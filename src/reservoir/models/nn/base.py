"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/base.py
Flax-based BaseModel adapter optimized with jax.lax.scan and tqdm logging."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state
from tqdm import tqdm  # 進行状況表示用

from reservoir.training.presets import TrainingConfig


class BaseModel(ABC):
    """Minimal training/evaluation contract shared by Flax adapters."""

    def train(self, inputs: Any, targets: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute internal pre-training phase (e.g., Distillation, Backprop). Returns metrics/logs.
        Defaults to a no-op so models without a pre-training stage can conform to the interface.
        """
        return {}

    @abstractmethod
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        ...

    @abstractmethod
    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        ...

    def get_topology_meta(self) -> Dict[str, Any]:
        """Optional topology metadata for visualization."""
        return {}


class BaseFlaxModel(BaseModel, ABC):
    """Adapter that turns a flax.linen Module into a BaseModel."""

    def __init__(self, model_config: Dict[str, Any], training_config: TrainingConfig) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.learning_rate: float = float(training_config.learning_rate)
        self.epochs: int = int(training_config.epochs)
        self.batch_size: int = int(training_config.batch_size)
        self.classification: bool = bool(training_config.classification)
        self.seed: int = int(model_config.get("seed", training_config.seed))
        self._model_def = self._create_model_def()
        self._state: Optional[train_state.TrainState] = None
        self.trained: bool = False

    @abstractmethod
    def _create_model_def(self) -> Any:
        """Return the flax.linen Module definition."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Training utilities                                                 #
    # ------------------------------------------------------------------ #
    def _init_train_state(self, key: jnp.ndarray, sample_input: jnp.ndarray) -> train_state.TrainState:
        variables = self._model_def.init(key, sample_input)
        params = variables["params"]
        tx = optax.adam(self.learning_rate)
        return train_state.TrainState.create(
            apply_fn=self._model_def.apply,
            params=params,
            tx=tx,
        )

    @staticmethod
    def _train_step(state: train_state.TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray, classification: bool):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch_x)
            if classification:
                labels = batch_y
                if labels.ndim == 1:
                    labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
                loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
            else:
                loss = jnp.mean((logits - batch_y) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    # ------------------------------------------------------------------ #
    # BaseModel API                                                      #
    # ------------------------------------------------------------------ #
    def train(self, inputs: Any, targets: Optional[Any] = None, **_: Any) -> Dict[str, Any]:
        if targets is None:
            raise ValueError("BaseFlaxModel.train requires 'targets'.")

        # 1. ここで一括してGPUに転送してしまう (MNIST程度なら余裕で乗ります)
        print("    [JAX] Transferring data to GPU...")
        inputs = jax.device_put(jnp.asarray(inputs, dtype=jnp.float32))
        targets = jax.device_put(jnp.asarray(targets, dtype=jnp.float32))  # 回帰ならfloat, 分類ならint注意

        num_samples = inputs.shape[0]
        # ... (初期化ロジックは同じ) ...

        # Initialize State
        rng = jax.random.PRNGKey(self.seed)
        init_key, _ = jax.random.split(rng)
        sample_input = inputs[:1]  # 既にGPUにあるデータを使う

        if self._state is None:
            print(f"    [JAX] Initializing parameters...")
            self._state = self._init_train_state(init_key, sample_input)

        # JIT function
        @jax.jit
        def train_step_jit(state, b_x, b_y):
            # classificationフラグの扱いに注意 (self.classificationがboolならstatic引数化など検討)
            # ここではクロージャでキャプチャしているので再コンパイルは起きないはずですが
            # static_argnumsを使うのがベストプラクティスです。
            new_state, loss = BaseFlaxModel._train_step(state, b_x, b_y, self.classification)
            return new_state, loss

        loss_history = []
        num_batches = num_samples // self.batch_size
        limit = num_batches * self.batch_size

        print(f"    [JAX] Starting Loop: {self.epochs} epochs, {num_batches} batches/epoch.")
        pbar = tqdm(range(self.epochs), desc="[Train]", unit="ep")

        for _ in pbar:
            batch_losses = []

            # Pythonループだが、データは既にGPUにあるためスライシングは高速
            for i in range(0, limit, self.batch_size):
                # GPU上の配列をスライス (データ転送は発生しない)
                b_x = inputs[i: i + self.batch_size]
                b_y = targets[i: i + self.batch_size]

                self._state, loss = train_step_jit(self._state, b_x, b_y)
                batch_losses.append(float(loss))  # LossをCPUに戻すコストのみ

            if batch_losses:
                avg_loss = float(np.mean(batch_losses))
                loss_history.append(avg_loss)
                pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

        self.trained = True
        return {"loss_history": loss_history}

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self._state is None:
            raise RuntimeError("Model not trained")
        X = jnp.asarray(X)
        @jax.jit
        def _predict_fn(params, inputs):
            return self._model_def.apply({"params": params}, inputs)

        return _predict_fn(self._state.params, X)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        y_arr = jnp.asarray(y)
        if self.classification:
            labels = y_arr
            if labels.ndim > 1:
                labels = jnp.argmax(labels, axis=-1)
            pred_labels = jnp.argmax(preds, axis=-1)
            acc = float(jnp.mean(pred_labels == labels))
            return {"accuracy": acc}
        mse = float(jnp.mean((preds - y_arr) ** 2))
        mae = float(jnp.mean(jnp.abs(preds - y_arr)))
        return {"mse": mse, "mae": mae}

    def get_topology_meta(self) -> Dict[str, Any]:
        return getattr(self, "topology_meta", {})