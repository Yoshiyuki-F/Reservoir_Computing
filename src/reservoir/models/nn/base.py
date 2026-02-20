"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/base.py
Flax-based BaseModel adapter optimized with jax.lax.scan and tqdm logging."""

from typing import Dict, Optional, List

from beartype import beartype
import jax
import jax.numpy as jnp
from reservoir.core.types import JaxF64, JaxKey, TrainLogs, EvalMetrics, ConfigDict
import optax
from flax.training import train_state
from tqdm import tqdm  # 進行状況表示用

from reservoir.training.presets import TrainingConfig


@beartype
class BaseModel(ABC):
    """Minimal training/evaluation contract shared by Flax adapters."""

    def train(self, inputs: JaxF64, targets: Optional[JaxF64] = None) -> TrainLogs:
        """
        Execute internal pre-training phase (e.g., Distillation, Backprop). Returns metrics/logs.
        Defaults to a no-op so models without a pre-training stage can conform to the interface.
        """
        return {}

    @abstractmethod
    def predict(self, X: JaxF64) -> JaxF64:
        ...

    @abstractmethod
    def evaluate(self, X: JaxF64, y: JaxF64) -> EvalMetrics:
        ...

    def get_topology_meta(self) -> ConfigDict:
        """Optional topology metadata for visualization."""
        return getattr(self, "topology_meta", {})

    @property
    def input_window_size(self) -> int:
        """Required input history/window size. Defaults to 0 (no history)."""
        return 0


@beartype
class BaseFlaxModel(BaseModel, ABC):
    """Adapter that turns a flax.linen Module into a BaseModel."""

    def __init__(self, model_config: ConfigDict, training_config: TrainingConfig, classification: bool = False) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.learning_rate: float = float(training_config.learning_rate)
        self.epochs: int = int(training_config.epochs)
        self.batch_size: int = int(training_config.batch_size)
        self.classification: bool = classification
        self.seed: int = int(model_config.get("seed", training_config.seed))
        self._model_def = self._create_model_def()
        self._state: Optional[train_state.TrainState] = None
        self.trained: bool = False

    @abstractmethod
    def _create_model_def(self):
        """Return the flax.linen Module definition."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Training utilities                                                 #
    # ------------------------------------------------------------------ #
    def _build_optimizer(self, num_train_steps: int) -> optax.GradientTransformation:
        """Build optimizer with optional learning rate schedule."""
        scheduler_type = getattr(self.training_config, 'scheduler_type', None)
        warmup_epochs = getattr(self.training_config, 'warmup_epochs', 0)
        warmup_steps = warmup_epochs * (num_train_steps // self.epochs) if warmup_epochs > 0 else 0
        
        if scheduler_type == "cosine":
            # Cosine decay with optional warmup
            if warmup_steps > 0:
                schedule = optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=self.learning_rate,
                    warmup_steps=warmup_steps,
                    decay_steps=num_train_steps,
                    end_value=0,
                )
            else:
                schedule = optax.cosine_decay_schedule(
                    init_value=self.learning_rate,
                    decay_steps=num_train_steps,
                )
            return optax.adam(schedule)
        else:
            # Constant learning rate
            return optax.adam(self.learning_rate)

    def _init_train_state(self, key: JaxKey, sample_input: JaxF64, num_train_steps: int) -> train_state.TrainState:
        variables = self._model_def.init(key, sample_input)
        params = variables["params"]
        tx = self._build_optimizer(num_train_steps)
        return train_state.TrainState.create(
            apply_fn=self._model_def.apply,
            params=params,
            tx=tx,
        )

    @staticmethod
    def _train_step(state: train_state.TrainState, batch_x: JaxF64, batch_y: JaxF64, classification: bool):
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
    def train(self, inputs: JaxF64, targets: Optional[JaxF64] = None, **_) -> TrainLogs:
        if targets is None:
            raise ValueError("BaseFlaxModel.train requires 'targets'.")

        print(f"\n=== Step 5: Model Dynamics (Training/Warmup) [] ===")
        # Inputs/Targets are already JaxF64 (Device Domain)
        num_samples = inputs.shape[0]
        num_batches = num_samples // self.batch_size
        num_train_steps = self.epochs * num_batches  # Total training steps for scheduler

        # Initialize State
        rng = jax.random.PRNGKey(self.seed)
        init_key, _ = jax.random.split(rng)
        sample_input = inputs[:1]

        if self._state is None:
            print(f"    [JAX] Initializing parameters...")
            self._state = self._init_train_state(init_key, sample_input, num_train_steps)

        # JIT function
        @jax.jit
        def train_step_jit(state, b_x, b_y):
            new_state, loss = BaseFlaxModel._train_step(state, b_x, b_y, self.classification)
            return new_state, loss

        loss_history: List[float] = []
        limit = num_batches * self.batch_size

        print(f"    [JAX] Starting Loop: {self.epochs} epochs, {num_batches} batches/epoch.")
        pbar = tqdm(range(self.epochs), desc="[Train]", unit="ep")

        for _ in pbar:
            batch_losses = []

            for i in range(0, limit, self.batch_size):
                b_x = inputs[i: i + self.batch_size]
                b_y = targets[i: i + self.batch_size]

                self._state, loss = train_step_jit(self._state, b_x, b_y)
                batch_losses.append(float(loss))

            if batch_losses:
                avg_loss = float(jnp.mean(jnp.array(batch_losses)))
                loss_history.append(avg_loss)
                pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

        self.trained = True
        final_loss = loss_history[-1] if loss_history else 0.0
        return {"loss_history": loss_history, "final_loss": final_loss}


    def predict(self, X: JaxF64) -> JaxF64:
        if self._state is None:
            raise RuntimeError("Model not trained")
        @jax.jit
        def _predict_fn(params, inputs):
            return self._model_def.apply({"params": params}, inputs)

        return _predict_fn(self._state.params, X)

    def evaluate(self, X: JaxF64, y: JaxF64) -> EvalMetrics:
        preds = self.predict(X)
        y_arr = y
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

    def get_topology_meta(self) -> ConfigDict:
        return getattr(self, "topology_meta", {})