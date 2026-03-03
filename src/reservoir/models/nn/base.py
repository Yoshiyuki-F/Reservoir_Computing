"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/base.py
Flax-based BaseModel adapter optimized with jax.lax.scan and tqdm logging."""


from beartype import beartype
import jax
import jax.numpy as jnp
from reservoir.core.types import JaxF64, JaxKey, TrainLogs, EvalMetrics, ConfigDict, TopologyMeta
import optax
import flax.linen as nn
from flax.training import train_state
from tqdm import tqdm  # 進行状況表示用
from abc import ABC, abstractmethod

from reservoir.training.presets import TrainingConfig


@beartype
class BaseModel(ABC):
    """Minimal training/evaluation contract shared by Flax adapters."""
    
    topology_meta: TopologyMeta = {}

    def train(self, inputs: JaxF64, targets: JaxF64 | None = None, log_prefix: str = "4", **kwargs) -> TrainLogs:
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

    def get_topology_meta(self) -> TopologyMeta:
        """Optional topology metadata for visualization."""
        return self.topology_meta

    @property
    def input_window_size(self) -> int:
        """Required input history/window size. Defaults to 0 (no history)."""
        return 0


@beartype
class BaseFlaxModel(BaseModel, ABC):
    """Adapter that turns a flax.linen Module into a BaseModel."""

    def __init__(self, model_config: ConfigDict, classification: bool = False, training_config: TrainingConfig | None = None) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.learning_rate: float = float(getattr(training_config, "learning_rate", 0.001))
        self.epochs: int = int(getattr(training_config, "epochs", 1))
        self.batch_size: int = int(getattr(training_config, "batch_size", 32))
        self.classification: bool = classification
        
        seed_val = model_config.get("seed", getattr(training_config, "seed", 0))
        self.seed: int = 0 if seed_val is None else int(float(str(seed_val)))

        self._model_def = self._create_model_def()
        self._state: train_state.TrainState | None = None
        self.trained: bool = False

    @abstractmethod
    def _create_model_def(self) -> nn.Module:
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
    def train(self, inputs: JaxF64, targets: JaxF64 | None = None, log_prefix: str = "4", **kwargs) -> TrainLogs:
        if targets is None:
            raise ValueError("BaseFlaxModel.train requires 'targets'.")

        import time

        print(f"\n[nn.base.py] === Training {self.__class__.__name__} ===")

        projection_layer = kwargs.get("projection_layer")
        adapter = kwargs.get("adapter")

        # ==============================================================
        # Phase 1: Pre-compute adapter if safe (no projection → no OOM)
        # Projection is kept inside scan body to avoid OOM on large outputs.
        # ==============================================================
        processed_x = inputs
        processed_y = targets
        apply_proj_in_scan = projection_layer is not None
        apply_adapter_in_scan = False

        if adapter is not None:
            if apply_proj_in_scan:
                # Projection output is large → adapter must also be in scan
                apply_adapter_in_scan = True
                print(f"[nn.base.py] Projection + adapter ({adapter.__class__.__name__}) will be applied per-batch inside JIT")
            else:
                # No projection → safe to pre-compute adapter
                print(f"[nn.base.py] Pre-applying adapter ({adapter.__class__.__name__})...")
                processed_x = adapter(processed_x)
                processed_y = adapter.align_targets(processed_y)

        # ==============================================================
        # Phase 2: Setup batching
        # ==============================================================
        num_samples = processed_x.shape[0]
        batch_size = self.batch_size
        num_batches = num_samples // batch_size if batch_size > 0 else 0
        num_train_steps = self.epochs * num_batches

        if num_batches == 0:
            print("[nn.base.py] Warning: Not enough samples for a single batch.")
            return {}

        # Initialize model state using a processed sample
        rng = jax.random.PRNGKey(self.seed)
        init_key, _ = jax.random.split(rng)

        if self._state is None:
            print("[nn.base.py] Initializing parameters...")
            sample = processed_x[:1]
            if apply_proj_in_scan:
                sample = projection_layer(sample)
            if apply_adapter_in_scan:
                sample = adapter(sample)
            self._state = self._init_train_state(init_key, sample, num_train_steps)

        # Reshape into batches
        limit = num_batches * batch_size
        inputs_batched = processed_x[:limit].reshape(num_batches, batch_size, *processed_x.shape[1:])
        targets_batched = processed_y[:limit].reshape(num_batches, batch_size, *processed_y.shape[1:])

        # ==============================================================
        # Phase 3: Nested jax.lax.scan (epochs × batches) — max speed
        # Projection + adapter applied per-batch inside compiled scan.
        # GPU→CPU sync only once per chunk (50 epochs).
        # ==============================================================
        classification = self.classification
        scan_chunk = getattr(self.training_config, 'scan_chunk_size', 50)
        CHUNK_SIZE = min(scan_chunk, self.epochs)
        num_chunks = self.epochs // CHUNK_SIZE
        remainder = self.epochs % CHUNK_SIZE

        def _make_train_fn(length):
            @jax.jit
            def train_fn(state, data_x, data_y):
                def epoch_body(st, _):
                    def batch_body(s, batch):
                        bx, by = batch
                        if apply_proj_in_scan:
                            bx = projection_layer(bx)
                        if apply_adapter_in_scan:
                            bx = adapter(bx)
                            by = adapter.align_targets(by)
                        new_s, loss = BaseFlaxModel._train_step(s, bx, by, classification)
                        return new_s, loss
                    s, losses = jax.lax.scan(batch_body, st, (data_x, data_y))
                    return s, jnp.mean(losses)
                return jax.lax.scan(epoch_body, state, None, length=length)
            return train_fn

        train_chunk = _make_train_fn(CHUNK_SIZE)

        loss_history: list[float] = []
        proj_info = " (proj+adapt per-batch)" if apply_proj_in_scan else ""
        print(f"[nn.base.py] Training: {self.epochs} epochs × {num_batches} batches/epoch{proj_info}")
        print(f"[nn.base.py] (Nested jax.lax.scan: {CHUNK_SIZE}-epoch chunks)")
        t_start = time.time()
        pbar = tqdm(total=self.epochs, desc="[Train]", unit="ep")

        for _ in range(num_chunks):
            self._state, chunk_losses = train_chunk(self._state, inputs_batched, targets_batched)
            chunk_list = chunk_losses.tolist()
            loss_history.extend(chunk_list)
            pbar.update(CHUNK_SIZE)
            pbar.set_postfix({"loss": f"{chunk_list[-1]:.6f}"})

        if remainder > 0:
            train_rem = _make_train_fn(remainder)
            self._state, rem_losses = train_rem(self._state, inputs_batched, targets_batched)
            rem_list = rem_losses.tolist()
            loss_history.extend(rem_list)
            pbar.update(remainder)
            pbar.set_postfix({"loss": f"{rem_list[-1]:.6f}"})

        pbar.close()
        elapsed = time.time() - t_start
        print(f"[nn.base.py] Training complete in {elapsed:.1f}s. Final loss: {loss_history[-1]:.6f}")

        self.trained = True
        return {"loss_history": loss_history, "final_loss": loss_history[-1] if loss_history else 0.0}


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

    def get_topology_meta(self) -> TopologyMeta:
        return self.topology_meta
