"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/distillation/model.py
Distill a teacher feature extractor into a student feed-forward network.
Implements strict teacher-student distillation: student mimics teacher outputs, labels are ignored.
Updated with clear logging phases.
"""

from __future__ import annotations


from beartype import beartype
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from reservoir.models.generative import ClosedLoopGenerativeModel
from typing import TYPE_CHECKING

from reservoir.core.types import JaxF64, TrainLogs, EvalMetrics, KwargsDict, TopologyMeta
from reservoir.models.reservoir.classical import ClassicalReservoir
from reservoir.models.nn.fnn import FNNModel
from reservoir.training.presets import TrainingConfig

if TYPE_CHECKING:
    pass #ruff safe breaks this

@beartype
class DistillationModel(ClosedLoopGenerativeModel):
    """
    Distills reservoir dynamics into a student FNN.
    Teacher: ClassicalReservoir (handles aggregation internally; no gradients).
    Student: FNNModel trained to regress onto teacher targets.
    
    Implements ClosedLoopGenerativeModel to allow autonomous closed-loop generation.
    State representation: Sliding window of input features.
    """

    @property
    def input_window_size(self) -> int:
        return self._window_size

    def __init__(
        self,
        teacher: ClassicalReservoir,
        student: FNNModel,
        training_config: TrainingConfig | None = None,
    ):
        self.teacher = teacher
        self.student = student
        self.training_config = training_config
        
        # State management for closed-loop generation
        self._input_dim: int | None = None
        # Get window_size from student's adapter if available
        self._window_size: int = getattr(student, 'window_size', 1) or 1

    def __call__(self, inputs: JaxF64, params: KwargsDict | None = None) -> JaxF64:
        return self.predict(inputs, params=params)

    def predict(self, X: JaxF64, params: KwargsDict | None = None) -> JaxF64:
        """Delegate to student's predict (which handles adapter internally)."""
        return self.student.predict(X)

    def _compute_teacher_targets(self, inputs: JaxF64) -> JaxF64:
        """Legacy single-batch implementation (prone to OOM on large datasets)."""
        teacher_outputs = self.teacher(inputs)
        return jax.lax.stop_gradient(teacher_outputs)

    def _compute_teacher_targets_batched(self, inputs: JaxF64, batch_size: int) -> JaxF64:
        """Compute teacher targets in batches to allow CPU offloading and avoid OOM."""
        n_samples = inputs.shape[0]

        # 1. Infer output shape from a small dummy batch
        dummy_in = inputs[:1]
        # Force JIT compilation for shape inference
        dummy_out = self.teacher(dummy_in)

        # --- LOGGING STEP 5A (Raw Dynamics) ---
        # Capture raw states for the first batch to visualize reservoir dynamics
        # Use return_sequences=True to get (Batch, Time, Features)
        first_batch_size = min(batch_size, n_samples)
        first_batch_in = inputs[:first_batch_size]
        raw_states = self.teacher(first_batch_in, return_sequences=True, split_name="teacher_raw")
        jax.debug.print("    [Teacher] Raw Dynamics (First Batch) Stats: mean={m:.4f} std={s:.4f}", m=jnp.mean(raw_states), s=jnp.std(raw_states))
        # --------------------------------------

        output_shape = (n_samples,) + dummy_out.shape[1:]
        # print(f"    [Teacher] Generating targets (Total: {n_samples}, Batch: {batch_size})...")
        targets = jnp.empty(output_shape)

        # 2. Define JIT step
        @jax.jit
        def step(x):
             return self.teacher(x)

        # 3. Loop
        with tqdm(total=n_samples, desc="[Teacher]", unit="samples") as pbar:
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                batch_x = inputs[i:end]
                # Move to GPU, compute, move back to CPU
                batch_out = step(batch_x)
                targets = targets.at[i:end].set(batch_out)
                pbar.update(end - i)

        return targets

    def train(self, inputs: JaxF64, targets: JaxF64 | None = None, log_prefix: str = "4", **kwargs) -> TrainLogs:
        """
        Orchestrate the distillation process with clear phase separation in logs.
        
        Inputs/Targets are already JaxF64 (Device Domain).
        """
        # Phase A: Teacher Target Generation
        print("\n    [Distillation] ==========================================")
        print("    [Distillation] Phase A: Teacher Target Generation")
        print("    [Distillation] ==========================================")

        # Capture input dimension for closed-loop generation state init
        batch_sz = self.training_config.batch_size if self.training_config else 32
        teacher_targets = self._compute_teacher_targets_batched(inputs, batch_size=batch_sz)
        
        jax.debug.print("    [Distillation] Teacher Target Stats: mean={m:.4f} std={s:.4f}", m=jnp.mean(teacher_targets), s=jnp.std(teacher_targets))

        # --- Phase B: Student Model Training ---
        print("\n    [Distillation] ==========================================")
        print("    [Distillation] Phase B: Student Model Training")
        print("    [Distillation] ==========================================")
        print(f"    [Student] Training {self.student.__class__.__name__} to mimic Teacher...")

        # Student training - FNNModel.train() handles adapter and alignment internally
        student_logs = self.student.train(inputs, teacher_targets, log_prefix="4B") or {}

        # --- Generate 5B: Student Output (Predicted State) ---
        # To verify Distillation, we show the student's output stats
        student_outputs = self.student.predict(inputs)
        jax.debug.print("    [Distillation] Student Output Stats: mean={m:.4f} std={s:.4f}", m=jnp.mean(student_outputs), s=jnp.std(student_outputs))

        # Optional: Compute final distillation MSE for logging
        distill_mse = float(student_logs.get("final_loss", 0.0))
        student_logs.setdefault("distill_mse", distill_mse)
        student_logs.setdefault("final_loss", distill_mse)
        return student_logs

    def evaluate(self, X: JaxF64, y: JaxF64 | None = None) -> EvalMetrics:
        """
        Distillation evaluation: compare student output with teacher output.
        """
         # Use batched generation if dataset is large to prevent OOM during evaluation
        if X.shape[0] > 4096:
             teacher_targets = self._compute_teacher_targets_batched(X, batch_size=2048)
        else:
             teacher_targets = self._compute_teacher_targets(X)

        # Delegate to student's evaluate (handles adapter and alignment internally)
        student_metrics = self.student.evaluate(X, teacher_targets)

        return student_metrics

    def get_topology_meta(self) -> TopologyMeta:
        if self.topology_meta:
            return self.topology_meta
        if self.student.topology_meta:
            return self.student.topology_meta
        return self.teacher.topology_meta

    # ------------------------------------------------------------------ #
    # ClosedLoopGenerativeModel Interface                                #
    # ------------------------------------------------------------------ #

    def initialize_state(self, batch_size: int = 1) -> JaxF64:
        """
        Initialize the sliding window state.
        CAUTION: Requires self._input_dim to be set via train().
        """
        if self._input_dim is None:
            # Fallback for inference-only usage? 
            # We cannot create a concrete array without dimension.
            # Assuming training happened or input_dim provided manually.
            raise RuntimeError("DistillationModel._input_dim not set. Call train() first.")
            
        return jnp.zeros((batch_size, self._window_size, self._input_dim))

    def step(self, state: JaxF64, inputs: JaxF64) -> tuple[JaxF64, JaxF64]:
        """
        Single step execution with window management.
        Args:
            state: (batch, window_size, features)
            inputs: (batch, features)
        Returns:
            next_state: (batch, window_size, features)
            output: (batch, features)
        """
        # Update window buffer by shifting and appending new input
        # state[:, 1:] -> drops oldest, adds space
        # inputs[:, None, :] -> adds time dim to input
        next_state = jnp.concatenate([state[:, 1:], inputs[:, None, :]], axis=1)
        
        # Flatten state for FNN prediction: (batch, window_size*features)
        batch_size = next_state.shape[0]
        flat_input = next_state.reshape(batch_size, -1)
        
        # Call BaseFlaxModel.predict directly to bypass FNN's adapter
        # (flat_input is already windowed)
        from reservoir.models.nn.base import BaseFlaxModel
        output = BaseFlaxModel.predict(self.student, flat_input)
        return next_state, output

    def forward(self, state: JaxF64, input_data: JaxF64) -> tuple[JaxF64, JaxF64]:
        """
        Process sequence using JAX scan.
        Args:
            state: Initial window (batch, window_size, features)
            input_data: Sequence inputs (batch, time, features)
        """
        # Scan over time dimension
        inputs_transposed = jnp.swapaxes(input_data, 0, 1)  # (time, batch, feat)
        final_state, stacked_outputs = jax.lax.scan(self.step, state, inputs_transposed)
        
        # Swap back to (batch, time, feat)
        stacked_outputs = jnp.swapaxes(stacked_outputs, 0, 1)
        return final_state, stacked_outputs
    
    # generate_closed_loop is inherited from ClosedLoopGenerativeModel