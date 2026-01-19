"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/distillation/model.py
Distill a teacher feature extractor into a student feed-forward network.
Implements strict teacher-student distillation: student mimics teacher outputs, labels are ignored.
Updated with clear logging phases.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from reservoir.models.reservoir.classical import ClassicalReservoir
from reservoir.models.nn.fnn import FNNModel
from reservoir.models.nn.base import BaseModel
from reservoir.training.presets import TrainingConfig
from reservoir.utils.reporting import print_feature_stats


class DistillationModel(BaseModel):
    """
    Distills reservoir dynamics into a student FNN.
    Teacher: ClassicalReservoir (handles aggregation internally; no gradients).
    Student: FNNModel trained to regress onto teacher targets.
    """

    def __init__(
        self,
        teacher: ClassicalReservoir,
        student: FNNModel,
        training_config: TrainingConfig,
        student_adapter: Any = None,
    ):
        self.teacher = teacher
        self.student = student
        self.training_config = training_config
        self.student_adapter = student_adapter

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.predict(inputs)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.student_adapter is not None:
             # Preserve batch dimension if input is 3D sequence
             is_sequence = (X.ndim == 3)
             batch_size = X.shape[0] if is_sequence else 1

             X_in = self.student_adapter(X)
             out = self.student.predict(X_in)

             if is_sequence:
                 # Reshape (N*T', F) -> (N, T', F)
                 return out.reshape(batch_size, -1, out.shape[-1])
             return out
        return self.student.predict(X)

    def _prepare_student_data(self, inputs: jnp.ndarray, targets: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Aligns inputs and targets when an adapter (e.g. TimeDelay) changes the time dimension."""
        if self.student_adapter is None:
            return inputs, targets

        # 1. Transform Inputs
        # TimeDelayEmbedding returns (Batch * (Time-Window+1), Window*Feat)
        student_X = self.student_adapter(inputs)

        # 2. Align Targets
        # Targets are teacher outputs (Batch, Time, TeacherFeat)
        # We need to slice off the initial window accumulation
        # and flatten to match student_X samples.
        
        # Infer window size from adapter if possible, or deduce from shape change
        # Assuming linear mapping: N_out = N_in - N_per_window * (W-1) ?
        # Harder to guess. 
        # But TimeDelayEmbedding class has `window_size`.
        if hasattr(self.student_adapter, "window_size"):
             w = self.student_adapter.window_size
             # Slice targets: discard first w-1 steps
             # targets[:, w-1:, :]
             if targets.ndim == 3:
                 targets_sliced = targets[:, w-1:, :]
                 student_y = targets_sliced.reshape(-1, targets_sliced.shape[-1])
                 return student_X, student_y
        
        # Fallback if logic is generic or flat
        # If flattened, just return targets? No, might mismatch.
        # This fallback is risky but minimal.
        return student_X, targets

    def _compute_teacher_targets(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Legacy single-batch implementation (prone to OOM on large datasets)."""
        teacher_outputs = self.teacher(inputs)
        return jax.lax.stop_gradient(teacher_outputs)

    def _compute_teacher_targets_batched(self, inputs: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """Compute teacher targets in batches to allow CPU offloading and avoid OOM."""
        inputs_np = np.asarray(inputs)
        n_samples = inputs_np.shape[0]

        # 1. Infer output shape from a small dummy batch
        dummy_in = jnp.array(inputs_np[:1])
        # Force JIT compilation for shape inference
        dummy_out = self.teacher(dummy_in)

        output_shape = (n_samples,) + dummy_out.shape[1:]
        # print(f"    [Teacher] Generating targets (Total: {n_samples}, Batch: {batch_size})...")
        targets = np.empty(output_shape, dtype=np.float32)

        # 2. Define JIT step
        @jax.jit
        def step(x):
             return self.teacher(x)

        # 3. Loop
        with tqdm(total=n_samples, desc="[Teacher]", unit="samples") as pbar:
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                batch_x = inputs_np[i:end]
                # Move to GPU, compute, move back to CPU
                batch_out = step(jnp.array(batch_x))
                targets[i:end] = np.asarray(batch_out, dtype=np.float32)
                pbar.update(end - i)

        return jnp.array(targets)

    def train(self, inputs: jnp.ndarray, targets: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Orchestrate the distillation process with clear phase separation in logs.
        """
        # --- Phase 1: Teacher Target Generation ---
        print("\n    [Distillation] ==========================================")
        print("    [Distillation] Phase 1: Teacher Target Generation")
        print("    [Distillation] ==========================================")

        # Use batched computation for teacher targets (safe for large datasets)
        teacher_targets = self._compute_teacher_targets_batched(inputs, batch_size=self.training_config.batch_size)
        print_feature_stats(teacher_targets, "5:teacher")

        # Apply Adapter and Align Targets
        student_X, student_targets = self._prepare_student_data(inputs, teacher_targets)

        # --- Phase 2: Student Model Training ---
        print("\n    [Distillation] ==========================================")
        print("    [Distillation] Phase 2: Student Model Training")
        print("    [Distillation] ==========================================")
        print(f"    [Student] Training {self.student.__class__.__name__} to mimic Teacher...")

        # Student training (uses its own progress bar)
        student_logs = self.student.train(student_X, student_targets, **kwargs) or {}

        # Optional: Compute final distillation MSE for logging
        # If dataset is huge, consider batching this predict as well, but usually student is faster.
        distill_mse = student_logs.get("final_loss", 0.0)
        logs = dict(student_logs) if isinstance(student_logs, dict) else {}
        logs.setdefault("distill_mse", distill_mse)
        logs.setdefault("final_loss", distill_mse)
        return logs

    def evaluate(self, X: jnp.ndarray, y: Any = None) -> Dict[str, float]:
        """
        Distillation evaluation aligns student outputs with teacher features (not labels).
        """
        # Use batched generation if dataset is large to prevent OOM during evaluation
        if X.shape[0] > 4096:
             teacher_targets = self._compute_teacher_targets_batched(X, batch_size=2048)
        else:
             teacher_targets = self._compute_teacher_targets(X)

        # Align for student
        student_X, teacher_targets_aligned = self._prepare_student_data(X, teacher_targets)
        
        student_out = self.student.predict(student_X)
        
        if student_out.shape != teacher_targets_aligned.shape:
             # Just in case fallback didn't work
            raise ValueError(
                f"Distillation shape mismatch: student {student_out.shape} vs teacher {teacher_targets_aligned.shape}"
            )

        distill_mse = float(jnp.mean((student_out - teacher_targets_aligned) ** 2))
        student_metrics = self.student.evaluate(student_X, teacher_targets_aligned)

        if isinstance(student_metrics, dict):
            metrics: Dict[str, float] = dict(student_metrics)
        else:
            metrics = {}
        metrics.setdefault("distill_mse", distill_mse)
        return metrics

    def get_topology_meta(self) -> Dict[str, Any]:
        return (
            getattr(self, "topology_meta", {})
            or getattr(self.student, "topology_meta", {})
            or getattr(self.teacher, "topology_meta", {})
            or {}
        )