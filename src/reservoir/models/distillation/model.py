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
        
        self._input_dim: int | None = None
        self._window_size: int = getattr(student, 'window_size', 1) or 1

    def __call__(self, inputs: JaxF64, params: KwargsDict | None = None, **kwargs) -> JaxF64:
        return self.predict(inputs, params=params, **kwargs)

    def predict(self, X: JaxF64, params: KwargsDict | None = None, **kwargs) -> JaxF64:
        return self.student.predict(X)

    def _compute_teacher_targets(self, inputs: JaxF64, **kwargs) -> JaxF64:
        """Legacy single-batch implementation."""
        teacher_outputs = self.teacher(inputs, **kwargs)
        return jax.lax.stop_gradient(teacher_outputs)

    def _compute_teacher_targets_batched(self, inputs: JaxF64, batch_size: int, **kwargs) -> JaxF64:
        """Compute teacher targets in batches to allow CPU offloading and avoid OOM."""
        n_samples = inputs.shape[0]

        # 改善点: jax.eval_shape を使い、再計算なしで出力シェイプのみを取得
        dummy_out = jax.eval_shape(lambda x: self.teacher(x, **kwargs), inputs[:1])
        output_shape = (n_samples,) + dummy_out.shape[1:]
        
        targets = jnp.empty(output_shape)

        @jax.jit
        def step(x):
             return self.teacher(x, **kwargs)

        with tqdm(total=n_samples, desc="[Teacher]", unit="samples") as pbar:
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                batch_x = inputs[i:end]
                batch_out = step(batch_x)
                targets = targets.at[i:end].set(batch_out)
                pbar.update(end - i)

        return targets

    def train(self, inputs: JaxF64, targets: JaxF64 | None = None, log_prefix: str = "4", **kwargs) -> TrainLogs:
        """
        Orchestrate the distillation process with clear phase separation in logs.
        """
        print("\n    [Distillation] ==========================================")
        print("    [Distillation] Phase A: Teacher Target Generation")
        print("    [Distillation] ==========================================")

        # 改善点: kwargsを伝播させてprojection_layer等をTeacherに適用させる
        batch_sz = self.training_config.batch_size
        teacher_targets = self._compute_teacher_targets_batched(inputs, batch_size=batch_sz, **kwargs)
        
        jax.debug.print("    [Distillation] Teacher Target Stats: mean={m:.4f} std={s:.4f}", m=jnp.mean(teacher_targets), s=jnp.std(teacher_targets))

        print("\n    [Distillation] ==========================================")
        print("    [Distillation] Phase B: Student Model Training")
        print("    [Distillation] ==========================================")
        print(f"    [Student] Training {self.student.__class__.__name__} to mimic Teacher...")

        student_logs = self.student.train(inputs, teacher_targets, log_prefix="4B", **kwargs) or {}

        # 改善点: student側のpredictにも念のためkwargsを渡す
        student_outputs = self.student.predict(inputs)
        jax.debug.print("    [Distillation] Student Output Stats: mean={m:.4f} std={s:.4f}", m=jnp.mean(student_outputs), s=jnp.std(student_outputs))

        distill_mse = float(student_logs.get("final_loss", 0.0))
        student_logs.setdefault("distill_mse", distill_mse)
        student_logs.setdefault("final_loss", distill_mse)
        return student_logs

    def evaluate(self, X: JaxF64, y: JaxF64 | None = None, **kwargs) -> EvalMetrics:
        """
        Distillation evaluation: compare student output with teacher output.
        """
        if X.shape[0] > 4096:
             teacher_targets = self._compute_teacher_targets_batched(X, batch_size=self.training_config.batch_size, **kwargs)
        else:
             teacher_targets = self._compute_teacher_targets(X, **kwargs)

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
        if self._input_dim is None:
            raise RuntimeError("DistillationModel._input_dim not set. Call train() first.")
        return jnp.zeros((batch_size, self._window_size, self._input_dim))

    def step(self, state: JaxF64, inputs: JaxF64) -> tuple[JaxF64, JaxF64]:
        next_state = jnp.concatenate([state[:, 1:], inputs[:, None, :]], axis=1)
        batch_size = next_state.shape[0]
        flat_input = next_state.reshape(batch_size, -1)
        
        from reservoir.models.nn.base import BaseFlaxModel
        output = BaseFlaxModel.predict(self.student, flat_input)
        return next_state, output

    def forward(self, state: JaxF64, input_data: JaxF64) -> tuple[JaxF64, JaxF64]:
        inputs_transposed = jnp.swapaxes(input_data, 0, 1)
        final_state, stacked_outputs = jax.lax.scan(self.step, state, inputs_transposed)
        stacked_outputs = jnp.swapaxes(stacked_outputs, 0, 1)
        return final_state, stacked_outputs