
from functools import partial

import jax.numpy as jnp
from reservoir.core.types import NpF64, JaxF64, to_jax_f64, to_np_f64

from reservoir.models.generative import ClosedLoopGenerativeModel
from reservoir.models.presets import PipelineConfig
from reservoir.pipelines.config import DatasetMetadata, FrontendContext, ModelStack
from reservoir.layers.projection import Projection
from reservoir.pipelines.evaluation import Evaluator
from reservoir.pipelines.strategies import ReadoutStrategyFactory
from reservoir.pipelines.components.data_coordinator import DataCoordinator
from reservoir.utils.batched_compute import batched_compute
from reservoir.utils.reporting import print_feature_stats
from reservoir.models.distillation.model import DistillationModel
from reservoir.core.types import ResultDict


class PipelineExecutor:
    """
    Orchestrates the pipeline execution.
    Acts as a Site Supervisor:
    1. Asks DataCoordinator for materials (Data).
    2. Directs Worker (Model) to process materials.
    3. Asks StrategyFactory for specialized tooling (Readout Strategy).
    4. Executes the finish work.
    """

    def __init__(self, stack: ModelStack, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata, coordinator: DataCoordinator):
        self.stack = stack
        self.frontend_ctx = frontend_ctx
        self.dataset_meta = dataset_meta
        self.evaluator = Evaluator()
        # Dependency Injection (OCP/DIP)
        self.coordinator = coordinator

    def run(self, config: PipelineConfig) -> ResultDict:
        """Run the execution phase."""
        
        # Step 5: Model Dynamics (Training/Warmup)
        # Coordinator provides raw training data
        train_X = self.coordinator.get_train_inputs()
        projection = self.frontend_ctx.projection_layer
        
        # Pass projection_layer to model.train() so models can compose it internally
        # Explicit transition to Device Domain (JaxF64)
        train_y = self.frontend_ctx.processed_split.train_y
        train_logs = self.stack.model.train(
            to_jax_f64(train_X), 
            to_jax_f64(train_y) if train_y is not None else None,
            projection_layer=projection,
        )

        # Delegate extraction (Model does work, Coordinator provides input)
        train_Z, val_Z, test_Z = self._extract_all_features(self.stack.model)

        print("\n=== Step 6: Extract Features (Output) ===")
        if train_Z is not None:
            print_feature_stats(train_Z, "6:Z:train")
            if jnp.std(train_Z) < 0.1:
                raise ValueError(f"Feature collapse detected! train_Z std ({jnp.std(train_Z):.4f}) < 0.1. "
                                 "This usually indicates the Reservoir state is saturated or not responding to input.")

        if val_Z is not None: 
            print_feature_stats(val_Z, "6:Z:val")
            if jnp.std(val_Z) < 0.1:
                 print(f"    [Warning] val_Z std ({jnp.std(val_Z):.4f}) is very low.")

        if test_Z is not None : 
            print_feature_stats(test_Z, "6:Z:test")
            if jnp.std(test_Z) < 0.1:
                 print(f"    [Warning] test_Z std ({jnp.std(test_Z):.4f}) is very low.")

        if train_Z is None:
             raise ValueError("train_Z is None. Execution aborted.")

        # Step 6.5: Target Alignment (Delegate to Coordinator)
        print("\n=== Step 6.5: Target Alignment (Auto-Align) ===")
        train_y = self.coordinator.align_targets(train_Z, "train")
        val_y = self.coordinator.align_targets(val_Z, "val")
        test_y = self.coordinator.align_targets(test_Z, "test")

        # Step 7: Fit Readout (Strategy Pattern)
        readout_name = type(self.stack.readout).__name__ if self.stack.readout else "None"
        print(f"\n=== Step 7: Readout ({readout_name}) with val data ===")
        
        strategy = ReadoutStrategyFactory.create_strategy(
            self.stack.readout,
            self.dataset_meta, 
            self.evaluator, 
            self.stack.metric
        )

        fit_result = strategy.fit_and_evaluate(
            self.stack.model, self.stack.readout,
            train_Z, val_Z, test_Z,
            train_y, val_y, test_y,
            self.frontend_ctx, self.dataset_meta,
            config
        )
        
        # Step 8.5: Capture Quantum Trace (for Visualization)
        quantum_trace = None
        # Check if it's a QuantumReservoir (duck typing or class check)
        if hasattr(self.stack.model, "n_qubits") and hasattr(self.stack.model, "measurement_basis"):
             print("\n=== Step 8.5: Capturing Quantum Dynamics Trace ===")
             try:
                 # Take first test sample or train sample
                 test_data = self.frontend_ctx.processed_split.test_X
                 if test_data is None:
                     test_data = self.frontend_ctx.processed_split.train_X
                     
                 sample_input = None
                 if test_data is not None and len(test_data) > 0:
                      # Heuristic: If 2D (Batch, Feat), treat as sequence of length min(100, N)
                      if test_data.ndim == 2:
                           # Use full sequence for visualization as requested
                           sample_input = test_data[None, :, :]
                      # If 3D (Batch, Time, Feat), take first sample -> (1, T, F)
                      elif test_data.ndim == 3:
                           sample_input = test_data[0:1]
                 
                 if sample_input is not None:
                     # Force return_sequences=True to get time evolution
                     # Use batch_size=1
                     # Convert to JAX array to satisfy strictly typed model
                     sample_input_jax = to_jax_f64(sample_input)
                     trace = self.stack.model(sample_input_jax, return_sequences=True)
                     quantum_trace = to_np_f64(trace)
                     print(f"    [Executor] Captured trace shape: {trace.shape}")
             except (ValueError, RuntimeError, TypeError) as e:
                 print(f"    [Executor] Failed to capture quantum trace: {e}")

        from typing import cast
        return cast("ResultDict", {
            "fit_result": fit_result,
            "train_logs": train_logs,
            "quantum_trace": quantum_trace,
        })

    def _extract_all_features(self, model: ClosedLoopGenerativeModel) -> tuple[NpF64 | None, ...]:
        """
        Orchestrate feature extraction.
        If projection_layer is deferred, fuse projection + model forward.
        """
        window_size = getattr(model, 'input_window_size', 0)
        batch_size = self.dataset_meta.training.batch_size
        projection = self.frontend_ctx.projection_layer  # May be None
        
        if window_size > 1:
            print(f"    [Executor] Requesting Halo Padding (Window={window_size}) from Coordinator...")
        
        if projection is not None:
            print(f"    [Executor] Fusing {type(projection).__name__} + {type(model).__name__} in batched_compute (OOM-safe)")
        else:
            print(f"    [Executor] Running {type(model).__name__} feature extraction in batched_compute...")

        # 1. Train
        train_in = self.coordinator.get_train_inputs()
        train_Z = self._compute_split(model, train_in, "train", batch_size, projection=projection)

        # 2. Validation
        val_in = self.coordinator.get_val_inputs(window_size)
        val_Z = self._compute_split(model, val_in, "val", batch_size, projection=projection)

        # 3. Test
        test_in = self.coordinator.get_test_inputs(window_size)
        test_Z = self._compute_split(model, test_in, "test", batch_size, projection=projection)
            
        return train_Z, val_Z, test_Z

    @staticmethod
    def _compute_split(
        model: ClosedLoopGenerativeModel, 
        inputs: NpF64 | None, 
        split_name: str, 
        batch_size: int,
        projection: Projection | None = None,
    ):
        """Helper to run batched computation. If projection is deferred, fuse it with model forward."""
        if inputs is None:
            return None
        
        # DistillationModel handles projection internally for its teacher, 
        # but its student (which is used during predict/extraction) takes RAW inputs.
        # Fusing projection here would cause shape mismatches in the student.
        is_distillation = isinstance(model, DistillationModel)

        if projection is not None and not is_distillation:
            # Fused: projection + model forward in a single GPU pass
            def fused_fn(x: JaxF64) -> JaxF64:
                return model(projection(x))
            return batched_compute(fused_fn, inputs, batch_size, desc=f"[Step 3 and 5 Proj+Extract] {split_name}")
        else:
            # No projection (already projected or no projection needed, or DistillationModel)
            fn = partial(model, split_name=None)
            return batched_compute(fn, inputs, batch_size, desc=f"[Step 5 Extracting] {split_name}")
