
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
        train_Z, val_Z, test_Z, val_final_info = self._extract_all_features(self.stack.model)

        print("\n=== Step 6: Extract Features (Output) ===")
        if train_Z is not None:
            if jnp.std(train_Z) < 0.1:
                raise ValueError(f"Feature collapse detected! train_Z std ({jnp.std(train_Z):.4f}) < 0.1. "
                                 "This usually indicates the Reservoir state is saturated or not responding to input.")

        if val_Z is not None: 
            if jnp.std(val_Z) < 0.1:
                 print(f"    [Warning] val_Z std ({jnp.std(val_Z):.4f}) is very low.")

        if test_Z is not None : 
            if jnp.std(test_Z) < 0.1:
                 print(f"    [Warning] test_Z std ({jnp.std(test_Z):.4f}) is very low.")

        if train_Z is None:
             raise ValueError("train_Z is None. Execution aborted.")

        # Step 6.5: Target Alignment (Delegate to Coordinator)
        print("\n=== Step 6.5: Target Alignment (Auto-Align) ===")
        train_y = self.coordinator.align_targets(train_Z, "train")
        val_y = self.coordinator.align_targets(val_Z, "val")
        
        # If test_Z was skipped, test_y still needs to be aligned to the test set length
        if test_Z is not None:
            test_y = self.coordinator.align_targets(test_Z, "test")
        else:
            # Fallback: test_y should match the length of the processed test_X
            test_y = self.frontend_ctx.processed_split.test_y
            print(f"    [Executor] Using raw test_y (Length: {len(test_y) if test_y is not None else 0})")

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
            self.stack.model,
            self.stack.readout,
            train_Z, val_Z, test_Z,
            train_y, val_y, test_y,
            self.frontend_ctx,
            self.dataset_meta,
            config,
            val_final_state=val_final_info
        )
        
        # Step 8.5: Capture Quantum Trace (for Visualization)
        quantum_trace = None
        # Check if it's a QuantumReservoir (duck typing or class check)
        if hasattr(self.stack.model, "n_qubits") and hasattr(self.stack.model, "measurement_basis"):
             print("\n=== Step 8.5: Capturing Quantum Dynamics Trace ===")
             try:
                 if fit_result.get("closed_loop_history") is not None:
                     # Use history from closed-loop generation (no re-computation needed)
                     quantum_trace = to_np_f64(fit_result["closed_loop_history"])
                     print(f"    [Executor] Using captured closed-loop history: {quantum_trace.shape}")
                 else:
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
                         print(f"[executor.py] Captured trace shape: {trace.shape}")
             except (ValueError, RuntimeError, TypeError) as e:
                 print(f"[executor.py] Failed to capture quantum trace: {e}")

        from typing import cast
        return cast("ResultDict", {
            "fit_result": fit_result,
            "train_logs": train_logs,
            "quantum_trace": quantum_trace,
        })

    def _extract_all_features(self, model: ClosedLoopGenerativeModel) \
            -> tuple[NpF64 | None, NpF64 | None, NpF64 | None, tuple | None]:
        """
        Orchestrate feature extraction.
        If projection_layer is deferred, fuse projection + model forward.
        """
        window_size = getattr(model, 'input_window_size', 0)
        batch_size = self.dataset_meta.training.batch_size
        projection = self.frontend_ctx.projection_layer  # May be None
        
        if window_size > 1:
            print(f"[executor.py] Requesting Halo Padding (Window={window_size}) from Coordinator...")
        
        if projection is not None:
            print(f"[executor.py] Fusing {type(projection).__name__} + {type(model).__name__} in batched_compute (OOM-safe)")
        else:
            print(f"[executor.py] Running {type(model).__name__} feature extraction in batched_compute...")

        # Check for Quantum Reservoir to enable state chaining
        is_quantum = "QuantumReservoir" in type(model).__name__
        current_state = None

        # 1. Train
        train_in = self.coordinator.get_train_inputs()
        train_Z, current_state, _ = self._compute_split(
            model, train_in, "train", batch_size, projection=projection, initial_state=current_state, return_state=is_quantum
        )

        # 2. Validation
        val_in = self.coordinator.get_val_inputs(window_size)
        val_Z, current_state, val_last_output = self._compute_split(
            model, val_in, "val", batch_size, projection=projection, initial_state=current_state, return_state=is_quantum
        )
        val_final_state = current_state

        # 3. Test
        test_Z = None
        if self.dataset_meta.classification:
            test_in = self.coordinator.get_test_inputs(window_size)
            test_Z, _, _ = self._compute_split(
                model, test_in, "test", batch_size, projection=projection, initial_state=None, return_state=False
            )
        else:
            print("    [Executor] Skipping Test feature extraction for Regression task (Closed-loop will be used).")
            
        return train_Z, val_Z, test_Z, (val_final_state, val_last_output)

    @staticmethod
    def _compute_split(
        model: ClosedLoopGenerativeModel, 
        inputs: NpF64 | None, 
        split_name: str, 
        batch_size: int,
        projection: Projection | None = None,
        initial_state: object | None = None,
        return_state: bool = False
    ) -> tuple: #TODO　型定義
        """Helper to run batched computation. If projection is deferred, fuse it with model forward."""
        if inputs is None:
            return None, None, None
        
        # Special Path for Quantum Reservoir State Chaining (Time Series Mode)
        # QuantumReservoir forward expects (1, Time, Feat)
        if return_state and inputs.ndim == 2:
            inputs_jax = to_jax_f64(inputs)
            # Reshape to (1, Time, Feat)
            if inputs_jax.ndim == 2:
                inputs_jax = inputs_jax[None, :, :]
            
            # Initialize state if not provided
            if initial_state is None:
                state = model.initialize_state(1)
            else:
                state = initial_state
            
            # Apply projection if exists (Fused)
            if projection is not None:
                inputs_jax = projection(inputs_jax) # Assuming projection handles (1, T, F) or vmap handles it
            
            # Run Forward (Directly, bypassing batched_compute for state access)
            # QuantumReservoir.forward returns (final_state, stacked_outputs)
            # Note: We must ensure we call the method that returns state.
            # 'forward' returns state. '__call__' does not.
            final_state, outputs_jax = model.forward(state, inputs_jax)
            
            # Output is (1, Time, Out) -> Flatten to (Time, Out)
            outputs_np = to_np_f64(outputs_jax[0])
            
            # Get last output for loop initialization
            last_output = outputs_jax[0, -1, :] # (Out,)
            
            # Log stats manually since we skipped batched_compute
            print_feature_stats(outputs_np, "executor.py",f"6:Z:{split_name}")
            
            return outputs_np, final_state, last_output

        # Standard Path
        # DistillationModel handles projection internally for its teacher, 
        # but its student (which is used during predict/extraction) takes RAW inputs.
        # Fusing projection here would cause shape mismatches in the student.
        is_distillation = isinstance(model, DistillationModel)

        if projection is not None and not is_distillation:
            # Fused: projection + model forward in a single GPU pass
            def fused_fn(x: JaxF64) -> JaxF64:
                return model(projection(x))
            return batched_compute(
                fused_fn,
                inputs,
                batch_size,
                desc=f"[Step 3 and 5 Proj+Extract] {split_name}",
                file="executor.py"
            ), None, None
        else:
            # No projection (already projected or no projection needed, or DistillationModel)
            fn = partial(model, split_name=None)
            return batched_compute(
                fn,
                inputs,
                batch_size,
                desc=f"[Step 5 Extracting] {split_name}",
                file="executor.py"
            ), None, None
