"""/home/yoshi/PycharmProjects/Reservoir/pipelines/generic_runner.py
Universal pipeline that treats models as feature extractors and owns the readout.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from reservoir.pipelines.config import ModelStack, FrontendContext, DatasetMetadata
from reservoir.models.presets import PipelineConfig
from reservoir.training.presets import TrainingConfig


class UniversalPipeline:
    """Runs the V2 flow: pre-train model -> extract features -> fit ridge -> evaluate."""

    def __init__(self, stack: ModelStack, config: PipelineConfig) -> None:
        self.model = stack.model
        self.readout = stack.readout
        self.metric_name = stack.metric
        self.config = config

    # ------------------------------------------------------------------ #
    # Utilities                                                          #
    # ------------------------------------------------------------------ #
    def _score(self, preds: jnp.ndarray, targets: jnp.ndarray) -> float:
        preds_arr = jnp.asarray(preds)
        targets_arr = jnp.asarray(targets)
        if self.metric_name == "accuracy":
            pred_labels = preds_arr if preds_arr.ndim == 1 else jnp.argmax(preds_arr, axis=-1)
            true_labels = targets_arr if targets_arr.ndim == 1 else jnp.argmax(targets_arr, axis=-1)
            return float(jnp.mean(pred_labels == true_labels))

        aligned_preds = preds_arr
        if preds_arr.shape != targets_arr.shape and preds_arr.size == targets_arr.size:
            aligned_preds = preds_arr.reshape(targets_arr.shape)
        return float(jnp.mean((aligned_preds - targets_arr) ** 2))

    def _feature_stats(self, features: Any, stage: str) -> None:
        # Prefer CPU stats to avoid host->device transfers when features are numpy.
        if isinstance(features, np.ndarray):
            feats = features
            stats = {
                "shape": feats.shape,
                "mean": float(np.mean(feats)),
                "std": float(np.std(feats)),
                "min": float(np.min(feats)),
                "max": float(np.max(feats)),
                "nans": int(np.isnan(feats).sum()),
            }
        else:
            feats = jnp.asarray(features, dtype=jnp.float64)
            stats = {
                "shape": feats.shape,
                "mean": float(jnp.mean(feats)),
                "std": float(jnp.std(feats)),
                "min": float(jnp.min(feats)),
                "max": float(jnp.max(feats)),
                "nans": int(jnp.isnan(feats).sum()),
            }
        print(
            f"[FeatureStats:{stage}] shape={stats['shape']}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}, nans={stats['nans']}"
        )
        if stats["std"] < 1e-6:
            print("Feature matrix has near-zero variance. Model output may be inactive.")

    def _extract_features(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if hasattr(self.model, "__call__"):
            try:
                return jnp.asarray(self.model(inputs))
            except TypeError:
                pass
        if hasattr(self.model, "predict"):
            return jnp.asarray(self.model.predict(inputs))
        raise AttributeError("Model must implement __call__ or predict to produce features.")

    def _extract_features_batched(self, inputs: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        arr = jnp.asarray(inputs)
        if batch_size is None or batch_size <= 0 or arr.shape[0] <= batch_size:
            return self._extract_features(arr)

        n = arr.shape[0]
        pad = (-n) % batch_size
        if pad == 0:
            arr_padded = arr
        else:
            pad_width = [(0, pad)] + [(0, 0)] * (arr.ndim - 1)
            arr_padded = jnp.pad(arr, pad_width)
        num_batches = arr_padded.shape[0] // batch_size
        reshaped = arr_padded.reshape((num_batches, batch_size) + arr.shape[1:])

        def batch_fn(batch):
            return self._extract_features(batch)

        mapped = jax.lax.map(batch_fn, reshaped)  # (num_batches, batch_size, ...)
        flat = mapped.reshape((arr_padded.shape[0],) + mapped.shape[2:])
        return flat[:n]

    def batch_transform(self, inputs: Any, batch_size: Optional[int] = None, *, to_numpy: bool = True) -> Any:
        """
        Public batched feature extraction that optionally moves results to CPU (numpy).
        """
        features = self._extract_features_batched(inputs, batch_size or 0)
        if to_numpy:
            return np.asarray(features)
        return features

    def _fit_readout(
        self,
        train_Z: jnp.ndarray,
        train_y: jnp.ndarray,
        val_Z: Optional[jnp.ndarray],
        val_y: Optional[jnp.ndarray],
        ridge_lambdas: Optional[Sequence[float]],
    ) -> tuple[float, Dict[float, float], Dict[float, float]]:
        lambda_candidates = list(ridge_lambdas) if ridge_lambdas is not None else []
        config_init = getattr(getattr(self, "config", None), "readout", None)
        initial_lambda = float(getattr(config_init, "init_lambda", getattr(self.readout, "ridge_lambda", 1.0)))
        single_candidate = lambda_candidates[0] if len(lambda_candidates) == 1 else None

        # Tier 1: no validation or no search space (empty/single) -> direct fit.
        direct_fit = val_Z is None or val_y is None or len(lambda_candidates) <= 1
        if direct_fit:
            chosen = float(single_candidate if single_candidate is not None else initial_lambda)
            self.readout.ridge_lambda = chosen
            if val_Z is None or val_y is None:
                print("No validation set provided. Skipping hyperparameter search to prevent overfitting.")
            elif len(lambda_candidates) <= 1:
                print("Single ridge_lambda candidate provided. Running direct fit without search.")
            self.readout.fit(train_Z, train_y)
            return chosen, {}, {}

        # Tier 2: validation present and multiple candidates -> search.
        print(
            f"Search active: overriding initial ridge_lambda={initial_lambda} with {len(lambda_candidates)} candidates."
        )
        best_lambda, search_history, weight_norms = self.readout.fit_and_search(
            train_Z,
            train_y,
            val_Z,
            val_y,
            lambda_candidates,
            metric=self.metric_name,
        )
        return best_lambda, search_history, weight_norms

    # ------------------------------------------------------------------ #
    # Run                                                               #
    # ------------------------------------------------------------------ #
    def run(
        self,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata,
    ) -> Dict[str, Dict[str, Any]]:
        processed = frontend_ctx.processed_split
        train_X = processed.train_X
        train_y = processed.train_y
        test_X = processed.test_X
        test_y = processed.test_y

        cfg = dataset_meta.training or TrainingConfig()
        ridge_lambdas = cfg.ridge_lambdas
        feature_batch_size = int(cfg.batch_size or 0)

        start = time.time()

        print(f"\n=== Step 5: Model Dynamics (Training/Warmup) [{self.config.model_type.value}] ===")
        start_train = time.time()
        train_logs = self.model.train(train_X, train_y) or {}
        train_time = time.time() - start_train

        final_loss = train_logs.get("final_loss") or train_logs.get("final_mse") or train_logs.get("loss")
        if final_loss is not None:
            print(f"[Step 5] Model Dynamics completed in {train_time:.2f}s. Final Loss: {final_loss}")
        else:
            print(f"[Step 5] Model Dynamics completed in {train_time:.2f}s.")

        print("\n=== Step 6: Aggregation (Feature Extraction) ===")
        train_features = self.batch_transform(train_X, batch_size=feature_batch_size)
        self._feature_stats(train_features, "post_train_features")

        val_Z = None
        val_y = processed.val_y
        if processed.val_X is not None and processed.val_y is not None:
            val_Z = self.batch_transform(processed.val_X, batch_size=feature_batch_size)
            self._feature_stats(val_Z, "post_val_features")

        test_features = self.batch_transform(test_X, batch_size=feature_batch_size)
        self._feature_stats(test_features, "post_test_features")

        print("\n=== Step 7: Readout (Ridge Regression) ===")

        if self.readout is None: #for end-to-end models without readout
            print("Readout is None. Using model output directly as predictions (End-to-End mode).")
            # Readout学習はスキップ
            best_lambda = None
            search_history = {}
            weight_norms = {}
            # Step 6 の出力をそのまま予測値とする
            test_pred = test_features

            results = {
                "train": {"best_lambda": best_lambda, "search_history": search_history, "weight_norms": weight_norms},
                "test": {self.metric_name: self._score(test_pred, test_y)},
            }

        else:
            best_lambda, search_history, weight_norms = self._fit_readout(
                train_features,
                train_y,
                val_Z,
                val_y,
                ridge_lambdas,
            )
            if search_history:
                print(f"Search complete. Best lambda: {best_lambda}")
            self.readout.ridge_lambda = best_lambda

            # Evaluate
            test_pred = self.readout.predict(test_features)

            results = {
                "train": {"best_lambda": best_lambda, "search_history": search_history, "weight_norms": weight_norms},
                "test": {self.metric_name: self._score(test_pred, test_y)},
            }

            if val_Z is not None and val_y is not None:
                results["validation"] = {self.metric_name: self._score(self.readout.predict(val_Z), val_y)}
            results["readout"] = self.readout

        results["training_logs"] = train_logs
        elapsed = time.time() - start
        results["meta"] = {"metric": self.metric_name, "elapsed_sec": elapsed, "pretrain_sec": train_time}


        return results
