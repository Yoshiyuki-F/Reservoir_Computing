"""/home/yoshi/PycharmProjects/Reservoir/pipelines/generic_runner.py
Universal pipeline that treats models as feature extractors and owns the readout.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import asdict

from reservoir.readout.ridge import RidgeRegression
from reservoir.training.presets import TrainingConfig


class UniversalPipeline:
    """Runs the V2 flow: pre-train model -> extract features -> fit ridge -> evaluate."""

    def __init__(
        self,
        model: Any,
        readout: RidgeRegression,
        save_path: Optional[Path | str] = None,
        *,
        metric: str = "mse",
    ) -> None:
        if readout is None:
            raise ValueError("UniversalPipeline requires a readout instance (RidgeRegression).")
        self.model = model
        self.readout = readout
        self.save_path = Path(save_path) if save_path is not None else None
        self.metric_name = metric

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

    def _feature_stats(self, features: jnp.ndarray, stage: str) -> None:
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
        initial_lambda = float(self.readout.ridge_lambda)

        # No validation -> skip search to avoid overfitting
        if val_Z is None or val_y is None:
            print("No validation set provided. Skipping hyperparameter search to prevent overfitting.")
            self.readout.fit(train_Z, train_y)
            return float(initial_lambda), {}, {}

        if not lambda_candidates:
            lambda_candidates = [initial_lambda]
        else:
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
        train_X: Any,
        train_y: Any,
        test_X: Any,
        test_y: Any,
        *,
        validation: Optional[tuple[Any, Any]] = None,
        training_cfg: Optional[TrainingConfig] = None,
        training_extras: Optional[Dict[str, Any]] = None,
        model_label: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        train_X = jnp.asarray(train_X)
        train_y = jnp.asarray(train_y)
        test_X = jnp.asarray(test_X)
        test_y = jnp.asarray(test_y)

        cfg = training_cfg or TrainingConfig()
        extras = dict(training_extras or {})
        model_label = model_label or self.model.__class__.__name__
        ridge_lambdas = cfg.ridge_lambdas
        feature_batch_size = int(extras.get("feature_batch_size", cfg.batch_size or 0))

        cfg_dict = asdict(cfg)
        # Model owns its TrainingConfig (injected via Factory). Do not pass duplicate params to train().
        exclude_keys = {
            "ridge_lambda",
            "ridge_lambdas",
            "batch_size",
            "epochs",
            "learning_rate",
            "seed",
            "classification",
            "train_size",
            "val_size",
            "test_ratio",
            "task_type",
            "name",
        }
        train_params = {k: v for k, v in cfg_dict.items() if k not in exclude_keys}

        start = time.time()

        print(f"\n=== Step 5: Train Model ({model_label}) ===")
        start_train = time.time()
        train_logs = self.model.train(train_X, train_y, **train_params) or {}
        train_time = time.time() - start_train

        final_loss = train_logs.get("final_loss") or train_logs.get("final_mse") or train_logs.get("loss")
        if final_loss is not None:
            print(f"Model training completed in {train_time:.2f}s. Final Loss: {final_loss}")
        else:
            print(f"Model training completed in {train_time:.2f}s.")

        print("\n=== Step 6: Extract Features (Aggregated Model Output) ===")
        train_features = self.batch_transform(train_X, batch_size=feature_batch_size)
        self._feature_stats(train_features, "post_train_features")

        val_Z = None
        val_y = None
        if validation:
            val_X, val_y = validation
            val_Z = self.batch_transform(val_X, batch_size=feature_batch_size)

        test_features = self.batch_transform(test_X, batch_size=feature_batch_size)

        print("\n=== Step 7: Fit Readout (Ridge Regression) ===")
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
        train_pred = self.readout.predict(train_features)
        test_pred = self.readout.predict(test_features)

        results = {
            "train": {"best_lambda": best_lambda, "search_history": search_history, "weight_norms": weight_norms},
            "test": {self.metric_name: self._score(test_pred, test_y)},
        }
        if validation:
            results["validation"] = {self.metric_name: self._score(self.readout.predict(val_Z), val_y)}

        elapsed = time.time() - start
        results["meta"] = {"metric": self.metric_name, "elapsed_sec": elapsed, "pretrain_sec": train_time}

        # Optional: save model/readout if a path is provided
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            # Placeholder for save logic

        return results
