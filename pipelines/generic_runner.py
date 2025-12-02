"""/home/yoshi/PycharmProjects/Reservoir/pipelines/generic_runner.py
Universal pipeline that treats models as feature extractors and owns the readout.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import jax.numpy as jnp

from reservoir.components.readout.ridge import RidgeRegression


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
            print("⚠️ Feature matrix has near-zero variance. Model output may be inactive.")

    def _extract_features(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if hasattr(self.model, "__call__"):
            try:
                return jnp.asarray(self.model(inputs))
            except TypeError:
                pass
        if hasattr(self.model, "predict"):
            return jnp.asarray(self.model.predict(inputs))
        raise AttributeError("Model must implement __call__ or predict to produce features.")

    def _fit_readout(
        self,
        train_Z: jnp.ndarray,
        train_y: jnp.ndarray,
        val_Z: Optional[jnp.ndarray],
        val_y: Optional[jnp.ndarray],
        ridge_lambdas: Optional[Sequence[float]],
    ) -> tuple[float, Dict[float, float]]:
        lambdas = list(ridge_lambdas) if ridge_lambdas is not None else []
        if not lambdas:
            lambdas = [float(getattr(self.readout, "alpha", 1.0))]

        best_lambda: float = float(lambdas[0])
        best_score: Optional[float] = None
        best_readout: Optional[RidgeRegression] = None
        search_history: Dict[float, float] = {}

        for lam in lambdas:
            candidate = RidgeRegression(alpha=float(lam), use_intercept=getattr(self.readout, "use_intercept", True))
            candidate.fit(train_Z, train_y)

            eval_Z = val_Z if val_Z is not None else train_Z
            eval_y = val_y if val_y is not None else train_y
            score = self._score(candidate.predict(eval_Z), eval_y)
            search_history[float(lam)] = score

            if best_score is None:
                best_score = score
                best_lambda = float(lam)
                best_readout = candidate
            else:
                if self.metric_name == "accuracy":
                    if score > best_score:
                        best_score = score
                        best_lambda = float(lam)
                        best_readout = candidate
                else:
                    if score < best_score:
                        best_score = score
                        best_lambda = float(lam)
                        best_readout = candidate

        if best_readout is None:
            best_readout = RidgeRegression(alpha=float(best_lambda), use_intercept=getattr(self.readout, "use_intercept", True))
            best_readout.fit(train_Z, train_y)

        self.readout = best_readout
        return best_lambda, search_history

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
        training_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        train_X = jnp.asarray(train_X)
        train_y = jnp.asarray(train_y)
        test_X = jnp.asarray(test_X)
        test_y = jnp.asarray(test_y)

        extra_kwargs = dict(training_cfg or {})
        dataset_name = str(extra_kwargs.pop("_meta_dataset", "dataset"))
        model_label = str(extra_kwargs.pop("_meta_model_type", self.model.__class__.__name__))
        ridge_lambdas = extra_kwargs.pop("ridge_lambdas", None)

        start = time.time()

        # Phase 1: Pre-training
        print(f"\n=== [Phase 1] Pre-training Model ({model_label}) ===")
        start_train = time.time()
        train_logs = self.model.train(train_X, train_y, validation=validation, metric=self.metric_name, **extra_kwargs) or {}
        train_time = time.time() - start_train
        final_loss = train_logs.get("final_loss") or train_logs.get("final_mse") or train_logs.get("loss")
        print(
            f"Pre-training completed. Duration: {train_time:.2f}s."
            + (f" Final Loss: {final_loss}" if final_loss is not None else "")
        )

        # Phase 2: Feature Extraction
        print("\n=== [Phase 2] Generating Features (Design Matrix) ===")
        train_Z = self._extract_features(train_X)
        test_Z = self._extract_features(test_X)
        val_Z = None
        val_y = None
        if validation is not None:
            val_X, val_targets = validation
            val_Z = self._extract_features(val_X)
            val_y = jnp.asarray(val_targets)
        self._feature_stats(train_Z, stage="post_train_features")

        # Phase 3: Readout Training
        print("\n=== [Phase 3] Ridge Regression (Readout Training) ===")
        if ridge_lambdas is None:
            print("Training with default lambda (no search provided).")
        else:
            try:
                num_candidates = len(list(ridge_lambdas))
                if num_candidates > 1:
                    print(f"Searching optimal lambda over {num_candidates} candidates...")
                else:
                    first_val = list(ridge_lambdas)[0]
                    print(f"Training with fixed lambda: {first_val}...")
            except Exception:
                print("Training with provided ridge_lambdas configuration.")

        best_lambda, search_history = self._fit_readout(train_Z, train_y, val_Z, val_y, ridge_lambdas)
        print(f"Readout training completed. Best lambda: {best_lambda}")

        # Phase 4: Evaluation
        train_pred = self.readout.predict(train_Z)
        test_pred = self.readout.predict(test_Z)
        val_pred = self.readout.predict(val_Z) if val_Z is not None else None

        train_metric = self._score(train_pred, train_y)
        test_metric = self._score(test_pred, test_y)
        val_metric = self._score(val_pred, val_y) if val_pred is not None and val_y is not None else None

        if self.save_path is not None and hasattr(self.model, "save"):
            self.model.save(self.save_path)

        elapsed = time.time() - start

        results: Dict[str, Dict[str, float]] = {
            "train": {
                self.metric_name: train_metric,
                "best_lambda": best_lambda,
                "search_history": search_history,
            },
            "test": {self.metric_name: test_metric},
        }
        if val_metric is not None:
            results["validation"] = {self.metric_name: val_metric}
        if train_logs:
            results["training_logs"] = train_logs
        results["meta"] = {"metric": self.metric_name, "elapsed_sec": elapsed, "pretrain_sec": train_time}
        # Expose fitted readout for downstream consumers (e.g., plotting)
        results["readout"] = self.readout
        return results
