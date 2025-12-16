"""/home/yoshi/PycharmProjects/Reservoir/pipelines/generic_runner.py
Universal pipeline that treats models as feature extractors and owns the readout.
Updated to distinguish between Aggregation (Reservoir) and Inference (FNN/Distillation) in logs.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from reservoir.pipelines.config import ModelStack, FrontendContext, DatasetMetadata
from reservoir.models.presets import PipelineConfig

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

    def _extract_features_batched(self, inputs: Any, batch_size: int) -> np.ndarray:
        """
        CPUメモリ(Numpy)上のデータを少しずつGPU(JAX)に送り、計算結果を再びCPUに戻す。
        巨大なデータセットでVRAM不足(OOM)を防ぐための実装。
        """
        # 1. 入力をNumpy配列として確保 (絶対に jnp.array(inputs) してはいけない)
        inputs_np = np.asarray(inputs)
        n_samples = inputs_np.shape[0]

        # print(f"    [FeatureExtraction] CPU Manual Batching (Batch Size: {batch_size}, Total: {n_samples})")

        if n_samples == 0:
            return np.array([])

        # 2. 出力の形状と型を決定するためのダミー実行 (最初の1件だけGPUで計算)
        # JITコンパイルのトリガーにもなります
        dummy_in = jnp.array(inputs_np[:1])
        dummy_out = self._extract_features(dummy_in)

        # 出力形状の計算: (54000, Features...)
        out_shape = (n_samples,) + dummy_out.shape[1:]
        out_dtype = dummy_out.dtype

        # 3. 結果格納用の巨大なNumpy配列をCPUメモリに確保
        output = np.empty(out_shape, dtype=out_dtype)

        # 4. 高速化用JIT関数
        @jax.jit
        def step(batch_in):
            return self._extract_features(batch_in)

        # 5. CPUループ実行 (tqdm適用)
        # ログが冗長にならないよう、descをシンプルに
        with tqdm(total=n_samples, desc="[Pipeline] Extracting", unit="samples") as pbar:
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                current_batch_size = batch_end - i

                # (A) CPUでスライス
                batch_in_cpu = inputs_np[i: batch_end]

                # (B) GPUへ転送 & 計算
                batch_out_jax = step(jnp.array(batch_in_cpu))

                # (C) 結果を即座にCPUへ戻す
                output[i: batch_end] = np.asarray(batch_out_jax)

                # プログレスバーを進める
                pbar.update(current_batch_size)

        return output

    def batch_transform(self, inputs: Any, batch_size: int, to_numpy: bool = True) -> Any:
        """
        Public batched feature extraction that optionally moves results to CPU (numpy).
        """
        features = self._extract_features_batched(inputs, batch_size)
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
            if train_Z.ndim == 3:
                train_Z = train_Z.reshape(-1, train_Z.shape[-1])
            if train_y.ndim == 3:
                train_y = train_y.reshape(-1, train_y.shape[-1])
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

        cfg = dataset_meta.training
        ridge_lambdas = cfg.ridge_lambdas
        feature_batch_size = int(cfg.batch_size)

        start = time.time()

        print(f"\n=== Step 5: Model Dynamics (Training/Warmup) [{self.config.model_type.value}] ===")
        start_train = time.time()
        train_logs = self.model.train(train_X, train_y) or {}
        train_time = time.time() - start_train

        # ログ出力
        final_loss = train_logs.get("final_loss") or train_logs.get("final_mse") or train_logs.get("loss")
        if final_loss is not None:
            print(f"[Step 5] Model Dynamics completed in {train_time:.2f}s. Final Loss: {final_loss}")
        else:
            print(f"[Step 5] Model Dynamics completed in {train_time:.2f}s.")

        # --- ログ表示の分岐ロジック ---
        # モデル名に "FNN" や "Distillation" が含まれる場合は Aggregation ではなく Inference とみなす
        model_cls_name = self.model.__class__.__name__
        is_static_model = "FNN" in model_cls_name or "Distillation" in model_cls_name or "CNN" in model_cls_name

        if is_static_model:
            print("\n=== Step 6: Feature Extraction (Inference) ===")
            print("    Generating static features for Readout...")
        else:
            print("\n=== Step 6: Aggregation (Time-Collapse) ===")
            print("    Aggregating temporal states...")

        # 特徴量抽出 (1回目かつ最後にしたい)
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

        results = {}
        train_pred = None
        test_pred = None
        val_pred = None

        if self.readout is None:
            print("Readout is None. Using model output directly as predictions (End-to-End mode).")
            # End-to-Endの場合は特徴量＝予測値とみなす
            best_lambda = None
            search_history = {}
            weight_norms = {}

            train_pred = train_features
            test_pred = test_features
            if val_Z is not None:
                val_pred = val_Z

            results["test"] = {self.metric_name: self._score(test_pred, test_y)}

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

            # Evaluate & Store Predictions
            print("Generating final predictions...")
            train_pred = self.readout.predict(train_features)
            test_pred = self.readout.predict(test_features)

            results["train"] = {
                "best_lambda": best_lambda,
                "search_history": search_history,
                "weight_norms": weight_norms
            }
            results["test"] = {self.metric_name: self._score(test_pred, test_y)}

            if val_Z is not None and val_y is not None:
                val_pred = self.readout.predict(val_Z)
                results["validation"] = {self.metric_name: self._score(val_pred, val_y)}

            results["readout"] = self.readout

        results["outputs"] = {
            "train_pred": train_pred,
            "test_pred": test_pred,
            "val_pred": val_pred
        }

        results["training_logs"] = train_logs
        elapsed = time.time() - start
        results["meta"] = {"metric": self.metric_name, "elapsed_sec": elapsed, "pretrain_sec": train_time}

        del train_features, test_features, val_Z

        return results