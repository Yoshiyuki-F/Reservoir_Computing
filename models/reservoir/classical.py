"""
Classical Reservoir Computer implementation using JAX.
"""

import jax
import numpy as np
import json
from pathlib import Path
from functools import lru_cache

from typing import Optional, Dict, Any, Sequence, Iterable

from pipelines.jax_config import ensure_x64_enabled

ensure_x64_enabled()

import jax.numpy as jnp
from jax import random, lax, device_put

from tqdm.auto import tqdm

def tqdm(iterable, *args, **kwargs):
    return iterable

from .base_reservoir import BaseReservoirComputer
from .config import ReservoirConfig, parse_ridge_lambdas


@lru_cache()
def _load_shared_defaults() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "configs/models/shared_reservoir_params.json"
    data = json.loads(path.read_text())
    return dict(data.get("params", {}))

class ReservoirComputer(BaseReservoirComputer):
    """JAXベースのEcho State Network (ESN) 実装。
    Attributes:
        config: リザーバーの設定パラメータ
        backend: 計算バックエンド種別（'cpu' or 'gpu'）
        W_in: 入力重み行列 (n_reservoir, n_inputs)
        W_res: リザーバー重み行列 (n_reservoir, n_reservoir)
        W_out: 出力重み行列 (n_reservoir+1, n_outputs) 訓練後に設定
    """
    
    def __init__(self, config: Sequence[Dict[str, Any]], backend: Optional[str] = None):
        """
        Reservoir Computerを初期化します。

        Args:
            config: ReservoirConfigオブジェクト
            backend: 計算バックエンド ('cpu', 'gpu', または None で自動選択)
        """
        super().__init__()  # Initialize base class

        # Merge shared defaults with user config
        merged: Dict[str, Any] = _load_shared_defaults().copy()
        config_sequence: Iterable[Dict[str, Any]] = [config] if isinstance(config, dict) else config  # type: ignore[arg-type]
        for cfg in config_sequence:
            cfg_dict = dict(cfg)
            merged.update({k: v for k, v in cfg_dict.items() if k not in {'name', 'description', 'params'}})
            params = cfg_dict.get('params', {}) or {}
            merged.update(params)

        # Create config object - this performs all validation
        cfg = ReservoirConfig(**merged)

        self.config = cfg

        # Extract validated parameters
        params = cfg.params

        self.n_inputs: int = params['n_inputs']
        self.n_reservoir: int = params['n_reservoir']
        self.n_outputs: int = params['n_outputs']
        self.spectral_radius: float = float(params['spectral_radius'])
        self.input_scaling: float = float(params['input_scaling'])
        self.noise_level: float = float(params['noise_level'])
        self.alpha: float = float(params['alpha'])
        self.reservoir_weight_range: float = float(params['reservoir_weight_range'])
        self.sparsity: float = float(params['sparsity'])
        self.input_bias: float = float(params['input_bias'])
        self.nonlinearity: str = str(params['nonlinearity'])
        random_seed: int = int(params['random_seed'])
        self.state_aggregation: str = str(params.get('state_aggregation', 'last')).lower()

        # 乱数キーの初期化
        self.key = random.PRNGKey(random_seed)

        # バックエンドの設定（CLI層で確認済み）
        self.backend = backend

        # reservoirの重みを初期化
        self._initialize_weights()

        # 出力重みは後で訓練で設定
        self.W_out = None
        self.best_ridge_lambda: Optional[float] = None
        self.ridge_search_log: list[Dict[str, float]] = []
        self.last_training_mse: Optional[float] = None
        self.classification_mode: bool = False
        self.num_classes: Optional[int] = None

        self.initial_random_seed = random_seed
        self.washout_steps: int = 3

        # Parse ridge_lambdas using common validation function
        self.ridge_lambdas: Sequence[float] = parse_ridge_lambdas(params)

        # Feature normalization state
        self._feature_mu_: Optional[np.ndarray] = None
        self._feature_sigma_: Optional[np.ndarray] = None
        self._feature_keep_mask_: Optional[np.ndarray] = None
        self._design_keep_mask_: Optional[np.ndarray] = None


    def _initialize_weights(self):
        """reservoirの重みを初期化します。JAXのデフォルトデバイス配置を使用。"""
        key1, key2, new_key = random.split(self.key, 3)
        
        W_in = random.uniform(
            key1, 
            (self.n_reservoir, self.n_inputs), 
            minval=-self.input_scaling, 
            maxval=self.input_scaling,
            dtype=jnp.float64
        )
        
        W_res = random.uniform(
            key2, 
            (self.n_reservoir, self.n_reservoir), 
            minval=-self.reservoir_weight_range,
            maxval=self.reservoir_weight_range,
            dtype=jnp.float64
        )
        
        # スペクトル半径の調整（NumPyで安定した計算を行う）
        W_res_np = np.array(W_res)
        try:
            eigenvalues = np.linalg.eigvals(W_res_np)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            max_eigenvalue = max(max_eigenvalue, 1e-8)
            W_res_scaled = (self.spectral_radius / max_eigenvalue) * W_res_np
        except:
            frobenius_norm = np.linalg.norm(W_res_np, 'fro')
            frobenius_norm = max(frobenius_norm, 1e-8)
            W_res_scaled = (self.spectral_radius / frobenius_norm) * W_res_np
        
        self.W_in = W_in
        self.W_res = jnp.array(W_res_scaled)
        self.key = new_key
        
    def _reservoir_step(self, carry, input_data):
        """reservoirの1ステップを実行します（JAX scan用）。"""
        state, key = carry # 前の状態 h(t-1)
        key, subkey = random.split(key)
        
        noise = random.normal(subkey, (self.n_reservoir,), dtype=jnp.float64) * self.noise_level
        
        # 新しい状態 h(t) を計算
        res_contribution = jnp.dot(self.W_res, state)      # W_res * h(t-1)
        input_contribution = jnp.dot(self.W_in, input_data) # W_in * u(t)
        
        pre_activation = res_contribution + input_contribution + noise
        new_state = (1 - self.alpha) * state + self.alpha * jnp.tanh(pre_activation)
        
        return (new_state, key), new_state  # h(t)を返す
        
    def run_reservoir(self, input_sequence: jnp.ndarray) -> jnp.ndarray:
        """
        入力シーケンスに対してreservoirを実行します。
        
        Args:
            input_sequence: 形状 (time_steps, n_inputs) の入力データ
            
        Returns:
            reservoir状態のシーケンス (time_steps, n_reservoir)
        """
        # 入力データをfloat64に変換
        input_sequence = input_sequence.astype(jnp.float64)
        
        # 初期状態
        initial_state = jnp.zeros(self.n_reservoir, dtype=jnp.float64)
        initial_carry = (initial_state, self.key)
        
        # JAXのscanを使用して効率的に計算
        carry, states = lax.scan(self._reservoir_step, initial_carry, input_sequence)
        
        # キーを更新
        _, self.key = carry
        
        return states

    @staticmethod
    @jax.jit
    def _train_unified(X: jnp.ndarray, target_data: jnp.ndarray, ridge_lambda: float) -> jnp.ndarray:
        """統合されたRidge回帰訓練（JAXのデフォルトデバイス配置を使用）"""
        XTX = X.T @ X
        XTY = X.T @ target_data

        A = XTX + ridge_lambda * jnp.eye(XTX.shape[0], dtype=jnp.float64)

        try:
            return jnp.linalg.solve(A, XTY)
        except:
            return jnp.linalg.pinv(A) @ XTY

    def _prepare_design_matrix(
        self,
        feature_matrix: jnp.ndarray,
        fit: bool = False,
        washout: bool = False,
    ) -> jnp.ndarray:
        """Standardize features and filter zero-variance columns."""
        features_np = np.asarray(feature_matrix, dtype=np.float64)

        if features_np.ndim != 2:
            features_np = features_np.reshape(features_np.shape[0], -1)

        if washout and features_np.shape[0] > getattr(self, "washout_steps", 0):
            features_np = features_np[self.washout_steps :, ...]

        eps = 1e-8
        if fit or self._feature_mu_ is None or self._feature_sigma_ is None:
            mu = features_np.mean(axis=0)
            sigma = features_np.std(axis=0)
            sigma_adj = sigma + eps
            keep = np.ones_like(sigma_adj, dtype=bool)
            self._feature_mu_ = mu
            self._feature_sigma_ = sigma_adj
            self._feature_keep_mask_ = keep

            kept_sigma = sigma_adj[keep]
            print(
                f"[classical-rc] feature std range (kept) -> min={float(kept_sigma.min()):.3e}, max={float(kept_sigma.max()):.3e}"
            )
        else:
            if self._feature_mu_ is None or self._feature_sigma_ is None or self._feature_keep_mask_ is None:
                raise RuntimeError("Feature scaler has not been fitted. Call train() before predict().")
            mu = self._feature_mu_
            sigma_adj = self._feature_sigma_
            keep = self._feature_keep_mask_

        centered = (features_np - mu) / sigma_adj
        centered = centered[:, keep]

        expanded = np.concatenate([centered, centered**2], axis=1)
        bias = np.ones((expanded.shape[0], 1), dtype=np.float64)
        design = np.concatenate([expanded, bias], axis=1)

        if fit or self._design_keep_mask_ is None:
            col_std = design.std(axis=0)
            design_keep = np.ones(design.shape[1], dtype=bool)
            if design.shape[1] > 1:
                design_keep[:-1] = col_std[:-1] > 1e-3
            design_keep[-1] = True
            self._design_keep_mask_ = design_keep
        else:
            if self._design_keep_mask_ is None:
                raise RuntimeError("Design matrix filter has not been fitted. Call train() before predict().")
            design_keep = self._design_keep_mask_

        design = design[:, design_keep]
        return jnp.array(design, dtype=jnp.float64)

    def _fit_ridge_with_grid(
        self,
        X: jnp.ndarray,
        target_data: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
        use_kfold: bool = False,
        n_folds: int = 5,
        classification_mode: bool = False,
    ) -> None:
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(target_data, dtype=np.float64)

        lambdas = ridge_lambdas or self.ridge_lambdas
        lambda_candidates = [float(lam) for lam in lambdas if lam is not None and lam > 0]

        if not lambda_candidates:
            lambda_candidates = list(self.ridge_lambdas)

        lambda_candidates = sorted(set(lambda_candidates))

        n_samples = X_np.shape[0]
        if n_samples < 2:
            raise ValueError("Not enough samples to perform ridge regression")

        # Feature statistics on full training data
        feature_part = X_np[:, :-1]
        if feature_part.size:
            sigma = feature_part.std(axis=0)
            print(
                f"[classical-rc] feature std range (full) -> min={float(sigma.min()):.3e}, max={float(sigma.max()):.3e}"
            )
            try:
                sv = np.linalg.svd(feature_part, compute_uv=False)
                cond_number = float(sv.max() / max(sv.min(), 1e-12))
                print(f"[classical-rc] design matrix singular values -> {sv}")
                print(f"[classical-rc] design matrix cond -> {cond_number:.3e}")
            except np.linalg.LinAlgError:
                print("[classical-rc] SVD failed; skipping condition number log.")

        def ridge_with_bias(design: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
            if design.shape[1] == 0:
                raise ValueError("Design matrix must have at least one column (bias term expected).")

            features = design[:, :-1]
            if features.size == 0:
                bias = targets.mean(axis=0, keepdims=False, dtype=np.float64)
                return np.vstack([np.zeros((0, targets.shape[1]), dtype=np.float64), bias])

            U, s, Vt = np.linalg.svd(features, full_matrices=False)
            UT_y = U.T @ targets
            denom = s * s + lam
            filt = (s / denom)[:, None]
            weights = Vt.T @ (filt * UT_y)
            bias = (targets - features @ weights).mean(axis=0, keepdims=False, dtype=np.float64)
            return np.vstack([weights, bias])

        metric_name = "accuracy" if classification_mode else "MSE"
        metric_fmt = ".4f" if classification_mode else ".6e"

        def compute_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            if classification_mode:
                true_labels = np.argmax(y_true, axis=1)
                pred_labels = np.argmax(y_pred, axis=1)
                return float(np.mean(true_labels == pred_labels))
            diff = y_true - y_pred
            return float(np.mean(diff * diff))

        choose_best = np.argmax if classification_mode else np.argmin

        if use_kfold and n_samples >= n_folds:
            # K-Fold Cross Validation
            print(f"[classical-rc] Using {n_folds}-Fold CV for lambda selection")
            fold_size = n_samples // n_folds
            cv_scores: Dict[float, list[float]] = {lam: [] for lam in lambda_candidates}

            lambda_iter = tqdm(
                lambda_candidates,
                desc=f"Ridge λ search ({n_folds}-Fold CV)",
                leave=False,
                unit="λ",
            )
            for lam in lambda_iter:
                for fold in range(n_folds):
                    val_start = fold * fold_size
                    val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples

                    val_indices = list(range(val_start, val_end))
                    train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))

                    X_fold_train = X_np[train_indices]
                    y_fold_train = y_np[train_indices]
                    X_fold_val = X_np[val_indices]
                    y_fold_val = y_np[val_indices]

                    weights = ridge_with_bias(X_fold_train, y_fold_train, lam)
                    y_pred = X_fold_val @ weights
                    fold_score = compute_metric(y_fold_val, y_pred)
                    cv_scores[lam].append(fold_score)

            # Average CV scores
            val_scores = [float(np.mean(cv_scores[lam])) for lam in lambda_candidates]
            train_scores = val_scores.copy()  # Placeholder

            best_index = int(choose_best(val_scores))
            best_lambda = lambda_candidates[best_index]

            print(f"🔍 Ridge λ grid search ({n_folds}-Fold CV)")
            for lam, cv_score in zip(lambda_candidates, val_scores):
                print(f"  λ={lam:.2e} -> CV {metric_name}={cv_score:{metric_fmt}}")

            # Retrain on full data with best lambda
            print(f"[classical-rc] Retraining on full data with λ*={best_lambda:.2e}")
            best_weights_np = ridge_with_bias(X_np, y_np, best_lambda)

        else:
            # Simple 90/10 split
            split_idx = int(0.9 * n_samples)
            split_idx = max(1, min(split_idx, n_samples - 1))

            X_train, X_val = X_np[:split_idx], X_np[split_idx:]
            y_train, y_val = y_np[:split_idx], y_np[split_idx:]

            train_scores: list[float] = []
            val_scores: list[float] = []
            weights_by_lambda: Dict[float, np.ndarray] = {}

            lambda_iter = tqdm(
                lambda_candidates,
                desc="Ridge λ search (VAL)",
                leave=False,
                unit="λ",
            )
            for lam in lambda_iter:
                weights = ridge_with_bias(X_train, y_train, lam)
                yhat_tr = X_train @ weights
                train_score = compute_metric(y_train, yhat_tr)

                if X_val.size > 0:
                    yhat_val = X_val @ weights
                    val_score = compute_metric(y_val, yhat_val)
                else:
                    val_score = train_score

                train_scores.append(train_score)
                val_scores.append(val_score)
                weights_by_lambda[lam] = weights

            best_index = int(choose_best(val_scores))
            best_lambda = lambda_candidates[best_index]
            best_weights_np = weights_by_lambda[best_lambda]

            print("🔍 Ridge λ grid search (VAL)")
            for lam, val_score in zip(lambda_candidates, val_scores):
                print(f"  λ={lam:.2e} -> val {metric_name}={val_score:{metric_fmt}}")

            # Retrain on full train+val with best lambda
            print(f"[classical-rc] Retraining on train+val with λ*={best_lambda:.2e}")
            best_weights_np = ridge_with_bias(X_np, y_np, best_lambda)

        self.W_out = jnp.array(best_weights_np, dtype=jnp.float64)
        self.best_ridge_lambda = best_lambda
        self.last_training_mse = float(val_scores[best_index])

        metric_key_train = "train_accuracy" if classification_mode else "train_mse"
        metric_key_val = "val_accuracy" if classification_mode else "val_mse"

        self.ridge_search_log = []
        for i, lam in enumerate(lambda_candidates):
            if i >= len(val_scores):
                break
            train_val_score = train_scores[i] if i < len(train_scores) else val_scores[i]
            entry = {
                "lambda": lam,
                metric_key_train: train_val_score,
                metric_key_val: val_scores[i],
            }
            if classification_mode:
                entry["train_mse"] = train_val_score
                entry["val_mse"] = val_scores[i]
            self.ridge_search_log.append(entry)

    def _encode_sequences(
        self,
        sequences: jnp.ndarray,
        desc: Optional[str] = None,
    ) -> jnp.ndarray:
        """Run reservoir on sequences and collect final states."""
        encoded_states = []
        try:
            total = int(sequences.shape[0])
        except Exception:  # pragma: no cover - dynamic shapes
            total = None

        iterator = tqdm(
            sequences,
            desc=desc or "Encoding sequences",
            total=total,
            leave=False,
            unit="seq",
        )
        for seq in iterator:
            seq_arr = jnp.array(seq, dtype=jnp.float64)
            states = self.run_reservoir(seq_arr)
            encoded_states.append(self._aggregate_states(states))
        return jnp.stack(encoded_states, axis=0)

    def _aggregate_states(self, states: jnp.ndarray) -> jnp.ndarray:
        if self.state_aggregation == 'last':
            return states[-1]
        if self.state_aggregation == 'mean':
            return jnp.mean(states, axis=0)
        if self.state_aggregation in {'last_mean', 'mts'}:
            last = states[-1]
            mean = jnp.mean(states, axis=0)
            return jnp.concatenate([last, mean], axis=0)
        # concat
        return states.reshape(-1)

    def train_classification(
        self,
        sequences: jnp.ndarray,
        labels: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
        num_classes: int = 10,
    ) -> None:
        features = self._encode_sequences(sequences, desc="Encoding train sequences")
        design_matrix = self._prepare_design_matrix(features, fit=True)

        labels = labels.astype(jnp.int32)
        targets = jnp.zeros((labels.shape[0], num_classes), dtype=jnp.float64)
        targets = targets.at[jnp.arange(labels.shape[0]), labels].set(1.0)

        self._fit_ridge_with_grid(
            design_matrix,
            targets,
            ridge_lambdas,
            classification_mode=True,
        )
        self.classification_mode = True
        self.num_classes = num_classes
        self.trained = True

    def predict_classification(self, sequences: jnp.ndarray) -> jnp.ndarray:
        if not self.classification_mode or self.num_classes is None:
            raise ValueError("Classification mode not enabled. Call train_classification first.")
        if self.W_out is None:
            raise ValueError("Model has not been trained.")

        features = self._encode_sequences(sequences, desc="Encoding eval sequences")
        design_matrix = self._prepare_design_matrix(features, fit=False)
        logits = design_matrix @ self.W_out
        return logits

    def train(
        self,
        input_data: jnp.ndarray,
        target_data: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
    ) -> None:
        """
        出力層を訓練します（Ridge回帰）。JAXのデフォルトデバイス配置を使用。

        Args:
            input_data: 形状 (time_steps, n_inputs) の入力データ
            target_data: 形状 (time_steps, n_outputs) の目標データ
            ridge_lambdas: 正則化パラメータ候補リスト
        """
        # データをfloat64に変換
        input_data = input_data.astype(jnp.float64)
        target_data = target_data.astype(jnp.float64)

        # ①固定されたランダム重みを使ってreservoir状態を計算
        reservoir_states = self.run_reservoir(input_data)

        # Feature standardization and zero-variance filtering
        design_matrix = self._prepare_design_matrix(reservoir_states, fit=True, washout=True)

        self._fit_ridge_with_grid(
            design_matrix,
            target_data,
            ridge_lambdas,
            classification_mode=False,
        )
        self.classification_mode = False
        self.num_classes = None
        self.trained = True  # Mark as trained
        
    def predict(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """
        入力データに対して予測を行います。

        Args:
            input_data: 形状 (time_steps, n_inputs) の入力データ

        Returns:
            予測結果 (time_steps, n_outputs)
        """
        # Use base class validation
        super().predict(input_data)

        if self.W_out is None:
            raise ValueError("モデルが訓練されていません。先にtrain()を呼び出してください。")

        # 入力データをfloat64に変換
        input_data = input_data.astype(jnp.float64)

        # reservoirを実行
        reservoir_states = self.run_reservoir(input_data)

        # Feature standardization and zero-variance filtering
        design_matrix = self._prepare_design_matrix(reservoir_states, fit=False, washout=True)

        # 予測を計算（単純な行列積）
        predictions = jnp.dot(design_matrix, self.W_out)
        return predictions

    def reset_state(self) -> None:
        """Reset the reservoir to initial state."""
        super().reset_state()  # Reset base class state
        self.W_out = None
        self.best_ridge_lambda = None
        self.ridge_search_log = []
        self.last_training_mse = None
        self.classification_mode = False
        self.num_classes = None
        self._feature_mu_ = None
        self._feature_sigma_ = None
        self._feature_keep_mask_ = None
        self._design_keep_mask_ = None
        # Reinitialize weights with new random seed
        self.key = random.PRNGKey(self.initial_random_seed)
        self._initialize_weights()

    def get_reservoir_info(self) -> Dict[str, Any]:
        """reservoirの情報を返します。"""
        return {
            **(self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config),
            "backend": self.backend,
            "trained": self.trained
        } 
