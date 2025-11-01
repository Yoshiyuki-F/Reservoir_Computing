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

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .base_reservoir import BaseReservoirComputer


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
        
    Examples:
        基本的な使用例:
        
        >>> from reservoir import ReservoirConfig, ReservoirComputer
        >>> import numpy as np
        
        >>> config = ReservoirConfig(
        ...     n_inputs=1, n_reservoir=100, n_outputs=1
        ... )
        >>> rc = ReservoirComputer(config)
        
        >>> # 訓練データ準備
        >>> time = np.linspace(0, 10, 1000)
        >>> inputs = np.sin(time).reshape(-1, 1)
        >>> targets = np.cos(time).reshape(-1, 1)
        
        >>> # 訓練と予測
        >>> rc.train(inputs, targets)
        >>> predictions = rc.predict(inputs)
    """
    
    def __init__(self, config: Sequence[Dict[str, Any]], backend: Optional[str] = None):
        """
        Reservoir Computerを初期化します。

        Args:
            config: ReservoirConfigオブジェクト
            backend: 計算バックエンド ('cpu', 'gpu', または None で自動選択)
        """
        super().__init__()  # Initialize base class

        merged: Dict[str, Any] = _load_shared_defaults().copy()
        required_keys = {
            "n_inputs",
            "n_reservoir",
            "n_outputs",
            "spectral_radius",
            "input_scaling",
            "noise_level",
            "alpha",
            "reservoir_weight_range",
            "sparsity",
            "input_bias",
            "nonlinearity",
            "random_seed",
            "state_aggregation",
        }
        config_sequence: Iterable[Dict[str, Any]] = [config] if isinstance(config, dict) else config  # type: ignore[arg-type]
        for cfg in config_sequence:
            cfg_dict = dict(cfg)
            merged.update({k: v for k, v in cfg_dict.items() if k not in {'name', 'description', 'params'}})
            params = cfg_dict.get('params', {}) or {}
            merged.update(params)

        missing_core = [key for key in required_keys if key not in merged]
        if missing_core:
            formatted = ", ".join(sorted(missing_core))
            raise ValueError(
                "Classical reservoir configuration missing required keys: "
                f"{formatted}"
            )

        self.config = merged

        self.n_inputs = merged['n_inputs']
        self.n_reservoir = merged['n_reservoir']
        self.n_outputs = merged['n_outputs']
        self.spectral_radius = merged['spectral_radius']
        self.input_scaling = merged['input_scaling']
        self.noise_level = merged['noise_level']
        self.alpha = merged['alpha']
        self.reservoir_weight_range = merged['reservoir_weight_range']
        self.sparsity = merged['sparsity']
        self.input_bias = merged['input_bias']
        self.nonlinearity = merged['nonlinearity']
        random_seed = merged['random_seed']

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
        self.state_aggregation = str(merged.get('state_aggregation', 'last')).lower()
        if self.state_aggregation not in {'last', 'mean', 'concat'}:
            raise ValueError("state_aggregation must be 'last', 'mean', or 'concat'")

        self.initial_random_seed = random_seed

        # Parse ridge_lambdas: supports (-14, 2, 25), [-14, 2, 25], or explicit list
        ridge_cfg = merged.get("ridge_lambdas", [-14, 2, 25])
        if isinstance(ridge_cfg, (list, tuple)) and len(ridge_cfg) == 3:
            # Format: [start, stop, num] or (start, stop, num)
            start, stop, num = ridge_cfg
            self.ridge_lambdas: Sequence[float] = tuple(np.logspace(start, stop, int(num)))
        elif isinstance(ridge_cfg, dict):
            # Format: {"start": -14, "stop": 2, "num": 25}
            start = ridge_cfg.get("start", -14)
            stop = ridge_cfg.get("stop", 2)
            num = ridge_cfg.get("num", 25)
            self.ridge_lambdas = tuple(np.logspace(start, stop, int(num)))
        else:
            # Explicit list of lambda values
            self.ridge_lambdas = tuple(float(l) for l in ridge_cfg)

        # Feature normalization state
        self._feature_mu_: Optional[np.ndarray] = None
        self._feature_sigma_: Optional[np.ndarray] = None
        self._feature_keep_mask_: Optional[np.ndarray] = None


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
    ) -> jnp.ndarray:
        """Standardize features and filter zero-variance columns."""
        features_np = np.asarray(feature_matrix, dtype=np.float64)

        if features_np.ndim != 2:
            features_np = features_np.reshape(features_np.shape[0], -1)

        if fit or self._feature_mu_ is None or self._feature_sigma_ is None:
            mu = features_np.mean(axis=0)
            sigma = features_np.std(axis=0)
            keep = sigma >= 1e-12
            if not np.any(keep):
                keep = np.ones_like(sigma, dtype=bool)
            sigma_adj = sigma.copy()
            sigma_adj[~keep] = 1.0
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

        bias = np.ones((centered.shape[0], 1), dtype=np.float64)
        design = np.concatenate([centered, bias], axis=1)
        return jnp.array(design, dtype=jnp.float64)

    def _fit_ridge_with_grid(
        self,
        X: jnp.ndarray,
        target_data: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
        use_kfold: bool = False,
        n_folds: int = 5,
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
                print(f"[classical-rc] design matrix cond -> {cond_number:.3e}")
            except np.linalg.LinAlgError:
                print("[classical-rc] SVD failed; skipping condition number log.")

        def ridge_via_svd(design: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
            U, s, Vt = np.linalg.svd(design, full_matrices=False)
            UT_y = U.T @ targets
            denom = s ** 2 + lam
            coeff = (s / denom)[:, None]
            scaled = coeff * UT_y
            return Vt.T @ scaled

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

                    weights = ridge_via_svd(X_fold_train, y_fold_train, lam)
                    y_pred = X_fold_val @ weights
                    fold_mse = float(np.mean((y_fold_val - y_pred) ** 2))
                    cv_scores[lam].append(fold_mse)

            # Average CV scores
            val_mse_list = [float(np.mean(cv_scores[lam])) for lam in lambda_candidates]
            train_mse_list = val_mse_list.copy()  # Placeholder

            best_index = int(np.argmin(val_mse_list))
            best_lambda = lambda_candidates[best_index]

            print(f"🔍 Ridge λ grid search ({n_folds}-Fold CV)")
            for lam, cv_mse in zip(lambda_candidates, val_mse_list):
                print(f"  λ={lam:.2e} -> CV MSE={cv_mse:.6e}")

            # Retrain on full data with best lambda
            print(f"[classical-rc] Retraining on full data with λ*={best_lambda:.2e}")
            best_weights_np = ridge_via_svd(X_np, y_np, best_lambda)

        else:
            # Simple 90/10 split
            split_idx = int(0.9 * n_samples)
            split_idx = max(1, min(split_idx, n_samples - 1))

            X_train, X_val = X_np[:split_idx], X_np[split_idx:]
            y_train, y_val = y_np[:split_idx], y_np[split_idx:]

            train_mse_list: list[float] = []
            val_mse_list: list[float] = []
            weights_by_lambda: Dict[float, np.ndarray] = {}

            lambda_iter = tqdm(
                lambda_candidates,
                desc="Ridge λ search (VAL)",
                leave=False,
                unit="λ",
            )
            for lam in lambda_iter:
                weights = ridge_via_svd(X_train, y_train, lam)
                yhat_tr = X_train @ weights
                train_mse = float(np.mean((y_train - yhat_tr) ** 2))

                if X_val.size > 0:
                    yhat_val = X_val @ weights
                    val_mse = float(np.mean((y_val - yhat_val) ** 2))
                else:
                    val_mse = train_mse

                train_mse_list.append(train_mse)
                val_mse_list.append(val_mse)
                weights_by_lambda[lam] = weights

            best_index = int(np.argmin(val_mse_list))
            best_lambda = lambda_candidates[best_index]
            best_weights_np = weights_by_lambda[best_lambda]

            print("🔍 Ridge λ grid search (VAL)")
            for lam, val_mse in zip(lambda_candidates, val_mse_list):
                print(f"  λ={lam:.2e} -> val MSE={val_mse:.6e}")

            # Retrain on full train+val with best lambda
            print(f"[classical-rc] Retraining on train+val with λ*={best_lambda:.2e}")
            best_weights_np = ridge_via_svd(X_np, y_np, best_lambda)

        self.W_out = jnp.array(best_weights_np, dtype=jnp.float64)
        self.best_ridge_lambda = best_lambda
        self.last_training_mse = float(val_mse_list[best_index])
        self.ridge_search_log = [
            {
                "lambda": lam,
                "train_mse": train_mse_list[i] if i < len(train_mse_list) else val_mse_list[i],
                "val_mse": val,
            }
            for i, (lam, val) in enumerate(zip(lambda_candidates, val_mse_list))
        ]

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

        self._fit_ridge_with_grid(design_matrix, targets, ridge_lambdas)
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
        design_matrix = self._prepare_design_matrix(reservoir_states, fit=True)

        self._fit_ridge_with_grid(design_matrix, target_data, ridge_lambdas)
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
        design_matrix = self._prepare_design_matrix(reservoir_states, fit=False)

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
