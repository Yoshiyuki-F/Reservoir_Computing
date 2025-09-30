"""
Classical Reservoir Computer implementation using JAX.
"""

import jax
import numpy as np

from typing import Optional, Dict, Any

from pipelines.jax_config import ensure_x64_enabled

ensure_x64_enabled()

import jax.numpy as jnp
from jax import random, lax, device_put

from .config import ReservoirConfig
from .base_reservoir import BaseReservoirComputer

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
        >>> rc.train(inputs, targets, reg_param=1e-6)
        >>> predictions = rc.predict(inputs)
    """
    
    def __init__(self, config: ReservoirConfig, backend: Optional[str] = None):
        """
        Reservoir Computerを初期化します。

        Args:
            config: ReservoirConfigオブジェクト
            backend: 計算バックエンド ('cpu', 'gpu', または None で自動選択)
        """
        super().__init__()  # Initialize base class
        self.config = config

        # Handle both dict and config object formats
        if isinstance(config, dict):
            self.n_inputs = config.get('n_inputs', 1)
            self.n_reservoir = config.get('n_reservoir', 100)
            self.n_outputs = config.get('n_outputs', 1)
            self.spectral_radius = config.get('spectral_radius', 0.95)
            self.input_scaling = config.get('input_scaling', 1.0)
        else:
            self.n_inputs = config.n_inputs
            self.n_reservoir = config.n_reservoir
            self.n_outputs = config.n_outputs
            self.spectral_radius = config.spectral_radius
            self.input_scaling = config.input_scaling
        # Handle both dict and config object formats for remaining parameters
        if isinstance(config, dict):
            self.noise_level = config.get('noise_level', 0.001)
            self.alpha = config.get('alpha', 1.0)
            self.reservoir_weight_range = config.get('reservoir_weight_range', 1.0)
            self.sparsity = config.get('sparsity', 1.0)
            self.input_bias = config.get('input_bias', 0.0)
            self.nonlinearity = config.get('nonlinearity', 'tanh')
            random_seed = config.get('random_seed', 42)
        else:
            self.noise_level = config.noise_level
            self.alpha = config.alpha
            self.reservoir_weight_range = config.reservoir_weight_range
            self.sparsity = getattr(config, 'sparsity', 1.0)
            self.input_bias = getattr(config, 'input_bias', 0.0)
            self.nonlinearity = getattr(config, 'nonlinearity', 'tanh')
            random_seed = config.random_seed

        # 乱数キーの初期化
        self.key = random.PRNGKey(random_seed)
        
        # バックエンドの設定（CLI層で確認済み）
        self.backend = backend
        
        # reservoirの重みを初期化
        self._initialize_weights()
        
        # 出力重みは後で訓練で設定
        self.W_out = None


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
    def _train_unified(X: jnp.ndarray, target_data: jnp.ndarray, reg_param: float) -> jnp.ndarray:
        """統合されたRidge回帰訓練（JAXのデフォルトデバイス配置を使用）"""
        XTX = X.T @ X
        XTY = X.T @ target_data
        
        A = XTX + reg_param * jnp.eye(XTX.shape[0], dtype=jnp.float64)
        
        try:
            return jnp.linalg.solve(A, XTY)
        except:
            return jnp.linalg.pinv(A) @ XTY

    def train(self, input_data: jnp.ndarray, target_data: jnp.ndarray, reg_param: float):
        """
        出力層を訓練します（Ridge回帰）。JAXのデフォルトデバイス配置を使用。
        
        Args:
            input_data: 形状 (time_steps, n_inputs) の入力データ
            target_data: 形状 (time_steps, n_outputs) の目標データ
            reg_param: 正則化パラメータ
        """
        # データをfloat64に変換
        input_data = input_data.astype(jnp.float64)
        target_data = target_data.astype(jnp.float64)
        
        # ①固定されたランダム重みを使ってreservoir状態を計算
        reservoir_states = self.run_reservoir(input_data)
        
        # バイアス項を追加
        bias_column = jnp.ones((reservoir_states.shape[0], 1), dtype=jnp.float64)
        X = jnp.concatenate([reservoir_states, bias_column], axis=1)
        
        # ②訓練：reservoir状態から目標データへの線形写像（W_out）を学習
        # JAXのデフォルトデバイス配置を使用
        self.W_out = self._train_unified(X, target_data, reg_param)
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

        # バイアス項を追加
        bias_column = jnp.ones((reservoir_states.shape[0], 1), dtype=jnp.float64)
        X = jnp.concatenate([reservoir_states, bias_column], axis=1)

        # 予測を計算（単純な行列積）
        predictions = jnp.dot(X, self.W_out)
        return predictions

    def reset_state(self) -> None:
        """Reset the reservoir to initial state."""
        super().reset_state()  # Reset base class state
        self.W_out = None
        # Reinitialize weights with new random seed
        self.key = random.PRNGKey(self.config.random_seed)
        self._initialize_weights()

    def get_reservoir_info(self) -> Dict[str, Any]:
        """reservoirの情報を返します。"""
        return {
            **(self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config),
            "backend": self.backend,
            "trained": self.trained
        } 
