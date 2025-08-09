"""
Reservoir Computer implementation using JAX.
"""

import jax
import jax.numpy as jnp
from jax import random, lax, device_put
import numpy as np

# JAXの設定
jax.config.update("jax_enable_x64", True)


from typing import Optional
from .config import ReservoirConfig

class ReservoirComputer:
    """
    JAXを使ったReservoir Computingの実装クラス。
    
    Reservoir Computingは、固定されたランダムなリカレント層（reservoir）と
    訓練可能な出力層から構成されるニューラルネットワークです。
    """
    
    def __init__(self, config: ReservoirConfig):
        """
        Reservoir Computerを初期化します。
        
        Args:
            config: ReservoirConfigオブジェクト
        """
        self.config = config
        self.n_inputs = config.n_inputs
        self.n_reservoir = config.n_reservoir
        self.n_outputs = config.n_outputs
        self.spectral_radius = config.spectral_radius
        self.input_scaling = config.input_scaling
        self.noise_level = config.noise_level
        self.alpha = config.alpha
        
        # 乱数キーの初期化
        self.key = random.PRNGKey(config.random_seed)
        
        # reservoirの重みを初期化
        self._initialize_weights()
        
        # 出力重みは後で訓練で設定
        self.W_out = None
        
    def _initialize_weights(self):
        """reservoir内部の重みを初期化します。"""
        from .utils import print_gpu_info
        
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        
        if gpu_devices:
            print_gpu_info()
            self._initialize_on_gpu()
        else:
            # CPU環境で実行
            self._initialize_on_cpu()
    
    def _initialize_on_cpu(self):
        """CPU上で重み初期化（安定性重視）"""
        # 入力重みをランダムに初期化
        key1, key2, self.key = random.split(self.key, 3)
        
        W_in_cpu = random.uniform(
            key1, 
            (self.n_reservoir, self.n_inputs), 
            minval=-self.input_scaling, 
            maxval=self.input_scaling,
            dtype=jnp.float64
        )
        
        # reservoir内部の重みを初期化
        W_res_cpu = random.uniform(
            key2, 
            (self.n_reservoir, self.n_reservoir), 
            minval=-1, 
            maxval=1,
            dtype=jnp.float64
        )
        
        # NumPyで固有値計算（より安定）
        W_res_np = np.array(W_res_cpu)
        try:
            eigenvalues = np.linalg.eigvals(W_res_np)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            max_eigenvalue = max(max_eigenvalue, 1e-8)
            W_res_scaled = (self.spectral_radius / max_eigenvalue) * W_res_np
        except:
            # 固有値計算が失敗した場合
            frobenius_norm = np.linalg.norm(W_res_np, 'fro')
            frobenius_norm = max(frobenius_norm, 1e-8)
            W_res_scaled = (self.spectral_radius / frobenius_norm) * W_res_np
        
        # GPUに転送
        self.W_in = device_put(jnp.array(W_in_cpu), jax.devices()[0])
        self.W_res = device_put(jnp.array(W_res_scaled), jax.devices()[0])
    
    def _initialize_on_gpu(self):
        """GPU上で重み初期化（GPU専用環境用）"""
        # GPU上で直接初期化
        key1, key2, self.key = random.split(self.key, 3)
        
        self.W_in = random.uniform(
            key1, 
            (self.n_reservoir, self.n_inputs), 
            minval=-self.input_scaling, 
            maxval=self.input_scaling,
            dtype=jnp.float64
        )
        
        # reservoir内部の重みを初期化
        W_res = random.uniform(
            key2, 
            (self.n_reservoir, self.n_reservoir), 
            minval=-1, 
            maxval=1,
            dtype=jnp.float64
        )
        
        # GPU上で固有値計算（安定性に注意）
        try:
            eigenvalues = jnp.linalg.eigvals(W_res)
            max_eigenvalue = jnp.max(jnp.abs(eigenvalues))
            max_eigenvalue = jnp.maximum(max_eigenvalue, 1e-8)
            self.W_res = (self.spectral_radius / max_eigenvalue) * W_res
        except:
            # GPU固有値計算が失敗した場合、Frobenius normで代替
            frobenius_norm = jnp.linalg.norm(W_res, 'fro')
            frobenius_norm = jnp.maximum(frobenius_norm, 1e-8)
            self.W_res = (self.spectral_radius / frobenius_norm) * W_res
        




        
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
        
        # JAXのscanを使用して効率的に計算 /reservoir状態 h(1) から h(n) まで連続的に計算
        carry, states = lax.scan(self._reservoir_step, initial_carry, input_sequence)
        
        # キーを更新
        _, self.key = carry
        
        return states
        
    def train(self, input_data: jnp.ndarray, target_data: jnp.ndarray, reg_param: float = 1e-8):
        """
        出力層を訓練します（Ridge回帰）。
        
        Args:
            input_data: 形状 (time_steps, n_inputs) の入力データ
            target_data: 形状 (time_steps, n_outputs) の目標データ
            reg_param: 正則化パラメータ
        """
        # データをfloat64に変換
        input_data = input_data.astype(jnp.float64)
        target_data = target_data.astype(jnp.float64)
        
        # ①【ここで実行】固定されたランダム重みを使ってreservoir状態を計算
        reservoir_states = self.run_reservoir(input_data)  # h(1), h(2), ..., h(n)　(time_steps-1, n_reservoir)
        
        # バイアス項を追加
        bias_column = jnp.ones((reservoir_states.shape[0], 1), dtype=jnp.float64)
        X = jnp.concatenate([reservoir_states, bias_column], axis=1) # (time_steps-1, n_reservoir+1)
        
        # ②訓練：reservoir状態から目標データへの線形写像（W_out）を学習
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        
        if gpu_devices:
            self._train_on_gpu(X, target_data, reg_param)
        else:
            self._train_on_cpu(X, target_data, reg_param)
    
    def _train_on_cpu(self, X, target_data, reg_param):
        """CPU上でRidge回帰を実行（安定性重視）"""
        with jax.default_device(jax.devices('cpu')[0]):
            X_cpu = jax.device_get(X)
            target_cpu = jax.device_get(target_data)
            
            # NumPyでRidge回帰を実行
            XTX = X_cpu.T @ X_cpu # X^T X (グラム行列)
            XTY = X_cpu.T @ target_cpu # X^T Y  
            
            # 正則化項を追加　 # X^T X + λI
            A = XTX + reg_param * np.eye(XTX.shape[0], dtype=np.float64)
            
            try:
                W_out_cpu = np.linalg.solve(A, XTY)
            except:
                W_out_cpu = np.linalg.pinv(A) @ XTY
            
            # GPUに転送
            self.W_out = device_put(jnp.array(W_out_cpu), jax.devices()[0])
    
    def _train_on_gpu(self, X, target_data, reg_param):
        """GPU上でRidge回帰を実行（GPU専用環境用）"""
        # GPU上で直接Ridge回帰
        XTX = X.T @ X  # X^T X (グラム行列)
        XTY = X.T @ target_data  # X^T Y
        
        # 正則化項を追加
        A = XTX + reg_param * jnp.eye(XTX.shape[0], dtype=jnp.float64)
        
        try:
            self.W_out = jnp.linalg.solve(A, XTY)
        except:
            # GPU上でも失敗する場合は擬似逆行列を使用
            self.W_out = jnp.linalg.pinv(A) @ XTY
        
    def predict(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """
        入力データに対して予測を行います。
        
        Args:
            input_data: 形状 (time_steps, n_inputs) の入力データ
            
        Returns:
            予測結果 (time_steps, n_outputs)
        """
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
        
    def get_reservoir_info(self) -> dict:
        """reservoirの情報を返します。"""
        return {
            **self.config.to_dict(),
            "trained": self.W_out is not None
        } 