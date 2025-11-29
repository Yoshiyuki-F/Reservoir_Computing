"""
src/reservoir/core/interfaces.py
Core protocols defining the contract for system components.
"""
from typing import Protocol, runtime_checkable, Any, Dict, Tuple
import jax.numpy as jnp


@runtime_checkable
class Transformer(Protocol):
    """
    データ変換を行うコンポーネントの共通インターフェース。
    Scaler, DesignMatrixBuilder, PCAなどがこれに該当します。
    """

    def fit(self, features: jnp.ndarray) -> "Transformer":
        """
        データから統計量（平均、分散など）を学習します。

        Args:
            features: 形状 (samples, features) の入力データ
        Returns:
            self (メソッドチェーン用)
        """
        ...

    def transform(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        学習した統計量に基づいてデータを変換します。

        Args:
            features: 入力データ
        Returns:
            変換後のデータ
        """
        ...

    def fit_transform(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        学習と変換を一度に行います。
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        設定や学習済みパラメータを辞書形式でエクスポートします（保存用）。
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transformer":
        """
        辞書データからインスタンスを復元します。
        """
        ...


@runtime_checkable
class ReservoirNode(Protocol):
    """
    リザバー（力学系）モデルの共通インターフェース。
    ClassicalReservoir, QuantumReservoir, ESNなどがこれに該当します。
    """

    @property
    def output_dim(self) -> int:
        """リザバーが出力する特徴量（状態）の次元数を返します。"""
        ...

    def initialize_state(self, batch_size: int) -> jnp.ndarray:
        """
        初期状態ベクトルを生成します。

        Args:
            batch_size: バッチサイズ
        Returns:
            形状 (batch_size, state_dim) のゼロ状態または初期状態
        """
        ...

    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        1ステップ（または1バッチ）の時間の更新を行います。

        Args:
            state: 現在の隠れ状態 (batch_size, state_dim)
            input_data: 現在の入力 (batch_size, input_dim)

        Returns:
            (next_state, output_features)
            - next_state: 次の隠れ状態（次のステップに渡す用）
            - output_features: リードアウト層に渡す特徴量（preprocess等が適用される前の生の状態）
        """
        ...

    def generate_trajectory(self, initial_state: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        時系列入力全体に対する状態遷移（軌跡）を生成します。
        JAXの scan 等を使用して高速化されることを想定しています。

        Args:
            initial_state: 初期状態
            inputs: 時系列入力 (time_steps, features) または (batch, time, features)

        Returns:
            全時刻の状態系列
        """
        ...


@runtime_checkable
class ReadoutModule(Protocol):
    """
    リザバーの状態から最終出力を予測する読み出し層のインターフェース。
    RidgeRegression, LinearRegressionなどがこれに該当します。
    """

    def fit(self, states: jnp.ndarray, targets: jnp.ndarray) -> "ReadoutModule":
        """
        リザバー状態と教師データから重みを学習します。
        """
        ...

    def predict(self, states: jnp.ndarray) -> jnp.ndarray:
        """
        リザバー状態から値を予測します。
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """学習済み重みなどを保存用辞書にします。"""
        ...

