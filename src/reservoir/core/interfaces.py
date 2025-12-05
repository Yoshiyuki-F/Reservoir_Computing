"""
src/reservoir/core/interfaces.py
Core protocols defining the contract for system components.
"""
from typing import Protocol, runtime_checkable, Any, Dict
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
