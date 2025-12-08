"""Factory for creating readout instances from configuration."""
from __future__ import annotations

from typing import Optional, Any

from reservoir.models.config import RidgeReadoutConfig, ReadoutConfig
from reservoir.readout.ridge import RidgeRegression

class ReadoutFactory:
    """Builds readout modules from ReadoutConfig."""

    @staticmethod
    def create_readout(config: Optional[ReadoutConfig]) -> Any:
        # None (End-to-End) の場合
        if config is None:
            return None

        # Ridgeの場合
        if isinstance(config, RidgeReadoutConfig):
            # ここで Config(データ) を Instance(オブジェクト) に変換する
            return RidgeRegression(ridge_lambda=config.init_lambda, use_intercept=config.use_intercept)

        # 将来の拡張 (FNN, SVD etc.)
        # elif isinstance(config, FNNReadoutConfig):
        #     return ...

        raise TypeError(f"ReadoutFactory received unknown config type: {type(config)}")

__all__ = ["ReadoutFactory"]