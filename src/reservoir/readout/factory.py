"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/readout/factory.py
Factory for creating readout instances from configuration."""
from __future__ import annotations


from reservoir.models.config import RidgeReadoutConfig, PolyRidgeReadoutConfig, FNNReadoutConfig, ReadoutConfig
from reservoir.readout.ridge import RidgeCV
from reservoir.readout.poly_ridge import PolyRidgeReadout
from reservoir.readout.fnn_readout import FNNReadout
from reservoir.core.interfaces import ReadoutModule
from reservoir.training.config import TrainingConfig

class ReadoutFactory:
    """Builds readout modules from ReadoutConfig."""

    @staticmethod
    def create_readout(
        config: ReadoutConfig | None,
        classification: bool,
        training_config: TrainingConfig | None = None,
    ) -> ReadoutModule | None:
        # None (End-to-End) の場合
        if config is None:
            return None

        # PolyRidgeの場合 (must check before RidgeReadoutConfig)
        if isinstance(config, PolyRidgeReadoutConfig):
            candidates = config.lambda_candidates
            return PolyRidgeReadout(
                use_intercept=config.use_intercept,
                lambda_candidates=candidates,
                degree=config.degree,
                mode=config.mode,
            )

        # Ridgeの場合
        if isinstance(config, RidgeReadoutConfig):
            candidates = config.lambda_candidates
            if candidates is None:
                candidates = (config.init_lambda,)
            
            return RidgeCV(
                use_intercept=config.use_intercept,
                lambda_candidates=candidates
            )

        # FNNの場合
        elif isinstance(config, FNNReadoutConfig):
            return FNNReadout(
                hidden_layers=config.hidden_layers,
                training_config=training_config,
                classification=classification
            )

        raise TypeError(f"ReadoutFactory received unknown config type: {type(config)}")

__all__ = ["ReadoutFactory"]