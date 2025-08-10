"""
Reservoir Computing用の設定クラス（Pydantic版）
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator

class ReservoirConfig(BaseModel):
    """Reservoir Computerのハイパーパラメータ設定。
    
    Echo State Network (ESN) におけるリザーバー層の構成と動作を制御する
    パラメータを定義します。全てのパラメータは妥当性が検証されます。
    """
    n_inputs: int = Field(..., gt=0, description="入力次元数")
    n_reservoir: int = Field(..., gt=0, description="リザーバーノード数")
    n_outputs: int = Field(..., gt=0, description="出力次元数")
    spectral_radius: float = Field(..., gt=0, le=2.0, description="スペクトラル半径")
    input_scaling: float = Field(..., gt=0, description="入力スケーリング")
    noise_level: float = Field(..., ge=0, description="ノイズレベル")
    alpha: float = Field(..., gt=0, le=1.0, description="リーク率")
    random_seed: int = Field(..., ge=0, description="乱数シード")
    reservoir_weight_range: float = Field(..., gt=0, description="リザーバー重み初期化範囲 [-range, range]")

    class Config:
        extra = "forbid"

class DataGenerationConfig(BaseModel):
    """時系列データ生成のための設定。
    
    サイン波やLorenz方程式などの時系列データ生成に使用される
    パラメータを定義します。デモやテストに使用されます。
    """
    name: str = Field(..., description="データ生成関数の名前 ('sine_wave', 'lorenz', 'mackey_glass')")
    time_steps: int = Field(..., gt=0, description="時系列の長さ")
    dt: float = Field(..., gt=0, description="時間ステップ")
    frequencies: List[float] = Field(..., description="サイン波の周波数リスト")
    noise_level: float = Field(..., ge=0, description="ノイズレベル")
    
    # Lorenz parameters
    sigma: float = Field(..., gt=0, description="Lorenzパラメータσ")
    rho: float = Field(..., gt=0, description="Lorenzパラメータρ")
    beta: float = Field(..., gt=0, description="Lorenzパラメータβ")
    
    # Optional dimension filtering
    use_dimensions: Optional[List[int]] = Field(None, description="使用する次元のインデックス")

    @validator('name')
    def validate_name(cls, v):
        valid_names = {'sine_wave', 'lorenz', 'mackey_glass'}
        if v not in valid_names:
            raise ValueError(f"name は {valid_names} のいずれかである必要があります")
        return v

    @validator('frequencies')
    def validate_frequencies(cls, v):
        if not v:
            raise ValueError("少なくとも1つの周波数が必要です")
        if any(f <= 0 for f in v):
            raise ValueError("周波数は正の値である必要があります")
        return v

    class Config:
        extra = "forbid"

class TrainingConfig(BaseModel):
    """Reservoir Computerの訓練に関する設定。
    
    Ridge回帰による出力層の学習に使用されるパラメータを定義します。
    """
    train_size: int = Field(..., gt=0, description="訓練データサイズ")
    reg_param: float = Field(..., gt=0, description="正則化パラメータ")

    class Config:
        extra = "forbid"

class DemoConfig(BaseModel):
    """デモ表示設定"""
    title: str = Field(..., description="デモのタイトル")
    filename: str = Field(..., description="出力ファイル名")
    show_training: bool = Field(..., description="訓練結果も表示するか")

    @validator('filename')
    def validate_filename(cls, v):
        if not v.strip():
            raise ValueError("ファイル名は空にできません")
        return v.strip()

    class Config:
        extra = "forbid"

class ExperimentConfig(BaseModel):
    """実験の完全設定管理"""
    data_generation: DataGenerationConfig
    reservoir: ReservoirConfig
    training: TrainingConfig
    demo: DemoConfig

    def get_data_params(self) -> Dict[str, Any]:
        """データ生成パラメータを辞書として取得"""
        return self.data_generation.model_dump()

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'ExperimentConfig':
        """JSONファイルから設定を読み込み"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, json_path: Union[str, Path]) -> None:
        """JSONファイルに設定を保存"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    class Config:
        extra = "forbid"