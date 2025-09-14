"""
Reservoir Computing用の設定クラス（Pydantic版）
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

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

    model_config = ConfigDict(extra="forbid")


class DataGenerationConfig(BaseModel):
    """時系列データ生成のための設定。
    
    各データタイプに必要な追加パラメータは `params` フィールドに格納します。
    """
    name: str = Field(..., description="データ生成関数の名前 ('sine_wave', 'lorenz', 'mackey_glass')")
    time_steps: int = Field(..., gt=0, description="時系列の長さ")
    dt: float = Field(..., gt=0, description="時間ステップ")
    noise_level: float = Field(..., ge=0, description="ノイズレベル")
    use_dimensions: Optional[List[int]] = Field(None, description="使用する次元のインデックス")
    
    # 動的パラメータ - データタイプごとに異なる
    params: Dict[str, Any] = Field(default_factory=dict, description="データタイプ固有のパラメータ")
    
    @field_validator('name')
    def validate_name(cls, v):
        valid_names = {'sine_wave', 'lorenz', 'mackey_glass'}
        if v not in valid_names:
            raise ValueError(f"name は {valid_names} のいずれかである必要があります")
        return v
    
    @field_validator('params', mode='before')
    def validate_params(cls, v, info):
        name = info.data.get('name')
        if not name:
            return v
            
        # データタイプごとの必須パラメータをチェック
        if name == 'sine_wave':
            if 'frequencies' not in v:
                raise ValueError("sine_wave には frequencies パラメータが必要です")
            freqs = v['frequencies']
            if not isinstance(freqs, list) or not freqs:
                raise ValueError("frequencies は空でないリストである必要があります")
            if any(f <= 0 for f in freqs):
                raise ValueError("周波数は正の値である必要があります")
                
        elif name == 'lorenz':
            required = {'sigma', 'rho', 'beta'}
            missing = required - set(v.keys())
            if missing:
                raise ValueError(f"lorenz には {missing} パラメータが必要です")
            for param in required:
                if v[param] <= 0:
                    raise ValueError(f"{param} は正の値である必要があります")
                    
        elif name == 'mackey_glass':
            defaults = {'tau': 17.0, 'n': 10.0, 'beta': 0.2, 'gamma': 0.1}
            for param, default in defaults.items():
                if param not in v:
                    v[param] = default
                elif v[param] <= 0:
                    raise ValueError(f"{param} は正の値である必要があります")
        
        return v
    
    def get_param(self, key: str, default=None):
        """パラメータを安全に取得"""
        return self.params.get(key, default)
    
    model_config = ConfigDict(extra="forbid")



class TrainingConfig(BaseModel):
    """Reservoir Computerの訓練に関する設定。
    
    Ridge回帰による出力層の学習に使用されるパラメータを定義します。
    """
    train_size: int = Field(..., gt=0, description="訓練データサイズ")
    reg_param: float = Field(..., gt=0, description="正則化パラメータ")

    model_config = ConfigDict(extra="forbid")

class DemoConfig(BaseModel):
    """デモ表示設定"""
    title: str = Field(..., description="デモのタイトル")
    filename: str = Field(..., description="出力ファイル名")
    show_training: bool = Field(..., description="訓練結果も表示するか")

    @field_validator('filename')
    def validate_filename(cls, v):
        if not v.strip():
            raise ValueError("ファイル名は空にできません")
        return v.strip()

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")
