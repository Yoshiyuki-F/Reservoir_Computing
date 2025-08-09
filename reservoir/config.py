from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import json
from pathlib import Path


@dataclass
class ReservoirConfig:
    """リザーバーコンピュータの設定パラメータ"""
    n_inputs: int
    n_reservoir: int
    n_outputs: int
    spectral_radius: float = 0.95
    input_scaling: float = 1.0
    noise_level: float = 0.001
    alpha: float = 1.0
    random_seed: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ReservoirConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """設定を辞書として返します。"""
        return {
            'n_inputs': self.n_inputs,
            'n_reservoir': self.n_reservoir,
            'n_outputs': self.n_outputs,
            'spectral_radius': self.spectral_radius,
            'input_scaling': self.input_scaling,
            'noise_level': self.noise_level,
            'alpha': self.alpha,
            'random_seed': self.random_seed
        }


@dataclass 
class BaseConfig:
    """汎用的な設定管理クラス"""
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __getattr__(self, name: str) -> Any:
        """ドット記法でのパラメータアクセスを可能にする"""
        if name in self.params:
            value = self.params[name]
            # ネストした辞書もBaseConfigに変換
            if isinstance(value, dict):
                return BaseConfig(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """辞書形式でのアクセスも可能"""
        return self.params[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """デフォルト値付きでの取得"""
        return self.params.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return self.params.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """辞書から作成"""
        return cls(params=config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'BaseConfig':
        """JSONファイルから設定を読み込み"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """JSONファイルに設定を保存"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False)


@dataclass
class DemoConfig:
    """デモの完全設定管理"""
    config: BaseConfig
    
    def __init__(self, json_path: Optional[Union[str, Path]] = None, config_dict: Optional[Dict[str, Any]] = None):
        if json_path:
            self.config = BaseConfig.from_json(json_path)
        elif config_dict:
            self.config = BaseConfig.from_dict(config_dict)
        else:
            self.config = BaseConfig()
    
    @property
    def data_generation(self) -> BaseConfig:
        return BaseConfig(self.config.params.get('data_generation', {}))
    
    @property
    def reservoir(self) -> ReservoirConfig:
        reservoir_params = self.config.params.get('reservoir', {})
        return ReservoirConfig.from_dict(reservoir_params)
    
    @property
    def training(self) -> BaseConfig:
        return BaseConfig(self.config.params.get('training', {}))
    
    @property
    def demo(self) -> BaseConfig:
        return BaseConfig(self.config.params.get('demo', {}))
    
    def get_data_params(self) -> Dict[str, Any]:
        """データ生成パラメータを辞書として取得"""
        return self.data_generation.to_dict()
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'DemoConfig':
        """JSONファイルから作成"""
        return cls(json_path=json_path)


# 使用例を示すヘルパー関数
def create_demo_config_template(task_type: str, output_path: Union[str, Path]) -> None:
    """デモ設定テンプレートを作成"""
    templates = {
        'sine_wave': {
            "data_generation": {
                "time_steps": 2000,
                "dt": 0.01,
                "frequencies": [1.0, 2.0, 0.5],
                "noise_level": 0.05
            },
            "reservoir": {
                "n_inputs": 1,
                "n_reservoir": 100,
                "n_outputs": 1,
                "spectral_radius": 0.95,
                "input_scaling": 1.0,
                "noise_level": 0.001,
                "alpha": 1.0,
                "random_seed": 42
            },
            "training": {
                "train_size": 1500,
                "reg_param": 1e-8
            },
            "demo": {
                "title": "サイン波予測のデモンストレーション",
                "filename": "sine_wave_prediction.png",
                "show_training": True
            }
        },
        'lorenz': {
            "data_generation": {
                "time_steps": 3000,
                "dt": 0.01,
                "sigma": 10.0,
                "rho": 28.0,
                "beta": 2.666667
            },
            "reservoir": {
                "n_inputs": 1,
                "n_reservoir": 200,
                "n_outputs": 1,
                "spectral_radius": 0.9,
                "input_scaling": 0.5,
                "noise_level": 0.001,
                "alpha": 0.8,
                "random_seed": 42
            },
            "training": {
                "train_size": 2000,
                "reg_param": 1e-6
            },
            "demo": {
                "title": "Lorenz Attractor (X-coordinate) Prediction Results",
                "filename": "lorenz_prediction.png",
                "show_training": False
            }
        }
    }
    
    if task_type in templates:
        config = BaseConfig(templates[task_type])
        config.to_json(output_path)
        print(f"テンプレート設定ファイル '{output_path}' を作成しました。")
    else:
        raise ValueError(f"未対応のタスクタイプ: {task_type}。対応タイプ: {list(templates.keys())}")