"""
Classical Reservoir Computer implementation using JAX.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Iterable, Callable, cast

from pipelines.jax_config import ensure_x64_enabled

ensure_x64_enabled()

import jax.numpy as jnp
from jax import random, lax, vmap
from tqdm.auto import tqdm

from reservoirs.preprocess import FeatureScaler, DesignMatrixBuilder
from reservoirs.preprocess.aggregator import AggregationMode, aggregate_states
from reservoirs.readout import BaseReadout, RidgeReadoutNumpy
from reservoirs.utils.spectral import spectral_radius_scale

from .base_reservoir import BaseReservoirComputer
from .config import ReservoirConfig, parse_ridge_lambdas


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
    """
    
    def __init__(
        self,
        config: Sequence[Dict[str, Any]],
        backend: Optional[str] = None,
        readout: Optional[BaseReadout] = None,
    ):
        """
        Reservoir Computerを初期化します。

        Args:
            config: ReservoirConfigオブジェクト
            backend: 計算バックエンド ('cpu', 'gpu', または None で自動選択)
        """
        super().__init__()  # Initialize base class

        # Merge shared defaults with user config
        merged: Dict[str, Any] = _load_shared_defaults().copy()
        config_sequence: Iterable[Dict[str, Any]] = [config] if isinstance(config, dict) else config  # type: ignore[arg-type]
        for cfg in config_sequence:
            cfg_dict = dict(cfg)
            merged.update({k: v for k, v in cfg_dict.items() if k not in {'name', 'description', 'params'}})
            params = cfg_dict.get('params', {}) or {}
            merged.update(params)

        # Create config object - this performs all validation
        cfg = ReservoirConfig(**merged)

        self.config = cfg

        # Extract validated parameters
        params = cfg.params

        self.n_inputs: int = params['n_inputs']
        self.n_reservoir: int = params['n_reservoir']
        self.n_outputs: int = params['n_outputs']
        self.spectral_radius: float = float(params['spectral_radius'])
        self.input_scaling: float = float(params['input_scaling'])
        self.noise_level: float = float(params['noise_level'])
        self.alpha: float = float(params['alpha'])
        self.reservoir_weight_range: float = float(params['reservoir_weight_range'])
        self.sparsity: float = float(params['sparsity'])
        self.input_bias: float = float(params['input_bias'])
        self.nonlinearity: str = str(params['nonlinearity'])
        self.encode_batch_size: int = max(1, int(params.get('encode_batch_size', 1024)))
        random_seed: int = int(params['random_seed'])
        state_agg = str(params.get('state_aggregation', 'last')).lower()
        self.state_aggregation: AggregationMode = cast(AggregationMode, state_agg)

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

        self.initial_random_seed = random_seed
        self.washout_steps: int = 3

        # Parse ridge_lambdas using common validation function
        self.ridge_lambdas: Sequence[float] = parse_ridge_lambdas(params)

        # Preprocess & readout components
        design_cfg = {
            "poly_mode": params.get("poly_mode", "square"),
            "degree": int(params.get("poly_degree", 2)),
            "include_bias": True,
            "std_threshold": float(params.get("std_threshold", 1e-3)),
        }
        self._design_cfg = design_cfg
        self.scaler = FeatureScaler()
        self.design_builder = DesignMatrixBuilder(**design_cfg)
        self._readout_cv = params.get("readout_cv", "holdout")
        self._readout_n_folds = int(params.get("readout_n_folds", 5))
        self._external_readout = readout
        self.readout: BaseReadout = readout or RidgeReadoutNumpy(
            default_cv=self._readout_cv,
            default_n_folds=self._readout_n_folds,
        )
        self.last_training_score: Optional[float] = None
        self.last_training_score_name: Optional[str] = None


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
            dtype=jnp.float64,
        )

        W_res_scaled = spectral_radius_scale(W_res, self.spectral_radius)

        self.W_in = W_in
        self.W_res = W_res_scaled
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
        
    def _run_sequence_with_key(
        self,
        input_sequence: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run reservoir for a single sequence with the provided key.
        """
        input_sequence = input_sequence.astype(jnp.float64)
        initial_state = jnp.zeros(self.n_reservoir, dtype=jnp.float64)
        initial_carry = (initial_state, key)
        carry, states = lax.scan(self._reservoir_step, initial_carry, input_sequence)
        return states
        
    def run_reservoir(self, input_sequence: jnp.ndarray) -> jnp.ndarray:
        """
        入力シーケンスに対してreservoirを実行します。
        """
        seq_key, new_master = random.split(self.key)
        states = self._run_sequence_with_key(input_sequence, seq_key)
        self.key = new_master
        return states

    def _encode_sequences(
        self,
        sequences: jnp.ndarray,
        desc: Optional[str] = None,
        *,
        leave: bool = False,
        batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """Run reservoir on sequences (batched) and collect aggregated features."""
        sequences = jnp.asarray(sequences, dtype=jnp.float64)
        total = sequences.shape[0]
        if total == 0:
            raise ValueError("Cannot encode empty sequence batch.")

        batch = int(batch_size or self.encode_batch_size)
        batch = max(1, batch)

        RunVmapped = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        run_vmapped: RunVmapped = cast(
            RunVmapped,
            vmap(self._run_sequence_with_key, in_axes=(0, 0)),
        )
        AggFn = Callable[[jnp.ndarray], jnp.ndarray]
        agg_fn: AggFn = lambda st: aggregate_states(st, self.state_aggregation)
        aggregate_batch: Callable[[jnp.ndarray], jnp.ndarray] = cast(
            Callable[[jnp.ndarray], jnp.ndarray],
            vmap(agg_fn, in_axes=0),
        )

        features_list = []
        master_key = self.key
        iterator = tqdm(
            range(0, total, batch),
            desc=desc or "Encoding sequences (batched)",
            leave=leave,
            unit="batch",
        )

        for start in iterator:
            end = min(start + batch, total)
            batch_sequences = sequences[start:end]
            split_keys = random.split(master_key, batch_sequences.shape[0] + 1)
            seq_keys = split_keys[:-1]
            master_key = split_keys[-1]

            batch_states = run_vmapped(batch_sequences, seq_keys)
            batch_features = aggregate_batch(batch_states)
            features_list.append(batch_features)

        self.key = master_key
        concatenated = jnp.concatenate(features_list, axis=0)
        return concatenated

    def _build_design_matrix(
        self,
        features: jnp.ndarray,
        *,
        fit: bool,
        washout: bool,
    ) -> jnp.ndarray:
        data = features
        if washout and data.shape[0] > self.washout_steps:
            data = data[self.washout_steps :, ...]
        if fit:
            normalized = self.scaler.fit_transform(data)
            design = self.design_builder.fit_transform(normalized)
        else:
            normalized = self.scaler.transform(data)
            design = self.design_builder.transform(normalized)
        return design

    def _train_readout(
        self,
        design_matrix: jnp.ndarray,
        target_data: jnp.ndarray,
        *,
        classification: bool,
        ridge_lambdas: Optional[Sequence[float]],
    ) -> None:
        result = self.readout.fit(
            design_matrix,
            jnp.asarray(target_data, dtype=jnp.float64),
            classification=classification,
            lambdas=ridge_lambdas or self.ridge_lambdas,
            cv=self._readout_cv,
            n_folds=self._readout_n_folds,
            random_state=self.initial_random_seed,
        )
        self.W_out = jnp.asarray(result.weights, dtype=jnp.float64)
        self.best_ridge_lambda = result.best_lambda
        self.last_training_score = result.score_val
        self.last_training_score_name = result.score_name
        self.last_training_mse = (
            result.score_val if result.score_name.lower() == "mse" else None
        )
        self.ridge_search_log = result.logs

    def train_classification(
        self,
        sequences: jnp.ndarray,
        labels: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
        num_classes: int = 10,
        return_features: bool = False,
    ) -> Optional[jnp.ndarray]:
        features = self._encode_sequences(
            sequences,
            desc="[TRAIN] Encoding sequences",
            leave=True
        )
        design_matrix = self._build_design_matrix(features, fit=True, washout=False)

        labels = labels.astype(jnp.int32)
        targets = jnp.zeros((labels.shape[0], num_classes), dtype=jnp.float64)
        targets = targets.at[jnp.arange(labels.shape[0]), labels].set(1.0)

        self._train_readout(
            design_matrix,
            targets,
            classification=True,
            ridge_lambdas=ridge_lambdas,
        )
        self.classification_mode = True
        self.num_classes = num_classes
        self.trained = True
        if return_features:
            return features
        return None

    def _ensure_classification_ready(self) -> None:
        """Ensure classification inference is only run after training."""
        if not self.classification_mode or self.num_classes is None:
            raise ValueError("Classification mode not enabled. Call train_classification first.")
        if self.W_out is None:
            raise ValueError("Model has not been trained.")

    def predict_classification(
        self,
        sequences: Optional[jnp.ndarray] = None,
        *,
        precomputed_features: Optional[jnp.ndarray] = None,
        progress_desc: Optional[str] = None,
    ) -> jnp.ndarray:
        self._ensure_classification_ready()

        if precomputed_features is None and sequences is None:
            raise ValueError("Either sequences or precomputed_features must be provided.")
        if precomputed_features is not None and sequences is not None:
            raise ValueError("Specify only one of sequences or precomputed_features.")

        phase_desc = progress_desc or "Encoding eval sequences"
        desc_label = f"[PREDICT] {phase_desc}"
        if precomputed_features is not None:
            features = jnp.asarray(precomputed_features, dtype=jnp.float64)
        else:
            features = self._encode_sequences(
                sequences,  # type: ignore[arg-type]
                desc=desc_label,
                leave=True
            )
        design_matrix = self._build_design_matrix(features, fit=False, washout=False)
        logits = self.readout.predict(design_matrix)
        return jnp.asarray(logits, dtype=jnp.float64)

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

        design_matrix = self._build_design_matrix(reservoir_states, fit=True, washout=True)

        target_aligned = target_data
        target_len = target_data.shape[0]
        design_len = design_matrix.shape[0]
        if target_len > design_len:
            start = target_len - design_len
            target_aligned = target_data[start:, ...]
        elif target_len < design_len:
            raise ValueError(
                f"Aligned target data shorter than design matrix ({target_len} vs {design_len})"
            )

        self._train_readout(
            design_matrix,
            target_aligned,
            classification=False,
            ridge_lambdas=ridge_lambdas,
        )
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

        design_matrix = self._build_design_matrix(reservoir_states, fit=False, washout=True)

        predictions = self.readout.predict(design_matrix)
        return jnp.asarray(predictions, dtype=jnp.float64)

    def reset_state(self) -> None:
        """Reset the reservoir to initial state."""
        super().reset_state()  # Reset base class state
        self.W_out = None
        self.best_ridge_lambda = None
        self.ridge_search_log = []
        self.last_training_mse = None
        self.last_training_score = None
        self.last_training_score_name = None
        self.classification_mode = False
        self.num_classes = None
        self.scaler = FeatureScaler()
        self.design_builder = DesignMatrixBuilder(**self._design_cfg)
        if self._external_readout is None:
            self.readout = RidgeReadoutNumpy(
                default_cv=self._readout_cv,
                default_n_folds=self._readout_n_folds,
            )
        else:
            self.readout = self._external_readout
            if hasattr(self.readout, "weights"):
                setattr(self.readout, "weights", None)
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
