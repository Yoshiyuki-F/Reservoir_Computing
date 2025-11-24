"""Pipeline (b): FNN pretraining and fixed-feature ridge readout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Sequence, Optional

import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import train_state
import optax
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import numpy as np
import json

from core_lib.utils import ensure_x64_enabled, calculate_mse
from core_lib.components import FeatureScaler, DesignMatrixBuilder, RidgeReadoutNumpy
from core_lib.models.fnn import FNN, FNNPipelineConfig
from core_lib.models.reservoir.config import parse_ridge_lambdas
from pipelines.plotting import plot_epoch_metric
from pipelines.datasets.mnist_loader import get_mnist_datasets, image_to_sequence
from core_lib.models.reservoir.classical import ReservoirComputer

ensure_x64_enabled()


# --- Helper Functions for De-duplication ---

def _train_and_evaluate_ridge(
        X_train: jnp.ndarray,
        Y_train_onehot: jnp.ndarray,
        X_test: jnp.ndarray,
        Y_test_onehot: jnp.ndarray,
        labels_train: jnp.ndarray,
        labels_test: jnp.ndarray,
        lambdas: Sequence[float],
        description: str = "Ridge Readout",
) -> Dict[str, float]:
    """Ridge回帰の学習、ログ出力、評価を行う共通ロジック"""
    readout = RidgeReadoutNumpy()

    # 学習
    result = readout.fit(
        X_train,
        Y_train_onehot,
        classification=True,
        lambdas=lambdas,
    )

    # ログ出力 (これが欲しかった機能)
    if result.logs:
        print("Ridge λ grid search")
        for entry in result.logs:
            lam = entry.get("lambda")
            val_acc = entry.get("val_accuracy")
            if lam is not None and val_acc is not None:
                print(f"  λ={lam:.2e} -> val Acc={val_acc:.4f}")
        if result.best_lambda is not None:
            print(f"Selected λ={float(result.best_lambda):.2e}")

    # 推論と評価
    train_logits = readout.predict(X_train)
    test_logits = readout.predict(X_test)

    train_mse = calculate_mse(train_logits, Y_train_onehot)
    test_mse = calculate_mse(test_logits, Y_test_onehot)

    train_pred = jnp.argmax(train_logits, axis=1)
    test_pred = jnp.argmax(test_logits, axis=1)

    train_acc = float(np.mean(np.array(train_pred) == np.array(labels_train)))
    test_acc = float(np.mean(np.array(test_pred) == np.array(labels_test)))

    print(
        f"[{description}] "
        f"(best λ={float(result.best_lambda):.2e}) "
        f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}"
    )

    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "best_ridge_lambda": float(result.best_lambda),
        "train_predictions": np.array(train_pred),
        "test_predictions": np.array(test_pred),
        "train_labels": np.array(labels_train),
        "test_labels": np.array(labels_test),
    }


# グローバルな特徴抽出ヘルパー (既存の _predict_features があればそれを使うか、なければここに追加)
def _predict_features(
        model: FNN,
        params: Any,
        loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    labels_list = []
    for batch in loader:
        batch_x = batch[0]
        batch_labels = batch[2]
        x = jnp.asarray(batch_x.numpy(), dtype=jnp.float32)

        # applyの戻り値が (logits, hidden) か features かで分岐
        out = model.apply({"params": params}, x)
        if isinstance(out, tuple):
            _, features = out
        else:
            features = out

        feats.append(np.array(features))
        labels_list.append(batch_labels.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels_list, axis=0)


def _save_params(params: Any, path: Path) -> None:
    bytes_data = serialization.to_bytes(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes_data)


def _load_params(model: FNN, input_dim: int, path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Pretrained FNN weights not found at {path}")
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, jnp.ones((1, input_dim), jnp.float32))
    params_template = variables["params"]
    bytes_data = path.read_bytes()
    return serialization.from_bytes(params_template, bytes_data)


# --- FNN Training Logic ---

def pretrain_fnn(
    config: FNNPipelineConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[float, ...]]:
    """End-to-end pretraining of the FNN classifier (baseline b)."""
    input_dim = config.model.input_dim
    model = FNN(layer_dims=config.model.hidden_dims, return_hidden=False)
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, jnp.ones((1, input_dim), jnp.float32))
    params = variables["params"]
    tx = optax.adam(config.training.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(
        state: train_state.TrainState,
        images: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
        def loss_fn(p):
            logits = state.apply_fn({"params": p}, images)
            one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
            loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        return state, loss, accuracy

    @jax.jit
    def eval_step(
        params_eval: Any,
        images: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logits = model.apply({"params": params_eval}, images)
        one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        return loss, accuracy

    best_params = state.params
    best_test_acc = 0.0
    epoch_indices = []
    test_acc_history = []
    train_acc_history = []

    for epoch in range(1, config.training.num_epochs + 1):
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        for batch_images, batch_labels in train_loader:
            images_np = batch_images.view(batch_images.size(0), -1).numpy()
            labels_np = batch_labels.numpy()
            images = jnp.asarray(images_np, dtype=jnp.float32)
            labels = jnp.asarray(labels_np, dtype=jnp.int32)
            state, loss, acc = train_step(state, images, labels)
            train_loss_sum += float(loss)
            train_acc_sum += float(acc)
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        train_acc = train_acc_sum / max(train_batches, 1)

        test_loss_sum = 0.0
        test_acc_sum = 0.0
        test_batches = 0

        for batch_images, batch_labels in test_loader:
            images_np = batch_images.view(batch_images.size(0), -1).numpy()
            labels_np = batch_labels.numpy()
            images = jnp.asarray(images_np, dtype=jnp.float32)
            labels = jnp.asarray(labels_np, dtype=jnp.int32)
            loss, acc = eval_step(state.params, images, labels)
            test_loss_sum += float(loss)
            test_acc_sum += float(acc)
            test_batches += 1

        test_loss = test_loss_sum / max(test_batches, 1)
        test_acc = test_acc_sum / max(test_batches, 1)

        epoch_indices.append(epoch)
        test_acc_history.append(test_acc)
        train_acc_history.append(train_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_params = state.params

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    weights_path = config.weights_path
    _save_params(best_params, weights_path)
    print(f"[Phase 1] FNN pre-training complete. Best test accuracy={best_test_acc:.4f}.")
    print(f"[Phase 1] Weights saved to: {weights_path}")

    return tuple(epoch_indices), tuple(train_acc_history), tuple(test_acc_history)


def run_fnn_fixed_feature_pipeline(
    config: FNNPipelineConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: Tuple[int, ...],
    train_acc_history: Tuple[float, ...],
    test_acc_history: Tuple[float, ...],
) -> Dict[str, float]:
    """Use pretrained FNN as fixed feature extractor and train ridge readout."""
    input_dim = config.model.input_dim
    model = FNN(layer_dims=config.model.hidden_dims, return_hidden=True)
    params = _load_params(model, input_dim, config.weights_path)

    # 内部関数 _extract_features を削除し、グローバルの _predict_features を使用
    train_features, train_labels = _predict_features(model, params, train_loader)
    test_features, test_labels = _predict_features(model, params, test_loader)

    # Design Matrix (FeatureScalerのロジックはそのまま)
    if config.use_preprocessing:
        scaler = FeatureScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        design_builder = DesignMatrixBuilder(
            poly_mode="square", degree=2, include_bias=True, std_threshold=1e-3
        )
        X_train = design_builder.fit_transform(train_scaled)
        X_test = design_builder.transform(test_scaled)
    else:
        # 手動の _append_bias などを削除し、Builderに統一
        design_builder = DesignMatrixBuilder(
            poly_mode="none", degree=1, include_bias=True, std_threshold=0.0
        )
        X_train = design_builder.fit_transform(train_features)
        X_test = design_builder.transform(test_features)

    # One-hot化
    num_classes = int(jnp.max(train_labels).item()) + 1
    train_one_hot = jax.nn.one_hot(train_labels, num_classes=num_classes).astype(jnp.float64)
    test_one_hot = jax.nn.one_hot(test_labels, num_classes=num_classes).astype(jnp.float64)

    lambda_candidates = parse_ridge_lambdas(
        {"ridge_lambdas": config.ridge_lambdas}, key="ridge_lambdas", default=None
    )

    # 共通関数で学習・評価・ログ出力
    results = _train_and_evaluate_ridge(
        X_train, train_one_hot, X_test, test_one_hot,
        train_labels, test_labels,
        lambda_candidates, description="Phase 2 FNN Fixed-Feature"
    )

    if epochs and test_acc_history:
        curve_filename = f"{config.weights_path.stem}_test_acc_curve.png"
        plot_epoch_metric(
            epochs,
            list(test_acc_history),
            title=f"FNN Pretraining + Phase 2 (best λ={results['best_ridge_lambda']:.2e})",
            filename=curve_filename,
            ylabel="Accuracy",
            metric_name="phase1_test_acc",
            extra_metrics={"phase1_train_acc": list(train_acc_history)},
            phase2_test_acc=results['test_accuracy'],
            phase2_train_acc=results['train_accuracy'],
        )

    return results


# === Reservoir emulation (regression) mode ===

def _load_shared_reservoir_params() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "presets/models/shared_reservoir_params.json"
    return json.loads(path.read_text()).get("params", {})


def _load_training_preset(name: str = "standard") -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "presets/training" / f"{name}.json"
    return json.loads(path.read_text())


def _create_teacher_reservoir(
    reservoir_size: int,
    backend: str,
    ridge_lambdas: Sequence[float],
) -> ReservoirComputer:
    shared = _load_shared_reservoir_params()
    params = {
        **shared,
        "n_inputs": 28,
        "n_hidden_layer": reservoir_size,
        "n_outputs": 1,
        "state_aggregation": "last",
        "ridge_lambdas": ridge_lambdas,
        "use_preprocessing": False,
        "readout_cv": "holdout",
        "readout_n_folds": 5,
    }
    cfg = {
        "name": "classical_reservoir_emulation_teacher",
        "model_type": "reservoir",
        "params": params,
    }
    return ReservoirComputer(cfg, backend=backend)


def _build_teacher_pairs(
    reservoir: ReservoirComputer,
    dataset: Sequence[Any],
    time_steps: int,
    master_key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
    """Generate (flattened projected inputs, final reservoir states, labels) pairs."""
    sequences = []
    labels = []
    for img_tensor, label in dataset:
        img_np = img_tensor.numpy()
        seq = image_to_sequence(img_np, n_steps=time_steps)
        sequences.append(seq.astype(np.float64))
        labels.append(int(label))

    seq_array = jnp.asarray(np.stack(sequences, axis=0), dtype=jnp.float64)
    label_array = jnp.asarray(np.array(labels, dtype=np.int32))

    split_keys = jax.random.split(master_key, seq_array.shape[0] + 1)
    seq_keys = split_keys[:-1]
    next_key = split_keys[-1]

    run_vmapped = jax.vmap(reservoir._run_sequence_with_key, in_axes=(0, 0))
    states = run_vmapped(seq_array, seq_keys)
    final_states = states[:, -1, :]

    projected = jnp.einsum("bti,hi->bth", seq_array, reservoir.W_in)
    flat_inputs = projected.reshape(projected.shape[0], -1)
    return flat_inputs, final_states, label_array, next_key


def _to_dataloader(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    labels: jnp.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_np = np.asarray(inputs, dtype=np.float32)
    y_np = np.asarray(targets, dtype=np.float32)
    lbl_np = np.asarray(labels, dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(x_np),
        torch.from_numpy(y_np),
        torch.from_numpy(lbl_np),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_reservoir_emulation_pipeline(
        config: FNNPipelineConfig,
        *,
        reservoir_size: int,
        time_steps: int,
        backend: str = "cpu",
) -> Dict[str, float]:
    """Train an FNN to emulate a reservoir's final state given projected inputs."""
    training_preset = _load_training_preset("standard")
    preset_lambdas = training_preset.get("ridge_lambdas")
    preset_lambda_grid = None
    if preset_lambdas is not None:
        preset_lambda_grid = parse_ridge_lambdas(
            {"ridge_lambdas": preset_lambdas}, key="ridge_lambdas", default=None
        )

    teacher = _create_teacher_reservoir(
        reservoir_size,
        backend=backend,
        ridge_lambdas=preset_lambda_grid,
    )
    train_set, test_set = get_mnist_datasets()

    master_key = teacher.key
    train_inputs, train_targets, train_labels, key_after_train = _build_teacher_pairs(
        teacher, train_set, time_steps, master_key
    )
    test_inputs, test_targets, test_labels, final_key = _build_teacher_pairs(
        teacher, test_set, time_steps, key_after_train
    )
    teacher.key = final_key

    expected_input_dim = train_inputs.shape[1]
    expected_output_dim = train_targets.shape[1]
    if config.model.input_dim != expected_input_dim:
        raise ValueError(f"Config mismatch: input {config.model.input_dim} vs {expected_input_dim}")
    if config.model.hidden_dims[-1] != expected_output_dim:
        raise ValueError(f"Config mismatch: output {config.model.hidden_dims[-1]} vs {expected_output_dim}")

    train_loader = _to_dataloader(
        train_inputs, train_targets, train_labels, config.training.batch_size, shuffle=True
    )
    test_loader = _to_dataloader(
        test_inputs, test_targets, test_labels, config.training.batch_size, shuffle=False
    )

    model = FNN(layer_dims=config.model.hidden_dims, return_hidden=False)
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, jnp.ones((1, expected_input_dim), jnp.float32))
    params = variables["params"]
    tx = optax.adam(config.training.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch_x, batch_y):
        def loss_fn(p):
            preds = state.apply_fn({"params": p}, batch_x)
            return jnp.mean(jnp.square(preds - batch_y))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_step(params_eval, batch_x, batch_y):
        preds = model.apply({"params": params_eval}, batch_x)
        return jnp.mean(jnp.square(preds - batch_y))

    best_params = state.params
    best_test_mse = float("inf")

    # --- History Recording ---
    epoch_indices = []
    train_mse_history = []
    test_mse_history = []

    print("[Phase 1] Training FNN Regression...")
    for epoch in range(1, config.training.num_epochs + 1):
        train_loss_sum = 0.0
        train_batches = 0
        for batch_x, batch_y, _ in train_loader:
            x = jnp.asarray(batch_x.numpy(), dtype=jnp.float32)
            y = jnp.asarray(batch_y.numpy(), dtype=jnp.float32)
            state, loss = train_step(state, x, y)
            train_loss_sum += float(loss)
            train_batches += 1
        train_mse = train_loss_sum / max(train_batches, 1)

        test_loss_sum = 0.0
        test_batches = 0
        for batch_x, batch_y, _ in test_loader:
            x = jnp.asarray(batch_x.numpy(), dtype=jnp.float32)
            y = jnp.asarray(batch_y.numpy(), dtype=jnp.float32)
            loss = eval_step(state.params, x, y)
            test_loss_sum += float(loss)
            test_batches += 1
        test_mse = test_loss_sum / max(test_batches, 1)

        # Record history
        epoch_indices.append(epoch)
        train_mse_history.append(float(train_mse))
        test_mse_history.append(float(test_mse))

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_params = state.params
        print(f"[Epoch {epoch:03d}] train_mse={train_mse:.6f}, test_mse={test_mse:.6f}")

    weights_path = config.weights_path
    _save_params(best_params, weights_path)
    print(f"[Reservoir Emulation] Weights saved to: {weights_path}")

    # --- Plot Phase 1 MSE Curve (New) ---
    weights_path = Path(config.weights_path)
    curve_filename = weights_path.with_name(f"{weights_path.stem}_mse_curve.png")
    plot_epoch_metric(
        tuple(epoch_indices),
        tuple(test_mse_history),
        title=f"Reservoir Emulation Phase 1 (H={config.model.hidden_dims[1]})",
        filename=str(curve_filename),
        ylabel="MSE (Loss)",
        metric_name="test_mse",
        extra_metrics={"train_mse": tuple(train_mse_history)},
    )
    print(f"[Reservoir Emulation] Plot saved to: {curve_filename}")

    # Phase 2: Use emulated reservoir states as features for classification
    print("\n[Phase 2] Training ridge readout on FNN-emulated states...")

    train_feats, train_labels_aligned = _predict_features(model, best_params, train_loader)
    test_feats, test_labels_aligned = _predict_features(model, best_params, test_loader)

    design_builder = DesignMatrixBuilder(
        poly_mode="none", degree=1, include_bias=True, std_threshold=0.0
    )
    train_feats_b = design_builder.fit_transform(jnp.asarray(train_feats, dtype=jnp.float64))
    test_feats_b = design_builder.transform(jnp.asarray(test_feats, dtype=jnp.float64))

    num_classes = 10
    train_labels_onehot = jax.nn.one_hot(train_labels_aligned, num_classes=num_classes).astype(jnp.float64)
    test_labels_onehot = jax.nn.one_hot(test_labels_aligned, num_classes=num_classes).astype(jnp.float64)

    lambda_candidates = teacher.ridge_lambdas or parse_ridge_lambdas(
        {"ridge_lambdas": config.ridge_lambdas}, key="ridge_lambdas", default=None
    )

    results = _train_and_evaluate_ridge(
        train_feats_b, train_labels_onehot, test_feats_b, test_labels_onehot,
        train_labels_aligned, test_labels_aligned,
        lambda_candidates, description="Phase 2 Emulation Readout"
    )

    results["train_mse"] = train_mse
    results["test_mse"] = best_test_mse
    return results
