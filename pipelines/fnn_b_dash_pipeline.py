"""Pipeline (b'): FNN with frozen input mapping and two hidden layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import train_state
import optax
from torch.utils.data import DataLoader

from core_lib.utils import ensure_x64_enabled, calculate_mse
from core_lib.components import FeatureScaler, DesignMatrixBuilder, RidgeReadoutNumpy
from core_lib.models.fnn import FNN, FNNPipelineConfig
from core_lib.models.reservoir.config import parse_ridge_lambdas
from pipelines.plotting import plot_epoch_metric


ensure_x64_enabled()


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


def pretrain_fnn_b_dash(
    config: FNNPipelineConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Pretraining for pipeline (b'): freeze input->first hidden weights."""
    input_dim = config.model.input_dim
    model = FNN(features=config.model.hidden_dims, return_hidden=False)
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, jnp.ones((1, input_dim), jnp.float32))
    params = variables["params"]
    tx = optax.adam(config.training.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Build a gradient mask that freezes the first Dense layer (input -> first hidden)
    grad_mask = jax.tree_util.tree_map(jnp.ones_like, state.params)
    if isinstance(grad_mask, dict) and "Dense_0" in grad_mask:
        grad_mask = dict(grad_mask)
        grad_mask["Dense_0"] = jax.tree_util.tree_map(
            jnp.zeros_like,
            grad_mask["Dense_0"],
        )

    @jax.jit
    def train_step(
        state: train_state.TrainState,
        images: jnp.ndarray,
        labels: jnp.ndarray,
        mask: Any,
    ) -> Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
        def loss_fn(p):
            logits = state.apply_fn({"params": p}, images)
            one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
            loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        masked_grads = jax.tree_util.tree_map(
            lambda g, m: g * m,
            grads,
            mask,
        )
        state = state.apply_gradients(grads=masked_grads)
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
            state, loss, acc = train_step(state, images, labels, grad_mask)
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
            f"[b' Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    weights_path = config.weights_path
    _save_params(best_params, weights_path)
    print(f"[Phase 1 b'] FNN pre-training complete. Best test accuracy={best_test_acc:.4f}.")
    print(f"[Phase 1 b'] Weights saved to: {weights_path}")

    epochs_out = tuple(int(e) for e in epoch_indices)
    train_hist_out = tuple(float(a) for a in train_acc_history)
    test_hist_out = tuple(float(a) for a in test_acc_history)
    return epochs_out, train_hist_out, test_hist_out


def run_fnn_fixed_feature_pipeline_b_dash(
    config: FNNPipelineConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: Tuple[int, ...],
    train_acc_history: Tuple[float, ...],
    test_acc_history: Tuple[float, ...],
) -> Dict[str, float]:
    """Use pretrained FNN (b') as fixed feature extractor and train ridge readout."""
    input_dim = config.model.input_dim
    model = FNN(features=config.model.hidden_dims, return_hidden=True)
    params = _load_params(model, input_dim, config.weights_path)

    @jax.jit
    def feature_step(
        params_eval: Any,
        images: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logits, hidden = model.apply({"params": params_eval}, images)
        return logits, hidden

    def _extract_features(loader: DataLoader) -> Tuple[jnp.ndarray, jnp.ndarray]:
        feats_list = []
        labels_list = []
        for batch_images, batch_labels in loader:
            images_np = batch_images.view(batch_images.size(0), -1).numpy()
            labels_np = batch_labels.numpy()
            images = jnp.asarray(images_np, dtype=jnp.float32)
            logits, hidden = feature_step(params, images)
            feats_list.append(jnp.asarray(hidden, dtype=jnp.float64))
            labels_list.append(jnp.asarray(labels_np, dtype=jnp.int32))
        features = jnp.concatenate(feats_list, axis=0)
        labels = jnp.concatenate(labels_list, axis=0)
        return features, labels

    train_features, train_labels = _extract_features(train_loader)
    test_features, test_labels = _extract_features(test_loader)

    if config.use_preprocessing:
        scaler = FeatureScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        design_builder = DesignMatrixBuilder(
            poly_mode="square",
            degree=2,
            include_bias=True,
            std_threshold=1e-3,
        )
        X_train = design_builder.fit_transform(train_scaled)
        X_test = design_builder.transform(test_scaled)
    else:
        X_train_core = jnp.asarray(train_features, dtype=jnp.float64)
        X_test_core = jnp.asarray(test_features, dtype=jnp.float64)
        bias_train = jnp.ones((X_train_core.shape[0], 1), dtype=X_train_core.dtype)
        bias_test = jnp.ones((X_test_core.shape[0], 1), dtype=X_test_core.dtype)
        X_train = jnp.concatenate([X_train_core, bias_train], axis=1)
        X_test = jnp.concatenate([X_test_core, bias_test], axis=1)

    num_classes = int(jnp.max(train_labels).item()) + 1
    train_one_hot = jax.nn.one_hot(train_labels, num_classes=num_classes).astype(jnp.float64)
    test_one_hot = jax.nn.one_hot(test_labels, num_classes=num_classes).astype(jnp.float64)

    readout = RidgeReadoutNumpy()
    lambda_candidates = parse_ridge_lambdas(
        {"ridge_lambdas": config.ridge_lambdas},
        key="ridge_lambdas",
        default=None,
    )
    result = readout.fit(
        X_train,
        train_one_hot,
        classification=True,
        lambdas=lambda_candidates,
    )

    train_logits = readout.predict(X_train)
    test_logits = readout.predict(X_test)

    train_mse = calculate_mse(train_logits, train_one_hot)
    test_mse = calculate_mse(test_logits, test_one_hot)

    train_pred = jnp.argmax(train_logits, axis=1)
    test_pred = jnp.argmax(test_logits, axis=1)
    train_accuracy = float(jnp.mean(train_pred == train_labels))
    test_accuracy = float(jnp.mean(test_pred == test_labels))

    print(
        f"[Phase 2 b'] FNN fixed-feature ridge readout "
        f"(best λ={float(result.best_lambda):.2e}) "
        f"train_acc={train_accuracy:.4f}, test_acc={test_accuracy:.4f}"
    )

    if epochs and test_acc_history:
        curve_filename = f"{config.weights_path.stem}_test_acc_curve.png"
        plot_epoch_metric(
            epochs,
            list(test_acc_history),
            title=f"FNN Pretraining b' + Phase 2 (best λ={float(result.best_lambda):.2e})",
            filename=curve_filename,
            ylabel="Accuracy",
            metric_name="phase1_test_acc_b_dash",
            extra_metrics={
                "phase1_train_acc_b_dash": list(train_acc_history),
            },
            phase2_test_acc=test_accuracy,
            phase2_train_acc=train_accuracy,
        )

    results: Dict[str, float] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "best_ridge_lambda": float(result.best_lambda),
    }
    return results
