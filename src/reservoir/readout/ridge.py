"""
src/reservoir/components/readout/ridge.py
Ridge regression implementation compliant with ReadoutModule protocol.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from reservoir.core.interfaces import ReadoutModule


class RidgeRegression(ReadoutModule):
    """Ridge regression readout solved with JAX linear algebra."""

    def __init__(self, ridge_lambda: float, use_intercept: bool = True) -> None:
        if ridge_lambda is None:
            raise ValueError("RidgeRegression requires an explicit, positive ridge_lambda.")
        lambda_val = float(ridge_lambda)
        if lambda_val <= 0.0:
            raise ValueError(f"RidgeRegression ridge_lambda must be positive, got {lambda_val}.")
        self.ridge_lambda = lambda_val
        self.use_intercept = bool(use_intercept)
        self.coef_: Optional[jnp.ndarray] = None
        self.intercept_: Optional[jnp.ndarray] = None
        self.input_dim_: Optional[int] = None

    def _prepare_xy(self, states: jnp.ndarray, targets: jnp.ndarray, *, update_dim: bool) \
            -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        X = jnp.asarray(states, dtype=jnp.float64)
        if X.ndim != 2:
            raise ValueError(f"States must be 2D, got {X.shape}")
        y_arr = jnp.asarray(targets, dtype=jnp.float64)
        y_is_1d = y_arr.ndim == 1
        if y_is_1d:
            y_arr = y_arr[:, None]
        if y_arr.shape[0] != X.shape[0]:
            raise ValueError(f"Mismatched samples: states have {X.shape[0]}, targets have {y_arr.shape[0]}.")

        n_samples, n_features = X.shape
        if update_dim:
            self.input_dim_ = n_features
        elif self.input_dim_ is not None and n_features != self.input_dim_:
            raise ValueError(f"Expected states with {self.input_dim_} features, got {n_features}.")

        if self.use_intercept:
            ones = jnp.ones((n_samples, 1), dtype=X.dtype)
            X = jnp.concatenate([ones, X], axis=1)
        return X, y_arr, y_is_1d

    def fit(self, states: jnp.ndarray, targets: jnp.ndarray) -> "RidgeRegression":
        """Fit a single ridge model without validation search."""
        X, y_arr, y_is_1d = self._prepare_xy(states, targets, update_dim=True)
        n_features = X.shape[1]
        eye = jnp.eye(n_features, dtype=X.dtype)
        XtX = X.T @ X
        Xty = X.T @ y_arr
        w = jax.scipy.linalg.solve(XtX + self.ridge_lambda * eye, Xty, assume_a="pos")
        if not jnp.all(jnp.isfinite(w)):
            w = jnp.zeros_like(w)
        if self.use_intercept:
            self.intercept_ = jnp.asarray(w[0], dtype=jnp.float64)
            self.coef_ = jnp.asarray(w[1:], dtype=jnp.float64)
        else:
            self.intercept_ = jnp.zeros(w.shape[1], dtype=jnp.float64)
            self.coef_ = jnp.asarray(w, dtype=jnp.float64)
        if y_is_1d:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = self.intercept_.ravel()
        return self

    def fit_and_search(
        self,
        train_states: jnp.ndarray,
        train_targets: jnp.ndarray,
        val_states: jnp.ndarray,
        val_targets: jnp.ndarray,
        lambdas: Optional[jnp.ndarray],
        *,
        metric: str = "mse",
    ) -> tuple[float, Dict[float, float], Dict[float, float]]:
        if val_states is None or val_targets is None:
            raise ValueError("Validation data is required for hyperparameter search.")

        lambda_candidates = [float(lam) for lam in (lambdas or [self.ridge_lambda])]
        if not lambda_candidates:
            raise ValueError("At least one lambda candidate must be provided.")
        if any(lam <= 0.0 for lam in lambda_candidates):
            raise ValueError(f"All ridge lambdas must be positive, got {lambda_candidates}.")
        lambdas_arr = jnp.asarray(lambda_candidates, dtype=jnp.float64)

        X_train, y_train, y_is_1d = self._prepare_xy(train_states, train_targets, update_dim=True)
        X_val, y_val, y_val_is_1d = self._prepare_xy(val_states, val_targets, update_dim=False)

        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_train
        eye = jnp.eye(XtX.shape[0], dtype=XtX.dtype)

        def solve_lambda(lam: jnp.ndarray) -> jnp.ndarray:
            lam_val = float(lam)
            if lam_val < 1e-7:  # 閾値は適宜調整
                # 簡易対策: LU分解(solve) 遅いお
                # default = "gem" (一般行列)
                # A = L*U not root calculation
                return jax.scipy.linalg.solve(XtX + lam * eye, Xty)  # assume_a="gen"
            else: # positive-definite matrix Cholesky(pos)
                # A = LL^T calculation
                # (it might be doing root calculation of negative number internally) which might cause NaN
                return jax.scipy.linalg.solve(XtX + lam * eye, Xty, assume_a="pos")


        # --- 【修正前】 vmapで一括解決 ---
        # weights_all = jax.vmap(solve_lambda, in_axes=0, out_axes=0)(lambdas_arr)

        # --- 【修正後】 Pythonループで1つずつ解く ---
        weights_list = []
        for lam in lambdas_arr:
            # 1つ解く -> メモリ解放 を繰り返すことでピークメモリを抑える
            w = solve_lambda(lam)
            weights_list.append(w)

        weights_all = jnp.stack(weights_list)
        preds_all = jnp.einsum("nd,kdo->kno", X_val, weights_all)

        if metric == "accuracy":
            true_labels = y_val.ravel() if y_val_is_1d else jnp.argmax(y_val, axis=-1)
            pred_labels = preds_all if preds_all.ndim == 2 else jnp.argmax(preds_all, axis=-1)
            accs = jnp.mean(pred_labels == true_labels, axis=-1)
            best_idx = int(jnp.argmax(accs))
            search_history = {float(lambdas_arr[i]): float(accs[i]) for i in range(accs.shape[0])}
        else:
            errors = jnp.mean((preds_all - y_val[None, ...]) ** 2, axis=(1, 2))
            best_idx = int(jnp.argmin(errors))
            search_history = {float(lambdas_arr[i]): float(errors[i]) for i in range(errors.shape[0])}

        weight_norms = {float(lambdas_arr[i]): float(jnp.linalg.norm(weights_all[i])) for i in range(weights_all.shape[0])}

        best_lambda = float(lambdas_arr[best_idx])
        best_weights = weights_all[best_idx]
        if self.use_intercept:
            self.intercept_ = jnp.asarray(best_weights[0], dtype=jnp.float64)
            self.coef_ = jnp.asarray(best_weights[1:], dtype=jnp.float64)
        else:
            self.intercept_ = jnp.zeros(best_weights.shape[1], dtype=jnp.float64)
            self.coef_ = jnp.asarray(best_weights, dtype=jnp.float64)
        if y_is_1d:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = self.intercept_.ravel()
        self.ridge_lambda = best_lambda
        return best_lambda, search_history, weight_norms

    def predict(self, states: jnp.ndarray) -> jnp.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeRegression is not fitted yet.")
        X = jnp.asarray(states, dtype=jnp.float64)
        return X @ self.coef_ + self.intercept_ if self.use_intercept else X @ self.coef_

    def to_dict(self) -> Dict[str, Any]:
        data = {"ridge_lambda": self.ridge_lambda, "use_intercept": self.use_intercept}
        if self.coef_ is not None:
            data["coef"] = jnp.asarray(self.coef_).tolist()
            data["intercept"] = jnp.asarray(self.intercept_).tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RidgeRegression":
        if "ridge_lambda" not in data:
            raise ValueError("Serialized RidgeRegression is missing required 'ridge_lambda'.")
        model = cls(ridge_lambda=float(data["ridge_lambda"]), use_intercept=bool(data.get("use_intercept", True)))
        if "coef" in data:
            model.coef_ = jnp.asarray(data["coef"], dtype=jnp.float64)
            model.intercept_ = jnp.asarray(data.get("intercept", 0.0), dtype=jnp.float64)
        return model
