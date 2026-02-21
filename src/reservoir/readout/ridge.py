"""
src/reservoir/components/readout/ridge.py
Refactored Ridge regression designed with SOLID principles.
"""
import jax
import jax.numpy as jnp
from reservoir.core.types import JaxF64, ConfigDict
import jax.scipy.linalg

from reservoir.core.interfaces import ReadoutModule

# Robustness: Ridge Regression often requires x64 for matrix inversion stability.
def _ensure_x64() -> None:
    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)

class RidgeRegression(ReadoutModule):
    """
    Single Ridge Regression model (Pure JAX).
    Responsible ONLY for fitting parameters given a fixed lambda (SRP).
    """

    def __init__(self, ridge_lambda: float, use_intercept: bool = True) -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.use_intercept = bool(use_intercept)
        self.coef_: JaxF64 | None = None
        self.intercept_: JaxF64 | None = None
        self._input_dim: int | None = None

    def _add_intercept(self, X: JaxF64) -> JaxF64:
        if not self.use_intercept:
            return X
        n_samples = X.shape[0]
        ones = jnp.ones((n_samples, 1))
        return jnp.concatenate([ones, X], axis=1)

    def fit(self, states: JaxF64, targets: JaxF64) -> "RidgeRegression":
        _ensure_x64()
        # Inputs are already JaxF64 per type hint
        X = states
        y = targets
        
        if y.ndim == 1:
            y = y[:, None]
        
        self._input_dim = int(X.shape[1])
        
        # Prepare Data
        X_design = self._add_intercept(X)
        n_features = X_design.shape[1]

        # Solve Normal Equations: (X^T X + lambda I) w = X^T y
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y
        
        # Regularization matrix
        eye = jnp.eye(n_features)
        reg_matrix = eye * self.ridge_lambda
        
        # JAX Solver (assuming x64 enabled externally if needed)
        solve_mat = XtX + reg_matrix
        
        # Robust solver
        if self.ridge_lambda < 1e-7:
             w = jax.scipy.linalg.solve(solve_mat, Xty)
        else:
             w = jax.scipy.linalg.solve(solve_mat, Xty, assume_a="pos")

        # Handle NaNs
        w = jnp.where(jnp.isfinite(w), w, jnp.zeros_like(w))

        # Extract weights
        if self.use_intercept:
            self.intercept_ = w[0].ravel()
            self.coef_ = w[1:]
        else:
            self.intercept_ = jnp.zeros(w.shape[1])
            self.coef_ = w

        # Flatten if original target was 1D
        if targets.ndim == 1:
            self.coef_ = self.coef_.ravel()
        
        return self

    def predict(self, states: JaxF64) -> JaxF64:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        
        X = states
        pred = X @ self.coef_
        if self.intercept_ is not None:
             pred = pred + self.intercept_
        return pred

    def to_dict(self) -> ConfigDict:
        # Use .tolist() which is a safe JAX method to get primitives for JSON/serialization
        return {
            "ridge_lambda": self.ridge_lambda,
            "use_intercept": self.use_intercept,
            "coef": tuple(self.coef_.tolist()) if self.coef_ is not None else None,
            "intercept": tuple(self.intercept_.tolist()) if self.intercept_ is not None else None
        }

    @classmethod
    def from_dict(cls, data: ConfigDict) -> "RidgeRegression":
        lam_val = data.get("ridge_lambda")
        r_lam = float(lam_val) if isinstance(lam_val, (int, float, str)) else 0.0
        model = cls(
            ridge_lambda=r_lam,
            use_intercept=bool(data.get("use_intercept", True))
        )
        if data.get("coef") is not None:
            model.coef_ = jnp.array(data["coef"], dtype=jnp.float64)
            model.intercept_ = jnp.array(data.get("intercept", 0.0), dtype=jnp.float64)
        return model


class RidgeCV(ReadoutModule):
    """
    Orchestrates validation search (Pure JAX).
    Acts as the primary ReadoutModule that wraps RidgeRegression.
    """
    def __init__(
        self, 
        lambda_candidates: tuple[float, ...],
        use_intercept: bool = True
    ):
        if not lambda_candidates:
            raise ValueError("lambda_candidates must not be empty.")

        self.lambda_candidates = lambda_candidates
        self.use_intercept = use_intercept
        self.best_model: RidgeRegression | None = None

        # Initialize default model with first candidate (state consistency)
        if self.lambda_candidates:
            self.best_model = RidgeRegression(self.lambda_candidates[0], use_intercept)

    @property
    def ridge_lambda(self) -> float:
        return self.best_model.ridge_lambda if self.best_model else float(self.lambda_candidates[0])

    @property
    def coef_(self) -> JaxF64 | None:
        return self.best_model.coef_ if self.best_model else None

    @property
    def intercept_(self) -> JaxF64 | None:
        return self.best_model.intercept_ if self.best_model else None

    def fit(self, states: JaxF64, targets: JaxF64) -> "RidgeCV":
        """Fit using current best/default lambda."""
        if self.best_model is None:
             self.best_model = RidgeRegression(self.lambda_candidates[0], self.use_intercept)
        self.best_model.fit(states, targets)
        return self

    def predict(self, states: JaxF64) -> JaxF64:
        if self.best_model is None:
            raise RuntimeError("RidgeCV model is not fitted.")
        return self.best_model.predict(states)

    def to_dict(self) -> ConfigDict:
        data = self.best_model.to_dict() if self.best_model else {}
        res: ConfigDict = dict(data)
        res["lambda_candidates"] = tuple(self.lambda_candidates)
        return res

    @classmethod
    def from_dict(cls, data: ConfigDict) -> "RidgeCV":
        candidates_list = data.get("lambda_candidates")
        if candidates_list is None:
             lam_val = data.get("ridge_lambda")
             candidates = (float(lam_val) if isinstance(lam_val, (int, float, str)) else 0.0,)
        elif isinstance(candidates_list, (list, tuple)):
             candidates = tuple(float(x) for x in candidates_list if isinstance(x, (int, float, str)))
        else:
             candidates = (0.0,)
             
        instance = cls(lambda_candidates=candidates, use_intercept=bool(data.get("use_intercept", True)))
        instance.best_model = RidgeRegression.from_dict(data)
        return instance
