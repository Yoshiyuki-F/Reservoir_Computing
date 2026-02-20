"""
src/reservoir/components/readout/ridge.py
Refactored Ridge regression designed with SOLID principles.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Callable
import jax
import jax.numpy as jnp
from reservoir.core.types import JaxF64
import jax.scipy.linalg
import numpy as np
from tqdm import tqdm

from reservoir.core.interfaces import ReadoutModule

# Robustness: Ridge Regression often requires x64 for matrix inversion stability.
# We enable it lazily to avoid unnecessary overhead if not used, 
# though JAX config is global.
def _ensure_x64():
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)

# --- Helper Types ---
ScoringFn = Callable[[np.ndarray, np.ndarray], float]

class RidgeRegression(ReadoutModule):
    """
    Single Ridge Regression model.
    Responsible ONLY for fitting parameters given a fixed lambda (SRP).
    """

    def __init__(self, ridge_lambda: float, use_intercept: bool = True) -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.use_intercept = bool(use_intercept)
        self.coef_: Optional[JaxF64] = None
        self.intercept_: Optional[JaxF64] = None
        self._input_dim: Optional[int] = None

    def _add_intercept(self, X: JaxF64) -> JaxF64:
        if not self.use_intercept:
            return X
        n_samples = X.shape[0]
        ones = jnp.ones((n_samples, 1))
        return jnp.concatenate([ones, X], axis=1)

    def fit(self, states: JaxF64, targets: JaxF64) -> "RidgeRegression":
        _ensure_x64()
        # Ensure input formats
        X = jnp.asarray(states)
        y = jnp.asarray(targets)
        
        if y.ndim == 1:
            y = y[:, None]
        
        self._input_dim = X.shape[1]
        
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
        
        X = jnp.asarray(states)
        pred = X @ self.coef_
        if self.intercept_ is not None:
             pred = pred + self.intercept_
        return pred

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ridge_lambda": self.ridge_lambda,
            "use_intercept": self.use_intercept,
            "coef": jnp.asarray(self.coef_).tolist() if self.coef_ is not None else None,
            "intercept": jnp.asarray(self.intercept_).tolist() if self.intercept_ is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RidgeRegression":
        model = cls(
            ridge_lambda=float(data["ridge_lambda"]),
            use_intercept=bool(data["use_intercept"])
        )
        if data.get("coef") is not None:
            model.coef_ = jnp.asarray(data["coef"])
            model.intercept_ = jnp.asarray(data.get("intercept", 0.0))
        return model


class RidgeCV(ReadoutModule):
    """
    Orchestrates validation search (OCP/DIP).
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
        self.best_model: Optional[RidgeRegression] = None

        # Initialize default model with first candidate (state consistency)
        if self.lambda_candidates:
            self.best_model = RidgeRegression(self.lambda_candidates[0], use_intercept)

    @property
    def ridge_lambda(self) -> float:
        return self.best_model.ridge_lambda if self.best_model else float(self.lambda_candidates[0])

    @property
    def coef_(self):
        return self.best_model.coef_ if self.best_model else None

    @property
    def intercept_(self):
        return self.best_model.intercept_ if self.best_model else None

    def fit_with_validation(
        self, 
        train_Z: JaxF64, 
        train_y: JaxF64, 
        val_Z: JaxF64, 
        val_y: JaxF64, 
        scoring_fn: ScoringFn, 
        maximize_score: bool = True
    ) -> tuple[float, float, Dict[float, float], Dict[float, float]]:
        
        best_score = float('-inf') if maximize_score else float('inf')
        best_lambda = self.lambda_candidates[0]
        search_history = {}
        weight_norms = {}
        residuals_history = {} # For BoxPlot

        print(f"    [RidgeCV] Optimizing over {len(self.lambda_candidates)} candidates...")

        for lam in tqdm(self.lambda_candidates, desc="[RidgeCV Search]"):
            lam_val = float(lam)
            model = RidgeRegression(ridge_lambda=lam_val, use_intercept=self.use_intercept)
            model.fit(train_Z, train_y)
            
            # Predict & Score
            val_pred = model.predict(val_Z)
            score_out = scoring_fn(np.asarray(val_pred), np.asarray(val_y))
            
            if isinstance(score_out, tuple):
                score, res_sq = score_out
            else:
                score = score_out
                # Default behavior: residuals in the space passed to fit
                vp = np.asarray(val_pred).ravel()
                vy = np.asarray(val_y).ravel()
                res_sq = (vp - vy) ** 2
            
            search_history[lam_val] = score
            residuals_history[lam_val] = res_sq
            
            if model.coef_ is not None:
                weight_norms[lam_val] = float(jnp.linalg.norm(model.coef_))

            # Compare
            is_better = (score > best_score) if maximize_score else (score < best_score)
            if is_better:
                best_score = score
                best_lambda = lam_val
                
        print(f"    [RidgeCV] Best Lambda: {best_lambda:.5e} (Score: {best_score:.5f})")
        
        # Final Fit
        self.best_model = RidgeRegression(ridge_lambda=best_lambda, use_intercept=self.use_intercept)
        self.best_model.fit(train_Z, train_y)
        
        return best_lambda, best_score, search_history, weight_norms, residuals_history

    def fit(self, states: JaxF64, targets: JaxF64) -> "RidgeCV":
        """Fallback fit without validation (uses current best/default lambda)."""
        if self.best_model is None:
             self.best_model = RidgeRegression(self.lambda_candidates[0], self.use_intercept)
        self.best_model.fit(states, targets)
        return self

    def predict(self, states: JaxF64) -> JaxF64:
        if self.best_model is None:
            raise RuntimeError("RidgeCV model is not fitted.")
        return self.best_model.predict(states)

    def to_dict(self) -> Dict[str, Any]:
        data = self.best_model.to_dict() if self.best_model else {}
        # Decorate with candidates for full restoration
        data["lambda_candidates"] = list(self.lambda_candidates)
        # Ensure ridge_lambda is set (RidgeRegression.to_dict does it, but purely for safety)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RidgeCV":
        candidates_list = data.get("lambda_candidates")
        # Legacy handling
        if candidates_list is None:
             candidates = (float(data["ridge_lambda"]),)
        else:
             candidates = tuple(candidates_list)
             
        instance = cls(lambda_candidates=candidates, use_intercept=bool(data.get("use_intercept", True)))
        
        # Restore inner best model
        instance.best_model = RidgeRegression.from_dict(data)
        return instance
