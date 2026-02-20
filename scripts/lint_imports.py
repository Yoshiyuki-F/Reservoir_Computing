#!/usr/bin/env python3
"""
scripts/lint_imports.py
Import Boundary Enforcement — 物理的制約

Hexagonal Architecture for ML: NumPy/JAX import境界を自動監視。
Mapper以外のファイルで両方をimportしていたらエラーで弾く。

Usage:
    uv run python scripts/lint_imports.py
"""
import re
import sys
from pathlib import Path

# ===== Domain Registry =====
# Mapper files: BOTH np and jax imports explicitly allowed (boundary converters)
MAPPERS = {
    "core/types.py",                # Type alias bridge: NpF64, JaxF64
    "utils/batched_compute.py",     # The Gateway: np → jax → np
    "utils/metrics.py",             # np inputs → jax computation → float output
    "utils/reporting.py",           # np stats formatting  
    "pipelines/strategies.py",      # Orchestrates np frontend ↔ jax readout
    "pipelines/components/executor.py",  # Delegates between domains
    "readout/ridge.py",             # Ridge solve uses both domains
}

# Files where jax import inside conditional/function is acceptable
CONDITIONAL_JAX_OK = {
    "data/generators.py",           # Legacy: conditional jax import in function body
}

SRC_ROOT = Path(__file__).parent.parent / "src" / "reservoir"

_import_np  = re.compile(r"^\s*(import numpy|from numpy|import numpy\.)")
# jaxtyping and beartype are TYPE ANNOTATION tools, not JAX computation.
# They are allowed in NumPy-domain files for Float64[np.ndarray] enforcement.
_import_jax = re.compile(r"^\s*(import jax\b|from jax\b|import jaxlib)")


def check_file(path: Path) -> list[str]:
    """Return list of violation messages for a file."""
    rel = str(path.relative_to(SRC_ROOT))
    
    # Skip __init__.py and __pycache__
    if path.name == "__init__.py" or "__pycache__" in str(path):
        return []
    
    # Mappers are exempt
    if rel in MAPPERS:
        return []

    text = path.read_text()
    lines = text.splitlines()
    
    has_np = any(_import_np.match(line) for line in lines)
    has_jax = any(_import_jax.match(line) for line in lines)
    
    if has_np and has_jax:
        if rel in CONDITIONAL_JAX_OK:
            return []  # Known exception
        return [f"❌ BOUNDARY VIOLATION: {rel} imports BOTH numpy AND jax (not a registered Mapper)"]
    
    return []


def main():
    violations = []
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        violations.extend(check_file(py_file))
    
    if violations:
        print("=" * 60)
        print("Import Boundary Violations Detected")
        print("=" * 60)
        for v in violations:
            print(v)
        print(f"\nTotal: {len(violations)} violation(s)")
        print("\nFix: Move the file to MAPPERS in scripts/lint_imports.py")
        print("     or remove the forbidden import.")
        sys.exit(1)
    else:
        print("✅ All import boundaries clean. No violations.")
        sys.exit(0)


if __name__ == "__main__":
    main()
