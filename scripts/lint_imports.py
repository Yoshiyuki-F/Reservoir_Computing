#!/usr/bin/env python3
"""
scripts/lint_imports.py
Import Boundary & Type Enforcement — 物理的制約

Hexagonal Architecture for ML:
1. NumPy/JAX import境界を自動監視。
2. 厳格な型エイリアス (JaxF64, NpF64) の使用を強制。
3. Any, Union, float32, 生のndarray/Array型ヒントを禁止。

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
    "models/nn/base.py",            # Training loop handles data transfer
    "models/nn/fnn.py",             # Training loop handles data transfer
    "models/distillation/model.py", # Training loop handles data transfer
    "models/passthrough/passthrough.py", # Training loop handles data transfer
    "models/reservoir/base.py", # Abstract Base Class
    "models/reservoir/classical/classical.py", # Classical Reservoir
}

# Files where jax import inside conditional/function is acceptable
CONDITIONAL_JAX_OK = {
    "data/generators.py",           # Legacy: conditional jax import in function body
}

SRC_ROOT = Path(__file__).parent.parent / "src" / "reservoir"

_import_np  = re.compile(r"^\s*(import numpy|from numpy|import numpy\.)")
_import_jax = re.compile(r"^\s*(import jax\b|from jax\b|import jaxlib)")

# Forbidden patterns
_forbidden_union = re.compile(r"Union\[")
_forbidden_any = re.compile(r":\s*Any\b|->\s*Any\b")
_allowed_any_kwargs = re.compile(r"\*\*kwargs:\s*Any")
_allowed_any_default = re.compile(r"Any\s*=\s*None") # e.g. target: Any = None
_forbidden_float32 = re.compile(r"\bfloat32\b")
_forbidden_raw_hints = re.compile(r":\s*(?:jnp\.|np\.|jax\.|numpy\.)?(?:ndarray|Array)\b")
_forbidden_asarray = re.compile(r"jnp\.asarray")
_forbidden_np_asarray = re.compile(r"np\.asarray")

def check_file(path: Path) -> list[str]:
    """Return list of violation messages for a file."""
    rel = str(path.relative_to(SRC_ROOT))
    
    # Skip __init__.py and __pycache__
    if path.name == "__init__.py" or "__pycache__" in str(path):
        return []
    
    text = path.read_text()
    lines = text.splitlines()
    violations = []

    has_np = False
    has_jax = False

    for i, line in enumerate(lines, 1):
        # 1. Boundary Check
        if _import_np.match(line): has_np = True
        if _import_jax.match(line): has_jax = True

        # 2. Type Hints Check (Any, Union)
        if _forbidden_union.search(line):
            violations.append(f"L{i}: ❌ Forbidden 'Union' detected. Use specific types or NpF64/JaxF64.")
        
        if _forbidden_any.search(line):
            # Allow **kwargs: Any and Any = None (default value)
            if not _allowed_any_kwargs.search(line) and not _allowed_any_default.search(line):
                 violations.append(f"L{i}: ❌ Forbidden 'Any' detected. Use specific types.")

        # 3. Float32 Check
        if _forbidden_float32.search(line):
             violations.append(f"L{i}: ❌ Forbidden 'float32' detected. Use float64 (JaxF64/NpF64).")

        # 4. Raw Array Hint Check
        if _forbidden_raw_hints.search(line):
             violations.append(f"L{i}: ❌ Forbidden raw array hint (ndarray/Array). Use NpF64 or JaxF64.")
        
        # 5. jnp.asarray Check (Only allowed in Mappers)
        if _forbidden_asarray.search(line) and rel not in MAPPERS:
             violations.append(f"L{i}: ❌ Forbidden 'jnp.asarray' outside Mapper. Use 'to_jax_f64' from core.types.")

        # 6. np.asarray Check (Only allowed in Mappers)
        if _forbidden_np_asarray.search(line) and rel not in MAPPERS:
             violations.append(f"L{i}: ❌ Forbidden 'np.asarray' outside Mapper. Use 'to_np_f64' from core.types or ensure input is NpF64.")

    # Boundary Violation
    if has_np and has_jax and rel not in MAPPERS and rel not in CONDITIONAL_JAX_OK:
        violations.append(f"❌ BOUNDARY VIOLATION: Imports BOTH numpy AND jax (not a registered Mapper)")
    
    if violations:
        return [f"\n📄 {rel}:"] + violations
    
    return []


def main():
    violations = []
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        violations.extend(check_file(py_file))
    
    if violations:
        print("=" * 60)
        print("Import & Type Violations Detected")
        print("=" * 60)
        for v in violations:
            print(v)
        print(f"\nTotal Files with Violations: {len([v for v in violations if v.startswith('📄')])}")
        print("\nACTION REQUIRED: Fix strict type violations.")
        sys.exit(1)
    else:
        print("✅ All import boundaries and types are clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
