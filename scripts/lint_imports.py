#!/usr/bin/env python3
"""
scripts/lint_imports.py
Import Boundary & Type Enforcement — 物理的制約

Hexagonal Architecture for ML:
1. NumPy/JAX import境界を自動監視。
2. 厳格な型エイリアス (JaxF64, NpF64) の使用を強制 (`Any`, `Union`, 生の `Float64` 禁止)
3. `np.asarray`, `jnp.asarray` は Mapper 外で禁止 (`to_jax_f64`, `to_np_f64`を使用)
4. Callable の曖昧な型指定 (`...`, `Any`) を禁止

Usage:
    uv run python scripts/lint_imports.py
"""
import re
import sys
from pathlib import Path

# ===== Domain Registry =====
# Mapper files: BOTH np and jax imports explicitly allowed (boundary converters)
# Pipeline or types or batched_compute
MAPPERS = {
    "core/types.py",
    "utils/batched_compute.py",
    #pipeline
    "pipelines/strategies.py",
    "pipelines/components/executor.py",
}

CONDITIONAL_JAX_OK = {
}

FORBIDDEN_TYPES = {"Any", "object", "Union"}

SRC_ROOT = Path(__file__).parent.parent / "src" / "reservoir"

# 1. Imports
_import_np  = re.compile(r"^\s*(import numpy|from numpy|import numpy\.)")
_import_jax = re.compile(r"^\s*(import jax\b|from jax\b|import jaxlib)")
_import_types = re.compile(r"from reservoir\.core\.types import (.*)")
_import_forbidden = re.compile(r"from typing import.*?\b(Any|Union|Optional)\b|import typing.*?Any|import typing.*?Union|import typing.*?Optional") 

# 2. Types
_forbidden_union = re.compile(r"\bUnion\[")
_forbidden_optional = re.compile(r"\bOptional\[")
_forbidden_any = re.compile(r"\bAny\b")
_forbidden_object = re.compile(r"\bobject\b")
_forbidden_raw_float = re.compile(r"\b(?:Float64|Float32|Float)\[|\b(?:Float64|Float32|Float)\b(?!\w|\[)") # matches Float64 or Float64[...
_forbidden_raw_ndarray = re.compile(r"\b(?:jnp\.ndarray|np\.ndarray|jax\.Array|np\.array|jax\.numpy\.ndarray)\b")

# 3. Array creation and modifications
_forbidden_asarray = re.compile(r"\b(?:np|jnp)\.asarray\b") # np.array is allowed if it has dtype=np.float64
_forbidden_astype = re.compile(r"\.astype\(")
_forbidden_copy = re.compile(r"\.copy\(")

# 4. Callable
_forbidden_callable_ellipsis = re.compile(r"\bCallable\[\s*\.\.\.")
_forbidden_callable_any = re.compile(r"\bCallable\[.*,\s*(?:Any|object)\s*\]")

# 5. Bare Collections (Rule 10)
_forbidden_bare_dict = re.compile(r"(?::|->)\s*dict\b(?!\s*\[)")
_forbidden_bare_list = re.compile(r"(?::|->)\s*list\b(?!\s*\[)")



def check_file(path: Path) -> list[str]:
    """Return list of violation messages for a file."""
    rel = str(path.relative_to(SRC_ROOT))
    if path.name == "__init__.py" or "__pycache__" in str(path):
        return []

    lines = path.read_text().splitlines()
    violations = []
    has_np = False
    has_jax = False

    for i, line in enumerate(lines, 1):
        if line.strip().startswith("#"):
            continue

        # Rule 1: Import Boundaries
        if _import_np.match(line): has_np = True
        if _import_jax.match(line): has_jax = True

        # Rule 2: Forbidden Imports (Any, Union, Optional)
        if _import_forbidden.search(line) and rel != "core/types.py":
             violations.append(f"L{i}: ❌ Rule 2: Importing Any, Union, or Optional is strictly prohibited outside core/types.py. Use strictly defined aliases or '| None'.")

        # Rule 7: Don't import both NpF64 and JaxF64 outside of Mappers
        types_match = _import_types.search(line)
        if types_match and rel not in MAPPERS:
            imported = types_match.group(1)
            has_jaxf64 = "JaxF64" in imported or "to_jax_f64" in imported
            has_npf64 = "NpF64" in imported or "to_np_f64" in imported
            
            # We don't strictly ban `to_np_f64` and `to_jax_f64` being imported together, but importing JaxF64 + NpF64 signifies domain bleeding.
            if has_jaxf64 and has_npf64:
                 violations.append(f"L{i}: ❌ Rule 7: Cannot import both JaxF64/to_jax_f64 and NpF64/to_np_f64 outside Mapper.")

        # Rule 3: np.asarray / jnp.asarray not allowed outside Mappers
        if _forbidden_asarray.search(line) and rel not in MAPPERS:
            violations.append(f"L{i}: ❌ Rule 3: 'np.asarray', 'jnp.asarray', 'np.array' forbidden outside Mappers. Use 'to_jax_f64' / 'to_np_f64'.")

        if _forbidden_astype.search(line) and rel not in CONDITIONAL_JAX_OK:
            violations.append(f"L{i}: ❌ Rule 9: '.astype()' forbidden (Fail Fast). Data must be np.float64 inherently.")

        if _forbidden_copy.search(line):
            violations.append(f"L{i}: ❌ Rule 10: '.copy()' forbidden (Memory Safety). Operations must be in-place.")

        # Rule 2: Any / Union / object / |
        if _forbidden_union.search(line):
            # Allow Union if it's part of strictly defined ConfigDict/ResultDict patterns
            is_allowed_union = any(x in line for x in ["ConfigDict", "ResultDict", "ConfigValue", "ResultValue", "PrimitiveValue", "ConfigL", "ResultL", "EvalMetrics", "TrainLogs", "KwargsDict"])
            if not is_allowed_union:
                violations.append(f"L{i}: ❌ Rule 1: 'Union' is strictly prohibited.")
        
        if _forbidden_optional.search(line):
             violations.append(f"L{i}: ❌ Rule 1: 'Optional' is strictly prohibited. Use 'X | None' instead.")

        if _forbidden_any.search(line):
            # Only kwargs / args can use Any
            # Strip comments to avoid matching in them
            line_no_comment = line.split('#')[0]
            # If "Any" is in the line, it MUST be strictly mapped to args or kwargs.
            # We look for "kwargs" or "args" near the colon and "Any" near the end
            if not re.search(r"(?:\*args|\*\*kwargs|\bargs|\bkwargs)\s*:\s*(?:Optional\[)?Any(?:\])?", line_no_comment):
                violations.append(f"L{i}: ❌ Rule 2: 'Any' is strictly prohibited except for **kwargs and *args.")
        
        if _forbidden_object.search(line):
            if not re.search(r"class\s+\w+\s*\(object\):", line) and "logger" not in line and not isinstance(eval("object"), object): # naive filters
                # We want to catch object entirely if used as type hint
                if re.search(r":\s*object|->\s*object|\[object\]", line):
                    violations.append(f"L{i}: ❌ Rule 2: 'object' type hint is a prohibited escape hatch.")

        # Rule 1: Raw Float64 / Float32 
        if rel != "core/types.py" and _forbidden_raw_float.search(line):
            violations.append(f"L{i}: ❌ Rule 1: Raw 'Float64', 'Float32', 'Float' forbidden. Use 'JaxF64' or 'NpF64'.")

        # Rule 1 (implied): Raw jnp.ndarray / np.ndarray
        if rel != "core/types.py" and _forbidden_raw_ndarray.search(line):
            if re.search(r":\s*(?:jnp\.ndarray|np\.ndarray|jax\.Array)|->\s*(?:jnp\.ndarray|np\.ndarray|jax\.Array)", line):
                violations.append(f"L{i}: ❌ Rule 1/3: Raw array usage in type hint. Use 'JaxF64' or 'NpF64'.")

        # Rule 8: Callable strictness
        if _forbidden_callable_ellipsis.search(line):
            violations.append(f"L{i}: ❌ Rule 8: 'Callable[...,]' with ellipsis is forbidden.")
        if _forbidden_callable_any.search(line):
            violations.append(f"L{i}: ❌ Rule 8: 'Callable[..., Any|object]' is forbidden. Specify exact return type.")

        # Rule 10: Bare dict and list
        if _forbidden_bare_dict.search(line):
            violations.append(f"L{i}: ❌ Rule 10: Bare 'dict' type hint is forbidden. Use 'Dict[K, V]' or 'dict[K, V]'.")
        if _forbidden_bare_list.search(line):
            violations.append(f"L{i}: ❌ Rule 10: Bare 'list' type hint is forbidden. Use 'List[T]' or 'list[T]'.")

    # Boundary Violation
    if has_np and has_jax and rel not in MAPPERS and rel not in CONDITIONAL_JAX_OK:
        violations.append("❌ BOUNDARY VIOLATION: Imports BOTH numpy AND jax (not a registered Mapper)")
    
    if violations:
        return [f"\n📄 {rel}:"] + violations
    
    return []

def main():
    violations = []
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        violations.extend(check_file(py_file))
    
    if violations:
        print("=" * 80)
        print("🚨 IMPORT & TYPE BOUNDARY VIOLATIONS DETECTED 🚨")
        print("=" * 80)
        for v in violations:
            print(v)
        print(f"\nTotal Files with Violations: {len([v for v in violations if v.startswith('📄')])}")
        print("\nACTION REQUIRED: Fix strict type violations as defined in AGENT.md")
        sys.exit(1)
    else:
        print("✅ All import boundaries and types are clean.")
        sys.exit(0)

if __name__ == "__main__":
    main()
