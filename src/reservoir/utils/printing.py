"""
printing.py
Utilities for pretty-printing model topology and status.
"""
from typing import Dict, Any, Tuple, Optional, Iterable


def _fmt_dim(shape: Optional[Tuple[int, ...]]) -> str:
    if shape is None:
        return "?"
    return "[" + "x".join(str(d) for d in shape) + "]"


def _fmt_layers(layers: Optional[Iterable[int]]) -> str:
    if not layers:
        return "None"
    return "-".join(str(int(v)) for v in layers)


def print_topology(meta: Dict[str, Any]) -> None:
    """
    Prints a detailed 6-step topology report:
      1. Input
      2. Preprocessing
      3. Input Projection
      4. Model
      5. Aggregation
      6. Readout
    """
    if not meta:
        return

    shapes = meta.get("shapes", {}) or {}
    details = meta.get("details", {}) or {}

    s_in = shapes.get("input")
    s_pre = shapes.get("preprocessed") or s_in
    s_proj = shapes.get("projected")
    s_internal = shapes.get("internal")
    s_adapter = shapes.get("adapter")
    s_feat = shapes.get("feature")
    s_out = shapes.get("output")

    preprocess_method = details.get("preprocess") or "None"
    agg_mode = details.get("agg_mode") or "None"
    student_layers = _fmt_layers(details.get("student_layers"))

    print(f"=== Model Topology: {meta.get('type', 'UNKNOWN')} ===")
    print(f"1. Input Data      : {_fmt_dim(s_in)}")
    print(f"2. Preprocessing   : {_fmt_dim(s_in)} -> {_fmt_dim(s_pre)} (Method: {preprocess_method})")
    if s_proj is not None:
        print(f"3. Input Projection: {_fmt_dim(s_pre)} -> {_fmt_dim(s_proj)}")
    else:
        print("3. Input Projection: Skipped")

    if s_adapter is not None:
        print(f"   (Adapter)       : {_fmt_dim(s_proj or s_pre)} -> {_fmt_dim(s_adapter)}")

    model_desc = ""
    if student_layers != "None" and s_adapter is not None:
        model_desc = f"{_fmt_dim(s_adapter)} -> [{student_layers}] -> {_fmt_dim(s_internal or s_feat)}"
    elif student_layers != "None":
        model_desc = f"Student Layers: {student_layers}"
    elif s_internal is not None:
        model_desc = f"{_fmt_dim(s_proj or s_pre)} -> {_fmt_dim(s_internal)}"
    else:
        model_desc = "n/a"
    print(f"4. Model           : {model_desc}")

    prev_dim = s_internal or s_proj or s_pre
    if not agg_mode or agg_mode == "None":
        print("5. Aggregation     : Skipped (Architecture implies flat output)")
    else:
        print(f"5. Aggregation     : {_fmt_dim(prev_dim)} -> {_fmt_dim(s_feat)} (Mode: {agg_mode})")

    print(f"6. Readout         : {_fmt_dim(s_feat)} -> {_fmt_dim(s_out)} (Ridge Regression)")
    print("=" * 60)
