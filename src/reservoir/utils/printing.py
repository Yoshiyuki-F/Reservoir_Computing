"""Topology and shape printing utilities for pipeline summaries."""

from __future__ import annotations

from typing import Any, Dict, Optional


def format_shape(shape: Optional[tuple[int, ...]]) -> str:
    if shape is None:
        return "None"
    if len(shape) == 1:
        return f"[{shape[0]}]"
    return f"[{'x'.join(str(d) for d in shape)}]"


def print_topology(topo_meta: Optional[Dict[str, Any]]) -> None:
    if not topo_meta:
        return

    flow_parts = []
    shapes = topo_meta.get("shapes", {})
    typ = topo_meta.get("type", "").upper()
    details = topo_meta.get("details", {})

    print("=" * 40)
    print(f"Model Architecture: {typ or 'MODEL'}")
    print("=" * 40)

    input_shape = shapes.get("input")
    projected = shapes.get("projected")
    internal = shapes.get("internal")
    feature = shapes.get("feature")
    output = shapes.get("output")

    preprocess = details.get("preprocess") or "None (Pass-through)"
    agg_mode = details.get("agg_mode")
    student_layers = details.get("student_layers")

    print(f"1. Input Data      : { format_shape(input_shape) }")
    print(f"2. Preprocessing   : { preprocess }")

    if typ == "FNN_DISTILLATION": #TODO identifier for PIPELINE.FNN_DISTILLATION
        flow_parts.append(format_shape(input_shape))
        flow_parts.append(format_shape(projected))
        if isinstance(internal, tuple):
            flat_shape = internal
            flow_parts.append(format_shape(flat_shape))
        else:
            flat_shape = None
        hidden_str = "-".join(str(h) for h in (student_layers or [])) or "None"
        print(f"3. Input Projection: {format_shape(input_shape)} -> {format_shape(projected)} (time-distributed)")
        print(f"4. Student Model   : {format_shape(flat_shape)} -> [{hidden_str}] -> {format_shape(feature)}")
        print(f"5. Readout         : {format_shape(feature)} -> {format_shape(output)} outputs")
        flow_parts.append(f"[{hidden_str}]")
        flow_parts.append(format_shape(feature))
        flow_parts.append(format_shape(output))
    elif typ in {"RESERVOIR", "ESN", "CLASSICAL"}:
        flow_parts.append(format_shape(input_shape))
        flow_parts.append(format_shape(projected))
        flow_parts.append(format_shape(internal))
        agg_desc = f"{format_shape(internal)} -> {format_shape(feature)}"
        if agg_mode:
            agg_desc += f" (mode={agg_mode})"
        print(f"3. Reservoir       : {format_shape(projected)} -> {format_shape(internal)} (Recurrent)")
        print(f"4. Aggregation     : {agg_desc}")
        print(f"5. Readout         : {format_shape(feature)} -> {format_shape(output)} outputs")
        flow_parts.append(format_shape(feature))
        flow_parts.append(format_shape(output))
    else:
        flow_parts.append(format_shape(input_shape))
        flow_parts.append(format_shape(internal))
        flow_parts.append(format_shape(feature))
        flow_parts.append(format_shape(output))
        print(f"3. Internal Struct : {format_shape(internal)}")
        print(f"4. Readout         : {format_shape(feature)} -> {format_shape(output)} outputs")

    print("-" * 40)
    flow_str = " -> ".join(str(p) for p in flow_parts if p)
    if flow_str:
        print(f"Tensor Flow     : {flow_str}")
    print("=" * 40)
