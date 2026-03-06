# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import tempfile
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from mosaic.cmd.entry_point import (
    get_memory_profile,
    get_memory_usage_by_annotation_stage,
    get_memory_usage_diff,
    get_memory_usage_peak,
)

_GIB = float(1024**3)


def _bytes_to_gib(num_bytes: float) -> float:
    return num_bytes / _GIB


def _serialize_obj(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _serialize_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_obj(v) for v in value]
    return value


def _serialize_peak_events(call_stack_hash_set: dict[int, Any]) -> list[dict[str, Any]]:
    events = []
    for callstack_hash, event in call_stack_hash_set.items():
        event_dict = _serialize_obj(event)
        events.append(
            {
                "callstack_hash": callstack_hash,
                "num_calls": event_dict.get("num_call"),
                "memory_bytes": event_dict.get("mem_size"),
                "memory_gib": _bytes_to_gib(float(event_dict.get("mem_size", 0.0))),
                "memory_bytes_per_call": event_dict.get("mem_size_per_call", {}),
                "allocation_type": event_dict.get("alloc_type"),
                "call_stack": event_dict.get("call_stack", []),
            }
        )
    return events


def _error_payload(
    tool_name: str, snapshot_path: str, exc: Exception
) -> dict[str, Any]:
    return {
        "status": "error",
        "tool": tool_name,
        "snapshot_path": snapshot_path,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def analyze_peak_memory(snapshot_path: str, print_stack: bool = True) -> dict[str, Any]:
    """Analyze peak memory usage and return stack traces contributing to peak."""
    try:
        memory_abstract = get_memory_usage_peak(
            snapshot=snapshot_path,
            trace="",
            allocation="",
            action="alloc",
            paste=False,
            print_stack=print_stack,
            upload_result=False,
        )

        memory_snapshot = memory_abstract.memory_snapshot
        return {
            "status": "ok",
            "tool": "peak_memory_analysis",
            "snapshot_path": snapshot_path,
            "summary": {
                "dynamic_peak_bytes": memory_snapshot.dynamic_memory_peak,
                "dynamic_peak_gib": _bytes_to_gib(
                    float(memory_snapshot.dynamic_memory_peak)
                ),
                "static_memory_bytes": memory_snapshot.static_memory,
                "static_memory_gib": _bytes_to_gib(
                    float(memory_snapshot.static_memory)
                ),
                "overall_peak_bytes": (
                    memory_snapshot.dynamic_memory_peak + memory_snapshot.static_memory
                ),
                "overall_peak_gib": _bytes_to_gib(
                    float(
                        memory_snapshot.dynamic_memory_peak
                        + memory_snapshot.static_memory
                    )
                ),
                "peak_event_count": len(memory_snapshot.call_stack_hash_set),
            },
            "events": _serialize_peak_events(memory_snapshot.call_stack_hash_set),
        }
    except Exception as exc:
        return _error_payload("peak_memory_analysis", snapshot_path, exc)


def analyze_categorical(snapshot_path: str) -> dict[str, Any]:
    """Analyze memory usage by allocation categories."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=True) as output_file:
            memory_abstract = get_memory_profile(
                snapshot=snapshot_path,
                out_path=output_file.name,
                profile="categories",
            )

        peak_usage = memory_abstract.memory_snapshot.max_memory_usage
        category_usage = {
            str(category): bytes_used
            for category, bytes_used in peak_usage.per_category_alloc_sum.items()
        }

        return {
            "status": "ok",
            "tool": "categorical_profiling",
            "snapshot_path": snapshot_path,
            "summary": {
                "peak_total_allocated_bytes": peak_usage.total_alloc,
                "peak_total_allocated_gib": _bytes_to_gib(
                    float(peak_usage.total_alloc)
                ),
                "category_count": len(category_usage),
            },
            "categories": {
                category: {
                    "memory_bytes": value,
                    "memory_gib": _bytes_to_gib(float(value)),
                }
                for category, value in category_usage.items()
            },
        }
    except Exception as exc:
        return _error_payload("categorical_profiling", snapshot_path, exc)


def analyze_annotations(
    snapshot_path: str,
    annotation: Optional[str] = None,
) -> dict[str, Any]:
    """Analyze memory usage by annotation stage, optionally filtered by annotation."""
    try:
        annotation_filter: tuple[str, ...] = (annotation,) if annotation else ()
        usage_by_annotation = get_memory_usage_by_annotation_stage(
            snapshot=snapshot_path,
            annotation=annotation_filter,
            paste=False,
        )

        stages = []
        for stage_name, (annotation_data, memory_bytes) in usage_by_annotation.items():
            stages.append(
                {
                    "stage": stage_name,
                    "annotation": annotation_data,
                    "memory_bytes": memory_bytes,
                    "memory_gib": _bytes_to_gib(float(memory_bytes)),
                }
            )

        return {
            "status": "ok",
            "tool": "annotation_analysis",
            "snapshot_path": snapshot_path,
            "annotation_filter": annotation,
            "stage_count": len(stages),
            "stages": stages,
        }
    except Exception as exc:
        return _error_payload("annotation_analysis", snapshot_path, exc)


def analyze_diff(snapshot_path_1: str, snapshot_path_2: str) -> dict[str, Any]:
    """Compute memory peak call-stack diff between two snapshots."""
    try:
        diff = get_memory_usage_diff(
            snapshot_base=snapshot_path_1,
            snapshot_diff=snapshot_path_2,
            paste=False,
        )

        return {
            "status": "ok",
            "tool": "memory_diff",
            "snapshot_path_1": snapshot_path_1,
            "snapshot_path_2": snapshot_path_2,
            "diff_event_count": len(diff),
            "diff_events": _serialize_peak_events(diff),
        }
    except Exception as exc:
        return {
            "status": "error",
            "tool": "memory_diff",
            "snapshot_path_1": snapshot_path_1,
            "snapshot_path_2": snapshot_path_2,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
