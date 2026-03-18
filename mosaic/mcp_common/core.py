# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import contextlib
import io
import re
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


@contextlib.contextmanager
def _suppress_stdout() -> Any:
    """Prevent helper CLI utilities from writing to MCP stdout channel."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _serialize_obj(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _serialize_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_obj(v) for v in value]
    return value


_PATH_PREFIX_RE: re.Pattern[str] = re.compile(
    r".*/(?=(?:torch|torchrec|hammer|minimal_viable_ai|mosaic|aiplatform)/)"
)

_TORCH_INTERNAL_NAMES: frozenset[str] = frozenset({"_call_impl", "_wrapped_call_impl"})


def _simplify_frame(frame: dict[str, Any]) -> dict[str, Any]:
    filename = frame.get("filename", "")
    if filename:
        frame = {**frame, "filename": _PATH_PREFIX_RE.sub("", filename)}
    return frame


def _serialize_peak_events(
    call_stack_hash_set: dict[int, Any],
    collapse_torch_internals: bool = True,
) -> list[dict[str, Any]]:
    events = []
    for callstack_hash, event in call_stack_hash_set.items():
        event_dict = _serialize_obj(event)
        raw_stack = event_dict.get("call_stack", [])
        stack = []
        for frame in raw_stack:
            frame = _simplify_frame(frame)
            if collapse_torch_internals and frame.get("name") in _TORCH_INTERNAL_NAMES:
                continue
            stack.append(frame)
        events.append(
            {
                "callstack_hash": callstack_hash,
                "num_calls": event_dict.get("num_call"),
                "memory_bytes": event_dict.get("mem_size"),
                "memory_gib": _bytes_to_gib(float(event_dict.get("mem_size", 0.0))),
                "memory_bytes_per_call": event_dict.get("mem_size_per_call", {}),
                "allocation_type": event_dict.get("alloc_type"),
                "call_stack": stack,
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


def analyze_peak_memory(
    snapshot_path: str, print_stack: bool = True, top_n: int = 0
) -> dict[str, Any]:
    """Analyze peak memory usage and return stack traces contributing to peak."""
    try:
        with _suppress_stdout():
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
        events = _serialize_peak_events(memory_snapshot.call_stack_hash_set)
        total_count = len(events)
        events.sort(key=lambda e: e.get("memory_bytes", 0), reverse=True)
        if top_n > 0:
            events = events[:top_n]
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
                "peak_event_count": total_count,
                "returned_event_count": len(events),
            },
            "events": events,
        }
    except Exception as exc:
        return _error_payload("peak_memory_analysis", snapshot_path, exc)


def analyze_categorical(snapshot_path: str) -> dict[str, Any]:
    """Analyze memory usage by allocation categories."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=True) as output_file:
            with _suppress_stdout():
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
        with _suppress_stdout():
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


def analyze_diff(
    snapshot_path_1: str, snapshot_path_2: str, top_n: int = 0
) -> dict[str, Any]:
    """Compute bidirectional memory peak call-stack diff between two snapshots."""
    try:
        with _suppress_stdout():
            added_raw = get_memory_usage_diff(
                snapshot_base=snapshot_path_1,
                snapshot_diff=snapshot_path_2,
                paste=False,
            )
            removed_raw = get_memory_usage_diff(
                snapshot_base=snapshot_path_2,
                snapshot_diff=snapshot_path_1,
                paste=False,
            )

        added = _serialize_peak_events(added_raw)
        removed = _serialize_peak_events(removed_raw)

        added_total = sum(e.get("memory_bytes", 0) for e in added)
        removed_total = sum(e.get("memory_bytes", 0) for e in removed)

        added.sort(key=lambda e: e.get("memory_bytes", 0), reverse=True)
        removed.sort(key=lambda e: e.get("memory_bytes", 0), reverse=True)
        if top_n > 0:
            added = added[:top_n]
            removed = removed[:top_n]

        return {
            "status": "ok",
            "tool": "memory_diff",
            "snapshot_path_1": snapshot_path_1,
            "snapshot_path_2": snapshot_path_2,
            "summary": {
                "added_count": len(added_raw),
                "removed_count": len(removed_raw),
                "added_total_bytes": added_total,
                "added_total_gib": _bytes_to_gib(float(added_total)),
                "removed_total_bytes": removed_total,
                "removed_total_gib": _bytes_to_gib(float(removed_total)),
                "net_delta_bytes": added_total - removed_total,
                "net_delta_gib": _bytes_to_gib(float(added_total - removed_total)),
            },
            "added": added,
            "removed": removed,
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


def compare_snapshots(
    snapshot_path_1: str, snapshot_path_2: str, top_n: int = 10
) -> dict[str, Any]:
    """All-in-one comparison: peak memory, categories, and bidirectional diff."""
    try:
        with _suppress_stdout():
            peak_1 = get_memory_usage_peak(
                snapshot=snapshot_path_1,
                trace="",
                allocation="",
                action="alloc",
                paste=False,
                print_stack=False,
                upload_result=False,
            )
            peak_2 = get_memory_usage_peak(
                snapshot=snapshot_path_2,
                trace="",
                allocation="",
                action="alloc",
                paste=False,
                print_stack=False,
                upload_result=False,
            )

        snap_1 = peak_1.memory_snapshot
        snap_2 = peak_2.memory_snapshot
        peak_1_bytes = float(snap_1.dynamic_memory_peak + snap_1.static_memory)
        peak_2_bytes = float(snap_2.dynamic_memory_peak + snap_2.static_memory)

        # Categorical profiling
        with _suppress_stdout():
            with tempfile.NamedTemporaryFile(suffix=".html", delete=True) as f1:
                cat_1 = get_memory_profile(
                    snapshot=snapshot_path_1, out_path=f1.name, profile="categories"
                )
            with tempfile.NamedTemporaryFile(suffix=".html", delete=True) as f2:
                cat_2 = get_memory_profile(
                    snapshot=snapshot_path_2, out_path=f2.name, profile="categories"
                )

        cat_1_usage = {
            str(k): v
            for k, v in cat_1.memory_snapshot.max_memory_usage.per_category_alloc_sum.items()
        }
        cat_2_usage = {
            str(k): v
            for k, v in cat_2.memory_snapshot.max_memory_usage.per_category_alloc_sum.items()
        }
        all_categories = sorted(set(cat_1_usage) | set(cat_2_usage))
        categories = {}
        for cat in all_categories:
            v1 = float(cat_1_usage.get(cat, 0))
            v2 = float(cat_2_usage.get(cat, 0))
            categories[cat] = {
                "snapshot_1_gib": _bytes_to_gib(v1),
                "snapshot_2_gib": _bytes_to_gib(v2),
                "delta_gib": _bytes_to_gib(v2 - v1),
            }

        # Bidirectional diff
        with _suppress_stdout():
            added_raw = get_memory_usage_diff(
                snapshot_base=snapshot_path_1,
                snapshot_diff=snapshot_path_2,
                paste=False,
            )
            removed_raw = get_memory_usage_diff(
                snapshot_base=snapshot_path_2,
                snapshot_diff=snapshot_path_1,
                paste=False,
            )

        added = _serialize_peak_events(added_raw)
        removed = _serialize_peak_events(removed_raw)
        added_total = sum(e.get("memory_bytes", 0) for e in added)
        removed_total = sum(e.get("memory_bytes", 0) for e in removed)

        added.sort(key=lambda e: e.get("memory_bytes", 0), reverse=True)
        removed.sort(key=lambda e: e.get("memory_bytes", 0), reverse=True)
        if top_n > 0:
            added = added[:top_n]
            removed = removed[:top_n]

        return {
            "status": "ok",
            "tool": "compare_snapshots",
            "snapshot_path_1": snapshot_path_1,
            "snapshot_path_2": snapshot_path_2,
            "peak_memory": {
                "snapshot_1_gib": _bytes_to_gib(peak_1_bytes),
                "snapshot_2_gib": _bytes_to_gib(peak_2_bytes),
                "delta_gib": _bytes_to_gib(peak_2_bytes - peak_1_bytes),
            },
            "categories": categories,
            "diff": {
                "added_count": len(added_raw),
                "removed_count": len(removed_raw),
                "added_total_gib": _bytes_to_gib(float(added_total)),
                "removed_total_gib": _bytes_to_gib(float(removed_total)),
                "net_delta_gib": _bytes_to_gib(float(added_total - removed_total)),
                "top_added": added,
                "top_removed": removed,
            },
        }
    except Exception as exc:
        return {
            "status": "error",
            "tool": "compare_snapshots",
            "snapshot_path_1": snapshot_path_1,
            "snapshot_path_2": snapshot_path_2,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
