# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import importlib
import json
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from mosaic.mcp_common import core as _core
else:
    try:
        _core = importlib.import_module("mosaic.mcp_common.core")
    except ModuleNotFoundError:
        # OSS pytest can import this module as top-level `mcp.server` from the
        # repo root (`.../mosaic/mosaic`), where `mosaic.*` is not importable.
        _core = importlib.import_module("mcp_common.core")

analyze_peak_memory: Callable[..., dict[str, Any]] = _core.analyze_peak_memory
analyze_categorical: Callable[..., dict[str, Any]] = _core.analyze_categorical
analyze_annotations: Callable[..., dict[str, Any]] = _core.analyze_annotations
analyze_diff: Callable[..., dict[str, Any]] = _core.analyze_diff


def _json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2)


def peak_memory_analysis(snapshot_path: str, print_stack: bool = True) -> str:
    return _json(analyze_peak_memory(snapshot_path, print_stack))


def categorical_profiling(snapshot_path: str) -> str:
    return _json(analyze_categorical(snapshot_path))


def annotation_analysis(snapshot_path: str, annotation: str | None = None) -> str:
    return _json(analyze_annotations(snapshot_path, annotation))


def memory_diff(snapshot_path_1: str, snapshot_path_2: str) -> str:
    return _json(analyze_diff(snapshot_path_1, snapshot_path_2))


def _build_server() -> Any:
    # Import lazily so module import does not fail in environments that don't
    # have the MCP SDK available during test collection.
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mosaic")
    mcp.tool(
        description=(
            "Analyze peak memory usage from PyTorch memory snapshot. "
            "Returns stack traces of allocations contributing to peak usage."
        )
    )(peak_memory_analysis)
    mcp.tool(
        description=(
            "Classify memory allocations into categories: activation, gradient, "
            "optimizer state, parameter."
        )
    )(categorical_profiling)
    mcp.tool(description="Get memory usage at each annotation stage during training.")(
        annotation_analysis
    )
    mcp.tool(description="Compare two memory snapshots to identify memory imbalances.")(
        memory_diff
    )
    return mcp


def main() -> None:
    _build_server().run(transport="stdio")


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
