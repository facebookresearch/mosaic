# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from mosaic.libmosaic.utils.data_utils import MemoryEvent


# apply diff on memory call stack hash of snapshot_diff to snapshot_base
def find_memory_snapshot_peak_call_stack_diffs(
    snapshot_base: dict[int, MemoryEvent], snapshot_diff: dict[int, MemoryEvent]
) -> dict[int, MemoryEvent]:
    call_stack_diff = set(snapshot_diff.keys()).difference(snapshot_base.keys())
    return dict(
        sorted(
            ((k, snapshot_diff[k]) for k in call_stack_diff),
            key=lambda item: item[1].mem_size,
            reverse=True,
        )
    )
    pass


# apply intersection on memory call stack hash of snapshot_diff and snapshot_base
def find_memory_snapshot_peak_call_stack_intersection(
    snapshot_base: dict[int, MemoryEvent], snapshot_diff: dict[int, MemoryEvent]
) -> dict[int, MemoryEvent]:
    call_stack_diff = set(snapshot_diff.keys()).intersection(snapshot_base.keys())
    return dict(
        sorted(
            ((k, snapshot_diff[k]) for k in call_stack_diff),
            key=lambda item: item[1].mem_size,
            reverse=True,
        )
    )
    pass


def order_call_stack_by_memory_size(
    traces: dict[int, MemoryEvent],
) -> dict[int, MemoryEvent]:
    return dict(sorted(traces.items(), key=lambda item: item[1].mem_size, reverse=True))
    pass
