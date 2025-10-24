# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from typing import Dict, Union

from mosaic.libmosaic.utils.data_utils import Frame, MemoryEvent


def get_printable_stack_trace(
    stack_trace: list[Union[Dict[str, Union[str, int]], Frame]],
) -> str:
    """
    Get a printable string representation of the stack trace.

    Args:
        stack_trace (list[dict[str, Any]]): The stack trace to be printed.

    Returns:
        str: The printable string representation of the stack trace.
    """

    prefix = ""
    stack_trace_parts = []
    for trace in reversed(stack_trace):
        if isinstance(trace, Dict):
            stack_trace_parts.append(
                f"{prefix}{trace['name']}, {trace['filename']}:{trace['line']}"
            )
        elif isinstance(trace, Frame):
            stack_trace_parts.append(
                f"{prefix}{trace.name}, {trace.filename}:{trace.line}"
            )
        prefix += "  "

    return "\n" + "\n".join(stack_trace_parts) + "\n"


def get_printable_memory_event_set(memory_event_set: Dict[int, MemoryEvent]) -> str:
    """
    Get a printable string representation of the memory event set.

    Args:
        memory_event_set (Dict[int, MemoryEvent]): The memory event set to be printed.

    Returns:
        str: The printable string representation of the memory event set.
    """

    GiB_CONVERSION_FACTOR = 1024 * 1024 * 1024
    memory_event_set_parts = []

    for _key, value in memory_event_set.items():
        memory_event_str = (
            f"Num of Calls: {value.num_call}, "
            f"Memory Usage: {value.mem_size / GiB_CONVERSION_FACTOR} GiB"
        )
        memory_event_str += get_printable_stack_trace(value.call_stack)
        memory_event_set_parts.append(memory_event_str)

    return "\n" + "\n".join(memory_event_set_parts)


def get_memory_peak_event_in_json(memory_event_set: Dict[int, MemoryEvent]) -> str:
    """
    Get a json string representation of the memory event set.

    Args:
        memory_event_set (Dict[int, MemoryEvent]): The memory event set to be printed.

    Returns:
        str: The json string representation of the memory event set.
    """

    total_memory = 0
    num_of_unique_call_stacks = 0
    details = []
    for _, event in memory_event_set.items():
        total_memory += event.mem_size
        num_of_unique_call_stacks += 1
        details.append(
            {
                "num_of_calls": event.num_call,
                "memory_usage": event.mem_size,
                "memory_usage_per_call": event.mem_size_per_call,
                "memory_usage_percentage": 0,
                "call_stack": event.call_stack,
            }
        )

    for item in details:
        item["memory_usage_percentage"] = item["memory_usage"] / total_memory * 100

    details.sort(key=lambda x: x["memory_usage"], reverse=True)

    return json.dumps(
        {
            "total_memory": total_memory,
            "num_of_unique_call_stacks": num_of_unique_call_stacks,
            "call_stacks": details,
        },
        indent=4,
    )
