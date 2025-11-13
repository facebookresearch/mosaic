# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This script generates a query to check memory usage in ODS for large-scale jobs.
It is used when MemorySnapshot is not available for all the jobs.

"""

import copy
import logging

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from mosaic.libmosaic.utils.data_utils import (
    Allocation,
    BaseMemoryUsage,
    FloatMemoryUsage,
    Frame,
    MemoryEvent,
    MemoryUsage,
    NUM_GPUS_PER_HOST,
    TraceEvent,
)
from mosaic.libmosaic.utils.snapshot_loader import SnapshotLoader
from mosaic.libmosaic.utils.snapshot_utils import order_call_stack_by_memory_size

# Memory viz seems to only use alloc and free_completed events, to calculate memory usage
# so lets do the same here
ALLOC_EVENTS = ["alloc"]  # "segment_alloc", "segment_map"
FREE_EVENTS = ["free_completed"]  # "segment_free", "segment_unmap"


class MemorySnapshot(SnapshotLoader):
    def __init__(self, filename: Optional[str]) -> None:
        super().__init__(filename)

        # TODO: there are a lot of temp variables here, need to clean up
        self.memory_usage_history: dict[int, BaseMemoryUsage] = {}
        self.max_memory_usage: BaseMemoryUsage = FloatMemoryUsage(0.0)
        self.call_stack: dict[object, str] = {}
        self.memory_peak = -1
        self.memory_usage: dict[str, float] = {}
        self.memory_usage_call_stack: dict[int, dict[str, Any]] = {}
        self.call_stack_hash_set: dict[int, MemoryEvent] = {}
        self.max_allocation_info: Allocation = Allocation("", "", 0)

    def stat_memory_usage_range(
        self, device_idx: int = 0, start: int = -1, end: int = -1
    ) -> None:
        self._stat_memory_usage_with_range(device_idx, start, end)

    def analyze_memory_snapshot(
        self,
        opt: str = "memory_peak",
        allocation: str = "",
        action: str = "",
        profile_types: Optional[List[str]] = None,
        custom_rules: Optional[Dict[str, str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        preserve_allocation_order: bool = False,
    ) -> None:
        if not profile_types:
            profile_types = ["categories"]
        if opt == "memory_peak":
            self.memory_usage_call_stack, self.memory_peak, self.max_allocation_info = (
                self._get_memory_usage_peak(
                    allocation_str=allocation,
                    action=action,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            self._stat_call_stack()
            self.call_stack_hash_set = order_call_stack_by_memory_size(
                self.call_stack_hash_set
            )
        elif opt == "memory_usage":
            self._get_memory_usage(allocation, action)
            self._stat_call_stack()
            self.call_stack_hash_set = order_call_stack_by_memory_size(
                self.call_stack_hash_set
            )
        elif opt == "alloc_history":
            self._stat_memory_usage_categorization(
                profile_types=profile_types,
                custom_rules=custom_rules,
                preserve_allocation_order=preserve_allocation_order,
            )
        else:
            logging.error("Invalid option, please check the options")

    # Returns a dict of annotation with key {name}_{stage}(version) to a tuple of the annotation and the memory usage at the start of the annotation stage
    # Pass in a list of substrings to match the annotation name(s) desired to be included in the output. If empty, all annotations will be included.
    def get_memory_usage_by_annotation_stage(
        self, annotation_substring: Set[str]
    ) -> Dict[str, Tuple[Dict[str, Any], float]]:
        version = defaultdict(int)
        trace_events = self._get_pytorch_memory_allocator_event_trace()
        if len(annotation_substring) > 0:
            external_annotation_events = [
                annotation
                for annotation in self.snapshot_data["external_annotations"]
                if any(
                    substring in annotation["name"]
                    for substring in annotation_substring
                )
            ]
        else:
            external_annotation_events = self.snapshot_data.get(
                "external_annotations", []
            )
        snapshot = {}
        annotation_to_memory_usage = {}
        if trace_events:
            active_memory = 0
            trace_event_index = 0
            for annotation in external_annotation_events:
                # Process all trace events that occurred before the current annotation
                while (
                    trace_event_index < len(trace_events)
                    and trace_events[trace_event_index]["time_us"]
                    <= annotation["time_us"]
                ):
                    event = trace_events[trace_event_index]
                    if event["action"] in ALLOC_EVENTS:
                        active_memory += event["size"]
                        snapshot[event["addr"]] = event
                    elif event["action"] in FREE_EVENTS:
                        if event["addr"] in snapshot:
                            snapshot.pop(event["addr"])
                            active_memory -= event["size"]
                    trace_event_index += 1
                name_stage = f"{annotation['name']}_{annotation['stage']}"
                key = (
                    f"{name_stage}({version[f'{name_stage}']})"
                    if version[name_stage] > 0
                    else f"{name_stage}"
                )
                annotation_to_memory_usage[key] = (
                    annotation,
                    active_memory,
                )
                version[f"{name_stage}"] += 1
        return annotation_to_memory_usage

    def _convert_stack_trace_to_str(
        self, stack_trace_dict: list[dict[str, str]]
    ) -> str:
        stack_trace_str = ""
        for item in stack_trace_dict:
            stack_trace_str += f"{item['filename']}: {item['line']}" + "<br>"
        return stack_trace_str
        pass

    def _convert_trace_event_to_str(self, stack_trace_dict: list[Frame]) -> str:
        stack_trace_str = ""
        for item in stack_trace_dict:
            stack_trace_str += f"{item.filename}: {item.line}" + "<br>"
        return stack_trace_str
        pass

    def _stat_memory_usage_with_range(
        self, device_idx: int = 0, start: int = -1, end: int = -1
    ) -> None:
        # currently only considered memory alloc and free_completed events
        # TODO: add more detailed memory events tracking
        #   - allocated memory: alloc, free, free_completed
        #   - reserved memory: segment_alloc, segment_free,  segment_free
        # TODO: if we are using expandable segments, they may start using segment_map instead of segment_alloc, and segment_unmap instead of segment_free
        memory_usage: float = 0.0
        self.memory_usage_history = {}
        self.max_memory_usage: FloatMemoryUsage = FloatMemoryUsage(0.0)
        self.call_stack = {}

        if len(self.snapshot_data.keys()) == 0:
            logging.error("Empty snapshot data, need to load snapshot first.")
            return

        for item in self.snapshot_data["device_traces"][device_idx]:
            if "time_us" not in item.keys():
                continue
            if (
                start < 0
                and end < 0
                or item["time_us"] >= start
                and item["time_us"] <= end
            ):
                if item["action"] == "alloc":
                    memory_usage += item["size"]
                elif item["action"] == "free_completed":
                    memory_usage -= item["size"]

                if "frames" in item and len(item["frames"]) > 0:
                    self.call_stack[item["time_us"]] = self._convert_stack_trace_to_str(
                        item["frames"]
                    )
                else:
                    self.call_stack[item["time_us"]] = ""
                self.memory_usage_history[item["time_us"]] = FloatMemoryUsage(
                    memory_usage
                )
                # record the max memory usage
                self.max_memory_usage = max(
                    self.max_memory_usage, FloatMemoryUsage(memory_usage)
                )
        pass

    # Gets last index of target in stack, returns -1 if not found
    def _get_last_index(self, stack: list[str], target: str) -> int:
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] == target:
                if index != len(stack) - 1:
                    logging.warning(
                        f"annotation {target} overlaps with {stack[-1]}. This may result in allocations being attributed to incorrect annotations!"
                    )
                return index
        logging.warning(
            f"END stage for {target} annotation found without START stage. Skipping..."
        )
        return -1

    def _get_annotation(
        self,
        item: dict[str, Any],
        external_annotation_events: list[dict[str, Any]],
        annotation_stack: list[str],
    ) -> str:
        while (
            external_annotation_events
            and item["time_us"] >= external_annotation_events[0]["time_us"]
        ):
            annotation = external_annotation_events.pop(0)
            if annotation["stage"] == "START":
                annotation_stack.append(annotation["name"])
            elif annotation["stage"] == "END":
                idx = self._get_last_index(annotation_stack, annotation["name"])
                # only pop if we find the corresponding start annotation
                if idx >= 0:
                    annotation_stack.pop(idx)
            else:
                logging.error(
                    f"Invalid annotation stage: {annotation['stage']} for {annotation['name']}"
                )
        return annotation_stack[-1] if annotation_stack else ""

    def _stat_memory_usage_categorization(
        self,
        profile_types: List[str],
        custom_rules: Optional[Dict[str, str]] = None,
        device_idx: int = 0,
        start: int = -1,
        end: int = -1,
        preserve_allocation_order: bool = False,
    ) -> None:
        memory_usage: MemoryUsage = MemoryUsage(
            save_profile=True, track_allocation_order=preserve_allocation_order
        )
        self.memory_usage_history: dict[int, MemoryUsage] = {}
        self.max_memory_usage: MemoryUsage = MemoryUsage()
        self.call_stack: dict[int, str] = {}
        annotation_stack = []
        external_annotation_events = self.snapshot_data.get("external_annotations", [])
        if len(self.snapshot_data.keys()) == 0:
            logging.error("Empty snapshot data, need to load snapshot first.")
            return

        for item in self.snapshot_data["device_traces"][device_idx]:
            if "time_us" not in item.keys():
                continue
            if "annotations" in profile_types:
                annotation = self._get_annotation(
                    item, external_annotation_events, annotation_stack
                )
            else:
                annotation = ""
            # TODO: handle oom case
            if item["action"] == "oom":
                continue
            evt = TraceEvent.from_raw(item, annotation, custom_rules)
            if start < 0 and end < 0 or evt.time_us >= start and evt.time_us <= end:
                memory_usage.update(evt, profile_types)

                current_memory_usage = memory_usage.partial_copy()
                self.memory_usage_history[evt.time_us] = current_memory_usage
                if memory_usage.total_alloc > self.max_memory_usage.total_alloc:
                    self.max_memory_usage = memory_usage.partial_copy()

    def _stat_memory_reserve_with_range(
        self, device_idx: int = 0, start: int = -1, end: int = -1
    ) -> None:
        # currently only considered memory alloc and free_completed events
        # TODO: add more detailed memory events tracking
        #   - allocated memory: alloc, free, free_completed
        #   - reserved memory: segment_alloc, segment_free,  segment_free
        # TODO: if we are using expandable segments, they may start using segment_map instead of segment_alloc, and segment_unmap instead of segment_free
        memory_usage: float = 0.0
        self.memory_usage_history = {}
        self.max_memory_usage: float = 0.0
        self.call_stack = {}

        if len(self.snapshot_data.keys()) == 0:
            logging.error("Empty snapshot data, need to load snapshot first.")
            return

        for item in self.snapshot_data["device_traces"][device_idx]:
            if "time_us" not in item.keys():
                continue
            if (
                start < 0
                and end < 0
                or item["time_us"] >= start
                and item["time_us"] <= end
            ):
                if item["action"] == "segment_alloc":
                    memory_usage += item["size"]
                elif item["action"] == "segment_free":
                    memory_usage -= item["size"]

                if len(item["frames"]) > 0:
                    self.call_stack[item["time_us"]] = (
                        item["frames"][0]["filename"]
                        + ":"
                        + str(item["frames"][0]["line"])
                    )
                else:
                    self.call_stack[item["time_us"]] = ""
                self.memory_usage_history[item["time_us"]] = FloatMemoryUsage(
                    memory_usage
                )
                # record the max memory usage
                self.max_memory_usage = max(
                    self.max_memory_usage, FloatMemoryUsage(memory_usage)
                )
        pass

    def _get_memory_usage_peak(
        self,
        device_idx: int = -1,
        allocation_str: str = "",
        action: str = "",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], int, Allocation]:
        usage = 0
        memory_peak = 0
        snapshot = {}
        max_snapshot = {}
        version = defaultdict(int)
        find_allocation = Allocation.from_str(allocation_str, action)
        (
            hex_addr,
            max_allocation_addr,
            max_allocation_size,
            max_allocation_action,
            max_allocation_time,
        ) = (
            "",
            "",
            0,
            "",
            0,
        )
        trace_events = self._get_pytorch_memory_allocator_event_trace()

        if not trace_events:
            logging.info("No device traces found")
            raise Exception("No device traces found")

        # In first scan, we find the max memory usage to avoid repeatedly deep copy the snapshot
        allocated_addr_set = set()
        for event in trace_events:
            time_us = event["time_us"]

            if end_time and time_us > end_time:
                break

            if "addr" not in event:
                continue

            cur_addr = event["addr"]
            cur_action = event["action"]
            cur_size = event["size"]
            hex_addr = "{:x}".format(event["addr"])

            if cur_action in ALLOC_EVENTS:
                usage += cur_size
                allocated_addr_set.add(cur_addr)
            elif cur_action in FREE_EVENTS:
                if cur_addr in allocated_addr_set:
                    allocated_addr_set.remove(cur_addr)
                    usage -= cur_size

            if start_time and time_us < start_time:
                continue

            memory_peak = max(memory_peak, usage)

            if (
                find_allocation.addr
                and hex_addr == find_allocation.addr
                and version[hex_addr] == find_allocation.version
                and cur_action == find_allocation.action
            ):
                break

            if cur_action in FREE_EVENTS:
                version[hex_addr] += 1

        # In the second scan, we record and return the snapshot at the first max memory uage
        usage = 0
        for event in trace_events:
            time_us = event["time_us"]

            if end_time and time_us > end_time:
                break

            if "addr" not in event:
                continue

            cur_addr = event["addr"]
            cur_action = event["action"]
            cur_size = event["size"]
            hex_addr = "{:x}".format(event["addr"])

            if cur_action in ALLOC_EVENTS:
                usage += cur_size
                snapshot[cur_addr] = event
            elif cur_action in FREE_EVENTS:
                if cur_addr in snapshot:
                    snapshot.pop(cur_addr)
                    usage -= cur_size
            elif cur_action == "oom":
                logging.info(
                    f"OOM detected! (requested {event['size']}, CUDA has {event['device_free']} memory free) - stream({event['stream']})"
                )
                continue

            if start_time and time_us < start_time:
                continue

            if usage == memory_peak:
                max_allocation_addr = hex_addr
                max_allocation_action = cur_action
                max_allocation_size = cur_size
                max_allocation_time = time_us
                max_snapshot = copy.deepcopy(snapshot)
                if len(max_snapshot) != len(snapshot):
                    print("Deep copy failed")
                break

            if (
                find_allocation.addr
                and hex_addr == find_allocation.addr
                and version[hex_addr] == find_allocation.version
                and cur_action == find_allocation.action
            ):
                break

            if cur_action in FREE_EVENTS:
                version[hex_addr] += 1

        logging.info(
            f"Total Peak Memory Usage (Relative to Start): {memory_peak / 1024 / 1024 / 1024} GiB at {max_allocation_addr}_{version[hex_addr]} ({max_allocation_action}) - size {max_allocation_size} bytes at {max_allocation_time} us"
        )
        max_allocation_info = Allocation(
            max_allocation_action, max_allocation_addr, version[hex_addr]
        )

        return max_snapshot, memory_peak, max_allocation_info

    def _get_memory_usage(
        self,
        allocation_str: str = "",
        action: str = "",
        device_idx: int = -1,
    ) -> None:
        usage = 0
        snapshot = {}
        version = defaultdict(int)
        allocation = Allocation.from_str(allocation_str, action)
        trace_events = self._get_pytorch_memory_allocator_event_trace()

        if trace_events:
            for event in trace_events:
                if event["action"] in ALLOC_EVENTS:
                    usage += event["size"]
                    snapshot[event["addr"]] = event
                elif event["action"] in FREE_EVENTS:
                    try:
                        snapshot.pop(event["addr"])
                        usage -= event["size"]
                    except KeyError:
                        pass

                hex_addr = "{:x}".format(event["addr"])
                if (
                    hex_addr == allocation.addr
                    and version[hex_addr] == allocation.version
                    and event["action"] == allocation.action
                ):
                    self.memory_usage[allocation.to_str()] = usage
                    self.memory_usage_call_stack = copy.deepcopy(snapshot)
                    logging.info(
                        f"Memory Usage (Relative to Start) at '{allocation.to_str()}' - {usage / 1024 / 1024 / 1024} GiB"
                    )
                    return
                if event["action"] in FREE_EVENTS:
                    version[hex_addr] += 1
            logging.info("Allocation not found!")
        else:
            logging.info("No device traces found")
        pass

    def _stat_call_stack(self) -> None:
        for stack_call in self.memory_usage_call_stack.values():
            line_str = ""
            if isinstance(stack_call, dict) and "frames" not in stack_call.keys():
                continue
            for line in stack_call["frames"]:
                line_str += str(line["filename"]) + str(line["line"])
            hash_val = hash(line_str)
            if hash_val not in self.call_stack_hash_set.keys():
                self.call_stack_hash_set[hash_val] = MemoryEvent(
                    call_stack=stack_call["frames"],
                    mem_size=stack_call["size"],
                    mem_size_per_call={stack_call["addr"]: stack_call["size"]},
                    num_call=1,
                )
            else:
                self.call_stack_hash_set[hash_val].num_call += 1
                self.call_stack_hash_set[hash_val].mem_size += stack_call["size"]
                self.call_stack_hash_set[hash_val].mem_size_per_call[
                    stack_call["addr"]
                ] = stack_call["size"]

        pass

    def _get_pytorch_memory_allocator_event_trace(
        self, device_idx: int = -1
    ) -> Optional[list[dict[str, Any]]]:
        if not self.snapshot_data:
            logging.error("Empty snapshot data, need to load snapshot first.")
            return

        if device_idx == -1:
            while device_idx < NUM_GPUS_PER_HOST:
                device_idx += 1
                if self.snapshot_data["device_traces"][device_idx]:
                    break

        if device_idx < NUM_GPUS_PER_HOST:
            return self.snapshot_data["device_traces"][device_idx]
        return None
