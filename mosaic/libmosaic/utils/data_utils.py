# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import enum
import logging

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

NUM_GPUS_PER_HOST = 8

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Allocation:
    action: str
    addr: str
    version: int = 0

    def to_str(self) -> str:
        return f"{self.action}: {self.addr}_{self.version}"

    @staticmethod
    def from_str(allocation: str, action: str) -> "Allocation":
        if allocation == "":
            return Allocation(action=action, addr="", version=0)
        addr, version = allocation.split("_")
        return Allocation(action=action, addr=addr, version=int(version))


@dataclass
class Frame:
    name: str
    filename: str
    line: int


class AllocationType(enum.Enum):
    PARAMETER = 0
    ACTIVATION = 1
    BACKWARD = 2
    OPTIMIZER = 3
    NET = 4
    UNKNOWN = 5
    NET_SETUP = 6
    STATS = 7
    FSDP = 8
    CUSTOM = 9

    @classmethod
    def from_frame_stack(cls, frame_stack: List[Frame]) -> "AllocationType":
        has_fsdp = any(
            "fully_sharded_data_parallel.py" in frame.filename for frame in frame_stack
        )
        if has_fsdp:
            return cls.FSDP

        for frame in frame_stack:
            if "forward" in frame.name:
                return cls.ACTIVATION
            if "param" in frame.name:
                return cls.PARAMETER
            if "flatten_params_wrapper" in frame.filename:
                return cls.PARAMETER
            if "build_norm_fn" in frame.name:
                return cls.PARAMETER
            if "__init__" in frame.name and "parallel/layers.py" in frame.filename:
                return cls.PARAMETER
            if "backward" in frame.name:
                return cls.BACKWARD
            if frame.name in ["clip_grad_norm_", "calc_grad_norm"]:
                return cls.BACKWARD
            if frame.name in ["custom_adamw", "_init_group"]:
                return cls.OPTIMIZER
            if frame.name in ["dist_max", "dist_mean", "dist_sum", "_aggregate"]:
                return cls.NET
            if "validate_process_group" in frame.name:
                return cls.NET_SETUP
            if frame.name == "compute_grad_stats":
                return cls.STATS

        return cls.UNKNOWN

    @classmethod
    def from_frame_stack_with_custom(
        cls, frame_stack: List[Frame], custom_rules: Optional[dict[str, str]] = None
    ) -> tuple["AllocationType", Optional[str]]:
        """Returns (allocation_type, custom_category_name)"""
        if custom_rules:
            for category_name, pattern in custom_rules.items():
                if cls._matches_custom_pattern(frame_stack, pattern):
                    return cls.CUSTOM, category_name

        # Fall back to existing logic
        return cls.from_frame_stack(frame_stack), None

    @classmethod
    def _matches_custom_pattern(cls, frame_stack: List[Frame], pattern: str) -> bool:
        """Check if any frame matches the custom pattern using regex"""
        import re

        try:
            compiled_pattern = re.compile(pattern)
            for frame in frame_stack:
                if compiled_pattern.search(frame.name) or compiled_pattern.search(
                    frame.filename
                ):
                    return True
        except re.error:
            # Invalid regex pattern, skip this rule
            logger.warning(f"Invalid regex pattern: {pattern}")
            return False
        return False


@dataclass
class TraceEvent:
    action: str
    addr: int
    size: int
    stream: int
    time_us: int
    classification: AllocationType = AllocationType.UNKNOWN
    custom_category: str = "unknown"
    annotation: str = "None"
    compile_context: str = "None"

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        annotation: str = "None",
        custom_rules: Optional[dict[str, str]] = None,
    ) -> "TraceEvent":
        raw_modified = {**raw}
        if "frames" in raw:
            raw_modified["frames"] = [Frame(**f) for f in raw["frames"]]
        else:
            raw_modified["frames"] = []

        if "compile_context" not in raw:
            raw_modified["compile_context"] = "N/A"

        # Use custom classification if custom rules provided
        classification, custom_category = AllocationType.from_frame_stack_with_custom(
            raw_modified["frames"], custom_rules
        )

        del raw_modified["frames"]
        raw_modified.pop("user_metadata", None)

        return cls(
            **raw_modified,
            classification=classification,
            custom_category=custom_category or "unknown",
            annotation=annotation,
        )

    # Not implemented, just needed for typing
    def __getitem__(self, key: str) -> None:
        return None


@dataclass
class MemoryEvent:
    # Dictionary keys of each call stack element include: "name", "filename", "line"
    call_stack: list[Union[dict[str, Any], Frame]]
    num_call: int
    mem_size: float
    mem_size_per_call: dict[str, float] = field(default_factory=dict)
    alloc_type: Optional[AllocationType] = None


@dataclass
class CategoryStackElement:
    """
    Represents an aggregated segment of memory allocations for a single category.
    Maintains the total size and tracks individual allocations tied to this segment.
    """

    category: str  # Category name (depends on profile type)
    total_size: float  # Total memory size in bytes for this stack element
    allocation_ids: set[int]  # Set of allocation IDs tied to this segment


class CategoryStackOrderTracker:
    """
    Optimized tracker that maintains a stack of aggregated category segments.

    Instead of tracking every individual allocation, this tracker aggregates consecutive
    allocations of the same category into stack elements. This dramatically reduces
    memory usage and improves plotting performance.

    Key properties:
    - When allocations occur for the same category as the top of stack, they're added to that element
    - When allocations occur for a different category, a new stack element is created
    - Stack elements can be removed from anywhere when all their allocations are freed
    - Uses a dictionary to track which stack element each allocation belongs to
    """

    def __init__(self) -> None:
        # Ordered list of category stack elements (maintains temporal ordering)
        self.stack: list[CategoryStackElement] = []

        # Maps address to (stack_index, allocation_id, size, category)
        self.addr_to_info: dict[int, tuple[int, int, float, str]] = {}

        # Counter for assigning unique allocation IDs
        self.allocation_counter: int = 0

    def track_allocation(self, evt: TraceEvent, category: str) -> int:
        """
        Track a new allocation and add it to the appropriate stack element.

        Only adds to the top stack element if it matches the category.
        Otherwise, always creates a new stack element on top.

        Args:
            evt: The trace event for this allocation
            category: The category string for this allocation

        Returns:
            The unique allocation_id assigned to this allocation
        """
        allocation_id = self.allocation_counter
        self.allocation_counter += 1

        # Check if we can add to existing top stack element
        if self.stack and self.stack[-1].category == category:
            # Same category as top of stack - add to existing element
            stack_idx = len(self.stack) - 1
            self.stack[-1].total_size += evt.size
            self.stack[-1].allocation_ids.add(allocation_id)
        else:
            # Different category or empty stack - always create new element
            new_element = CategoryStackElement(
                category=category,
                total_size=evt.size,
                allocation_ids={allocation_id},
            )
            self.stack.append(new_element)
            stack_idx = len(self.stack) - 1

        # Track this allocation's location
        self.addr_to_info[evt.addr] = (stack_idx, allocation_id, evt.size, category)

        return allocation_id

    def track_deallocation(self, addr: int) -> Optional[int]:
        """
        Track a deallocation and update the corresponding stack element.
        Remove the stack element if all its allocations have been freed.
        If removal results in adjacent elements with the same category, merge them.

        Args:
            addr: The memory address being freed

        Returns:
            The allocation_id that was freed, or None if address wasn't tracked
        """
        if addr not in self.addr_to_info:
            return None

        stack_idx, allocation_id, size, category = self.addr_to_info.pop(addr)

        # Update the stack element
        if stack_idx < len(self.stack):
            element = self.stack[stack_idx]
            element.total_size -= size
            element.allocation_ids.discard(allocation_id)

            # If this stack element is now empty, remove it
            if len(element.allocation_ids) == 0:
                self.stack.pop(stack_idx)

                # Check if we should merge adjacent elements with the same category
                # After removing stack_idx, elements at (stack_idx-1) and stack_idx are now adjacent
                # Note: the element now at position stack_idx was originally at stack_idx+1
                if (
                    stack_idx > 0
                    and stack_idx < len(self.stack)
                    and self.stack[stack_idx - 1].category
                    == self.stack[stack_idx].category
                ):
                    # Merge stack_idx into stack_idx-1
                    prev_element = self.stack[stack_idx - 1]
                    curr_element = self.stack[stack_idx]

                    prev_element.total_size += curr_element.total_size
                    prev_element.allocation_ids.update(curr_element.allocation_ids)

                    # Remove the current element (at stack_idx)
                    self.stack.pop(stack_idx)

                    # Update indices in addr_to_info
                    # Remember: old_idx values are from BEFORE the first pop(stack_idx)
                    # - Element at original stack_idx was removed (empty)
                    # - Element at original stack_idx+1 is being merged into stack_idx-1
                    # - Elements at original stack_idx+2+ need to shift down by 2
                    for addr_key in list(self.addr_to_info.keys()):
                        old_idx, alloc_id, alloc_size, alloc_cat = self.addr_to_info[
                            addr_key
                        ]
                        if old_idx == stack_idx + 1:
                            # Move to merged element at stack_idx-1
                            self.addr_to_info[addr_key] = (
                                stack_idx - 1,
                                alloc_id,
                                alloc_size,
                                alloc_cat,
                            )
                        elif old_idx > stack_idx + 1:
                            # Allocations from elements after the merged one
                            # Need to shift down by 2 (one for empty removal, one for merge)
                            self.addr_to_info[addr_key] = (
                                old_idx - 2,
                                alloc_id,
                                alloc_size,
                                alloc_cat,
                            )
                else:
                    # No merge needed, just update indices for elements after the removed one
                    for addr_key in list(self.addr_to_info.keys()):
                        old_idx, alloc_id, alloc_size, alloc_cat = self.addr_to_info[
                            addr_key
                        ]
                        if old_idx > stack_idx:
                            self.addr_to_info[addr_key] = (
                                old_idx - 1,
                                alloc_id,
                                alloc_size,
                                alloc_cat,
                            )

        return allocation_id

    def get_current_stack_snapshot(self) -> list[tuple[str, float]]:
        """
        Get the current state of the stack as a list of (category, size).

        Returns:
            List of tuples representing the current stack state
        """
        return [(element.category, element.total_size) for element in self.stack]


class BaseMemoryUsage:
    def __init__(self) -> None:
        self.active_alloc_events: dict[int, TraceEvent] = {}
        self.active_reserve_events: dict[int, TraceEvent] = {}
        self.per_category_alloc_sum: dict[AllocationType, float] = defaultdict(float)
        self.per_annotation_alloc_sum: dict[str, float] = defaultdict(float)
        self.per_compile_context_alloc_sum: dict[str, float] = defaultdict(float)

    def update(self, evt: TraceEvent, profile_types: List[str]) -> None:
        pass

    def __copy__(self) -> "BaseMemoryUsage":
        return self

    def __gt__(self, other: "BaseMemoryUsage") -> bool:
        return False

    @property
    def total_alloc(self) -> float:
        return 0.0

    @property
    def total_reserved(self) -> float:
        return 0.0

    def __str__(self) -> str:
        return ""

    def __truediv__(self, other: Union["BaseMemoryUsage", float, int]) -> float:
        return 0.0


class FloatMemoryUsage(BaseMemoryUsage):
    def __init__(self, float_val: float) -> None:
        super().__init__()
        self.value = float_val

    def __gt__(self, other: "BaseMemoryUsage") -> bool:
        if isinstance(other, FloatMemoryUsage):
            # Perform action specific to FloatMemoryUsage
            return self.value > other.value
        else:
            return False

    def __str__(self) -> str:
        return f"{self.value}"

    def __truediv__(self, other: Union["BaseMemoryUsage", float, int]) -> float:
        if isinstance(other, int):
            return self.value / other
        if isinstance(other, FloatMemoryUsage):
            return self.value / other.value
        else:
            return 0


class MemoryUsage(BaseMemoryUsage):
    def __init__(
        self, save_profile: bool = False, track_allocation_order: bool = False
    ) -> None:
        super().__init__()

        # Flag to determine if we should save the categorization profile or not
        self.save_profile: bool = save_profile

        # Custom profile tracking
        self.per_custom_alloc_sum: dict[str, float] = defaultdict(float)

        # Temporal tracking - use optimized CategoryStackOrderTracker
        self.track_allocation_order = track_allocation_order
        self.category_stack_tracker: Optional[CategoryStackOrderTracker] = None
        # Snapshot of stack state at this point in time (for history entries)
        self.stack_snapshot: Optional[list[tuple[str, float]]] = None
        if track_allocation_order:
            self.category_stack_tracker = CategoryStackOrderTracker()

        # These values only used when no profile saved (needed to lower memory usage)
        self.alloc_usage = 0
        self.reserved_usage = 0

        # Performance tracking
        self._update_count = 0

        # Track profile types for category determination
        self._profile_types: List[str] = []

    def _get_category_for_event(self, evt: TraceEvent, profile_types: List[str]) -> str:
        # TODO: Enable concurrently categorizing events for multiple profile types
        if "categories" in profile_types:
            return evt.classification.name
        elif "annotations" in profile_types:
            return evt.annotation
        elif "compile_context" in profile_types:
            return evt.compile_context
        elif "custom" in profile_types and hasattr(evt, "custom_category"):
            return evt.custom_category
        else:
            return "unknown"

    def _update_alloc_with_profile(
        self, evt: TraceEvent, profile_types: List[str]
    ) -> None:
        if evt.addr not in self.active_alloc_events:
            self.active_alloc_events[evt.addr] = evt
            if "categories" in profile_types:
                self.per_category_alloc_sum[evt.classification] += evt.size
            if "annotations" in profile_types:
                self.per_annotation_alloc_sum[evt.annotation] += evt.size
            if "compile_context" in profile_types:
                self.per_compile_context_alloc_sum[evt.compile_context] += evt.size
            if "custom" in profile_types and hasattr(evt, "custom_category"):
                self.per_custom_alloc_sum[evt.custom_category] += evt.size

            # Track in category stack if enabled
            category = self._get_category_for_event(evt, profile_types)
            if self.category_stack_tracker is not None:
                self.category_stack_tracker.track_allocation(evt, category)
        else:
            logger.warning(f"Double alloc at addr: {evt.addr}")

    def _update_free_with_profile(
        self, evt: TraceEvent, profile_types: List[str]
    ) -> None:
        if evt.addr in self.active_alloc_events:
            alloc_evt = self.active_alloc_events[evt.addr]
            if "categories" in profile_types:
                self.per_category_alloc_sum[alloc_evt.classification] -= evt.size
            if "annotations" in profile_types:
                if evt.addr in self.active_alloc_events:
                    annotation = self.active_alloc_events[evt.addr].annotation
                else:
                    annotation = evt.annotation
                self.per_annotation_alloc_sum[annotation] -= evt.size
            if "compile_context" in profile_types:
                if evt.addr in self.active_alloc_events:
                    compile_context = self.active_alloc_events[evt.addr].compile_context
                else:
                    compile_context = evt.compile_context
                self.per_compile_context_alloc_sum[compile_context] -= evt.size
            if "custom" in profile_types and hasattr(alloc_evt, "custom_category"):
                self.per_custom_alloc_sum[alloc_evt.custom_category] -= evt.size

            # Track deallocation in category stack if enabled
            if self.category_stack_tracker is not None:
                self.category_stack_tracker.track_deallocation(evt.addr)

            if evt.addr in self.active_alloc_events:
                del self.active_alloc_events[evt.addr]
        else:
            logger.warning("Free for unallocated address")

    def _update_with_profile(self, evt: TraceEvent, profile_types: List[str]) -> None:
        if evt.action == "alloc":
            self._update_alloc_with_profile(evt, profile_types)
        elif evt.action == "free_completed":
            self._update_free_with_profile(evt, profile_types)
        elif evt.action == "segment_alloc":
            if evt.addr not in self.active_reserve_events:
                self.active_reserve_events[evt.addr] = evt
        elif evt.action == "segment_free":
            if evt.addr in self.active_reserve_events:
                del self.active_reserve_events[evt.addr]

    def _update_no_profile(self, evt: TraceEvent) -> None:
        if evt.action == "alloc":
            self.alloc_usage += evt.size
        elif evt.action == "free_completed":
            self.alloc_usage -= evt.size
        elif evt.action == "segment_alloc":
            self.reserved_usage += evt.size
        elif evt.action == "segment_free":
            self.reserved_usage -= evt.size

    def update(self, evt: TraceEvent, profile_types: List[str]) -> None:
        if self.save_profile:
            self._update_with_profile(evt, profile_types)
        else:
            self._update_no_profile(evt)

    def __copy__(self) -> "MemoryUsage":
        u = MemoryUsage()
        u.active_alloc_events = self.active_alloc_events.copy()
        u.active_reserve_events = self.active_reserve_events.copy()
        u.per_category_alloc_sum = self.per_category_alloc_sum.copy()
        u.per_annotation_alloc_sum = self.per_annotation_alloc_sum.copy()
        u.per_compile_context_alloc_sum = self.per_compile_context_alloc_sum.copy()
        u.per_custom_alloc_sum = self.per_custom_alloc_sum.copy()
        u.alloc_usage = self.alloc_usage
        u.reserved_usage = self.reserved_usage
        u.save_profile = self.save_profile
        return u

    def __gt__(self, other: "BaseMemoryUsage") -> bool:
        if isinstance(other, MemoryUsage):
            # Perform action specific to MemoryUsage
            return self.total_alloc > other.total_alloc
        else:
            return False

    @property
    def total_alloc(self) -> float:
        if self.save_profile:
            if len(self.per_category_alloc_sum) != 0:
                return sum(self.per_category_alloc_sum.values())
            if len(self.per_annotation_alloc_sum) != 0:
                return sum(self.per_annotation_alloc_sum.values())
            if len(self.per_compile_context_alloc_sum) != 0:
                return sum(self.per_compile_context_alloc_sum.values())
            if len(self.per_custom_alloc_sum) != 0:
                return sum(self.per_custom_alloc_sum.values())
            return sum(e.size for e in self.active_alloc_events.values())
        else:
            return self.alloc_usage

    @property
    def total_reserved(self) -> float:
        if self.save_profile:
            return sum(e.size for e in self.active_reserve_events.values())
        else:
            return self.reserved_usage

    def _convert_bytes(self, size: float) -> str:
        if size >= 1024**3:
            return f"{round(size/1024**3, 2)}GiB"
        elif size >= 1024**2:
            return f"{round(size/1024**2, 2)}MB"
        elif size >= 1024:
            return f"{round(size/1024, 2)}KB"
        else:
            return f"{size}B"

    def partial_copy(self) -> "MemoryUsage":
        new_instance = MemoryUsage()
        new_instance.per_category_alloc_sum = self.per_category_alloc_sum.copy()
        new_instance.per_annotation_alloc_sum = self.per_annotation_alloc_sum.copy()
        new_instance.per_compile_context_alloc_sum = (
            self.per_compile_context_alloc_sum.copy()
        )
        new_instance.per_custom_alloc_sum = self.per_custom_alloc_sum.copy()
        new_instance.save_profile = self.save_profile
        new_instance.track_allocation_order = self.track_allocation_order

        # For temporal tracking, capture the current stack snapshot
        # Don't share the tracker reference - instead store a snapshot
        new_instance.category_stack_tracker = None  # Don't share the reference

        # Capture the current stack state as a snapshot for this timestamp
        if self.category_stack_tracker is not None:
            new_instance.stack_snapshot = (
                self.category_stack_tracker.get_current_stack_snapshot()
            )
        else:
            new_instance.stack_snapshot = None

        return new_instance

    def __str__(self) -> str:
        lines = []
        lines.append(f"Total Allocated: {round(self.total_alloc / 1024**3, 2)}GiB")
        lines.append("Category Profile:")
        for cat, size in self.per_category_alloc_sum.items():
            lines.append(f"{cat}: {self._convert_bytes(size)}")
        lines.append("Annotation Profile:")
        for annotation, size in self.per_annotation_alloc_sum.items():
            lines.append(f"{annotation}: {self._convert_bytes(size)}")
        lines.append("Compile Context Profile:")
        for compile_context, size in self.per_compile_context_alloc_sum.items():
            lines.append(f"{compile_context}: {self._convert_bytes(size)}")
        lines.append("Custom Profile:")
        for custom_category, size in self.per_custom_alloc_sum.items():
            lines.append(f"{custom_category}: {self._convert_bytes(size)}")
        return "\n".join(lines)
