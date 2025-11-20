# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import sys
from typing import Any, Dict, List

import altair as alt
import pandas as pd
from mosaic.libmosaic.utils.data_utils import MemoryUsage


class ProfilerPlotter:
    """
    A class to manage plotting memory usage history
    """

    def __init__(
        self,
        history: dict[int, MemoryUsage],
        profile: str = "categories",
        sampling_rate: int = 1,
        start_idx: int = 0,
        end_idx: int = sys.maxsize,
        preserve_allocation_order: bool = False,
    ) -> None:
        self.history = history
        self.profile = profile
        self.sampling_rate = sampling_rate
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.preserve_allocation_order = preserve_allocation_order

    def _plot_rows(self, rows: List[Dict[str, Any]], filename: str) -> None:
        df = pd.DataFrame(rows)

        # Handle empty DataFrame case - return early without plotting
        if df.empty:
            # Create an empty chart or return without doing anything
            return

        selection = alt.selection_point(fields=["cat"], bind="legend")

        # Get unique categories and sort them
        unique_cats = sorted(df["cat"].unique())

        c = (
            alt.Chart(df)
            .mark_area()
            .encode(
                alt.X(
                    "event_idx:Q",
                    axis=alt.Axis(
                        title=f"Event (Sampling Rate = {self.sampling_rate})"
                    ),
                ),
                alt.Y(
                    "sum:Q",
                    stack="zero",  # Explicitly set stack order to start from zero
                    axis=alt.Axis(title="Memory (GiB)"),
                ),
                alt.Color(
                    "cat:N",
                    sort=unique_cats,  # Explicitly provide the sort order
                    legend=alt.Legend(
                        title="Category",
                        orient="right",
                        labelLimit=400,
                        symbolLimit=0,
                    ),
                ),
                order=alt.Order(
                    "cat:N",
                    sort="ascending",  # Ensure stack order follows category order
                ),
                tooltip=[
                    alt.Tooltip("cat:N", title="Category"),
                    alt.Tooltip("sum:Q", title="Value"),
                ],
                opacity=alt.when(selection)
                .then(alt.value(0.8))
                .otherwise(alt.value(0.2)),
            )
            .add_params(selection)
            .properties(width=800, height=600)
            .configure_axis(grid=True, gridColor="lightgray")
            .configure_view(strokeWidth=0)
        ).interactive()

        c.save(filename)

    def plot_values(self, filename: str, attribute: str) -> None:
        if self.sampling_rate < 1:
            raise ValueError("Sampling rate must be a positive integer")
        rows = []
        time_sorted = sorted(self.history.keys())
        for ii, time in enumerate(time_sorted):
            if ii % self.sampling_rate != 0:
                continue
            if ii < self.start_idx or ii > self.end_idx:
                continue
            value_dict = getattr(self.history[time], attribute)
            for value in value_dict:
                if value_dict[value] / 1024**3 > 0:
                    # Handle different value types for category names
                    if self.profile == "custom":
                        # Custom categories are already strings
                        category_name = str(value)
                    elif isinstance(value, str):
                        category_name = value.replace("Torch-Compiled Region: ", "")
                    elif hasattr(value, "name"):
                        category_name = value.name
                    else:
                        category_name = str(value)

                    rows.append(
                        {
                            "event_idx": ii / self.sampling_rate,
                            "sum": round(
                                value_dict[value] / 1024**3,
                                3,
                            ),
                            "cat": category_name,
                        }
                    )

        self._plot_rows(rows, filename)

    def plot(self, filename: str) -> None:
        # If preserve_allocation_order is enabled and we have temporal data, use temporal plotting
        if self.preserve_allocation_order and self._has_temporal_data():
            self._plot_temporal(filename)
        else:
            self._plot_categorical(filename)

    def _has_temporal_data(self) -> bool:
        for memory_usage in self.history.values():
            if memory_usage.stack_snapshot is not None:
                return True
        return False

    def _plot_categorical(self, filename: str) -> None:
        attribute = ""
        if self.profile == "categories":
            attribute = "per_category_alloc_sum"
        elif self.profile == "annotations":
            attribute = "per_annotation_alloc_sum"
        elif self.profile == "compile_context":
            attribute = "per_compile_context_alloc_sum"
        elif self.profile == "custom":
            attribute = "per_custom_alloc_sum"
        else:
            raise ValueError(f"Unsupported profile type: {self.profile}")

        self.plot_values(filename, attribute)

    def _plot_temporal(self, filename: str) -> None:
        if self.sampling_rate < 1:
            raise ValueError("Sampling rate must be a positive integer")

        rows = []
        time_sorted = sorted(self.history.keys())

        # Track statistics for logging
        max_stack_depth = 0
        category_occurrences: Dict[str, int] = {}
        timestamps_with_snapshots = 0

        # First pass: determine the maximum stack depth to calculate padding
        for time in time_sorted:
            memory_usage = self.history[time]
            stack_snapshot = memory_usage.stack_snapshot
            if stack_snapshot is not None:
                max_stack_depth = max(max_stack_depth, len(stack_snapshot))

        # Calculate padding width based on max stack depth
        # e.g., max_stack_depth=100 needs 3 digits: "000", "001", ..., "099"
        padding_width = len(str(max_stack_depth - 1)) if max_stack_depth > 0 else 1

        for ii, time in enumerate(time_sorted):
            if ii % self.sampling_rate != 0:
                continue
            if ii < self.start_idx or ii > self.end_idx:
                continue

            memory_usage = self.history[time]
            # Use the captured stack snapshot from this timestamp
            stack_snapshot = memory_usage.stack_snapshot
            if stack_snapshot is not None:
                timestamps_with_snapshots += 1

                # Convert stack snapshot to rows
                for stack_idx, (category, total_size) in enumerate(stack_snapshot):
                    size_gib = total_size / 1024**3
                    if size_gib > 0:
                        # Use zero-padded stack index as prefix to maintain temporal order
                        # This ensures alphabetical sorting respects the stack order even with >10 items
                        # e.g., "00_unknown", "01_ncclx", "02_unknown" for stack [unknown, ncclx, unknown]
                        category_with_order = (
                            f"{stack_idx:0{padding_width}d}_{category}"
                        )

                        # Track category occurrences for statistics
                        if category not in category_occurrences:
                            category_occurrences[category] = 0
                        category_occurrences[category] += 1

                        rows.append(
                            {
                                "event_idx": ii / self.sampling_rate,
                                "sum": round(size_gib, 3),
                                "cat": category_with_order,
                            }
                        )

        if rows:
            self._plot_rows(rows, filename)
        else:
            # Fallback to categorical if no temporal data
            self._plot_categorical(filename)
