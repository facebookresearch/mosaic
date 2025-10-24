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
    ) -> None:
        self.history = history
        self.profile = profile
        self.sampling_rate = sampling_rate
        self.start_idx = start_idx
        self.end_idx = end_idx

    def _plot_rows(self, rows: List[Dict[str, Any]], filename: str) -> None:
        df = pd.DataFrame(rows)
        selection = alt.selection_point(fields=["cat"], bind="legend")
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
                alt.Y("sum:Q", axis=alt.Axis(title="Memory (GiB)")),
                alt.Color(
                    "cat:N",
                    legend=alt.Legend(
                        title="Category",
                        orient="right",  # Change to 'bottom' for horizontal orientation
                        labelLimit=400,  # Increase label limit to avoid truncation
                        symbolLimit=0,  # Show all legend entries
                    ),
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
