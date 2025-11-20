# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
from unittest.mock import patch

from later.unittest import TestCase
from mosaic.libmosaic.utils.data_utils import AllocationType, MemoryUsage
from mosaic.libmosaic.utils.profiler_plotter import ProfilerPlotter


class TestProfilerPlotter(TestCase):
    def _create_memory_usage_with_categories(
        self, categories: dict[AllocationType, float]
    ) -> MemoryUsage:
        memory_usage = MemoryUsage(save_profile=True)
        memory_usage.per_category_alloc_sum = categories
        return memory_usage

    def _create_memory_usage_with_custom(
        self, custom_categories: dict[str, float]
    ) -> MemoryUsage:
        memory_usage = MemoryUsage(save_profile=True)
        memory_usage.per_custom_alloc_sum = custom_categories
        return memory_usage

    def _create_memory_usage_with_temporal(
        self, stack_snapshot: list[tuple[str, float]]
    ) -> MemoryUsage:
        memory_usage = MemoryUsage(save_profile=True, track_allocation_order=True)
        memory_usage.stack_snapshot = stack_snapshot
        return memory_usage

    def test_plot_categorical_categories_with_sampling(self) -> None:
        # Create history with 10 timestamps, each with two categories
        history: dict[int, MemoryUsage] = {
            i: self._create_memory_usage_with_categories(
                {
                    AllocationType.ACTIVATION: (1 + 0.5 * i) * 1024**3,
                    AllocationType.PARAMETER: (2 + 0.5 * i) * 1024**3,
                }
            )
            for i in range(10)
        }
        # Use sampling_rate=3 to test sampling, and profile="categories" to test categories
        plotter = ProfilerPlotter(history, profile="categories", sampling_rate=3)

        with patch.object(plotter, "_plot_rows") as mock_plot_rows:
            with tempfile.NamedTemporaryFile(suffix=".html") as f:
                plotter.plot(f.name)

        mock_plot_rows.assert_called_once()
        rows = mock_plot_rows.call_args[0][0]
        # There should be 4 sampled event indices (0, 3, 6, 9), each with 2 categories
        event_indices = {row["event_idx"] for row in rows}
        self.assertEqual(len(event_indices), 4)
        categories = {row["cat"] for row in rows}
        self.assertIn("ACTIVATION", categories)
        self.assertIn("PARAMETER", categories)
        self.assertEqual(len(rows), 8)  # 4 sampled timestamps × 2 categories

    def test_plot_categorical_custom_profile(self) -> None:
        history: dict[int, MemoryUsage] = {
            0: self._create_memory_usage_with_custom(
                {"fsdp_kernel": 1 * 1024**3, "nccl_kernel": 0.5 * 1024**3}
            ),
            1: self._create_memory_usage_with_custom(
                {"fsdp_kernel": 1.2 * 1024**3, "nccl_kernel": 0.7 * 1024**3}
            ),
        }
        plotter = ProfilerPlotter(history, profile="custom")

        with patch.object(plotter, "_plot_rows") as mock_plot_rows:
            with tempfile.NamedTemporaryFile(suffix=".html") as f:
                plotter.plot(f.name)

        mock_plot_rows.assert_called_once()
        rows = mock_plot_rows.call_args[0][0]
        categories = {row["cat"] for row in rows}
        self.assertIn("fsdp_kernel", categories)
        self.assertIn("nccl_kernel", categories)

    def test_plot_temporal_with_allocation_order(self) -> None:
        history: dict[int, MemoryUsage] = {
            0: self._create_memory_usage_with_temporal(
                [("activation", 1 * 1024**3), ("parameter", 0.5 * 1024**3)]
            ),
            1: self._create_memory_usage_with_temporal(
                [
                    ("activation", 1.2 * 1024**3),
                    ("parameter", 0.7 * 1024**3),
                    ("optimizer", 0.3 * 1024**3),
                ]
            ),
        }
        plotter = ProfilerPlotter(history, preserve_allocation_order=True)

        with patch.object(plotter, "_plot_rows") as mock_plot_rows:
            with tempfile.NamedTemporaryFile(suffix=".html") as f:
                plotter.plot(f.name)

        mock_plot_rows.assert_called_once()
        rows = mock_plot_rows.call_args[0][0]
        # Should have 5 total rows: 2 categories at t0 + 3 categories at t1
        self.assertEqual(len(rows), 5)
        # Verify categories have stack index prefix for temporal ordering
        categories = {row["cat"] for row in rows}
        self.assertTrue(any("_activation" in cat for cat in categories))
        self.assertTrue(any("_parameter" in cat for cat in categories))
