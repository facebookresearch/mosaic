# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from mosaic.libmosaic.utils.data_utils import (
    AllocationType,
    CategoryStackOrderTracker,
    Frame,
    MemoryUsage,
    TraceEvent,
)


class TestAllocationType(unittest.TestCase):
    def test_from_frame_stack_fsdp(self) -> None:
        frames = [
            Frame(
                name="forward", filename="/path/fully_sharded_data_parallel.py", line=10
            )
        ]

        result = AllocationType.from_frame_stack(frames)
        self.assertEqual(result, AllocationType.FSDP)

    def test_from_frame_stack_with_custom_match(self) -> None:
        frames = [Frame(name="nccl_init", filename="/communication/nccl.py", line=5)]
        custom_rules = {"ncclx": "nccl.*"}
        result_type, result_category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )
        self.assertEqual(result_type, AllocationType.CUSTOM)
        self.assertEqual(result_category, "ncclx")


class TestTraceEvent(unittest.TestCase):
    def test_from_raw_with_frames_and_compile_context(self) -> None:
        raw_data = {
            "action": "alloc",
            "addr": 12345,
            "size": 1024,
            "stream": 0,
            "time_us": 1000,
            "frames": [{"name": "forward", "filename": "/model/net.py", "line": 10}],
            "compile_context": "torch.compile",
        }

        result = TraceEvent.from_raw(raw_data, annotation="test_annotation")

        self.assertEqual(result.action, "alloc")
        self.assertEqual(result.addr, 12345)
        self.assertEqual(result.size, 1024)
        self.assertEqual(result.annotation, "test_annotation")
        self.assertEqual(result.classification, AllocationType.ACTIVATION)
        self.assertEqual(result.compile_context, "torch.compile")


class TestCategoryStackOrderTracker(unittest.TestCase):
    def test_track_allocation(self) -> None:
        tracker = CategoryStackOrderTracker()
        evt1 = TraceEvent(action="alloc", addr=100, size=1024, stream=0, time_us=1000)
        evt2 = TraceEvent(action="alloc", addr=200, size=2048, stream=0, time_us=2000)
        evt3 = TraceEvent(action="alloc", addr=200, size=2048, stream=0, time_us=2000)
        alloc_id1 = tracker.track_allocation(evt1, "category_a")
        alloc_id2 = tracker.track_allocation(evt2, "category_a")
        alloc_id3 = tracker.track_allocation(evt3, "category_b")

        self.assertEqual(len(tracker.stack), 2)
        # Assert: adds to existing stack element
        self.assertEqual(tracker.stack[0].total_size, 3072)
        self.assertEqual(len(tracker.stack[0].allocation_ids), 2)
        self.assertIn(alloc_id1, tracker.stack[0].allocation_ids)
        self.assertIn(alloc_id2, tracker.stack[0].allocation_ids)
        # Assert: creates new stack element
        self.assertIn(alloc_id3, tracker.stack[1].allocation_ids)

    def test_track_deallocation_removes_element(self) -> None:
        tracker = CategoryStackOrderTracker()
        evt = TraceEvent(action="alloc", addr=100, size=1024, stream=0, time_us=1000)
        tracker.track_allocation(evt, "category_a")

        result = tracker.track_deallocation(100)

        self.assertIsNotNone(result)
        self.assertEqual(len(tracker.stack), 0)

    def test_track_deallocation_partial_element(self) -> None:
        tracker = CategoryStackOrderTracker()
        evt1 = TraceEvent(action="alloc", addr=100, size=1024, stream=0, time_us=1000)
        evt2 = TraceEvent(action="alloc", addr=200, size=2048, stream=0, time_us=2000)
        tracker.track_allocation(evt1, "category_a")
        tracker.track_allocation(evt2, "category_a")

        result = tracker.track_deallocation(100)

        self.assertIsNotNone(result)
        self.assertEqual(len(tracker.stack), 1)
        self.assertEqual(tracker.stack[0].total_size, 2048)
        self.assertEqual(len(tracker.stack[0].allocation_ids), 1)

    def test_track_deallocation_merges_adjacent_same_category(self) -> None:
        # Create tracker with pattern [A, B, A]
        tracker = CategoryStackOrderTracker()
        evt1 = TraceEvent(action="alloc", addr=100, size=1024, stream=0, time_us=1000)
        evt2 = TraceEvent(action="alloc", addr=200, size=2048, stream=0, time_us=2000)
        evt3 = TraceEvent(action="alloc", addr=300, size=512, stream=0, time_us=3000)
        tracker.track_allocation(evt1, "category_a")
        tracker.track_allocation(evt2, "category_b")
        tracker.track_allocation(evt3, "category_a")

        # Deallocate B, which should merge adjacent A elements
        tracker.track_deallocation(200)

        # Confirming B is removed and adjacent A's merged
        self.assertEqual(len(tracker.stack), 1)
        self.assertEqual(tracker.stack[0].category, "category_a")
        self.assertEqual(tracker.stack[0].total_size, 1536)

    def test_get_current_stack_snapshot(self) -> None:
        tracker = CategoryStackOrderTracker()
        evt1 = TraceEvent(action="alloc", addr=100, size=1024, stream=0, time_us=1000)
        evt2 = TraceEvent(action="alloc", addr=200, size=2048, stream=0, time_us=2000)
        evt3 = TraceEvent(action="alloc", addr=300, size=512, stream=0, time_us=3000)
        tracker.track_allocation(evt1, "category_a")
        tracker.track_allocation(evt2, "category_b")
        tracker.track_allocation(evt3, "category_c")

        snapshot = tracker.get_current_stack_snapshot()

        self.assertEqual(len(snapshot), 3)
        self.assertEqual(snapshot[0], ("category_a", 1024))
        self.assertEqual(snapshot[1], ("category_b", 2048))
        self.assertEqual(snapshot[2], ("category_c", 512))


class TestMemoryUsage(unittest.TestCase):
    def test_update_alloc_with_profile(self) -> None:
        memory_usage = MemoryUsage(save_profile=True)
        evt = TraceEvent(
            action="alloc",
            addr=100,
            size=1024,
            stream=0,
            time_us=1000,
            classification=AllocationType.ACTIVATION,
        )

        memory_usage.update(evt, ["categories"])

        self.assertEqual(
            memory_usage.per_category_alloc_sum[AllocationType.ACTIVATION], 1024
        )
        self.assertIn(100, memory_usage.active_alloc_events)

    def test_update_free_with_profile(self) -> None:
        memory_usage = MemoryUsage(save_profile=True)
        alloc_evt = TraceEvent(
            action="alloc",
            addr=100,
            size=1024,
            stream=0,
            time_us=1000,
            classification=AllocationType.PARAMETER,
        )
        free_evt = TraceEvent(
            action="free_completed",
            addr=100,
            size=1024,
            stream=0,
            time_us=2000,
        )
        memory_usage.update(alloc_evt, ["categories"])
        memory_usage.update(free_evt, ["categories"])

        self.assertEqual(
            memory_usage.per_category_alloc_sum[AllocationType.PARAMETER], 0
        )
        self.assertNotIn(100, memory_usage.active_alloc_events)

    def test_update_with_annotations(self) -> None:
        memory_usage = MemoryUsage(save_profile=True)
        evt = TraceEvent(
            action="alloc",
            addr=200,
            size=2048,
            stream=0,
            time_us=1000,
            annotation="forward_pass",
        )
        memory_usage.update(evt, ["annotations"])
        self.assertEqual(memory_usage.per_annotation_alloc_sum["forward_pass"], 2048)

    def test_update_with_custom_rules(self) -> None:
        memory_usage = MemoryUsage(save_profile=True)
        evt = TraceEvent(
            action="alloc",
            addr=300,
            size=4096,
            stream=0,
            time_us=1000,
            classification=AllocationType.CUSTOM,
            custom_category="nccl",
        )
        memory_usage.update(evt, ["custom"])
        self.assertEqual(memory_usage.per_custom_alloc_sum["nccl"], 4096)

    def test_total_alloc_with_categories(self) -> None:
        memory_usage = MemoryUsage(save_profile=True)
        evt1 = TraceEvent(
            action="alloc",
            addr=100,
            size=1024,
            stream=0,
            time_us=1000,
            classification=AllocationType.ACTIVATION,
        )
        evt2 = TraceEvent(
            action="alloc",
            addr=200,
            size=2048,
            stream=0,
            time_us=2000,
            classification=AllocationType.PARAMETER,
        )
        memory_usage.update(evt1, ["categories"])
        memory_usage.update(evt2, ["categories"])
        total = memory_usage.total_alloc
        self.assertEqual(total, 3072)

    def test_partial_copy(self) -> None:
        memory_usage = MemoryUsage(save_profile=True)
        evt = TraceEvent(
            action="alloc",
            addr=100,
            size=1024,
            stream=0,
            time_us=1000,
            classification=AllocationType.ACTIVATION,
        )
        memory_usage.update(evt, ["categories"])

        copy = memory_usage.partial_copy()

        self.assertEqual(copy.per_category_alloc_sum[AllocationType.ACTIVATION], 1024)
        self.assertTrue(copy.save_profile)
