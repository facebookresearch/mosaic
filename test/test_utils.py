# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from later.unittest import TestCase
from later.unittest.mock import MagicMock, patch
from mosaic.libmosaic.utils.data_utils import MemoryEvent
from mosaic.libmosaic.utils.utils import (
    get_printable_memory_event_set,
    get_printable_stack_trace,
)


class TestUtils(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_get_printable_stack_trace(self) -> None:
        stack_trace = [
            {"name": "function1", "filename": "file1.py", "line": 10},
            {"name": "function2", "filename": "file2.py", "line": 20},
            {"name": "function3", "filename": "file3.py", "line": 30},
        ]
        expected_output = (
            "\nfunction3, file3.py:30"
            "\n  function2, file2.py:20"
            "\n    function1, file1.py:10\n"
        )
        self.assertEqual(
            get_printable_stack_trace(stack_trace),  # pyre-ignore
            expected_output,
        )

    def test_get_printable_stack_trace_empty(self) -> None:
        stack_trace = []
        expected_output = "\n\n"
        self.assertEqual(get_printable_stack_trace(stack_trace), expected_output)

    @patch("mosaic.libmosaic.utils.utils.get_printable_stack_trace")
    def test_get_printable_memory_event_set(
        self, mock_get_printable_stack_trace: MagicMock
    ) -> None:
        memory_event_set = {
            1: MemoryEvent(
                num_call=10,
                mem_size=1024 * 1024 * 1024,
                call_stack=[{"name": "function1", "filename": "file1.py", "line": 10}],
            ),
            2: MemoryEvent(
                num_call=20,
                mem_size=2048 * 1024 * 1024,
                call_stack=[{"name": "function2", "filename": "file2.py", "line": 20}],
            ),
        }
        mock_get_printable_stack_trace.side_effect = [
            "\nfunction1, file1.py:10\n",
            "\nfunction2, file2.py:20\n",
        ]
        expected_output = (
            "\nNum of Calls: 10, Memory Usage: 1.0 GiB"
            "\nfunction1, file1.py:10\n"
            "\nNum of Calls: 20, Memory Usage: 2.0 GiB"
            "\nfunction2, file2.py:20\n"
        )
        self.assertEqual(
            get_printable_memory_event_set(memory_event_set), expected_output
        )

    def test_get_printable_memory_event_set_empty(self) -> None:
        memory_event_set = {}
        expected_output = "\n"
        self.assertEqual(
            get_printable_memory_event_set(memory_event_set), expected_output
        )
