# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import mock_open

import pandas as pd

from later.unittest import TestCase
from later.unittest.mock import MagicMock, patch

from mosaic.libmosaic.analyzer.abstract_syntax_analyzer import AbstractSyntaxAnalyzer
from mosaic.libmosaic.analyzer.memory_snapshot import MemorySnapshot
from mosaic.libmosaic.utils.data_utils import MemoryEvent


class TestAbstractSyntaxAnalyzer(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model_path = "path/to/model.py"

        self.memory_snapshot = MagicMock(spec=MemorySnapshot)
        self.memory_snapshot.call_stack_hash_set = {
            1: MagicMock(
                spec=MemoryEvent,
                call_stack=[
                    {"name": "model_function1", "filename": "model.py", "line": 7}
                ],
                num_call=1,
                mem_size=0.1,
            ),
            2: MagicMock(
                spec=MemoryEvent,
                call_stack=[
                    {"name": "function2", "filename": "non_model.py", "line": 0},
                    {"name": "function2", "filename": "test.py", "line": 0},
                ],
                num_call=2,
                mem_size=0.1,
            ),
            3: MagicMock(
                spec=MemoryEvent,
                call_stack=[
                    {"name": "--", "filename": "test.py", "line": 0},
                    {"name": "model_function2", "filename": "model.py", "line": 16},
                ],
                num_call=5,
                mem_size=0.1,
            ),
        }

        self.test_file_contents = """
class ModelClass1:
    def __init__(self):
        pass

    def model_function1(self):
        x = 5
        y = 10
        result = x + y
        return result


class ModelClass2:
    def __init__(self):
        pass

    def model_function2(self):
        data = [1, 2, 3, 4, 5]
        for i in range(len(data)):
            print(data[i])


def standalone_function1():
    pass


def standalone_function2():
    a = 20
    b = 30
    c = a* b
    return  c
        """
        with patch("builtins.open", new=mock_open(read_data=self.test_file_contents)):
            self.classifier = AbstractSyntaxAnalyzer(
                memory_snapshot=self.memory_snapshot,
                file_path=self.model_path,
                allocation="",
                action="alloc",
            )

    def tearDown(self) -> None:
        super().tearDown()

    def test_init_file_name_parsing(self) -> None:
        self.assertEqual(self.classifier.file_name, "model.py")

    def test_memory_snapshot_analsis(self) -> None:
        self.memory_snapshot.analyze_memory_snapshot.assert_called_once_with(
            opt="memory_peak", allocation="", action="alloc"
        )

    def test_parse_classes_and_functions(self) -> None:
        """Test that AST parser correctly identifies all classes and functions."""
        # Test classes are found
        self.assertIn("ModelClass1", self.classifier.classes)
        self.assertIn("ModelClass2", self.classifier.classes)

        # Test class methods are found
        self.assertIn(
            "model_function1", self.classifier.classes["ModelClass1"]["functions"]
        )
        self.assertIn(
            "model_function2", self.classifier.classes["ModelClass2"]["functions"]
        )

        # Test standalone functions are found
        self.assertIn("standalone_function1", self.classifier.functions)
        self.assertIn("standalone_function2", self.classifier.functions)

    def test_ast_parser_sets_class_end_lines(self) -> None:
        """Test that AST parser automatically sets correct end lines for classes."""
        for class_info in self.classifier.classes.values():
            self.assertIsNotNone(class_info["end_line"])
            self.assertGreater(class_info["end_line"], class_info["start_line"])

    def test_ast_parser_sets_function_end_lines(self) -> None:
        """Test that AST parser automatically sets correct end lines for functions."""
        # Test standalone functions
        for func_info in self.classifier.functions.values():
            self.assertIsNotNone(func_info["end_line"])
            self.assertGreater(func_info["end_line"], func_info["start_line"])

        # Test class methods
        for class_info in self.classifier.classes.values():
            for func_info in class_info["functions"].values():
                self.assertIsNotNone(func_info["end_line"])
                self.assertGreater(func_info["end_line"], func_info["start_line"])

    def test_process_memory_events(self) -> None:
        memory_events_df = self.classifier._process_memory_events()
        self.assertIsInstance(memory_events_df, pd.DataFrame)
        self.assertEqual(
            len(memory_events_df), 3
        )  # Check that there are two rows in the DataFrame (specified in setup function)
        expected_columns = [
            "line_numbers",
            "line_content",
            "class_names",
            "function_names",
            "stack_trace",
            "memory_utilization",
        ]
        self.assertEqual(list(memory_events_df.columns), expected_columns)

    def test_aggregate_memory_events(self) -> None:
        aggregated_df = self.classifier._aggregate_memory_events(
            ["line_content", "line_numbers"]
        )
        self.assertIsInstance(aggregated_df, pd.DataFrame)
        expected_columns = [
            "line_content",
            "line_numbers",
            "occurrences",
            "total_memory_utilization_gb",
        ]
        self.assertEqual(list(aggregated_df.columns), expected_columns)
        self.assertEqual(len(aggregated_df), 2)

    def test_aggregate_without_occurrences(self) -> None:
        aggregated_df = self.classifier._aggregate_without_occurrences()
        self.assertIsInstance(aggregated_df, pd.DataFrame)
        expected_columns = ["stack_trace", "total_memory_utilization", "occurrences"]
        self.assertEqual(list(aggregated_df.columns), expected_columns)
        self.assertGreaterEqual(
            len(aggregated_df), 1
        )  # Check that the DataFrame is not empty

    def test_display_parsed_model_info(self) -> None:
        with patch("builtins.print") as mock_print:
            self.classifier.display_parsed_model_info()
            mock_print.assert_called()

    def test_get_printable_dataframes_multiple_dfs(self) -> None:
        dfs = [
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            pd.DataFrame({"C": [5, 6], "D": [7, 8]}),
        ]
        titles = ["Test DataFrame 1", "Test DataFrame 2"]
        printable_df = self.classifier.get_printable_dataframes(dfs, titles)
        self.assertIsInstance(printable_df, str)
        self.assertIn("Test DataFrame 1", printable_df)
        self.assertIn("Test DataFrame 2", printable_df)
        self.assertIn("A", printable_df)
        self.assertIn("8", printable_df)

    def test_get_printable_classification_verbose(self) -> None:
        # Mock the default dataframes
        self.classifier.with_occurence_memory_event_aggregation_by_lines = pd.DataFrame(
            {"A": [1, 2], "B": [3, 4]}
        )
        self.classifier.with_occurence_memory_event_aggregation_by_class_and_function = pd.DataFrame(
            {"C": [5, 6], "D": [7, 8]}
        )
        self.classifier.with_occurence_memory_event_aggregation_by_class = pd.DataFrame(
            {"E": [9, 10], "F": [11, 12]}
        )
        self.classifier.with_occurence_memory_event_aggregation_per_stack_trace = (
            pd.DataFrame({"G": [13, 14], "H": [15, 16]})
        )
        self.classifier.without_occurence_memory_event_aggregation = pd.DataFrame(
            {"I": [17, 18], "J": [19, 20]}
        )
        printable_info = self.classifier.get_printable_classification_info(
            [], [], verbose=True
        )
        self.assertIsInstance(printable_info, str)
        self.assertIn(
            "with_occurence_memory_event_aggregation_per_line_content", printable_info
        )
        self.assertIn(
            "with_occurence_memory_event_aggregation_by_class_and_function",
            printable_info,
        )
        self.assertIn(
            "with_occurence_memory_event_aggregation_by_class", printable_info
        )
        self.assertIn(
            "with_occurence_memory_event_aggregation_per_stack_trace", printable_info
        )
        self.assertIn("without_occurence_memory_event_aggregation", printable_info)

    def test_sum_memory_utilization(self) -> None:
        self.assertEqual(
            self.classifier._sum_memory_utilization(
                self.classifier.memory_events_df,
                7,  # line number of model_function1
            ),
            0.1,  # memory utilization of model_function1
        )

    def test_get_augmented_model_file_view(self) -> None:
        augmented_model_file_view = self.classifier.get_augmented_model_file_view()
        for line_num, (input_line, output_line) in enumerate(
            zip(
                self.test_file_contents.splitlines(),
                augmented_model_file_view.splitlines(),
            )
        ):
            self.assertIn(input_line, output_line)
            self.assertIn("|", output_line)
            if line_num == 7 - 1:
                self.assertIn("GiB", output_line)

    def test_compare_with_occurence_memory_event_aggregation_by_lines(self) -> None:
        memory_snapshot2 = MagicMock(spec=MemorySnapshot)
        memory_snapshot2.call_stack_hash_set = {
            1: MagicMock(
                spec=MemoryEvent,
                call_stack=[
                    {"name": "model_function1", "filename": "model.py", "line": 7}
                ],
                num_call=2,
                mem_size=0.2,
            ),
        }
        with patch("builtins.open", new=mock_open(read_data=self.test_file_contents)):
            classifier2 = AbstractSyntaxAnalyzer(
                memory_snapshot=memory_snapshot2,
                file_path=self.model_path,
            )
        # Call the compare method and check the output
        output = (
            self.classifier.compare_with_occurence_memory_event_aggregation_by_lines(
                classifier2
            )
        )
        self.assertIn("occurrences_df1", output)
        self.assertIn("occurrences_df2", output)
        self.assertIn("total_memory_utilization_gb_df1", output)
        self.assertIn("total_memory_utilization_gb_df2", output)
        self.assertIn("occurrences_diff", output)
        self.assertIn("total_memory_utilization_gb_diff", output)
        # Check the number of rows in the output table
        rows = output.split("\n")
        model_file_referencing_memory_events = 2
        self.assertEqual(
            len([row for row in rows if "|" in row]),
            model_file_referencing_memory_events + 2,  # +2 for header
        )
        # Check the values of occurrences diff and total_memory_utilization_gb_diff
        for row in rows:
            # Test Memory Utilization Different:
            # 0.2 - 0.1 = 0.1 -> 0.1 / 1024 / 1024 / 1024 = 9.31323e-11
            if "x = 5" in row:
                self.assertIn("9.31323e-11", row)
            elif "ModelClass2" in row:
                self.assertIn(
                    "1",
                    row.split("|")[11],  # 11th column is the occurrences_diff
                )

    def test_content_mode_produces_same_results_as_file_mode(self) -> None:
        """Test that both input modes produce identical analysis results."""
        # Create classifier using file path mode
        with patch("builtins.open", new=mock_open(read_data=self.test_file_contents)):
            classifier_path = AbstractSyntaxAnalyzer(
                memory_snapshot=self.memory_snapshot,
                file_path=self.model_path,
                allocation="",
                action="alloc",
            )

        # Create classifier using content mode
        classifier_content = AbstractSyntaxAnalyzer(
            memory_snapshot=self.memory_snapshot,
            file_name="model.py",
            file_content=self.test_file_contents,
            allocation="",
            action="alloc",
        )

        # Compare classes
        self.assertEqual(
            set(classifier_path.classes.keys()),
            set(classifier_content.classes.keys()),
        )

        # Compare functions
        self.assertEqual(
            set(classifier_path.functions.keys()),
            set(classifier_content.functions.keys()),
        )

        # Compare memory events dataframe
        self.assertEqual(
            len(classifier_path.memory_events_df),
            len(classifier_content.memory_events_df),
        )
        self.assertEqual(
            list(classifier_path.memory_events_df.columns),
            list(classifier_content.memory_events_df.columns),
        )
