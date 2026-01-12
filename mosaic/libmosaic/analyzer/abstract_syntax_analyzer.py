# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import ast
import logging
import os
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Union

import pandas as pd
from mosaic.libmosaic.analyzer.memory_snapshot import MemorySnapshot
from mosaic.libmosaic.utils.data_utils import Frame, MemoryEvent
from tabulate import tabulate

logging_level: str = os.environ.get("LOGGING_LEVEL", "INFO")
logger: logging.Logger = logging.getLogger()


@dataclass
class CodeContext:
    """
    Class to represent additional code context for a given line in the model file.
    In addition to the line number and content, this class also stores the correlated
    class and function information.
    """

    line_number: int
    line_content: str
    class_name: Optional[str] = None
    function_name: Optional[str] = None


class ASTCodeParser:
    """
    AST-based parser that handles file reading and creates a line-to-context mapping.
    Single source of truth for all file operations and parsing.

    Supports two input modes:
    1. File path mode: Provide model_path to read from disk
    2. Content mode: Provide source_content directly (for snapshot-based analysis)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        source_content: Optional[str] = None,
        file_name: Optional[str] = None,
    ):
        if model_path is None and source_content is None:
            raise ValueError("Either model_path or source_content must be provided")
        if model_path is not None and source_content is not None:
            raise ValueError(
                "Only one of model_path or source_content should be provided"
            )

        self.model_path = model_path
        self.file_name = (
            file_name
            if file_name
            else (model_path.split("/")[-1] if model_path else None)
        )

        if source_content is not None:
            self.source_code = source_content
        else:
            self.source_code = self._read_file()

        self.lines = self.source_code.splitlines()
        self.line_to_context: Dict[int, CodeContext] = {}
        self._parse()

    def _read_file(self) -> str:
        """Read the model file and return its contents as a string."""
        if not self.model_path:
            raise ValueError(
                "model_path is required when source_content is not provided"
            )
        try:
            with open(self.model_path, "r") as file:
                return file.read()
        except FileNotFoundError:
            logging.warning(
                f"Path to model definition not found: {self.model_path}, please check the path. An example path is: /data/users/{os.environ['USER']}/fbsource/fbcode/aps_models/ads/launchers/fm/models/experimental/joint_arch_exploration_cmf_model.py"
            )
            raise FileNotFoundError()

    def _parse(self):
        """Parse the source code and build line-to-context mapping using AST."""
        tree = ast.parse(self.source_code)

        # Parse top-level definitions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self._process_class(node)
            elif isinstance(node, ast.FunctionDef):
                self._process_function(node, parent_class=None)

    def _process_class(self, class_node: ast.ClassDef):
        """Process a class and its methods."""
        class_name = class_node.name
        start_line = class_node.lineno  # AST uses 1-based indexing
        end_line = class_node.end_lineno if class_node.end_lineno else len(self.lines)

        # Mark class lines (functions will override with more specific context)
        for line_num in range(start_line, end_line + 1):
            if line_num <= len(self.lines):
                line_content = self.lines[
                    line_num - 1
                ].strip()  # Convert to 0-based for array access
                self.line_to_context[line_num] = CodeContext(
                    line_number=line_num,
                    line_content=line_content,
                    class_name=class_name,
                    function_name=None,
                )

        # Process methods within the class (these will override class-only context)
        for method_node in class_node.body:
            if isinstance(method_node, ast.FunctionDef):
                self._process_function(method_node, parent_class=class_name)

    def _process_function(
        self, func_node: ast.FunctionDef, parent_class: Optional[str]
    ):
        """Process a function (standalone or method)."""
        func_name = func_node.name
        start_line = func_node.lineno  # AST uses 1-based indexing
        end_line = func_node.end_lineno if func_node.end_lineno else len(self.lines)

        # Mark function lines with more specific context
        for line_num in range(start_line, end_line + 1):
            if line_num <= len(self.lines):
                line_content = self.lines[
                    line_num - 1
                ].strip()  # Convert to 0-based for array access
                self.line_to_context[line_num] = CodeContext(
                    line_number=line_num,
                    line_content=line_content,
                    class_name=parent_class,
                    function_name=func_name,
                )

    def get_context_for_line(self, line_number: int) -> Optional[CodeContext]:
        """Get class and function context for a given line number (1-based)."""
        return self.line_to_context.get(line_number)  # Direct 1-based lookup

    def get_classes(self) -> Dict[str, Dict]:
        """Derive class information from line-to-context mapping on demand."""
        classes = {}
        for context in self.line_to_context.values():
            if context.class_name and context.class_name not in classes:
                class_lines = [
                    ctx
                    for ctx in self.line_to_context.values()
                    if ctx.class_name == context.class_name
                ]
                start_line = min(ctx.line_number for ctx in class_lines)  # Keep 1-based
                end_line = max(ctx.line_number for ctx in class_lines)  # Keep 1-based

                # Get functions within this class
                functions = {}
                for ctx in class_lines:
                    if ctx.function_name and ctx.function_name not in functions:
                        func_lines = [
                            c
                            for c in class_lines
                            if c.function_name == ctx.function_name
                        ]
                        func_start = min(
                            c.line_number for c in func_lines
                        )  # Keep 1-based
                        func_end = max(
                            c.line_number for c in func_lines
                        )  # Keep 1-based
                        functions[ctx.function_name] = {
                            "name": ctx.function_name,
                            "start_line": func_start,
                            "end_line": func_end,
                        }

                classes[context.class_name] = {
                    "name": context.class_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "functions": functions,
                }
        return classes

    def get_functions(self) -> Dict[str, Dict]:
        """Derive standalone function information from line-to-context mapping on demand."""
        functions = {}
        for context in self.line_to_context.values():
            # Only standalone functions (not class methods)
            if context.function_name and not context.class_name:
                if context.function_name not in functions:
                    func_lines = [
                        ctx
                        for ctx in self.line_to_context.values()
                        if ctx.function_name == context.function_name
                        and not ctx.class_name
                    ]
                    start_line = min(
                        ctx.line_number for ctx in func_lines
                    )  # Keep 1-based
                    end_line = max(
                        ctx.line_number for ctx in func_lines
                    )  # Keep 1-based
                    functions[context.function_name] = {
                        "name": context.function_name,
                        "start_line": start_line,
                        "end_line": end_line,
                    }
        return functions


class AbstractSyntaxAnalyzer:
    """
    Class to classify model components based on memory utilization.

    Supports two input modes:
    1. File path mode: Provide file_path to read from disk (requires same revision as snapshot)
    2. Content mode: Provide file_name + file_content for snapshot-based analysis
    """

    def __init__(
        self,
        memory_snapshot: MemorySnapshot,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_content: Optional[str] = None,
        allocation: str = "",
        action: str = "alloc",
    ) -> None:
        # Validate input parameters
        if file_path is None and (file_name is None or file_content is None):
            raise ValueError(
                "Either file_path must be provided, or both file_name and file_content must be provided"
            )
        if file_path is not None and (
            file_name is not None or file_content is not None
        ):
            raise ValueError(
                "Cannot provide both file_path and file_name/file_content. Choose one input mode."
            )

        # Set file_path and file_name based on input mode
        if file_path is not None:
            self.file_path = file_path
            self.file_name = file_path.split("/")[-1]
        else:
            self.file_path = None
            self.file_name = file_name

        self.memory_snapshot = memory_snapshot
        self.memory_snapshot.analyze_memory_snapshot(
            opt="memory_peak", allocation=allocation, action=action
        )

        # Initialize AST parser based on input mode
        if file_path is not None:
            # Mode 1: File path mode
            self.ast_parser = ASTCodeParser(model_path=file_path)
        else:
            # Mode 2: Content mode
            self.ast_parser = ASTCodeParser(
                source_content=file_content, file_name=file_name
            )

        # Expose AST parser's derived data structures on-demand (no redundant storage)
        self.classes = self.ast_parser.get_classes()
        self.functions = self.ast_parser.get_functions()

        # Step 2: For each memory event, find the corresponding class and function based on the latest instance of the model file in the stack trace.
        #    If no instance of the model file is found, we just store the stack trace.
        #    If an instance of the model file is found, we store the specific line contents, class, function, stack trace and memory utilization.
        self.memory_events_df: pd.DataFrame = self._process_memory_events()

        # Step 3: Aggregate stack traces unrelated to the model.
        self.without_occurence_memory_event_aggregation = (
            self._aggregate_without_occurrences()
        )
        # Step 4: Aggregate stack traces related to the model by line number.
        self.with_occurence_memory_event_aggregation_by_lines = (
            self._aggregate_memory_events(
                groupby_columns=[
                    "line_content",
                    "line_numbers",
                    "class_names",
                    "function_names",
                ]
            )
        )

        # Step 5: Aggregate stack traces related to the model by class and function.
        self.with_occurence_memory_event_aggregation_by_class_and_function = (
            self._aggregate_memory_events(
                groupby_columns=["class_names", "function_names"]
            )
        )
        # Step 6: Aggregate the contents of the memory events by class.
        self.with_occurence_memory_event_aggregation_by_class = (
            self._aggregate_memory_events(groupby_columns=["class_names"])
        )
        # Step 7: Aggregate the contents of the memory events per stack trace.
        self.with_occurence_memory_event_aggregation_per_stack_trace = (
            self._aggregate_memory_events(
                groupby_columns=[
                    "stack_trace",
                    "line_numbers",
                    "class_names",
                    "function_names",
                ],
                sort_by_column="occurrences",
            )
        )

        # Step 8: Simplify per line aggregation by just taking the first valus.
        self.simplified_with_occurence_memory_event_aggregation_by_lines = (
            self.with_occurence_memory_event_aggregation_by_lines[
                ["line_content", "occurrences", "total_memory_utilization_gb"]
            ].assign(
                line_numbers=self.with_occurence_memory_event_aggregation_by_lines[
                    "line_numbers"
                ]
                .str.split(",")
                .str[0],
                class_names=self.with_occurence_memory_event_aggregation_by_lines[
                    "class_names"
                ]
                .str.split(",")
                .str[0],
                function_names=self.with_occurence_memory_event_aggregation_by_lines[
                    "function_names"
                ]
                .str.split(",")
                .str[0],
            )
        )

    def _get_context(self, line_number: int) -> Optional[CodeContext]:
        """
        Get the context for a given line number in the model file using AST parser.
        Fast O(1) lookup using pre-computed line-to-context mapping.
        """
        return self.ast_parser.get_context_for_line(line_number)

    def _build_stack_trace_string(
        self, memory_event_call_stack: List[Union[Dict[str, Union[str, int]], Frame]]
    ) -> str:
        if not memory_event_call_stack:
            return ""

        if isinstance(memory_event_call_stack[0], Dict):
            return "\n".join(
                [
                    f"{f['name']}, {f['filename']}:{f['line']}"  # pyre-ignore
                    for f in memory_event_call_stack
                ]
            )
        else:
            return "\n".join(
                [
                    f"{f.name}, {f.filename}:{f.line}"  # pyre-ignore
                    for f in memory_event_call_stack
                ]
            )

    def _process_memory_events(self) -> pd.DataFrame:
        memory_events: dict[int, MemoryEvent] = self.memory_snapshot.call_stack_hash_set
        event_data = []
        for event in memory_events.values():
            stack_trace = event.call_stack
            memory_utilization = event.mem_size
            stack_trace_string = self._build_stack_trace_string(stack_trace)

            # Find all instances of the model file in the stack trace
            model_frames = [
                frame
                for frame in stack_trace
                if self.file_name
                == (
                    frame.filename.split("/")[-1]
                    if isinstance(frame, Frame)
                    else frame.get("filename", "").split("/")[-1]
                )
            ]

            if model_frames:
                # Process the most recent occurrence of the model file
                contexts = []
                for frame in model_frames:
                    context = self._get_context(
                        frame.line if isinstance(frame, Frame) else frame["line"]
                    )
                    if context:
                        contexts.append(context)

                if contexts:
                    event_data.append(
                        {
                            "line_numbers": ",".join(
                                map(
                                    str,
                                    [
                                        f.line if isinstance(f, Frame) else f["line"]
                                        for f in model_frames
                                    ],
                                )
                            ),
                            "line_content": contexts[0].line_content,
                            "class_names": ",".join(
                                [
                                    c.class_name if c.class_name else "_"
                                    for c in contexts
                                ]
                            ),
                            "function_names": ",".join(
                                [
                                    c.function_name if c.function_name else "_"
                                    for c in contexts
                                ]
                            ),
                            "stack_trace": stack_trace_string,
                            "memory_utilization": memory_utilization,
                        }
                    )
            else:
                # If no instance of the model file is found, store the stack trace
                event_data.append(
                    {
                        "line_numbers": None,
                        "line_content": None,
                        "class_names": None,
                        "function_names": None,
                        "stack_trace": stack_trace_string,
                        "memory_utilization": memory_utilization,
                    }
                )

        # Convert the list of dictionaries to a DataFrame
        return pd.DataFrame(event_data)

    def _aggregate_memory_events(
        self,
        groupby_columns: List[str],
        sort_by_column: str = "total_memory_utilization_gb",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Aggregate memory events by specified columns.
        Args:
            groupby_columns (List[str]): Columns to group by.
            sort_by_column (str): Column to sort by. Defaults to "total_memory_utilization_gb".
            ascending (bool): Whether to sort in ascending order. Defaults to False.
        Returns:
            pd.DataFrame: Aggregated memory events.
        """
        with_occurrence = self.memory_events_df.dropna(subset=groupby_columns)
        aggregated = (
            with_occurrence.groupby(groupby_columns)
            .agg(
                total_memory_utilization=pd.NamedAgg(
                    column="memory_utilization", aggfunc="sum"
                ),
                occurrences=pd.NamedAgg(column="line_content", aggfunc="count"),
                # stack_trace=pd.NamedAgg(
                #     column="stack_trace", aggfunc=lambda x: x.iloc[0]
                # ),
            )
            .reset_index()
        )

        # Convert total memory utilization from bytes to GiB
        aggregated["total_memory_utilization_gb"] = aggregated[
            "total_memory_utilization"
        ] / (1024 * 1024 * 1024)
        aggregated = aggregated.drop(columns=["total_memory_utilization"])

        return aggregated.sort_values(by=sort_by_column, ascending=ascending)

    def _aggregate_without_occurrences(self) -> pd.DataFrame:
        """
        Aggregate stack traces unrelated to the model.
        Returns:
            pd.DataFrame: Aggregated stack traces.
        """
        without_occurrence = self.memory_events_df[
            self.memory_events_df[
                ["line_content", "line_numbers", "class_names", "function_names"]
            ]
            .isnull()
            .all(axis=1)
        ]
        aggregated = (
            without_occurrence.groupby("stack_trace")
            .agg(
                total_memory_utilization=pd.NamedAgg(
                    column="memory_utilization", aggfunc="sum"
                ),
                occurrences=pd.NamedAgg(column="stack_trace", aggfunc="count"),
            )
            .reset_index()
        )
        return aggregated.sort_values(by="total_memory_utilization", ascending=False)

    def _sum_memory_utilization(self, df: pd.DataFrame, line_number: int) -> float:
        """
        This function calculates the sum of memory utilization for all rows in the dataframe
        where the given line number is at the beginning of the 'line_numbers' column.
        Args:
            df (pd.DataFrame): The input dataframe.
            line_number (int): The line number to filter by.
        Returns:
            float: The sum of memory utilization.
        """
        # Convert the line number to string for comparison
        line_number_str = str(line_number)
        # Filter out rows with empty or None 'line_numbers'
        filtered_df = df[df["line_numbers"].notna() & (df["line_numbers"] != "")]
        # Filter the dataframe to include only rows where the line number is at the beginning of 'line_numbers'
        filtered_df = filtered_df[
            (filtered_df["line_numbers"].str.startswith(line_number_str + ","))
            | (filtered_df["line_numbers"].eq(line_number_str))
        ]
        # Calculate the sum of memory utilization
        total_memory_utilization = filtered_df["memory_utilization"].sum()
        return total_memory_utilization

    def display_parsed_model_info(self) -> None:
        print("Classes and their line ranges:")
        for class_name, class_info in self.classes.items():
            print(
                f"Class: {class_name}, Start Line: {class_info['start_line']}, End Line: {class_info['end_line']}"
            )
            for func_name, func_info in class_info["functions"].items():
                print(
                    f"  Function: {func_name}, Start Line: {func_info['start_line']}, End Line: {func_info['end_line']}"
                )
        print("\nFunctions outside of classes and their line ranges:")
        for func_name, func_info in self.functions.items():
            print(
                f"Function: {func_name}, Start Line: {func_info['start_line']}, End Line: {func_info['end_line']}"
            )

    def get_printable_dataframes(
        self, dfs: List[pd.DataFrame], titles: List[str], max_rows: int = 100
    ):
        """
        Accepts a list of pandas DataFrames, converts them to strings,
        and returns the concatenated string.

        Args:
            dfs (list[pandas.DataFrame]): List of DataFrames
            titles (list[str], optional): List of titles corresponding to each DataFrame. Defaults to None.
            max_rows (int, optional): Maximum number of rows to display. Defaults to 100.

        Returns:
            str: A string representation of the input DataFrames
        """
        output = StringIO()
        for i, df in enumerate(dfs):
            title = f"DataFrame {i + 1}" if titles is None else titles[i]
            output.write(f"\n{title}\n{'=' * len(title)}\n")
            table = tabulate(
                df.head(max_rows).to_dict(orient="records"),
                headers="keys",
                tablefmt="psql",
            )
            output.write(table + "\n\n")
        return output.getvalue()

    def get_printable_classification_info(
        self,
        specific_dfs: List[pd.DataFrame],
        display_titles: List[str],
        verbose: bool = False,
    ):
        """
        Returns a string representation of the memory event aggregation dataframes.
        If specific_dfs is not provided, returns the string representation of all the memory event aggregation dataframes.
        If specific_dfs is provided, returns the string representation of the specific memory event aggregation dataframes.
        Args:
            specific_dfs (list[pd.DataFrame], optional): List of specific memory event aggregation dataframes to return. Defaults
                to None, which means all the memory event aggregation dataframes will be returned.
            display_titles (list[str], optional): List of titles to display for the specific memory event aggregation dataframes. Defaults
                to None, which means the default titles will be used.
            verbose (bool, optional): Whether to display all the memory event aggregation dataframes. Defaults to False.
        Returns:
            str: A string representation of the memory event aggregation dataframes.
        """
        if verbose:
            dfs = [
                self.with_occurence_memory_event_aggregation_by_lines,
                self.with_occurence_memory_event_aggregation_by_class_and_function,
                self.with_occurence_memory_event_aggregation_by_class,
                self.with_occurence_memory_event_aggregation_per_stack_trace,
                self.without_occurence_memory_event_aggregation,
            ]
            titles = [
                "with_occurence_memory_event_aggregation_per_line_content",
                "with_occurence_memory_event_aggregation_by_class_and_function",
                "with_occurence_memory_event_aggregation_by_class",
                "with_occurence_memory_event_aggregation_per_stack_trace",
                "without_occurence_memory_event_aggregation",
            ]
            return self.get_printable_dataframes(dfs, titles)
        else:
            return self.get_printable_dataframes(specific_dfs, display_titles)

    def get_augmented_model_file_view(self) -> str:
        """
        Returns a string representation of the model file with line numbers and memory information.
        """
        output = StringIO()
        for i, line in enumerate(self.ast_parser.lines):
            memory_usage_gb = self._sum_memory_utilization(
                self.memory_events_df, i + 1
            ) / (1024 * 1024 * 1024)
            memory_usage = f"{memory_usage_gb:.3f} GiB" if memory_usage_gb > 0 else ""
            output.write(f"{i + 1:>5} {memory_usage:<15} | {line}\n")

        return output.getvalue()

    def compare_with_occurence_memory_event_aggregation_by_lines(
        self, other_classifier: "AbstractSyntaxAnalyzer"
    ) -> str:
        """
        Compare the with_occurence_memory_event_aggregation_by_lines dataframes of two AbstractSyntaxAnalyzer instances.

        Args:
            other_classifier (AbstractSyntaxAnalyzer): The other instance to compare with.

        Returns:
            str: A string representation of the comparison of the two tables.
        """

        # Get the dataframes to compare
        df1 = self.with_occurence_memory_event_aggregation_by_lines
        df2 = other_classifier.with_occurence_memory_event_aggregation_by_lines

        # Merge the dataframes on the line_content column
        merged_df = pd.merge(
            df1,
            df2,
            on=["line_content", "line_numbers"],
            how="outer",
            suffixes=("_df1", "_df2"),
        )

        # Fill NaN values in the occurrences and total_memory_utilization_gb columns
        merged_df[["occurrences_df1", "total_memory_utilization_gb_df1"]] = merged_df[
            ["occurrences_df1", "total_memory_utilization_gb_df1"]
        ].fillna(0)
        merged_df[["occurrences_df2", "total_memory_utilization_gb_df2"]] = merged_df[
            ["occurrences_df2", "total_memory_utilization_gb_df2"]
        ].fillna(0)

        # Calculate the differences in occurrences and total memory utilization
        merged_df["occurrences_diff"] = (
            merged_df["occurrences_df1"] - merged_df["occurrences_df2"]
        )
        merged_df["total_memory_utilization_gb_diff"] = (
            merged_df["total_memory_utilization_gb_df1"]
            - merged_df["total_memory_utilization_gb_df2"]
        )

        # Create a printable version of the comparison table
        output = StringIO()
        output.write(
            "Comparison of with_occurence_memory_event_aggregation_by_lines dataframes\n"
        )
        output.write(
            "=====================================================================\n"
        )
        table = tabulate(
            merged_df.to_dict(orient="records"),
            headers="keys",
            tablefmt="psql",
        )
        output.write(table + "\n")

        return output.getvalue()
