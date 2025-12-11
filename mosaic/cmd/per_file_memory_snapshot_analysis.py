# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging

import click
from mosaic.libmosaic.analyzer.abstract_syntax_analyzer import AbstractSyntaxAnalyzer
from mosaic.libmosaic.analyzer.memory_abstract import MemoryAbstract
from mosaic.libmosaic.utils.upload_utils import create_paste


def _setup_memory_abstract_and_analyzer(
    snapshot: str,
    file_path: str | None = None,
    file_name: str | None = None,
    file_content_path: str | None = None,
    allocation: str = "",
    action: str = "alloc",
) -> tuple[MemoryAbstract, AbstractSyntaxAnalyzer]:
    """
    Shared setup logic for both analyze and compare commands.

    Supports two input modes:
    1. File path mode: Provide file_path to read from disk
    2. Content mode: Provide file_name + file_content_path for snapshot-based analysis
    """
    if action == "free":
        logging.info(
            "Overloading action 'free' to 'free_completed' for memory analysis"
        )
        action = "free_completed"

    memory_abstract = MemoryAbstract(memory_snapshot_file=snapshot)
    memory_abstract.load_memory_snapshot()

    # Read file content if content mode is used
    file_content = None
    if file_content_path:
        with open(file_content_path, "r") as f:
            file_content = f.read()

    analyzer = AbstractSyntaxAnalyzer(
        memory_snapshot=memory_abstract.memory_snapshot,
        file_path=file_path,
        file_name=file_name,
        file_content=file_content,
        allocation=allocation,
        action=action,
    )
    return memory_abstract, analyzer


def _handle_output(
    content: str, paste: bool, paste_title: str = "Per-File Memory Analysis Results"
) -> None:
    """Shared output handling logic."""
    if paste:
        create_paste(content=content, paste_title=paste_title)
    else:
        print(content)


@click.group()
def per_file_memory_analysis() -> None:
    """Per-file memory snapshot analysis tools."""
    logging.basicConfig(level=logging.INFO)


@per_file_memory_analysis.command(name="analyze")
@click.option("--snapshot", required=True, help="Path to memory snapshot file")
@click.option(
    "--file-path",
    help="Path to file definition file (Mode 1: reads from disk, requires same revision as snapshot)",
)
@click.option(
    "--file-name",
    help="File name to search for in snapshot (Mode 2: use with --file-content)",
)
@click.option(
    "--file-content",
    help="Path to file containing the file contents for referencing (Mode 2: use with --file-name)",
)
@click.option("--paste", is_flag=True, help="Output to paste")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option("--allocation", default="", help="Allocation to filter")
@click.option("--action", default="alloc", help="Action to filter")
@click.option("--augmented-view", is_flag=True, help="Show augmented model view")
def analyze_single(
    snapshot: str,
    file_path: str | None,
    file_name: str | None,
    file_content: str | None,
    paste: bool,
    verbose: bool,
    allocation: str,
    action: str,
    augmented_view: bool,
) -> None:
    """
    Analyze a single memory snapshot.

    Two input modes are supported:
    1. Provide --file-path: Reads the file from disk (requires same revision as snapshot)
    2. Provide --file-name and --file-content: Search for file in snapshot using provided contents
    """
    # Validate input parameters
    if file_path is None and (file_name is None or file_content is None):
        raise click.UsageError(
            "Either --file-path must be provided, or both --file-name and --file-content must be provided"
        )
    if file_path is not None and (file_name is not None or file_content is not None):
        raise click.UsageError(
            "Cannot provide both --file-path and --file-name/--file-content. Choose one input mode."
        )

    memory_abstract, analyzer = _setup_memory_abstract_and_analyzer(
        snapshot=snapshot,
        file_path=file_path,
        file_name=file_name,
        file_content_path=file_content,
        allocation=allocation,
        action=action,
    )

    output_dataframes = [
        analyzer.simplified_with_occurence_memory_event_aggregation_by_lines,
        analyzer.without_occurence_memory_event_aggregation,
    ]
    output_titles = [
        "Aggregated Stack Traces by Model References",
        "Aggregated Stack Traces that do not Have Model References",
    ]

    content = (
        f"Total Peak Memory Usage (Relative to Start): {memory_abstract.memory_snapshot.dynamic_memory_peak / 1024 / 1024 / 1024} GiB\n"
        + f"Total Static Memory Usage (estimated by Pytorch visualizer): {memory_abstract.memory_snapshot.static_memory / 1024 / 1024 / 1024} GiB\n"
        + f"Total Overall Peak Memory Usage (Dynamic + Static): {(memory_abstract.memory_snapshot.dynamic_memory_peak + memory_abstract.memory_snapshot.static_memory) / 1024 / 1024 / 1024} GiB\n"
        + analyzer.get_printable_classification_info(
            output_dataframes,
            output_titles,
            verbose=verbose,
        )
    )

    _handle_output(content, paste, "Peak Memory Classification: Table View")

    if augmented_view:
        create_paste(
            content=analyzer.get_augmented_model_file_view(),
            paste_title="Peak Memory Classification: Augmented Model File View",
        )


@per_file_memory_analysis.command(name="compare")
@click.option("--base-snapshot", required=True, help="Base snapshot file")
@click.option("--diff-snapshot", required=True, help="Comparison snapshot file")
@click.option(
    "--file-path",
    help="Path to file definition file (Mode 1: reads from disk, requires same revision as snapshot)",
)
@click.option(
    "--file-name",
    help="File name to search for in snapshot (Mode 2: use with --file-content)",
)
@click.option(
    "--file-content",
    help="Path to file containing the file contents for referencing (Mode 2: use with --file-name)",
)
@click.option("--paste", is_flag=True, help="Output to paste")
def compare_snapshots(
    base_snapshot: str,
    diff_snapshot: str,
    file_path: str | None,
    file_name: str | None,
    file_content: str | None,
    paste: bool,
) -> None:
    """
    Compare two memory snapshots.

    Two input modes are supported:
    1. Provide --file-path: Reads the file from disk (requires same revision as snapshot)
    2. Provide --file-name and --file-content: Search for file in snapshot using provided contents
    """
    # Validate input parameters
    if file_path is None and (file_name is None or file_content is None):
        raise click.UsageError(
            "Either --file-path must be provided, or both --file-name and --file-content must be provided"
        )
    if file_path is not None and (file_name is not None or file_content is not None):
        raise click.UsageError(
            "Cannot provide both --file-path and --file-name/--file-content. Choose one input mode."
        )

    # Setup analyzer for "base" snapshot
    _, analyzer_base = _setup_memory_abstract_and_analyzer(
        snapshot=base_snapshot,
        file_path=file_path,
        file_name=file_name,
        file_content_path=file_content,
    )

    # Setup analyzer for "diff" snapshot
    _, analyzer_diff = _setup_memory_abstract_and_analyzer(
        snapshot=diff_snapshot,
        file_path=file_path,
        file_name=file_name,
        file_content_path=file_content,
    )

    content = analyzer_base.compare_with_occurence_memory_event_aggregation_by_lines(
        analyzer_diff
    )

    _handle_output(content, paste, "Memory Event Aggregation by Lines Comparison")


if __name__ == "__main__":
    per_file_memory_analysis()
