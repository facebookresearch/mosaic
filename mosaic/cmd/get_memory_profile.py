# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import sys

import click
from mosaic.cmd.entry_point import get_memory_profile


@click.command()
@click.option(
    "--snapshot",
    required=True,
    help="Path to first memory snapshot file",
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False),
    default="snapshot_graph.html",
    help="Path to output png",
)
@click.option(
    "--profile",
    default="categories",
    type=click.Choice(["annotations", "categories", "compile_context", "custom"]),
    help="Type of profile to generate",
)
@click.option(
    "--custom-profile",
    type=str,
    help="""
    Custom Profile Format:
        The custom_profile parameter accepts either:

        1. Simple JSON dictionary (backward compatibility):
           '{"category_name": "regex_pattern", "fsdp": "fsdp.*", "optimizer": "adam|sgd"}'

        2. Structured YAML/JSON with detailed configuration:
           ```yaml
           rules:
             - name: "fsdp_forward"
               pattern: "fsdp.*forward"
               description: "FSDP forward pass operations"
               priority: 1
             - name: "fsdp_general"
               pattern: "fsdp"
               description: "General FSDP operations"
               priority: 2
             - name: "all_operations"
               pattern: ".*"
               description: "Catch-all for remaining allocations"
               priority: 999
           ```

        Pattern Matching:
        - Regex patterns are matched against both function names and filenames in stack traces
        - Rules are processed in order - first match wins (enables hierarchical categorization)
        - Use specific patterns first, general patterns last
        - Invalid regex patterns are logged as warnings and skipped

    Example Usage:
        # Simple custom profiling
        get_memory_profile(
            snapshot="memory.pickle",
            out_path="profile.html",
            profile="custom",
            custom_profile='{"fsdp": "fsdp.*", "optimizer": "adam|sgd", "other": ".*"}'
        )

        # Advanced custom profiling with structured config
        custom_config = '''
        rules:
          - name: "fsdp_forward"
            pattern: "fsdp.*forward"
            description: "FSDP forward operations"
          - name: "fsdp_backward"
            pattern: "fsdp.*backward"
            description: "FSDP backward operations"
          - name: "optimizer"
            pattern: "adam|sgd|adamw"
            description: "Optimizer operations"
          - name: "general"
            pattern: ".*"
            description: "All other operations"
        '''
        get_memory_profile(
            snapshot="memory.pickle",
            out_path="profile.html",
            profile="custom",
            custom_profile=custom_config
        )
        """,
)
@click.option(
    "--plotter_sampling_rate", default=1, type=int, help="Sampling rate for the plotter"
)
@click.option(
    "--plotter_start_idx", default=0, type=int, help="Start index for the plotter"
)
@click.option(
    "--plotter_end_idx",
    default=sys.maxsize,
    type=int,
    help="End index for the plotter",
)
def main(
    snapshot: str,
    out_path: str,
    profile: str,
    custom_profile: str,
    plotter_sampling_rate: int,
    plotter_start_idx: int,
    plotter_end_idx: int,
) -> None:
    # Validate custom profile arguments
    if profile == "custom" and not custom_profile:
        raise click.BadParameter("--custom-profile required when --profile=custom")
    if profile != "custom" and custom_profile:
        raise click.BadParameter("--custom-profile only valid with --profile=custom")

    get_memory_profile(
        snapshot=snapshot,
        out_path=out_path,
        profile=profile,
        custom_profile=custom_profile,
        sampling_rate=plotter_sampling_rate,
        start_idx=plotter_start_idx,
        end_idx=plotter_end_idx,
    )


if __name__ == "__main__":
    main()
