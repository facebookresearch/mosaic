# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import click
from mosaic.cmd.entry_point import get_memory_usage


@click.command(
    help="Gets the memory usage of a given snapshot at a given allocation *relative* to the start of the trace."
    "If you see a discrepancy between the memory usage in the visualizer and the memory usage reported by this script, it is likely because the visualizer"
    "tries to infer the starting memory usage from the first allocation in the trace, whereas MOSAIC just starts at 0. Enable print-stack to see the stack trace at given allocation."
)
@click.option("--snapshot", type=str, default="", help="Path to memory snapshot file")
@click.option("--trace", type=str, default="", help="Path to gpu trace file")
@click.option("--allocation", type=str, help="Allocation to stop at", required=True)
@click.option(
    "--action",
    type=click.Choice(
        [
            "alloc",
            "segment_alloc",
            "segment_map",
            "free_completed",
            "segment_free",
            "segment_unmap",
            "free",
        ]
    ),
    help="Action to stop at",
    required=True,
)
@click.option("--print_stack", default=True, help="Print stack trace")
@click.option(
    "--paste",
    flag_value=True,
    default=False,
    help="Output peak memory snapshot information to paste",
)
def main(
    snapshot: str,
    trace: str,
    allocation: str,
    action: str,
    paste: bool,
    print_stack: bool,
) -> None:
    get_memory_usage(
        snapshot=snapshot,
        trace=trace,
        allocation=allocation,
        action=action,
        paste=paste,
        print_stack=print_stack,
    )


if __name__ == "__main__":
    main()
