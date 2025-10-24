# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import click
from mosaic.cmd.entry_point import get_memory_usage_diff


@click.command()
@click.option(
    "--snapshot_base", type=str, default="", help="Path to first memory snapshot file"
)
@click.option(
    "--snapshot_diff", type=str, default="", help="Path to second memory snapshot file"
)
@click.option(
    "--paste",
    flag_value=True,
    default=False,
    help="Output peak memory snapshot information to paste",
)
def main(
    snapshot_base: str,
    snapshot_diff: str,
    paste: bool,
) -> None:
    get_memory_usage_diff(
        snapshot_base=snapshot_base, snapshot_diff=snapshot_diff, paste=paste
    )


if __name__ == "__main__":
    main()
