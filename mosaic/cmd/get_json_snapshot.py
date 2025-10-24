# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import sys

import click
from mosaic.cmd.entry_point import get_json_snapshot


@click.command()
@click.option(
    "--snapshot",
    required=True,
    help="Path to first memory snapshot file",
)
@click.option(
    "--output_file",
    type=click.Path(dir_okay=False),
    default="/tmp/snapshot.json",
    help="Path to output png",
)
@click.option(
    "--upload_result",
    flag_value=True,
    default=False,
    help="Output peak memory snapshot information using upload handler",
)
def main(
    snapshot: str,
    output_file: str,
    upload_result: bool,
) -> None:
    get_json_snapshot(
        snapshot=snapshot,
        output_file=output_file,
        upload_result=upload_result,
    )


if __name__ == "__main__":
    main()
