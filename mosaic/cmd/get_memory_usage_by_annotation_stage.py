# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import click
from mosaic.cmd.entry_point import get_memory_usage_by_annotation_stage


@click.command()
@click.option("--snapshot", type=str, default="", help="Path to memory snapshot file")
@click.option(
    "--annotation",
    type=str,
    multiple=True,
    default=[],
    help="Annotation to filter memory usage",
)
@click.option(
    "--paste",
    flag_value=True,
    default=False,
    help="Output memory usage by annotation to paste",
)
def main(snapshot: str, annotation: str, paste: bool) -> None:
    get_memory_usage_by_annotation_stage(
        snapshot=snapshot, annotation=annotation, paste=paste
    )


if __name__ == "__main__":
    main()
