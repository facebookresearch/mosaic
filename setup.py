# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

setup(
    name="mosaic",
    version="0.1.0",
    packages=[
        "mosaic",
        "mosaic.cmd",
        "mosaic.libmosaic",
        "mosaic.libmosaic.analyzer",
        "mosaic.libmosaic.utils",
        "mosaic.mcp",
        "mosaic.mcp_common",
    ],
    package_dir={
        "mosaic.cmd": "cmd",
        "mosaic.libmosaic": "libmosaic",
        "mosaic.libmosaic.analyzer": "libmosaic/analyzer",
        "mosaic.libmosaic.utils": "libmosaic/utils",
    },
    install_requires=[
        "altair>=5.5.0",
        "click>=8.3.0",
        "mcp>=1.0.0",
        "omegaconf>=2.3.0",
        "pandas>=2.3.3",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "mosaic_get_memory_profile = mosaic.cmd.get_memory_profile:main",
            "mosaic_get_json_snapshot = mosaic.cmd.get_json_snapshot:main",
            "mosaic_get_memory_usage_by_annotation_stage = mosaic.cmd.get_memory_usage_by_annotation_stage:main",
            "mosaic_get_memory_usage_diff = mosaic.cmd.get_memory_usage_diff:main",
            "mosaic_get_memory_usage_peak = mosaic.cmd.get_memory_usage_peak:main",
            "mosaic_get_memory_usage = mosaic.cmd.get_memory_usage:main",
            "mosaic_per_file_memory_snapshot_analysis = mosaic.cmd.per_file_memory_snapshot_analysis:per_file_memory_analysis",
            "mosaic_mcp = mosaic.mcp.server:cli",
        ],
    },
)
