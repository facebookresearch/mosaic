# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
import os
import subprocess


async def upload_viz_html(
    local_file_path: str,
    upload_name: str,
    upload_folder: str,
    bucket_name: str,
) -> None:
    raise NotImplementedError("upload_viz_html is not implemented")


async def upload_report(
    report: str,
    upload_folder: str,
    upload_name: str,
    bucket_name: str,
) -> None:
    raise NotImplementedError("upload_report is not implemented")


def get_upload_folder(job_name: str) -> str:
    return ""


def get_jobname() -> str:
    return ""


def get_upload_file_name(path_to_viz: str) -> str:
    return os.path.basename(path_to_viz)


def create_paste(content: str, paste_title: str = "") -> str:
    raise NotImplementedError("create_paste is not implemented")
