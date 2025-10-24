# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import os
import pickle
from typing import Any, Optional


class SnapshotLoader:
    def __init__(self, filename: Optional[str]) -> None:
        self.snapshot_filename: str = ""
        self.snapshot_data: dict[str, Any] = {}
        if filename is not None:
            self.register_memory_snapshot(filename)

    # external APIs
    def register_memory_snapshot(self, filename: str) -> None:
        self.snapshot_filename = filename
        pass

    def _load_memory_snapshot_from_pickle_file(self) -> None:
        # Load the pickle file
        # return the tensor life cycle history
        # At a high level, the data is a dict, with two main keys: "segments" and "device_traces"
        file_size = os.path.getsize(self.snapshot_filename) / (1024 * 1024)
        logging.info(
            f"Loading snapshot {self.snapshot_filename}, size {file_size:.2f}MB ..."
        )
        with open(self.snapshot_filename, "rb") as f:
            self.snapshot_data = pickle.load(f)
            f.close()
        logging.info("Snapshot loaded successfully.")

    def _load_memory_snapshot_from_json_file(self) -> None:
        # Load the json file
        # return the tensor life cycle history
        # At a high level, the data is a dict, with two main keys: "segments" and "device_traces"
        file_size = os.path.getsize(self.snapshot_filename) / (1024 * 1024)
        logging.info(
            f"Loading snapshot {self.snapshot_filename}, size {file_size:.2f}MB ..."
        )
        with open(self.snapshot_filename, "r") as f:
            self.snapshot_data = json.load(f)
            f.close()
        logging.info("Snapshot loaded successfully.")

    def load_memory_snapshot(self, stream_load: bool = False) -> None:
        # Load the pickle file

        if not os.path.exists(self.snapshot_filename):
            logging.error(
                "Memory Snapshot file , call register_memory_snapshot() to register."
            )
            return

        if not stream_load:
            logging.info(f"Loading snapshot {self.snapshot_filename} using io read")
            if self.snapshot_filename.endswith(
                ".pickle"
            ) or self.snapshot_filename.endswith(".pkl"):
                self._load_memory_snapshot_from_pickle_file()
            else:
                # if not pickle assume it is a json file (used for testing)
                self._load_memory_snapshot_from_json_file()
        else:
            # TODO: add stream read support
            logging.info(f"Loading snapshot {self.snapshot_filename} using stream read")
        pass
