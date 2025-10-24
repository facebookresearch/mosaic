# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import gzip
import json
import logging
import os
from typing import Optional


class GPUTrace:
    def __init__(self, filename: Optional[str] = None) -> None:
        self.trace_file_name: str
        self.trace_file_size: Optional[int] = None
        self.trace_data: Optional[dict[str, list[dict[str, str]]]] = None

        if filename is not None:
            self.register_trace_file(filename)
            return

    def register_trace_file(self, file_name: str) -> None:
        self.trace_file_name: str = file_name
        self.trace_file_size: float = os.path.getsize(self.trace_file_name) / (
            1024 * 1024
        )
        logging.info(
            f"Kineto trace {self.trace_file_name} registered, size {self.trace_file_size:.2f}MB."
        )

    # load the trace data entirely to memory
    # [TODO] currently load as json object, adopt HTA method for efficient query in the future
    def load_gpu_trace(self) -> None:
        # read gpu traces
        # Open the compressed file for reading
        if self.trace_file_name is None:
            logging.error(
                "Kineto Trace is not registered, call register_trace_file() to fix."
            )
            return None
        logging.info(
            f"Loading kineto trace {self.trace_file_name}, size {self.trace_file_size:.2f}MB ..."
        )

        with gzip.open(self.trace_file_name, "r") as f:
            # Read the entire contents of the file into a string
            data_str = f.read()
            f.close()
            # Parse the JSON data
            self.trace_data = json.loads(data_str)

    def query_op(self, op_name: str, num_op: int = 1) -> Optional[list[dict[str, str]]]:
        if self.trace_file_name is None:
            logging.error(
                "Kineto Trace is not registered, call register_trace_file() to fix."
            )
            return None

        if self.trace_data is None:
            logging.info("Trace is not loaded entirely, query using streaming method.")
            return self._query_op_from_streaming(op_name, num_op)
        else:
            logging.info("Trace is entirely loaded, query from trace data directly.")
            return self._query_op_from_trace_data(op_name, num_op)

    # [TODO] adopt HTA method for efficient query in the future
    def _query_op_from_trace_data(
        self, op_name: str, num_op: int = 1
    ) -> Optional[list[dict[str, str]]]:
        # get all events from trace
        ops = []
        if self.trace_data is None:
            logging.error("Trace data is not loaded.")
            return None
        trace_events: list[dict[str, str]] = self.trace_data["traceEvents"]

        # find the step level events
        for event in trace_events:
            if op_name in event["name"] and event["cat"] == "user_annotation":
                # TODO: current step number in xlformer has +2 offiste, need to correct later when fix in xlformer
                ops.append(event)
                num_op -= 1
                if num_op == 0:
                    break
        return ops
        pass

    # streaming based trace processing, fast in deal with large traces with limited number of ops
    # find num_op (default = 1) of ops with name op_name (default = "ProfilerStep" for step info)
    def _query_op_from_streaming(
        self, op_name: Optional[str] = None, num_op: int = 1
    ) -> Optional[list[dict[str, str]]]:
        if op_name is None:
            logging.error("op_name is not specified.")
            return None
        ops = []
        with gzip.open(self.trace_file_name, "rb") as f:
            line = f.readline()
            while line:
                if line == b"  {\n":
                    block_lines = line
                    line = f.readline()
                    while line != b"  },\n":
                        block_lines += line
                        line = f.readline()
                    end = "}"
                    object_lines = block_lines.decode("utf-8")
                    object_lines += end
                    # process block
                    if op_name in object_lines:
                        obj = json.loads(object_lines)
                        ops.append(obj)
                        num_op -= 1
                        if num_op == 0:
                            break
                line = f.readline()
            f.close()

        return ops
        pass
