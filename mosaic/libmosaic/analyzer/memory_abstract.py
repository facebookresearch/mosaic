# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
from typing import Optional

from mosaic.libmosaic.analyzer.gpu_trace import GPUTrace

from mosaic.libmosaic.analyzer.memory_snapshot import MemorySnapshot


class MemoryAbstract:
    def __init__(
        self,
        memory_snapshot_file: str,
        gpu_trace_file: Optional[str] = None,
    ) -> None:
        if gpu_trace_file is not None:
            self.gpu_trace: GPUTrace = GPUTrace(filename=gpu_trace_file)
        self.memory_snapshot = MemorySnapshot(filename=memory_snapshot_file)

        self.memory_abstract: Optional[MemoryAbstract] = None

        # trainer info
        self.global_rank: Optional[int] = None
        self.op_cache: list[object] = []
        self.rank: int = 0  # global rank of the device

    # load & analyze
    # load memory snapshot from a pickle file
    def load_memory_snapshot(self) -> None:
        self.memory_snapshot.load_memory_snapshot()

    # load trace from a kineto trace
    def load_gpu_trace(self) -> None:
        self.gpu_trace.load_gpu_trace()

    def query_op(self, op_name: str, num_op: int) -> Optional[list[dict[str, str]]]:
        ops = self.gpu_trace.query_op(op_name, num_op)
        # TODO: cache queried results somewhere
        return ops

    # output memory abstract to a json file
    def output_memory_abstract(self, output_filename: str) -> None:
        if self.memory_abstract is None:
            logging.error("Memory abstract is not generated.")
            return
        with open(output_filename, "w") as outfile:
            json.dump(self.memory_abstract, outfile)

    # load existing memory abstract from a json file
    def load_memory_abstract(self, filename: str) -> None:
        self.memory_abstract = json.loads(filename)
