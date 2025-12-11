# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from mosaic.libmosaic.analyzer.memory_abstract import MemoryAbstract
from mosaic.libmosaic.utils.data_utils import MemoryEvent
from mosaic.libmosaic.utils.profiler_plotter import ProfilerPlotter
from mosaic.libmosaic.utils.snapshot_utils import (
    find_memory_snapshot_peak_call_stack_diffs,
)

from mosaic.libmosaic.utils.upload_utils import (
    create_paste,
    get_jobname,
    get_upload_file_name,
    get_upload_folder,
    upload_report,
)

from mosaic.libmosaic.utils.utils import (
    get_memory_peak_event_in_json,
    get_printable_memory_event_set,
)

from omegaconf import DictConfig, OmegaConf


logging.getLogger().setLevel(logging.INFO)


@dataclass
class CustomProfileRule:
    """
    A single custom profiling rule that maps a category name to a regex pattern.

    Attributes:
        name: Category name for memory allocations matching this rule
        pattern: Regex pattern to match against stack trace frame names and filenames
        description: Optional human-readable description of what this rule captures
        priority: Optional priority level (lower values = higher priority)
    """

    name: str
    pattern: str
    description: Optional[str] = None
    priority: int = 0


@dataclass
class CustomProfile:
    """
    Configuration for custom memory profiling rules.

    This dataclass defines a structured way to specify custom categorization rules
    for memory allocations based on stack trace patterns.

    Attributes:
        rules: List of custom profiling rules, processed in order

    Note:
        Rules are processed in the order they appear in the list.
        The first matching rule wins, enabling hierarchical categorization
        from specific to general patterns.
    """

    rules: List[CustomProfileRule] = field(default_factory=list)

    def to_dict(self) -> Dict[str, str]:
        """Convert to simple dictionary format for backward compatibility."""
        return {rule.name: rule.pattern for rule in self.rules}


def get_memory_usage(
    snapshot: str,
    trace: str,
    allocation: str,
    action: str,
    paste: bool,
    print_stack: bool,
) -> MemoryAbstract:
    # a free action is just requested followed by freed so just change it to free_completed
    if action == "free":
        logging.info(
            "Overloading action 'free' to 'free_completed' for memory analysis"
        )
        action = "free_completed"
    allocation_str = f"{action}: {allocation}"
    memory_abstract = MemoryAbstract(memory_snapshot_file=snapshot)
    memory_abstract.load_memory_snapshot()
    memory_abstract.memory_snapshot.analyze_memory_snapshot(
        opt="memory_usage", allocation=allocation, action=action
    )

    if allocation_str not in memory_abstract.memory_snapshot.memory_usage:
        logging.error(f"{allocation_str} not found in memory snapshot")
        return memory_abstract
    memory_usage_str = f"Memory Usage: {memory_abstract.memory_snapshot.memory_usage[allocation_str] / 1024 / 1024 / 1024} GiB\n"
    if paste:
        create_paste(
            memory_usage_str
            + get_printable_memory_event_set(
                memory_abstract.memory_snapshot.call_stack_hash_set
            )
        )
    else:
        if print_stack:
            print(
                get_printable_memory_event_set(
                    memory_abstract.memory_snapshot.call_stack_hash_set
                )
            )
    return memory_abstract


def get_memory_usage_peak(
    snapshot: str,
    trace: str,
    allocation: str,
    action: str,
    paste: bool,
    print_stack: bool,
    upload_result: bool,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> MemoryAbstract:
    # a free action is just requested followed by freed so just change it to free_completed
    if action == "free":
        logging.info(
            "Overloading action 'free' to 'free_completed' for memory analysis"
        )
        action = "free_completed"
    memory_abstract = MemoryAbstract(memory_snapshot_file=snapshot)
    memory_abstract.load_memory_snapshot()
    memory_abstract.memory_snapshot.analyze_memory_snapshot(
        opt="memory_peak",
        allocation=allocation,
        action=action,
        start_time=start_time,
        end_time=end_time,
    )

    if upload_result:
        data = get_memory_peak_event_in_json(
            memory_abstract.memory_snapshot.call_stack_hash_set
        )

        upload_folder = get_upload_folder(get_jobname())
        upload_name = f"{get_upload_file_name(snapshot)}.json"

        asyncio.run(
            upload_report(data, upload_folder=upload_folder, upload_name=upload_name)
        )
    elif paste:
        create_paste(
            f"Total Peak Dynamic Memory Usage (Relative to Start): {memory_abstract.memory_snapshot.dynamic_memory_peak / 1024 / 1024 / 1024} GiB\n"
            + f"Total Static Memory Usage (estimated by Pytorch visualizer): {memory_abstract.memory_snapshot.static_memory / 1024 / 1024 / 1024} GiB\n"
            + f"Total Overall Peak Memory Usage (Dynamic + Static): {(memory_abstract.memory_snapshot.dynamic_memory_peak + memory_abstract.memory_snapshot.static_memory) / 1024 / 1024 / 1024} GiB\n"
            + get_printable_memory_event_set(
                memory_abstract.memory_snapshot.call_stack_hash_set
            )
        )
    else:
        if print_stack:
            print(
                get_printable_memory_event_set(
                    memory_abstract.memory_snapshot.call_stack_hash_set
                )
            )
    return memory_abstract


def get_memory_usage_diff(
    snapshot_base: str,
    snapshot_diff: str,
    paste: bool,
) -> Dict[int, MemoryEvent]:
    memory_abstract_base = MemoryAbstract(memory_snapshot_file=snapshot_base)
    memory_abstract_diff = MemoryAbstract(memory_snapshot_file=snapshot_diff)

    memory_abstract_base.load_memory_snapshot()
    memory_abstract_diff.load_memory_snapshot()

    memory_abstract_base.memory_snapshot.analyze_memory_snapshot(opt="memory_peak")
    memory_abstract_diff.memory_snapshot.analyze_memory_snapshot(opt="memory_peak")

    call_stack_diff = find_memory_snapshot_peak_call_stack_diffs(
        memory_abstract_base.memory_snapshot.call_stack_hash_set,
        memory_abstract_diff.memory_snapshot.call_stack_hash_set,
    )

    if paste:
        create_paste(get_printable_memory_event_set(call_stack_diff))
    else:
        print(get_printable_memory_event_set(call_stack_diff))
    return call_stack_diff


def get_memory_usage_by_annotation_stage(
    snapshot: str, annotation: Union[str, Tuple[str, ...]], paste: bool
) -> Dict[str, Tuple[Dict[str, Any], float]]:
    memory_abstract = MemoryAbstract(memory_snapshot_file=snapshot)
    memory_abstract.load_memory_snapshot()
    if type(annotation) is str:
        annotation = (annotation,)
    memory_usage_by_annotation = (
        memory_abstract.memory_snapshot.get_memory_usage_by_annotation_stage(
            set(annotation)
        )
    )
    if paste:
        create_paste(
            f"Memory Usage by Annotation: {annotation}\n"
            + "\n".join(
                f"{key}: {value[1] / 1024 / 1024 / 1024} GiB. Annotation: {value[0]}"
                for key, value in memory_usage_by_annotation.items()
            )
        )
    else:
        for key, value in memory_usage_by_annotation.items():
            print(f"{key}: {value[1] / 1024 / 1024 / 1024} GiB. Annotation: {value[0]}")
    return memory_usage_by_annotation


def get_memory_profile(
    snapshot: str,
    out_path: str,
    profile: str,
    custom_profile: Optional[str] = None,
    sampling_rate: int = 1,
    start_idx: int = 0,
    end_idx: int = sys.maxsize,
    preserve_allocation_order: bool = False,
) -> MemoryAbstract:
    """
    Generate a memory profiling report with customizable categorization.

    Args:
        snapshot: Path to the PyTorch memory snapshot file (.pickle format)
        out_path: Output path for the generated HTML visualization
        profile: Profile type - one of ["annotations", "categories", "compile_context", "custom"]
        custom_profile: JSON/YAML string defining custom profiling rules (required when profile="custom")
        sampling_rate: Sampling rate for visualization (1 = no sampling)
        start_idx: Start index for time-based filtering
        end_idx: End index for time-based filtering

    Returns:
        MemoryAbstract: Analyzed memory snapshot object

    """

    # Parse custom profile if provided
    custom_rules = None
    if profile == "custom":
        if not custom_profile:
            raise ValueError(
                "Custom profile configuration required when profile type is 'custom'. "
                'Provide either a JSON dictionary like \'{"category": "pattern"}\' or '
                "a structured YAML/JSON config with 'rules' list."
            )

        try:
            # Try to parse as structured OmegaConf config first
            try:
                config = OmegaConf.create(custom_profile)
                if isinstance(config, DictConfig) and "rules" in config:
                    # Structured format with rules list
                    custom_profile_obj = CustomProfile(**config)
                    custom_rules = custom_profile_obj.to_dict()
                else:
                    # Simple dictionary format (backward compatibility)
                    if isinstance(config, DictConfig):
                        custom_rules = dict(config)
                    else:
                        custom_rules = config

                    # Validate simple dictionary format
                    if not isinstance(custom_rules, dict):
                        raise ValueError(
                            "Custom profile must be a dictionary or structured config with 'rules'"
                        )

                    for key, value in custom_rules.items():
                        if not isinstance(key, str) or not isinstance(value, str):
                            raise ValueError(
                                "Custom profile dictionary keys and values must be strings (regex patterns)"
                            )

            except Exception as omegaconf_error:
                # Fallback to JSON parsing for backward compatibility
                try:
                    custom_rules = json.loads(custom_profile)
                    if not isinstance(custom_rules, dict):
                        raise ValueError("Custom profile must be a JSON dictionary")

                    for key, value in custom_rules.items():
                        if not isinstance(key, str) or not isinstance(value, str):
                            raise ValueError(
                                "Custom profile dictionary keys and values must be strings"
                            )

                except json.JSONDecodeError:
                    # Re-raise the original OmegaConf error with more context
                    raise ValueError(
                        f"Invalid custom profile format. Expected JSON dictionary or YAML config. "
                        f"OmegaConf error: {omegaconf_error}"
                    )

        except Exception as e:
            raise ValueError(f"Failed to parse custom profile: {e}")

    start_time = time.time()
    memory_abstract = MemoryAbstract(memory_snapshot_file=snapshot)
    memory_abstract.load_memory_snapshot()
    memory_abstract.memory_snapshot.analyze_memory_snapshot(
        "alloc_history",
        profile_types=[profile],
        custom_rules=custom_rules,
        preserve_allocation_order=preserve_allocation_order,
    )

    peak = memory_abstract.memory_snapshot.max_memory_usage
    print("Memory Usage At Peak:")
    print(peak)

    pp = ProfilerPlotter(
        # pyre-fixme[6]: For 1st argument expected `Dict[int, MemoryUsage]` but got
        #  `Dict[int, BaseMemoryUsage]`.
        memory_abstract.memory_snapshot.memory_usage_history,
        profile=profile,
        sampling_rate=sampling_rate,
        start_idx=start_idx,
        end_idx=end_idx,
        preserve_allocation_order=preserve_allocation_order,
    )
    pp.plot(out_path)
    end_time = time.time()
    logging.info(
        "Profiling function took: " + str(end_time - start_time) + " seconds to run"
    )
    return memory_abstract


def get_json_snapshot(
    snapshot: str,
    output_file: str,
    upload_result: bool,
) -> None:
    memory_abstract = MemoryAbstract(memory_snapshot_file=snapshot)
    memory_abstract.load_memory_snapshot()
    snapshot_data = memory_abstract.memory_snapshot.snapshot_data

    if output_file:
        with open(output_file, "w") as file:
            json.dump(snapshot_data, file, indent=4)

    if upload_result:
        json_data = json.dumps(snapshot_data, indent=4)
        upload_folder = get_upload_folder(get_jobname())
        upload_name = f"{get_upload_file_name(snapshot)}.json"

        asyncio.run(
            upload_report(
                json_data, upload_folder=upload_folder, upload_name=upload_name
            )
        )
