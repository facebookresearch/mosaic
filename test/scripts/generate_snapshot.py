# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import pickle
import random
import time


def random_frame():
    frame_names = ["foo", "bar", "baz"]
    filenames = ["/path/to/foo.py", "/path/to/bar.py", "/path/to/baz.py"]
    return {
        "name": random.choice(frame_names),
        "filename": random.choice(filenames),
        "line": random.randint(100, 2000),
    }


def generate_alloc_event(start_time):
    compile_context = ["context1", "context2", "context3"]
    return {
        "action": "alloc",
        "addr": random.randint(int(1e12), int(1e15)),
        "size": random.randint(int(1e6), int(1e8)),
        "stream": random.randint(0, 3),
        "time_us": start_time,
        "frames": [random_frame() for _ in range(random.randint(2, 5))],
        "compile_context": random.choice(compile_context),
    }


def generate_free_completed_event(alloc_event, end_time):
    return {
        "action": "free_completed",
        "addr": alloc_event["addr"],
        "size": alloc_event["size"],
        "stream": alloc_event["stream"],
        "time_us": end_time,
        "frames": [random_frame() for _ in range(random.randint(2, 5))],
    }


def generate_device_events(n, overlap_chance=0.5):
    events = []
    active_allocs = []
    current_time = int(time.time() * 1e6)
    for i in range(n):
        # Decide whether to free an active allocation (overlap) or create a new one
        if active_allocs and random.random() < overlap_chance:
            # Free a random active allocation
            alloc_event = random.choice(active_allocs)
            end_time = alloc_event["time_us"] + random.randint(1, int(1e6))
            free_event = generate_free_completed_event(alloc_event, end_time)
            events.append(free_event)
            active_allocs.remove(alloc_event)
        else:
            # Create a new allocation
            start_time = current_time + random.randint(1, int(1e6))
            alloc_event = generate_alloc_event(start_time)
            events.append(alloc_event)
            active_allocs.append(alloc_event)
        # Advance current_time for next event
        current_time += random.randint(1, int(5e5))
    # Free any remaining allocations
    for alloc_event in active_allocs:
        end_time = current_time + random.randint(1, int(1e6))
        free_event = generate_free_completed_event(alloc_event, end_time)
        events.append(free_event)
        current_time += random.randint(1, int(5e5))

    events.sort(key=lambda x: x["time_us"])
    return [events]


def random_block(base_address, index):
    return {
        "address": base_address + index * 512,
        "size": 512,
        "requested_size": random.randint(1, 64),
        "state": "active_allocated",
        "frames": [],
    }


def generate_segment(
    device=0, stream=0, segment_type="small", pool_id=[0, 0], num_blocks=3
):
    base_address = random.randint(int(1e10), int(1e12))
    total_size = 52428800
    allocated_size = random.randint(int(1e7), total_size)
    active_size = allocated_size
    requested_size = random.randint(int(1e7), allocated_size)
    blocks = [random_block(base_address, i) for i in range(num_blocks)]
    return {
        "device": device,
        "address": base_address,
        "total_size": total_size,
        "allocated_size": allocated_size,
        "active_size": active_size,
        "requested_size": requested_size,
        "stream": stream,
        "segment_type": segment_type,
        "segment_pool_id": pool_id,
        "is_expandable": True,
        "frames": [],
        "blocks": blocks,
    }


def generate_segments(n):
    return [generate_segment(num_blocks=random.randint(2, 5)) for _ in range(n)]


def generate_external_annotations(device_traces, num_annotations=3):
    annotations = []
    device = 0
    names = [
        "Annotation1",
        "Annotation2",
        "Annotation3",
    ]
    for _ in range(num_annotations):
        # Pick a random subset of device_traces for this annotation
        subset = random.sample(
            device_traces[0], random.randint(2, max(2, len(device_traces[0]) // 2))
        )
        times = [event["time_us"] for event in subset]
        start_time = min(times) - random.randint(10, 100)
        end_time = max(times) + random.randint(10, 100)
        name = random.choice(names)
        # START annotation
        annotations.append(
            {"stage": "START", "name": name, "device": device, "time_us": start_time}
        )
        # END annotation
        annotations.append(
            {"stage": "END", "name": name, "device": device, "time_us": end_time}
        )
    # Sort annotations by time_us for readability
    annotations.sort(key=lambda x: x["time_us"])
    return annotations


if __name__ == "__main__":
    num_events = 1000  # Number of device_traces events
    num_segments = 5  # Number of segments
    num_annotations = 3  # Number of annotation pairs

    device_traces = generate_device_events(num_events)
    segments = generate_segments(num_segments)
    external_annotations = generate_external_annotations(device_traces, num_annotations)

    output = {
        "device_traces": device_traces,
        "segments": segments,
        "external_annotations": external_annotations,
    }
    print(json.dumps(output, indent=2))
    with open("output.pickle", "wb") as f:
        pickle.dump(output, f)
