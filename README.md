# Mosaic - PyTorch Memory Profiling and Analysis Tool

Mosaic is a comprehensive memory profiling and analysis tool designed for PyTorch workloads. It offers a suite of utilities for analyzing [Pytorch Memory Snapshots](https://pytorch.org/blog/understanding-gpu-memory-1/):

- *Visualize* memory usage patterns
- *Identify* memory peaks and bottlenecks

With Mosaic, users can better understand how their PyTorch models use memory. This makes it easier to identify problems and improve efficiency during both training and inference.

## Features

- **Memory Peak Analysis**: Identify when and where peak memory usage occurs during model execution
- **Memory Usage Tracking**: Identify memory usage at specific memory allocation
- **Call Stack Analysis**: Analyze memory usage by call stack to identify memory-intensive operations
- **Custom Profiling**: Create custom categorization rules to profile memory by specific code patterns
- **Memory Comparison**: Compare memory usage between different snapshots or code versions
- **Annotation-based Analysis**: Track memory usage across custom annotations and training stages
- **Visualization**: Generate interactive HTML visualizations of memory usage over time

## Installation

From the `mosaic` directory:

```bash
pip install -e .
```

## Command-Line Tools

Mosaic provides several command-line utilities:

### Get Memory Profile
```bash
mosaic-get-memory --snapshot <path_to_snapshot.pickle> --out-path <output.html> --profile <profile_type>
```

Profile types:
- `annotations`: Profile by external annotations
- `categories`: Profile by PyTorch memory categories
- `compile_context`: Profile by torch.compile context
- `custom`: Profile using custom regex patterns

### Get JSON Snapshot
```bash
mosaic-get-json-snapshot --snapshot <path_to_snapshot.pickle> --output-file <output.json>
```

### Get Memory Usage
```bash
mosaic_get_memory_usage --snapshot <path_to_snapshot.pickle> --allocation <address> --action <alloc|free>
```

### Get Memory Usage Peak
```bash
mosaic_get_memory_usage_peak --snapshot <path_to_snapshot.pickle>
```

### Get Memory Usage Diff
```bash
mosaic_get_memory_usage_diff --snapshot-base <base.pickle> --snapshot-diff <diff.pickle>
```

### Get Memory Usage by Annotation Stage
```bash
mosaic_usage_by_annotation_stage --snapshot <path_to_snapshot.pickle> --annotation <annotation_name>
```

## Python API

### Basic Usage

```python
from mosaic.libmosaic.analyzer.memory_abstract import MemoryAbstract

# Load and analyze a memory snapshot
memory_abstract = MemoryAbstract(memory_snapshot_file="snapshot.pickle")
memory_abstract.load_memory_snapshot()

# Analyze peak memory usage
memory_abstract.memory_snapshot.analyze_memory_snapshot(opt="memory_peak")
peak_memory = memory_abstract.memory_snapshot.memory_peak
print(f"Peak memory: {peak_memory / 1024**3:.2f} GiB")
```

### Custom Profiling

```python
from mosaic.cmd.entry_point import get_memory_profile

# Define custom profiling rules
custom_rules = {
    "Model Forward": ".*forward.*",
    "Optimizer": ".*optimizer.*",
    "Data Loading": ".*DataLoader.*"
}

# Generate profile with custom categories
get_memory_profile(
    snapshot="snapshot.pickle",
    out_path="profile.html",
    profile="custom",
    custom_profile=json.dumps(custom_rules)
)
```

### Memory Usage by Annotation

```python
from mosaic.cmd.entry_point import get_memory_usage_by_annotation_stage

# Get memory usage at each training stage
memory_by_stage = get_memory_usage_by_annotation_stage(
    snapshot="snapshot.pickle",
    annotation=("forward", "backward", "optimizer"),
    paste=False
)

for stage, (annotation, memory_bytes) in memory_by_stage.items():
    print(f"{stage}: {memory_bytes / 1024**3:.2f} GiB")
```

## Core Components

### libmosaic.analyzer
- `MemoryAbstract`: High-level interface for memory analysis
- `MemorySnapshot`: Core snapshot analysis and processing
- `gpu_trace`: GPU trace event analysis

### libmosaic.utils
- `snapshot_loader`: Load PyTorch memory snapshots
- `snapshot_utils`: Utilities for snapshot manipulation
- `plotting`: Generate interactive visualizations
- `data_utils`: Data structures for memory events

### cmd
Command-line entry points for various memory analysis tasks

## Use Cases

1. **Identify Memory Bottlenecks**: Find which operations consume the most memory during training
2. **Optimize Model Memory**: Analyze memory usage patterns to reduce peak memory consumption
3. **Debug OOM Errors**: Understand what causes out-of-memory errors in your models
4. **Compare Memory Impact**: Compare memory usage before and after code changes
5. **Profile Large-Scale Training**: Analyze memory patterns in distributed training workloads

## Requirements

- Valid Memory Snapshot(s) generated from PyTorch
- Python 3.8+
- Additional dependencies specified in setup.py

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## Contributors

- [Shivam Raikundalia](https://github.com/sraikund16)
- [Basil Wong](https://github.com/basilwong)
- [Feng Tian](https://github.com/tianfengfrank)
- [Zizeng Meng](https://github.com/mzzchy)
- [Aaron Shi](https://github.com/aaronenyeshi)
- [Chuanhao Zhuge](https://github.com/chuanhaozhuge)


## License
Mosaic has a BSD-style license, as found in the [LICENSE](LICENSE) file.
