# Benchmark Integration Guide

This document guides developers on how to integrate new benchmarks into this evaluation framework.

## Integration Overview

To integrate a new benchmark, you need to subclass the abstract base class [`BenchmarkEvaluator`](../client/evaluation.py) and implement all of its abstract methods. The main steps are:

1. Create a new benchmark directory
2. Implement the evaluator class (subclassing `BenchmarkEvaluator`)
3. Register the new benchmark
4. Provide the necessary dataset and utility functions

## Example Directory Structure

```plaintext
benchmarks/
├── __init__.py
├── registry.py
└── my_benchmark/
    ├── __init__.py
    ├── eval.py        # Contains the evaluator implementation
    ├── data.json      # Dataset (if there's no official hf_link, place the prepared dataset here)
    └── utils.py       # Helper utility functions
```

## Abstract Method Descriptions

All abstract methods in the `BenchmarkEvaluator` class must be implemented in your subclass. Below are detailed descriptions for each method:

### 1. `benchmark_name`

```python
@property
@abstractmethod
def benchmark_name(self) -> str:
    """Return the name of the benchmark."""
    pass
```

**Purpose**: Defines the unique identifier for the benchmark, used for registration and result storage.

### 2. `load_dataset`

```python
@abstractmethod
def load_dataset(self, split: str = "test") -> Any:
    """Load the benchmark dataset."""
    pass
```

**Purpose**: Loads and preprocesses the dataset required for the benchmark.

### 3. `prepare_input_item`

```python
@abstractmethod
def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
    """Prepare the input for each data item."""
    pass
```

**Purpose**: Processes a single data instance from the dataset and organizes it into `system_content` and `user_content`.

### 4. `evaluate_item`

```python
@abstractmethod
def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
    """Evaluate the model's output for a single data item."""
    pass
```

**Purpose**: Assesses the model’s output on a single item to determine correctness or other evaluation metrics.

## Optional Methods to Override

In addition to the required abstract methods, you can optionally override the following methods to add custom functionality:

### 1. `calculate_metrics`

```python
def calculate_metrics(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics from detailed results."""
```

**Purpose**: Summarizes the results of all evaluated items and computes overall metrics (e.g., accuracy, F1 score).

### 2. `prepare_messages`

```python
def prepare_messages(self, system_content: str, user_content: str) -> List[Dict[str, Any]]:
    """Prepare the message format for the model API."""
```

**Purpose**: Converts system and user prompts into the message format required by the model API. Override if your benchmark requires a special messaging format (typically not needed).

## Registering a New Benchmark

Use the `BenchmarkRegistry.register` decorator to register your new benchmark:

```python
from benchmarks.registry import BenchmarkRegistry

@BenchmarkRegistry.register("my_benchmark")
class MyBenchmarkEvaluator(BenchmarkEvaluator):
    # Implement abstract methods
    ...
```

Also, you need to import the new evaluator class in `benchmarks/__init__.py` so the framework can automatically discover and register it:

```python
# __init__.py
from benchmarks.my_benchmark.eval import MyBenchmarkEvaluator
```

## Full Implementation Example

For a complete implementation example, please refer to [this example](./latent_jailbreak).

## Testing the New Benchmark

Once integration is complete, you can run the scripts in the `scripts` folder to test your benchmark. See [this script](../scripts/run_safety.sh) for reference.

## Notes

1. **Correctness**: Aim to keep the evaluation process consistent with the official implementation. If the official evaluation code is not open-sourced, refer to other open-source evaluation frameworks for guidance.
2. **Ray-Related Issues**: This framework uses Ray for parallel execution. If you encounter hanging processes, open a new terminal and run `ray stop`.
3. **Datasets**: If the dataset is officially available on HuggingFace, use the `hf_link` to load it directly instead of integrating it into the framework.
4. **Code Development**: Always develop benchmark code by subclassing and overriding methods within the benchmark’s subfolder. If you need to modify other parts of the framework, please discuss it in advance.
