"""
Benchmarks package initialization.

This module initializes the benchmarks package and handles registration of all evaluators.
"""

# First import the registry
from benchmarks.registry import BenchmarkRegistry

import benchmarks.mathvista
import benchmarks.mathvision  
import benchmarks.mmmu
import benchmarks.cv_bench
import benchmarks.latent_jailbreak