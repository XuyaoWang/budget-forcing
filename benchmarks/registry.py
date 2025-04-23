import logging
from typing import Dict, Type, Any

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkRegistry:
    """Registry for benchmark evaluators."""
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """
        Decorator to register a benchmark evaluator class.
        
        Args:
            name: Name to register the evaluator under (defaults to class name)
            
        Returns:
            Decorator function
        """
        def decorator(benchmark_class):
            registered_name = name or benchmark_class.__name__
            cls._registry[registered_name] = benchmark_class
            logger.debug(f"Registered benchmark evaluator: {registered_name}")
            return benchmark_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type:
        """
        Get a benchmark evaluator class by name.
        
        Args:
            name: Name of the benchmark evaluator
            
        Returns:
            Benchmark evaluator class
        
        Raises:
            KeyError: If the benchmark evaluator is not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Benchmark evaluator '{name}' not registered. Available evaluators: {list(cls._registry.keys())}")
        return cls._registry[name]
    
    @classmethod
    def list(cls) -> Dict[str, Type]:
        """
        List all registered benchmark evaluators.
        
        Returns:
            Dictionary mapping benchmark names to evaluator classes
        """
        return cls._registry.copy()
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """
        Create an instance of a benchmark evaluator.
        
        Args:
            name: Name of the benchmark evaluator
            **kwargs: Arguments to pass to the evaluator constructor
            
        Returns:
            Instance of the benchmark evaluator
        """
        evaluator_class = cls.get(name)
        return evaluator_class(**kwargs)