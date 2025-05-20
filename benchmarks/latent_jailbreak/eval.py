import os
from typing import Dict, Any, List, Tuple, Optional
import logging
from datasets import load_dataset

from client.evaluation import BenchmarkEvaluator
from benchmarks.registry import BenchmarkRegistry
from utils.cached_requests import cached_requests
from benchmarks.latent_jailbreak.utils import (
    parse_json,
    EVALUATE_PROMPT
)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def gpt_evaluate(response: str, 
                   item: Dict[str, Any], 
                   model: str = "gpt-4o", 
                   api_key: Optional[str] = None, 
                   api_base: Optional[str] = None) -> str:
    """
    Extract an answer from a model response for a specific item.
        
    Args:
        response: Model's response
        item: Dataset item
        model: Name of the extractor model (e.g., "gpt-4o-mini")
        api_key: API key for the extractor model
        api_base: Base URL for the extractor model API
            
    Returns:
        Extracted answer
    """

    api_key = os.getenv("API_KEY")
    api_base = os.getenv("API_BASE")

    user_prompt = EVALUATE_PROMPT.format(category=item['category'], response=response)
    
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    extraction = cached_requests(
        messages=messages,
        model=model, 
        max_completion_tokens=1024,
        temperature=0.0,
        api_key=api_key,
        api_base=api_base
    )
    with open("extraction.txt", "w+") as f:
        f.write(extraction + "\n======\n")
    return extraction

@BenchmarkRegistry.register("latent_jailbreak")
class LatenJailbreakEvaluator(BenchmarkEvaluator):
    """Evaluator for the LatenJailbreak benchmark."""
    
    def __init__(self, model_name: str, results_dir: str = "./results", **kwargs):
        super().__init__(model_name, results_dir, **kwargs)
        self.data_path = kwargs.get("data_path", "benchmarks/latent_jailbreak/")

    @property
    def benchmark_name(self) -> str:
        """Return the name of the benchmark."""
        return "latent_jailbreak"

    def load_dataset(self, split: str = "train") -> Any:
        """
        Load the LatenJailbreak dataset.
        """
        dataset = load_dataset("json", data_dir=self.data_path, data_files="data.json", split=split)
        return dataset

    def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Prepare input for a item from the dataset.
        """
        system_content = ""
        user_content = item['prompt']

        return system_content, user_content
    
    def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a model output against an item from the dataset.
        """
        evaluate_output = gpt_evaluate(
            response, 
            item,
        )
        try:
            eval_data = parse_json(evaluate_output)
        except Exception as e:
            return {
                "category": item['category'],
                "prompt": item['prompt'],
                "response": response,
                "is_safe": None,
                "is_following": None,
            }

        safety = eval_data['safety'].lower() == 'yes'
        following = eval_data['following'].lower() == 'yes'

        return {
            "category": item['category'],
            "prompt": item['prompt'],
            "response": response,
            "is_safe": safety,
            "is_following": following,
        }

    def calculate_metrics(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics based on detailed evaluation results.
        """
        safety_results = []
        following_results = []
        for result in detailed_results:
            if not result:
                continue
            if result['is_safe'] is not None:
                safety_results.append(result['is_safe'])
            if result['is_following'] is not None:
                following_results.append(result['is_following'])

        return {
            "safety_rate": sum(safety_results) / len(safety_results),
            "following_rate": sum(following_results) / len(following_results)
        }
