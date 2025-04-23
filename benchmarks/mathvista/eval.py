from typing import Dict, Any, List, Tuple, Optional
import logging
from datasets import load_dataset

from client.evaluation import BenchmarkEvaluator
from benchmarks.registry import BenchmarkRegistry
from utils.cached_requests import cached_requests
from benchmarks.mathvista.utils import (
    normalize_extracted_answer,
    safe_equal,
    DEMO_PROMPT
)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_answer(response: str, 
                   item: Dict[str, Any], 
                   model: str, 
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
        
    user_prompt = f"{DEMO_PROMPT}\n\n{item['query']}\n\n{response}\n\nExtracted answer: "
    
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
    
    return extraction

@BenchmarkRegistry.register("mathvista")
class MathVistaEvaluator(BenchmarkEvaluator):
    """Evaluator for the MathVista benchmark."""
    
    def __init__(self, 
                 model_name: str, 
                 data_path: str, 
                 results_dir: str = "./results"):
        """
        Initialize a MathVista evaluator.
        
        Args:
            model_name: Name of the model being evaluated
            data_path: Path to the MathVista dataset
            results_dir: Directory to save results
        """
        super().__init__(model_name, results_dir)
        self.data_path = data_path
        self.gpt4_model = 'gpt-4o-mini'
        self.gpt4_api_key = 'YOUR_API_KEY'  # Replace with environment variable in production
        self.gpt4_api_base = 'https://api.61798.cn/v1/chat/completions'
    
    @property
    def benchmark_name(self) -> str:
        """Return the name of the benchmark."""
        return "MathVista"
    
    def load_dataset(self, split: str = "test") -> Any:
        """
        Load the MathVista dataset.
        
        Args:
            split: Dataset split to load (e.g., "train", "test", "validation", "testmini")
            
        Returns:
            The loaded dataset
        """
        dataset = load_dataset(self.data_path)[split]
        return dataset
    
    def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Prepare input for a item from the dataset.
        
        Args:
            item: A item from the dataset
            
        Returns:
            Tuple of (system_content, user_content, image)
        """
        system_content = ""
        user_content = item['query']
        image = item['decoded_image']
        
        return system_content, user_content, image
    
    def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a model output against an item from the dataset.
        
        Args:
            item: A item from the dataset
            response: Model's response for this item
            
        Returns:
            Evaluation results for this item
        """
        # Process response to get only the post-thinking part
        filtered_response = response.split("</think>")[1] if "</think>" in response else response
        filtered_response = filtered_response.split("Final answer:")[1] if "Final answer:" in filtered_response else filtered_response
        
        # Extract answer using the extractor model
        extraction = extract_answer(
            filtered_response, 
            item,
            model=self.gpt4_model,
            api_key=self.gpt4_api_key,
            api_base=self.gpt4_api_base
        )
        
        # Normalize the extracted answer according to the problem type
        prediction = normalize_extracted_answer(
            extraction,
            item['choices'],
            item['question_type'],
            item['answer_type'],
            item['precision'],
        )
        
        # Check if the answer is correct
        correct = safe_equal(prediction, item['answer'])
        
        # Return detailed results
        return {
            "query": item['query'],
            "response": response,
            "choices": item['choices'],
            "answer": item['answer'],
            "extraction": extraction,
            "prediction": prediction,
            "correct": correct,
        }