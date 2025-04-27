from typing import Dict, Any, List, Tuple
import logging
from datasets import load_dataset
import re

from client.evaluation import BenchmarkEvaluator
from benchmarks.registry import BenchmarkRegistry
from utils.common import normalize_response, normalize_extracted_answer, MULTILINGUAL_ANSWER_REGEXES, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE, ANSWER_PATTERN_MULTICHOICE

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@BenchmarkRegistry.register("cv_bench")
class CVBenchEvaluator(BenchmarkEvaluator):
    """Evaluator for the CVBench benchmark."""
    
    def __init__(self, 
                 model_name: str, 
                 data_path: str, 
                 results_dir: str = "./results"):
        """
        Initialize a CVBench evaluator.
        
        Args:
            model_name: Name of the model being evaluated
            data_path: Path to the CVBench dataset
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
        return "CVBench"
    
    def load_dataset(self, split: str = "test") -> Any:
        """
        Load the CVBench dataset.
        
        Args:
            split: Dataset split to load (e.g., "train", "test", "validation", "testmini")
            
        Returns:
            The loaded dataset
        """
        dataset = load_dataset(self.data_path)['test']
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
        user_content = item['prompt']
        image = item['image']
        
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
        filtered_response = filtered_response.strip()
        
        choices = item['choices']
        normalized_response = normalize_response(filtered_response)
        
        extracted_answer = None
        
        # Try to match option in parentheses format (A)-(F)
        match = re.search(r'(?i)\(([A-Fa-f])\)', normalized_response)
        if match:
            extracted_answer = f"({match.group(1).upper()})"
        else:
            # Check if the response has just the letter (A-F) with space around it
            match = re.search(r'(?i)(?:^|\s)([A-Fa-f])(?:\s|$)', normalized_response)
            if match:
                extracted_answer = f"({match.group(1).upper()})"
            else:
                # Try to match from multilingual patterns
                for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                    regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                    match = re.search(regex, normalized_response)
                    if match:
                        extracted_letter = normalize_extracted_answer(match.group(1))
                        extracted_answer = f"({extracted_letter.upper()})"
                        break
                        
                if not extracted_answer:
                    match = re.search(ANSWER_PATTERN_MULTICHOICE, normalized_response)
                    if match:
                        extracted_letter = match.group(1).strip().upper()
                        extracted_answer = f"({extracted_letter})"
        
        correct_answer = item['answer']
        
        if extracted_answer:
            score = 1.0 if extracted_answer.upper() == correct_answer.upper() else 0.0
        else:
            # Check if response contains the text of one of the answer choices
            score = 0.0
            correct_option_idx = ord(correct_answer.strip('()').upper()) - ord('A')
            if 0 <= correct_option_idx < len(choices):
                correct_choice_text = choices[correct_option_idx]
                # Check if correct choice text is in the response
                if correct_choice_text.lower() in normalized_response.lower():
                    score = 1.0
                    extracted_answer = correct_answer
        
        # Return detailed results
        return {
            'question_id': item['idx'],
            'question': item['prompt'],
            'source': item['source'],
            'correct': score > 0,
            'correct_answer': item['answer'],
            'response': response,
            'extracted_answer': extracted_answer,
            'normalized_response': normalized_response
        }
    
    def calculate_metrics(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics based on detailed evaluation results.
        
        Args:
            detailed_results: List of detailed evaluation results for each item
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Initialize counters
        counts = {'ade': [0, 0], 'coco': [0, 0], 'omni': [0, 0]}  # [total, correct]
        
        # Map for source name normalization
        source_map = {
            'ADE20K': 'ade',
            'COCO': 'coco',
            'Omni3D': 'omni'
        }
        
        # Count results
        for result in detailed_results:
            source = result['source']
            # Normalize source name
            normalized_source = source_map.get(source, source.lower())
            
            if normalized_source in counts:
                counts[normalized_source][0] += 1
                counts[normalized_source][1] += int(result['correct'])
        
        # Calculate accuracies
        acc = {src: counts[src][1]/counts[src][0] if counts[src][0] > 0 else 0 for src in counts}
        
        # Calculate combined metrics
        acc_2d = (acc['ade'] + acc['coco']) / 2
        acc_3d = acc['omni']
        
        return {
            'ade_accuracy': acc['ade'],
            'coco_accuracy': acc['coco'],
            'omni_accuracy': acc['omni'],
            'accuracy_2d': acc_2d,
            'accuracy_3d': acc_3d,
            'accuracy': (acc_2d + acc_3d) / 2,
            'detailed_results': detailed_results
        }
    