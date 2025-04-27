import os
import json
import time
import logging
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from utils.parallel_processing import parallel_processing_backend
from utils.visualization import plot_accuracy_and_length, plot_detailed_metrics, format_results_for_plotting

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators."""
    
    def __init__(self, model_name: str, results_dir: str = "./results"):
        """
        Initialize a benchmark evaluator.
        
        Args:
            model_name: Name of the model being evaluated
            results_dir: Directory to save results
        """
        self.model_name = model_name
        self.results_dir = os.path.join(results_dir, self.benchmark_name, model_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Return the name of the benchmark."""
        pass
    
    @abstractmethod
    def load_dataset(self, split: str = "test") -> Any:
        """
        Load the benchmark dataset.
        
        Args:
            split: Dataset split to load (e.g., "train", "test", "validation")
            
        Returns:
            The loaded dataset
        """
        pass
    
    @abstractmethod
    def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Prepare input for a single item from the dataset.
        
        Args:
            item: A single item from the dataset
            
        Returns:
            Tuple of (system_content, user_content, image)
        """
        pass
    
    @abstractmethod
    def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a single model output against an item from the dataset.
        
        Args:
            item: A single item from the dataset
            response: Model's response for this item
            
        Returns:
            Evaluation results for this item
        """
        pass
    
    @staticmethod
    def encode_image(image: Union[str, Image.Image]) -> str:
        """
        Encode an image as a base64 data URL.
        
        Args:
            image: Path to an image or a PIL Image object
            
        Returns:
            Base64 encoded image as a data URL
        """
        if isinstance(image, str):
            image_input = Image.open(image)
        else:
            image_input = image
        
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        buffer = BytesIO()
        image_input.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        base64_data = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"
    
    def prepare_messages(self, system_content: str, user_content: str, image: Any) -> List[Dict[str, Any]]:
        """
        Prepare messages from the tuple of (system_content, user_content, image).
        This is the default implementation that can be overridden by derived classes.
        
        Args:
            system_content: System prompt content
            user_content: User prompt content
            image: Image or list of images
            
        Returns:
            List of message dictionaries ready for the API
        """
        # Process images
        images = []
        if image is not None:
            if isinstance(image, list):
                images = [self.encode_image(img) for img in image]
            else:
                images = [self.encode_image(image)]
        
        # Construct the content array for the user message
        content = [{"type": "image_url", "image_url": {"url": img}} for img in images]
        content.append({"type": "text", "text": user_content})
        
        # Build the messages list
        messages = [{'role': 'user', 'content': content}]
        if system_content:
            messages.insert(0, {'role': 'system', 'content': system_content})
            
        return messages
    
    def _prepare_input_item_wrapper(self, param: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Wrapper function for prepare_input_item to use with parallel processing.
        
        Args:
            param: Dictionary containing the item
            
        Returns:
            Tuple of (system_content, user_content, image)
        """
        return self.prepare_input_item(param['item'])
    
    def prepare_inputs(self, dataset: Any, num_workers: int = 100) -> Tuple[List[str], List[str], List[Any]]:
        """
        Prepare inputs for the budget forcing client using parallel processing.
        
        Args:
            dataset: The loaded dataset
            num_workers: Number of parallel workers
            
        Returns:
            Tuple of (system_contents, user_contents, images)
        """
        # Prepare parameter list for parallel processing
        params = [{'item': item} for item in dataset]
        
        # Run parallel processing
        logger.info(f"Preparing inputs for {len(params)} examples using {num_workers} workers")
        results = parallel_processing_backend(
            params=params,
            fn=self._prepare_input_item_wrapper,
            num_workers=num_workers,
            desc="Preparing Inputs"
        )
        
        # Unpack results
        system_contents = []
        user_contents = []
        images = []
        
        for system_content, user_content, image in results:
            system_contents.append(system_content)
            user_contents.append(user_content)
            images.append(image)
            
        return system_contents, user_contents, images
    
    def _prepare_messages_wrapper(self, param: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Wrapper function for prepare_messages to use with parallel processing.
        
        Args:
            param: Dictionary containing system_content, user_content, and image
            
        Returns:
            List of message dictionaries
        """
        return self.prepare_messages(
            param['system_content'], 
            param['user_content'], 
            param['image']
        )
    
    def prepare_all_messages(self, 
                            system_contents: List[str], 
                            user_contents: List[str], 
                            images: List[Any],
                            num_workers: int = 100) -> List[List[Dict[str, Any]]]:
        """
        Prepare all messages for the budget forcing client using parallel processing.
        
        Args:
            system_contents: List of system prompts
            user_contents: List of user prompts
            images: List of images
            num_workers: Number of parallel workers
            
        Returns:
            List of message lists ready for the API
        """
        # Prepare parameter list for parallel processing
        params = [
            {
                'system_content': system_content,
                'user_content': user_content,
                'image': image
            } 
            for system_content, user_content, image in zip(system_contents, user_contents, images)
        ]
        
        # Run parallel processing
        logger.info(f"Preparing messages for {len(params)} examples using {num_workers} workers")
        results = parallel_processing_backend(
            params=params,
            fn=self._prepare_messages_wrapper,
            num_workers=num_workers,
            desc="Preparing Messages"
        )
            
        return results
    
    def _evaluate_item_wrapper(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper function for evaluate_item to use with parallel processing.
        
        Args:
            param: Dictionary containing the item and response
            
        Returns:
            Evaluation results for this item
        """
        return self.evaluate_item(param['item'], param['response'])
    
    def evaluate_outputs(self, 
                         dataset: Any, 
                         responses: List[str], 
                         num_workers: int = 100) -> Dict[str, Any]:
        """
        Evaluate model outputs against the benchmark using parallel processing.
        
        Args:
            dataset: The loaded dataset
            responses: Model responses from budget forcing
            num_workers: Number of parallel workers
            
        Returns:
            Dictionary with evaluation results
        """
        num_total = len(responses)
        
        # Prepare parameter list for parallel processing
        params = [{'item': item, 'response': response} for item, response in zip(dataset, responses)]
        
        # Run parallel processing
        logger.info(f"Evaluating {len(params)} examples using {num_workers} workers")
        detailed_results = parallel_processing_backend(
            params=params,
            fn=self._evaluate_item_wrapper,
            num_workers=num_workers,
            desc="Evaluating Outputs"
        )
        
        # Calculate metrics using the potentially overridden method
        return self.calculate_metrics(detailed_results)
    
    def calculate_metrics(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics based on detailed evaluation results.
        
        This method can be overridden by derived classes to implement custom metrics.
        
        Args:
            detailed_results: List of detailed evaluation results for each item
            
        Returns:
            Dictionary with evaluation metrics
        """
        num_total = len(detailed_results)

        # Count correct answers
        num_match = sum(1 for result in detailed_results if result.get('correct', False))
        accuracy = num_match / num_total if num_total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "num_correct": num_match,
            "num_total": num_total,
            "detailed_results": detailed_results
        }
    
    def run_evaluation(self, 
                     budget_forcing_client,
                     num_ignore: int = 1,
                     temperature: float = 0.3,
                     top_p: float = 0.9,
                     repetition_penalty: float = 1.05,
                     max_tokens_thinking: int = 32000,
                     api_key: Optional[str] = None,
                     split: str = "test",
                     num_workers: int = 100,
                     visualize: bool = True,
                     tokenizer = None) -> Dict[str, Any]:
        """
        Run an evaluation on the benchmark.
        
        Args:
            budget_forcing_client: BudgetForcingClient instance
            num_ignore: Number of times to ignore the stop token
            temperature: Temperature parameter for sampling
            top_p: Top-p parameter for sampling
            repetition_penalty: Repetition penalty parameter
            max_tokens_thinking: Maximum tokens for thinking
            api_key: API key (if needed)
            split: Dataset split to evaluate on
            num_workers: Number of parallel workers for input preparation and evaluation
            visualize: Whether to generate visualization plots
            tokenizer: Optional tokenizer for calculating token statistics
            
        Returns:
            Dictionary with evaluation results
        """
        # Load dataset
        dataset = self.load_dataset(split)
        
        # Prepare inputs using parallel processing
        system_contents, user_contents, images = self.prepare_inputs(dataset, num_workers=num_workers)
        
        # Prepare messages for the API
        messages_list = self.prepare_all_messages(
            system_contents, 
            user_contents, 
            images,
            num_workers=num_workers
        )
        
        # Run budget forcing
        logger.info(f"Running budget forcing on {len(messages_list)} examples with num_ignore={num_ignore}")
        responses = budget_forcing_client.run(
            messages_list=messages_list,
            max_tokens_thinking=max_tokens_thinking,
            num_ignore=num_ignore,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        # Process results for each num_ignore value
        num_ignore2responses = {}
        for i in range(num_ignore):
            num_ignore2responses[str(i)] = []
            for num_ignore2response in responses:
                if str(i) in num_ignore2response:
                    num_ignore2responses[str(i)].append(num_ignore2response[str(i)])
                else:
                    # Handle case where some responses might not have all ignore values
                    last_ignore = max(int(k) for k in num_ignore2response.keys())
                    num_ignore2responses[str(i)].append(num_ignore2response[str(last_ignore)])
        
        # Evaluate outputs for each num_ignore value using parallel processing
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(
            self.results_dir, 
            f"{timestamp}_temp_{str(temperature).replace('.', '_')}_top_p_{str(top_p).replace('.', '_')}_rep_penalty_{str(repetition_penalty).replace('.', '_')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        results = {}
        for ignore_idx, ignore_responses in num_ignore2responses.items():
            logger.info(f"Evaluating outputs for num_ignore={ignore_idx}")
            eval_results = self.evaluate_outputs(
                dataset, 
                ignore_responses, 
                num_workers=num_workers
            )
            results[ignore_idx] = eval_results
            
            result_file = os.path.join(
                self.results_dir, 
                f"num_ignore_{ignore_idx}.json"
            )
            
            # Prepare results dictionary with metadata
            full_results = {
                "hyperparameters": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "num_ignore": int(ignore_idx),
                    "max_tokens_thinking": max_tokens_thinking,
                    "num_workers": num_workers,
                },
                "timestamp": timestamp,
                "benchmark": self.benchmark_name,
                "model": self.model_name,
                "split": split,
                **eval_results
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Evaluation complete for num_ignore={ignore_idx}, accuracy={eval_results.get('accuracy', 0):.4f}")
            logger.info(f"Results saved to {result_file}")
            
        # Generate visualizations if requested
        if visualize:
            try:
                logger.info("Generating visualization plots")
                # Format results for plotting
                plot_data = format_results_for_plotting(results, tokenizer)
                
                # Generate plots
                plot_accuracy_and_length(self.model_name, plot_data, self.results_dir)
                plot_detailed_metrics(self.model_name, plot_data, self.results_dir)
                
                logger.info(f"Visualization plots saved to {self.results_dir}")
            except Exception as e:
                logger.error(f"Error generating visualization plots: {str(e)}")
        
        # Calculate summary statistics
        try:
            logger.info("Calculating summary statistics")
            summary_stats = self._calculate_summary_statistics(results, tokenizer)
            
            # Save summary statistics
            summary_file = os.path.join(self.results_dir, "summary_statistics.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Summary statistics saved to {summary_file}")
            
            # Add summary statistics to results
            results["summary_statistics"] = summary_stats
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
                
        return results
    
    def _calculate_summary_statistics(self, results: Dict[str, Any], tokenizer=None) -> Dict[str, Any]:
        """
        Calculate summary statistics for the evaluation results.
        
        Args:
            results: Dictionary with evaluation results for each num_ignore value
            tokenizer: Optional tokenizer for calculating token statistics
            
        Returns:
            Dictionary with summary statistics
        """
        
        # Extract num_ignore values and convert to integers for sorting
        ignore_values = sorted([int(k) for k in results.keys()])
        
        # 1. Accuracy list as a function of num_ignore
        accuracy_list = [results[str(i)]["accuracy"] for i in ignore_values]
        
        # 2. Calculate token statistics if tokenizer is available
        token_stats = {"avg_tokens": [], "token_variance": []}
        if tokenizer:
            for i in ignore_values:
                responses = [item.get("response", "") for item in results[str(i)].get("detailed_results", [])]
                if responses:
                    # Calculate token counts for each response
                    token_counts = [len(tokenizer(resp)["input_ids"]) for resp in responses]
                    token_stats["avg_tokens"].append(np.mean(token_counts))
                    token_stats["token_variance"].append(np.var(token_counts))
                else:
                    token_stats["avg_tokens"].append(0)
                    token_stats["token_variance"].append(0)
        
        # 3. Calculate slope of accuracy vs. num_ignore using least squares
        slope = 0
        if len(ignore_values) > 1:
            # Use linear regression to find the slope
            slope, _, _, _, _ = stats.linregress(ignore_values, accuracy_list)
        
        # 4. Calculate difference between max accuracy and accuracy at num_ignore=0
        max_acc = max(accuracy_list)
        baseline_acc = accuracy_list[0] if ignore_values and ignore_values[0] == 0 else 0
        max_acc_diff = max_acc - baseline_acc
        
        # Create summary dictionary
        summary = {
            "accuracy_by_num_ignore": dict(zip([str(i) for i in ignore_values], accuracy_list)),
            "token_stats_by_num_ignore": {
                "avg_tokens": dict(zip([str(i) for i in ignore_values], token_stats["avg_tokens"])),
                "token_variance": dict(zip([str(i) for i in ignore_values], token_stats["token_variance"]))
            },
            "accuracy_slope": float(slope),
            "max_accuracy": float(max_acc),
            "baseline_accuracy": float(baseline_acc),
            "max_accuracy_improvement": float(max_acc_diff),
            "max_accuracy_at_num_ignore": str(ignore_values[accuracy_list.index(max_acc)])
        }
        logger.info(f"Accuracy Slope: {summary['accuracy_slope']}")
        logger.info(f"Max Accuracy: {summary['max_accuracy']}")
        logger.info(f"Baseline Accuracy: {summary['baseline_accuracy']}")
        logger.info(f"Max Accuracy Improvement: {summary['max_accuracy_improvement']}")
        logger.info(f"Max Accuracy at num_ignore: {summary['max_accuracy_at_num_ignore']}")
        
        return summary