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

    def __init__(self, model_name: str, results_dir: str = "./results", **kwargs):
        """
        Initialize a benchmark evaluator.

        Args:
            model_name: Name of the model being evaluated
            results_dir: Directory to save results
        """
        self.model_name = model_name
        self.data_path = kwargs.get("data_path", None)
        if self.data_path is None:
            raise ValueError("data_path must be provided")
        self.results_dir_base = os.path.join(results_dir, self.benchmark_name, model_name)
        os.makedirs(self.results_dir_base, exist_ok=True)

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

    def prepare_messages(self, system_content: str, user_content: str) -> List[Dict[str, Any]]:
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
        # images = []
        # if image is not None:
        #     if isinstance(image, list):
        #         images = [self.encode_image(img) for img in image]
        #     else:
        #         images = [self.encode_image(image)]

        # Construct the content array for the user message
        # content = [{"type": "image_url", "image_url": {"url": img}} for img in images]
        content = [{"type": "text", "text": user_content}]

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

        for system_content, user_content in results:
            system_contents.append(system_content)
            user_contents.append(user_content)

        return system_contents, user_contents

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
        )

    def prepare_all_messages(self,
                            system_contents: List[str],
                            user_contents: List[str],
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
                'user_content': user_content
            }
            for system_content, user_content in zip(system_contents, user_contents)
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

    def run_evaluation(
        self,
        budget_forcing_client,
        #  num_ignore: int = 1,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        max_tokens: int = 32000,
        api_key: Optional[str] = None,
        split: str = "test",
        num_workers: int = 100,
        #  visualize: bool = True,
        #  tokenizer = None
    ) -> Dict[str, Any]:
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
        logger.info(
            f"Loading dataset for {self.benchmark_name} split '{split}'"
        )
        dataset = self.load_dataset(split)

        # Prepare inputs using parallel processing
        system_contents, user_contents= self.prepare_inputs(
            dataset, num_workers=num_workers
        )

        # Prepare messages for the API
        messages_list = self.prepare_all_messages(
            system_contents,
            user_contents,
            num_workers=num_workers
        )

        # Run budget forcing
        logger.info(f"Running inference on {len(messages_list)} examples")
        responses = budget_forcing_client.run(
            messages_list=messages_list,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

        # # Process results for each num_ignore value
        # num_ignore2responses = {}
        # for i in range(num_ignore):
        #     num_ignore2responses[str(i)] = []
        #     for num_ignore2response in responses:
        #         if str(i) in num_ignore2response:
        #             num_ignore2responses[str(i)].append(num_ignore2response[str(i)])
        #         else:
        #             # Handle case where some responses might not have all ignore values
        #             last_ignore = max(int(k) for k in num_ignore2response.keys())
        #             num_ignore2responses[str(i)].append(num_ignore2response[str(last_ignore)])

        # Evaluate outputs for each num_ignore value using parallel processing
        logger.info(f"Evaluating {len(responses)} outputs")
        eval_results = self.evaluate_outputs(
            dataset, responses, num_workers=num_workers
        )
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        current_results_dir = os.path.join(
            self.results_dir_base,
            (
                f"{timestamp}_temp_{str(temperature).replace('.', '_')}"
                f"_top_p_{str(top_p).replace('.', '_')}"
                f"_rep_penalty_{str(repetition_penalty).replace('.', '_')}"
            ),
        )
        os.makedirs(current_results_dir, exist_ok=True)

        result_file = os.path.join(current_results_dir, f"result.json")

        # Consolidate results into a single dictionary
        full_results = {
                "hyperparameters": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                "max_tokens": max_tokens,
                    "num_workers": num_workers,
                },
                "timestamp": timestamp,
                "benchmark": self.benchmark_name,
                "model": self.model_name,
                "split": split,
            **eval_results,
        }

        logger.info(f"Saving full results to {result_file}")
        with open(result_file, "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=4, ensure_ascii=False)

        logger.info(
            f"Evaluation complete for {self.benchmark_name} on model {self.model_name}"
        )
        logger.info(f"Accuracy: {eval_results.get('accuracy', 'N/A')}")

        return full_results
