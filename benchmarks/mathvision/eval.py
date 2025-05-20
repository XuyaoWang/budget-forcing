import re
from typing import Dict, Any, List, Tuple
from datasets import load_dataset

from client.evaluation import BenchmarkEvaluator
from benchmarks.registry import BenchmarkRegistry
from benchmarks.mathvision.utils import find_math_answer, is_equal, is_number

@BenchmarkRegistry.register("mathvision")
class MathVisionEvaluator(BenchmarkEvaluator):
    """Evaluator for the MathVision benchmark."""

    # def __init__(self, model_name: str, data_path: str, results_dir: str = "./results"):
    #     """
    #     Initialize a MathVision evaluator.

    #     Args:
    #         model_name: Name of the model being evaluated
    #         data_path: Path to the MathVision dataset
    #         results_dir: Directory to save results
    #     """
    #     super().__init__(model_name, results_dir)
    #     self.data_path = data_path

    @property
    def benchmark_name(self) -> str:
        """Return the name of the benchmark."""
        return "MathVision"

    def load_dataset(self, split: str = "test") -> Any:
        """
        Load the MathVision dataset.

        Args:
            split: Dataset split to load (e.g., "train", "test", "validation")

        Returns:
            The loaded dataset
        """
        dataset = load_dataset(self.data_path)['test'].select(range(20))
        return dataset

    def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Prepare input for a single item from the dataset.

        Args:
            item: A single item from the dataset

        Returns:
            Tuple of (system_content, user_content, image)
        """
        # Remove image tags from question
        question = re.sub(r'<image\s*\d+\s*>', '', item['question'])

        # Format options if available
        options = ''
        if len(item['options']) > 0:
            assert len(item['options']) == 5, item
            if ''.join(item['options']) != 'ABCDE':
                options = f"(A) {item['options'][0]}\n(B) {item['options'][1]}\n(C) {item['options'][2]}\n(D) {item['options'][3]}\n(E) {item['options'][4]}\n"

        # Create the input prompt
        user_content = ('Please solve the problem step by step and put your answer in one "\\boxed{}". '
                       'If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'
                       f"{question}\n{options}")

        system_content = ""
        image = item['decoded_image']

        return system_content, user_content, image

    def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a single model output against an item from the dataset.

        Args:
            item: A single item from the dataset
            response: Model's response for this item

        Returns:
            Evaluation results for this item
        """
        # Get the prediction from the response
        filtered_response = response.split("</think>")[1] if "</think>" in response else response
        filtered_response = filtered_response.split("Final answer:")[1] if "Final answer:" in filtered_response else filtered_response
        filtered_response = filtered_response.strip()

        # Extract the answer from the response
        gt_answer = str(item['answer'])
        if len(item['options']) > 0:
            gt_answer_value = item['options'][ord(gt_answer)-ord('A')]
        else:
            gt_answer_value = ''

        # Process the response to extract the answer
        processed_response = filtered_response

        # Check for multiple choice format
        for c in 'ABCDE':
            if filtered_response.endswith(f" {c}.") or filtered_response.endswith(f" ({c}).") or filtered_response.startswith(f"{c}\n") or filtered_response.startswith(f"({c})\n") or filtered_response.startswith(f"({c}) {c}\n"):
                processed_response = c

        # Check for numeric format (e.g., "The answer is 42")
        if is_number(filtered_response.split('is ')[-1].rstrip('.')):
            processed_response = filtered_response.split('is ')[-1].rstrip('.')

        # Look for LaTeX boxed answers
        if 'oxed{' not in processed_response:
            for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be', 'Final answer:']:
                raw_response = processed_response
                processed_response = processed_response.split(flag)[-1].strip()
                if flag in raw_response:
                    processed_response = processed_response.split('\n')[0].split('. ')[0]
                flag = flag.replace('the', 'The')
                raw_response = processed_response
                processed_response = processed_response.split(flag)[-1].strip()
                if flag in raw_response:
                    processed_response = processed_response.split('\n')[0].split('. ')[0]
        elif processed_response.count('oxed{') > 1:
            processed_response = '\\boxed{' + processed_response.split('oxed{')[-1]

        # Extract the final answer using a helper function
        processed_response = find_math_answer(processed_response)
        processed_response = processed_response.replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e')
        processed_response = processed_response.replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e')
        processed_response = processed_response.rstrip('.').lstrip(':').strip()

        # Check if the answer is correct
        correct = is_equal(gt_answer, processed_response) or is_equal(gt_answer_value, processed_response)

        # Return detailed results
        return {
            "question": item['question'],
            "response": response,
            "options": item['options'],
            "post_processed_response": processed_response,
            "gt_answer": gt_answer,
            "gt_answer_value": gt_answer_value,
            "correct": correct,
        }
