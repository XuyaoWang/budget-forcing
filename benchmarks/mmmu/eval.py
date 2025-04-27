from typing import Dict, Any, List, Tuple, Optional

import re
from datasets import load_dataset, concatenate_datasets

from client.evaluation import BenchmarkEvaluator
from benchmarks.registry import BenchmarkRegistry
from benchmarks.mmmu.utils import (
    parse_multi_choice_response, 
    parse_open_response,
    eval_multi_choice,
    eval_open,
    CAT_SHORT2LONG
)

@BenchmarkRegistry.register("mmmu")
class MMMUEvaluator(BenchmarkEvaluator):
    """Evaluator for the MMMU benchmark."""
    
    def __init__(self, model_name: str, data_path: str, results_dir: str = "./results"):
        """
        Initialize a MMMU evaluator.
        
        Args:
            model_name: Name of the model being evaluated
            data_path: Path to the MMMU dataset
            results_dir: Directory to save results
        """
        super().__init__(model_name, results_dir)
        self.data_path = data_path
    
    @property
    def benchmark_name(self) -> str:
        """Return the name of the benchmark."""
        return "MMMU"
    
    def load_dataset(self, split: str = "test") -> Any:
        """
        Load the MMMU dataset.
        
        Args:
            split: Dataset split to load (e.g., "train", "test", "validation")
            
        Returns:
            The loaded dataset
        """
        categories = list(CAT_SHORT2LONG.values())
        return concatenate_datasets([
            (lambda d: d.add_column('category', [category] * len(d)))(
                load_dataset(self.data_path, category, split='validation')
            )
            for category in categories
        ])
    
    def get_image_indice(self, text: str)->List[int]:
        pattern = r'<image (\d+)>'
        matches = re.findall(pattern, text)
        return [int(num) for num in matches]
    
    def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Prepare input for an item from the dataset.
        
        Args:
            item: An item from the dataset
            
        Returns:
            Tuple of (system_content, user_content, image)
        """

        system_content = ""
        question = item['question']
        if item['question_type'] == 'multiple-choice':
            options = eval(item['options'])
            example = ""
            letter_to_option = {}
                
            for idx, option in enumerate(options):
                option_letter = chr(ord('A') + idx)
                example += f"({option_letter}) {option}\n"
                letter_to_option[option_letter] = option
                
            user_content = f"{question}\n\n{example}\n\nAnswer with the option's letter from the given choices directly."
        else:
            user_content = f"{question}\n\nAnswer the question using a single word or phrase."
            
        image_ids = self.get_image_indice(user_content)
        image = [item[f'image_{id}'] for id in image_ids]

        return system_content, user_content, image
    
    def prepare_messages(self, system_content: str, user_content: str, image: Any) -> List[Dict[str, Any]]:
        """
        Prepare messages for the API.
        
        Args:
            system_content: System prompt content
            user_content: User prompt content
            image: Image or list of images

        Returns:
            List of messages ready for the API
        """
        content_parts = []
        matches = list(re.finditer(r'<image\s*(\d*)>', user_content))
        images = image if isinstance(image, list) else [image]
        
        if matches:
            assert len(images) == len(matches), f"Number of images ({len(images)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_content}"
            
            last_end = 0
            for i, match in enumerate(matches):
                if match.start() > last_end:
                    content_parts.append({"type": "text", "text": user_content[last_end:match.start()]})
                content_parts.append({"type": "image_url", "image_url": {"url": self.encode_image(images[i])}})
                last_end = match.end()
                
            if last_end < len(user_content):
                content_parts.append({"type": "text", "text": user_content[last_end:]})
        else:
            content_parts.extend([{"type": "image_url", "image_url": {"url": self.encode_image(img)}} for img in images])
            if user_content:
                content_parts.append({"type": "text", "text": user_content})

        messages = [{"role": "user", "content": content_parts}]
        if system_content:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_content}]})

        return messages
    
    def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a model output against an item from the dataset.
        
        Args:
            item: An item from the dataset
            response: Model's response for this item
            
        Returns:
            Evaluation results for this item
        """
        # Process response to get only the post-thinking part
        filtered_response = response.split("</think>")[1] if "</think>" in response else response
        filtered_response = filtered_response.split("Final answer:")[1] if "Final answer:" in filtered_response else filtered_response
        filtered_response = filtered_response.strip()
        
        result = {
            "id": item.get('id', ''),
            "category": item['category'],
            "question": item['question'],
            "response": response,
            "filtered_response": filtered_response
        }
        
        # Process based on question type
        if item['question_type'] == 'multiple-choice':
            options = eval(item['options'])

            all_choices = [chr(ord('A') + i) for i in range(len(options))]
            index2ans = {chr(ord('A') + idx): option for idx, option in enumerate(options)}
            
            # Parse the prediction from the response
            parsed_pred = parse_multi_choice_response(filtered_response, all_choices, index2ans)
            result["parsed_pred"] = parsed_pred
            result["all_choices"] = all_choices
            result["index2ans"] = index2ans
            result["answer"] = item['answer']
            
            # Evaluate the prediction
            correct = eval_multi_choice(item['answer'], parsed_pred)
            result["correct"] = correct
            
        else:  # open-ended question
            # Parse the prediction from the response
            parsed_pred = parse_open_response(filtered_response)
            result["parsed_pred"] = parsed_pred
            result["all_choices"] = []
            result["index2ans"] = {}
            result["answer"] = item['answer']
            
            # Evaluate the prediction
            correct = eval_open(item['answer'], parsed_pred)
            result["correct"] = correct
        
        return result
