from __future__ import annotations
import logging
from typing import Any, List, Union, Dict, Optional

import copy
from transformers import AutoTokenizer
from utils.parallel_processing import parallel_processing_backend
from utils.cached_requests import cached_requests

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BudgetForcingClient:
    """Client for budget forcing algorithm using parallel processing."""
    
    def __init__(self, 
                 api_base: str,
                 model: str,
                 api_key: str = "EMPTY",
                 tokenizer_path: Optional[str] = None,
                 reasoning: bool = False,
                 bot: str = "<think>",
                 eot: str = "</think>",
                 ignore_str: str = "Wait",
                 cache_dir: str = "./cache",
                 num_workers: int = 100):
        """
        Initialize the budget forcing client.
        
        Args:
            api_base: Base URL for the API
            api_key: API key for the API
            model: Name of the model to use
            tokenizer_path: Path to the tokenizer (if None, uses model for path)
            reasoning: Whether the model contains both bot and eot tokens
            bot: Beginning of thinking token
            eot: End of thinking token
            ignore_str: String to use when ignoring the stop token
            cache_dir: Directory to cache results
            num_workers: Maximum number of parallel workers
        """
        self.api_base = api_base
        self.model = model
        self.reasoning = reasoning
        self.bot = bot
        self.eot = eot
        self.ignore_str = ignore_str
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        
        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = model
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def _budget_forcing(self, param: Dict[str, Any]) -> Any:
        """
        Execute budget forcing for a single request.
        
        Args:
            param: Dictionary containing all necessary parameters for the task
            
        Returns:
            Dictionary mapping ignore index to full response
        """
        messages = param["messages"]
        max_tokens_thinking = param["max_tokens_thinking"]
        num_ignore = param["num_ignore"]
        temperature = param["temperature"]
        top_p = param["top_p"]
        repetition_penalty = param["repetition_penalty"]
        
        if self.reasoning:
            messages.append({'role': 'assistant', 'content': f'{self.bot}\n'})
        else:
            messages.append({'role': 'assistant', 'content': ''})
        
        end_think_token = self.eot
        begin_think_token = self.bot
        
        current_tokens_thinking = max_tokens_thinking
        response = cached_requests(
            messages=messages,
            model=self.model,
            max_completion_tokens=current_tokens_thinking,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            api_key=self.api_key,
            api_base=self.api_base,
            cache_dir=self.cache_dir
        )
        messages[-1]['content'] = messages[-1]['content'] + response
        
        # Reduce token budget by the length of the response
        current_tokens_thinking -= len(self.tokenizer(response)["input_ids"])

        num_ignore2response = {}
        
        for i in range(num_ignore):  # Num of times to skip stop token
            final_messages = copy.deepcopy(messages)
            
            # Prepare message for final answer generation
            if self.reasoning:
                if end_think_token in final_messages[-1]['content']:
                    final_messages[-1]['content'] = final_messages[-1]['content'].split(end_think_token)[0]
                    final_messages[-1]['content'] = final_messages[-1]['content'] + end_think_token + '\n\nFinal answer: '
                else:
                    # If the begin_think_token is present but no end_think_token, first add the stop token
                    if begin_think_token in final_messages[-1]['content'] and end_think_token not in final_messages[-1]['content']:
                        final_messages[-1]['content'] = final_messages[-1]['content'] + end_think_token + '\n\nFinal answer: '
                    else:
                        # If no begin_think_token or end_think_token, just add the final answer prompt
                        final_messages[-1]['content'] = final_messages[-1]['content'] + '\n\nFinal answer: '
            else:
                final_messages[-1]['content'] = final_messages[-1]['content'] + '\n\nFinal answer: '
            
            # Get the final answer
            final_response = cached_requests(
                messages=final_messages,
                model=self.model,
                max_completion_tokens=current_tokens_thinking,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                api_key=self.api_key,
                api_base=self.api_base,
                cache_dir=self.cache_dir
            )
            num_ignore2response[str(i)] = final_messages[-1]['content'] + final_response

            # Break if we've used up our token budget
            if current_tokens_thinking <= 0:
                break

            # Break if this is the last iteration
            if i == num_ignore - 1:
                break

            # Prepare for next round of thinking
            if self.reasoning:
                if end_think_token in messages[-1]['content']:
                    messages[-1]['content'] = messages[-1]['content'].split(end_think_token)[0]
                # If the model starts thinking with begin_think_token but doesn't finish with end_think_token,
                # we want to make sure we don't add another begin_think_token
                if begin_think_token in messages[-1]['content'] and messages[-1]['content'].strip().endswith(begin_think_token):
                    # Remove the last begin_think_token to avoid duplication
                    messages[-1]['content'] = messages[-1]['content'].rstrip(begin_think_token)
            
            messages[-1]['content'] = messages[-1]['content'] + self.ignore_str
            
            # Continue thinking
            response = cached_requests(
                messages=messages,
                model=self.model,
                max_completion_tokens=current_tokens_thinking,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                api_key=self.api_key,
                api_base=self.api_base,
                cache_dir=self.cache_dir
            )
            
            # Update token count
            current_tokens_thinking -= len(self.tokenizer(response)["input_ids"])
            
            messages[-1]['content'] = messages[-1]['content'] + response

        # Fill in missing responses with the last response
        if len(num_ignore2response) < num_ignore:
            last_key = max(int(k) for k in num_ignore2response.keys()) if num_ignore2response else -1
            last_response = num_ignore2response[str(last_key)] if last_key >= 0 else ""
            for j in range(last_key + 1, num_ignore):
                num_ignore2response[str(j)] = last_response
        
        return num_ignore2response

    def run(self,
            messages_list: List[List[Dict[str, Any]]],
            max_tokens_thinking: int = 32000,
            num_ignore: int = 1,
            temperature: float = 0.3,
            top_p: float = 0.9,
            repetition_penalty: float = 1.05,
            num_workers: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Run budget forcing on multiple inputs with prepared messages in parallel.
        
        Args:
            messages_list: List of prepared message lists
            max_tokens_thinking: Maximum tokens for thinking phase
            num_ignore: Number of times to ignore the stop token
            temperature: Temperature parameter for sampling
            top_p: Top-p parameter for sampling
            repetition_penalty: Repetition penalty parameter
            num_workers: Number of parallel workers (uses self.num_workers if None)
            
        Returns:
            List of dictionaries mapping ignore index to full response
        """
        # Use class num_workers if not provided
        if num_workers is None:
            num_workers = self.num_workers
        
        # Prepare parameter list for parallel processing
        params = []
        for messages in messages_list:
            params.append({
                "messages": copy.deepcopy(messages),
                "max_tokens_thinking": max_tokens_thinking,
                "num_ignore": num_ignore,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            })

        logger.info(f"Running budget forcing on {len(params)} examples using {num_workers} workers")
        results = parallel_processing_backend(
            params=params,
            fn=self._budget_forcing,
            num_workers=num_workers,
            desc="Budget Forcing"
        )
        
        logger.info(f'Budget forcing completed for {len(results)} examples')
        
        return results