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
                #  tokenizer_path: Optional[str] = None,
                #  reasoning: bool = False,
                #  bot: str = "<think>",
                #  eot: str = "</think>",
                #  ignore_str: str = "Wait",
                 cache_dir: str = "./cache",
                 num_workers: int = 100,
                 **kwargs,
        ):
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
        # self.reasoning = reasoning
        # self.bot = bot
        # self.eot = eot
        # self.ignore_str = ignore_str
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.num_workers = num_workers

        # Load tokenizer
        # if tokenizer_path is None:
        #     tokenizer_path = model

        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def inference(self,
                  messages_list: List[List[Dict[str, Any]]],
                  temperature: float,
                  top_p: float,
                  repetition_penalty: float,
                  max_tokens: int,
                  ) -> List[Any]:
        """
        Run inference on the model using parallel processing.

        Args:
            messages_list: List of message lists
            temperature: Temperature parameter for sampling
            top_p: Top-p parameter for sampling
            repetition_penalty: Repetition penalty parameter
            max_tokens: Maximum tokens for the response

        Returns:
            List of responses
        """
        params = [{
            'messages': messages,
            'temperature': temperature,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'max_tokens': max_tokens,
            } for messages in messages_list]

        results = parallel_processing_backend(
            params=params,
            fn=self._inference_item_wrapper,
            num_workers=self.num_workers,
            desc="Inferring"
        )
        return results

    def _inference_item_wrapper(self, param: Dict[str, Any]) -> Any:
        """Wrapper for inference_item for parallel processing."""
        return self.inference_item(
            messages=param["messages"],
            temperature=param["temperature"],
            top_p=param["top_p"],
            repetition_penalty=param["repetition_penalty"],
            max_tokens=param["max_tokens"]
        )

    def inference_item(self,
                       messages: List[Dict[str, Any]],
                       temperature: float,
                       top_p: float,
                       repetition_penalty: float,
                       max_tokens: int
                       ) -> Any:
        """Calls the cached request function for a single inference."""
        response = cached_requests(
            messages=messages,
            model=self.model,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_completion_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
            cache_dir=self.cache_dir
        )
        return response

    def run(
        self,
        messages_list: List[List[Dict[str, Any]]],
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
    ):
        return self.inference(
            messages_list=messages_list,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens
        )
