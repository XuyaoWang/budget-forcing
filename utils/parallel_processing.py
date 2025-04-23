import ray
import os
from tqdm import tqdm
from typing import List, Dict, Any, Callable, TypeVar, Generic, Optional
import logging

T = TypeVar('T')
R = TypeVar('R')

logger = logging.getLogger(__name__)

def parallel_processing_backend(
    params: List[Dict[str, Any]], 
    fn: Callable[[Dict[str, Any]], R], 
    num_workers: int = 10, 
    desc: str = "Processing tasks"
) -> List[Optional[R]]:
    """
    A backend for parallel processing using Ray with progress bar and controllable parallelism.
    
    Args:
        params: A list of dictionaries, each dictionary is an input to fn
        fn: A function that processes a single input dictionary
        num_workers: Maximum number of parallel workers
        desc: Description for the progress bar
        
    Returns:
        A list of results, where each result corresponds to fn(param) for each param in params. 
        Returns None for tasks that encountered an exception.
    """
    
    if not ray.is_initialized():
        # ray_temp_dir = "/aifs4su/yaodong/xuyao/ray_temp"
        # os.makedirs(ray_temp_dir, exist_ok=True)
        # ray.init(_temp_dir=ray_temp_dir)
        ray.init()
    
    @ray.remote
    def remote_fn(index, param):
        try:
            return fn(param)
        except Exception as e:
            logger.error(f"Task failed for index {index} with error: {e}", exc_info=True) 
            return None 
    
    bar = tqdm(total=len(params), desc=desc)
    results = [None] * len(params)
    
    contents = list(enumerate(params))
    not_finished = []
    
    while True:
        if len(not_finished) == 0 and len(contents) == 0:
            break
        
        while len(not_finished) < num_workers and len(contents) > 0:
            index, param = contents.pop(0)
            future = remote_fn.remote(index, param) 
            not_finished.append([index, future]) 
        
        if len(not_finished) == 0:
            continue
        
        finished_futures, not_finished_futures_list = ray.wait(
            [future for _, future in not_finished], 
            num_returns=len(not_finished), 
            timeout=1.0
        )
        
        future_id_to_index = {future.hex(): index for index, future in not_finished}

        new_not_finished = []
        processed_indices = set()

        if finished_futures:
            finished_results = ray.get(finished_futures)
            for i, future in enumerate(finished_futures):
                original_index = future_id_to_index.get(future.hex())
                if original_index is not None:
                    results[original_index] = finished_results[i]
                    processed_indices.add(original_index)
                    bar.update(1)

        for index, future in not_finished:
             if index not in processed_indices:
                 new_not_finished.append([index, future])
        not_finished = new_not_finished
    
    bar.close()
    
    return results