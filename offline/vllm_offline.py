# -- coding: utf-8 --
from typing import Any, Dict
import json
import numpy as np
import ray
import hashlib
import os
import time
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams,EngineArgs
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Any, Dict, List




def get_batch_hash(batch_data):
    """Generate hash for a batch of data"""
    # long_str = ''.join(batch_data)
    long_str = str(batch_data)
    batch_str = str(long_str).encode('utf-8')
    return hashlib.md5(batch_str).hexdigest()

def load_json(file):
    try:
        with open(file, 'r', encoding="utf8") as load_f:
            data = json.load(load_f)
        return data
    except Exception as e:
        print("Error loading JSON:", e)
        return []

def write_json(file, data):
    try:
        with open(file, "w", encoding="utf8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Error writing JSON:", e)

def get_completed_batches(CACHE_DIR):
    """Get list of completed batch hashes from cache directory"""
    completed = set()
    for filename in os.listdir(CACHE_DIR["completed"]):
        if filename.endswith('.json'):
            completed.add(filename[:-5])  # Remove .json extension
    print(f"Found {len(completed)} completed batches")
    return completed

def cleanup_stale_processing(CACHE_DIR):
    """Clean up stale processing flags"""
    current_time = time.time()
    cleaned = 0
    for flag in os.listdir(CACHE_DIR["processing"]):
        flag_path = os.path.join(CACHE_DIR["processing"], flag)
        if current_time - os.path.getmtime(flag_path) > 3600:  # 1小时过期
            try:
                os.remove(flag_path)
                cleaned += 1
            except Exception as e:
                print(f"Error cleaning up {flag}: {e}")
    if cleaned > 0:
        print(f"Cleaned up {cleaned} stale processing flags")

def combine_json_lists(folder_path):
    combined_list = []
    
    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取json文件
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
                # 确保读取的数据是list类型
                if isinstance(json_data, list):
                    combined_list.extend(json_data)
                else:
                    print(f"Warning: {filename} does not contain a list")
    
    return combined_list

        
def process_with_requests(
    config: Dict[str, Any],
    input_list: List[Any],
    related_list: List[Any],
    num_workers: int = 4,
    cache_dir: str = "./cache"
) -> List[Dict[str, Any]]:
    """
    离线批处理推理（接口与 vllm_online 保持一致）

    参数：
      - config: 包含模型、采样参数等设定的字典。示例键包括：
            {
                "model": "模型路径或名称",
                "temperature": 0,
                "top_p": 0,
                "max_completion_tokens": 1048,
                "tensor_parallel_size": 8,             # 可选，默认为8
                "gpu_memory_utilization": 0.9,           # 可选
                "max_model_len": 8192,                   # 可选
                "batch_size": 256                        # 可选，每个批次处理的请求条数
            }
      - input_list: 每个元素应为聊天消息列表，例如：
            [
                [{"role": "user", "content": "你好"}],
                [{"role": "user", "content": "今天天气怎么样？"}],
                ...
            ]
         如果 input_list 元素为字符串，则自动包装为聊天格式。
      - num_workers: 并行处理批次数量（建议根据 GPU 数量合理分配）。
      - cache_dir: 缓存目录路径，处理结果会保存到 cache_dir 下的 completed 子目录中
                   防止重复推理。

    返回：一个列表，每个元素是字典，包含输入和生成回答及相关参数（与 online 版本返回结果一致）
    """
    
    

    model_path = config['model']
    temperature = config['temperature']
    top_p = config['top_p']
    repetition_penalty = config['repetition_penalty']
    gpu_memory_utilization = config['gpu_memory_utilization']
    tensor_parallel_size = config['tensor_parallel_size']

    max_completion_tokens = config['max_completion_tokens']
    
    
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, max_tokens=max_completion_tokens)
    
    class LLMPredictor:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = EngineArgs(model=model_path,
                           tensor_parallel_size=tensor_parallel_size,
                           gpu_memory_utilization=gpu_memory_utilization,
                           max_model_len=max_completion_tokens,
                           limit_mm_per_prompt={"image": 5})
            print("Model initialized successfully.")

        def try_acquire_batch(self, batch_hash):
            """Try to acquire a batch for processing"""
            # 检查是否已完成
            if os.path.exists(os.path.join(CACHE_DIR["completed"], f"{batch_hash}.json")):
                return False
                
            # 创建processing标记
            processing_flag = os.path.join(CACHE_DIR["processing"], f"{batch_hash}.flag")
            
            try:
                fd = os.open(processing_flag, os.O_CREAT | os.O_EXCL)
                os.close(fd)
                return True
            except FileExistsError:
                return False

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            batch_hash = get_batch_hash(batch["data"])
            
            # 如果已完成或正在处理，直接跳过
            if not self.try_acquire_batch(batch_hash):
                return {'results': []}
            
            try:
                prompts = []
                for str_messages in batch["data"]:
                    messages = [json.loads(item) for item in str_messages] 
                    prompt = self.tokenizer.apply_chat_template(
                        [messages[0]],
                        tokenize=False,
                        add_generation_prompt=True,    
                    )
                    prompt += messages[1]["content"]
                    prompts.append(prompt)

                outputs = self.llm.generate(prompts, sampling_params)
                generated_text = [output.outputs[0].text for output in outputs]
                
                # 保存结果
                result = []
                for i, str_messages in enumerate(batch["data"]):
                    messages = [json.loads(item) for item in str_messages] 
                    save_sample = {
                        "input_messages": messages,
                        "generated_text": generated_text[i],
                        "annotator":model_path,
                        "temperature":temperature,
                        "top_p":top_p,
                        "repetition_penalty":repetition_penalty,
                        "max_completion_tokens":max_completion_tokens
                    }
                    save_sample.update(related_list[i])
                    result.append(save_sample)
                write_json(os.path.join(CACHE_DIR["completed"], f"{batch_hash}.json"), result)
                
            finally:
                # 清理processing标记
                processing_flag = os.path.join(CACHE_DIR["processing"], f"{batch_hash}.flag")
                try:
                    os.remove(processing_flag)
                except Exception as e:
                    print(f"Error removing processing flag: {e}")
            
            return {'results': []}  # 返回值并不重要


    BASE_CACHE_DIR = cache_dir
    CACHE_DIR = {
        "completed": os.path.join(cache_dir, "completed"),
        "processing": os.path.join(cache_dir, "processing"),
        "failed": os.path.join(cache_dir, "failed")
    }
    for dir_path in CACHE_DIR.values():
        os.makedirs(dir_path, exist_ok=True)
    cleanup_stale_processing(CACHE_DIR)

    BATCH_SIZE = config['batch_size']
    batches = [input_list[i:i + BATCH_SIZE] for i in range(0, len(input_list), BATCH_SIZE)]
    batch_hashes = [get_batch_hash(batch) for batch in batches]
    # Get completed batches
    completed_batches = get_completed_batches(CACHE_DIR)
    
    # Filter out completed queries
    remaining_queries = []
    for i, batch_hash in enumerate(batch_hashes):
        if batch_hash not in completed_batches:
            remaining_queries.extend(batches[i])
        else:
            print(f"Batch {i} already processed")

    print(f"Total queries: {len(input_list)}")
    print(f"Remaining queries: {len(remaining_queries)}")
    
    if not remaining_queries:
        print("All batches already processed. Combining results...")
    else:
        CHUNK_SIZE = 100
        chunks = [remaining_queries[i:i + CHUNK_SIZE*BATCH_SIZE] for i in range(0, len(remaining_queries), CHUNK_SIZE*BATCH_SIZE)]
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}, one chunk has {len(chunk)} queries")
            ray.init()
            assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be at least 2.22.0"
            processed_chunk = [[json.dumps(item) for item in messages] for messages in chunk]
            ds = ray.data.from_numpy(np.array(processed_chunk))
            print("running ds")
            
            def scheduling_strategy_fn():
                pg = ray.util.placement_group(
                    [{"GPU": 1, "CPU": 20}] * tensor_parallel_size,
                    strategy="STRICT_PACK",
                )
                return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
                    pg, placement_group_capture_child_tasks=True))

            resources_kwarg: Dict[str, Any] = {}
            if tensor_parallel_size == 1:
                resources_kwarg["num_gpus"] = 1
            else:
                resources_kwarg["num_gpus"] = 0
                resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn
                
            print("running map_batches")
            num_instances = 1
            ds = ds.map_batches(
                LLMPredictor,
                concurrency=num_instances,
                batch_size=BATCH_SIZE,
                **resources_kwarg,
            )

            outputs = ds.take_all()
            ray.shutdown()
        
    
    # read all the data under   CACHE_DIR["completed"]
    final_result = combine_json_lists(CACHE_DIR["completed"])
    return final_result