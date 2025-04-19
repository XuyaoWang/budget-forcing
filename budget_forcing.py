# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations
import os
import hashlib

import logging
import os
import time
from typing import Any, Callable, Union
import json

import ray

import requests
from tqdm import tqdm
import base64
import json
import PIL
from PIL import Image
from io import BytesIO

from transformers import AutoTokenizer
import copy



@ray.remote(num_cpus=1)
def budget_forcing(
    messages,
    tok,
    max_tokens_thinking:int = 30000,
    num_ignore: int = 1,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    api_key = None,
    api_base = None,
    model = None,
    stop_think_token = "\n\n**答案**\n\n",
    ignore_str = "\n\n等等"
) -> Any:
    
        
    def request_api(
        messages: list[dict[str, Any]],
        max_completion_tokens: int,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        api_key: str,
        api_base: str,
        model: str,
    ) -> str:
        max_try = 3
        while max_try > 0:
            try:
                response = requests.post(
                    f"{api_base}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Connection": "close",
                    },
                    json={
                        "model": model,
                        "max_completion_tokens": max_completion_tokens,
                        "messages": messages,
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                        "top_p": top_p, #not set top_p  
                        "add_generation_prompt": False,
                    },
                )
                if response.status_code == 200:
                    response = response.json()['choices'][0]['message']['content']
                    logging.info(response)
                    break
                elif response.status_code == 400:
                    err_msg = f"Access error, status code: {response.status_code}\nresponse: {response.json()['error']['message']}\nmessages: {messages}\n"
                    response = response.json()['error']['message']
                    logging.error(err_msg)
                    break
                else:
                    logging.error(response.json())
                    time.sleep(3)
                    max_try -= 1
                    continue
            except Exception as e:
                logging.error(e)
                logging.error(response)                
                time.sleep(3)
                max_try -= 1
                continue
        else:
            logging.error('API Failed...')
            response = ''
        return response

    
    current_tokens_thinking = max_tokens_thinking
    response = request_api(
        messages=messages,
        temperature=temperature,
        max_completion_tokens=current_tokens_thinking,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        api_key=api_key,
        api_base=api_base,
        model=model,
    )

    messages.append({'role': 'assistant', 'content': response})
    current_tokens_thinking -= len(tok(response)["input_ids"])

    num_ignore2response = {}

    for i in range(num_ignore): # Num of times to skip stop token
        final_messages = copy.deepcopy(messages)
        if stop_think_token in final_messages[-1]['content']:
            final_messages[-1]['content'] = final_messages[-1]['content'].split(stop_think_token)[0]
        final_messages[-1]['content'] = final_messages[-1]['content'] + stop_think_token 
        final_response = request_api(
            messages=final_messages,
            temperature=temperature,
            max_completion_tokens=current_tokens_thinking + 2000,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )
        num_ignore2response[i] = final_messages[-1]['content'] + final_response


        if current_tokens_thinking <= 0:
            break

        if i == num_ignore - 1:
            break

        if stop_think_token in messages[-1]['content']:
            messages[-1]['content'] = messages[-1]['content'].split(stop_think_token)[0]
        messages[-1]['content'] = messages[-1]['content'] + ignore_str
        
        response = request_api(
            messages=messages,
            temperature=temperature,
            max_completion_tokens=current_tokens_thinking,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )
        current_tokens_thinking -= len(tok(response)["input_ids"])
        messages[-1]['content'] = messages[-1]['content'] + response

    if len(num_ignore2response) < num_ignore:
        last_response = num_ignore2response[max(num_ignore2response.keys())] if num_ignore2response else ""
        for j in range(max(num_ignore2response.keys()) + 1 if num_ignore2response else 0, num_ignore):
            num_ignore2response[j] = last_response

    return num_ignore2response
    
        
def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given arguments."""
    # Convert the input to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid


def call_budget_forcing(
    config,
    tok,
    input_list,
    related_list,
    stop_think_token,
    ignore_str,
    num_workers: int = 10, #30 oom
    cache_dir: str = './cache',
):
    """API"""
    # if len(system_contents) != len(user_contents):
    #     raise ValueError('Length of system_contents and user_contents should be equal.')
    server = budget_forcing
    print("now calling")
    # assert api_key != "", "Please provide API key"
    # assert api_base != "", "Please provide API base"
    # assert model != "", "Please provide model name"

    api_interaction_count = 0
    ray.init()
    # ray.init(_temp_dir="/home/hansirui_2nd/ray_temp")
    
    model = config['model']
    api_key = config['api_key']
    api_base = config['api_base']
    temperature = config['temperature']
    top_p = config['top_p']
    max_tokens_thinking = config['max_tokens_thinking']
    num_ignore = config['num_ignore']
    repetition_penalty = config['repetition_penalty']
    
    bar = tqdm(total=len(input_list))
    results = [None] * len(input_list)

        
    uids = [
      generate_hash_uid({
        'input_list': c, 
        'temperature': temperature,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'max_tokens_thinking': max_tokens_thinking,
        'num_ignore': num_ignore,
        'model': model,
        'stop_think_token':stop_think_token,
        'ignore_str':ignore_str
      }) for c in input_list]
    
    agg_results = []
    not_finished = []
    input_list_ori = input_list.copy() # 用于 save to json
    input_list = list(enumerate(input_list)) # to have index
    
    while True:
        if len(not_finished) == 0 and len(input_list) == 0:
            break
        while len(not_finished) < num_workers and len(input_list) > 0:
            index, content = input_list.pop()
            uid = uids[index]
            
            # temp = temperature[index] if temperature is not None else 0.2
            # p = top_p[index]
        
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    try:
                        result = json.load(f)
                    except:
                        print("error in loading cache")
                        os.remove(cache_path)
                        continue  # 这里continue 有点奇怪, 上面执行了 input_list.pop(), 这样这条问题的数据不会再被处理的
                                    # 不过猜测应该很难走到except这里
                results[index] = result
                agg_results.append(result)
                bar.update(1)
                continue

            future = server.remote(
                content,
                tok,
                max_tokens_thinking,
                num_ignore,
                temperature, 
                top_p,
                repetition_penalty,
                api_key,
                api_base,
                model,
                stop_think_token,
                ignore_str
            )
            not_finished.append([index, future])
            api_interaction_count += 1

        if len(not_finished) == 0:
            continue

        indices, futures = zip(*not_finished)
        finished, not_finished_futures = ray.wait(list(futures), timeout=1.0)
        finished_indices = [indices[futures.index(task)] for task in finished]

        for i, task in enumerate(finished):
            results[finished_indices[i]] = ray.get(task)
            uid = uids[finished_indices[i]]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            result_sample = {
                "index": finished_indices[i],
                "input_messages": input_list_ori[finished_indices[i]],
                "output": results[finished_indices[i]],
                "max_tokens_thinking": max_tokens_thinking,
                "num_ignore":num_ignore,
                "temperature": temperature,
                "top_p": top_p,
                "model": model,
                'stop_think_token':stop_think_token,
                'ignore_str':ignore_str
                }
            result_sample.update(related_list[finished_indices[i]])
            agg_results.append(result_sample)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result_sample, f, ensure_ascii=False, indent=4)

        not_finished = [(index, future) for index, future in not_finished if future not in finished]

        bar.update(len(finished))
    bar.close()

    assert all(result is not None for result in results)

    ray.shutdown()
    print(f'API interaction count: {api_interaction_count}')

    return agg_results