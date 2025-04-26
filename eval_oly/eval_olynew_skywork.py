import glob
import json
import os
import sys
from datasets import load_dataset
from tqdm import tqdm, trange
import re
from transformers import AutoTokenizer

from util_oly_math_judger import MathJudger
judger = MathJudger()

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-R1V-38B", trust_remote_code=True)

def count(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# 数据集子集列表
subset_list = [
    "OE_MM_maths_en_COMP", "OE_MM_maths_zh_CEE", "OE_MM_maths_zh_COMP",
    "OE_MM_physics_en_COMP", "OE_MM_physics_zh_CEE", "OE_TO_maths_en_COMP",
    "OE_TO_maths_zh_CEE", "OE_TO_maths_zh_COMP", "OE_TO_physics_en_COMP",
    "OE_TO_physics_zh_CEE", 
]

# 用于存储最终结果的列表
results_data = []

# 为每个wait值计算分数和token长度
for wait in range(7):
    win_count = 0
    total_count = 0
    total_token_count = 0
    
    for subset in subset_list:
        file_path = f"/home/hansirui_2nd/pcwen_workspace/s1m_assemble/result/R1V-30168_{subset}-{wait}wait.json"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            
            # ds = load_dataset("Hothan/OlympiadBench", subset)
            # train_ = ds["train"].to_list()
            
            
            for result in tqdm(all_results, desc=f"Processing {subset} with wait={wait}"):
                total_count += 1
                
                # 计算token长度
                token_length = count(result.get('my_model_thinking', '') + result.get('output', ''))
                total_token_count += token_length
                
                # 计算准确率
                # input_query = result['instruction']
                response = result['my_model_output']
                
                # the_sample = find_the_correspoding_qa_in_dataset(input_query)
                if result:
                    ground_truth = result['final_answer']
                    
                    answer_type = result['answer_type']
                    if 'Tuple' in answer_type:
                        judge_result = judger.judge(response, result['final_answer'][0])
                    else:
                        if result['error']:
                            if ',' in result['error']:
                                precisions = result['error'].split(',')
                                precisions = [float(p) if p else 1e-8 for p in precisions]
                                judge_result = judger.judge(response, result['final_answer'][0], precisions)
                            else:
                                precision = float(result['error'])
                                judge_result = judger.judge(response, result['final_answer'][0], precision)
                        else:
                            judge_result = judger.judge(response, result['final_answer'][0])
                    
                    if judge_result:
                        win_count += 1
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # 如果有数据则计算分数和平均token长度
    if total_count > 0:
        accuracy = win_count / total_count
        avg_token_length = total_token_count / total_count
        
        # 将结果添加到列表
        results_data.append({
            "model_name": "Skywork-R1V-38B",
            "wait": wait,
            "accuracy": accuracy,
            "avg_token_length": avg_token_length
        })
        
        print(f"Wait: {wait}, Accuracy: {accuracy:.4f}, Avg Token Length: {avg_token_length:.2f}")

# 将结果保存为JSON文件
output_file = "/home/hansirui_2nd/pcwen_workspace/s1m_assemble/scores/Skywork-R1V-38B_oly_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, indent=4)

print(f"Results saved to {output_file}")