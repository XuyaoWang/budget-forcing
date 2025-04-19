# %%
from datasets import load_dataset
import glob
import json
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
from util_mathv_utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number

# 加载数据集
dataset = load_dataset("MathLLMs/MathVision")

# 获取问题和选项列表
question_list = dataset['test']["question"]
options_list = dataset['test']["options"]



def eval(model_name, result_prefix):
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def count(text):
        """计算文本的token数量"""
        tokens = tokenizer.encode(text)
        return len(tokens)

    def find_sample(inputq):
        """根据问题查找对应的选项"""
        for q, o in zip(question_list, options_list):
            if inputq == q:
                return o
        return []

    # 创建结果列表，用于存储最终的JSON数据
    results_data = []

    # 为每个wait值评估模型性能
    for wait in range(7):
        file_path = f"{result_prefix}{wait}xwait.json"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            
            # 评估准确率
            win_count = 0
            total_token_count = 0
            
            for result in tqdm(all_results, desc=f"Processing results with wait={wait}"):
                # 提取问题、回答和正确答案
                input_query = result['instruction']
                response = result['output']
                ground_truth = result['gt']
                
                # 获取token长度
                thinking = result.get('thinking', '')
                token_length = count(thinking + response)
                total_token_count += token_length
                
                # 提取模型答案
                options = find_sample(input_query)
                if len(options) > 0:
                    gt_answer_value = options[ord(ground_truth)-ord('A')]
                else:
                    gt_answer_value = ''
                
                model_answer = '\\boxed{' + response.split('oxed{')[-1]
                model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
                
                # 判断答案是否正确
                if is_equal(ground_truth, model_answer):
                    win_count += 1
            
            # 计算准确率和平均token长度
            accuracy = win_count / len(all_results) if len(all_results) > 0 else 0
            avg_token_length = total_token_count / len(all_results) if len(all_results) > 0 else 0
            
            # 将结果添加到列表
            results_data.append({
                "model_name": model_name,
                "wait": wait,
                "accuracy": accuracy,
                "avg_token_length": avg_token_length
            })
            
            print(f"Wait: {wait}, Accuracy: {accuracy:.4f}, Avg Token Length: {avg_token_length:.2f}")
        
        except Exception as e:
            print(f"Error processing wait={wait}: {e}")

    # 将结果保存为JSON文件
    model = model_name.split("/")[1]
    output_file = f"/home/hansirui_2nd/pcwen_workspace/s1m_assemble/scores/{model}-mathv_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to {output_file}")


# eval("Qwen/QVQ-72B-Preview", "/aifs4su/hansirui/pcwen_workspace/t/MATH-V/results/qvq_mathv_")
eval("Fancy-MLLM/R1-Onevision-7B-RL", "/aifs4su/hansirui/pcwen_workspace/t/MATH-V/results/r1-ov-7b-30168_")
