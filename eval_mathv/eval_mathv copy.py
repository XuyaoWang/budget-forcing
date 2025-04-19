# %%
from datasets import load_dataset

dataset = load_dataset("MathLLMs/MathVision")

# %%
import glob
import json
import os
import sys

question_list = dataset['test']["question"]
options_list = dataset['test']["options"]

def find_sample(inputq):
    for q,o in zip(question_list, options_list):
        if inputq == q:
            return o
        


from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("Qwen/QVQ-72B-Preview", trust_remote_code=True)
def count(text):
    tokens = tokenizer.encode(text)
    return len(tokens)



file_path = f"/aifs4su/hansirui/pcwen_workspace/t/MATH-V/results/qvq_mathv_{sys.argv[1]}wait.json"
with open(file_path, 'r', encoding='utf-8') as f:
    all_results = json.load(f)


# %%
from util_mathv_utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number
from tqdm import tqdm
win_count = 0
for result in tqdm(all_results):

    input_query = result['instruction']
    response = result['output']
    ground_truth = result['gt']
    options = find_sample(input_query)
    if len(options) > 0:
        gt_answer_value = options[ord(ground_truth)-ord('A')]
    else:
        gt_answer_value = ''
    
    model_answer = '\\boxed{' + response.split('oxed{')[-1]
    model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()

    if  is_equal(ground_truth, model_answer) or is_equal(ground_truth, model_answer) :
        win_count+=1
        
print(win_count/len(all_results)) 




for i in range(7):
    tokencout= 0
    all_count = 0
    for subset in subset_list:
        file  = f"/aifs4su/hansirui/pcwen_workspace/t/new_Oly/qvq-stts/result/QVQ-30168_{subset}-{i}wait.json"
        with open(file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)


        
        for result in tqdm(all_results):
            all_count+=1
            input_query = result['instruction']
            response = result['output']
            model_answer = response
            c = count(result['thinking']+result['output'])
            tokencout+=c


    print(i)
    print("wait")
    print(tokencout/all_count)