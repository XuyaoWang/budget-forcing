# prepare input messages, 支持断点重跑, 最好从huggingface上弄dataset下来




#############################################
# Prepare input messages
#############################################

from datasets import load_dataset
import base64, io
from tqdm import tqdm
from PIL import Image
import json
import hashlib
import re
import copy
def write_json(file, data):
    try:
        with open(file, "w", encoding="utf8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Error writing JSON:", e)

def encode_pil_image(pil_img, format="JPEG"):
    # 如果图像不是 RGB 模式，则转换为 RGB 模式
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    return base64.b64encode(img_bytes).decode("utf-8")

def split_markdown(prompt, sample):
    # use regex to extract image property
    items = re.split('(<image_\d+>)', prompt)

    # 从分割后的字符串列表中去除空的元素（完全由空白符、换行符或制表符构成的字符串）
    items = [item for item in items if item and not item.isspace()]
    message_items = []
    for item in items:
        if item.startswith('<image_'):
            image_id = item.replace('<', '').replace('>', '')
            pil_img = sample[image_id]
            base64_image = encode_pil_image(pil_img)
            message_items.append({
                'type': 'image_url',
                'image_url': {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                }
            })
        else:
            message_items.append({
                'type': 'text',
                'text': item.strip()
            })
    return message_items

# Dictionary mappings for answer types
chinese_answer_type_dict = {
    'Numerical': '数值',
    'Expression': '表达式',
    'Equation': '方程',
    'Interval': '区间'
}
english_answer_type_dict = {
    'Numerical': 'a numerical value',
    'Expression': 'an expression',
    'Equation': 'an equation',
    'Interval': 'an interval'
}

def get_single_answer_type_text(answer_type, is_chinese):
    if '-' in answer_type:    # No need now
        answer_type = answer_type[:answer_type.find('-')]
    for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
        if t in answer_type:
            if is_chinese:
                return chinese_answer_type_dict[t]
            else:
                return english_answer_type_dict[t]
    exit(f'Error parsing answer type {answer_type}!')

def get_answer_type_text(answer_type, is_chinese, multiple_answer):
    if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):
        full_answer_text = ''
    else:
        if not multiple_answer:
            answer_text = get_single_answer_type_text(answer_type, is_chinese)
            if is_chinese:
                full_answer_text = f'，答案类型为{answer_text}'
            else:
                full_answer_text = f"The answer of The problem should be {answer_text}. "
        else:
            if ',' not in answer_type:    # Same answer type for all answers
                answer_text = get_single_answer_type_text(answer_type, is_chinese)
                if is_chinese:
                    full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                else:
                    full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
            else:
                answer_types = answer_type.split(',')
                answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
                if len(set(answer_types)) == 1:
                    answer_text = answer_types[0]
                    if is_chinese:
                        full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                    else:
                        full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
                else:
                    if is_chinese:
                        answer_text = '、'.join(answer_types)
                        full_answer_text = f'，题目有多个答案，答案类型分别为{answer_text}'
                    else:
                        answer_text = ', '.join(answer_types)
                        full_answer_text = f'The problem has multiple answers, with the answers in order being {answer_text}. '
    return full_answer_text

def make_prompt(sample, is_chinese, is_math, is_theorem_proving):
    if is_chinese:
        subject_content = '数学' if is_math else '物理'
        if is_theorem_proving:
            prompt = f'以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。证明过程中使用的变量和公式请使用LaTeX格式表示。'
        else:
            answer_type_text = get_answer_type_text(sample['answer_type'], is_chinese=True, multiple_answer=sample['is_multiple_answer'])
            if sample['is_multiple_answer']:
                multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
            else:
                multiple_answer_text = '\\boxed{答案}'
            unit_text = ''
            if sample['unit']:
                multiple_answer_text += '(单位)'
                unit_text = '，注意答案的单位不要放在\\boxed{}中'
            prompt = f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以"所以最终答案是{multiple_answer_text}。"显式给出结果{unit_text}。'
    else:
        subject_content = 'Math' if is_math else 'Physics'
        if is_theorem_proving:
            prompt = f'The following is a theorem proving problem from an International {subject_content} competition. Please use logical reasoning and common theorems to prove the proposition in the problem according to the given requirements. Please use LaTeX format to represent the variables and formulas used in the proof.'
        else:
            if sample['is_multiple_answer']:
                multiple_answer_text = '\\boxed{multiple answers connected with commas}'
            else:
                multiple_answer_text = '\\boxed{answer}'
            unit_text = ''
            if sample['unit']:
                multiple_answer_text += '(unit)'
                unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
            answer_type_text = get_answer_type_text(sample['answer_type'], is_chinese=False, multiple_answer=sample['is_multiple_answer'])
            prompt = f'The following is an open-ended problem from an International {subject_content} competition. {answer_type_text}Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "So the final answer is {multiple_answer_text}." and give the result explicitly{unit_text}.'
    return prompt

def make_input(prompt, sample, is_chinese, is_math):
    if is_chinese:
        subject = '数学' if is_math else '物理'
        question_message = split_markdown(prompt, sample)
        messages = [
            {
                'role': 'system',
                'content': f'你是一个中文人工智能助手。请根据要求，完成下面的{subject}竞赛题目。'
            },
            {
                'role': 'user',
                'content': question_message
            }
        ]
    else:
        subject = 'Math' if is_math else 'Physics'
        question_message = split_markdown(prompt, sample)
        messages = [
            {
                'role': 'system',
                'content': f'You are an AI assistant. Please answer the following {subject} competition problems as required.'
            },
            {
                'role': 'user',
                'content': question_message
            }
        ]
    return messages



def write_json(file, data):
    try:
        with open(file, "w", encoding="utf8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Error writing JSON:", e)



subset_list = [
    "OE_MM_maths_en_COMP", "OE_MM_maths_zh_CEE", "OE_MM_maths_zh_COMP",
    "OE_MM_physics_en_COMP", "OE_MM_physics_zh_CEE", "OE_TO_maths_en_COMP",
    "OE_TO_maths_zh_CEE", "OE_TO_maths_zh_COMP", "OE_TO_physics_en_COMP",
    "OE_TO_physics_zh_CEE"
]
from tqdm import trange
from budget_forcing import call_budget_forcing

from transformers import AutoTokenizer

for subset in subset_list:
    print(f"Processing subset: {subset}")
    ds = load_dataset("Hothan/OlympiadBench", subset)
    is_theorem_proving = 'TP' in subset
    is_math = 'maths' in subset
    is_chinese = 'zh' in subset
    final_answers = []
    input_list = []
    related_list = []
    for i in trange(len(ds["train"])):
        sample = ds["train"][i]
        prefix_prompt = make_prompt(sample, is_chinese, is_math, is_theorem_proving)
        if is_math:
            prefix_prompt = prefix_prompt + '\n' + sample['question']
        else:
            if 'context' in sample.keys() and sample['context']: # cannot be null
                prefix_prompt = prefix_prompt + '\n' + sample['context']+'\n'+sample['question']
            else:
                prefix_prompt = prefix_prompt + '\n' + sample['question']
        messages = make_input(prefix_prompt,sample, is_chinese, is_math)
        input_list.append(messages)
        final_answers.append(sample["final_answer"])
        related_sample = copy.deepcopy(sample)
        # dont need image
        del related_sample["image_1"]
        del related_sample["image_2"]
        del related_sample["image_3"]
        del related_sample["image_4"]
        del related_sample["image_5"]
        
        related_list.append(related_sample)

#############################################
# send to next stage and call 
#############################################


    # ref https://github.com/SkyworkAI/Skywork-R1V/blob/main/inference/inference_with_vllm.py#L10-33
    config = {
        "api_key" : "EMPTY",
        "api_base" :"http://dgx-050:9000/v1/chat/completions",
        "model" :"QVQ-72B-Preview",
        'temperature': 0.,
        'top_p': 0.95,
        'max_tokens_thinking': 30168,
        'repetition_penalty' : 1.05,
        'num_ignore': 7
    }
    tokenizer = AutoTokenizer.from_pretrained("Qwen/QVQ-72B-Preview", trust_remote_code=True)
    if not is_chinese:
        results = call_budget_forcing(config,tokenizer,input_list,related_list,stop_think_token="\n\n**Final Answer**\n\n",ignore_str="\n\nWait",num_workers=100,cache_dir = "/aifs4su/hansirui/pcwen_workspace/s1m/cache/qvq_oly")
    else:
        results = call_budget_forcing(config,tokenizer,input_list,related_list,stop_think_token="\n\n**答案**\n\n",ignore_str="\n\n等等",num_workers=100,cache_dir = "/aifs4su/hansirui/pcwen_workspace/s1m/cache/qvq_oly")
    try:
        generated_chunks = []
        for res,related_info in tqdm(zip(results,related_list)):     
            for wait in res["output"].keys():
                ta = res["output"][wait]
                try:
                    thinking = ta.split("\n\n**Final Answer**\n\n")[0]
                    answer = ta.split("\n\n**Final Answer**\n\n")[1]
                except:
                    try:
                        thinking = ta.split("\n\n**答案**\n\n")[0]
                        answer = ta.split("\n\n**答案**\n\n")[1]
                    except:
                        answer = ta
                        thinking = ""
                    # raise ValueError("Error in splitting thinking and answer")
                try:
                    generated_chunks[wait]
                except:
                    generated_chunks[wait] = []
                result_sample = copy.deepcopy(related_info)
                result_sample.update({
                    "my_model_thinking": thinking,
                    "my_model_output": answer,
                    "generator": "QvQ"
                    })
                generated_chunks[wait].append(result_sample)
                
        for wait in generated_chunks.keys():
            config_output_path = f"/aifs4su/hansirui/pcwen_workspace/s1m/result/QvQ-30168_{subset}-{wait}wait.json"
            output_filename = config_output_path
            write_json(output_filename, generated_chunks[wait])
    except:
        continue
    # (vllm-server) hansirui_2nd@dgx-106:~/pcwen_workspace/s1m/OlympiadBench/my_scripts_totests1m$ python input_prepared_120.py 