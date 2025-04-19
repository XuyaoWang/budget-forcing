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


# Define the function to process each example
def gen_text_input(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    return input


def gen_messages(example):
    text_input = gen_text_input(example)
    base64_image = encode_pil_image(example['decoded_image'])
    messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": text_input},
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
              },
            },
          ],
        }
    ]
    return messages


dataset = load_dataset("MathLLMs/MathVision")
input_list = []
related_list = []
print("preparing input")
for sample in tqdm(dataset['test']):
    messages = gen_messages(sample)
    # for skywork R1V
    messages.append({'role': 'assistant', 'content': "<think>\n"})
    input_list.append(messages)
    related_sample = copy.deepcopy(sample)
    # dont need image
    del related_sample["decoded_image"]
    related_list.append(related_sample)
#############################################
# send to next stage and call 
#############################################
from budget_forcing import call_budget_forcing
from transformers import AutoTokenizer
# ref https://github.com/SkyworkAI/Skywork-R1V/blob/main/inference/inference_with_vllm.py#L10-33
config = {
    "api_key" : "EMPTY",
    "api_base" :"http://localhost:8000/v1/chat/completions",
    "model" :"Skywork-R1V-38B",
    'temperature': 0.,
    'top_p': 0.95,
    'max_tokens_thinking': 30168,
    'repetition_penalty' : 1.05,
    'num_ignore': 7
}
print(config)
tokenizer = AutoTokenizer.from_pretrained("/ceph/home/yaodong01/s1-m/Models/Skywork-R1V-38B", trust_remote_code=True)
results = call_budget_forcing(config,tokenizer,input_list,related_list,stop_think_token="\n</think>\n\n",ignore_str="\n\nWait",num_workers=50,cache_dir = "/home/hansirui_2nd/pcwen_workspace/s1m_assemble/cache/r1v_mathv")
try:
    generated_chunks = {}
    for res,related_info in tqdm(zip(results,related_list)):
        for wait in res["output"].keys():
            ta = res["output"][wait]
            try:
                thinking = ta.split("\n</think>\n\n")[0]
                answer = ta.split("\n</think>\n\n")[1]
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
                "generator": "R1V"
                })
            generated_chunks[wait].append(result_sample)
            
    for wait in generated_chunks.keys():
        config_output_path = f"result/R1V-mathv-30168-{wait}wait.json"
        output_filename = config_output_path
        write_json(output_filename, generated_chunks[wait])
except:
    pass
