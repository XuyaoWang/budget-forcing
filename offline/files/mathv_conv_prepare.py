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
    input_list.append(messages)
    related_sample = copy.deepcopy(sample)
    # dont need image
    del related_sample["decoded_image"]
    related_list.append(related_sample)
#############################################
# send to next stage and call 
#############################################
big_collect = []
for input, related in zip(input_list,related_list):
    sample = copy.deepcopy(related)
    sample["input"] = input

    big_collect.append(sample)
    
output_filename = "offline/files/mathv_0.json"
write_json(output_filename, big_collect)
