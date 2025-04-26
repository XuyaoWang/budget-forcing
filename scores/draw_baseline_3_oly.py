import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

BASE_DIR = "/home/hansirui_2nd/pcwen_workspace/s1m_assemble/scores"
SAVE_DIR = "/home/hansirui_2nd/pcwen_workspace/s1m_assemble/scores/images"



def anaylze_results(result_dir):
    results = []
    data_path = os.path.join(BASE_DIR, result_dir)
    with open(data_path, "r") as json_file:
        data = json.load(json_file)
        for sample in data:
            results.append({
                "num_ignore": sample['wait'],
                "accuracy": sample['accuracy'],
                "token_mean": sample['avg_token_length']
            })
    return results

# 分析结果
model_7b_result_dir = "r1-ov-7b_oly_results.json"
model_7b_results = anaylze_results(model_7b_result_dir)
model_7b_results.sort(key=lambda x: x["num_ignore"])

model_38b_result_dir = "Skywork-R1V-38B_oly_results.json"
model_38b_results = anaylze_results(model_38b_result_dir)
model_38b_results.sort(key=lambda x: x["num_ignore"])

model_72b_result_dir = "QVQ_oly_results.json"
model_72b_results = anaylze_results(model_72b_result_dir)
model_72b_results.sort(key=lambda x: x["num_ignore"])

# 创建3个子图，每个子图对应一个模型
models = [
    ("R1-Onevision-7B-RL", model_7b_results),
    ("Skywork-R1V-38B", model_38b_results),
    ("Qwen/QVQ-72B-Preview", model_72b_results)
]

# 设置绘图参数
fig = plt.figure(figsize=(15, 5))  # 一行三个子图的宽高比
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

for i, (model_name, results) in enumerate(models, 1):
    # 提取数据
    num_ignore = [item["num_ignore"] for item in results]
    accuracy = [item["accuracy"] * 100 for item in results]  # 转换为百分比
    token_mean = [item["token_mean"] / 1000 for item in results]  # 转换为千为单位
    
    # 创建子图
    ax = plt.subplot(1, 3, i)
    
    # 绘制准确率（左y轴）
    acc_line, = ax.plot(num_ignore, accuracy, '-', color='indigo', marker='o', markersize=4, 
                       markerfacecolor='indigo', linewidth=1.5)
    
    # 不在每个子图上添加xlabel和ylabel
    ax.tick_params(axis='y', labelcolor='indigo')
    
    # 设置网格线
    ax.grid(True, linestyle='--', alpha=0.3, color='lightgray')
    
    # 创建右y轴绘制平均token长度
    ax2 = ax.twinx()
    len_line, = ax2.plot(num_ignore, token_mean, '-', color='darkorange', marker='o', markersize=4, 
                        markerfacecolor='darkorange', linewidth=1.5)
    
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    # 添加标题
    ax.set_title(model_name, fontweight='bold')

# 添加统一的图例
plt.figlegend([acc_line, len_line], ['Accuracy', 'Response Length'], 
             loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)

# 添加单个x轴标签在底部
fig.text(0.5, 0.04, 'Budget Forcing Times (OlympiadBench)', ha='center', fontweight='bold')

# 添加左侧y轴标签
fig.text(0.02, 0.5, 'Accuracy (%)', va='center', rotation='vertical', 
         color='indigo', fontweight='bold')

# 添加右侧y轴标签
fig.text(0.98, 0.5, 'Response Length (K)', va='center', rotation='vertical', 
         color='darkorange', fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.07, right=0.93)  # 调整边距，为标签留出空间


# 保存为PNG格式
plt.savefig(os.path.join(SAVE_DIR, 'OlympiadBench_comparison_results.png'), dpi=300, bbox_inches='tight')

# # # 保存为PDF格式
# plt.savefig(os.path.join(SAVE_DIR, 'qwen2_5_vl_7b_sft_0401_results.pdf'), bbox_inches='tight')

plt.show()