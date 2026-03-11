import json

def convert_to_sharegpt(infile, outfile, task_name):
    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for sample in data:
        instruction = sample.get("instruction", "").strip()
        inp = sample.get("input", "").strip()
        out = sample.get("output", "").strip()

        conversations = [
            {
                "from": "human",
                "value": f"{instruction} {inp}"
            },
            {
                "from": "gpt",
                "value": out
            }
        ]

        new_data.append({
            "conversations": conversations,
            "tools": []
        })

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print("Saved:", outfile, " samples:", len(new_data))

source_sinr = "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_room3-6_task1.json"
source_aps = "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_room3-6_task2.json"
source_ra = "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_room3-6_task3.json"
new_sinr = "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_room3-6_task1_conversation.json"
new_aps = "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_room3-6_task2_conversation.json"
new_ra = "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_room3-6_task3_conversation.json"

convert_to_sharegpt(source_sinr, new_sinr, "SINR")
convert_to_sharegpt(source_aps, new_aps, "APS")
convert_to_sharegpt(source_ra, new_ra, "RA")


#%%
import json
import re
from copy import deepcopy

def merge_datasets(task1_path, task2_path, task3_path, output_path):
    # 加载数据
    with open(task1_path, 'r') as f:
        task1 = json.load(f)
    with open(task2_path, 'r') as f:
        task2 = json.load(f)
    with open(task3_path, 'r') as f:
        task3 = json.load(f)
    
    # 按最小长度处理
    min_len = min(len(task1), len(task2), len(task3))
    merged = []
    
    for i in range(min_len):
        # 复制任务1的基础对话
        merged_item = deepcopy(task1[i])
        
        # 从任务2提取rate和aps
        conv2 = task2[i]["conversations"]
        human2 = conv2[0]["value"]
        gpt2 = conv2[1]["value"]
        
        # 提取rate "data rate requirement vector for all users is"之后的内容）
        rate_start = human2.find("data rate requirement vector for all users is")
        if rate_start != -1:
            rate_text = human2[rate_start + len("data rate requirement vector for all users is"):].strip()
            # 取第一个向量
            rate_vec = rate_text.split('.Please')[0].strip()
        else:
            rate_vec = None
        
        # 提取aps "APS result is"之后的内容）
        aps_start = gpt2.find("APS result is")
        if aps_start != -1:
            aps_vec = gpt2[aps_start + len("APS result is"):].strip()
        else:
            aps_vec = None
        
        # 从任务3提取ra
        conv3 = task3[i]["conversations"]
        gpt3 = conv3[1]["value"]
        
        # 提取ra "RA result is"之后的内容）
        ra_start = gpt3.find("RA result is")
        if ra_start != -1:
            ra_mat = gpt3[ra_start + len("RA result is"):].strip()
        else:
            ra_mat = None
        
        # 追加对话
        merged_conv = merged_item["conversations"]
        
        if rate_vec and aps_vec:
            merged_conv.extend([
                {
                    "from": "human",
                    "value": f"And data rate requirement vector for all users is {rate_vec}. Please provide the optimal access point selection matrix for all users to obtain highest network capacity."
                },
                {
                    "from": "gpt",
                    "value": f"APS result is {aps_vec}"
                }
            ])
        
        if ra_mat:
            merged_conv.extend([
                {
                    "from": "human",
                    "value": "Please provide the optimal resource allocation results in matrix for all APs to obtain highest network capacity."
                },
                {
                    "from": "gpt",
                    "value": f"RA result is {ra_mat}"
                }
            ])
        
        merged.append(merged_item)
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

# 使用示例
merge_datasets(
    "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_room3-6_task1_conversation.json",
    "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_room3-6_task2_conversation.json", 
    "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_room3-6_task3_conversation.json",
    "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room3-6_combined_sharegpt.json"
)

'''
#%%
# split dataset into training and test datset
import json
import random

def split_train_test(input_path, train_output_path, test_output_path, train_size=10000):
    # 加载合并后的数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.shuffle(data)
    
    # 检查数据总长度
    total_size = len(data)
    
    # 分割数据
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"训练集: {len(train_data)}条")
    print(f"测试集: {len(test_data)}条")
    
    # 保存训练集
    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存测试集
    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

# 使用示例
split_train_test(
    input_path="/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_12k_combined_sharegpt.json",
    train_output_path="/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_sharegpt_training_10k.json", 
    test_output_path="/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/dataset3_sharegpt_test_2k.json",
    train_size = 10000
)
'''