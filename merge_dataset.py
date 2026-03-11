import json
import os

# Merge datasets for different traces 
# 直接列出所有文件并手动分组
dataset_dir = "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/"
all_files = os.listdir(dataset_dir)
json_files = [f for f in all_files if f.endswith('.json') and f.startswith('Room3-6')]

sample_num = 10 # maximum samples used for each trace

# 按任务分组
tasks_data = {}
for filename in json_files:
    parts = filename.split('_')
    if len(parts) >= 2:
        task_name = parts[1]  # task1, task2, task3
        if task_name not in tasks_data:
            tasks_data[task_name] = []
        tasks_data[task_name].append(filename)

# 合并文件
for task, files in tasks_data.items():
    merged_data = []
    # 排序
    files.sort(key=lambda x: int(x.split('trace')[-1].split('.')[0]))
    
    for filename in files:
        filepath = os.path.join(dataset_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data[0:sample_num])  # 展平列表
    
    output_filename = os.path.join(dataset_dir, f"dataset3_room3-6_{task}.json")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    print(f"已创建: {output_filename}, 包含 {len(merged_data)} 个轨迹")

'''
#%% Merge datasets for different rooms 
dataset_dir = "/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/"
all_files = os.listdir(dataset_dir)
json_files = [f for f in all_files if f.endswith('.json') and f.startswith('dataset3')]

sample_num = 2000 # maximum samples used for each trace

# 按任务分组
tasks_data = {}
for filename in json_files:
    parts = filename.split('_')
    if len(parts) >= 2:
        task_name = parts[2]  # task1, task2, task3
        if task_name not in tasks_data:
            tasks_data[task_name] = []
        tasks_data[task_name].append(filename)

# 合并文件
for task, files in tasks_data.items():
    merged_data = []
    
    for filename in files:
        filepath = os.path.join(dataset_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data[0:sample_num])  # 展平列表
    
    output_filename = os.path.join(dataset_dir, f"dataset3_{task}_12k.json")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    print(f"已创建: {output_filename}, 包含 {len(merged_data)} 个轨迹")
'''
