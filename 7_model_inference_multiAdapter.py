import warnings
warnings.filterwarnings("ignore")
import argparse
import datetime
import os
import random

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    import torch
    torch.cuda.empty_cache()
    from utils_new import split_dataset, sequencial_inference_multiAdapter

    adapters = [
        {'adapter_name': 'multiAdapter_task1',
        'adapter_path': '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_task1-8B-full-20260106-1239AM'},
        {'adapter_name': 'multiAdapter_task2',
        'adapter_path': '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_task2-8B-full-20260105-1120AM'},
        {'adapter_name': 'multiAdapter_task3',
        'adapter_path': '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_task3-8B-full-20260105-0341PM'}
    ]

    test_datasets = [
        {
            "room_type": "dataset3_room1-1",
            "dataset": [
                {'dataset_name': 'dataset3_room1-1_task1',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room1-1_task1.json'},
                {'dataset_name': 'dataset3_room1-1_task2',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room1-1_task2.json'},
                {'dataset_name': 'dataset3_room1-1_task3',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room1-1_task3.json'}
            ]
        },
        {
            "room_type": "dataset3_room2-1",
            "dataset": [
                {'dataset_name': 'dataset3_room2-1_task1',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room2-1_task1.json'},
                {'dataset_name': 'dataset3_room2-1_task2',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room2-1_task2.json'},
                {'dataset_name': 'dataset3_room2-1_task3',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room2-1_task3.json'}
            ]
        },
        {
            "room_type": "dataset3_room3-1",
            "dataset": [
                {'dataset_name': 'dataset3_room3-1_task1',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room3-1_task1.json'},
                {'dataset_name': 'dataset3_room3-1_task2',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room3-1_task2.json'},
                {'dataset_name': 'dataset3_room3-1_task3',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room3-1_task3.json'}
            ]
        },
        {
            "room_type": "dataset3_room4-1",
            "dataset": [
                {'dataset_name': 'dataset3_room4-1_task1',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room4-1_task1.json'},
                {'dataset_name': 'dataset3_room4-1_task2',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room4-1_task2.json'},
                {'dataset_name': 'dataset3_room4-1_task3',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room4-1_task3.json'}
            ]
        },
        {
            "room_type": "dataset3_room5-1",
            "dataset": [
                {'dataset_name': 'dataset3_room5-1_task1',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room5-1_task1.json'},
                {'dataset_name': 'dataset3_room5-1_task2',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room5-1_task2.json'},
                {'dataset_name': 'dataset3_room5-1_task3',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room5-1_task3.json'}
            ]
        },
        {
            "room_type": "dataset3_room6-1",
            "dataset": [
                {'dataset_name': 'dataset3_room6-1_task1',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room6-1_task1.json'},
                {'dataset_name': 'dataset3_room6-1_task2',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room6-1_task2.json'},
                {'dataset_name': 'dataset3_room6-1_task3',
                'dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room6-1_task3.json'}
            ]
        }
    ]

    save_path = '/data/LLM_indoor/LLaMA-Factory-main/LLM-train/Multitask/results/Dataset3_10k_RArevised_multiAdapter_8B_Full%s/'%datetime.datetime.now().strftime("-%Y%m%d-%I%M%p")
    os.makedirs(save_path, exist_ok=True)

    # 调用适配器处理不同测试数据集
    for i in range(len(test_datasets)):
        parser = argparse.ArgumentParser()
        # ====== 基本参数 ======
        parser.add_argument('--model_name_or_path', default=adapters)
        parser.add_argument('--datasets', default=test_datasets[i]['dataset'])
        parser.add_argument('--adapter_name_or_path', default=None)
        parser.add_argument('--result_path', default='%s/%s'%(save_path, test_datasets[i]['room_type'])) ###
        parser.add_argument('--evaluation_config', default='/data/LLM_indoor/LLaMA-Factory-main/examples/inference/llama3_lora_sft.yaml')
        # ====== vLLM参数 ======
        parser.add_argument('--vllm_dtype', default="bfloat16", type=str)
        parser.add_argument('--vllm_max_model_len', default=4096, type=int) # 所有对话前后总长度
        parser.add_argument('--vllm_disable_custom_all_reduce', default=True, type=bool)
        parser.add_argument('--vllm_tensor_parallel_size', default=2, type=int)
        parser.add_argument('--GPU_device', default='0,1,2,3', type=str)
        parser.add_argument('--vllm_gpu_memory_utilization', default=0.95, type=float)
        args = parser.parse_args()

        # 加载测试数据集
        used_datasets = split_dataset(args.datasets, 1000, 'train')
        # 测试数据集打乱
        n_samples = len(list(used_datasets.values())[0])
        indices = list(range(n_samples))
        random.shuffle(indices)
        for key in used_datasets:
            used_datasets[key] = [used_datasets[key][i] for i in indices]

        test_samples = 100
        used_datasets = {k: v[:test_samples] for k, v in used_datasets.items()}
        # 多适配器并行推理
        sequencial_inference_multiAdapter(args, used_datasets, test_samples)

if __name__ == "__main__":
    main()
