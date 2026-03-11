import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import datetime
import os
import random

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    import torch
    torch.cuda.empty_cache()
    from utils_new import get_target_inout_from_samples_sequential, sequential_inference_RArevised

    # 对于70B的模型推理 只能执行一次推理后重新运行脚本
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_RArevised_Llama-3.2-1B-Instruct_Qlora_bf16-20251231-0120AM'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_RArevised_Llama-3.2-3B-Instruct_Qlora_bf16-20251231-0139AM'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_RArevised_Llama-3.3-70B-Instruct_Qlora_bf16-20251231-0219AM'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_RArevised_Llama-3.3-70B-Instruct_lora_bf16-20251231-0711PM'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_RArevised_Llama-3.2-1B-Instruct_full_bf16-20251230-1147PM'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_RArevised_Llama-3.2-3B-Instruct_full_bf16-20251230-1019PM'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10k_RArevised_Llama-3.1-8B-Instruct_full_bf16-20251230-0309PM'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10k_RArevised_Llama-3.2-1B-Instruct_lora_bf16'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_RArevised_Llama-3.2-3B-Instruct_lora_bf16-20251230-1239PM'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10k_RArevised_Llama-3.1-8B-Instruct_Qlora_bf16-20251230-0134PM'
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_12k_RArevised-20251229-1033AM'
    adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10k_RArevised-20251228-0228AM' # revised RA model with 10k samples
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_8k_RArevised-20251228-0904PM' # revised RA model with 5k samples
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_5k_RArevised-20251227-1156PM' # revised RA model with 5k samples
    # adapter_path = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_2k_RArevised-20251228-0514PM' # revised RA model with 2k samples
    
    save_path = '/data/LLM_indoor/LLaMA-Factory-main/LLM-train/Multitask/results/Dataset3_10k_RArevised_8B_lora%s/'%datetime.datetime.now().strftime("-%Y%m%d-%I%M%p")
    os.makedirs(save_path, exist_ok=True)
    
    test_datasets = [{'dataset_name': 'Dataset3_room1-1_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room1-1_combined_sharegpt.json'},
                    #{'dataset_name': 'Dataset3_room1-2_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room1-2_combined_sharegpt.json'},
                    {'dataset_name': 'Dataset3_room2-1_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room2-1_combined_sharegpt.json'},
                    #{'dataset_name': 'Dataset3_room2-2_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room2-2_combined_sharegpt.json'},
                    {'dataset_name': 'Dataset3_room3-1_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room3-1_combined_sharegpt.json'},
                    #{'dataset_name': 'Dataset3_room3-2_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room3-2_combined_sharegpt.json'},
                    {'dataset_name': 'Dataset3_room4-1_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room4-1_combined_sharegpt.json'},
                    #{'dataset_name': 'Dataset3_room4-2_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room4-2_combined_sharegpt.json'},
                    {'dataset_name': 'Dataset3_room5-1_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room5-1_combined_sharegpt.json'},
                    #{'dataset_name': 'Dataset3_room5-2_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room5-2_combined_sharegpt.json'},
                    {'dataset_name': 'Dataset3_room6-1_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room6-1_combined_sharegpt.json'},
                    #{'dataset_name': 'Dataset3_room6-2_sharegpt_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/dataset3_room6-2_combined_sharegpt.json'}
                    #{'dataset_name': 'Test1_Room1_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/Test1_room1-1_combined_sharegpt.json'},
                    #{'dataset_name': 'Test2_Room2_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/Test2_room4-3_combined_sharegpt.json'},
                    #{'dataset_name': 'Test4_Room4_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/Test4_room7_combined_sharegpt.json'},
                    #{'dataset_name': 'Test3_Room3_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/Test3_room4-4_combined_sharegpt.json'},
                    #{'dataset_name': 'Test5_Room5_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/Test5_room8_combined_sharegpt.json'},
                    #{'dataset_name': 'Test6_Room5_RArevised','dataset_path': '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset/Test6_room9_combined_sharegpt.json'}
                    ]

    # 调用适配器处理不同测试数据集
    for i in range(len(test_datasets)):
        parser = argparse.ArgumentParser()
        # ====== 基本参数 ======
        parser.add_argument('--model_name_or_path', default='/data/LLM_indoor/LLaMA-Factory-main/models/Llama-3.1-8B-Instruct')
        parser.add_argument('--adapter_name_or_path', default=adapter_path)
        parser.add_argument('--result_path', default='%s/%s'%(save_path, test_datasets[i]['dataset_name']))
        parser.add_argument('--evaluation_config', default='/data/LLM_indoor/LLaMA-Factory-main/examples/inference/llama3_lora_sft.yaml')
        # ====== vLLM参数 ======
        parser.add_argument('--vllm_dtype', default="bfloat16", type=str)
        parser.add_argument('--vllm_max_model_len', default=4096, type=int) # 所有对话前后总长度
        parser.add_argument('--vllm_disable_custom_all_reduce', default=True, type=bool)
        parser.add_argument('--vllm_tensor_parallel_size', default=1, type=int) # 70B-QLoRa需要设为2（2卡推理），70B-LoRA需要3卡推理
        parser.add_argument('--GPU_device', default='0,1', type=str)
        parser.add_argument('--vllm_gpu_memory_utilization', default=0.95, type=float)
        args = parser.parse_args()
    
        max_samples = 100
        # 加载测试数据集
        with open(test_datasets[i]['dataset_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        random.shuffle(data)
        used_datasets = data[0:max_samples]

        prompts, output_labels = get_target_inout_from_samples_sequential(used_datasets)

        sequential_inference_RArevised(args, prompts, output_labels)

if __name__ == "__main__":
    main()
