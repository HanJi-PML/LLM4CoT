import warnings
warnings.filterwarnings("ignore")
import argparse
from utils import model_train_DDP, model_train_DSZ
import datetime
import torch
torch.cuda.empty_cache()

###### 多任务并行微调：每个任务独立训练 ######
dataset_name = ['Dataset3_sharegpt_training_12k_RArevised'] # Dataset3_sharegpt_training_10k
saved_name = ['Dataset3_12k_12k_RArevised']
num_samples = [12000] # 10000
current_adapter = None
training_mode = "DDP"
for i in range(len(dataset_name)):
    dataset_name_now = dataset_name[i]
    ########################  Training parameters  ########################
    # revise hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='Meta-Llama-3.1-8B-Instruct') # base-model
    parser.add_argument('--training_config', default='/data/LLM_indoor/LLaMA-Factory-main/examples/train_lora/llama3_lora_sft.yaml') # raw configuration yaml file
    out_dir = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/%s'%saved_name[i] + datetime.datetime.now().strftime("-%Y%m%d-%I%M%p")
    parser.add_argument('--expected_output_dir', default=out_dir) # saved model directory
    parser.add_argument('--dataset_name', default=dataset_name_now) # dataset directory
    parser.add_argument('--max_samples', default=num_samples[i], type=int) # number of training samples
    parser.add_argument('--epoch_num', default='3', type=int) # set training epoch number
    parser.add_argument('--cutoff_len', default=2048) # 4096训练时间太久（4小时）并且占现存太大
    parser.add_argument('--learning_rate', default=1.0e-4)
    parser.add_argument('--lora_rank', default=8) # supported max_lora_rank value for VLLM is 16
    parser.add_argument('--GPU_device', default='0,1,2,3', type=str) # e.g. 0-1 take 3 hours, 0-1-2-3 take 3 hour
    parser.add_argument('--deepspeed', default='examples/deepspeed/ds_z2_config_rong.json')
    parser.add_argument('--gradient_checkpointing', default=False, type=bool)
    args = parser.parse_args()
    if current_adapter:
        args.adapter_path = current_adapter
        print(f"Fine-turning from the last Adapter: {current_adapter}")
    else:
        args.adapter_path = None
        print(f"Fine-turning from the base model: {args.model_name_or_path}")
    # current_adapter = args.expected_output_dir # fine-tuning from the last SFT model
    # start training process
    if training_mode == "DDP":
        model_train_DDP(args)
    else:
        model_train_DSZ(args)
