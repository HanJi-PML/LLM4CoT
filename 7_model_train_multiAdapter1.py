import warnings
warnings.filterwarnings("ignore")
import argparse
import gc
import datetime
gc.collect()

# 70B模型10k数据集 LoRa 4卡训练预估需要18小时
# 70B模型10k数据集 QLoRa 4卡训练需要9.5小时
# 8B full 10k数据集 3卡训练 需要3小时
def main():
    import torch
    torch.cuda.empty_cache()
    from utils_new import model_train_DSZ_multi_finetuning
    # ####################### training variables #######################
    # dataset of training, name from 'data/dataset_info.json', eg: 'Dataset1-RA'
    dataset_name = ['Dataset3_sharegpt_training_12k_task1', 'Dataset3_sharegpt_training_12k_task2','Dataset3_sharegpt_training_12k_task3', 'Dataset3_sharegpt_training_12k_task1']
    # training samples for each dataset, eg: 4000
    num_samples = [10000, 10000, 10000, 10000]
    # base models of training, name from base models dir: 'models/', eg: 'SmolLM2-135M-Instruct', 'Llama-3.2-1B-Instruct', 'Llama-3.1-8B-Instruct', 'Llama-3.3-70B-Instruct'
    base_model_name = ['Llama-3.1-8B-Instruct', 'Llama-3.1-8B-Instruct','Llama-3.1-8B-Instruct', 'Llama-3.1-8B-Instruct'] 
    # instruct template corrsponding to the base model of training, eg: 'chatml', 'llama3', 'llama3', 'llama3'
    template = ['llama3', 'llama3', 'llama3', 'llama3']
    multi_fine_tuning = ['qlora-4bnb', 'qlora-4bnb', 'qlora-4bnb', 'full']
    lora_rank = [8, 8, 8, 8]
    batch_size = [12, 12, 12, 4]

    dsz3_gpu = '0,1' # model size > 30B / full 8B: 3卡
    dsz3_config_path = '/data/LLM_indoor/LLaMA-Factory-main/examples/deepspeed/ds_z3_config_rong.json'

    out_dir = ['/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_task1-8B-Qlora',
                '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_task2-8B-Qlora',
                '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_task3-8B-Qlora',
                '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/Dataset3_12k_10K_task1-8B-full']

    #####################################################################
    model_name_or_path = [f'/data/LLM_indoor/LLaMA-Factory-main/models/{name}' for name in base_model_name]
    for i in range(len(base_model_name)):
        parser = argparse.ArgumentParser()
        # ====== 基本参数设置 ======
        parser.add_argument('--model_name_or_path', default=model_name_or_path[i]) # base-model
        parser.add_argument('--template', default=template[i])
        parser.add_argument('--training_config', default='/data/LLM_indoor/LLaMA-Factory-main/examples/train_lora/llama3_lora_sft.yaml') # raw configuration yaml file
        parser.add_argument('--expected_output_dir', default=out_dir[i]+datetime.datetime.now().strftime("-%Y%m%d-%I%M%p")) # saved model directory
        parser.add_argument('--dataset_path', default=dataset_name[i]) # dataset directory
        parser.add_argument('--max_samples', default=num_samples[i]) # number of training samples
        parser.add_argument('--epoch_num', default=2) # set training epoch number
        parser.add_argument('--per_device_train_batch_size', default=batch_size[i]) # 70B-Lora 4卡训练需要设为4
        parser.add_argument('--gradient_accumulation_steps', default=1)
        parser.add_argument('--gradient_checkpointing', default=False)
        parser.add_argument('--GPU_device', default=dsz3_gpu)
        parser.add_argument('--lora_rank', default=lora_rank[i])
        parser.add_argument('--bf16', default=True, type=bool)
        parser.add_argument('--fp16', default=False, type=bool)
        parser.add_argument('--deepspeed', default=dsz3_config_path)
        # ====== 微调方法个性化设置 ======
        parser.add_argument('--fine_tuning_now', default=multi_fine_tuning[i])
        if multi_fine_tuning[i] == 'full':
            parser.add_argument('--finetuning_type', default='full')
        elif multi_fine_tuning[i] == 'lora':
            parser.add_argument('--finetuning_type', default='lora')
        elif multi_fine_tuning[i] == 'qlora-4bnb':
            parser.add_argument('--finetuning_type', default='lora')
            parser.add_argument('--quantization_bit', default=4)
            parser.add_argument('--quantization_method', default='bnb')
        elif multi_fine_tuning[i] == 'qlora-8bnb':
            parser.add_argument('--finetuning_type', default='lora')
            parser.add_argument('--quantization_bit', default=8)
            parser.add_argument('--quantization_method', default='bnb')

        args = parser.parse_args()
        # start training process
        model_train_DSZ_multi_finetuning(args)

if __name__ == "__main__":
    main()
