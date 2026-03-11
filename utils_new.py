import csv
import json
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
import time, datetime
import torch
torch.backends.cudnn.benchmark = True
import yaml
import os
import re
import math
import tempfile
import random
import numpy as np
from scipy.optimize import minimize
from datasets import load_dataset
import yaml
import pandas as pd
import ast
from llamafactory.chat.chat_model import ChatModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

############ 训练函数 ############
def model_train_DSZ(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_device
    os.environ["FORCE_TORCHRUN"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

    os.environ["TORCH_DISTRIBUTED_DEFAULT_BACKEND"] = "nccl"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_DEBUG"] = "warn"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_DISABLE"] = "0"

    with open(args.training_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['model_name_or_path'] = args.model_name_or_path
    config['output_dir'] = args.expected_output_dir
    config['dataset'] = args.dataset_path
    config['max_samples'] = args.max_samples
    config['num_train_epochs'] = args.epoch_num
    config['learning_rate'] = args.learning_rate
    config['lora_rank'] = args.lora_rank
    config['per_device_train_batch_size'] = args.per_device_train_batch_size
    config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    config['ddp_timeout'] = 1800        # 可选
    config['template'] = args.template  # 对话模板
    config['gradient_checkpointing'] = args.gradient_checkpointing
    config['deepspeed'] = args.deepspeed

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.dump(config, tmp, default_flow_style=False, allow_unicode=True)
        temp_config_path = tmp.name

    print("****** Starting training at " + datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p ******"))

    # 4) 直接调用 llamafactory-cli；它会根据可见 GPU 数量用 torchrun 启多进程并行
    env = os.environ.copy()
    process = subprocess.Popen(
        ["llamafactory-cli", "train", temp_config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()

    # 5) 结果与保存
    if process.returncode != 0:
        print("训练过程中出现错误:")
        try:
            # 再读一遍缓冲（若还有剩余）
            print(process.stdout.read())
        except Exception:
            pass
    else:
        time.sleep(1)
        if os.path.exists(args.expected_output_dir) and os.listdir(args.expected_output_dir):
            print("训练完成，检测到输出目录存在文件。")
            args_dict = vars(args)
            json_path = os.path.join(args.expected_output_dir, 'training_parameters.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(args_dict, f, indent=4, ensure_ascii=False)
            print(f"训练参数已保存至: {json_path}")
        else:
            print("训练完成，但输出目录不存在或为空。")
        print("****** Finish training at " + datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p ******"))

def model_train_DSZ_multi_finetuning(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_device
    os.environ["FORCE_TORCHRUN"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501" # sometimes need to revise this port number if multiple scripts are running
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

    os.environ["TORCH_DISTRIBUTED_DEFAULT_BACKEND"] = "nccl"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_DEBUG"] = "warn"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_DISABLE"] = "0"

    with open(args.training_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['model_name_or_path'] = args.model_name_or_path
    config['output_dir'] = args.expected_output_dir
    config['dataset'] = args.dataset_path
    config['max_samples'] = args.max_samples
    config['num_train_epochs'] = args.epoch_num
    config['per_device_train_batch_size'] = args.per_device_train_batch_size
    config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    config['ddp_timeout'] = 1800        # 可选
    config['template'] = args.template  # 对话模板
    config['gradient_checkpointing'] = args.gradient_checkpointing
    config['deepspeed'] = args.deepspeed
    config['lora_rank'] = args.lora_rank
    config['bf16'] = args.bf16
    config['fp16'] = args.fp16
    config['finetuning_type'] = args.finetuning_type

    if args.fine_tuning_now in ['qlora-4bnb', 'qlora-8bnb']:
        config['quantization_method'] = args.quantization_method
        config['quantization_bit'] = args.quantization_bit

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.dump(config, tmp, default_flow_style=False, allow_unicode=True)
        temp_config_path = tmp.name

    print("****** Starting training at " + datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p ******"))

    # 4) 直接调用 llamafactory-cli；它会根据可见 GPU 数量用 torchrun 启多进程并行
    env = os.environ.copy()
    process = subprocess.Popen(
        ["llamafactory-cli", "train", temp_config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()

    # 5) 结果与保存
    if process.returncode != 0:
        print("训练过程中出现错误:")
        try:
            # 再读一遍缓冲（若还有剩余）
            print(process.stdout.read())
        except Exception:
            pass
    else:
        time.sleep(1)
        if os.path.exists(args.expected_output_dir) and os.listdir(args.expected_output_dir):
            print("训练完成，检测到输出目录存在文件。")
            args_dict = vars(args)
            json_path = os.path.join(args.expected_output_dir, 'training_parameters.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(args_dict, f, indent=4, ensure_ascii=False)
            print(f"训练参数已保存至: {json_path}")
        else:
            print("训练完成，但输出目录不存在或为空。")
        print("****** Finish training at " + datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p ******"))


############ 推理函数 ############
def sequential_inference(args, prompts, outputs):
    # 定义vllm框架
    llm = LLM(
        model = args.model_name_or_path,
        tensor_parallel_size = args.vllm_tensor_parallel_size,
        dtype = args.vllm_dtype,
        max_model_len = args.vllm_max_model_len, # 前后对话长度
        gpu_memory_utilization = args.vllm_gpu_memory_utilization, # 限定占用显存比例
        disable_custom_all_reduce = args.vllm_disable_custom_all_reduce,
        enable_lora = True
    ) # max_lora_rank value for VLLM is 16
    # 创建LoRA请求
    lora_request = LoRARequest(
        args.adapter_name_or_path,  # 适配器名称，随便取
        1,                  # 适配器ID，随便取
        args.adapter_name_or_path        # 适配器路径
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=5120
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # 设置Llama的对话模板
    llama31_template = """{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"""
    tokenizer.chat_template = llama31_template

    # 开始推理
    history_theory_texts = [""] * len(prompts)
    history_infer_texts = [""] * len(prompts)
    history_messages = [[] for _ in range(len(prompts))]
    error_idx = {0: set(), 1: set(), 2: set()} # 不同任务错误数据样本索引
    error_ue_num = {0: [], 1: [], 2: []} # 不同任务错误数据样本对应的UE数
    result_paths = [] # 所有任务mae和cosine similarity结果以及throughput保存路径
    runtime_paths = [] # 所有任务runtime结果保存路径
    for task_idx, task in enumerate(["task1", "task2", "task3"]):
        print('\n============ Current Inference Task is: %s ============' % task)
        # 将历史输入输出和新prompt拼接作为总输入
        input_prompts = []
        formatted_prompts = []

        # 自定义构建对话模板
        for idx, prompt in enumerate(prompts):
            # 批量推理下实际输入的prompt不能简单字符串拼接，要保留多轮对话的格式
            messages_now = history_messages[idx] + [{"role": "user", "content": prompt[task]}]
            input_prompts.append(messages_now)
            formatted_prompts.append(build_prompt_from_messages(tokenizer, messages_now))

        ###### 执行推理 ######
        full_texts = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)

        ###### 评估性能 ######
        # 定义当前任务各结果保存路径
        os.makedirs(args.result_path, exist_ok=True)
        runtime_path = f'{args.result_path}/runtime_{task}.csv'
        result_path = f'{args.result_path}/result_{task}.csv'
        result_paths.append(result_path)
        runtime_paths.append(runtime_path)

        # 统计所有样本平均推理的token数和TPS
        all_samples_runtime = []
        all_samples_result = []

        for idx, (prompt, target_text, full_text) in enumerate(zip(prompts, outputs, full_texts)):
            # 统计输入token
            formatted_prompt_tokens = full_text.prompt_token_ids
            prompt_tokens_num = len(formatted_prompt_tokens[31:])

            # 规范输出
            full_output = full_text.outputs[0].text
            full_header = "<|start_header_id|>assistant<|end_header_id|>"
            if full_header in full_output:
                gen_text = full_output.split(full_header, 1)[1].strip()
            else:
                gen_text = full_output.strip()

            # 将输出更新到历史文本中以供下次输入 (对话格式)
            history_messages[idx].append({"role": "user", "content": prompt[task]})
            history_messages[idx].append({"role": "assistant", "content": gen_text}) # 使用预测的输出作为下一个任务,会有错误传播

            # 统计历史所有理论和实际的输入输出 (字符串格式)
            history_theory_texts[idx] += prompt[task] + target_text[task] # 评估元适配器在多任务上的预测性能
            history_infer_texts[idx] += prompt[task] + gen_text

            # 统计输出token
            gen_text_tokens = full_text.outputs[0].token_ids
            gen_text_tokens_num = len(gen_text_tokens[4:])

            # 统计推理时间
            metrics = full_text.metrics
            sample_inference_time = metrics.finished_time - metrics.first_token_time

            # 输出日志
            print("***** Sample: %s *****" % idx)
            print("*** Input Prompt *** %s" % input_prompts[idx])
            print("*** Input Token Num *** %d" % prompt_tokens_num)
            # print("*** Full Text *** %s" % full_text)
            print("*** Target Output *** %s" % target_text[task])
            print("*** Predict Output *** %s" % gen_text)
            print("*** Predict Output Token Num *** %s" % gen_text_tokens_num)
            print("*** Inference Time *** %s" % sample_inference_time)
            print("-" * 50)
            try:
                # task1: SINR Estimation
                if task == "task1":
                    evaluate_result_SINR(idx, history_theory_texts[idx], target_text[task], gen_text, result_path, all_samples_result)
                # task2: AP Selection
                elif task == "task2":
                    evaluate_result_APS(idx, history_theory_texts[idx], target_text[task], gen_text, result_path, all_samples_result)
                # task3: Resource Allocation
                elif task == "task3":
                    evaluate_result_RA(idx, history_theory_texts[idx], target_text[task], gen_text, result_path, all_samples_result)
            # 推理出错误数据(不合规范的数据格式)
            except ValueError as error_info:
                print(f"❌ 样本{idx}在任务{task}计算通信性能时有ValueError, 报错信息如下:\n{error_info}")
                msg = str(error_info)
                error_ue_num[task_idx].append(int(msg.split("UE_num=")[-1]))
                error_idx[task_idx].add(idx)
                continue
            # 评估推理速度性能
            evaluate_runtime1(idx, gen_text_tokens_num, sample_inference_time, runtime_path, all_samples_runtime)

    # 数据对齐，去除所有任务文件中的错误样本
    print(f"错误数据索引集合error_idx: {error_idx} \n错误数据UE数列表error_ue_num: {error_ue_num}")
    all_error_idx = set()
    for task_key in error_idx:  # error_idx 是一个字典 {0: set(), 1: set(), 2: set()}
        all_error_idx.update(error_idx[task_key])

    # sorted_error_idx = sorted(all_error_idx) # 原all_error_idx是一个集合, index乱序被检索
    # for result_path_now, runtime_path_now in zip(result_paths, runtime_paths):
    #     del_error_samples(result_path_now, sorted_error_idx) # 错误在于删除样本部分,重复删除了某些samples导致报错: 已解决
    #     del_error_samples(runtime_path_now, sorted_error_idx)

    # 优化: 传入删除函数的错误索引字典中除去了当前任务自己的错误索引, 因为文件中本来就不存在这些索引的样本, 不影响结果
    # 目的: 输出的删除索引有区分，可以直接看出各个任务执行的正确率, 真正的"删除"
    for task_idx, (result_path_now, runtime_path_now) in enumerate(zip(result_paths, runtime_paths)):
        delete_idx_now = all_error_idx - error_idx[task_idx]
        sorted_error_idx = sorted(delete_idx_now)
        del_error_samples(result_path_now, sorted_error_idx)
        del_error_samples(runtime_path_now, sorted_error_idx)

    history_theory_texts = [val for i, val in enumerate(history_theory_texts) if i not in all_error_idx]

    # 使用所有推理值计算网络吞吐量
    thr_path = f'{args.result_path}/result_throuput.csv'
    result_paths.append(thr_path)
    evalulate_throughput(history_theory_texts, result_paths, thr_path)

    # 定义文件路径
    file_paths = {
        'task1': result_paths[0],
        'task2': result_paths[1],
        'task3': result_paths[2],
        'Network': result_paths[3]
    }
    # 输出文件路径
    output_file = f'{args.result_path}/final_average_results.csv'
    file_metrics = {
        'task1': ['Cosine_Similarity', 'MAE'],
        'task2': ['UE_num', 'Accuracy Value'],
        'task3': ['Cosine_Similarity', 'MAE'],
        'Network': ['Performance_Gap']
    }
    # 计算所有推理样本的平均统计指标
    calculate_average(file_metrics, file_paths, output_file)



def sequential_inference_RArevised(args, prompts, outputs):
    # 修改任务3的RA数据格式,除去无关的0值后,把二维向量变成一维
    llm_kwargs = {
    "model": args.model_name_or_path,
    "tensor_parallel_size": args.vllm_tensor_parallel_size,
    "dtype": args.vllm_dtype,
    "max_model_len": args.vllm_max_model_len,
    "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
    "disable_custom_all_reduce": args.vllm_disable_custom_all_reduce}

    use_lora = hasattr(args, 'adapter_name_or_path') and args.adapter_name_or_path
    if use_lora:
        llm_kwargs["enable_lora"] = True

    llm = LLM(**llm_kwargs) # max_lora_rank value for VLLM is 16

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=5120
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # 设置Llama的对话模板
    llama31_template = """{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"""
    tokenizer.chat_template = llama31_template

    if use_lora:
        # 创建LoRA请求
        lora_request = LoRARequest(
            args.adapter_name_or_path,  # 适配器名称
            1,                          # 适配器ID
            args.adapter_name_or_path   # 适配器路径
        )

    # 开始推理
    history_theory_texts = [""] * len(prompts)
    history_infer_texts = [""] * len(prompts)
    history_messages = [[] for _ in range(len(prompts))]
    error_idx = {0: set(), 1: set(), 2: set()} # 不同任务错误数据样本索引
    error_ue_num = {0: [], 1: [], 2: []} # 不同任务错误数据样本对应的UE数
    result_paths = [] # 所有任务mae和cosine similarity结果以及throughput保存路径
    runtime_paths = [] # 所有任务runtime结果保存路径
    for task_idx, task in enumerate(["task1", "task2", "task3"]):
        print('\n============ Current Inference Task is: %s ============' % task)
        # 将历史输入输出和新prompt拼接作为总输入
        input_prompts = []
        formatted_prompts = []

        # 自定义构建对话模板
        for idx, prompt in enumerate(prompts):
            # 批量推理下实际输入的prompt不能简单字符串拼接，要保留多轮对话的格式
            messages_now = history_messages[idx] + [{"role": "user", "content": prompt[task]}]
            input_prompts.append(messages_now)
            formatted_prompts.append(build_prompt_from_messages(tokenizer, messages_now))

        ###### 执行推理 ######
        if use_lora:
            full_texts = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
        else:
            full_texts = llm.generate(formatted_prompts, sampling_params)

        ###### 评估性能 ######
        # 定义当前任务各结果保存路径
        os.makedirs(args.result_path, exist_ok=True)
        runtime_path = f'{args.result_path}/runtime_{task}.csv'
        result_path = f'{args.result_path}/result_{task}.csv'
        result_paths.append(result_path)
        runtime_paths.append(runtime_path)

        # 统计所有样本平均推理的token数和TPS
        all_samples_runtime = []
        all_samples_result = []

        for idx, (prompt, target_text, full_text) in enumerate(zip(prompts, outputs, full_texts)):
            # 统计输入token
            formatted_prompt_tokens = full_text.prompt_token_ids
            prompt_tokens_num = len(formatted_prompt_tokens[31:])

            # 统计输出token
            gen_text_tokens = full_text.outputs[0].token_ids
            gen_text_tokens_num = len(gen_text_tokens[4:])

            # 统计推理时间
            metrics = full_text.metrics
            sample_inference_time = metrics.finished_time - metrics.first_token_time

            # 规范输出
            full_output = full_text.outputs[0].text
            full_header = "<|start_header_id|>assistant<|end_header_id|>"
            if full_header in full_output:
                gen_text = full_output.split(full_header, 1)[1].strip()
            else:
                gen_text = full_output.strip()

            # 将输出更新到历史文本中以供下次输入 (对话格式)
            history_messages[idx].append({"role": "user", "content": prompt[task]})
            history_messages[idx].append({"role": "assistant", "content": gen_text}) # 使用预测的输出作为下一个任务,会有错误传播
            # 统计历史所有理论和实际的输入输出 (字符串格式)
            history_theory_texts[idx] += prompt[task] + target_text[task] # 评估元适配器在多任务上的预测性能
            history_infer_texts[idx] += prompt[task] + gen_text
            # 统计输出token
            gen_text_tokens = full_text.outputs[0].token_ids
            gen_text_tokens_num = len(gen_text_tokens[4:])

            # 统计推理时间
            metrics = full_text.metrics
            sample_inference_time = metrics.finished_time - metrics.first_token_time

            # 输出日志
            print("\n******************** Sample: %s ********************" % idx)
            print("*** Input Prompt *** %s" % prompt)
            print("*** Input Token Num *** %d" % prompt_tokens_num)
            # print("*** Full Text *** %s" % full_text)
            print("*** Target Output *** %s" % target_text)
            print("*** Predict Output *** %s" % gen_text)
            print("*** Predict Output Token Num *** %s" % gen_text_tokens_num)
            print("*** Inference Time *** %s" % sample_inference_time)
            print("-" * 50)

            try:
                # task1: SINR Estimation
                if task == "task1":
                    evaluate_result_SINR(idx, history_theory_texts[idx], target_text[task], gen_text, result_path, all_samples_result)
                # task2: AP Selection
                elif task == "task2":
                    evaluate_result_APS(idx, history_theory_texts[idx], target_text[task], gen_text, result_path, all_samples_result)
                # task3: Resource Allocation
                elif task == "task3":
                    revised_gen_text = restore_ra_matrix_multiAdapter(history_infer_texts[idx], gen_text)
                    evaluate_result_RA(idx, history_theory_texts[idx], target_text[task], revised_gen_text, result_path, all_samples_result)
            # 推理出错误数据(不合规范的数据格式)
            except ValueError as error_info:
                print(f"❌ 样本{idx}在任务{task}计算通信性能时有ValueError, 报错信息如下:\n{error_info}")
                msg = str(error_info)
                error_ue_num[task_idx].append(int(msg.split("UE_num=")[-1]))
                error_idx[task_idx].add(idx)
                continue
            # 评估推理速度性能
            evaluate_runtime1(idx, gen_text_tokens_num, sample_inference_time, runtime_path, all_samples_runtime)

    # 数据对齐，去除所有任务文件中的错误样本
    print(f"错误数据索引集合error_idx: {error_idx} \n错误数据UE数列表error_ue_num: {error_ue_num}")
    all_error_idx = set()
    for task_key in error_idx:  # error_idx 是一个字典 {0: set(), 1: set(), 2: set()}
        all_error_idx.update(error_idx[task_key])

    # 优化: 传入删除函数的错误索引字典中除去了当前任务自己的错误索引, 因为文件中本来就不存在这些索引的样本, 不影响结果
    # 目的: 输出的删除索引有区分，可以直接看出各个任务执行的正确率, 真正的"删除"
    for task_idx, (result_path_now, runtime_path_now) in enumerate(zip(result_paths, runtime_paths)):
        delete_idx_now = all_error_idx - error_idx[task_idx]
        sorted_error_idx = sorted(delete_idx_now)
        del_error_samples(result_path_now, sorted_error_idx)
        del_error_samples(runtime_path_now, sorted_error_idx)

    history_theory_texts = [val for i, val in enumerate(history_theory_texts) if i not in all_error_idx]

    # 使用所有推理值计算网络吞吐量
    thr_path = f'{args.result_path}/result_throuput.csv'
    result_paths.append(thr_path)
    evalulate_throughput(history_theory_texts, result_paths, thr_path)

    # 定义文件路径
    file_paths = {
        'task1': result_paths[0],
        'task2': result_paths[1],
        'task3': result_paths[2],
        'Network': result_paths[3]
    }
    # 输出文件路径
    output_file = f'{args.result_path}/final_average_results.csv'
    file_metrics = {
        'task1': ['Cosine_Similarity', 'MAE'],
        'task2': ['UE_num', 'Accuracy Value'],
        'task3': ['Cosine_Similarity', 'MAE'],
        'Network': ['Performance_Gap']
    }
    # 计算所有推理样本的平均统计指标
    calculate_average(file_metrics, file_paths, output_file)


# 多适配器按样本并行推理,按任务串行推理
def sequencial_inference_multiAdapter(args, used_datasets, max_samples):
    history_theory_texts = [""] * max_samples
    history_infer_texts = [""] * max_samples
    error_idx = {0: set(), 1: set(), 2: set()} # 不同任务错误数据样本索引
    error_ue_num = {0: [], 1: [], 2: []} # 不同任务错误数据样本对应的UE数
    result_paths = [] # 所有任务mse和cosine similarity结果以及throughput保存路径
    runtime_paths = [] # 所有任务runtime结果保存路径

    # 不能放在任务循环里多次初始化, 显存会爆
    llm = LLM(
        model = args.model_name_or_path,
        tensor_parallel_size = args.vllm_tensor_parallel_size,
        dtype = args.vllm_dtype,
        max_model_len = args.vllm_max_model_len, # 前后对话长度
        gpu_memory_utilization = args.vllm_gpu_memory_utilization, # 限定占用显存比例
        disable_custom_all_reduce = args.vllm_disable_custom_all_reduce,
        enable_lora = True
    ) # max_lora_rank value for VLLM is 16

    # 分任务批量处理
    for task_idx in range(len(args.datasets)):
        task = args.datasets[task_idx]["dataset_name"]
        adapter = args.adapter_name_or_path[task_idx]["adapter_path"]
        print('\n============ Current Inference Task is: %s ============' % task)
        # 提取样本输入输出
        # 多适配器串行/并行推理
        samples = used_datasets[task]
        prompts, outputs = get_target_inout_from_samples(samples)

        if task_idx == 1: # replace SINR using the prediced output of task1
            revised_SINR_prompts = []
            for index, sample in enumerate(prompts):
                pattern = r"SINR matrix is (\[\[.*?\]\])"
                match = re.search(pattern, history_infer_texts[index], re.DOTALL)
                new_matrix_content = match.group(1)
                result = re.sub(
                    r"all user's SINR matrix in dB is \[\[.*?\]\]",
                    f"all user's SINR matrix in dB is {new_matrix_content}",
                    sample,
                    flags=re.DOTALL)
                revised_SINR_prompts.append(result)
            prompts = revised_SINR_prompts
        elif task_idx == 2: # replace APS using the prediced output of task2
            revised_APS_prompts = []
            for index, sample in enumerate(prompts):
                pattern = r"APS result is (\[.*?\])"
                match = re.search(pattern, history_infer_texts[index], re.DOTALL)
                new_matrix_content = match.group(1)
                result = re.sub(
                    r"access point selection vector is \[.*?\]",
                    f"access point selection vector is {new_matrix_content}",
                    sample,
                    flags=re.DOTALL)
                revised_APS_prompts.append(result)
            prompts = revised_APS_prompts
        else:
            pass

        # 创建LoRA请求
        lora_request = LoRARequest(
            adapter,  # 适配器名称
            task_idx+1,  # 适配器ID, 不能乱取, 不能取0 (代表基座模型)
            adapter   # 适配器路径
        )
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=5120
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        # 设置Llama的对话模板
        llama31_template = """{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"""
        tokenizer.chat_template = llama31_template

        # 链式推理,下一个任务的输入prompts用的是上一个任务的预测输出
        formatted_prompts = [build_prompt(tokenizer, p) for p in prompts]

        full_texts = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)

        ###### 评估性能 ######
        os.makedirs(args.result_path, exist_ok=True)
        runtime_path = f'{args.result_path}/runtime_{task}.csv'
        result_path = f'{args.result_path}/result_{task}.csv'
        result_paths.append(result_path)
        runtime_paths.append(runtime_path)

        # 统计所有样本推理的token数和TPS
        # all_samples_runtime = []
        all_samples_result = []

        # 分样本单个处理
        for idx, (prompt, target_text, full_text) in enumerate(zip(prompts, outputs, full_texts)):
            # 统计输入token
            formatted_prompt_tokens = full_text.prompt_token_ids
            prompt_tokens_num = len(formatted_prompt_tokens[31:]) # why 31?

            # 规范输出
            full_output = full_text.outputs[0].text

            full_header = "<|start_header_id|>assistant<|end_header_id|>"
            if full_header in full_output:
                gen_text = full_output.split(full_header, 1)[1].strip()
            else:
                gen_text = full_output.strip()

            # 统计输出token
            gen_text_tokens = full_text.outputs[0].token_ids
            gen_text_tokens_num = len(gen_text_tokens[4:]) # why 4?

            # 统计推理时间
            metrics = full_text.metrics
            sample_inference_time = metrics.finished_time - metrics.first_token_time

            if task_idx == 1:  # APS任务
                # 检查预测输出是否包含错误的标签
                invalid_patterns = ['<|start_header_id|>assistant', '<|end_header_id|>', '<|eot_id|>']
                is_invalid = any(pattern in gen_text for pattern in invalid_patterns)
                # 检查是否包含有效的APS结果格式
                has_aps_pattern = re.search(r'APS result is \[.*?\]', gen_text) or re.search(r'\[[\d\s]+\]', gen_text)
                # 如果输出无效，记录错误并跳过该样本
                if is_invalid or not has_aps_pattern:
                    print(f"❌ 样本{idx}在任务{task}输出无效: {gen_text}")
                    error_idx[task_idx].add(idx)
                    error_ue_num[task_idx].append(0)  # 未知UE数，设为0
                    # 使用占位符文本，避免后续处理出错
                    gen_text = "APS result is []"

            if task_idx == 0:
                # 直接修正文本
                if 'SINR matrix is [' in gen_text and not gen_text.endswith(']]'):
                    # 找到最后一个]的位置
                    last_bracket = gen_text.rfind(']')
                    if last_bracket != -1:
                        # 在最后一个]后面再加一个]
                        gen_text = gen_text[:last_bracket+1] + ']' + gen_text[last_bracket+1:]
                history_infer_texts[idx] += prompt + gen_text
            elif task_idx == 1:
                gen_text = gen_text if gen_text.endswith(']') else gen_text + ']'
                history_infer_texts[idx] += gen_text
            else:
                # 直接修正文本
                gen_text = gen_text if gen_text.endswith(']') else gen_text + ']'
                history_theory_texts[idx] += prompt + target_text # history_theory_texts is only used to get R info in evalulate_throughput

            # 输出日志
            print("\n******************** Sample: %s ********************" % idx)
            print("*** Input Prompt *** %s" % prompt)
            print("*** Input Token Num *** %d" % prompt_tokens_num)
            # print("*** Full Text *** %s" % full_text)
            print("*** Target Output *** %s" % target_text)
            print("*** Predict Output *** %s" % gen_text)
            print("*** Predict Output Token Num *** %s" % gen_text_tokens_num)
            print("*** Inference Time *** %s" % sample_inference_time)
            print("-" * 50)

            # 评估通信性能
            try:
                # task1: SINR Estimation
                if task_idx == 0:
                    evaluate_result_SINR(idx, prompt, target_text, gen_text, result_path, all_samples_result)
                # task2: AP Selection
                elif task_idx == 1:
                    evaluate_result_APS(idx, prompt, target_text, gen_text, result_path, all_samples_result)
                # task3: Resource Allocation
                elif task_idx == 2:
                    revised_gen_text = restore_ra_matrix_multiAdapter(history_infer_texts[idx], gen_text)
                    evaluate_result_RA(idx, prompt, target_text, revised_gen_text, result_path, all_samples_result)
            # 推理出错误数据(不合规范的数据格式)
            except ValueError as error_info:
                print(f"❌ 样本{idx}在任务{task}计算通信性能时有ValueError, 报错信息如下:\n{error_info}")
                msg = str(error_info)
                error_ue_num[task_idx].append(int(msg.split("UE_num=")[-1]))
                error_idx[task_idx].add(idx)
                continue

    # 数据对齐，去除所有任务文件中的错误样本
    print(f"错误数据索引集合error_idx: {error_idx} \n错误数据UE数列表error_ue_num: {error_ue_num}")
    all_error_idx = set()
    for task_key in error_idx:  # error_idx 是一个字典 {0: set(), 1: set(), 2: set()}
        all_error_idx.update(error_idx[task_key])

    for task_idx, (result_path_now, runtime_path_now) in enumerate(zip(result_paths, runtime_paths)):
        delete_idx_now = all_error_idx - error_idx[task_idx]
        sorted_error_idx = sorted(delete_idx_now)
        del_error_samples(result_path_now, sorted_error_idx)
        # del_error_samples(runtime_path_now, sorted_error_idx)

    history_theory_texts = [val for i, val in enumerate(history_theory_texts) if i not in all_error_idx]

    # 使用所有推理值计算网络吞吐量
    thr_path = f'{args.result_path}/result_throuput.csv'
    result_paths.append(thr_path)
    evalulate_throughput(history_theory_texts, result_paths, thr_path)

    # 定义文件路径
    file_paths = {
        'task1': result_paths[0],
        'task2': result_paths[1],
        'task3': result_paths[2],
        'Network': result_paths[3]
    }
    # 输出文件路径
    output_file = f'{args.result_path}/final_average_results.csv'
    file_metrics = {
        'task1': ['Cosine_Similarity', 'MAE'],
        'task2': ['UE_num', 'Accuracy Value'],
        'task3': ['Cosine_Similarity', 'MAE'],
        'Network': ['Performance_Gap']
    }
    # 计算所有推理样本的平均统计指标
    calculate_average(file_metrics, file_paths, output_file)


def sequencial_inference_multiAdapter_fullModels(args, used_datasets, max_samples):
    """
    全参数微调模型的链式推理函数（优化版）
    一次性加载所有模型到不同GPU,避免重复加载
    """
    global global_models, global_tokenizers

    history_theory_texts = [""] * max_samples
    history_infer_texts = [""] * max_samples
    error_idx = {0: set(), 1: set(), 2: set()}  # 不同任务错误数据样本索引
    error_ue_num = {0: [], 1: [], 2: []}  # 不同任务错误数据样本对应的UE数
    result_paths = []  # 所有任务mse和cosine similarity结果以及throughput保存路径
    runtime_paths = []  # 所有任务runtime结果保存路径

    # 获取模型路径列表
    model_paths = [model_info["adapter_path"] for model_info in args.model_name_or_path]

    # 一次性加载所有模型（如果尚未加载）
    if len(global_models) != len(model_paths):
        models, tokenizers = load_models_once(model_paths, gpu_ids=[0, 1])
    else:
        models = global_models
        tokenizers = global_tokenizers

    # 分任务批量处理
    for task_idx in range(len(args.datasets)):
        task = args.datasets[task_idx]["dataset_name"]
        print('\n' + '='*60)
        print(f'Current Inference Task: {task} (Task {task_idx+1})')
        print('='*60)

        # 提取样本输入输出
        samples = used_datasets[task]
        prompts, outputs = get_target_inout_from_samples(samples)

        # 链式推理：使用上一个任务的输出修正当前任务的输入
        if task_idx == 1:  # replace SINR using the predicted output of task1
            revised_SINR_prompts = []
            for index, sample in enumerate(prompts):
                pattern = r"SINR matrix is (\[\[.*?\]\])"
                match = re.search(pattern, history_infer_texts[index], re.DOTALL)
                if match:
                    new_matrix_content = match.group(1)
                    result = re.sub(
                        r"all user's SINR matrix in dB is \[\[.*?\]\]",
                        f"all user's SINR matrix in dB is {new_matrix_content}",
                        sample,
                        flags=re.DOTALL)
                    revised_SINR_prompts.append(result)
                else:
                    print(f"Warning: No SINR matrix found in history for sample {index}, using original prompt")
                    revised_SINR_prompts.append(sample)
            prompts = revised_SINR_prompts
        elif task_idx == 2:  # replace APS using the predicted output of task2
            revised_APS_prompts = []
            for index, sample in enumerate(prompts):
                pattern = r"APS result is (\[.*?\])"
                match = re.search(pattern, history_infer_texts[index], re.DOTALL)
                if match:
                    new_matrix_content = match.group(1)
                    result = re.sub(
                        r"access point selection vector is \[.*?\]",
                        f"access point selection vector is {new_matrix_content}",
                        sample,
                        flags=re.DOTALL)
                    revised_APS_prompts.append(result)
                else:
                    print(f"Warning: No APS result found in history for sample {index}, using original prompt")
                    revised_APS_prompts.append(sample)
            prompts = revised_APS_prompts

        # 获取当前任务对应的已加载模型和tokenizer
        llm = models[task_idx]
        tokenizer = tokenizers[task_idx]

        # 设置Llama的对话模板
        llama31_template = """{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"""
        # alpaca_template = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Instruction:\n' + message['content'] + '\n\n### Response:\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}"""
        tokenizer.chat_template = llama31_template

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=4096
        )

        # 链式推理
        formatted_prompts = [build_prompt(tokenizer, p) for p in prompts]
        print(f"Starting inference for {len(formatted_prompts)} samples...")
        full_texts = llm.generate(formatted_prompts, sampling_params)

        ###### 评估性能 ######
        os.makedirs(args.result_path, exist_ok=True)
        runtime_path = f'{args.result_path}/runtime_{task}.csv'
        result_path = f'{args.result_path}/result_{task}.csv'
        result_paths.append(result_path)
        runtime_paths.append(runtime_path)

        all_samples_result = []

        # 分样本单个处理
        for idx, (prompt, target_text, full_text) in enumerate(zip(prompts, outputs, full_texts)):
            # 统计输入token
            formatted_prompt_tokens = full_text.prompt_token_ids
            prompt_tokens_num = len(formatted_prompt_tokens[31:])

            # 规范输出
            full_output = full_text.outputs[0].text
            full_header = "<|start_header_id|>assistant<|end_header_id|>"
            if full_header in full_output:
                gen_text = full_output.split(full_header, 1)[1].strip()
            else:
                gen_text = full_output.strip()

            # 统计输出token
            gen_text_tokens = full_text.outputs[0].token_ids
            gen_text_tokens_num = len(gen_text_tokens[4:]) # why 4?

            # 统计推理时间
            metrics = full_text.metrics
            # sample_inference_time = metrics.finished_time - metrics.first_token_time

            if task_idx == 1:  # APS任务
                # 检查预测输出是否包含错误的标签
                invalid_patterns = ['<|start_header_id|>assistant', '<|end_header_id|>', '<|eot_id|>']
                is_invalid = any(pattern in gen_text for pattern in invalid_patterns)
                # 检查是否包含有效的APS结果格式
                has_aps_pattern = re.search(r'APS result is \[.*?\]', gen_text) or re.search(r'\[[\d\s]+\]', gen_text)
                # 如果输出无效，记录错误并跳过该样本
                if is_invalid or not has_aps_pattern:
                    print(f"❌ 样本{idx}在任务{task}输出无效: {gen_text[:100]}...")
                    error_idx[task_idx].add(idx)
                    error_ue_num[task_idx].append(0)  # 未知UE数，设为0
                    # 使用占位符文本，避免后续处理出错
                    gen_text = "APS result is []"

            if task_idx == 0:
                # 直接修正文本
                if 'SINR matrix is [' in gen_text and not gen_text.endswith(']]'):
                    # 找到最后一个]的位置
                    last_bracket = gen_text.rfind(']')
                    if last_bracket != -1:
                        # 在最后一个]后面再加一个]
                        gen_text = gen_text[:last_bracket+1] + ']' + gen_text[last_bracket+1:]
                history_infer_texts[idx] += prompt + gen_text
            elif task_idx == 1:
                gen_text = gen_text if gen_text.endswith(']') else gen_text + ']'
                history_infer_texts[idx] += gen_text
            else:
                # 直接修正文本
                gen_text = gen_text if gen_text.endswith(']') else gen_text + ']'
                history_theory_texts[idx] += prompt + target_text  # history_theory_texts is only used to get R info in evalulate_throughput

            # 输出日志
            print("\n******************** Sample: %s ********************" % idx)
            print("*** Input Prompt *** %s" % prompt)
            print("*** Input Token Num *** %d" % prompt_tokens_num)
            # print("*** Full Text *** %s" % full_text)
            print("*** Target Output *** %s" % target_text)
            print("*** Predict Output *** %s" % gen_text)
            print("*** Predict Output Token Num *** %s" % gen_text_tokens_num)
            # print("*** Inference Time *** %s" % sample_inference_time)
            print("-" * 50)

            # 评估通信性能
            try:
                # task1: SINR Estimation
                if task_idx == 0:
                    evaluate_result_SINR(idx, prompt, target_text, gen_text, result_path, all_samples_result)
                # task2: AP Selection
                elif task_idx == 1:
                    evaluate_result_APS(idx, prompt, target_text, gen_text, result_path, all_samples_result)
                # task3: Resource Allocation
                elif task_idx == 2:
                    revised_gen_text = restore_ra_matrix_multiAdapter(history_infer_texts[idx], gen_text)
                    evaluate_result_RA(idx, prompt, target_text, revised_gen_text, result_path, all_samples_result)
            # 推理出错误数据(不合规范的数据格式)
            except ValueError as error_info:
                print(f"❌ 样本{idx}在任务{task}计算通信性能时有ValueError, 报错信息如下:\n{error_info}")
                msg = str(error_info)
                if "UE_num=" in msg:
                    error_ue_num[task_idx].append(int(msg.split("UE_num=")[-1]))
                else:
                    error_ue_num[task_idx].append(0)
                error_idx[task_idx].add(idx)
                continue
            except Exception as e:
                print(f"❌ 样本{idx}在任务{task}计算通信性能时有其他错误: {e}")
                error_idx[task_idx].add(idx)
                error_ue_num[task_idx].append(0)
                continue

        print(f"✓ Task {task_idx+1} ({task}) completed")
        time.sleep(10)

    # 数据对齐，去除所有任务文件中的错误样本
    print(f"\n错误数据统计:")
    print(f"错误数据索引集合error_idx: {error_idx}")
    print(f"错误数据UE数列表error_ue_num: {error_ue_num}")

    all_error_idx = set()
    for task_key in error_idx:
        all_error_idx.update(error_idx[task_key])

    for task_idx, (result_path_now, runtime_path_now) in enumerate(zip(result_paths, runtime_paths)):
        delete_idx_now = all_error_idx - error_idx[task_idx]
        sorted_error_idx = sorted(delete_idx_now)
        if os.path.exists(result_path_now):
            del_error_samples(result_path_now, sorted_error_idx)

    history_theory_texts = [val for i, val in enumerate(history_theory_texts) if i not in all_error_idx]

    # 使用所有推理值计算网络吞吐量
    thr_path = f'{args.result_path}/result_throuput.csv'
    result_paths.append(thr_path)
    evalulate_throughput(history_theory_texts, result_paths, thr_path)

    # 定义文件路径
    file_paths = {
        'task1': result_paths[0],
        'task2': result_paths[1],
        'task3': result_paths[2],
        'Network': result_paths[3]
    }
    # 输出文件路径
    output_file = f'{args.result_path}/final_average_results.csv'
    file_metrics = {
        'task1': ['Cosine_Similarity', 'MAE'],
        'task2': ['UE_num', 'Accuracy Value'],
        'task3': ['Cosine_Similarity', 'MAE'],
        'Network': ['Performance_Gap']
    }
    # 计算所有推理样本的平均统计指标
    calculate_average(file_metrics, file_paths, output_file)
    print(f"Results saved to: {args.result_path}")


global_models = {}
global_tokenizers = {}
def load_models_once(model_paths, gpu_ids=None):
    """
    一次性加载所有模型到不同的GPU上
    model_paths: 模型路径列表 [task1_path, task2_path, task3_path]
    gpu_ids: 指定每个模型使用的GPU ID列表,如 [0, 1]
    """
    global global_models, global_tokenizers
    if gpu_ids is None:
        gpu_ids = [0, 1]  # 默认使用前三张卡
    if len(global_models) == len(model_paths):
        print("Models already loaded, skipping loading...")
        return global_models, global_tokenizers

    for task_idx, model_path in enumerate(model_paths):
        # gpu_id = gpu_ids[task_idx % len(gpu_ids)]
        gpu_id = [0, 1]
        print(f"\nLoading model {task_idx+1} to GPU {gpu_id}:")
        print(f"Model path: {model_path}")
        # 设置CUDA设备
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
        try:
            # 加载模型到指定GPU
            llm = LLM(
                model=model_path,
                tensor_parallel_size=1,  # 每个模型只用1张卡
                dtype="bfloat16",
                max_model_len=4096,
                gpu_memory_utilization=0.3,
                disable_custom_all_reduce=True,
                trust_remote_code=True
            )
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # 设置对话模板
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is None:
                llama31_template = """{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"""
                tokenizer.chat_template = llama31_template

            global_models[task_idx] = llm
            global_tokenizers[task_idx] = tokenizer

            print(f"✓ Model {task_idx+1} loaded successfully to GPU {gpu_id}")

        except Exception as e:
            print(f"✗ Failed to load model {task_idx+1} to GPU {gpu_id}: {e}")
            raise

    # 恢复所有GPU可见
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    return global_models, global_tokenizers


def restore_ra_matrix(prompt, predicted_ra_text):
    """
    从文本输入重建RA二维矩阵文本
    """
    AP_num, UE_num, _ = extract_ue_info1(prompt)
    try:
        aps_match = re.search(r'APS result is (\[[^\]]+\])', prompt)
        if aps_match is None:
            raise ValueError(f"APS result pattern not found in prompt.")
        aps_str = aps_match.group(1)
        aps_str = "[" + ",".join(aps_str.strip("[]").split()) + "]"
        # 解析RA和APS列表
        ra_list = eval(predicted_ra_text.split("is")[1].strip())
        aps_list = eval(aps_str)
        # 当任务2预测的APS结果的维度出错时,这个函数也会报错,需要修改这里
        revised_pred_aps = [min(ap, AP_num) for ap in aps_list] # APS预测值不超过最大值
        # revise APS to avoid dimension mismatch
        if len(revised_pred_aps) < UE_num or len(ra_list) < UE_num:
            raise ValueError(f"APS or RA dim mismatch, UE_num={UE_num}") # when dim of pred_aps is smaller than target_aps
        revised_aps_list = revised_pred_aps[0:UE_num] # when dim of pred_aps is larger than target_aps
        revised_ra_list = ra_list[0:UE_num]

        ra_matrix = [[0.0]*AP_num for _ in range(len(revised_aps_list))]
        for i, val in enumerate(revised_ra_list):
            ra_matrix[i][revised_aps_list[i]-1] = val
        # 生成输出 - 完整的二维矩阵，使用双括号包裹
        matrix_rows = []
        for row in ra_matrix:
            # 使用列表推导式格式化每个数字
            row_str = " ".join(["0.0" if x == 0 else f"{x:.2f}" for x in row])
            matrix_rows.append(f"[{row_str}]")
        return f"RA result is [{' '.join(matrix_rows)}]"
    except ValueError as e:
        raise ValueError(f"APS or RA dim mismatch, UE_num={UE_num}") from e


def restore_ra_matrix_multiAdapter(prompt, predicted_ra_text):
    """
    从文本输入重建RA二维矩阵文本
    """
    AP_num, UE_num, _ = extract_ue_info1(prompt)
    try:
        aps_match = re.search(r'APS result is (\[[^\]]+\])', prompt)
        if aps_match is None:
            raise ValueError(f"APS result pattern not found in prompt.")
        aps_str = aps_match.group(1)
        aps_str = "[" + ",".join(aps_str.strip("[]").split()) + "]"

        # 安全地提取RA数组
        ra_str = None
        # 尝试匹配 "RA result is [...]" 格式
        ra_match = re.search(r'RA result is\s*(\[.*?\])', predicted_ra_text)
        if ra_match:
            ra_str = ra_match.group(1)
        else:
            # 尝试直接匹配数组格式
            array_match = re.search(r'(\[.*?\])', predicted_ra_text)
            if array_match:
                ra_str = array_match.group(1)
            else:
                raise ValueError(f"Cannot find RA array in: {predicted_ra_text}")

        # 安全解析RA字符串
        ra_list = []
        try:
            ra_list = eval(ra_str)
        except:
            # 手动解析
            clean = ra_str.strip('[]').replace(',', ' ')
            parts = clean.split()
            ra_list = [float(x) if x.replace('.', '', 1).isdigit() else 0.0 for x in parts if x]

        aps_list = eval(aps_str)
        revised_pred_aps = [min(ap, AP_num) for ap in aps_list]

        if len(revised_pred_aps) < UE_num or len(ra_list) < UE_num:
            raise ValueError(f"APS or RA dim mismatch, UE_num={UE_num}")

        revised_aps_list = revised_pred_aps[0:UE_num]
        revised_ra_list = ra_list[0:UE_num]

        ra_matrix = [[0.0]*AP_num for _ in range(len(revised_aps_list))]
        for i, val in enumerate(revised_ra_list):
            ra_matrix[i][revised_aps_list[i]-1] = val

        matrix_rows = []
        for row in ra_matrix:
            row_str = " ".join(["0.0" if x == 0 else f"{x:.2f}" for x in row])
            matrix_rows.append(f"[{row_str}]")

        return f"RA result is [{' '.join(matrix_rows)}]"
    except Exception as e:
        if "UE_num" not in str(e):
            raise ValueError(f"Error in restore_ra_matrix: {str(e)}, UE_num={UE_num}") from e
        raise


def reward_dataset_collection(args, prompts, outputs):
    # dataset template for reward model
    """
    [{
    'instruction': 'human instruction (required)',
    'input': 'human input (optional)',
    'chosen': 'chosen answer (required)', labeled three tasks
    'rejected': 'rejected answer (required)' predicted output for three tasks
    }]
    """
    # 定义vllm框架
    llm = LLM(
        model = args.model_name_or_path,
        tensor_parallel_size = args.vllm_tensor_parallel_size,
        dtype = args.vllm_dtype,
        max_model_len = args.vllm_max_model_len, # 前后对话长度
        gpu_memory_utilization = args.vllm_gpu_memory_utilization, # 限定占用显存比例
        disable_custom_all_reduce = args.vllm_disable_custom_all_reduce,
        enable_lora = True
    )
    # 创建LoRA请求
    lora_request = LoRARequest(
        args.adapter_name_or_path,  # 适配器名称，随便取
        1,                  # 适配器ID，随便取
        args.adapter_name_or_path        # 适配器路径
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    history_theory_texts = [""] * len(prompts)
    history_infer_texts = [""] * len(prompts)
    history_messages = [[] for _ in range(len(prompts))]
    performance_gap_list = []

    for task in ["task1", "task2", "task3"]:
        print('\n============ Current Inference Task is: %s ============' % task)
        # 将历史输入输出和新prompt拼接作为总输入
        input_prompts = []
        formatted_prompts = []

        # 自定义构建对话模板
        for idx, prompt in enumerate(prompts):
            # 批量推理下实际输入的prompt不能简单字符串拼接，要保留多轮对话的格式
            messages_now = history_messages[idx] + [{"role": "user", "content": prompt[task]}]
            input_prompts.append(messages_now)
            formatted_prompts.append(build_prompt_from_messages(tokenizer, messages_now))

        ###### 执行推理 ######
        print("Start using adapter for inference...")
        full_texts = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)

        ###### 评估性能 ######
        for idx, (prompt, target_text, full_text) in enumerate(zip(prompts, outputs, full_texts)):
            # 统计输入token
            formatted_prompt_tokens = full_text.prompt_token_ids
            # 规范输出
            full_output = full_text.outputs[0].text # predicted output
            full_header = "<|start_header_id|>assistant<|end_header_id|>"
            if full_header in full_output:
                gen_text = full_output.split(full_header, 1)[1].strip()
            else:
                gen_text = full_output.strip()

            # 将输出更新到历史文本中以供下次输入 (对话格式)
            history_messages[idx].append({"role": "user", "content": prompt[task]})
            history_messages[idx].append({"role": "assistant", "content": gen_text})

            # 统计历史所有理论和实际的输入输出 (字符串格式)
            history_theory_texts[idx] += prompt[task] + target_text[task]
            history_infer_texts[idx] += prompt[task] + gen_text

            # calculate performance gap
            if task == "task3":
                # 计算理论throuput用样本中的，计算推理throughput用推理历史中的
                performance_gap = evaluate_performance(history_theory_texts[idx], history_infer_texts[idx]) # 检查是否正确？
                performance_gap_list.append(performance_gap)

    # save reward dataset
    reward_dataset = []
    for label_output, predicted_output in zip(history_theory_texts, history_infer_texts):
        # 提取instruction部分
        instruction_pattern = r'^(You are.*?interference\.)'
        instruction_match = re.search(instruction_pattern, label_output, re.DOTALL)
        instruction = instruction_match.group(1) if instruction_match else ""

        # 提取完整的input部分
        input_pattern1 = r'(At time step .*?\]\]\.)'
        input_match1 = re.search(input_pattern1, label_output, re.DOTALL)
        input_text1 = input_match1.group(1) if input_match1 else ""
        input_pattern2 = r'(And data rate.*?\]\.)'

        input_match2 = re.search(input_pattern2, label_output, re.DOTALL)
        input_text2 = input_match2.group(1) if input_match2 else ""

        input_text3 = 'Please estimate the signal-to-noise-plus-interference ratio (SINR) in dB for all users. Please provide the optimal access point selection matrix for all users to obtain highest network capacity. Please provide the optimal resource allocation results in matrix for all APs to obtain highest network capacity.'
        input_text = input_text1 + input_text2 + input_text3

        # 提取chosen部分
        SINR_pattern = r'SINR matrix is \[\[(.*?)\]\]'
        APS_pattern = r'APS result is \[(.*?)\]'
        RA_pattern = r'RA result is \[\[(.*?)\]\]'

        SINR_match = re.search(SINR_pattern, label_output)
        APS_match = re.search(APS_pattern, label_output)
        RA_match = re.search(RA_pattern, label_output)

        SINR_result = f"SINR matrix is [[{SINR_match.group(1)}]]" if SINR_match else ""
        APS_result = f"APS result is [{APS_match.group(1)}]" if APS_match else ""
        RA_result = f"RA result is [[{RA_match.group(1)}]]" if RA_match else ""

        chosen = f"{SINR_result} {APS_result} {RA_result}"

        # 提取rejected部分
        SINR_pattern = r'SINR matrix is \[\[(.*?)\]\]'
        APS_pattern = r'APS result is \[(.*?)\]'
        RA_pattern = r'RA result is \[\[(.*?)\]\]'

        SINR_match = re.search(SINR_pattern, predicted_output)
        APS_match = re.search(APS_pattern, predicted_output)
        RA_match = re.search(RA_pattern, predicted_output)

        SINR_result = f"SINR matrix is [[{SINR_match.group(1)}]]" if SINR_match else ""
        APS_result = f"APS result is [{APS_match.group(1)}]" if APS_match else ""
        RA_result = f"RA result is [[{RA_match.group(1)}]]" if RA_match else ""

        rejected = f"{SINR_result} {APS_result} {RA_result}"

        # 构建样本
        sample = {
            "instruction": instruction,
            "input": input_text,
            "chosen": chosen.strip(),
            "rejected": rejected}
        reward_dataset.append(sample)

    # sort reward dataset samples that has worst 50% performance gap
    sorted_indices = np.argsort(performance_gap_list)[::-1]
    selected_indices = sorted_indices[:len(performance_gap_list)//2]
    filtered_reward_dataset = [reward_dataset[i] for i in selected_indices]

    return reward_dataset, filtered_reward_dataset


def build_prompt(tokenizer, text):
    # 构造 messages：交给 llama3 模板生成 formatted prompt
    messages = [{"role":"user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_prompt_from_messages(tokenizer, messages):
    # 并行情况下要实现链式推理，不能简单将历史输入输出字符串拼接，要构建对话格式
    """
    messages: List[{"role": "user"|"assistant", "content": str}]
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

####################### 性能评估函数 #######################
def split_dataset(datasets, max_samples, mod, split_ratio=0.8):
    # [修改] 以len(datasets) * split_radio为分界，前面是训练区，后面是测试区
    # 训练调用时 (mod='train') 数据集为训练区的前max_samples条数据
    # 测试调用时 (mod='infer') 数据集为测试区的前max_samples条数据
    used_datasets = {}
    for dataset in datasets:
        dataset_name_now = dataset['dataset_name']
        dataset_path = dataset['dataset_path']
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 计算分割点
        split_idx = int(len(data) * split_ratio)
        train_dataset = data[:split_idx]
        infer_dataset = data[split_idx:]

        # 获取数据集
        if mod == 'train':
            if max_samples <= len(train_dataset):
                used_datasets[dataset_name_now] = train_dataset[:max_samples]
                print(f'训练数据集: {max_samples} 条')
            else:
                used_datasets[dataset_name_now] = train_dataset
                print(f"训练数据集长度不能超过 {len(train_dataset)}, 已使用全部的{len(train_dataset)}条训练数据集")
        elif mod == 'infer':
            if max_samples <= len(infer_dataset):
                used_datasets[dataset_name_now] = infer_dataset[:max_samples]
                print(f'测试数据集: {max_samples} 条')
            else:
                used_datasets[dataset_name_now] = infer_dataset
                print(f"测试数据集长度不能超过 {len(infer_dataset)}, 已使用全部的{len(infer_dataset)}条测试数据集")
        else:
            continue

    return used_datasets


def get_target_inout_from_samples(samples):
    # 每个样本一个字典，其中包含1个任务的prompt
    prompts = []
    outputs = []
    for sample in samples:
        sample_instruction = sample.get('instruction', '')
        sample_input = sample.get('input', '')
        sample_output = sample.get('output', '')
        sample_prompt = sample_instruction + sample_input
        prompts.append(sample_prompt)
        outputs.append(sample_output)
    return prompts, outputs


def get_target_inout_from_samples_sequential(samples):
    # 每个样本一个字典，其中包含3个任务的prompt
    """
    prompts / outputs = [
        {
            "task1": "...",
            "task2": "...",
            "task3": "..."
        },
        ...
    ]
    """
    prompts = []
    outputs = []
    for sample in samples:
        conversations = sample.get("conversations", [])
        prompts.append({
            "task1": conversations[0].get("value", ""),
            "task2": conversations[2].get("value", ""),
            "task3": conversations[4].get("value", "")
        })
        outputs.append({
            "task1": conversations[1].get("value", ""),
            "task2": conversations[3].get("value", ""),
            "task3": conversations[5].get("value", "")
        })

    return prompts, outputs


def extract_ue_info(text, seq_text = None):
    num_match = re.search(r'There are (\d+) APs serving (\d+) users', text)
    AP_num = int(num_match.group(1)) if num_match else None
    UE_num = int(num_match.group(2)) if num_match else None

    # 单独推理和链式推理的理论值, R都一样
    SINR_match = re.search(r'(SINR matrix is|SINR matrix in dB is) \s*\[\[(.*?)\]\]', text)
    if SINR_match:
        matrix_str = SINR_match.group(2)
        rows = matrix_str.split('] [')
        SINR = [list(map(float, row.split())) for row in rows]
    else:
        SINR = []

    R_match = re.search(r'data rate requirement vector for all users is \s*\[([\d.,\s]+)\]', text)
    if R_match:
        R = list(map(float, R_match.group(1).split()))
    else:
        R = []

    X_iu_match = re.search(r'(APS result is|access point selection vector is) \s*\[([\d.,\s]+)\]', text)
    if X_iu_match:
        X_iu = list(map(int, X_iu_match.group(2).split()))
    else:
        X_iu = []

    # 链式推理的推理值, R都一样
    if seq_text:
        # SINR
        SINR_match = re.search(r'SINR matrix is \s*\[\[(.*?)\]\]', text)
        if SINR_match:
            matrix_str = SINR_match.group(1)
            rows = matrix_str.split('] [')
            seq_SINR_pred = [list(map(float, row.split())) for row in rows]
        else:
            seq_SINR_pred  = []

        # APS
        X_iu_match = re.search(r'APS result is \s*\[([\d.,\s]+)\]', seq_text)
        if X_iu_match:
            seq_X_iu_pred = list(map(int, X_iu_match.group(1).split()))
        else:
            seq_X_iu_pred = []
    else:
        seq_SINR_pred, seq_X_iu_pred = [], []

    return AP_num, UE_num, X_iu, R, SINR, seq_SINR_pred, seq_X_iu_pred


def extract_ue_info1(sample_text):
    num_match = re.search(r'There are (\d+) APs serving (\d+) users', sample_text)
    AP_num = int(num_match.group(1)) if num_match else None
    UE_num = int(num_match.group(2)) if num_match else None

    R_match = re.search(r'data rate requirement vector for all users is \s*\[([\d.,\s]+)\]', sample_text)
    if R_match:
        R = list(map(float, R_match.group(1).split()))
        # print("R:", R)
    else:
        R = []
    return AP_num, UE_num, R


def evaluate_result_SINR(sample_idx, prompt, target_text, gen_text, result_path, all_samples_result):
    # 提取SINR
    AP_num, UE_num, R = extract_ue_info1(prompt)
    try:
        target_sinr = get_sinr_from_output_text(target_text)
        pred_sinr = get_sinr_from_output_text(gen_text)

        # revise SINR to avoid dimension mismatch
        if len(pred_sinr) < len(target_sinr):
            raise ValueError(f"SINR dim mismatch, UE_num={UE_num}") # when dim of pred_sinr is smaller than target_sinr
        pred_sinr_revised = pred_sinr[0:UE_num] # when dim of pred_sinr is larger than target_sinr

        # 计算MAE, cosine
        MAE_now = mae(pred_sinr_revised, target_sinr)
        cosine_now = cosine_similarity(pred_sinr_revised, target_sinr)

        print("********** Current Cosine Similarity: %s **********"%cosine_now)

        # 将结果写入csv
        # 单个样本值
        sample_header = [
            "Index", # keep index, can not revise this variable
            "UE_num",
            "Target_SINR_Estimation",
            "Pred_SINR_Estimation",
            "Cosine_Similarity",
            "MAE"
        ]
        sample_stats = {
            "Index": sample_idx,
            "UE_num": UE_num,
            "Target_SINR_Estimation": target_sinr,
            "Pred_SINR_Estimation": pred_sinr_revised,
            "Cosine_Similarity": cosine_now,
            "MAE": MAE_now
        }
        all_samples_result.append(sample_stats)
        save_csv_result(result_path, sample_header, all_samples_result, 'w')
    except ValueError as e:
        raise ValueError(f"SINR Error, UE_num={UE_num}") from e



def get_sinr_from_output_text(text):
    match = re.search(r'SINR matrix is \s*\[\[(.*?)(?=\]\]|[A-Z][a-z]+)', text)
    if match:
        matrix_str = match.group(1)
        rows = matrix_str.split('] [')
        sinr = [list(map(float, row.split())) for row in rows]
        return sinr
    else:
        print("No matching values")
        return []


def evaluate_result_APS(sample_idx, prompt, target_text, gen_text, result_path, all_samples_result):
    # 提取APS
    AP_num, UE_num, R = extract_ue_info1(prompt)
    try:
        target_aps = get_aps_from_output_text(target_text)
        pred_aps = get_aps_from_output_text(gen_text)
        revised_pred_aps = [min(ap, AP_num) for ap in pred_aps] # APS预测值不超过最大值

        # revise APS to avoid dimension mismatch
        if len(revised_pred_aps) < len(target_aps):
            raise ValueError(f"APS dim mismatch, UE_num={UE_num}") # when dim of pred_aps is smaller than target_aps
        pred_aps_revised = revised_pred_aps[0:UE_num] # when dim of pred_aps is larger than target_aps

        # 计算Top1-accuracy
        accuracy_list, accuracy_value = accuracy(pred_aps_revised, target_aps)

        print("********** Current Accuracy List: %s **********" % accuracy_list)
        print("********** Current Accuracy Value: %s **********" % accuracy_value)

        # 将结果写入csv
        # 单个样本值
        sample_header = [
            "Index", # keep index, can not revise this variable
            "UE_num",
            "Target_APS",
            "Pred_APS",
            "Accuracy List",
            "Accuracy Value"
        ]
        sample_stats = {
            "Index": sample_idx,
            "UE_num": UE_num,
            "Target_APS": target_aps,
            "Pred_APS": pred_aps_revised,
            "Accuracy List": accuracy_list,
            "Accuracy Value": accuracy_value
        }
        all_samples_result.append(sample_stats)
        save_csv_result(result_path, sample_header, all_samples_result, 'w')
    except ValueError as e:
        raise ValueError(f"SINR Error, UE_num={UE_num}") from e


def get_aps_from_output_text(text):
    match = re.search(r'APS result is \s*\[(.*?)(?=\]|[A-Z][a-z]+)', text)
    if match:
        aps_str = match.group(1)
        aps = list(map(int, aps_str.split()))
        return aps
    else:
        print("No matching values")
        return []


def accuracy(pred_list, target_list):
    accuracy_list = [1 if p == t else 0 for p, t in (zip(pred_list, target_list))]
    accuracy_value = sum(accuracy_list) / len(accuracy_list) if len(accuracy_list) > 0 else 0
    return accuracy_list, accuracy_value


def evaluate_result_RA(sample_idx, prompt, target_text, gen_text, result_path, all_samples_result):
    # 提取Rho
    AP_num, UE_num, R = extract_ue_info1(prompt)
    try:
        target_Rho = get_rho_from_output_text(target_text)
        pred_Rho = get_rho_from_output_text(gen_text)

        # revise Rho to avoid dimension mismatch
        if len(pred_Rho) < len(target_Rho):
            raise ValueError(f"RA dim mismatch, UE_num={UE_num}") # when dim of pred_Rho is smaller than target_Rho
        pred_Rho_revised = pred_Rho[0:UE_num] # when dim of pred_Rho is larger than target_Rho

        MAE_now = mae(pred_Rho_revised, target_Rho)
        cosine_rho = cosine_similarity(pred_Rho_revised, target_Rho)

        print("********** Current Cosine Similarity: %s **********"%cosine_rho)
        sample_header = [
            "Index", # keep index, can not revise this variable
            "UE_num",
            "Target_Rho",
            "Pred_Rho",
            "Cosine_Similarity",
            "MAE"
        ]
        sample_stats = {
            "Index": sample_idx,
            "UE_num": UE_num,
            "Target_Rho": target_Rho,
            "Pred_Rho": pred_Rho_revised,
            "Cosine_Similarity": cosine_rho,
            "MAE": MAE_now
        }
        all_samples_result.append(sample_stats)
        save_csv_result(result_path, sample_header, all_samples_result, 'w')
    except ValueError as e:
        raise ValueError(f"SINR Error, UE_num={UE_num}") from e


def evaluate_performance(theory_text, infer_text):
    # 提取Rho
    AP_num, UE_num, target_X_iu, R, target_SINR, [], [] = extract_ue_info(theory_text)
    _, _, pred_X_iu, _, pred_SINR, _, _ = extract_ue_info(infer_text)
    target_Rho = get_rho_from_output_text(theory_text)
    pred_Rho = get_rho_from_output_text(infer_text)

    nor_target_Rho = normalize_rho(target_X_iu, target_Rho, AP_num, UE_num)
    target_thr = throughput_cal(R, target_X_iu, target_SINR, nor_target_Rho)

    # 链式推理用APS和SINR推理值计算最终预测的throughput
    nor_pred_Rho = normalize_rho(pred_X_iu, pred_Rho, AP_num, UE_num)
    pred_thr = throughput_cal(R, pred_X_iu, pred_SINR, nor_pred_Rho)

    performance_gap = (target_thr - pred_thr)/target_thr

    print("********** Current throughput gap: %s **********"%performance_gap)
    return performance_gap


def get_rho_from_output_text(text):
    match = re.search(r'RA result is \s*\[\[(.*?)(?=\]\]|[A-Z][a-z]+)', text)
    if match:
        matrix_str = match.group(1)
        rows = matrix_str.split('] [')
        Rho = [list(map(float, row.split())) for row in rows]
        return Rho
    else:
        print("No matching values")
        return []


def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def mape(y_true, y_pred):
    # this metric has problem, not reliable
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((y_true - y_pred) / y_true) * 100
        mape_values = np.where(np.isnan(mape_values) | np.isinf(mape_values), 0, mape_values)
    return np.mean(mape_values)

def mae(y_true, y_pred):
    # 计算平均绝对误差
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae_values = np.abs(y_true - y_pred)  # 计算每个点的绝对误差
    return np.mean(mae_values)           # 计算所有绝对误差的平均值


def cosine_similarity(list1, list2):
    vec1 = np.array(list1)
    vec2 = np.array(list2)
    if vec1.ndim == 1 and vec2.ndim == 1:
        # input is 1-D vector
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)
    elif vec1.ndim == 2 and vec2.ndim == 2 and vec1.shape == vec2.shape:
        # input is 2-D matrix
        dot_products = np.sum(vec1 * vec2, axis=1)
        norms1 = np.linalg.norm(vec1, axis=1)
        norms2 = np.linalg.norm(vec2, axis=1)
        # avoid zero as divisor
        norms_product = norms1 * norms2
        norms_product[norms_product == 0] = 1
        similarities = dot_products / norms_product
        return np.mean(similarities)
    else:
        raise ValueError("输入必须是相同形状的一维向量或二维数组。")


def normalize_rho(X_iu, Rho, AP_num, UE_num):
    # TCP和MPTCP合二为一
    # 输出：TCP: 标准化的各UE的Rho一维列表; MPTCP: 二维列表
    # TCP
    if isinstance(X_iu[0], int):
        # RA的rho样式:一维列表
        if isinstance(Rho[0], float):
            Rho_matrix = np.zeros((AP_num, UE_num))
            for col, sel_AP in enumerate(X_iu):
                if 1 <= sel_AP <= UE_num:
                    Rho_matrix[sel_AP - 1, col] = Rho[col]
            row_sums = Rho_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            mat_normalized = Rho_matrix / row_sums
            nor_Rho = mat_normalized.sum(axis=0)
            return nor_Rho.tolist()
        # RA的rho样式:二维列表
        elif isinstance(Rho[0], list):
            Rho_matrix = np.zeros((AP_num, UE_num))
            for col, sel_AP in enumerate(X_iu):
                if 1 <= sel_AP <= UE_num:
                    Rho_matrix[sel_AP - 1, col] = Rho[col][sel_AP-1]
            row_sums = Rho_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            mat_normalized = Rho_matrix / row_sums
            nor_Rho = mat_normalized.sum(axis=0)
            return nor_Rho.tolist()

    # MPTCP
    elif isinstance(X_iu[0], list):
        Rho_matrix = np.zeros((AP_num, UE_num))
        for col1, scenario in enumerate(X_iu):
            for col2 in range(len(scenario)):
                # If the selected AP index is within the valid range, update the corresponding position in the matrix with the UE's rho value
                sel_AP = scenario[col2]
                if 1 <= sel_AP <= UE_num:
                    Rho_matrix[sel_AP - 1, col1] = Rho[col1][col2]

        row_sums = Rho_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat_normalized = Rho_matrix / row_sums
        return mat_normalized.T.tolist()


def throughput_cal(R, X_iu, SINR, Rho):
    # TCP和MPTCP合二为一
    # TCP
    if isinstance(X_iu[0], int):
        LiFi_BW = 40  # Mbps
        WiFi_BW = 20  # Mbps
        UE_num = len(R)
        thr_list = []
        SINR_linear = [[10 ** (x/10) for x in user_sinr] for user_sinr in SINR]
        for ue_idx, X_iu_now in enumerate(X_iu):
            ap_idx = X_iu_now - 1
            sinr = SINR_linear[ue_idx][ap_idx]
            rho = Rho[ue_idx]
            if X_iu_now == 1:
                # WiFi容量计算
                capacity = WiFi_BW * math.log2(1 + sinr) * rho
            else:
                # LiFi容量计算
                factor = math.exp(1) / (2 * math.pi)
                capacity = (LiFi_BW / 2) * math.log2(1 + factor * sinr) * rho

            user_throughput = min(capacity, R[ue_idx])
            thr_list.append(user_throughput)

        total_throughput = sum(thr_list)  # Mbps
        return total_throughput

    # MPTCP
    elif isinstance(X_iu[0], list):
        LiFi_BW = 40 # Mbps
        WiFi_BW = 20 # Mbps
        UE_num = len(R)
        thr_list = []
        SINR_linear = [[10 ** (x/10) for x in user_sinr] for user_sinr in SINR]
        for ue_idx, scenario in enumerate(X_iu):
            user_throughput = 0
            R_now = R[ue_idx]
            for ap_idx, X_iu_now in enumerate(scenario):
                sinr = SINR_linear[ue_idx][ap_idx]
                rho = Rho[ue_idx][ap_idx]
                if X_iu_now == 1:
                    # WiFi capacity
                    capacity = WiFi_BW * math.log2(1 + sinr) * rho
                else:
                    # LiFi capacity
                    factor = math.exp(1) / (2 * math.pi)
                    capacity = (LiFi_BW / 2) * math.log2(1 + factor * sinr) * rho
                user_throughput += capacity

            user_throughput = min(user_throughput, R_now)
            thr_list.append(user_throughput)

        thr = sum(thr_list)
        return thr


def save_csv_result(file_path, header, text, mode='w'):
    with open(file_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        # mode='w'时写入表头，mode='a'时只写内容
        if mode == 'w':
            writer.writeheader()
        # text为字典时单行写入，text为列表时多行写入
        if isinstance(text, dict):
            writer.writerow(text)
        elif isinstance(text, list):
            writer.writerows(text)
    # print(f"********** 结果已保存到 {file_path} **********")


def del_error_samples(file_path, error_indices):
    # 读取并筛选所有非错误索引的行
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        # 读取保存的3个csv文件，然后当Index列中有error_indices,才删除该数据
        new_rows = [row for row in rows if not row or int(row[0]) not in error_indices]
    # 写回文件
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_rows)
    print(f"已完成 {file_path} 的错误数据删除，删除的索引：{error_indices}")


# 统计token运行时间
def evaluate_runtime(output_tokens_num, inference_time, token_time_path, total_runtime_index, all_samples_runtime):
    # 单次TPS (token per second)，已修改只统计输出
    # in_out_tokens = input_tokens + output_tokens
    out_tps = output_tokens_num / inference_time
    # 每个token需要多少秒 #
    out_spt = inference_time / output_tokens_num

    # 将结果写入csv文件
    # 单个样本值
    sample_stats = {
        "total output tokens": output_tokens_num,
        "inference time seconds": inference_time,
        "inference tokens per second": out_tps,
        "inference second per token": out_spt
    }
    csv_headers = [
        "total output tokens",
        "inference time seconds",
        "inference tokens per second",
        "inference second per token"
    ]
    all_samples_runtime.append(sample_stats)
    save_csv_result(token_time_path, csv_headers, all_samples_runtime, 'w')

    # 样本平均值
    total_runtime_index["total_tokens"] += output_tokens_num
    total_runtime_index["total_inference_time"] += inference_time

    num_samples = len(all_samples_runtime)
    avg_tokens = int(total_runtime_index["total_tokens"] / num_samples)
    avg_inference_time = total_runtime_index["total_inference_time"] / num_samples
    avg_tps = int(avg_tokens / avg_inference_time)
    avg_spt = avg_inference_time / avg_tokens

    avg_row = {
        "total output tokens": avg_tokens,
        "inference time seconds": avg_inference_time,
        "inference tokens per second": avg_tps,
        "inference second per token": avg_spt
    }
    save_csv_result(token_time_path, csv_headers, avg_row, 'a')
    print("-----------------------------------\n")


def evaluate_runtime1(sample_idx, output_tokens_num, inference_time, token_time_path, all_samples_runtime):
    # 单次TPS (token per second)，已修改只统计输出
    # in_out_tokens = input_tokens + output_tokens
    out_tps = output_tokens_num / inference_time
    # 每个token需要多少秒 #
    out_spt = inference_time / output_tokens_num

    # 将结果写入csv文件
    # 单个样本值
    sample_stats = {
        "Index": sample_idx,
        "total output tokens": output_tokens_num,
        "inference time seconds": inference_time,
        "inference tokens per second": out_tps,
        "inference second per token": out_spt
    }
    csv_headers = [
        "Index",
        "total output tokens",
        "inference time seconds",
        "inference tokens per second",
        "inference second per token"
    ]
    all_samples_runtime.append(sample_stats)
    save_csv_result(token_time_path, csv_headers, all_samples_runtime, 'w')
    print("-----------------------------------\n")


# 统计理论和推理的网络吞吐量
def evalulate_throughput(history_theory_texts, result_paths, thr_path):
    # SINR
    with open(result_paths[0], 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        sinr_rows = list(reader)
    # APS
    with open(result_paths[1], 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        aps_rows = list(reader)
    # RA
    with open(result_paths[2], 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        ra_rows = list(reader)

    all_samples_throughput = []
    for idx, prompt in enumerate(history_theory_texts):
        # print('****** For index %s, current prompt is %s ******'%(idx, prompt))
        AP_num, UE_num, R = extract_ue_info1(prompt)
        # print('Extracted AP and UE numbers are %s and %s \n'%(AP_num, UE_num))
        target_sinr = ast.literal_eval(sinr_rows[idx][2])
        target_X_iu = ast.literal_eval(aps_rows[idx][2])
        target_rho = ast.literal_eval(ra_rows[idx][2])
        pred_sinr = ast.literal_eval(sinr_rows[idx][3])
        pred_X_iu = ast.literal_eval(aps_rows[idx][3])
        pred_rho = ast.literal_eval(ra_rows[idx][3])
        # 链式推理和单独推理都用APS和SINR推理值计算最终预测的throughput
        # 修复bug: 对于APS预测超出AP数导致的索引问题(21), 跳过此样本
        try:
            nor_target_Rho = normalize_rho(target_X_iu, target_rho, AP_num, UE_num)
            target_thr = throughput_cal(R, target_X_iu, target_sinr, nor_target_Rho)
            nor_pred_Rho = normalize_rho(pred_X_iu, pred_rho, AP_num, UE_num)
            pred_thr = throughput_cal(R, pred_X_iu, pred_sinr, nor_pred_Rho)
            performance_gap = (target_thr - pred_thr)/target_thr
            print(f"********** Sample {idx} throughput gap: {performance_gap} **********")
            # 将结果写入csv
            sample_header = [
                "Index",
                "UE_num",
                "Target_Thr",
                "Pred_Thr",
                "Performance_Gap"
            ]
            sample_stats = {
                "Index": idx,
                "UE_num": UE_num,
                "Target_Thr": target_thr,
                "Pred_Thr": pred_thr,
                "Performance_Gap": performance_gap
            }
            all_samples_throughput.append(sample_stats)
            save_csv_result(thr_path, sample_header, all_samples_throughput, 'w')
        except IndexError as error_info:
            print(f"❌ 样本{idx}在最终计算吞吐量有数据索引错误, 错误信息如下: {error_info}")
            continue


def calculate_average(file_metrics, file_paths, output_path):
    results = {}
    # 读取并计算每个文件的指标
    for file_name, metrics in file_metrics.items():
        df = pd.read_csv(file_paths[file_name])
        if file_name == 'task2': # calculate average acc
            UE_num_list = df[metrics[0]]
            Acc_list = df[metrics[1]]
            weighted_avg = np.sum(UE_num_list * Acc_list) / np.sum(UE_num_list)
            results[f'{file_name}_average_accuracy'] = weighted_avg
        else:
            for metric in metrics:
                if metric in df.columns:
                    avg_value = df[metric].mean()
                    results[f'{file_name}_{metric}'] = avg_value

    # 创建结果DataFrame
    result_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Average_Value'])
    # 保存结果
    result_df.to_csv(output_path, index=False)
    return result_df



