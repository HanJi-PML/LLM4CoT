import sys
# 请确保路径正确，如果有变化请修改
sys.path.append('/data/LLM_indoor/ATCNN-main')

from ATCNN_model import ATCNN, ATCNN_9LiFi, switch  # type: ignore
import os
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import math
import warnings
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from types import SimpleNamespace

warnings.filterwarnings("ignore")

# 指定GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_ID = 0

############################################
# 训练参数配置 
############################################
args = SimpleNamespace()
args.epochs = 100           # 训练轮数
args.lr = 0.001             # 学习率 
args.momentum = 0.95        # 动量
args.weight_decay = 1e-4    # 权重衰减
args.batch_size = 64        # 训练批次大小
args.test_batch_size = 256  # 测试批次大小


# 数据集路径配置
DATASET_BASE_PATH = '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate'

# 保存路径
SAVE_FOLDER = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task2_ATCNN'


SNR_max = 70
SNR_min = -10
R_max = 1000
R_min = 1

# device
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)
    device = torch.device(f'cuda:{GPU_ID}')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


def auto_detect_dataset_config(json_path):
    """自动检测数据集的AP数量和最大UE数量"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ap_nums = set()
    ue_nums = set()
    
    for sample in data:
        instr = sample['instruction']
        match = re.search(r'There are (\d+) APs serving (\d+) users', instr)
        if match:
            ap_nums.add(int(match.group(1)))
            ue_nums.add(int(match.group(2)))
    
    if len(ap_nums) != 1:
        raise ValueError(f"数据集中AP数量不一致: {ap_nums}")
    
    ap_size = list(ap_nums)[0]
    max_ue = max(ue_nums)
    
    return ap_size, max_ue, ue_nums


class JSONDataset(Dataset):
    """支持动态AP和UE数量的数据集类"""
    
    def __init__(self, json_path, AP_size, max_ue, SNR_max=70, SNR_min=-10, R_max=500, R_min=10):
        self.AP_size = AP_size
        self.max_ue = max_ue
        self.SNR_max = SNR_max
        self.SNR_min = SNR_min
        self.R_max = R_max
        self.R_min = R_min
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f'Loaded {len(self.data)} samples from {json_path}')
    
    def __len__(self):
        return len(self.data)
    
    def parse_sinr_matrix(self, input_str):
        start_idx = input_str.find('[[')
        end_idx = input_str.find(']]', start_idx) + 2
        
        if start_idx == -1 or end_idx == 1:
            raise ValueError(f"Cannot find SINR matrix")
        
        matrix_str = input_str[start_idx:end_idx]
        matrix_str = matrix_str.replace('[', '').replace(']', '').strip()
        
        numbers = re.findall(r'-?[\d.]+', matrix_str)
        numbers = [float(n) for n in numbers]
        num_users = len(numbers) // self.AP_size
        sinr_matrix = np.array(numbers).reshape(num_users, self.AP_size)
        
        return sinr_matrix
    
    def parse_rate_requirement(self, input_str):
        start_idx = input_str.find('data rate requirement vector')
        if start_idx == -1:
            raise ValueError(f"Cannot find rate requirement")
        
        start_bracket = input_str.find('[', start_idx)
        end_bracket = input_str.find(']', start_bracket)
        
        vector_str = input_str[start_bracket+1:end_bracket]
        rate_vector = np.array([float(x) for x in vector_str.split() if x])
        
        return rate_vector
    
    def parse_output(self, output_str):
        start_idx = output_str.find('[')
        end_idx = output_str.find(']')
        
        vector_str = output_str[start_idx+1:end_idx]
        aps_result = np.array([int(x) for x in vector_str.split() if x])
        
        return aps_result
    
    def mirror(self, input_data, output_data, num_users):
        M = self.max_ue
        AP_num = self.AP_size
        UE_num = num_users
        user_dim = AP_num + 1
        
        if UE_num >= M:
            return np.array(input_data[:M * user_dim], dtype=np.float32), \
                   np.array(output_data[:M * AP_num], dtype=np.float32)
        
        quotient = math.floor(M / UE_num)
        reminder = M % UE_num
        
        new_input = np.array(input_data, dtype=np.float32)
        
        if reminder == 0:
            for i in range(UE_num):
                rate_idx = i * user_dim + AP_num
                new_input[rate_idx] = new_input[rate_idx] / quotient
        else:
            for i in range(reminder):
                rate_idx = i * user_dim + AP_num
                new_input[rate_idx] = new_input[rate_idx] / (quotient + 1)
            for i in range(UE_num - reminder):
                rate_idx = (i + reminder) * user_dim + AP_num
                new_input[rate_idx] = new_input[rate_idx] / quotient
        
        new_input = list(new_input)
        
        mirroring_input = []
        if quotient == 1:
            mirroring_input.append(new_input)
            mirroring_input.append(new_input[0:user_dim * reminder])
        else:
            for i in range(quotient):
                mirroring_input.append(new_input)
            mirroring_input.append(new_input[0:user_dim * reminder])
        
        mirroring_input = sum(mirroring_input, [])
        
        new_output = list(output_data)
        mirroring_output = []
        if quotient == 1:
            mirroring_output.append(new_output)
            mirroring_output.append(new_output[0:AP_num * reminder])
        else:
            for i in range(quotient):
                mirroring_output.append(new_output)
            mirroring_output.append(new_output[0:AP_num * reminder])
        
        mirroring_output = sum(mirroring_output, [])
        
        return np.array(mirroring_input, dtype=np.float32), np.array(mirroring_output, dtype=np.float32)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sinr_matrix = self.parse_sinr_matrix(sample['input'])
        rate_vector = self.parse_rate_requirement(sample['input'])
        aps_result = self.parse_output(sample['output'])
        
        # 获取真实用户数
        num_users = len(rate_vector)
        
        input_data = []
        for i in range(num_users):
            sinr_norm = (sinr_matrix[i] - self.SNR_min) / (self.SNR_max - self.SNR_min)
            sinr_norm = np.clip(sinr_norm, 0, 1)
            
            rate_norm = (rate_vector[i] - self.R_min) / (self.R_max - self.R_min)
            rate_norm = np.clip(rate_norm, 0, 1)
            
            user_data = list(sinr_norm) + [rate_norm]
            input_data.extend(user_data)
        
        output_data = []
        for i in range(num_users):
            one_hot = np.zeros(self.AP_size, dtype=np.float32)
            ap_idx = aps_result[i] - 1
            if 0 <= ap_idx < self.AP_size:
                one_hot[ap_idx] = 1.0
            output_data.extend(one_hot)
        
        input_data, output_data = self.mirror(input_data, output_data, num_users)
        
        # [修改点 1] 增加返回 num_users (真实用户数)
        return input_data, output_data, num_users


def train_single_dataset(json_path, args, save_folder, device):
    """训练单个数据集"""
    
    room_match = re.search(r'(room\d+-\d+)', json_path)
    room_name = room_match.group(1) if room_match else 'unknown'
    
    AP_size, max_ue, ue_nums = auto_detect_dataset_config(json_path)
    
    print('\n' + '='*70)
    print(f'开始训练: {room_name}')
    print(f'  AP数量: {AP_size}, UE范围: {sorted(ue_nums)}, 最大UE: {max_ue}')
    print('='*70)
    
    exp_folder = os.path.join(save_folder, f'{room_name}_{AP_size}AP_{max_ue}UE' + datetime.now().strftime("-%Y%m%d-%I%M%p"))
    os.makedirs(exp_folder, exist_ok=True)
    
    user_dim = AP_size + 1
    output_dim = AP_size
    cond_dim = (AP_size + 1) * max_ue
    
    # 这里的参数使用了全局变量修正后的值
    full_dataset = JSONDataset(json_path, AP_size, max_ue, SNR_max, SNR_min, R_max, R_min)
    
    train_size = int(len(full_dataset) * 0.8)
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, test_subset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    
    print(f'训练集: {train_size}, 测试集: {test_size}')
    
    model = ATCNN(input_dim=user_dim, cond_dim=cond_dim, cond_out_dim=user_dim, output_dim=output_dim)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    with open(f'{exp_folder}/config.txt', 'w') as f:
        f.write(f'Room: {room_name}\n')
        f.write(f'AP_size: {AP_size}, max_UE: {max_ue}\n')
        f.write(f'SNR Range: [{SNR_min}, {SNR_max}], Rate Range: [{R_min}, {R_max}]\n')
        f.write(f'lr: {args.lr}, batch_size: {args.batch_size}, epochs: {args.epochs}\n')
        f.write(f'Model:\n{str(model)}\n')
    
    epochs_list, train_losses, val_losses = [], [], []
    start_time = datetime.now()
    
    # ==================== 训练循环 ====================
    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss, epoch_train_count = 0, 0
        
        # [修改点 2] 接收第三个参数 _ (num_users 在训练时用不到，忽略)
        for raw_dataset, raw_label, _ in train_loader:
            raw_dataset = raw_dataset.to(torch.float32).to(device)
            raw_label = raw_label.to(torch.float32).to(device)
            
            optimizer.zero_grad()
            
            total_batch_loss = torch.tensor(0.0, device=device)
            valid_steps = 0
            
            # 遍历所有可能的用户位置
            for ue_idx in range(max_ue):
                # 提取 Target
                Target = raw_dataset[..., ue_idx*user_dim:(ue_idx+1)*user_dim]
                label_sub = raw_label[..., ue_idx*output_dim:(ue_idx+1)*output_dim]
                
                # 关键：Switch时必须使用 .clone()
                condition_list = switch(raw_dataset.clone(), ue_idx, AP_size+1)
                condition_now = torch.tensor(condition_list).to(device)
                Condition = condition_now[..., 0:]
                
                opt = model(Target, Condition)
                loss = criterion(opt, label_sub)
                
                total_batch_loss += loss
                valid_steps += 1
            
            if valid_steps > 0:
                avg_batch_loss = total_batch_loss / valid_steps
                avg_batch_loss.backward()
                optimizer.step()
                
                epoch_train_loss += avg_batch_loss.item()
                epoch_train_count += 1
        
        # ==================== 验证循环 ====================
        model.eval()
        epoch_val_loss, epoch_val_count = 0, 0
        
        with torch.no_grad():
            # [修改点 2] 接收第三个参数 _
            for raw_dataset, raw_label, _ in test_loader:
                raw_dataset = raw_dataset.to(torch.float32).to(device)
                raw_label = raw_label.to(torch.float32).to(device)
                
                total_batch_loss = 0
                valid_steps = 0
                
                for ue_idx in range(max_ue):
                    Target = raw_dataset[..., ue_idx*user_dim:(ue_idx+1)*user_dim]
                    label_sub = raw_label[..., ue_idx*output_dim:(ue_idx+1)*output_dim]
                    
                    condition_list = switch(raw_dataset.clone(), ue_idx, AP_size+1)
                    condition_now = torch.tensor(condition_list).to(device)
                    Condition = condition_now[..., 0:]
                    
                    opt = model(Target, Condition)
                    loss = criterion(opt, label_sub)
                    
                    # 累加 Loss 数值
                    total_batch_loss += loss.item()
                    valid_steps += 1
                
                # 计算当前 Batch 所有用户的平均 Loss
                if valid_steps > 0:
                    avg_batch_loss = total_batch_loss / valid_steps
                    epoch_val_loss += avg_batch_loss
                    epoch_val_count += 1
        
        avg_train_loss = epoch_train_loss / epoch_train_count if epoch_train_count > 0 else 0
        avg_val_loss = epoch_val_loss / epoch_val_count if epoch_val_count > 0 else 0
        
        epochs_list.append(epoch + 1)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}')
        
        scheduler.step()
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f'训练完成, 耗时: {duration:.1f}s')
    
    torch.save(model.state_dict(), f'{exp_folder}/model.pth')
    
    loss_df = pd.DataFrame({'epoch': epochs_list, 'train_loss': train_losses, 'val_loss': val_losses})
    loss_df.to_csv(f'{exp_folder}/loss_history.csv', index=False)
    
    # 绘制曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, label='Train')
    plt.plot(epochs_list, val_losses, label='Val')
    plt.title(f'{room_name} - Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    start_epoch = min(10, len(train_losses))
    plt.plot(epochs_list[start_epoch-1:], train_losses[start_epoch-1:], label='Train')
    plt.plot(epochs_list[start_epoch-1:], val_losses[start_epoch-1:], label='Val')
    plt.title(f'Loss (From Epoch {start_epoch})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{exp_folder}/loss_curve.png', dpi=300)
    plt.close()
    
    # ==================== 测试阶段 (分类统计重构) ====================
    print('\n测试模型 (分类统计模式)...')
    model.load_state_dict(torch.load(f'{exp_folder}/model.pth', map_location=device))
    model.eval()
    
    # 初始化分类统计桶
    # 格式: {5: {'correct': 0, 'total': 0}, 10: ...}
    metrics_by_load = {ue: {'correct': 0, 'total': 0} for ue in sorted(ue_nums)}
    
    with torch.no_grad():
        # [修改点 3] 接收 real_ue_counts，并进行精准分类测试
        for raw_dataset, raw_label, real_ue_counts in test_loader:
            raw_dataset = raw_dataset.to(torch.float32).to(device)
            raw_label = raw_label.to(torch.float32).to(device)
            
            # 必须逐样本处理，因为一个Batch内可能包含不同用户数的样本（虽然Mirror后大小一样，但统计需不同）
            batch_size_curr = raw_dataset.shape[0]
            
            for i in range(batch_size_curr):
                # 获取该样本原本的真实用户数
                current_real_ue = int(real_ue_counts[i].item())
                
                # 如果这个样本的用户数不在我们的统计范围内，跳过
                if current_real_ue not in metrics_by_load:
                    continue
                
                # 提取单样本数据
                sample_data = raw_dataset[i]
                sample_label = raw_label[i]
                
                # [核心逻辑] 仅测试真实存在的用户位置
                # 例如：如果是5UE样本，只循环0-4；如果是20UE样本，循环0-19
                for ue_idx in range(current_real_ue):
                    
                    # 提取 Target: [1, user_dim]
                    Target = sample_data[ue_idx*user_dim : (ue_idx+1)*user_dim].unsqueeze(0)
                    
                    # 提取 Condition: 需要配合 switch
                    # 注意：switch 需要输入 [1, ...] 维度的 tensor
                    condition_list = switch(sample_data.unsqueeze(0).clone(), ue_idx, AP_size+1)
                    condition_now = torch.tensor(condition_list).to(device)
                    Condition = condition_now[..., 0:]
                    
                    # 提取 Label
                    label_sub = sample_label[ue_idx*output_dim : (ue_idx+1)*output_dim].unsqueeze(0)
                    
                    # 推理
                    opt = model(Target, Condition)
                    pred_class = torch.argmax(opt, dim=1)
                    target_class = torch.argmax(label_sub, dim=1)
                    
                    # 统计
                    if pred_class.item() == target_class.item():
                        metrics_by_load[current_real_ue]['correct'] += 1
                    metrics_by_load[current_real_ue]['total'] += 1
    
    # 计算并打印结果
    ue_accuracy_results = {}
    print(f'分类准确率结果:')
    for ue_count in sorted(metrics_by_load.keys()):
        stats = metrics_by_load[ue_count]
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total']) * 100
            ue_accuracy_results[ue_count] = acc
            print(f'  真实负载 {ue_count}UE: {acc:.2f}% (测试样本点: {stats["total"]})')
        else:
            print(f'  真实负载 {ue_count}UE: 无数据')
            ue_accuracy_results[ue_count] = 0
            
    avg_accuracy = np.mean(list(ue_accuracy_results.values())) if ue_accuracy_results else 0
    print(f'平均准确率: {avg_accuracy:.2f}%')
    
    # 保存结果
    test_results = {
        'room': room_name,
        'AP_size': AP_size,
        'max_UE_size': max_ue,
        'train_loss_final': train_losses[-1],
        'val_loss_final': val_losses[-1],
        'epochs': args.epochs,
    }
    
    for ue_count in sorted(ue_accuracy_results.keys()):
        test_results[f'accuracy_{ue_count}UE'] = ue_accuracy_results[ue_count]
    test_results['accuracy_avg'] = avg_accuracy
    
    test_results_df = pd.DataFrame([test_results])
    test_results_df.to_csv(f'{exp_folder}/test_results.csv', index=False)
    
    return test_results


def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    all_results = []
    
    dataset_files = [
         'dataset3_room1-1_task2.json',
         'dataset3_room2-1_task2.json',
         'dataset3_room3-1_task2.json',
         'dataset3_room4-1_task2.json',
         'dataset3_room5-1_task2.json',
         'dataset3_room6-1_task2.json',
     ]
    #dataset_files = ['dataset3_room1-1_task2.json']
    
    for dataset_file in dataset_files:
        json_path = os.path.join(DATASET_BASE_PATH, dataset_file)
        
        if not os.path.exists(json_path):
            print(f'警告: 文件不存在 {json_path}, 跳过')
            continue
        
        result = train_single_dataset(json_path, args, SAVE_FOLDER, device)
        all_results.append(result)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        base_cols = ['room', 'AP_size', 'max_UE_size', 'train_loss_final', 'val_loss_final', 'epochs']
        accuracy_cols = [col for col in summary_df.columns if col.startswith('accuracy_') and col != 'accuracy_avg']
        
        def extract_ue_num(col_name):
            match = re.search(r'accuracy_(\d+)UE', col_name)
            return int(match.group(1)) if match else 0
        
        accuracy_cols_sorted = sorted(accuracy_cols, key=extract_ue_num)
        final_columns = base_cols + accuracy_cols_sorted + ['accuracy_avg']
        summary_df = summary_df[final_columns]
        
        timestamp = datetime.now().strftime("%Y%m%d-%I%M%p")
        summary_path = f'{SAVE_FOLDER}/task2_ATCNN_test_results_{timestamp}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print('\n' + '='*70)
        print('所有训练完成! 汇总结果:')
        print('='*70)
        print(summary_df.to_string(index=False))
        print(f'\n汇总文件已保存到: {summary_path}')


if __name__ == "__main__":
    main()