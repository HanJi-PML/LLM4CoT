"""
MLP Performance Gap Analysis Script (仅DNN版本 - 完美Task2+Task3输入)
====================================================================
本脚本用于测试DNN模型单独的推理性能（AP选择和功率分配使用完美数据）：
- Task 1: DNN 进行 SINR 预测
- Task 2: 使用真实AP选择结果（完美，不使用ATCNN）
- Task 3: 使用真实功率分配结果（完美，不使用GNN）

最终计算预测结果的吞吐量，并与原始数据的吞吐量进行对比。
支持一次性测试6个数据集。
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import re
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import random

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_ID = 0

# ==================== 设备配置 ====================
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)
    device = torch.device(f'cuda:{GPU_ID}')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


# ==================== 全局配置参数 ====================
# 数据集基础路径
DATASET_BASE_PATH = '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset'

# 模型基础路径
DNN_MODEL_BASE = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN'
ATCNN_MODEL_BASE = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task2_ATCNN'
GNN_MODEL_BASE = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task3_GNN'

# 保存路径
SAVE_FOLDER = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/MLP_performance_gap'

# 数据归一化参数
SNR_MAX = 70
SNR_MIN = -10
R_MAX = 1000
R_MIN = 1

# 测试样本数
MAX_SAMPLES = 100

# 6个数据集配置
DATASET_CONFIGS = [
    {
        'name': 'room1-1',
        'dataset': 'dataset3_room1-1_combined_sharegpt.json',
        'dnn_model': 'room1-1_5AP-20251231-1201AM/model.pth',
        'atcnn_model': 'room1-1_5AP_20UE-20260122-1123AM/model.pth',
        'gnn_model': 'room1-1_5AP_20UE_GAT_VectorOutput_-20260125-0449PM/best_model.pth',
    },
    {
        'name': 'room2-1',
        'dataset': 'dataset3_room2-1_combined_sharegpt.json',
        'dnn_model': 'room2-1_5AP-20251231-1201AM/model.pth',  
        'atcnn_model': 'room2-1_5AP_20UE-20260122-1055AM/model.pth',  
        'gnn_model': 'room2-1_5AP_20UE_GAT_VectorOutput_-20260125-0451PM/best_model.pth',  
    },
    {
        'name': 'room3-1',
        'dataset': 'dataset3_room3-1_combined_sharegpt.json',
        'dnn_model': 'room3-1_6AP-20251231-1202AM/model.pth',  # TODO: 需要填写实际路径
        'atcnn_model': 'room3-1_6AP_30UE-20260122-1058AM/model.pth',  # TODO: 需要填写实际路径
        'gnn_model': 'room3-1_6AP_30UE_GAT_VectorOutput_-20260125-0452PM/best_model.pth',  # TODO: 需要填写实际路径
    },
    {
        'name': 'room4-1',
        'dataset': 'dataset3_room4-1_combined_sharegpt.json',
        'dnn_model': 'room4-1_7AP-20251231-1202AM//model.pth',  # TODO: 需要填写实际路径
        'atcnn_model': 'room4-1_7AP_30UE-20260122-1100AM/model.pth',  # TODO: 需要填写实际路径
        'gnn_model': 'room4-1_7AP_30UE_GAT_VectorOutput_-20260125-0453PM/best_model.pth',  # TODO: 需要填写实际路径
    },
    {
        'name': 'room5-1',
        'dataset': 'dataset3_room5-1_combined_sharegpt.json',
        'dnn_model': 'room5-1_8AP-20251231-1202AM/model.pth',  # TODO: 需要填写实际路径
        'atcnn_model': 'room5-1_8AP_40UE-20260122-1102AM/model.pth',  # TODO: 需要填写实际路径
        'gnn_model': 'room5-1_8AP_40UE_GAT_VectorOutput_-20260125-0454PM/best_model.pth',  # TODO: 需要填写实际路径
    },
    {
        'name': 'room6-1',
        'dataset': 'dataset3_room6-1_combined_sharegpt.json',
        'dnn_model': 'room6-1_9AP-20251231-1202AM/model.pth',  # TODO: 需要填写实际路径
        'atcnn_model': 'room6-1_9AP_40UE-20260122-1106AM/model.pth',  # TODO: 需要填写实际路径
        'gnn_model': 'room6-1_9AP_40UE_GAT_VectorOutput_-20260125-0455PM/best_model.pth',  # TODO: 需要填写实际路径
    },
]


# ==================== 1. DNN 模型定义 ====================
class DNNModel(nn.Module):
    def __init__(self, output_dim=5):
        super(DNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==================== 2. 自动检测数据集配置 ====================
def auto_detect_dataset_config(json_path):
    """自动检测数据集的AP数量和最大UE数量"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ap_nums = set()
    ue_nums = set()
    
    for sample in data:
        # combined_sharegpt格式使用conversations
        if 'conversations' in sample:
            text = sample['conversations'][0]['value']
        else:
            text = sample.get('instruction', '')
        
        match = re.search(r'There are (\d+) APs serving (\d+) users', text)
        if match:
            ap_nums.add(int(match.group(1)))
            ue_nums.add(int(match.group(2)))
    
    if len(ap_nums) != 1:
        raise ValueError(f"数据集中AP数量不一致: {ap_nums}")
    
    ap_size = list(ap_nums)[0]
    max_ue = max(ue_nums)
    
    return ap_size, max_ue, ue_nums


# ==================== 3. 数据解析函数 ====================
def parse_positions(text):
    """从对话中提取用户位置"""
    pos_match = re.search(r"positions are (\[\[.*?\]\])", text)
    if not pos_match:
        return None
    pos_str = pos_match.group(1)
    positions = np.array(eval(pos_str))
    return positions


def parse_sinr_matrix(text, AP_num):
    """从对话中提取SINR矩阵"""
    sinr_match = re.search(r"SINR matrix is (\[\[.*?\]\])", text)
    if not sinr_match:
        return None
    sinr_str = sinr_match.group(1)
    sinr_rows = []
    rows_str = sinr_str.strip('[]').split('] [')
    for row_str in rows_str:
        row_str = row_str.strip('[]')
        numbers = [float(x) for x in row_str.split() if x]
        sinr_rows.append(numbers)
    sinr_matrix = np.array(sinr_rows)
    return sinr_matrix


def parse_rate_requirement(text):
    """从对话中提取速率需求向量"""
    rate_match = re.search(r"data rate requirement vector.*?is \[(.*?)\]", text)
    if not rate_match:
        return None
    rate_str = rate_match.group(1)
    rate_vector = np.array([float(x) for x in rate_str.split() if x])
    return rate_vector


def parse_aps_result(text):
    """从对话中提取AP选择结果"""
    aps_match = re.search(r"APS result is \[(.*?)\]", text)
    if not aps_match:
        return None
    aps_str = aps_match.group(1)
    aps_result = np.array([int(x) for x in aps_str.split() if x])
    return aps_result


def parse_ra_result(text):
    """从对话中提取资源分配矩阵"""
    ra_match = re.search(r"RA result is (\[\[.*?\]\])", text)
    if not ra_match:
        return None
    ra_str = ra_match.group(1)
    ra_rows = []
    rows_str = ra_str.strip('[]').split('] [')
    for row_str in rows_str:
        row_str = row_str.strip('[]')
        numbers = [float(x) for x in row_str.split() if x]
        ra_rows.append(numbers)
    ra_matrix = np.array(ra_rows)
    return ra_matrix


def extract_sample_data(sample, ap_num):
    """从单个样本中提取所有必要数据"""
    conversations = sample['conversations']
    
    # 拼接所有对话文本
    all_text = ""
    for conv in conversations:
        all_text += conv['value'] + " "
    
    # 提取各项数据
    positions = parse_positions(all_text)
    sinr_matrix = parse_sinr_matrix(all_text, ap_num)
    rate_vector = parse_rate_requirement(all_text)
    aps_result = parse_aps_result(all_text)
    ra_result = parse_ra_result(all_text)
    
    return {
        'positions': positions,
        'sinr': sinr_matrix,
        'rate': rate_vector,
        'aps': aps_result,
        'ra': ra_result
    }


# ==================== 4. 吞吐量计算函数 ====================
def normalize_rho(X_iu, Rho, AP_num, UE_num):
    """归一化资源分配向量"""
    Rho_list = []
    for ue_idx, ap_selected in enumerate(X_iu):
        ap_idx = ap_selected - 1
        if 0 <= ap_idx < AP_num:
            rho_value = Rho[ue_idx, ap_idx] if Rho.ndim == 2 else Rho[ue_idx]
        else:
            rho_value = 0
        Rho_list.append(rho_value)
    
    ap_rhos = {i: [] for i in range(AP_num)}
    for ue_idx, ap_selected in enumerate(X_iu):
        ap_idx = ap_selected - 1
        if 0 <= ap_idx < AP_num:
            ap_rhos[ap_idx].append((ue_idx, Rho_list[ue_idx]))
    
    normalized_rho = [0.0] * UE_num
    for ap_idx, ue_list in ap_rhos.items():
        if ue_list:
            total = sum([r for _, r in ue_list])
            if total > 0:
                for ue_idx, r in ue_list:
                    normalized_rho[ue_idx] = r / total
            else:
                for ue_idx, _ in ue_list:
                    normalized_rho[ue_idx] = 1.0 / len(ue_list)
    
    return normalized_rho


def throughput_cal(R, X_iu, SINR, Rho):
    """计算吞吐量 (TCP模式)"""
    LiFi_BW = 40
    WiFi_BW = 20
    UE_num = len(R)
    thr_list = []
    
    SINR_linear = [[10 ** (x/10) for x in user_sinr] for user_sinr in SINR]
    
    for ue_idx, X_iu_now in enumerate(X_iu):
        ap_idx = X_iu_now - 1
        
        if ap_idx < 0 or ap_idx >= len(SINR_linear[ue_idx]):
            thr_list.append(0)
            continue
            
        sinr = SINR_linear[ue_idx][ap_idx]
        rho = Rho[ue_idx] if ue_idx < len(Rho) else 0
        
        if X_iu_now == 1:
            capacity = WiFi_BW * math.log2(1 + sinr) * rho
        else:
            factor = math.exp(1) / (2 * math.pi)
            capacity = (LiFi_BW / 2) * math.log2(1 + factor * sinr) * rho
        
        user_throughput = min(capacity, R[ue_idx])
        thr_list.append(user_throughput)
    
    total_throughput = sum(thr_list)
    return total_throughput


# ==================== 5. 单个数据集测试函数 ====================
def test_single_dataset(config, device):
    """测试单个数据集（仅DNN预测SINR，Task2和Task3使用完美数据）"""
    room_name = config['name']
    dataset_path = os.path.join(DATASET_BASE_PATH, config['dataset'])
    dnn_model_path = os.path.join(DNN_MODEL_BASE, config['dnn_model'])
    
    print("\n" + "=" * 70)
    print(f"测试数据集: {room_name}")
    print("使用DNN预测SINR（Task1）+ 完美AP选择（Task2）+ 完美功率分配（Task3）")
    print("=" * 70)
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        print(f"  ❌ 数据集文件不存在: {dataset_path}")
        return None
    if not os.path.exists(dnn_model_path):
        print(f"  ❌ DNN模型文件不存在: {dnn_model_path}")
        return None
    
    # 自动检测数据集配置
    ap_num, max_ue, ue_nums = auto_detect_dataset_config(dataset_path)
    print(f"  数据集配置: AP={ap_num}, 最大UE={max_ue}, UE范围={sorted(ue_nums)}")
    
    # 加载数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    random.seed(42)
    random.shuffle(data)
    samples = data[:MAX_SAMPLES]
    print(f"  测试样本数: {len(samples)}")
    
    # 只加载DNN模型
    print("  加载DNN模型...")
    
    # DNN
    dnn_model = DNNModel(output_dim=ap_num)
    dnn_model.load_state_dict(torch.load(dnn_model_path, map_location=device))
    dnn_model.to(device)
    dnn_model.eval()
    
    # 推理
    print("  开始推理...")
    
    all_dnn_mae = []
    all_gt_thr = []
    all_pred_thr = []
    all_gap = []
    
    valid_count = 0
    error_count = 0
    
    # ========== 关键：从数据集中提取真实的 pos_min/pos_max ==========
    # 与训练代码保持一致：使用数据集中所有位置的实际 min/max
    print("  计算位置归一化参数...")
    all_positions = []
    for sample in data:  # 使用完整数据集计算，与训练一致
        sample_data = extract_sample_data(sample, ap_num)
        if sample_data['positions'] is not None:
            all_positions.extend(sample_data['positions'].tolist())
    
    all_positions = np.array(all_positions)
    pos_min = all_positions.min(axis=0)
    pos_max = all_positions.max(axis=0)
    print(f"  位置范围: min={pos_min}, max={pos_max}")
    
    for idx, sample in enumerate(samples):
        try:
            sample_data = extract_sample_data(sample, ap_num)
            
            if any(v is None for v in sample_data.values()):
                error_count += 1
                continue
            
            positions = sample_data['positions']
            gt_sinr = sample_data['sinr']
            rate_vector = sample_data['rate']
            gt_aps = sample_data['aps']
            gt_ra = sample_data['ra']
            
            ue_num = len(rate_vector)
            
            # Task 1: DNN预测SINR
            pos_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)
            
            with torch.no_grad():
                pos_tensor = torch.tensor(pos_norm, dtype=torch.float32).to(device)
                pred_sinr = dnn_model(pos_tensor).cpu().numpy()
            
            dnn_mae = np.abs(pred_sinr - gt_sinr).mean()
            all_dnn_mae.append(dnn_mae)
            
            # Task 2: 使用完美的真实AP选择结果（不使用ATCNN）
            # Task 3: 使用完美的真实功率分配结果（不使用GNN）
            
            # 计算吞吐量
            # 真实吞吐量：使用真实SINR、真实AP选择、真实功率分配
            gt_rho = normalize_rho(gt_aps, gt_ra, ap_num, ue_num)
            gt_throughput = throughput_cal(rate_vector, gt_aps, gt_sinr, gt_rho)
            
            # 预测吞吐量：使用DNN预测的SINR、真实AP选择、真实功率分配
            # 这样可以测试DNN预测的SINR误差对最终吞吐量的影响
            pred_throughput = throughput_cal(rate_vector, gt_aps, pred_sinr, gt_rho)
            
            if gt_throughput > 0:
                performance_gap = (gt_throughput - pred_throughput) / gt_throughput
            else:
                performance_gap = 0
            
            all_gt_thr.append(gt_throughput)
            all_pred_thr.append(pred_throughput)
            all_gap.append(performance_gap)
            
            valid_count += 1
            
        except Exception as e:
            error_count += 1
            continue
    
    print(f"  推理完成! 有效: {valid_count}, 错误: {error_count}")
    
    if valid_count == 0:
        return None
    
    # 计算平均指标
    result = {
        'room': room_name,
        'AP_num': ap_num,
        'max_UE': max_ue,
        'samples': valid_count,
        'DNN_MAE': np.mean(all_dnn_mae),
        'Task2_APS': 'Perfect (Ground Truth)',
        'Task3_RA': 'Perfect (Ground Truth)',
        'avg_gt_throughput': np.mean(all_gt_thr),
        'avg_pred_throughput': np.mean(all_pred_thr),
        'avg_performance_gap': np.mean(all_gap)
    }
    
    print(f"\n  结果汇总:")
    print(f"    [Task1] DNN MAE:          {result['DNN_MAE']:.4f} dB")
    print(f"    [Task2] AP选择:           完美（真实值）")
    print(f"    [Task3] 功率分配:         完美（真实值）")
    print(f"    平均真实吞吐量:           {result['avg_gt_throughput']:.4f} Mbps")
    print(f"    平均预测吞吐量:           {result['avg_pred_throughput']:.4f} Mbps")
    print(f"    平均性能差距:             {result['avg_performance_gap']*100:.2f}%")
    
    return result


# ==================== 6. 主函数 ====================
def main():
    print("=" * 80)
    print("MLP Performance Gap Analysis - 6 Datasets")
    print("仅测试DNN（Task1），使用完美的AP选择（Task2）和功率分配（Task3）")
    print("=" * 80)
    
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    
    all_results = []
    
    for config in DATASET_CONFIGS:
        result = test_single_dataset(config, device)
        if result is not None:
            all_results.append(result)
    
    if all_results:
        # 汇总结果
        summary_df = pd.DataFrame(all_results)
        
        timestamp = datetime.now().strftime("%Y%m%d-%I%M%p")
        summary_path = f'{SAVE_FOLDER}/MLP_performance_gap_DNN_only_{timestamp}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print("\n" + "=" * 80)
        print("所有测试完成! 汇总结果:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print(f"\n汇总文件已保存到: {summary_path}")
        
        # 打印总体平均
        print("\n" + "-" * 60)
        print("总体平均:")
        print("-" * 60)
        print(f"  DNN MAE:            {summary_df['DNN_MAE'].mean():.4f} dB")
        print(f"  Task2 AP选择:       完美（真实值）")
        print(f"  Task3 功率分配:     完美（真实值）")
        print(f"  平均真实吞吐量:     {summary_df['avg_gt_throughput'].mean():.4f} Mbps")
        print(f"  平均预测吞吐量:     {summary_df['avg_pred_throughput'].mean():.4f} Mbps")
        print(f"  平均性能差距:       {summary_df['avg_performance_gap'].mean()*100:.2f}%")
    else:
        print("\n没有成功测试的数据集!")


if __name__ == "__main__":
    main()
