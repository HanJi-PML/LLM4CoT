import json
import re
import numpy as np
import pandas as pd
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings("ignore")

# ==================== 全局配置 ====================
# 指定GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_ID = 0

# 训练参数配置
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_DIM = 32      # 【优化点】: 增加到64，提升模型容量
HEADS = 4
DROPOUT = 0.5

# 数据归一化参数
SNR_MAX = 70
SNR_MIN = -10
R_MAX = 1000
R_MIN = 1

# 数据集路径配置
DATASET_BASE_PATH = '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate'

# 保存路径
SAVE_FOLDER = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task3_GNN'

# device
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)
    device = torch.device(f'cuda:{GPU_ID}')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

# ==================== 1. GAT模型定义 ====================
'''
class GAT_RA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.5):
        super(GAT_RA, self).__init__()
        
        # 第一层 GAT
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        
        # 第一层 BatchNorm (中间层可以使用BN加速收敛)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        
        # 第二层 GAT (输出层)
        # 注意: 输出维度是 output_dim (即 AP_num)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)
        
        # 【优化点】: 移除了输出层的 BatchNorm，因为它可能破坏回归值的分布

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x) # 【优化点】: 使用 ELU 替代 ReLU，防止中间层神经元死亡
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        
        # 【核心优化点】: 
        # 1. 移除了 ReLU (避免死区)
        # 2. 使用 Sigmoid。对于AP_num维向量，这会将每个元素都压缩到(0,1)。
        #    这非常适合预测功率系数，对于未连接的AP，模型会学着输出接近0的值。
        x = torch.sigmoid(x) 

        return x
'''

class GAT_RA(nn.Module):
    """
    修改后的 3层 GAT 模型结构
    Input -> [GAT+BN+ELU] -> [GAT+BN+ELU] -> [GAT+Sigmoid] -> Output
    """
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.5):
        super(GAT_RA, self).__init__()
        
        # ==================== 第 1 层 ====================
        # 输入层 -> 隐藏层
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        
        # ==================== 第 2 层 (新增) ====================
        # 隐藏层 -> 隐藏层
        # 输入维度必须等于上一层的输出维度: hidden_dim * heads
        # 输出维度保持不变: hidden_dim * heads (因为 concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        
        # ==================== 第 3 层 (输出层) ====================
        # 隐藏层 -> 输出层
        # 输入维度: hidden_dim * heads
        # 输出维度: output_dim (即 AP_num)
        # 注意: 最后一层通常 concat=False (也就是做平均或直接输出)，维度变为 output_dim
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # ----- Layer 1 -----
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x) # 推荐使用 ELU
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # ----- Layer 2 (新增) -----
        # 引入残差连接 (Residual Connection) 思想通常能缓解深层 GNN 的效果下降
        # 但只有当输入输出维度一致时才能直接相加。
        # 这里为了简单起见，先展示标准堆叠，不加 res连接。
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ----- Layer 3 (Output) -----
        x = self.conv3(x, edge_index)
        
        # 激活函数: 功率分配必须在 [0,1] 之间
        x = torch.sigmoid(x) 

        return x

# ==================== 2. 数据处理函数 ====================
def parse_sinr_matrix(input_str, AP_num):
    start_idx = input_str.find('[[')
    end_idx = input_str.find(']]', start_idx) + 2
    if start_idx == -1 or end_idx == 1:
        raise ValueError("Cannot find SINR matrix")
    matrix_str = input_str[start_idx:end_idx]
    matrix_str = matrix_str.replace('[', '').replace(']', '').strip()
    numbers = re.findall(r'-?[\d.]+', matrix_str)
    numbers = [float(n) for n in numbers]
    num_users = len(numbers) // AP_num
    sinr_matrix = np.array(numbers).reshape(num_users, AP_num)
    return sinr_matrix

def parse_rate_requirement(input_str):
    start_idx = input_str.find('data rate requirement vector')
    if start_idx == -1:
        raise ValueError("Cannot find rate requirement")
    start_bracket = input_str.find('[', start_idx)
    end_bracket = input_str.find(']', start_bracket)
    vector_str = input_str[start_bracket+1:end_bracket]
    rate_vector = np.array([float(x) for x in vector_str.split() if x])
    return rate_vector

def parse_aps_vector(input_str):
    start_idx = input_str.find('access point selection vector is')
    if start_idx == -1:
        raise ValueError("Cannot find APS vector")
    start_bracket = input_str.find('[', start_idx)
    end_bracket = input_str.find(']', start_bracket)
    vector_str = input_str[start_bracket+1:end_bracket]
    aps_vector = np.array([int(x) for x in vector_str.split() if x])
    return aps_vector

def parse_ra_matrix(output_str, AP_num, UE_num):
    ra_match = re.search(r"RA result is (\[\[.*?\]\])", output_str)
    if not ra_match:
        raise ValueError("Cannot find RA matrix")
    ra_str = ra_match.group(1)
    ra_rows = []
    rows_str = ra_str.strip('[]').replace('] [', '],[').split('],[')
    for row_str in rows_str:
        numbers = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', row_str)]
        ra_rows.append(numbers)
    ra_matrix = np.array(ra_rows)
    return ra_matrix

def auto_detect_dataset_config(json_path):
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


# ==================== 3. 图数据创建 ====================
def create_graph_data_list(json_path, SNR_max=SNR_MAX, SNR_min=SNR_MIN, R_max=R_MAX, R_min=R_MIN):
    AP_num, max_UE, ue_nums = auto_detect_dataset_config(json_path)
    room_match = re.search(r'(room\d+-\d+)', json_path)
    room_name = room_match.group(1) if room_match else 'unknown'

    print(f"数据集配置: AP数量={AP_num}, UE范围={sorted(ue_nums)}, 最大UE: {max_UE}")
    print(f"房间名称: {room_name}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    graph_data_list = []

    for idx, sample in enumerate(data):
        try:
            sinr_matrix = parse_sinr_matrix(sample['input'], AP_num)
            rate_vector = parse_rate_requirement(sample['input'])
            aps_vector = parse_aps_vector(sample['input'])
            UE_num = len(rate_vector)
            ra_matrix = parse_ra_matrix(sample['output'], AP_num, UE_num)

            # 归一化
            sinr_norm = (sinr_matrix - SNR_min) / (SNR_max - SNR_min)
            sinr_norm = np.clip(sinr_norm, 0, 1)
            rate_norm = (rate_vector - R_min) / (R_max - R_min)
            rate_norm = np.clip(rate_norm, 0, 1)

            # 节点特征构建
            node_dim = AP_num + 1
            ap_features = np.zeros((AP_num, node_dim), dtype=np.float32)
            ue_features = np.zeros((UE_num, node_dim), dtype=np.float32)
            for i in range(UE_num):
                ue_features[i, :AP_num] = sinr_norm[i]
                ue_features[i, AP_num] = rate_norm[i]

            node_features = np.concatenate([ap_features, ue_features], axis=0)

            # 边构建
            edge_list = []
            for ue_idx in range(UE_num):
                selected_ap = aps_vector[ue_idx] - 1
                ue_node = AP_num + ue_idx
                ap_node = selected_ap
                edge_list.append([ue_node, ap_node])
                edge_list.append([ap_node, ue_node])

            edge_index = np.array(edge_list, dtype=np.int64).T

            #
            # 标签维度: (AP_num + UE_num, AP_num)
            ap_labels = np.zeros((AP_num, AP_num), dtype=np.float32)
            
            # 这里的 ra_matrix 本身就是 (UE_num, AP_num)
            # 它包含了选中AP的功率值，以及其他位置的0
            node_labels = np.concatenate([ap_labels, ra_matrix], axis=0)

            graph_data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(node_labels, dtype=torch.float32),
                aps=torch.tensor(aps_vector, dtype=torch.long),
                num_aps=AP_num,
                num_ues=UE_num
            )
            graph_data_list.append(graph_data)

        except Exception as e:
            print(f"样本 {idx} 解析失败: {e}")
            continue

    print(f"成功创建 {len(graph_data_list)} 个图数据样本")
    return graph_data_list, AP_num, max_UE, room_name


# ==================== 4. 训练函数 ====================
def compute_loss_batch(pred, batch_data, AP_num, criterion):
    ptr = batch_data.ptr
    batch_size = batch_data.num_graphs
    total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
    valid_count = 0

    for i in range(batch_size):
        start_idx = ptr[i].item()
        end_idx = ptr[i + 1].item()
        ue_start = start_idx + AP_num
        ue_end = end_idx

        # 此时 pred_ue 和 label_ue 都是 (num_ues, AP_num) 维度
        pred_ue = pred[ue_start:ue_end]
        label_ue = batch_data.y[ue_start:ue_end]

        if pred_ue.numel() > 0 and label_ue.numel() > 0:
            if not (torch.isnan(pred_ue).any() or torch.isinf(pred_ue).any()):
                loss = criterion(pred_ue, label_ue)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss = total_loss + loss
                    valid_count += 1

    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)


def train_model(model, train_loader, val_loader, output_dir, AP_num, UE_num,
                epochs=300, lr=0.001, gpu_id=0):

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    train_losses, val_losses = [], []
    epochs_list = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epochs_list.append(epoch + 1)
        model.train()
        train_loss = 0
        train_count = 0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            pred = model(batch_data)
            loss = compute_loss_batch(pred, batch_data, AP_num, criterion)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_count += 1

        model.eval()
        val_loss = 0
        val_count = 0

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                pred = model(batch_data)
                loss = compute_loss_batch(pred, batch_data, AP_num, criterion)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_count += 1

        avg_train_loss = train_loss / max(train_count, 1)
        avg_val_loss = val_loss / max(val_count, 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if val_count > 0 and not np.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | '
                  f'Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

    torch.save(model.state_dict(), f'{output_dir}/final_model.pth')
    if not os.path.exists(f'{output_dir}/best_model.pth'):
        torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

    save_training_history(epochs_list, train_losses, val_losses, output_dir)
    return train_losses, val_losses


def save_training_history(epochs, train_losses, val_losses, output_dir):
    loss_df = pd.DataFrame({'epoch': epochs, 'train_loss': train_losses, 'val_loss': val_losses})
    loss_df.to_csv(f'{output_dir}/loss_history.csv', index=False)
    plt.figure(figsize=(10, 5))
    start_epoch = 10 if len(epochs) > 10 else 0
    plt.plot(epochs[start_epoch:], train_losses[start_epoch:], label='Train Loss')
    plt.plot(epochs[start_epoch:], val_losses[start_epoch:], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/loss_curve.png')
    plt.close()


# ==================== 5. 测试函数 ====================
def test_model(model, test_loader, AP_num, UE_num, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    total_mae = 0
    total_cos_sim = 0
    count = 0

    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            pred = model(batch_data)
            ptr = batch_data.ptr
            batch_size = batch_data.num_graphs

            for i in range(batch_size):
                start_idx = ptr[i].item()
                end_idx = ptr[i + 1].item()
                ue_start = start_idx + AP_num
                ue_end = end_idx

                pred_ue = pred[ue_start:ue_end]
                label_ue = batch_data.y[ue_start:ue_end]

                if pred_ue.numel() > 0 and label_ue.numel() > 0:
                    mse_loss = criterion(pred_ue, label_ue)
                    mae_loss = torch.abs(pred_ue - label_ue).mean()

                    pred_flat = pred_ue.flatten()
                    label_flat = label_ue.flatten()
                    
                    if pred_flat.norm() > 0 and label_flat.norm() > 0:
                        cos_sim = F.cosine_similarity(pred_flat.unsqueeze(0), label_flat.unsqueeze(0))
                        total_cos_sim += cos_sim.item()
                    else:
                        total_cos_sim += 0

                    if not (torch.isnan(mse_loss) or torch.isinf(mse_loss)):
                        total_loss += mse_loss.item()
                        total_mae += mae_loss.item()
                        count += 1

    avg_mse = total_loss / max(count, 1)
    avg_mae = total_mae / max(count, 1)
    avg_cos_sim = total_cos_sim / max(count, 1)
    return avg_mse, avg_mae, avg_cos_sim


# ==================== 6. 单数据集训练入口 ====================
def train_single_dataset(json_path, save_folder):
    try:
        graph_data_list, AP_num, max_UE, room_name = create_graph_data_list(
            json_path, SNR_MAX, SNR_MIN, R_MAX, R_MIN
        )

        if len(graph_data_list) == 0:
            print("警告: 未创建有效图数据，跳过")
            return None

        UE_num = max_UE
        timestamp = datetime.now().strftime("-%Y%m%d-%I%M%p")
        # 文件夹重命名，标记为 VectorOutput
        output_dir = os.path.join(save_folder, f'{room_name}_{AP_num}AP_{UE_num}UE_GAT_VectorOutput_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        random.seed(42)
        shuffled_list = graph_data_list.copy()
        random.shuffle(shuffled_list)

        split_ratio = 0.8
        split_idx = int(len(shuffled_list) * split_ratio)
        train_dataset = shuffled_list[:split_idx]
        val_dataset = shuffled_list[split_idx:]

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f'训练集: {len(train_dataset)}, 测试集: {len(val_dataset)}')

        node_dim = AP_num + 1
        output_dim = AP_num  # 【关键】：保持输出为 AP_num 维向量

        model = GAT_RA(
            input_dim=node_dim,
            hidden_dim=HIDDEN_DIM,
            output_dim=output_dim,
            heads=HEADS,
            dropout=DROPOUT
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params:,}")

        with open(f'{output_dir}/config.txt', 'w') as f:
            f.write(f'Room: {room_name}\n')
            f.write(f'AP_size: {AP_num}, max_UE: {UE_num}\n')
            f.write(f'Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}\n')
            f.write(f'Hidden: {HIDDEN_DIM}, Heads: {HEADS}, Output: {output_dim}\n')
            f.write(f'Activation: Sigmoid (Output), ELU (Hidden)\n')

        train_losses, val_losses = train_model(
            model, train_loader, val_loader, output_dir,
            AP_num, UE_num, epochs=EPOCHS, lr=LEARNING_RATE, gpu_id=GPU_ID
        )

        device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(f'{output_dir}/best_model.pth', map_location=device))
        model.to(device)

        test_mse, test_mae, test_cos_sim = test_model(model, val_loader, AP_num, UE_num, device)
        print(f"\n[{room_name}] 测试结果 - MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, Cosine Sim: {test_cos_sim:.6f}")

        result_data = {
            'room': room_name,
            'AP_size': AP_num,
            'max_UE_size': UE_num,
            'train_loss_final': train_losses[-1] if train_losses else 0,
            'val_loss_final': val_losses[-1] if val_losses else 0,
            'epochs': EPOCHS,
            'MSE': test_mse,
            'MAE': test_mae,
            'Cosine_Similarity': test_cos_sim
        }
        
        pd.DataFrame([result_data]).to_csv(f'{output_dir}/test_results.csv', index=False)
        return result_data

    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 7. 主函数 ====================
def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    all_results = []
    
    dataset_files = [
        'dataset3_room1-1_task3.json',
        'dataset3_room2-1_task3.json',
        'dataset3_room3-1_task3.json',
        'dataset3_room4-1_task3.json',
        'dataset3_room5-1_task3.json',
        'dataset3_room6-1_task3.json',
    ]
    
    for dataset_file in dataset_files:
        json_path = os.path.join(DATASET_BASE_PATH, dataset_file)
        if not os.path.exists(json_path):
            print(f'警告: 文件不存在 {json_path}, 跳过')
            continue
        
        print('\n' + '='*70)
        print(f'处理文件: {dataset_file}')
        print('='*70)
        
        result = train_single_dataset(json_path, SAVE_FOLDER)
        if result:
            all_results.append(result)
            
    if all_results:
        summary_df = pd.DataFrame(all_results)
        cols = ['room', 'AP_size', 'max_UE_size', 'train_loss_final', 'val_loss_final', 'MSE', 'MAE', 'Cosine_Similarity']
        summary_df = summary_df[cols]
        
        timestamp = datetime.now().strftime("%Y%m%d-%I%M%p")
        summary_path = f'{SAVE_FOLDER}/task3_GNN_test_results_SUMMARY_{timestamp}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print('\n' + '='*70)
        print('所有任务训练完成! 汇总结果如下:')
        print('='*70)
        print(summary_df.to_string(index=False))
        print(f'\n汇总文件已保存到: {summary_path}')
    else:
        print("未产生任何训练结果。")

if __name__ == "__main__":
    main()