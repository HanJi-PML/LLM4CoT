'''
import json
import re
import numpy as np
import pandas as pd
import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==================== 1. 数据预处理 ====================
def extract_data_from_json(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    positions_list, sinr_list = [], []
    path_parts = json_path.split('/')
    for part in path_parts:
        if 'room' in part:
            room_match = re.search(r'(room\d+-\d+)', part)
            if room_match:
                room_name = room_match.group(1)
                break
    else:
        room_name = "unknown_room"

    for sample in data:
        # 提取位置
        pos_match = re.search(r'positions are (\[\[.*?\]\])', sample['input'])
        if not pos_match:
            continue

        positions_str = pos_match.group(1)
        positions = np.array(eval(positions_str))

        # 提取SINR
        sinr_match = re.search(r'SINR matrix is (\[\[.*?\]\])', sample['output'])
        if not sinr_match:
            continue

        sinr_str = sinr_match.group(1)
        # 手动解析SINR矩阵（避免eval问题）
        sinr_rows = []
        rows_str = sinr_str.strip('[]').split('] [')
        for row_str in rows_str:
            row_str = row_str.strip('[]')
            numbers = [float(x) for x in row_str.split() if x]
            sinr_rows.append(numbers)

        sinr_matrix = np.array(sinr_rows)

        # 添加到列表
        for i in range(len(positions)):
            positions_list.append(positions[i])
            sinr_list.append(sinr_matrix[i])

    positions_array = np.array(positions_list)
    sinr_array = np.array(sinr_list)
    ap_num = sinr_array.shape[1]

    # 保存CSV
    csv_filename = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/'+f'{room_name}_{ap_num}AP_task1_dataset.csv'

    df = pd.DataFrame(
        np.hstack([positions_array, sinr_array]),
        columns=['pos_x', 'pos_y'] + [f'sinr_ap{i}' for i in range(1, ap_num+1)]
    )
    df.to_csv(csv_filename, index=False)

    return positions_array, sinr_array, room_name, ap_num

# ==================== 2. 数据集类 ====================
class SINRDataset(Dataset):
    def __init__(self, positions, sinr):
        self.positions = torch.FloatTensor(positions)
        self.sinr = torch.FloatTensor(sinr)
        # 归一化
        self.pos_min = self.positions.min(0)[0]
        self.pos_max = self.positions.max(0)[0]
        self.positions = (self.positions - self.pos_min) / (self.pos_max - self.pos_min + 1e-8)
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.sinr[idx]

# ==================== 3. DNN模型 ====================
class DNNModel(nn.Module):
    def __init__(self, output_dim = 5):
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

# ==================== 4. 训练函数 ====================
def train_and_save(model, train_loader, val_loader, output_dir, batch_size, gpu_id=0):
    # 指定GPU
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    train_losses, val_losses = [], []
    epochs_list = []
    start = datetime.datetime.now()
    for epoch in range(100):
        epochs_list.append(epoch+1)
        # 训练
        model.train()
        train_loss = 0
        for pos, sinr in train_loader:
            pos, sinr = pos.to(device), sinr.to(device)

            optimizer.zero_grad()
            outputs = model(pos)
            loss = criterion(outputs, sinr)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for pos, sinr in val_loader:
                pos, sinr = pos.to(device), sinr.to(device)
                outputs = model(pos)
                val_loss += criterion(outputs, sinr).item()

        train_losses.append(train_loss/len(train_loader)/batch_size)
        val_losses.append(val_loss/len(val_loader)/batch_size)

        if (epoch+1) % 1 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    end = datetime.datetime.now()
    # 保存模型
    duration = end - start  # 得到 timedelta 对象
    print('Total training time per epoch is %s'%str((duration.total_seconds()/100)))
    torch.save(model.state_dict(), f'{output_dir}/model.pth')

     # 保存损失到CSV
    loss_csv_path = f'{output_dir}/loss_history.csv'
    loss_df = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"损失历史已保存到: {loss_csv_path}")

    # 绘制损失曲线 - 从第10个epoch开始显示后面的变化
    plt.figure(figsize=(12, 5))

    # 子图1：完整曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Full Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：从第10个epoch开始的曲线（放大看后面变化）
    plt.subplot(1, 2, 2)
    start_epoch = 10  # 从第10个epoch开始
    plt.plot(range(start_epoch, len(train_losses)+1), train_losses[start_epoch-1:],
             label='Train Loss', linewidth=2, color='blue')
    plt.plot(range(start_epoch, len(val_losses)+1), val_losses[start_epoch-1:],
             label='Val Loss', linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve (From Epoch {start_epoch})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==================== 5. 主函数 ====================
def main():
    import os
    from sklearn.model_selection import train_test_split

    output_dir = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/'
    # 1. 数据预处理
    json_path = ['/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room1-1_task1.json',
                 '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room2-1_task1.json',
                 '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room3-1_task1.json',
                 '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room4-1_task1.json',
                 '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room5-1_task1.json',
                 '/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/test_dataset_seperate/dataset3_room6-1_task1.json'
                 ]
    for i in range(len(json_path)):
        positions, sinr, room_name, ap_num = extract_data_from_json(json_path[i], output_dir)

        # 创建输出目录
        output_dir = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/%s_%sAP'%(room_name, ap_num)+datetime.datetime.now().strftime("-%Y%m%d-%I%M%p")
        os.makedirs(output_dir, exist_ok=True)

        # 2. 数据集划分 (8:2)
        X_train, X_test, y_train, y_test = train_test_split(positions, sinr, test_size=0.2, random_state=42)

        # 3. 创建数据集和数据加载器
        train_dataset = SINRDataset(X_train, y_train)
        test_dataset = SINRDataset(X_test, y_test)

        batch_size = 1000
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 4. 创建模型
        model = DNNModel(output_dim=ap_num)

        # 5. 训练模型
        train_and_save(model, train_loader, test_loader, output_dir, batch_size,gpu_id=0)

        print(f"训练完成！结果保存在: {output_dir}")
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")

if __name__ == "__main__":
    main()


'''
# ****** test the trained DNN models ******
import torch
import numpy as np
import pandas as pd
import os
import re
import datetime
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==================== 2. 数据集类 ====================
class SINRDataset(Dataset):
    def __init__(self, positions, sinr):
        self.positions = torch.FloatTensor(positions)
        self.sinr = torch.FloatTensor(sinr)
        # 归一化
        self.pos_min = self.positions.min(0)[0]
        self.pos_max = self.positions.max(0)[0]
        self.positions = (self.positions - self.pos_min) / (self.pos_max - self.pos_min + 1e-8)
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.sinr[idx]

# ==================== 3. DNN模型 ====================
class DNNModel(nn.Module):
    def __init__(self, output_dim = 5):
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

def count_model_params(model, unit='M'):
    """
    计算模型参数量
    :param model: PyTorch模型
    :param unit: 输出单位，可选 'B'(个)、'K'(千)、'M'(百万)、'G'(十亿)
    :return: 参数量（带单位）、可训练参数量（带单位）
    """
    # 统计总参数和可训练参数
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        param_count = param.numel()  # 计算单个参数张量的元素个数
        total_params += param_count
        if param.requires_grad:  # 判断是否可训练
            trainable_params += param_count

    # 单位转换
    unit_map = {'B': 1, 'K': 1e3, 'M': 1e6, 'G': 1e9}
    if unit not in unit_map:
        raise ValueError(f"unit must be in {list(unit_map.keys())}")

    total_params_scaled = total_params / unit_map[unit]
    trainable_params_scaled = trainable_params / unit_map[unit]

    return (
        f"总参数量: {total_params_scaled:.4f} {unit} ({total_params:,} 个)",
        f"可训练参数量: {trainable_params_scaled:.4f} {unit} ({trainable_params:,} 个)"
    )

def test_all_models_from_csv(model_base_dir, csv_paths):
    """从CSV文件加载数据测试模型"""
    results = []
    for csv_path in csv_paths:
        # 从CSV文件名提取房间名
        csv_name = os.path.basename(csv_path)
        room_match = re.search(r'(room\d+-\d+)', csv_name)
        room_name = room_match.group(1)

        # 加载CSV数据
        df = pd.read_csv(csv_path)
        positions = df[['pos_x', 'pos_y']].values
        sinr = df[[col for col in df.columns if 'sinr_ap' in col]].values
        ap_num = sinr.shape[1]

        # 创建数据集
        test_dataset = SINRDataset(positions, sinr)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        # 找对应的模型目录
        model_dir = None
        for subdir in os.listdir(model_base_dir):
            if subdir.startswith(room_name) and os.path.isdir(os.path.join(model_base_dir, subdir)):
                model_path = os.path.join(model_base_dir, subdir, 'model.pth')
                if os.path.exists(model_path):
                    model_dir = os.path.join(model_base_dir, subdir)
                    break

        if not model_dir:
            continue

        # 加载模型
        model = DNNModel(output_dim=ap_num)
        model.load_state_dict(torch.load(f'{model_dir}/model.pth'))
        model.cuda()
        model.eval()

        # 计算并打印
        total_info, trainable_info = count_model_params(model, unit='K')
        print(total_info)
        print(trainable_info)

        # 测试
        all_preds, all_targets = [], []
        time_list = []
        with torch.no_grad():
            for pos, target in test_loader:
                start = datetime.datetime.now()
                pos, target = pos.cuda(), target
                pred = model(pos)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.numpy())
                end = datetime.datetime.now()
                duration = end - start
                time_list.append(duration.total_seconds())
                print('Total inference time per sample is %s ms'%str((duration.total_seconds())))
        tokens_per_sample = 26 # [56.83, 26.10, 24.23, 33.76, 30.03] has 26 tokens
        inference_time_per_token = sum(time_list)/len(time_list)/tokens_per_sample
        print('Average inference time per epoch is %s ms'%str(inference_time_per_token))

        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)

        # 计算指标
        mae = np.mean(np.abs(preds - targets))

        from scipy.spatial.distance import cosine
        cos_sims = [1 - cosine(targets[i], preds[i]) for i in range(len(targets))]
        avg_cos = np.mean(cos_sims)

        results.append({
            'room': room_name,
            'ap_num': ap_num,
            'mae': mae,
            'cosine_similarity': avg_cos
        })

    return results

# CSV文件路径
# csv_files = [
#     '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/room1-1_5AP_task1_dataset.csv',
#     '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/room2-1_5AP_task1_dataset.csv',
#     '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/room3-1_6AP_task1_dataset.csv',
#     '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/room4-1_7AP_task1_dataset.csv',
#     '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/room5-1_8AP_task1_dataset.csv',
#     '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/room6-1_9AP_task1_dataset.csv'
# ]

csv_files = [
    '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/room1-1_5AP_task1_dataset.csv'
]

model_base_dir = '/data/LLM_indoor/LLaMA-Factory-main/saves/LLM_Multitask/task1_DNN/'


# 测试
results = test_all_models_from_csv(model_base_dir, csv_files)

# 保存结果
df = pd.DataFrame(results)
df.to_csv(f'{model_base_dir}/task1_DNN_test_results.csv', index=False)

# 打印结果
for result in results:
    print(f"{result['room']}: AP={result['ap_num']}, MAE={result['mae']:.4f}, Cosine={result['cosine_similarity']:.4f}")
