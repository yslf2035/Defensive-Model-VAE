"""
条件变分自编码器(Conditional VAE)轨迹生成模型

模型原理：
1. 编码器：将轨迹数据（包含时间信息）和起点条件编码到潜在空间
2. 解码器：从潜在空间和条件信息重建轨迹
3. 变分推断：学习数据的概率分布，支持生成新轨迹
4. 条件约束：通过起点坐标条件控制生成轨迹的起点

训练目标：
- 重构损失：确保重建轨迹与原始轨迹（在相对坐标系下）相似
- KL散度：正则化潜在空间分布
- 起点损失：约束生成轨迹在相对坐标系下的起点为(0,0)

数据格式：
- 原始数据：(batch_size, seq_len, 3) - [时间t, x坐标, y坐标]
- 条件：起点坐标 (x, y) - 2维

本实现中的实际训练形式：
- 时间t 保持为绝对时间，不做平移
- 位置部分改为相对于起点的偏移量：
  dx = x - x_start, dy = y - y_start
- 模型学习的是随时间变化的偏移轨迹 [t, dx, dy]
- 生成时：先根据给定起点 (x_start, y_start) 和随机潜在向量生成 [t, dx, dy]，
  再通过 x = x_start + dx, y = y_start + dy 还原到全局坐标系
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from Tools import *

# ===================== 数据集定义 =====================
class TrajectoryDataset(Dataset):
    """轨迹数据集类，用于加载和处理轨迹数据"""
    def __init__(self, data_path):
        self.data = np.load(data_path)  # 假设数据为 (num_samples, seq_len, dim)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ===================== 条件VAE模型定义 =====================
class ConditionalTrajectoryVAE(nn.Module):
    """
    条件变分自编码器，用于轨迹生成
    特点：将起点坐标作为条件信息，控制生成轨迹的起点
    数据格式：(batch_size, seq_len, 3) - 其中3维为[时间t, x坐标, y坐标]
    """
    def __init__(self, seq_len, dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 起点编码器（只考虑起点的x和y坐标）
        self.condition_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 起点(x,y) = 2维
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 编码器：4层MLP网络 + 条件信息
        # 输入维度：seq_len * dim (包含时间信息)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * dim, hidden_dim),  # seq_len * 3
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 潜在空间映射层（加入条件信息）
        self.fc_mu = nn.Linear(hidden_dim + hidden_dim, latent_dim)  # 轨迹特征 + 条件特征
        self.fc_logvar = nn.Linear(hidden_dim + hidden_dim, latent_dim)
        
        # 解码器：4层MLP网络 + 条件信息
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),  # 潜在向量 + 条件向量
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * dim),  # 输出包含时间信息
            nn.Unflatten(1, (seq_len, dim))
        )

    def get_start_points(self, x):
        """
        提取轨迹的起点坐标（只考虑x和y）
        注意：
        - 当前训练流程会在外部将坐标转换为“相对起点偏移”(dx, dy)，
          此时该函数得到的通常是(0,0)，仅保留用于兼容/调试。
        """
        # x格式：(batch_size, seq_len, 3) - [时间t, x坐标或dx, y坐标或dy]
        start_points = x[:, 0, 1:3]
        return start_points

    def encode(self, x, start_points):
        """
        编码过程：将轨迹和起点条件编码到潜在空间
        输入x格式：(batch_size, seq_len, 3) - [时间t, dx, dy]（相对起点偏移）
        start_points格式：(batch_size, 2) - 全局起点坐标(x_start, y_start)
        """
        # 编码轨迹（包含时间信息 + 相对位置信息）
        h_traj = self.encoder(x)

        # 编码条件信息（使用全局起点坐标）
        h_condition = self.condition_encoder(start_points)

        # 合并轨迹特征和条件特征
        h_combined = torch.cat([h_traj, h_condition], dim=1)

        mu = self.fc_mu(h_combined)
        logvar = self.fc_logvar(h_combined)
        return mu, logvar, h_condition

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从正态分布采样潜在向量
        这是VAE的核心，使得模型可以进行反向传播
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        """
        解码过程：从潜在向量和条件信息重建轨迹
        输出格式：(batch_size, seq_len, 3) - [时间t, x坐标, y坐标]
        """
        # 将潜在向量和条件向量结合
        z_condition = torch.cat([z, condition], dim=1)
        return self.decoder(z_condition)

    def forward(self, x, start_points):
        """
        前向传播：完整的编码-解码过程
        输入x格式：(batch_size, seq_len, 3) - [时间t, dx, dy]（相对起点偏移）
        start_points格式：(batch_size, 2) - 全局起点坐标(x_start, y_start)
        """
        mu, logvar, condition = self.encode(x, start_points)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition)
        return recon_x, mu, logvar, condition

# ===================== 条件VAE损失函数 =====================
def conditional_vae_loss(recon_x, x, mu, logvar, condition, alpha=1.0, time_weight=0.1):
    """
    条件VAE损失函数，包含起点约束和时间约束
    Args:
        recon_x, x: 此处的x为“相对起点偏移轨迹”，格式为 [t, dx, dy]
        alpha: 起点约束的权重，控制起点约束的强度
        time_weight: 时间约束的权重，控制时间约束的强度
    """
    # 重构损失：确保重建轨迹与原始轨迹相似
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    # KL散度：正则化潜在空间分布，使其接近标准正态分布
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 起点约束损失：确保生成轨迹的起点符合条件
    start_loss = 0
    if alpha > 0:
        # 此时x和recon_x都是相对坐标，理论上起点应为(0,0)，
        # 该约束鼓励模型在相对坐标系中保持起点为零
        real_start = x[:, 0, 1:3]      # 真实相对起点(dx, dy)，应接近(0,0)
        recon_start = recon_x[:, 0, 1:3]  # 重构相对起点(dx, dy)
        start_loss = nn.functional.mse_loss(recon_start, real_start, reduction='mean')
    
    # 时间约束损失：确保生成的时间信息合理
    time_loss = 0
    if time_weight > 0:
        # 确保时间从0开始
        time_start_loss = nn.functional.mse_loss(recon_x[:, 0, 0], torch.zeros_like(recon_x[:, 0, 0]), reduction='mean')
        
        # 确保时间递增
        time_diff = recon_x[:, 1:, 0] - recon_x[:, :-1, 0]  # 时间差
        time_increasing_loss = torch.mean(torch.relu(-time_diff))  # 惩罚负的时间差
        
        time_loss = time_start_loss + time_increasing_loss
    
    # 总损失：重构损失 + KL散度 + 起点约束 + 时间约束
    total_loss = recon_loss + kld + alpha * start_loss + time_weight * time_loss
    return total_loss, recon_loss, kld, start_loss, time_loss

# ===================== 训练主程序 =====================
if __name__ == "__main__":
    # ====== 可修改参数 ======
    mode = 'training'  # 'training', 'visualization'
    data_path = 'training/DefensiveDataProcessed/trajectory_sce1_cond.npy'  # 轨迹数据集路径，需为numpy数组 (num_samples, seq_len, dim)
    seq_len = 10                  # 轨迹长度（每条轨迹包含的采样点数量）
    dim = 3                       # 每个点的维度（3:t,x,y）
    latent_dim = 8                # 潜在空间维度（VAE编码器输出的潜在向量维度）
    batch_size = 38               # 批大小（每次训练使用的样本数量）: sce1 = 38, sce2 = 16, sce3 = 66, sce4 = 135
    lr = 1e-3                     # 学习率（优化器更新参数时的步长）
    epochs = 3000                 # 训练轮数
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备
    device = 'cpu'
    model_name = data_path.split('/')[-1]
    model_name = model_name.split('.')[0]
    model_name = model_name.replace("trajectory_", "", 1)
    model_save_path = 'training/models/vae_offset_' + model_name + '_ld' + str(latent_dim) + '_epoch' + str(epochs) + '.pth'  # 模型保存路径
    loss_save_path = 'training/loss/vae_offset_' + model_name + '_ld' + str(latent_dim) + '_epoch' + str(epochs) + '.png'

    # ====== 起点终点控制参数 ======
    # 训练时：始终使用真实数据的起点坐标（推荐）
    # 生成时：可选择使用真实数据或自定义起点坐标
    
    # 生成时的起点控制
    use_training_start_end = True  # 生成时是否使用训练数据的起点坐标
    
    # 自定义起点设置（仅在use_training_start_end=False时生效）
    custom_start_end = [(155.0, -15.0), (155.0, 40.0)]  # 自定义起点坐标，格式为[(start_x, start_y), (end_x, end_y)]
    
    # 起点约束权重（训练时使用）
    start_end_weight = 2.0  # 控制起点约束的强度，值越大约束越强
    
    # 时间约束权重（训练时使用）
    time_weight = 2.0  # 控制时间约束的强度，确保时间信息合理

    # ====== 可视化控制参数 ======
    # 控制绘制训练数据的轨迹范围
    train_traj_start = 0  # 绘制训练数据的起始轨迹索引（从0开始）
    train_traj_end = 9    # 绘制训练数据的结束轨迹索引（不包含）
    # 例如：train_traj_start=0, train_traj_end=9 表示绘制第0-8条训练轨迹（共9条）
    axis_flip = 'y'  # 可选'none'（不翻转）、'x'（翻转x轴）、'y'（翻转y轴）、'xy'（同时翻转x轴和y轴）
    # =====================

    if mode == 'training':
        # ========== 训练模式 ==========
        print("开始训练条件VAE模型...")
        print(f"训练参数：轨迹长度={seq_len}, 潜在维度={latent_dim}, 批大小={batch_size}, 学习率={lr}")
        print(f"数据格式：原始数据为时间t + 绝对坐标(x,y) (3维)")
        print(f"训练时：内部会自动转换为时间t + 相对起点偏移(dx, dy)")
        print(f"条件约束：使用绝对起点坐标(x_start, y_start)作为条件")
        print(f"时间约束：确保生成的时间从0开始且递增")
        
        # 加载数据
        dataset = TrajectoryDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"数据集大小：{len(dataset)} 条轨迹")

        # 初始化模型
        model = ConditionalTrajectoryVAE(seq_len, dim, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print("模型初始化完成，开始训练...")

        # 训练循环
        model.train()
        loss_history = {'total_loss': [], 'recon_loss': [], 'kld_loss': [], 'start_loss': [], 'time_loss': []}
        for epoch in range(epochs):
            total_loss, total_recon, total_kld, total_start, total_time = 0, 0, 0, 0, 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = batch.to(device)  # 原始绝对坐标：[t, x, y]

                # ===== 将坐标转换为“相对起点偏移”形式 =====
                # 提取全局起点坐标 (batch_size, 2)
                start_points = batch[:, 0, 1:3]  # (x_start, y_start)
                # 构造相对坐标：dx = x - x_start, dy = y - y_start
                batch_rel = batch.clone()
                batch_rel[:, :, 1:3] = batch_rel[:, :, 1:3] - start_points.unsqueeze(1)

                # 前向传播：编码-解码过程（在相对坐标系中）
                optimizer.zero_grad()
                recon_batch, mu, logvar, condition = model(batch_rel, start_points)

                # 计算损失：重构损失 + KL散度 + 起点约束 + 时间约束
                # 注意：此处的x使用的是相对坐标 batch_rel
                loss, recon_loss, kld, start_loss, time_loss = conditional_vae_loss(
                    recon_batch, batch_rel, mu, logvar, condition,
                    alpha=start_end_weight, time_weight=time_weight
                )
                
                # 反向传播和参数更新
                loss.backward()
                optimizer.step()
                
                # 累计损失
                total_loss += loss.item() * batch.size(0)
                total_recon += recon_loss.item() * batch.size(0)
                total_kld += kld.item() * batch.size(0)
                total_start += start_loss.item() * batch.size(0)
                total_time += time_loss.item() * batch.size(0)
            
            # 打印训练进度
            print(f"Epoch {epoch+1}: Loss={total_loss/len(dataset):.4f}, Recon={total_recon/len(dataset):.4f}, KLD={total_kld/len(dataset):.4f}, Start={total_start/len(dataset):.4f}, Time={total_time/len(dataset):.4f}")
            
            # 记录损失
            loss_history['total_loss'].append(total_loss / len(dataset))
            loss_history['recon_loss'].append(total_recon / len(dataset))
            loss_history['kld_loss'].append(total_kld / len(dataset))
            loss_history['start_loss'].append(total_start / len(dataset))
            loss_history['time_loss'].append(total_time / len(dataset))

        # 绘制损失曲线
        loss_history_1 = {key: values[1:] for key, values in loss_history.items()}
        epochs_1 = epochs - 1
        plot_losses(loss_history_1, epochs_1, loss_save_path)

        # 保存训练好的模型
        torch.save(model.state_dict(), model_save_path)
        print(f"模型训练完成，已保存为 {model_save_path}")

    elif mode == 'visualization':
        # ========== 可视化模式 ==========
        print("开始可视化轨迹生成...")

        # 加载训练好的模型
        model = ConditionalTrajectoryVAE(seq_len, dim, latent_dim).to(device)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"模型加载完成：{model_save_path}")

        # 加载数据用于对比
        dataset = TrajectoryDataset(data_path)
        print(f"数据集加载完成：{len(dataset)} 条轨迹")

        # 计算轨迹数量
        num_samples = train_traj_end - train_traj_start
        print(f"将绘制 {num_samples} 条轨迹（索引范围：{train_traj_start}-{train_traj_end-1}）")

        # 根据设置选择起点终点策略
        if use_training_start_end:
            print("使用训练数据的起点坐标进行生成")
            print(f"将使用训练轨迹{train_traj_start+1}-{train_traj_end}的起点坐标作为条件")
            # 显示前几条轨迹的起点坐标作为示例
            train_data_sample = dataset.data[train_traj_start:train_traj_end]
            for i in range(min(3, len(train_data_sample))):
                start = train_data_sample[i, 0, 1:3]  # 起点(x,y)
                print(f"  轨迹{train_traj_start+i+1}: 起点({start[0]:.2f}, {start[1]:.2f})")
        else:
            print(f"使用自定义起点坐标进行生成：{custom_start_end}")
            print(f"所有{num_samples}条生成轨迹将使用相同的起点坐标条件")

        # 执行可视化
        visualize_trajectories(model, dataset, model_save_path, axis_flip=axis_flip,
                              use_training_start_end=use_training_start_end,
                              custom_start_end=custom_start_end,
                              train_traj_start=train_traj_start,
                              train_traj_end=train_traj_end)
    
    else:
        print("错误：mode参数必须是 'training' 或 'visualization'")
