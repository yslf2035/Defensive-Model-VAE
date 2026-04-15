"""
条件变分自编码器(Conditional VAE)轨迹生成模型

================================================================================
一、CVAE 原理（数学表达）
================================================================================

记号约定：
  x  : 观测轨迹（本实现中为相对起点偏移）x ∈ R^{T×3}，每行 [t, dx, dy]
  c  : 条件（起点坐标）c = (x_start, y_start) ∈ R^2
  z  : 潜在向量 z ∈ R^d（d = latent_dim）

1) 生成模型（解码器视角）
   - 先验：        z ~ N(0, I)
   - 条件生成：    x ~ p_θ(x | z, c)   （解码器用 z 和 c 生成 x）
   - 即：给定条件 c，从 N(0,I) 采样 z，再由解码器输出 x̂ = Decode(z, c)。

2) 推断模型（编码器视角）
   - 近似后验：    z ~ q_φ(z | x, c) = N(z; μ_φ(x,c), diag(σ²_φ(x,c)))
   - 编码器输入 (x, c)，输出 μ, log σ²，重参数化采样：
     z = μ + σ ⊙ ε,   ε ~ N(0, I)。

3) 训练目标（最大化证据下界 ELBO，等价于最小化负 ELBO）
   L = E_{q_φ(z|x,c)} [ -log p_θ(x|z,c) ] + D_KL( q_φ(z|x,c) || p(z) )
     = L_recon + L_KLD

   本实现中：
   - L_recon：用 MSE( Decode(z,c), x ) 近似 -log p_θ(x|z,c)（高斯观测）
   - L_KLD：  -0.5 * mean( 1 + log σ² - μ² - σ² )，闭式 KL(N(μ,σ²)||N(0,1))
   - 另加两项正则：
     L_start：相对坐标系下起点 (dx_0, dy_0) 的约束（接近 0）
     L_time： 时间从 0 开始且单调递增

   总损失（标量）：
   L_total = λ_recon·L_recon + λ_kld·L_KLD + λ_start·L_start + λ_time·L_time

4) 生成过程（推理时）
   - 给定任意起点 c = (x_start, y_start)：
     (1) 采样 z ~ N(0, I)
     (2) 条件编码 h_c = ConditionEncoder(c)
     (3) 相对轨迹 x̂_rel = Decode(z, h_c)，形状 [T, 3]，每行 [t, dx, dy]
     (4) 全局轨迹：x_glob(t) = x_start + dx(t), y_glob(t) = y_start + dy(t)

================================================================================
二、训练时优化的参数
================================================================================

以下均为可学习参数（由优化器如 Adam 更新）：

1) condition_encoder（条件编码器）
   - Linear(2 → hidden_dim) 的权重与偏置
   - Linear(hidden_dim → hidden_dim) 的权重与偏置
   - 将 c ∈ R^2 映射为 h_c ∈ R^{hidden_dim}

2) encoder（轨迹编码器，MLP）
   - Flatten 无参数
   - 4 个 Linear 层：Flatten(x) ∈ R^{seq_len*dim} → hidden_dim → … → hidden_dim
   - 每层含权重与偏置

3) fc_mu, fc_logvar（潜在空间映射）
   - fc_mu:    Linear(hidden_dim + hidden_dim → latent_dim)，输出 μ
   - fc_logvar: Linear(hidden_dim + hidden_dim → latent_dim)，输出 log σ²
   - 输入为 [h_traj; h_c] 的拼接

4) decoder（解码器，MLP）
   - Linear(latent_dim + hidden_dim → hidden_dim)
   - 若干 Linear(hidden_dim → hidden_dim)
   - 最后一层 Linear(hidden_dim → seq_len*dim) + Unflatten → (seq_len, dim)
   - 输出相对轨迹 [t, dx, dy]

汇总：所有上述 Linear 的 weight 和 bias 均在训练中更新，无冻结参数。

================================================================================
三、数据与实现约定（与原注释一致）
================================================================================

模型原理：
1. 编码器：将轨迹（相对偏移）与起点条件编码到潜在空间 (μ, log σ²)
2. 解码器：从 z 与条件生成相对轨迹 [t, dx, dy]
3. 变分推断：学习条件分布，支持给定起点 c 生成新轨迹
4. 条件约束：通过起点 c 控制生成轨迹的“锚点”，相对轨迹学 (dx, dy)

训练目标：
- 重构损失：重建相对轨迹与输入相对轨迹相似
- KL 散度：正则化潜在空间
- 起点损失：相对坐标系下起点接近 (0,0)
- 时间损失：时间从 0 起且递增

数据格式：
- 原始数据：(batch_size, seq_len, 3) - [时间t, x坐标, y坐标]
- 条件：起点坐标 (x_start, y_start) ∈ R^2
- 训练时内部转为相对偏移：dx = x - x_start, dy = y - y_start，时间 t 不变
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
def conditional_vae_loss(recon_x, x, mu, logvar, condition, recon_weight=0.1, kld_weight=0.1, start_weight=1.0, time_weight=0.5):
    """
    条件VAE损失函数，包含起点约束和时间约束
    Args:
        recon_x, x: 此处的x为“相对起点偏移轨迹”，格式为 [t, dx, dy]
        recon_weight: 重构损失的权重
        kld_weight: KL散度的权重
        start_weight: 起点约束的权重，控制起点约束的强度
        time_weight: 时间约束的权重，控制时间约束的强度
    """
    # 重构损失：确保重建轨迹与原始轨迹相似
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    # KL散度：正则化潜在空间分布，使其接近标准正态分布
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 起点约束损失：确保生成轨迹的起点符合条件
    start_loss = 0
    if start_weight > 0:
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
    total_loss = recon_weight * recon_loss + kld_weight * kld + start_weight * start_loss + time_weight * time_loss
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
    model_save_path = 'training/models/vae_offset_' + model_name + '_ld' + str(latent_dim) + '_epoch' + str(epochs) + '_loss2.pth'  # 模型保存路径
    loss_save_path = 'training/loss/vae_offset_' + model_name + '_ld' + str(latent_dim) + '_epoch' + str(epochs) + '_loss2.png'

    # ====== 起点终点控制参数 ======
    # 训练时：始终使用真实数据的起点坐标（推荐）
    # 生成时：可选择使用真实数据或自定义起点坐标
    
    # 生成时的起点控制
    use_training_start_end = True  # 生成时是否使用训练数据的起点坐标
    
    # 自定义起点设置（仅在use_training_start_end=False时生效）
    custom_start_end = [(155.0, -15.0), (155.0, 40.0)]  # 自定义起点坐标，格式为[(start_x, start_y), (end_x, end_y)]

    # 重构权重（训练时使用）
    recon_weight = 0.1
    # KL散度权重（训练时使用）
    kld_weight = 0.1
    # 起点约束权重（训练时使用）
    start_weight = 1.0  # 控制起点约束的强度，值越大约束越强
    # 时间约束权重（训练时使用）
    time_weight = 1.0  # 控制时间约束的强度，确保时间信息合理

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
                    recon_weight=recon_weight, kld_weight=kld_weight, start_weight=start_weight, time_weight=time_weight
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

        # loss_history_1 = {key: values[1:] for key, values in loss_history.items()}
        # epochs_1 = epochs - 1

        loss_history['recon_loss'] = [x * recon_weight for x in loss_history['recon_loss']]
        loss_history['kld_loss'] = [x * kld_weight for x in loss_history['kld_loss']]
        loss_history['start_loss'] = [x * start_weight for x in loss_history['start_loss']]
        loss_history['time_loss'] = [x * time_weight for x in loss_history['time_loss']]
        # 绘制损失曲线
        plot_losses(loss_history, epochs, loss_save_path)

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
