import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import CubicSpline, make_interp_spline


# ===================== 样条曲线函数 =====================
def create_smooth_curve(points, time_interval=0.015, start_angle=None):
    """
    使用样条曲线将给定的点连接成平滑的曲线，支持起点朝向角度约束

    Args:
        points: 输入点坐标，格式为 (n_points, 3) 的numpy数组，包含t,x和y坐标
        time_interval: 相邻点之间时间间隔，默认为0.015
        start_angle: 起点处的朝向角度（弧度），如果为None则不施加角度约束

    Returns:
        smooth_x, smooth_y: 平滑曲线的x和y坐标数组
        num_points: 平滑曲线点数
    """
    time_series = points[:, 0]
    x_coords = points[:, 1]
    y_coords = points[:, 2]

    # 生成新的时间序列
    start_time = 0
    end_time = time_series[-1]
    smooth_time = np.arange(start_time, end_time, time_interval)
    num_points = len(smooth_time)  # 计算插值后的点数

    # 参数化曲线（使用累积弦长）
    t = np.zeros(len(x_coords))
    t[1:] = np.cumsum(np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2))
    t /= t[-1]  # 归一化到 [0, 1]

    # 计算起点处的导数（如果指定了 start_angle）
    if start_angle is not None:
        # 计算导数方向向量
        dx_start = np.cos(start_angle)
        dy_start = np.sin(start_angle)
        # 计算导数缩放因子（基于点间距）
        avg_dist = np.mean(np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2))
        dx_start *= avg_dist
        dy_start *= avg_dist
        # 设置边界条件：起点处固定一阶导数，终点自然边界
        bc_type_x = ((1, dx_start), 'natural')
        bc_type_y = ((1, dy_start), 'natural')
    else:
        bc_type_x = 'natural'
        bc_type_y = 'natural'

    # 对 x 和 y 分别进行 CubicSpline 插值
    cs_x = CubicSpline(t, x_coords, bc_type=bc_type_x)
    cs_y = CubicSpline(t, y_coords, bc_type=bc_type_y)

    # 生成平滑曲线上的点
    new_t = np.linspace(0, 1, num_points)
    smooth_x = cs_x(new_t)
    smooth_y = cs_y(new_t)

    smooth_trajectory = np.column_stack([smooth_time, smooth_x, smooth_y])

    return smooth_trajectory


# ===================== 可视化函数 =====================
def visualize_trajectories(model, dataset, model_save_path, axis_flip='none',
                           use_training_start_end=True, custom_start_end=None,
                           train_traj_start=0, train_traj_end=9):
    """
    可视化轨迹对比（支持条件VAE）

    条件生成原理：
    1. 提取或指定起点坐标条件
    2. 将条件编码为条件向量
    3. 从潜在空间采样随机向量
    4. 结合条件向量和随机向量生成轨迹

    使用训练数据起点坐标的逻辑：
    - 取指定范围的训练轨迹
    - 第i条生成的轨迹使用第i条训练轨迹的起点坐标作为条件
    - 这样可以保持生成轨迹与训练数据的对应关系
    - 例如：生成轨迹1使用训练轨迹1的起点坐标，生成轨迹2使用训练轨迹2的起点坐标

    Args:
        model: 训练好的VAE模型
        dataset: 训练数据集
        model_save_path: 模型保存路径
        axis_flip: 坐标轴翻转选项
        use_training_start_end: 是否使用训练数据的起点坐标
        custom_start_end: 自定义起点坐标，格式为[(start_x, start_y), (end_x, end_y)]（注意：只使用起点坐标）
        train_traj_start: 绘制训练数据的起始轨迹索引
        train_traj_end: 绘制训练数据的结束轨迹索引（不包含）
    """
    model.eval()  # 设置模型为评估模式

    # 计算轨迹数量
    num_samples = train_traj_end - train_traj_start

    with torch.no_grad():  # 禁止梯度计算
        # 设置全局字体为Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12

        # 获取指定范围的训练数据作为对比
        train_data = dataset.data[train_traj_start:train_traj_end]  # 取指定范围的训练轨迹

        # 确定起点条件（这是条件生成的关键）
        if use_training_start_end:
            # 使用训练数据的起点坐标（保持数据真实性）
            # 注意：这里使用的是指定范围的训练轨迹的起点坐标
            # 每条生成的轨迹对应一条训练轨迹的起点坐标
            start_points = train_data[:, 0, 1:3]  # (num_samples, 2) - 起点(x,y) - numpy数组
        elif custom_start_end is not None:
            # 使用自定义起点坐标（可精确控制生成轨迹的起点）
            start_x, start_y = custom_start_end[0]
            start_points = torch.tensor([[start_x, start_y]] * num_samples, dtype=torch.float32)
        else:
            # 使用训练数据的起点坐标作为默认
            start_points = train_data[:, 0, 1:3]  # 起点(x,y)

        # 生成条件向量（将起点坐标编码为条件信息）
        # 需要将numpy数组转换为tensor
        if isinstance(start_points, np.ndarray):  # 检查是否是numpy数组类型
            start_points = torch.from_numpy(start_points).float()

        condition = start_points  # (num_samples, 2) - 只需要起点坐标
        condition = condition.to(next(model.parameters()).device)

        # 编码条件信息（这是条件VAE的核心）
        h_condition = model.condition_encoder(condition)

        # 从潜在空间采样随机向量
        z = torch.randn(num_samples, model.latent_dim).to(next(model.parameters()).device)

        # 条件生成：结合随机向量和条件向量生成轨迹
        generated_samples = model.decode(z, h_condition).cpu().numpy()  # (num_samples, seq_len, dim)
        
        # 调试信息：显示生成的时间信息
        print(f"\n=== 生成轨迹的时间信息 ===")
        for i in range(min(3, num_samples)):
            print(f"轨迹 {train_traj_start+i+1}:")
            print(f"  训练数据时间: {train_data[i, :, 0]}")
            print(f"  生成数据时间: {generated_samples[i, :, 0]}")
            print(f"  时间差异: {np.abs(train_data[i, :, 0] - generated_samples[i, :, 0])}")
            print(f"  训练时间范围: {train_data[i, 0, 0]:.2f}s - {train_data[i, -1, 0]:.2f}s")
            print(f"  生成时间范围: {generated_samples[i, 0, 0]:.2f}s - {generated_samples[i, -1, 0]:.2f}s")
            print()

        # 确定子图布局
        n_cols = int(np.ceil(np.sqrt(num_samples)))
        n_rows = int(np.ceil(num_samples / n_cols))

        plt.figure(figsize=(9, 9))
        for i in range(num_samples):
            plt.subplot(n_rows, n_cols, i+1)

            # 检查是否需要绘制车道线
            model_name = model_save_path.split('/')[-1]
            if "sce1" in model_name:
                # 绘制车道线（黑色实线）
                y_range = np.linspace(20, 73, 100)
                plt.plot([-193.31] * len(y_range), y_range, 'k-', linewidth=2, alpha=0.7)
                plt.plot([-196.81] * len(y_range), y_range, 'k-', linewidth=2, alpha=0.7)
            elif "sce2" in model_name:
                # 绘制车道线（黑色实线）
                x_range = np.linspace(-177, -110, 100)
                plt.plot(x_range, [-5.8] * len(x_range), 'k-', linewidth=2, alpha=0.7)
                plt.plot(x_range, [-2.3] * len(x_range), 'k--', linewidth=2, alpha=0.7)
                plt.plot(x_range, [1.2] * len(x_range), 'k-', linewidth=2, alpha=0.7)
                # 绘制背景车2的轨迹（深绿色）
                try:
                    # 读取背景车轨迹数据
                    bg_data = pd.read_csv(
                        'DefensiveData/DynamicBlindTown05/减速/exp_1_control_DynamicBlindTown05_3.csv')
                    bg_x = bg_data['sv2_x'].values
                    bg_y = bg_data['sv2_y'].values
                    # 绘制背景车轨迹（深绿色）
                    plt.plot(bg_x, bg_y, color='darkgreen', linewidth=2, alpha=0.8, label='BV1')
                except Exception as e:
                    print(f"无法读取背景车轨迹数据: {e}")
            elif "sce3" in model_name:
                # 绘制车道线（黑色实线）
                y_range = np.linspace(-100, 60, 100)
                plt.plot([153.3] * len(y_range), y_range, 'k-', linewidth=2, alpha=0.7)
                plt.plot([156.8] * len(y_range), y_range, 'k-', linewidth=2, alpha=0.7)
                # 绘制背景车1的轨迹（深绿色）
                try:
                    # 读取背景车轨迹数据
                    bg_data = pd.read_csv('DefensiveData/PredictableMovementTown05/减速/exp_12_control_PredictableMovementTown05_2.csv')
                    bg_x = bg_data['sv1_x'].values
                    bg_y = bg_data['sv1_y'].values
                    # 绘制背景车轨迹（深绿色）
                    plt.plot(bg_x, bg_y, color='darkgreen', linewidth=2, alpha=0.8, label='BV1')
                except Exception as e:
                    print(f"无法读取背景车轨迹数据: {e}")
            elif "sce4" in model_name:
                # 绘制车道线（黑色实线）
                y_range = np.linspace(-40, 120, 100)
                plt.plot([18] * len(y_range), y_range, 'k-', linewidth=2, alpha=0.7)
                plt.plot([14.5] * len(y_range), y_range, 'k--', linewidth=2, alpha=0.7)
                plt.plot([11] * len(y_range), y_range, 'k--', linewidth=2, alpha=0.7)
                plt.plot([7.5] * len(y_range), y_range, 'k--', linewidth=2, alpha=0.7)
                plt.plot([4] * len(y_range), y_range, 'k-', linewidth=2, alpha=0.7)
                # 绘制背景车1的轨迹（深绿色）
                try:
                    # 读取背景车轨迹数据
                    bg_data = pd.read_csv(
                        'DefensiveData/UnpredictableMovementTown04/减速/exp_14_control_UnpredictableMovementTown04_3.csv')
                    bg_x = bg_data['sv1_x'].values
                    bg_y = bg_data['sv1_y'].values
                    # 绘制背景车轨迹（深绿色）
                    plt.plot(bg_x, bg_y, color='darkgreen', linewidth=2, alpha=0.8, label='BV1')
                except Exception as e:
                    print(f"无法读取背景车轨迹数据: {e}")

            # 创建平滑曲线
            # 训练数据轨迹点
            train_points = train_data[i, :, 1:3]  # 只取x和y坐标
            
            # 计算训练数据的起点角度（从起点到第二个点的方向）
            if len(train_points) >= 2:
                dx_train = train_points[1, 0] - train_points[0, 0]
                dy_train = train_points[1, 1] - train_points[0, 1]
                train_start_angle = np.arctan2(dy_train, dx_train)
            else:
                train_start_angle = None
            
            train_smooth_x, train_smooth_y = create_smooth_curve(train_points, time_interval=0.015)
            
            # 生成轨迹点
            gen_points = generated_samples[i, :, 1:3]  # 只取x和y坐标
            
            gen_smooth_x, gen_smooth_y = create_smooth_curve(gen_points, time_interval=0.015, start_angle=(-90)*(math.pi/180))

            # 绘制平滑的训练数据轨迹（红色）
            plt.plot(train_smooth_x, train_smooth_y, 'r-', linewidth=2, alpha=0.8, label='Training Data')
            # 绘制训练数据的关键点（红色方块）
            plt.plot(train_data[i, :, 1], train_data[i, :, 2], 'rs', markersize=4, alpha=0.6)
            
            # 添加训练数据的时间标签（右侧）
            for j in range(len(train_data[i])):
                t_val = train_data[i, j, 0]  # 时间值
                x_val = train_data[i, j, 1]  # x坐标
                y_val = train_data[i, j, 2]  # y坐标
                # 只在关键点显示时间标签（每隔几个点显示一次）
                if j % 3 == 0 or j == 0 or j == len(train_data[i]) - 1:
                    plt.annotate(f't={t_val:.2f}s', (x_val, y_val), 
                               xytext=(8, 0), textcoords='offset points',
                               fontsize=7, color='red', alpha=0.8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # 绘制平滑的生成轨迹（蓝色）
            plt.plot(gen_smooth_x, gen_smooth_y, 'b-', linewidth=2, alpha=0.8, label='Generated')
            # 绘制生成轨迹的关键点（蓝色圆圈）
            plt.plot(generated_samples[i, :, 1], generated_samples[i, :, 2], 'bo', markersize=4, alpha=0.6)
            
            # 添加生成轨迹的时间标签（左侧）
            for j in range(len(generated_samples[i])):
                t_val = generated_samples[i, j, 0]  # 时间值
                x_val = generated_samples[i, j, 1]  # x坐标
                y_val = generated_samples[i, j, 2]  # y坐标
                # 只在关键点显示时间标签（每隔几个点显示一次）
                if j % 3 == 0 or j == 0 or j == len(generated_samples[i]) - 1:
                    plt.annotate(f't={t_val:.2f}s', (x_val, y_val), 
                               xytext=(-30, 0), textcoords='offset points',
                               fontsize=7, color='blue', alpha=0.8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

            plt.grid(True, alpha=0.3)
            if "sce1" in model_name:
                plt.xlim(-200, -190)
                plt.ylim(35, 100)
            elif "sce2" in model_name:
                plt.xlim(-220, -130)
                plt.ylim(-10, 5)
            elif "sce3" in model_name:
                plt.xlim(150, 160)
                plt.ylim(-100, 50)
            elif "sce4" in model_name:
                plt.xlim(-8, 32)
                plt.ylim(0, 110)
            else:
                plt.xlim(150, 160)
                plt.ylim(-100, 50)
            if i == 0:
                plt.legend(framealpha=0.6)
            plt.title(f'Trajectory {train_traj_start+i+1}')  # 显示实际的训练轨迹编号

            # 坐标轴翻转
            if axis_flip == 'x':
                plt.gca().invert_xaxis()
            elif axis_flip == 'y':
                plt.gca().invert_yaxis()
            elif axis_flip == 'xy':
                plt.gca().invert_xaxis()
                plt.gca().invert_yaxis()

        model_name = model_save_path.split('/')[-1]
        plt.suptitle(f'Generated vs Training Trajectories ({train_traj_start+1}-{train_traj_end}) -- {model_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
