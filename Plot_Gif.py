"""
车辆轨迹动图生成器

功能：
1. 加载训练好的VAE模型
2. 生成轨迹数据
3. 使用样条曲线拟合轨迹点
4. 根据时间信息生成GIF动图
5. 在动图中绘制车辆矩形和轨迹

数据格式：
- 模型输出：(batch_size, seq_len, 3) - [时间t, x坐标, y坐标]
- CSV数据：包含ego_x, ego_y, ego_yaw, sv1_vx, sv1_vy等列
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pandas as pd
import math
import os
from Tools import create_smooth_curve
from Training_VAE import ConditionalTrajectoryVAE, TrajectoryDataset

def load_model_and_generate_trajectory(model_path, start_x, start_y, seq_len=12, dim=3, latent_dim=8, device='cpu'):
    """
    加载模型并生成轨迹

    Args:
        model_path: 模型文件路径
        seq_len: 轨迹长度
        dim: 每个点的维度
        latent_dim: 潜在空间维度
        device: 计算设备

    Returns:
        generated_trajectory: 生成的轨迹数据 (seq_len, 3) - [时间, x, y]
    """
    # 加载模型
    model = ConditionalTrajectoryVAE(seq_len, dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 生成轨迹
    with torch.no_grad():
        # 从潜在空间采样随机向量
        z = torch.randn(1, latent_dim).to(device)

        # 创建起点条件
        start_points = np.array([start_x, start_y], ndmin=2)
        if isinstance(start_points, np.ndarray):
            start_points = torch.from_numpy(start_points).float()
        condition = start_points
        condition = condition.to(next(model.parameters()).device)
        # 编码条件信息
        h_condition = model.condition_encoder(condition)

        # 生成轨迹
        generated_trajectory = model.decode(z, h_condition).cpu().numpy()[0]  # (seq_len, 3)
    
    return generated_trajectory

def get_start_conditions_from_csv(csv_path, scenario_type="sce3"):
    """
    从CSV文件中获取起始条件
    
    Args:
        csv_path: CSV文件路径
        scenario_type: 场景类型 ("sce1", "sce3" 或 "sce4")
        
    Returns:
        start_x, start_y, start_angle: 起始x坐标、y坐标、角度（弧度）
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
    
        if scenario_type == "sce1":
            # sce1场景：ego_y >= 40
            mask = (df['ego_y'] >= 40)
        elif scenario_type == "sce2":
            # sce2场景：sv1_yaw < -170
            mask = (df['sv1_yaw'] < -170)
        elif scenario_type == "sce4":
            # sce4场景：sv1_x < 9 且 sv1_yaw > -89
            mask = (df['sv1_x'] < 9) & (df['sv1_yaw'] > -89)
        else:
            # sce3场景：sv1_vx != 0, sv1_vy != 0, ego_y <= 40, ego_y != 0
            mask = (
                (df['sv1_vx'] != 0) &
                (df['sv1_vy'] != 0) &
                (df['ego_y'] <= 40) &
                (df['ego_y'] != 0)
            )
        
        if not mask.any():
            print(f"警告：未找到满足条件的起始行，使用默认值")
            if scenario_type == "sce1":
                return -193.3, 50.0, -90 * math.pi / 180
            elif scenario_type == "sce2":
                return -155.0, -5.0, -90 * math.pi / 180
            elif scenario_type == "sce4":
                return 11.0, 0.0, -90 * math.pi / 180
            else:
                return 155.0, -15.0, -90 * math.pi / 180
        
        # 获取第一行满足条件的数据
        start_row = df[mask].iloc[0]
        
        start_x = start_row['ego_x']
        start_y = start_row['ego_y']
        start_angle = start_row['ego_yaw'] * math.pi / 180
        
        print(f"从CSV获取起始条件：x={start_x:.2f}, y={start_y:.2f}, angle={start_angle:.2f}rad")
        
        return start_x, start_y, start_angle
        
    except Exception as e:
        print(f"读取CSV文件失败：{e}")
        print("使用默认起始条件")
        if scenario_type == "sce1":
            return -193.3, 50.0, -90 * math.pi / 180
        elif scenario_type == "sce2":
            return -155.0, -5.0, -90 * math.pi / 180
        elif scenario_type == "sce4":
            return 11.0, 0.0, -90 * math.pi / 180
        else:
            return 155.0, -15.0, -90 * math.pi / 180

def get_human_and_bg_trajectories_from_csv(csv_path, scenario_type="sce3"):
    """
    从CSV文件中获取人类轨迹和背景车轨迹
    
    Args:
        csv_path: CSV文件路径
        scenario_type: 场景类型 ("sce1", "sce3" 或 "sce4")
        
    Returns:
        human_trajectory: 人类轨迹数据 (num_points, 3) - [时间, x, y]
        bg1_trajectory: 背景车1轨迹数据 (num_points, 3) - [时间, x, y]
        bg2_trajectory: 背景车2轨迹数据 (num_points, 3) - [时间, x, y] (仅sce1)
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        if scenario_type == "sce1":
            # sce1场景：起始条件
            start_mask = (df['ego_y'] >= 40)
            # 终止条件
            end_mask = (df['sv2_x'] <= -199) & (df['ego_y'] >= 95)
        elif scenario_type == "sce2":
            # sce2场景：起始条件（sv1_yaw < -170）
            start_mask = (df['sv1_yaw'] < -170)
            # 终止条件：若缺省，则使用文件末尾
            end_mask = df['ego_x'] < -180
        elif scenario_type == "sce4":
            # sce4场景：起始条件
            start_mask = (df['sv1_x'] < 9) & (df['sv1_yaw'] > -89)
            # 终止条件
            end_mask = (df['sv1_x'] > 15) & (df['sv1_yaw'] < -85)
        else:
            # sce3场景：起始条件
            start_mask = (
                (df['sv1_vx'] != 0) &
                (df['sv1_vy'] != 0) &
                (df['ego_y'] <= 40) &
                (df['ego_y'] != 0)
            )
            # 终止条件
            end_mask = df['ego_y'] <= -90
        
        if not start_mask.any():
            print("警告：未找到满足条件的起始行")
            return None, None, None
        
        start_idx = df[start_mask].index[0]
        
        if not end_mask.any():
            print("警告：未找到满足条件的终止行，使用文件末尾")
            end_idx = len(df) - 1
        else:
            end_idx = df[end_mask].index[0]
        
        # 确保终止行在起始行之后
        if end_idx <= start_idx:
            print("警告：终止行在起始行之前或相同")
            return None, None, None
        
        # 提取轨迹数据
        trajectory_data = df.iloc[start_idx:end_idx+1]
        
        # 生成时间序列（从0开始，每行间隔0.015s）
        num_points = len(trajectory_data)
        times = np.arange(0, num_points * 0.015, 0.015)[:num_points]
        
        # 人类轨迹（ego_x, ego_y）
        human_x = trajectory_data['ego_x'].values
        human_y = trajectory_data['ego_y'].values
        human_trajectory = np.column_stack([times, human_x, human_y])
        
        # 背景车1轨迹（sv1_x, sv1_y）- sce2可以不绘制bg1
        if scenario_type == "sce2":
            bg1_trajectory = None
        else:
            bg1_x = trajectory_data['sv1_x'].values
            bg1_y = trajectory_data['sv1_y'].values
            bg1_trajectory = np.column_stack([times, bg1_x, bg1_y])
        
        # 背景车2轨迹（sv2_x, sv2_y）- sce1与sce2使用
        bg2_trajectory = None
        if scenario_type in ("sce1", "sce2"):
            bg2_x = trajectory_data['sv2_x'].values
            bg2_y = trajectory_data['sv2_y'].values
            bg2_trajectory = np.column_stack([times, bg2_x, bg2_y])
        
        print(f"提取轨迹数据：起始行{start_idx}，终止行{end_idx}，共{num_points}个点")
        print(f"时间范围：{times[0]:.3f}s - {times[-1]:.3f}s")
        
        return human_trajectory, bg1_trajectory, bg2_trajectory
        
    except Exception as e:
        print(f"读取CSV文件失败：{e}")
        return None, None, None

def create_vehicle_rectangle(center_x, center_y, yaw, length=4.0, width=2.0):
    """
    创建车辆矩形

    Args:
        center_x, center_y: 矩形中心坐标
        yaw: 航向角（弧度）
        length: 车长（米）
        width: 车宽（米）

    Returns:
        rectangle_corners: 矩形四个角的坐标
    """
    # 计算矩形的半长和半宽
    half_length = length / 2
    half_width = width / 2
    
    # 定义矩形的四个角（相对于中心点）
    corners_local = np.array([
        [-half_length, -half_width],  # 左下
        [half_length, -half_width],   # 右下
        [half_length, half_width],    # 右上
        [-half_length, half_width]    # 左上
    ])

    # 旋转矩阵
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])
    
    # 旋转矩形角点
    corners_rotated = np.dot(corners_local, rotation_matrix.T)

    # 平移到中心位置
    corners_global = corners_rotated + np.array([center_x, center_y])

    return corners_global

def interpolate_time_for_smooth_curve(trajectory_points, smooth_x, smooth_y, time_step=0.015):
    """
    为平滑曲线插值时间信息
    
    Args:
        trajectory_points: 原始轨迹点 (seq_len, 3) - [时间, x, y]
        smooth_x, smooth_y: 平滑曲线的x和y坐标
        time_step: 时间步长
        
    Returns:
        smooth_trajectory: 包含时间信息的平滑轨迹 (num_points, 3) - [时间, x, y]
    """
    # 原始轨迹的时间范围
    original_times = trajectory_points[:, 0]
    t_start = original_times[0]
    t_end = original_times[-1]
    
    # 为平滑曲线生成时间序列
    num_smooth_points = len(smooth_x)
    smooth_times = np.linspace(t_start, t_end, num_smooth_points)
    
    # 组合成完整的平滑轨迹
    smooth_trajectory = np.column_stack([smooth_times, smooth_x, smooth_y])
    
    return smooth_trajectory

def calculate_velocity_and_lateral_offset(smooth_trajectory, lane_center_x=155.05, scenario_type="sce3"):
    """
    计算速度曲线和横向偏移距离曲线
    
    Args:
        smooth_trajectory: 平滑轨迹数据 (num_points, 3) - [时间, x, y]
        lane_center_x: 车道中心线x坐标
        scenario_type: 场景类型，用于确定车道中心线
        
    Returns:
        times: 时间数组
        velocities: 速度数组 (m/s)
        lateral_offsets: 横向偏移距离数组 (m)
    """
    times = smooth_trajectory[:, 0]
    x_coords = smooth_trajectory[:, 1]
    y_coords = smooth_trajectory[:, 2]
    
    # 计算速度（使用相邻点之间的距离和时间差）
    velocities = np.zeros(len(times))
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        distance = np.sqrt(dx**2 + dy**2)
        velocities[i] = distance / dt if dt > 0 else 0
    
    # 第一个点的速度使用第二个点的速度
    if len(velocities) > 1:
        velocities[0] = velocities[1]
    
    # 根据场景类型确定偏移计算方式
    if scenario_type == "sce2":
        # sce2相对于直线y=-0.55
        lateral_offsets = y_coords - (-0.55)
    else:
        # 其他场景按x与车道中心线计算
        if scenario_type == "sce1":
            actual_lane_center_x = -195.05
        elif scenario_type == "sce4":
            actual_lane_center_x = 16.25
        else:
            actual_lane_center_x = lane_center_x
        lateral_offsets = x_coords - actual_lane_center_x
    
    return times, velocities, lateral_offsets

def plot_velocity_and_offset_curves(model_times, model_velocities, model_lateral_offsets, 
                                    human_times=None, human_velocities=None, human_lateral_offsets=None,
                                    output_path_prefix=""):
    """
    绘制速度曲线和横向偏移距离曲线（支持模型和人类对比）
    
    Args:
        model_times: 模型时间数组
        model_velocities: 模型速度数组
        model_lateral_offsets: 模型横向偏移距离数组
        human_times: 人类时间数组（可选）
        human_velocities: 人类速度数组（可选）
        human_lateral_offsets: 人类横向偏移距离数组（可选）
        output_path_prefix: 输出文件路径前缀
    """
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制速度曲线对比
    ax1.plot(model_times, model_velocities, 'b-', linewidth=2, alpha=0.8, label='Model')
    if human_times is not None and human_velocities is not None:
        ax1.plot(human_times, human_velocities, 'r-', linewidth=2, alpha=0.8, label='Human')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(model_times[0], model_times[-1])
    ax1.legend()
    
    # 绘制横向偏移距离曲线对比
    ax2.plot(model_times, model_lateral_offsets, 'b-', linewidth=2, alpha=0.8, label='Model')
    if human_times is not None and human_lateral_offsets is not None:
        ax2.plot(human_times, human_lateral_offsets, 'r-', linewidth=2, alpha=0.8, label='Human')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Lane Center')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Lateral Offset (m)')
    ax2.set_title('Lateral Offset')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(model_times[0], model_times[-1])
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图片
    velocity_path = f"{output_path_prefix}_velocity.png"
    offset_path = f"{output_path_prefix}_lateral_offset.png"
    
    # 分别保存速度曲线和横向偏移曲线
    fig1, ax1_single = plt.subplots(figsize=(8, 6))
    ax1_single.plot(model_times, model_velocities, 'b-', linewidth=2, alpha=0.8, label='Model')
    if human_times is not None and human_velocities is not None:
        ax1_single.plot(human_times, human_velocities, 'r-', linewidth=2, alpha=0.8, label='Human')
    ax1_single.set_xlabel('Time (s)')
    ax1_single.set_ylabel('Velocity (m/s)')
    ax1_single.set_title('Velocity')
    ax1_single.grid(True, alpha=0.3)
    ax1_single.set_xlim(model_times[0], model_times[-1])
    ax1_single.legend()
    plt.tight_layout()
    plt.savefig(velocity_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2, ax2_single = plt.subplots(figsize=(8, 6))
    ax2_single.plot(model_times, model_lateral_offsets, 'b-', linewidth=2, alpha=0.8, label='Model')
    if human_times is not None and human_lateral_offsets is not None:
        ax2_single.plot(human_times, human_lateral_offsets, 'r-', linewidth=2, alpha=0.8, label='Human')
    ax2_single.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Lane Center')
    ax2_single.set_xlabel('Time (s)')
    ax2_single.set_ylabel('Lateral Offset (m)')
    ax2_single.set_title('Lateral Offset')
    ax2_single.grid(True, alpha=0.3)
    ax2_single.set_xlim(model_times[0], model_times[-1])
    ax2_single.legend()
    plt.tight_layout()
    plt.savefig(offset_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"速度曲线已保存为：{velocity_path}")
    print(f"横向偏移曲线已保存为：{offset_path}")
    
    # 显示组合图
    # plt.show()

def create_trajectory_animation(trajectory_points, smooth_trajectory,
                              start_angle, time_step=0.01,
                              vehicle_length=4.0, vehicle_width=2.0,
                              xlim=(150, 160), ylim=(-100, 50),
                              human_trajectory=None, bg1_trajectory=None, bg2_trajectory=None, 
                              scenario_type="sce3", axis_flip='none'):
    """
    创建轨迹动画
    
    Args:
        trajectory_points: 原始轨迹点 (seq_len, 3) - [时间, x, y]
        smooth_x, smooth_y: 平滑曲线的x和y坐标
        start_angle: 起始角度（弧度）
        time_step: 时间步长
        vehicle_length: 车长
        vehicle_width: 车宽
        xlim, ylim: 坐标轴范围
        human_trajectory: 人类轨迹数据
        bg1_trajectory: 背景车1轨迹数据
        bg2_trajectory: 背景车2轨迹数据（仅sce1）
        scenario_type: 场景类型
        axis_flip: 坐标轴翻转选项
        
    Returns:
        animation: matplotlib动画对象
    """
    # 设置图形
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 绘制车道线
    if scenario_type == "sce1":
        # sce1场景：三条车道线，x坐标分别为-196.8、-193.3、-189.8，y范围[0,73.2]
        y_range = np.linspace(0, 73.2, 100)
        ax.plot([-196.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)  # 左侧实线
        ax.plot([-193.3] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)  # 中间虚线
        ax.plot([-189.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)  # 右侧实线
    elif scenario_type == "sce2":
        # sce2场景：三条车道线，y坐标分别为-5.8、-2.3、1.2，x范围[-177,-110]
        x_range = np.linspace(-177, -110, 200)
        ax.plot(x_range, [-5.8] * len(x_range), 'k-', linewidth=1.5, alpha=0.7)  # 下方实线
        ax.plot(x_range, [-2.3] * len(x_range), 'k--', linewidth=1.5, alpha=0.7) # 中间虚线
        ax.plot(x_range, [1.2] * len(x_range), 'k-', linewidth=1.5, alpha=0.7)   # 上方实线
    elif scenario_type == "sce4":
        # sce4场景：五条车道线，x坐标分别为4、7.5、11、14.5、18，y范围[-40,120]
        y_range = np.linspace(-40, 120, 100)
        ax.plot([4] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)      # 最左边实线
        ax.plot([7.5] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)    # 虚线
        ax.plot([11] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)     # 虚线
        ax.plot([14.5] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)   # 虚线
        ax.plot([18] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)      # 最右边实线
    else:
        # sce3场景：三条车道线
        y_range = np.linspace(-100, 60, 100)
        ax.plot([153.3] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
        ax.plot([156.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
        ax.plot([149.7] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
    
    # 初始化绘图元素
    line_trajectory, = ax.plot([], [], 'b-', linewidth=2, alpha=0.8, label='Model')
    # 初始化车辆矩形（使用默认位置）
    initial_corners = create_vehicle_rectangle(0, 0, 0, vehicle_length, vehicle_width)
    vehicle_rect = patches.Polygon(initial_corners.tolist(), facecolor='blue', alpha=1, edgecolor='none')
    ax.add_patch(vehicle_rect)
    
    # 初始化人类轨迹和车辆
    human_line = None
    human_rect = None
    if human_trajectory is not None:
        human_line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.8, label='Human')
        human_initial_corners = create_vehicle_rectangle(0, 0, 0, vehicle_length, vehicle_width)
        human_rect = patches.Polygon(human_initial_corners.tolist(), facecolor='red', alpha=1, edgecolor='none')
        ax.add_patch(human_rect)
    
    # 初始化背景车1轨迹和车辆
    bg1_line = None
    bg1_rect = None
    if bg1_trajectory is not None:
        if scenario_type == "sce1":
            bg1_line, = ax.plot([], [], color='darkgreen', linewidth=2, alpha=0.8, label='BV')
        else:
            bg1_line, = ax.plot([], [], color='darkgreen', linewidth=2, alpha=0.8, label='BV1')
        if scenario_type == "sce1":
            # sce1场景：背景车1为静止车辆，长4m，宽2m，长边平行于y轴
            bg1_initial_corners = create_vehicle_rectangle(0, 0, 0, 4.0, 2.0)
        elif scenario_type == "sce4":
            # sce4场景：背景车1为动态车辆，长4m，宽2m
            bg1_initial_corners = create_vehicle_rectangle(0, 0, 0, 4.0, 2.0)
        else:
            # sce3场景：背景车1为动态车辆，长2.5m，宽1.5m
            bg1_initial_corners = create_vehicle_rectangle(0, 0, 0, 2.5, 1.5)
        bg1_rect = patches.Polygon(bg1_initial_corners.tolist(), facecolor='darkgreen', alpha=1, edgecolor='none')
        ax.add_patch(bg1_rect)
    
    # 初始化背景车2轨迹和车辆（sce1与sce2场景）
    bg2_line = None
    bg2_rect = None
    if bg2_trajectory is not None and scenario_type in ("sce1", "sce2"):
        bg2_line, = ax.plot([], [], color='darkgreen', linewidth=2, alpha=0.8, label='BV')
        # sce2要求BV2为长4宽2
        if scenario_type == "sce2":
            bg2_initial_corners = create_vehicle_rectangle(0, 0, 0, 4.0, 2.0)
        else:
            bg2_initial_corners = create_vehicle_rectangle(0, 0, 0, 2.5, 1.5)
        bg2_rect = patches.Polygon(bg2_initial_corners.tolist(), facecolor='darkgreen', alpha=1, edgecolor='none')
        ax.add_patch(bg2_rect)
    
    # 时间文本
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 处理图例，避免重复
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels)
    ax.set_title('Vehicle Trajectory Animation')
    
    # 坐标轴翻转
    if axis_flip == 'x':
        ax.invert_xaxis()
    elif axis_flip == 'y':
        ax.invert_yaxis()
    elif axis_flip == 'xy':
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    def animate(frame):
        # 计算当前时间
        current_time = frame * time_step
        
        # 找到当前时间对应的轨迹点
        current_idx = np.argmin(np.abs(smooth_trajectory[:, 0] - current_time))
        current_point = smooth_trajectory[current_idx]
        
        # 更新模型轨迹线（显示已经走过的路径）
        past_mask = smooth_trajectory[:, 0] <= current_time
        if past_mask.any():
            line_trajectory.set_data(smooth_trajectory[past_mask, 1], smooth_trajectory[past_mask, 2])
        
        # 计算当前航向角（使用相邻点的方向）
        if current_idx < len(smooth_trajectory) - 1:
            dx = smooth_trajectory[current_idx + 1, 1] - smooth_trajectory[current_idx, 1]
            dy = smooth_trajectory[current_idx + 1, 2] - smooth_trajectory[current_idx, 2]
            current_yaw = np.arctan2(dy, dx)
        else:
            # 最后一个点使用前一个方向
            dx = smooth_trajectory[current_idx, 1] - smooth_trajectory[current_idx - 1, 1]
            dy = smooth_trajectory[current_idx, 2] - smooth_trajectory[current_idx - 1, 2]
            current_yaw = np.arctan2(dy, dx)
        
        # 更新模型车辆矩形
        rect_corners = create_vehicle_rectangle(
            current_point[1], current_point[2], current_yaw,
            vehicle_length, vehicle_width
        )
        vehicle_rect.set_xy(rect_corners.tolist())
        
        # 更新人类轨迹和车辆
        if human_trajectory is not None and human_line is not None and human_rect is not None:
            human_past_mask = human_trajectory[:, 0] <= current_time
            if human_past_mask.any():
                human_line.set_data(human_trajectory[human_past_mask, 1], human_trajectory[human_past_mask, 2])
            
            # 找到人类轨迹当前时间对应的点
            human_current_idx = np.argmin(np.abs(human_trajectory[:, 0] - current_time))
            if human_current_idx < len(human_trajectory):
                human_current_point = human_trajectory[human_current_idx]
                
                # 计算人类车辆航向角
                if human_current_idx < len(human_trajectory) - 1:
                    human_dx = human_trajectory[human_current_idx + 1, 1] - human_trajectory[human_current_idx, 1]
                    human_dy = human_trajectory[human_current_idx + 1, 2] - human_trajectory[human_current_idx, 2]
                    human_yaw = np.arctan2(human_dy, human_dx)
                else:
                    human_dx = human_trajectory[human_current_idx, 1] - human_trajectory[human_current_idx - 1, 1]
                    human_dy = human_trajectory[human_current_idx, 2] - human_trajectory[human_current_idx - 1, 2]
                    human_yaw = np.arctan2(human_dy, human_dx)
                
                # 更新人类车辆矩形
                human_rect_corners = create_vehicle_rectangle(
                    human_current_point[1], human_current_point[2], human_yaw,
                    vehicle_length, vehicle_width
                )
                human_rect.set_xy(human_rect_corners.tolist())
        
        # 更新背景车1轨迹和车辆
        if bg1_trajectory is not None and bg1_line is not None and bg1_rect is not None:
            bg1_past_mask = bg1_trajectory[:, 0] <= current_time
            if bg1_past_mask.any():
                bg1_line.set_data(bg1_trajectory[bg1_past_mask, 1], bg1_trajectory[bg1_past_mask, 2])
            
            # 找到背景车1轨迹当前时间对应的点
            bg1_current_idx = np.argmin(np.abs(bg1_trajectory[:, 0] - current_time))
            if bg1_current_idx < len(bg1_trajectory):
                bg1_current_point = bg1_trajectory[bg1_current_idx]
                
                if scenario_type == "sce1":
                    # sce1场景：背景车1为静止车辆，长边平行于y轴
                    bg1_yaw = 90 * math.pi / 180  # 旋转90度，让长边平行于y轴
                    bg1_length, bg1_width = 4.0, 2.0
                elif scenario_type == "sce4":
                    # sce4场景：背景车1为动态车辆，长4m，宽2m
                    # 计算背景车1航向角
                    if bg1_current_idx < len(bg1_trajectory) - 1:
                        bg1_dx = bg1_trajectory[bg1_current_idx + 1, 1] - bg1_trajectory[bg1_current_idx, 1]
                        bg1_dy = bg1_trajectory[bg1_current_idx + 1, 2] - bg1_trajectory[bg1_current_idx, 2]
                        bg1_yaw = np.arctan2(bg1_dy, bg1_dx)
                    else:
                        bg1_dx = bg1_trajectory[bg1_current_idx, 1] - bg1_trajectory[bg1_current_idx - 1, 1]
                        bg1_dy = bg1_trajectory[bg1_current_idx, 2] - bg1_trajectory[bg1_current_idx - 1, 2]
                        bg1_yaw = np.arctan2(bg1_dy, bg1_dx)
                    bg1_length, bg1_width = 4.0, 2.0
                else:
                    # sce3场景：背景车1为动态车辆
                    # 计算背景车1航向角
                    if bg1_current_idx < len(bg1_trajectory) - 1:
                        bg1_dx = bg1_trajectory[bg1_current_idx + 1, 1] - bg1_trajectory[bg1_current_idx, 1]
                        bg1_dy = bg1_trajectory[bg1_current_idx + 1, 2] - bg1_trajectory[bg1_current_idx, 2]
                        # 检查背景车1是否静止（前后两帧坐标相同）
                        if abs(bg1_dx) < 1e-6 and abs(bg1_dy) < 1e-6:
                            bg1_yaw = -90 * math.pi / 180  # 静止时车头朝向-90°
                        else:
                            bg1_yaw = np.arctan2(bg1_dy, bg1_dx)
                    else:
                        bg1_dx = bg1_trajectory[bg1_current_idx, 1] - bg1_trajectory[bg1_current_idx - 1, 1]
                        bg1_dy = bg1_trajectory[bg1_current_idx, 2] - bg1_trajectory[bg1_current_idx - 1, 2]
                        # 检查背景车1是否静止（前后两帧坐标相同）
                        if abs(bg1_dx) < 1e-6 and abs(bg1_dy) < 1e-6:
                            bg1_yaw = -90 * math.pi / 180  # 静止时车头朝向-90°
                        else:
                            bg1_yaw = np.arctan2(bg1_dy, bg1_dx)
                    bg1_length, bg1_width = 2.5, 1.5
                
                # 更新背景车1车辆矩形
                bg1_rect_corners = create_vehicle_rectangle(
                    bg1_current_point[1], bg1_current_point[2], bg1_yaw,
                    bg1_length, bg1_width
                )
                bg1_rect.set_xy(bg1_rect_corners.tolist())
        
        # 更新背景车2轨迹和车辆（sce1与sce2场景）
        if bg2_trajectory is not None and bg2_line is not None and bg2_rect is not None and scenario_type in ("sce1", "sce2"):
            bg2_past_mask = bg2_trajectory[:, 0] <= current_time
            if bg2_past_mask.any():
                bg2_line.set_data(bg2_trajectory[bg2_past_mask, 1], bg2_trajectory[bg2_past_mask, 2])
            
            # 找到背景车2轨迹当前时间对应的点
            bg2_current_idx = np.argmin(np.abs(bg2_trajectory[:, 0] - current_time))
            if bg2_current_idx < len(bg2_trajectory):
                bg2_current_point = bg2_trajectory[bg2_current_idx]
                
                # 计算背景车2航向角
                if bg2_current_idx < len(bg2_trajectory) - 1:
                    bg2_dx = bg2_trajectory[bg2_current_idx + 1, 1] - bg2_trajectory[bg2_current_idx, 1]
                    bg2_dy = bg2_trajectory[bg2_current_idx + 1, 2] - bg2_trajectory[bg2_current_idx, 2]
                else:
                    bg2_dx = bg2_trajectory[bg2_current_idx, 1] - bg2_trajectory[bg2_current_idx - 1, 1]
                    bg2_dy = bg2_trajectory[bg2_current_idx, 2] - bg2_trajectory[bg2_current_idx - 1, 2]

                if scenario_type == "sce2" and abs(bg2_dx) < 1e-6 and abs(bg2_dy) < 1e-6:
                    bg2_yaw = -45 * math.pi / 180  # sce2静止时固定角度
                else:
                    bg2_yaw = np.arctan2(bg2_dy, bg2_dx)
                
                # 更新背景车2车辆矩形
                bg2_rect_corners = create_vehicle_rectangle(
                    bg2_current_point[1], bg2_current_point[2], bg2_yaw,
                    (4.0 if scenario_type == "sce2" else 2.5), (2.0 if scenario_type == "sce2" else 1.5)
                )
                bg2_rect.set_xy(bg2_rect_corners.tolist())
                animate._bg2_last_yaw = bg2_yaw
        
        # 更新时间文本
        time_text.set_text(f'Time: {current_time:.2f}s')
        
        # 返回所有需要更新的元素
        elements_to_return = [line_trajectory, vehicle_rect, time_text]
        if human_line is not None:
            elements_to_return.append(human_line)
        if human_rect is not None:
            elements_to_return.append(human_rect)
        if bg1_line is not None:
            elements_to_return.append(bg1_line)
        if bg1_rect is not None:
            elements_to_return.append(bg1_rect)
        if bg2_line is not None:
            elements_to_return.append(bg2_line)
        if bg2_rect is not None:
            elements_to_return.append(bg2_rect)
        
        return elements_to_return
    
    # 计算动画帧数
    total_time = trajectory_points[-1, 0] - trajectory_points[0, 0]
    num_frames = int(total_time / time_step) + 1
    
    # 创建动画
    animation = FuncAnimation(fig, animate, frames=num_frames,
                            interval=time_step*1000, blit=True, repeat=True)
    
    return animation, fig

def save_animation_as_gif(animation, fig, output_path, fps=100):
    """
    将动画保存为GIF文件

    Args:
        animation: matplotlib动画对象
        fig: matplotlib图形对象
        output_path: 输出文件路径
        fps: 帧率
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存GIF
        animation.save(output_path, writer='pillow', fps=fps)
        print(f"动画已保存为：{output_path}")

    except Exception as e:
        print(f"保存GIF失败：{e}")

if __name__ == "__main__":
    # ====== 可修改参数 ======
    # 模型参数
    model_path = 'training/models/vae_sce3_ld8_epoch1000.pth'  # 模型文件路径
    seq_len = 12                  # 轨迹长度
    dim = 3                       # 每个点的维度
    latent_dim = 8                # 潜在空间维度
    device = 'cpu'                # 计算设备

    # CSV文件参数
    csv_path = 'DefensiveData/PredictableMovementTown05/减速/exp_8_control_PredictableMovementTown05_2.csv'  # CSV文件路径
    time_interval = 0.015  # 相邻点之间时间间隔
    # sce1:减速11_3,转向1_2;sce2:减速1_3,减速+转向12_2; sce3:; sce4:减速16_3,减速17_2

    # 动画参数
    time_step = 0.01              # 时间步长（秒）
    fps = 66                     # GIF帧率（帧/秒）
    vehicle_length = 4.0          # 车长（米）
    vehicle_width = 2.0           # 车宽（米）

    # 坐标轴范围
    if "sce1" in model_path and "StaticBlind" in csv_path:
        xlim = (-225.05, -165.05)
        ylim = (40, 100)
    elif "sce2" in model_path and "DynamicBlind" in csv_path:
        xlim = (-210, -100)
        ylim = (-47.5, 42.5)
    elif "sce3" in model_path and "PredictableMovement" in csv_path:
        xlim = (80, 230)
        ylim = (-100, 50)
    elif "sce4" in model_path and "UnpredictableMovement" in csv_path:
        xlim = (-38, 62)
        ylim = (-10, 90)
    else:
        print("The model and the scenarios do not match with each other.")
    
    # 坐标轴翻转参数
    if "sce1" in model_path or "sce2" in model_path:
        axis_flip = 'x'  # 翻转x轴
    else:
        axis_flip = 'y'
    
    # 输出参数
    output_filename = 'sce3_trajectory_animation_both_3.gif'  # 输出文件名
    output_path = f'training/gif/{output_filename}'  # 输出路径
    
    # =====================
    
    print("开始生成车辆轨迹动图...")

    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    model_name = os.path.basename(model_path)
    
    # 获取起始条件
    human_trajectory = None
    bg1_trajectory = None
    bg2_trajectory = None
    
    if "sce1" in model_name:
        print("检测到sce1模型，使用CSV文件获取起始条件")
        start_x, start_y, start_angle = get_start_conditions_from_csv(csv_path, scenario_type="sce1")
        
        # 获取人类轨迹和背景车轨迹
        print("获取人类轨迹和背景车轨迹...")
        human_trajectory, bg1_trajectory, bg2_trajectory = get_human_and_bg_trajectories_from_csv(csv_path, scenario_type="sce1")
        
        if human_trajectory is not None and bg1_trajectory is not None:
            print("成功获取人类轨迹和背景车轨迹")
        else:
            print("警告：无法获取人类轨迹或背景车轨迹")
    elif "sce2" in model_name:
        print("检测到sce2模型，使用CSV文件获取起始条件")
        start_x, start_y, start_angle = get_start_conditions_from_csv(csv_path, scenario_type="sce2")
        
        # 获取人类轨迹和背景车2轨迹
        print("获取人类轨迹和背景车2轨迹...")
        human_trajectory, bg1_trajectory, bg2_trajectory = get_human_and_bg_trajectories_from_csv(csv_path, scenario_type="sce2")
        
        if human_trajectory is not None and bg2_trajectory is not None:
            print("成功获取人类轨迹和背景车2轨迹")
        else:
            print("警告：无法获取人类轨迹或背景车2轨迹")
    elif "sce4" in model_name:
        print("检测到sce4模型，使用CSV文件获取起始条件")
        start_x, start_y, start_angle = get_start_conditions_from_csv(csv_path, scenario_type="sce4")
        
        # 获取人类轨迹和背景车1轨迹
        print("获取人类轨迹和背景车1轨迹...")
        human_trajectory, bg1_trajectory, _ = get_human_and_bg_trajectories_from_csv(csv_path, scenario_type="sce4")
        
        if human_trajectory is not None and bg1_trajectory is not None:
            print("成功获取人类轨迹和背景车1轨迹")
        else:
            print("警告：无法获取人类轨迹或背景车1轨迹")
    elif "sce3" in model_name:
        print("检测到sce3模型，使用CSV文件获取起始条件")
        start_x, start_y, start_angle = get_start_conditions_from_csv(csv_path, scenario_type="sce3")
        
        # 获取人类轨迹和背景车1轨迹
        print("获取人类轨迹和背景车1轨迹...")
        human_trajectory, bg1_trajectory, _ = get_human_and_bg_trajectories_from_csv(csv_path, scenario_type="sce3")
        
        if human_trajectory is not None and bg1_trajectory is not None:
            print("成功获取人类轨迹和背景车1轨迹")
        else:
            print("警告：无法获取人类轨迹或背景车1轨迹")
    else:
        print("使用默认起始条件")
        start_x, start_y = 155.0, -15.0
        start_angle = -90 * math.pi / 180
    
    # 加载模型并生成轨迹
    print("加载模型并生成轨迹...")
    trajectory_points = load_model_and_generate_trajectory(
        model_path, start_x, start_y, seq_len, dim, latent_dim, device
    )
    
    # 使用样条曲线拟合轨迹点
    print("使用样条曲线拟合轨迹...")
    smooth_trajectory = create_smooth_curve(trajectory_points, time_interval=time_interval, start_angle=start_angle)

    if "sce1" in model_name:
        scenario_type = "sce1"
    elif "sce2" in model_name:
        scenario_type = "sce2"
    elif "sce4" in model_name:
        scenario_type = "sce4"
    else:
        scenario_type = "sce3"
    # 计算速度曲线和横向偏移距离曲线
    print("计算速度曲线和横向偏移距离曲线...")
    # smooth_trajectory = interpolate_time_for_smooth_curve(trajectory_points, smooth_x, smooth_y, time_step=time_interval)
    model_times, model_velocities, model_lateral_offsets = calculate_velocity_and_lateral_offset(smooth_trajectory, lane_center_x=155.05, scenario_type=scenario_type)
    
    # 计算人类轨迹的速度和横向偏移（如果存在）
    human_times = None
    human_velocities = None
    human_lateral_offsets = None
    if human_trajectory is not None:
        print("计算人类轨迹的速度和横向偏移...")
        human_times, human_velocities, human_lateral_offsets = calculate_velocity_and_lateral_offset(human_trajectory, lane_center_x=155.05, scenario_type=scenario_type)

    original_times, original_velocities, original_lateral_offsets = calculate_velocity_and_lateral_offset(trajectory_points,
                                                                                                          lane_center_x=155.05,
                                                                                                          scenario_type=scenario_type)

    # 绘制速度曲线和横向偏移距离曲线
    print("绘制速度曲线和横向偏移距离曲线...")
    output_path_prefix = output_path.replace('.gif', '')
    plot_velocity_and_offset_curves(model_times, model_velocities, model_lateral_offsets,
                                    human_times, human_velocities, human_lateral_offsets,
                                    output_path_prefix)

    plot_velocity_and_offset_curves(original_times, original_velocities, original_lateral_offsets,
                                    human_times, human_velocities, human_lateral_offsets,
                                    output_path_prefix)
    
    # 创建动画
    print("创建动画...")
    animation, fig = create_trajectory_animation(
        trajectory_points, smooth_trajectory, start_angle,
        time_interval, vehicle_length, vehicle_width, xlim, ylim,  # 使用0.015s时间步长，与背景车1轨迹一致
        human_trajectory, bg1_trajectory, bg2_trajectory, scenario_type, axis_flip
    )
    
    # 保存GIF
    print("保存GIF文件...")
    save_animation_as_gif(animation, fig, output_path, fps)
    
    print("车辆轨迹动图生成完成！")
    
    # 显示动画（可选）
    # plt.show()
