import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import CubicSpline, make_interp_spline
from Training_VAE import ConditionalTrajectoryVAE


# 调用模型生成轨迹
def load_model_and_generate_trajectory(model_path, start_x, start_y, seq_len=12, dim=3, latent_dim=8, device='cpu'):
    """
    加载模型并生成轨迹

    Args:
        model_path: 模型文件路径
        start_x: 起始点x坐标
        start_y: 起始点y坐标
        seq_len: 轨迹长度
        dim: 每个点的维度
        latent_dim: 潜在空间维度
        device: 计算设备

    Returns:
        generated_trajectory: 生成的轨迹数据 (seq_len, 3) - [时间, x, y]
        说明：
        - 条件VAE内部实际生成的是“相对起点偏移轨迹” [t, dx, dy]
        - 本函数在解码后会自动执行 x = start_x + dx, y = start_y + dy，
          返回的是加上起点后的全局轨迹，方便后续直接使用
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

        # 生成相对轨迹：[t, dx, dy]
        rel_traj = model.decode(z, h_condition).cpu().numpy()[0]  # (seq_len, 3)

        # 将相对轨迹转换为全局轨迹：[t, x, y]
        generated_trajectory = rel_traj.copy()
        generated_trajectory[:, 1] = start_x + rel_traj[:, 1]
        generated_trajectory[:, 2] = start_y + rel_traj[:, 2]

    return generated_trajectory


# 从csv文件中获取起始条件
def get_start_conditions_from_csv(csv_path, model_name):
    """
    从csv文件中获取起始条件

    Args:
        csv_path: csv文件路径
        model_name: 模型名称

    Returns:
        start_x, start_y, start_angle: 起始x坐标、y坐标、角度（弧度）
    """
    try:
        # 读取csv文件
        df = pd.read_csv(csv_path)

        if "sce1" in model_name:
            mask = (df['ego_y'] >= 18) & (df['sv2_vx'] != 0) & (df['sv2_vy'] != 0)
        elif "sce2" in model_name:
            mask = (df['sv1_yaw'] < -170)
        elif "sce4" in model_name:
            mask = (((df['ego_x'] - df['sv1_x']) ** 2 + (df['ego_y'] - df['sv1_y']) ** 2 <= 40 ** 2)
                    & (df['sv1_yaw'] >= -89.9))
        else:
            mask = (
                    (df['sv1_vx'] != 0) &
                    (df['sv1_vy'] != 0) &
                    (df['ego_y'] <= 40) &
                    (df['ego_y'] != 0)
            )

        if not mask.any():
            print(f"警告：未找到满足条件的起始行，使用默认值")
            if "sce1" in model_name:
                return -193.3, 50.0, -90 * math.pi / 180
            elif "sce2" in model_name:
                return -155.0, -5.0, -90 * math.pi / 180
            elif "sce4" in model_name:
                return 11.0, 0.0, -90 * math.pi / 180
            else:
                return 155.0, -15.0, -90 * math.pi / 180

        # 获取第一行满足条件的数据
        start_row = df[mask].iloc[0]

        start_x = start_row['ego_x']
        start_y = start_row['ego_y']
        start_angle = start_row['ego_yaw'] * math.pi / 180
        start_vx = start_row['ego_vx']
        start_vy = start_row['ego_vy']

        print(f"从CSV获取起始条件：x={start_x:.2f}, y={start_y:.2f}, angle={start_angle:.2f}rad,"
              f"vx={start_vx:.2f}, vy={start_vy:.2f}")

        return start_x, start_y, start_angle, start_vx, start_vy

    except Exception as e:
        print(f"读取CSV文件失败：{e}")
        print("使用默认起始条件")
        if "sce1" in model_name:
            return -193.3, 50.0, -90 * math.pi / 180
        elif "sce2" in model_name:
            return -155.0, -5.0, -90 * math.pi / 180
        elif "sce4" in model_name:
            return 11.0, 0.0, -90 * math.pi / 180
        else:
            return 155.0, -15.0, -90 * math.pi / 180


# 从csv中提取人类轨迹和背景车轨迹
def get_human_and_bv_trajectories(csv_path, model_name):
    """
    从csv文件中提取人类轨迹和背景车轨迹

    Args:
        csv_path: csv文件路径
        model_name: 模型名称

    Returns:
        human_traj: 人类轨迹(x,y,t)
    """
    try:
        # 读取csv文件
        df = pd.read_csv(csv_path)
        # 定义起点掩码
        if "sce1" in model_name:
            start_mask = (df['ego_y'] >= 18) & (df['sv2_vx'] != 0) & (df['sv2_vy'] != 0)
            time_step = 0.02
        elif "sce2" in model_name:
            start_mask = (df['sv1_yaw'] < -170)
            time_step = 0.025
        elif "sce4" in model_name:
            start_mask = ((df['ego_x'] - df['sv1_x']) ** 2 + (df['ego_y'] - df['sv1_y']) ** 2 <= 50 ** 2)
            time_step = 0.02
        else:
            start_mask = (df['sv1_vx'] != 0) & (df['sv1_vy'] != 0) & (df['ego_y'] <= 40) & (df['ego_y'] != 0)
            time_step = 0.015
        if not start_mask.any():
            print("警告：未找到满足条件的起始行")
            return None, None, None
        start_idx = df[start_mask].index[0]  # 第一个满足start_mask的行索引
        df_copy = df.iloc[start_idx:]
        # 定义终点掩码
        if "sce1" in model_name:
            end_mask = (df_copy['ego_y'] >= 95)
        elif "sce2" in model_name:
            end_mask = (df_copy['ego_x'] < -186)
        elif "sce4" in model_name:
            end_mask = (df_copy['sv1_x'] > 15) & (df_copy['sv1_yaw'] < -85)
        else:
            end_mask = (df_copy['ego_y'] <= -80)
        if not end_mask.any():
            print("警告：未找到满足条件的终止行，使用文件末尾行")
            end_idx = len(df) - 1
        else:
            end_idx = df_copy[end_mask].index[0]  # 第一个满足end_mask的行索引
        if start_idx >= end_idx:
            print("警告：终止行在起始行之前或相同")
            return None, None, None

        # 人类轨迹
        human_coord = df.loc[start_idx:end_idx, ['ego_x', 'ego_y']].to_numpy()
        time_column = np.arange(len(human_coord)) * time_step
        human_traj = np.column_stack((human_coord, time_column))
        # 背景车轨迹
        if "sce1" in model_name:
            bv1_coord = df.loc[start_idx:end_idx, ['sv1_x', 'sv1_y']].to_numpy()
            bv2_coord = df.loc[start_idx:end_idx, ['sv2_x', 'sv2_y']].to_numpy()
            bv1_traj = np.column_stack((bv1_coord, time_column))
            bv2_traj = np.column_stack((bv2_coord, time_column))
        elif "sce2" in model_name:
            bv1_coord = df.loc[start_idx:end_idx, ['sv1_x', 'sv1_y']].to_numpy()
            bv2_coord = df.loc[start_idx:end_idx, ['sv2_x', 'sv2_y']].to_numpy()
            bv1_traj = np.column_stack((bv1_coord, time_column))
            bv2_traj = np.column_stack((bv2_coord, time_column))
        elif "sce4" in model_name:
            bv1_coord = df.loc[start_idx:end_idx, ['sv1_x', 'sv1_y']].to_numpy()
            bv2_coord = None
            bv1_traj = np.column_stack((bv1_coord, time_column))
            bv2_traj = None
        else:
            bv1_coord = df.loc[start_idx:end_idx, ['sv1_x', 'sv1_y']].to_numpy()
            bv2_coord = df.loc[start_idx:end_idx, ['sv2_x', 'sv2_y']].to_numpy()
            bv1_traj = np.column_stack((bv1_coord, time_column))
            bv2_traj = np.column_stack((bv2_coord, time_column))

        return human_traj, bv1_traj, bv2_traj
    except Exception as e:
        print(f"读取CSV文件失败：{e}")


def process_model_trajectory(human_traj, start_x, start_y, model_states, time_step):
    """
        获取模型完整轨迹

        Args:
            human_traj: 人类轨迹
            start_x: 模型起点x坐标
            start_y: 模型起点y坐标
            model_states: 模型坐标等状态信息
            time_step: 时间步长

        Returns:
            model_trajectory: 含时间信息的模型轨迹
        """
    model_coord = model_states[:, :2]
    start_mask = (human_traj[:, 0] == start_x) & (human_traj[:, 1] == start_y)
    indices = np.where(start_mask)[0]
    if len(indices) == 0:
        print("警告：没有找到匹配的行")
        return None
    start_index = indices[0]
    model_effective_time = human_traj[start_index:, 2]
    coord_rows = model_coord.shape[0]
    time_rows = model_effective_time.shape[0]

    if time_rows > coord_rows:
        # model_effective_time行数多于model_coord，截断多余行
        model_effective_time = model_effective_time[:coord_rows]
    elif time_rows < coord_rows:
        # model_effective_time行数少于model_coord，补充缺失行
        last_time = model_effective_time[-1] if time_rows > 0 else 0
        additional_rows = coord_rows - time_rows
        additional_times = np.array([last_time + time_step * (i + 1) for i in range(additional_rows)])
        model_effective_time = np.concatenate([model_effective_time, additional_times])

    model_trajectory = np.column_stack((model_coord, model_effective_time))

    return model_trajectory



def create_vehicle_rectangle(center_x, center_y, yaw, length=4.0, width=2.0):
    """
    创建车辆矩形

    Args:
        center_x: 矩形中心x坐标
        center_y: 矩形中心y坐标
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
        [half_length, -half_width],  # 右下
        [half_length, half_width],  # 右上
        [-half_length, half_width]  # 左上
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


# 绘制人类轨迹和模型轨迹对比gif
def plot_gif_human_vs_model(human_traj, bv1_traj, bv2_traj, model_traj, model_name):
    """
    绘制gif，人类原始轨迹+模型生成轨迹，潜在风险点前共用轨迹，潜在风险点后分离为两条对比轨迹

    Args:
        human_traj: 人类轨迹
        bv1_traj: 背景车1轨迹
        bv2_traj: 背景车2轨迹
        model_traj: 模型生成轨迹
        model_name: 模型名称

    Returns:
        human_traj: 人类轨迹(x,y,t)
    """
    if "sce1" in model_name:
        # 坐标轴范围
        xlim = (-230, -150)
        ylim = (20, 100)
        time_step = 0.02
    elif "sce2" in model_name:
        xlim = (-200, -100)
        ylim = (-53, 47)
        time_step = 0.025
    elif "sce4" in model_name:
        xlim = (-45, 65)
        ylim = (-10, 100)
        time_step = 0.02
    else:
        xlim = (80, 230)
        ylim = (-100, 50)
        time_step = 0.015
    # 绘图图窗设置
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    # 绘制车道线
    if "sce1" in model_name:
        # sce1场景：三条车道线，x坐标分别为-196.8、-193.3、-189.8，y范围[0,73.2]
        y_range = np.linspace(0, 73.2, 100)
        ax.plot([-196.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)  # 左侧实线
        ax.plot([-193.3] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)  # 中间虚线
        ax.plot([-189.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)  # 右侧实线
    elif "sce2" in model_name:
        # sce2场景：三条车道线，y坐标分别为-5.8、-2.3、1.2，x范围[-177,-50]
        x_range = np.linspace(-177, -50, 200)
        ax.plot(x_range, [-5.8] * len(x_range), 'k-', linewidth=1.5, alpha=0.7)  # 下方实线
        ax.plot(x_range, [-2.3] * len(x_range), 'k--', linewidth=1.5, alpha=0.7)  # 中间虚线
        ax.plot(x_range, [1.2] * len(x_range), 'k-', linewidth=1.5, alpha=0.7)  # 上方实线
    elif "sce4" in model_name:
        # sce4场景：五条车道线，x坐标分别为3.5、7、10.5、14、17.5，y范围[-40,120]
        y_range = np.linspace(-40, 120, 100)
        ax.plot([3.5] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)  # 最左边实线
        ax.plot([7] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)  # 虚线
        ax.plot([10.5] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)  # 虚线
        ax.plot([14] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)  # 虚线
        ax.plot([17.5] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)  # 最右边实线
    else:
        # sce3场景：三条车道线
        y_range = np.linspace(-100, 60, 100)
        ax.plot([153.3] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
        ax.plot([156.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
        ax.plot([149.7] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)

    model_color = (0, 0.4470, 0.7410)  # 蓝
    human_color = (0.7961, 0.1255, 0.1765)  # 红
    bv_color = (0.4660, 0.6740, 0.1880)  # 绿

    # 初始化绘图元素
    line_trajectory, = ax.plot([], [], color=model_color, linestyle='-', linewidth=2, alpha=1, label='Model')
    # 初始化车辆矩形
    vehicle_length = 4.0
    vehicle_width = 2.0
    initial_corners = create_vehicle_rectangle(1000, 1000, 0, vehicle_length, vehicle_width)
    vehicle_rect = patches.Polygon(initial_corners.tolist(), facecolor=model_color, alpha=1, edgecolor='none')
    ax.add_patch(vehicle_rect)

    # 初始化人类轨迹和车辆
    human_line = None
    human_rect = None
    if human_traj is not None:
        human_line, = ax.plot([], [], color=human_color, linestyle='-', linewidth=2, alpha=1, label='Human')
        human_initial_corners = create_vehicle_rectangle(1000, 1000, 0, vehicle_length, vehicle_width)
        human_rect = patches.Polygon(human_initial_corners.tolist(), facecolor=human_color, alpha=1, edgecolor='none')
        ax.add_patch(human_rect)

    # 初始化背景车1轨迹和车辆
    bv1_line = None
    bv1_rect = None
    if bv1_traj is not None:
        bv1_line, = ax.plot([], [], color=bv_color, linewidth=2, alpha=1, label='BV')
        if "sce3" in model_name:
            # sce3场景：背景车1为自行车，长2.5m，宽1.5m
            bv1_initial_corners = create_vehicle_rectangle(1000, 1000, 0, 2.5, 1.5)
        else:
            # 其他场景：背景车1为车辆，长4m，宽2m
            bv1_initial_corners = create_vehicle_rectangle(1000, 1000, 0, 4.0, 2.0)
        bv1_rect = patches.Polygon(bv1_initial_corners.tolist(), facecolor=bv_color, alpha=1, edgecolor='none')
        ax.add_patch(bv1_rect)

    # 初始化背景车2轨迹和车辆
    bv2_line = None
    bv2_rect = None
    if bv2_traj is not None:
        bv2_line, = ax.plot([], [], color=bv_color, linewidth=2, alpha=1, label='BV')
        if "sce1" in model_name:
            # sce1场景：背景车2为自行车，长2.5m，宽1.5m
            bv2_initial_corners = create_vehicle_rectangle(1000, 1000, 0, 2.5, 1.5)
        else:
            # 其他场景：背景车2长4m，宽2m
            bv2_initial_corners = create_vehicle_rectangle(1000, 1000, 0, 4.0, 2.0)
        bv2_rect = patches.Polygon(bv2_initial_corners.tolist(), facecolor=bv_color, alpha=1, edgecolor='none')
        ax.add_patch(bv2_rect)

    # 时间文本
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=16,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    # 处理图例，避免重复
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels)
    ax.set_title('Human Trajectory VS Model Trajectory')

    # 坐标轴翻转
    if "sce1" in model_name or "sce2" in model_name:
        ax.invert_xaxis()
    elif "sce3" in model_name or "sce4" in model_name:
        ax.invert_yaxis()

    def animate(frame):
        # 计算当前时间
        current_time = frame * time_step

        # 找到当前时间对应的轨迹点
        current_idx = np.argmin(np.abs(model_traj[:, 2] - current_time))
        current_point = model_traj[current_idx]

        # 更新模型轨迹线（显示已经走过的路径）
        past_mask = model_traj[:, 2] <= current_time
        if past_mask.any():
            line_trajectory.set_data(model_traj[past_mask, 0], model_traj[past_mask, 1])

        # 计算当前航向角（使用相邻点的方向）
        if current_idx < len(model_traj) - 1:
            dx = model_traj[current_idx + 1, 0] - model_traj[current_idx, 0]
            dy = model_traj[current_idx + 1, 1] - model_traj[current_idx, 1]
            current_yaw = np.arctan2(dy, dx)
        else:
            # 最后一个点使用前一个方向
            dx = model_traj[current_idx, 0] - model_traj[current_idx - 1, 0]
            dy = model_traj[current_idx, 1] - model_traj[current_idx - 1, 1]
            current_yaw = np.arctan2(dy, dx)

        # 更新模型车辆矩形
        if past_mask.any():
            rect_corners = create_vehicle_rectangle(
                current_point[0], current_point[1], current_yaw,
                vehicle_length, vehicle_width
            )
        else:
            rect_corners = create_vehicle_rectangle(
                1000, 1000, 0,
                vehicle_length, vehicle_width
            )
        vehicle_rect.set_xy(rect_corners.tolist())

        # 更新人类轨迹和车辆
        if human_traj is not None and human_line is not None and human_rect is not None:
            human_past_mask = human_traj[:, 2] <= current_time
            if human_past_mask.any():
                human_line.set_data(human_traj[human_past_mask, 0], human_traj[human_past_mask, 1])

            # 找到人类轨迹当前时间对应的点
            human_current_idx = np.argmin(np.abs(human_traj[:, 2] - current_time))
            if human_current_idx < len(human_traj):
                human_current_point = human_traj[human_current_idx]

                # 计算人类车辆航向角
                if human_current_idx < len(human_traj) - 1:
                    human_dx = human_traj[human_current_idx + 1, 0] - human_traj[human_current_idx, 0]
                    human_dy = human_traj[human_current_idx + 1, 1] - human_traj[human_current_idx, 1]
                    if abs(human_dx) < 1e-3 and abs(human_dy) < 1e-3:
                        # 静止时使用上一帧的角度
                        human_dx = human_traj[human_current_idx, 0] - human_traj[human_current_idx - 1, 0]
                        human_dy = human_traj[human_current_idx, 1] - human_traj[human_current_idx - 1, 1]
                        human_yaw = np.arctan2(human_dy, human_dx)
                    else:
                        human_yaw = np.arctan2(human_dy, human_dx)
                else:
                    human_dx = human_traj[human_current_idx, 0] - human_traj[human_current_idx - 1, 0]
                    human_dy = human_traj[human_current_idx, 1] - human_traj[human_current_idx - 1, 1]
                    human_yaw = np.arctan2(human_dy, human_dx)

                # 更新人类车辆矩形
                human_rect_corners = create_vehicle_rectangle(
                    human_current_point[0], human_current_point[1], human_yaw,
                    vehicle_length, vehicle_width
                )
                human_rect.set_xy(human_rect_corners.tolist())

        # 更新背景车1轨迹和车辆
        if bv1_traj is not None and bv1_line is not None and bv1_rect is not None:
            bv1_past_mask = bv1_traj[:, 2] <= current_time
            if bv1_past_mask.any():
                bv1_line.set_data(bv1_traj[bv1_past_mask, 0], bv1_traj[bv1_past_mask, 1])

            # 找到背景车1轨迹当前时间对应的点
            bv1_current_idx = np.argmin(np.abs(bv1_traj[:, 2] - current_time))
            if bv1_current_idx < len(bv1_traj):
                bv1_current_point = bv1_traj[bv1_current_idx]

                if "sce1" in model_name:
                    # sce1场景：背景车1为静止车辆，长边平行于y轴
                    bv1_yaw = 90 * math.pi / 180  # 旋转90度，让长边平行于y轴
                    bv1_length, bv1_width = 4.0, 2.0
                elif "sce2" in model_name or "sce4" in model_name:
                    # sce2和sce4场景：背景车1为动态车辆，长4m，宽2m
                    # 计算背景车1航向角
                    if bv1_current_idx < len(bv1_traj) - 1:
                        bv1_dx = bv1_traj[bv1_current_idx + 1, 0] - bv1_traj[bv1_current_idx, 0]
                        bv1_dy = bv1_traj[bv1_current_idx + 1, 1] - bv1_traj[bv1_current_idx, 1]
                        bv1_yaw = np.arctan2(bv1_dy, bv1_dx)
                    else:
                        bv1_dx = bv1_traj[bv1_current_idx, 0] - bv1_traj[bv1_current_idx - 1, 0]
                        bv1_dy = bv1_traj[bv1_current_idx, 1] - bv1_traj[bv1_current_idx - 1, 1]
                        bv1_yaw = np.arctan2(bv1_dy, bv1_dx)
                    bv1_length, bv1_width = 4.0, 2.0
                else:
                    # sce3场景：背景车1为动态车辆
                    # 计算背景车1航向角
                    if bv1_current_idx < len(bv1_traj) - 1:
                        bv1_dx = bv1_traj[bv1_current_idx + 1, 0] - bv1_traj[bv1_current_idx, 0]
                        bv1_dy = bv1_traj[bv1_current_idx + 1, 1] - bv1_traj[bv1_current_idx, 1]
                        # 检查背景车1是否静止（前后两帧坐标相同）
                        if abs(bv1_dx) < 1e-4 and abs(bv1_dy) < 1e-4:
                            bv1_yaw = -90 * math.pi / 180  # 静止时车头朝向-90°
                        else:
                            bv1_yaw = np.arctan2(bv1_dy, bv1_dx)
                    else:
                        bv1_dx = bv1_traj[bv1_current_idx, 0] - bv1_traj[bv1_current_idx - 1, 0]
                        bv1_dy = bv1_traj[bv1_current_idx, 1] - bv1_traj[bv1_current_idx - 1, 1]
                        # 检查背景车1是否静止（前后两帧坐标相同）
                        if abs(bv1_dx) < 1e-4 and abs(bv1_dy) < 1e-4:
                            bv1_yaw = -90 * math.pi / 180  # 静止时车头朝向-90°
                        else:
                            bv1_yaw = np.arctan2(bv1_dy, bv1_dx)
                    bv1_length, bv1_width = 2.5, 1.5

                # 更新背景车1车辆矩形
                bv1_rect_corners = create_vehicle_rectangle(
                    bv1_current_point[0], bv1_current_point[1], bv1_yaw,
                    bv1_length, bv1_width
                )
                bv1_rect.set_xy(bv1_rect_corners.tolist())

        # 更新背景车2轨迹和车辆
        if bv2_traj is not None and bv2_line is not None and bv2_rect is not None:
            bv2_past_mask = bv2_traj[:, 2] <= current_time
            if bv2_past_mask.any():
                bv2_line.set_data(bv2_traj[bv2_past_mask, 0], bv2_traj[bv2_past_mask, 1])

            # 找到背景车2轨迹当前时间对应的点
            bv2_current_idx = np.argmin(np.abs(bv2_traj[:, 2] - current_time))
            if bv2_current_idx < len(bv2_traj):
                bv2_current_point = bv2_traj[bv2_current_idx]

                # 计算背景车2航向角
                if bv2_current_idx < len(bv2_traj) - 1:
                    bv2_dx = bv2_traj[bv2_current_idx + 1, 0] - bv2_traj[bv2_current_idx, 0]
                    bv2_dy = bv2_traj[bv2_current_idx + 1, 1] - bv2_traj[bv2_current_idx, 1]
                else:
                    bv2_dx = bv2_traj[bv2_current_idx, 0] - bv2_traj[bv2_current_idx - 1, 0]
                    bv2_dy = bv2_traj[bv2_current_idx, 1] - bv2_traj[bv2_current_idx - 1, 1]

                if abs(bv2_dx) < 1e-4 and abs(bv2_dy) < 1e-4:
                    # 静止时固定角度
                    if "sce1" in model_name:
                        bv2_yaw = -175 * math.pi / 180
                    elif "sce2" in model_name:
                        bv2_yaw = 135 * math.pi / 180
                    else:
                        bv2_yaw = -90 * math.pi / 180
                else:
                    bv2_yaw = np.arctan2(bv2_dy, bv2_dx)

                if "sce1" in model_name:
                    bv2_length, bv2_width = 2.5, 1.5
                else:
                    bv2_length, bv2_width = 4.0, 2.0
                # 更新背景车2车辆矩形
                bv2_rect_corners = create_vehicle_rectangle(
                    bv2_current_point[0], bv2_current_point[1], bv2_yaw,
                    bv2_length, bv2_width
                )
                bv2_rect.set_xy(bv2_rect_corners.tolist())
                animate._bv2_last_yaw = bv2_yaw

        # 更新时间文本
        time_text.set_text(f'Time: {current_time:.2f}s')

        # 返回所有需要更新的元素
        elements_to_return = [line_trajectory, vehicle_rect, time_text]
        if human_line is not None:
            elements_to_return.append(human_line)
        if human_rect is not None:
            elements_to_return.append(human_rect)
        if bv1_line is not None:
            elements_to_return.append(bv1_line)
        if bv1_rect is not None:
            elements_to_return.append(bv1_rect)
        if bv2_line is not None:
            elements_to_return.append(bv2_line)
        if bv2_rect is not None:
            elements_to_return.append(bv2_rect)

        return elements_to_return

    # 计算动画帧数
    total_time = model_traj[-1, 2] - human_traj[0, 2]
    num_frames = int(total_time / time_step) + 1

    # 创建动画
    animation = FuncAnimation(fig, animate, frames=num_frames,
                              interval=time_step * 1000, blit=True, repeat=True)

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


def plot_losses(loss_history, epochs, save_path="training/loss/loss.png"):
    """
    绘制训练损失曲线：左侧为 total_loss 单独图，右侧为其余 4 种 loss 合并在一张图。
    所有文字使用 Times New Roman 字体。

    Parameters:
    -----------
    loss_history : dict of lists
        键包括 'total_loss', 'recon_loss', 'kld_loss', 'start_loss', 'time_loss'
        每个值为长度为 `epochs` 的 list，记录每 epoch 的平均损失
    epochs : int
        总训练轮数（即横坐标最大值，也用于验证数据长度）
    save_path : str
        保存路径（含文件名），默认 "training/loss/loss.png"
    """
    # 验证输入长度一致性
    expected_len = epochs
    for key, values in loss_history.items():
        if len(values) != expected_len:
            raise ValueError(f"Length of loss_history['{key}'] ({len(values)}) does not match epochs ({expected_len})")

    # === 设置 Times New Roman 全局字体 ===
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Serif"],
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
    })

    # 创建保存目录（若不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 横坐标：epoch 从 1 开始
    x_epochs = list(range(1, epochs + 1))

    # 定义颜色（确保 total_loss 在左图仍用蓝色；右图 4 种 loss 各用高对比色）
    colors = {
        'total_loss':   'tab:blue',
        'recon_loss':   '#1f77b4',  # muted blue
        'kld_loss':     '#ff7f0e',  # orange
        'start_loss':   '#2ca02c',  # green
        'time_loss':    '#d62728',  # red
    }
    labels = {
        'total_loss':   'Total Loss',
        'recon_loss':   'Reconstruction Loss',
        'kld_loss':     'KLD Loss',
        'start_loss':   'Start Loss',
        'time_loss':    'Time Loss',
    }

    # 创建双子图（1行2列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # --- 左图：total_loss 单独绘制 ---
    ax1.plot(x_epochs, loss_history['total_loss'],
             color=colors['total_loss'], label=labels['total_loss'], linewidth=2.0)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    # --- 右图：其余 4 种 loss 合并绘制 ---
    for key in ['recon_loss', 'kld_loss', 'start_loss', 'time_loss']:
        ax2.plot(x_epochs, loss_history[key],
                 color=colors[key], label=labels[key], linewidth=1.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Component Losses', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))  # 右上角内嵌，避免裁剪

    # 统一主标题（可选，注释掉则无）
    # fig.suptitle('Training Loss Breakdown', fontweight='bold', fontsize=18)

    # 保存 & 清理
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Loss plots saved to: {save_path}")


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

        # 条件生成：结合随机向量和条件向量生成“相对起点偏移轨迹” [t, dx, dy]
        rel_generated_samples = model.decode(z, h_condition).cpu().numpy()  # (num_samples, seq_len, dim)

        # 将相对轨迹转换为全局轨迹，便于与训练数据进行对比
        # start_points 此时为 (num_samples, 2) 的tensor，包含绝对起点(x_start, y_start)
        start_points_np = start_points.cpu().numpy() if hasattr(start_points, 'cpu') else np.asarray(start_points)
        generated_samples = rel_generated_samples.copy()
        for i in range(num_samples):
            generated_samples[i, :, 1] = start_points_np[i, 0] + rel_generated_samples[i, :, 1]
            generated_samples[i, :, 2] = start_points_np[i, 1] + rel_generated_samples[i, :, 2]
        
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
                    plt.plot(bg_x, bg_y, color=(62/255, 175/255, 73/255), linewidth=2, alpha=0.8, label='BV1')
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
                    plt.plot(bg_x, bg_y, color=(62/255, 175/255, 73/255), linewidth=2, alpha=0.8, label='BV1')
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
                    plt.plot(bg_x, bg_y, color=(62/255, 175/255, 73/255), linewidth=2, alpha=0.8, label='BV1')
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
            
            train_smooth_x, train_smooth_y = create_smooth_curve(train_points, time_interval=0.02)
            
            # 生成轨迹点
            gen_points = generated_samples[i, :, 1:3]  # 只取x和y坐标
            
            gen_smooth_x, gen_smooth_y = create_smooth_curve(gen_points, time_interval=0.02, start_angle=(-90)*(math.pi/180))

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
