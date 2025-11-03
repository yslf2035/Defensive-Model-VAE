"""
绘图工具模块
包含绘制GIF动图等功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os

# 设置中文字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def draw_vehicle_rectangle(ax, x, y, heading, length=4, width=2, color=[0, 0.447, 0.741]):
    """
    绘制车辆矩形
    
    Args:
        ax: matplotlib轴对象
        x, y: 车辆中心坐标
        heading: 车辆航向角（弧度）
        length: 车辆长度
        width: 车辆宽度
        color: 车辆颜色
    
    Returns:
        创建的矩形补丁对象
    """
    # 计算矩形的四个顶点（相对于中心的偏移）
    half_length = length / 2
    half_width = width / 2
    
    # 矩形顶点（相对于车辆中心）
    corners = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])
    
    # 旋转矩阵
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    
    # 旋转并平移顶点
    rotated_corners = corners @ rotation_matrix.T
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    # 创建矩形补丁（无边框）
    rectangle = patches.Polygon(rotated_corners, closed=True, 
                               facecolor=color, edgecolor=None, linewidth=0)
    ax.add_patch(rectangle)
    
    return rectangle


def draw_lane_lines(ax, x_range=[-10, 90], y_positions=[-1.75, 1.75], line_width=1.5):
    """
    绘制车道线
    
    Args:
        ax: matplotlib轴对象
        x_range: x坐标范围 [x_min, x_max]
        y_positions: y坐标位置列表
        line_width: 线宽
    """
    for y_pos in y_positions:
        ax.plot([x_range[0], x_range[1]], [y_pos, y_pos], 
                'k-', linewidth=line_width, alpha=0.8)


def draw_background_vehicle(ax, x=58, y=-3, length=4, width=2, color=[0.466, 0.674, 0.188]):
    """
    绘制背景车辆
    
    Args:
        ax: matplotlib轴对象
        x: 车辆中心x坐标
        y: 车辆中心y坐标
        length: 车辆长度
        width: 车辆宽度
        color: 车辆颜色
    """
    # 背景车辆朝向x轴正方向
    heading = 0
    
    # 计算矩形的四个顶点
    half_length = length / 2
    half_width = width / 2
    
    corners = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])
    
    # 旋转并平移顶点
    rotated_corners = corners.copy()
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    # 创建矩形补丁（无边框）
    rectangle = patches.Polygon(rotated_corners, closed=True, 
                               facecolor=color, edgecolor=None, linewidth=0)
    ax.add_patch(rectangle)


def calculate_moving_bg_vehicle_trajectory(waypoints, speed=3.0, dt=0.05):
    """
    计算运动背景车的轨迹
    
    Args:
        waypoints: 轨迹点 [(x1,y1), (x2,y2), ...]
        speed: 车辆速度 (m/s)
        dt: 时间步长
    
    Returns:
        trajectory: 轨迹数据 [N, 3] - [x, y, t]
    """
    trajectory = []
    current_time = 0.0
    
    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i + 1]
        
        # 计算两点间距离
        distance = np.sqrt((end_point[0] - start_point[0])**2 + 
                          (end_point[1] - start_point[1])**2)
        
        # 计算所需时间
        segment_time = distance / speed
        
        # 生成该段的轨迹点
        num_points = int(segment_time / dt) + 1
        for j in range(num_points):
            t = j * dt
            if t <= segment_time:
                # 线性插值
                alpha = t / segment_time if segment_time > 0 else 0
                x = start_point[0] + alpha * (end_point[0] - start_point[0])
                y = start_point[1] + alpha * (end_point[1] - start_point[1])
                trajectory.append([x, y, current_time + t])
        
        current_time += segment_time
    
    # 添加最后一个点
    if len(waypoints) > 0:
        last_point = waypoints[-1]
        trajectory.append([last_point[0], last_point[1], current_time])
    
    return np.array(trajectory)


def draw_moving_bg_vehicle(ax, x, y, heading, length=2, width=1, color=[0.466, 0.674, 0.188]):
    """
    绘制运动背景车辆
    
    Args:
        ax: matplotlib轴对象
        x, y: 车辆中心坐标
        heading: 车辆航向角（弧度）
        length: 车辆长度
        width: 车辆宽度
        color: 车辆颜色
    
    Returns:
        创建的矩形补丁对象
    """
    # 计算矩形的四个顶点（相对于中心的偏移）
    half_length = length / 2
    half_width = width / 2
    
    # 矩形顶点（相对于车辆中心）
    corners = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])
    
    # 旋转矩阵
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    
    # 旋转并平移顶点
    rotated_corners = corners @ rotation_matrix.T
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    # 创建矩形补丁（无边框）
    rectangle = patches.Polygon(rotated_corners, closed=True, 
                               facecolor=color, edgecolor=None, linewidth=0)
    ax.add_patch(rectangle)
    
    return rectangle


def create_path_tracking_gif(actual_path, output_path="gifs/path_tracking.gif", 
                             fps=20, dpi=100, figsize=(12, 8)):
    """
    创建路径跟踪GIF动图
    
    Args:
        actual_path: 实际路径数据 [N, 3] - [x, y, t]
        output_path: 输出文件路径
        fps: 帧率
        dpi: 图像分辨率
        figsize: 图像尺寸
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 提取数据
    x_coords = actual_path[:, 0]
    y_coords = actual_path[:, 1]
    times = actual_path[:, 2]
    
    # 计算主车航向角（基于轨迹方向）
    headings = np.zeros(len(x_coords))
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        if dx != 0 or dy != 0:
            headings[i] = np.arctan2(dy, dx)
        else:
            headings[i] = headings[i-1]
    
    # 计算运动背景车轨迹
    moving_bg_waypoints = [(40, -3), (50, -3), (54, -1), (62, -1), (66, -3), (68, -3)]
    moving_bg_trajectory = calculate_moving_bg_vehicle_trajectory(moving_bg_waypoints, speed=3.0, dt=0.05)
    
    # 计算运动背景车航向角
    bg_x_coords = moving_bg_trajectory[:, 0]
    bg_y_coords = moving_bg_trajectory[:, 1]
    bg_headings = np.zeros(len(bg_x_coords))
    for i in range(1, len(bg_x_coords)):
        dx = bg_x_coords[i] - bg_x_coords[i-1]
        dy = bg_y_coords[i] - bg_y_coords[i-1]
        if dx != 0 or dy != 0:
            bg_headings[i] = np.arctan2(dy, dx)
        else:
            bg_headings[i] = bg_headings[i-1]
    
    # 设置图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置坐标轴范围
    margin = 5
    ax.set_xlim(min(min(x_coords), min(bg_x_coords)) - margin, 
                max(max(x_coords), max(bg_x_coords)) + margin)
    ax.set_ylim(min(min(y_coords), min(bg_y_coords)) - margin, 
                max(max(y_coords), max(bg_y_coords)) + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=16)
    ax.set_ylabel('Y (m)', fontsize=16)
    
    # 绘制车道线（底层）
    draw_lane_lines(ax, line_width=1.5)
    
    # 绘制静止背景车辆（底层）
    draw_background_vehicle(ax)
    
    # 初始化主车轨迹线
    trajectory_line, = ax.plot([], [], color=[0, 0.447, 0.741], linewidth=3, alpha=0.8)
    
    # 初始化运动背景车轨迹线
    bg_trajectory_line, = ax.plot([], [], color=[0.466, 0.674, 0.188], linewidth=2, alpha=0.8)
    
    # 初始化车辆矩形
    vehicle_patch = None
    bg_vehicle_patch = None
    
    def animate(frame):
        nonlocal vehicle_patch, bg_vehicle_patch
        
        # 清除之前的车辆矩形
        if vehicle_patch is not None:
            vehicle_patch.remove()
            vehicle_patch = None
        
        if bg_vehicle_patch is not None:
            bg_vehicle_patch.remove()
            bg_vehicle_patch = None
        
        # 更新主车轨迹线（显示历史轨迹）
        if frame > 0:
            trajectory_line.set_data(x_coords[:frame+1], y_coords[:frame+1])
        
        # 绘制当前帧的主车矩形
        if frame < len(x_coords):
            current_x = x_coords[frame]
            current_y = y_coords[frame]
            current_heading = headings[frame]
            
            # 绘制主车矩形并保存引用
            vehicle_patch = draw_vehicle_rectangle(ax, current_x, current_y, current_heading)
        
        # 绘制当前帧的运动背景车矩形和轨迹
        current_time = times[frame] if frame < len(times) else times[-1]
        
        # 找到运动背景车在当前时间的位置
        bg_frame = None
        for i, bg_time in enumerate(moving_bg_trajectory[:, 2]):
            if bg_time <= current_time:
                bg_frame = i
            else:
                break
        
        if bg_frame is not None and bg_frame < len(bg_x_coords):
            bg_x = bg_x_coords[bg_frame]
            bg_y = bg_y_coords[bg_frame]
            bg_heading = bg_headings[bg_frame]
            
            # 更新运动背景车轨迹线（显示历史轨迹）
            bg_trajectory_line.set_data(bg_x_coords[:bg_frame+1], bg_y_coords[:bg_frame+1])
            
            # 绘制运动背景车矩形
            bg_vehicle_patch = draw_moving_bg_vehicle(ax, bg_x, bg_y, bg_heading, length=2, width=1)
        
        return trajectory_line, bg_trajectory_line,
    
    # 创建动画
    print(f"正在创建GIF动画...")
    print(f"总帧数: {len(x_coords)}")
    print(f"帧率: {fps} fps")
    print(f"预计时长: {len(x_coords)/fps:.1f} 秒")
    
    anim = FuncAnimation(fig, animate, frames=len(x_coords), 
                        interval=1000/fps, blit=False, repeat=True)
    
    # 保存GIF
    print(f"正在保存GIF到: {output_path}")
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    
    plt.close(fig)
    print(f"GIF动画已保存完成！")
    
    return output_path


def create_path_tracking_gif_with_reference(actual_path, reference_path, 
                                          output_path="gifs/path_tracking_with_ref.gif",
                                          fps=20, dpi=100, figsize=(12, 8)):
    """
    创建包含参考路径的路径跟踪GIF动图
    
    Args:
        actual_path: 实际路径数据 [N, 3] - [x, y, t]
        reference_path: 参考路径数据 [N, 3] - [x, y, t]
        output_path: 输出文件路径
        fps: 帧率
        dpi: 图像分辨率
        figsize: 图像尺寸
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 提取数据
    x_coords = actual_path[:, 0]
    y_coords = actual_path[:, 1]
    times = actual_path[:, 2]
    
    ref_x_coords = reference_path[:, 0]
    ref_y_coords = reference_path[:, 1]
    
    # 计算车辆航向角
    headings = np.zeros(len(x_coords))
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        if dx != 0 or dy != 0:
            headings[i] = np.arctan2(dy, dx)
        else:
            headings[i] = headings[i-1]
    
    # 设置图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置坐标轴范围
    margin = 5
    ax.set_xlim(min(min(x_coords), min(ref_x_coords)) - margin, 
                max(max(x_coords), max(ref_x_coords)) + margin)
    ax.set_ylim(min(min(y_coords), min(ref_y_coords)) - margin, 
                max(max(y_coords), max(ref_y_coords)) + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=16)
    ax.set_ylabel('Y (m)', fontsize=16)
    
    # 绘制车道线
    draw_lane_lines(ax, line_width=3)
    
    # 绘制背景车辆
    draw_background_vehicle(ax)
    
    # 绘制参考路径（静态）
    ax.plot(ref_x_coords, ref_y_coords, 'r--', linewidth=2, alpha=0.8, label='参考路径')
    
    # 初始化实际轨迹线
    trajectory_line, = ax.plot([], [], color=[0, 0.447, 0.741], linewidth=3, alpha=0.8, label='实际路径')
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 初始化车辆矩形
    vehicle_patch = None
    
    def animate(frame):
        nonlocal vehicle_patch
        
        # 清除之前的车辆矩形
        if vehicle_patch is not None:
            vehicle_patch.remove()
            vehicle_patch = None
        
        # 更新轨迹线（显示历史轨迹）
        if frame > 0:
            trajectory_line.set_data(x_coords[:frame+1], y_coords[:frame+1])
        
        # 绘制当前帧的车辆矩形
        if frame < len(x_coords):
            current_x = x_coords[frame]
            current_y = y_coords[frame]
            current_heading = headings[frame]
            
            # 绘制车辆矩形并保存引用
            vehicle_patch = draw_vehicle_rectangle(ax, current_x, current_y, current_heading)
        
        return trajectory_line,
    
    # 创建动画
    print(f"正在创建GIF动画（含参考路径）...")
    print(f"总帧数: {len(x_coords)}")
    print(f"帧率: {fps} fps")
    print(f"预计时长: {len(x_coords)/fps:.1f} 秒")
    
    anim = FuncAnimation(fig, animate, frames=len(x_coords), 
                        interval=1000/fps, blit=False, repeat=True)
    
    # 保存GIF
    print(f"正在保存GIF到: {output_path}")
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    
    plt.close(fig)
    print(f"GIF动画已保存完成！")
    
    return output_path
