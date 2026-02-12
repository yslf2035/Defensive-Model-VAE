"""
空间分布相关的函数：坐标提取、RMSE计算、colorbar范围计算和空间分布绘制
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D plotting
from scipy import ndimage

# 全局字体配置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


def extract_coordinates_from_trajectories(trajectories):
    """
    从轨迹状态序列中提取所有(x, y)坐标点
    
    Args:
        trajectories: 轨迹列表，每个元素是 [N, 4] 的状态序列 [x, y, theta, v]
        
    Returns:
        coordinates: 所有坐标点 [M, 2] - [x, y]
    """
    coordinates = []
    for traj in trajectories:
        # 状态序列的前两列是x和y坐标
        xy = traj[:, [0, 1]]
        num_points = xy.shape[0]
        sampling_num = 150  # 采样点数

        if num_points <= sampling_num:
            sampled_xy = xy
        else:
            # 生成均匀采样索引（包含第一个和最后一个索引）
            # 生成0到num_points-1的均匀分布值，转整数去重
            indices = np.linspace(0, num_points - 1, num=sampling_num, dtype=int)
            # 去重（避免linspace因浮点精度导致重复索引）
            indices = np.unique(indices)
            # 极端情况：去重后不足20个点，随机补充不重复的索引（保证总数20）
            if len(indices) < sampling_num:
                all_indices = np.arange(num_points)
                unused_indices = all_indices[~np.isin(all_indices, indices)]
                extra_indices = np.random.choice(unused_indices, size=sampling_num - len(indices), replace=False)
                indices = np.sort(np.concatenate([indices, extra_indices]))
            # 根据索引采样
            sampled_xy = xy[indices]

        coordinates.append(sampled_xy)
    return np.vstack(coordinates)


def extract_human_coordinates(human_trajectories):
    """
    从人类轨迹数据中提取所有(x, y)坐标点
    
    Args:
        human_trajectories: 轨迹数组 (num_samples, target_points, 3) - [时间, x, y]
        
    Returns:
        coordinates: 所有坐标点 [M, 2] - [x, y]
    """
    coordinates = []
    for traj in human_trajectories:
        # traj shape: [x, y, t]
        xy = traj[:, [0, 1]]  # 提取x和y坐标
        num_points = xy.shape[0]
        sampling_num = 150  # 采样点数

        if num_points <= sampling_num:
            sampled_xy = xy
        else:
            # 生成均匀采样索引（包含第一个和最后一个索引）
            # 生成0到num_points-1的均匀分布值，转整数去重
            indices = np.linspace(0, num_points - 1, num=sampling_num, dtype=int)
            # 去重（避免linspace因浮点精度导致重复索引）
            indices = np.unique(indices)
            # 极端情况：去重后不足20个点，随机补充不重复的索引（保证总数20）
            if len(indices) < sampling_num:
                all_indices = np.arange(num_points)
                unused_indices = all_indices[~np.isin(all_indices, indices)]
                extra_indices = np.random.choice(unused_indices, size=sampling_num - len(indices), replace=False)
                indices = np.sort(np.concatenate([indices, extra_indices]))
            # 根据索引采样
            sampled_xy = xy[indices]

        coordinates.append(sampled_xy)
    return np.vstack(coordinates)


def calculate_rmse_frequency(generated_coordinates, human_coordinates, model_name, grid_size=1.0):
    """
    计算模型轨迹空间分布和人类轨迹空间分布之间的RMSE_frequency差异指标
    
    公式: RMSE_frequency = sqrt( (1/n) * sum_{i=1 to n} (f_sim,i - f_obs,i)^2 )
    其中:
    - f_sim,i: 模型轨迹在第i个方格中出现的频次
    - f_obs,i: 人类轨迹在第i个方格中出现的频次
    - n: 方格总数（只考虑存在模型轨迹或人类轨迹的方格，如果两个都是0则不考虑）
    
    Args:
        generated_coordinates: 模型生成轨迹的坐标数组 [N, 2] - [x, y]
        human_coordinates: 人类轨迹的坐标数组 [M, 2] - [x, y]
        model_name: 模型名称
        grid_size: 网格大小（米），默认1.0m
        
    Returns:
        rmse_frequency: RMSE_frequency值
    """
    # 确定网格边界（使用_get_grid_edges函数）
    x_edges, y_edges = _get_grid_edges(model_name, grid_size)
    
    # 计算模型轨迹的直方图（频次）
    if len(generated_coordinates) > 0:
        H_sim, _, _ = np.histogram2d(
            generated_coordinates[:, 0], generated_coordinates[:, 1],
            bins=[x_edges, y_edges]
        )
        H_sim = H_sim.T  # 转置以便正确显示
    else:
        H_sim = np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    
    # 计算人类轨迹的直方图（频次）
    if len(human_coordinates) > 0:
        H_obs, _, _ = np.histogram2d(
            human_coordinates[:, 0], human_coordinates[:, 1],
            bins=[x_edges, y_edges]
        )
        H_obs = H_obs.T  # 转置以便正确显示
    else:
        H_obs = np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    
    # 展平为一维数组
    f_sim = H_sim.flatten()
    f_obs = H_obs.flatten()
    
    # 只考虑存在模型轨迹或人类轨迹的方格（如果两个都是0则不考虑）
    mask = (f_sim > 0) | (f_obs > 0)
    f_sim_filtered = f_sim[mask]
    f_obs_filtered = f_obs[mask]

    
    # 计算方格总数n（只考虑至少有一个分布非零的方格）
    n = len(f_sim_filtered)

    if n == 0:
        print("Warning: No valid grids found (both distributions are zero in all grids)")
        return 0.0
    
    # 计算RMSE_frequency = sqrt( (1/n) * sum_{i=1 to n} (f_sim,i - f_obs,i)^2 )
    squared_diff = (f_sim_filtered - f_obs_filtered) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmse_frequency = np.sqrt(mean_squared_diff)
    
    print(f"\nSpatial Distribution RMSE_frequency: {rmse_frequency:.6f}")
    print(f"  Number of valid grids (n): {n}")
    
    return rmse_frequency


def calculate_unified_colorbar_range(coordinates_list, model_name, grid_size=1.0):
    """
    计算多个坐标分布的统一colorbar范围
    
    Args:
        coordinates_list: 坐标数组列表，每个元素是 [N, 2] - [x, y]
        model_name: 模型名称
        grid_size: 网格大小（米）
        
    Returns:
        vmin: 统一的最小值
        vmax: 统一的最大值
    """
    # 确定坐标范围（使用_get_grid_edges函数）
    x_edges, y_edges = _get_grid_edges(model_name, grid_size)
    
    # 计算所有分布的最大最小值
    all_max_counts = []
    all_min_counts = []
    
    for coordinates in coordinates_list:
        if len(coordinates) == 0:
            continue
        
        # 计算直方图
        H, _, _ = np.histogram2d(
            coordinates[:, 0], coordinates[:, 1],
            bins=[x_edges, y_edges]
        )
        H = H.T
        
        # 记录非零值的最小值和最大值
        non_zero_H = H[H > 0]
        if len(non_zero_H) > 0:
            all_max_counts.append(np.max(H))
            all_min_counts.append(np.min(non_zero_H))
    
    # 返回全局最大值和最小值
    if len(all_max_counts) > 0:
        vmax = max(all_max_counts)
        vmin = min(all_min_counts) if len(all_min_counts) > 0 else 0
    else:
        vmax = 1
        vmin = 0
    
    return vmin, vmax


def plot_spatial_distribution(coordinates, title, model_name, save_path=None, grid_size=1.0,
                              cmap=None, vmin=None, vmax=None):
    """
    绘制坐标在xOy坐标系中的分布热力图（1m×1m方格）
    
    Args:
        coordinates: 坐标点数组 [N, 2] - [x, y]
        title: 图标题
        model_name: 模型名称
        save_path: 保存路径（可选）
        grid_size: 网格大小（米），默认1.0m
        cmap: 可选的自定义颜色映射
        vmin: 可选的颜色归一化范围
        vmax: 可选的颜色归一化范围
    """
    if len(coordinates) == 0:
        print(f"Warning: No coordinates to plot for {title}")
        return None, None

    if "sce1" in model_name:
        # 坐标轴范围
        xlim = (-198, -188)
        ylim = (40, 80)
        base_size = 20  # 图窗基准尺寸
    elif "sce2" in model_name:
        xlim = (-200, -120)
        ylim = (-8, 6)
        base_size = 20
    elif "sce4" in model_name:
        xlim = (-45, 65)
        ylim = (-10, 100)
        base_size = 30
    else:
        xlim = (0, 20)
        ylim = (-20, 110)
        base_size = 40

    x_range = xlim[1] - xlim[0]  # x轴总长度
    y_range = ylim[1] - ylim[0]  # y轴总长度
    # 自动计算figsize（保证1:1比例下完整显示坐标轴范围）
    aspect_ratio = x_range / y_range  # x/y长度比
    if aspect_ratio >= 1:
        # x轴更长：宽=基准尺寸，高=基准尺寸/比例
        fig_width = base_size
        fig_height = base_size / aspect_ratio
    else:
        # y轴更长：宽=基准尺寸*比例，高=基准尺寸
        fig_width = base_size * aspect_ratio
        fig_height = base_size
    figsize = (fig_width, fig_height)

    # 坐标范围（使用_get_grid_edges函数）
    x_edges, y_edges = _get_grid_edges(model_name, grid_size)

    # 计算每个网格内的点数
    H, x_edges, y_edges = np.histogram2d(
        coordinates[:, 0], coordinates[:, 1],
        bins=[x_edges, y_edges]
    )
    
    # 转置以便正确显示（histogram2d返回的矩阵需要转置）
    H = H.T
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    
    # 绘制热力图（使用pcolormesh以更精确地显示1m×1m网格）
    im = ax.pcolormesh(
        x_edges, y_edges, H,
        cmap=(cmap if cmap is not None else LinearSegmentedColormap.from_list(
            'sky_to_darkred', ['#87CEEB', '#FFA07A', '#FF4500', '#8B0000'])
        ),
        vmin=vmin, vmax=vmax,
        shading='flat',
        edgecolors='none'
    )
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Points', fontsize=16)
    
    # 绘制网格边框（白色实线）
    for x in x_edges:
        ax.axvline(x, color='white', linewidth=0.5, linestyle='-', alpha=0.8)
    for y in y_edges:
        ax.axhline(y, color='white', linewidth=0.5, linestyle='-', alpha=0.8)

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
        # sce4场景：五条车道线，x坐标分别为4、7.5、11、14.5、18，y范围[-40,120]
        y_range = np.linspace(-40, 120, 100)
        ax.plot([4] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)  # 最左边实线
        ax.plot([7.5] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)  # 虚线
        ax.plot([11] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)  # 虚线
        ax.plot([14.5] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)  # 虚线
        ax.plot([18] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)  # 最右边实线
    else:
        # sce3场景：三条车道线
        y_range = np.linspace(-100, 60, 100)
        ax.plot([153.3] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
        ax.plot([156.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
        ax.plot([149.7] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
    
    # 设置标签和标题
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title(title, fontsize=16)
    if "sce1" in model_name or "sce2" in model_name:
        ax.invert_xaxis()
    elif "sce3" in model_name or "sce4" in model_name:
        ax.invert_yaxis()
    ax.axis('off')

    # 添加统计信息
    total_points = len(coordinates)
    max_count = np.max(H)
    stats_text = f'Total points: {total_points}\nMax count per grid: {int(max_count)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Spatial distribution plot saved to: {save_path}")
    
    # plt.show()
    
    return x_edges, y_edges


def _get_grid_edges(model_name, grid_size=1.0):
    """
    获取网格边界（辅助函数）
    
    Args:
        model_name: 模型名称
        grid_size: 网格大小（米）
        
    Returns:
        x_edges: x轴网格边界
        y_edges: y轴网格边界
    """
    if "sce1" in model_name:
        x_edges = np.arange(-198, -188 + 1, grid_size)
        y_edges = np.arange(40, 80 + 1, grid_size)
    elif "sce2" in model_name:
        x_edges = np.arange(-200, -120, grid_size)
        y_edges = np.arange(-8, 6, grid_size)
    elif "sce3" in model_name:
        x_edges = np.arange(148, 158, grid_size)
        y_edges = np.arange(-80, 0, grid_size)
    else:
        x_edges = np.arange(0, 20, grid_size)
        y_edges = np.arange(-20, 100, grid_size)
    return x_edges, y_edges


def _count_trajectories_per_grid(trajectories, model_name, grid_size=1.0):
    """
    统计每条轨迹经过的方格（每条轨迹经过某个方格只计数1次）
    
    Args:
        trajectories: 轨迹列表，每个元素是 [N, 4] 的状态序列 [x, y, theta, v] 或 [N, 3] 的 [x, y, t]
        model_name: 模型名称
        grid_size: 网格大小（米）
        
    Returns:
        H: 统计矩阵，H[i, j] 表示有多少条轨迹经过了第(i, j)个方格
        x_edges: x轴网格边界
        y_edges: y轴网格边界
    """
    x_edges, y_edges = _get_grid_edges(model_name, grid_size)
    
    # 初始化统计矩阵
    H = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=int)
    
    for traj in trajectories:
        # 提取坐标（支持两种格式：[x, y, theta, v] 或 [x, y, t]）
        if traj.shape[1] >= 2:
            xy = traj[:, [0, 1]]
        else:
            continue
        
        # 找出这条轨迹经过的所有唯一方格
        # 计算每个点所在的方格索引
        x_indices = np.digitize(xy[:, 0], x_edges) - 1
        y_indices = np.digitize(xy[:, 1], y_edges) - 1
        
        # 限制索引范围在有效范围内
        x_indices = np.clip(x_indices, 0, len(x_edges) - 2)
        y_indices = np.clip(y_indices, 0, len(y_edges) - 2)
        
        # 找出唯一的方格（每条轨迹经过的方格）
        unique_grids = set(zip(y_indices, x_indices))
        
        # 对每个经过的方格计数加1（每条轨迹只计数一次）
        for y_idx, x_idx in unique_grids:
            if 0 <= y_idx < H.shape[0] and 0 <= x_idx < H.shape[1]:
                H[y_idx, x_idx] += 1
    
    return H, x_edges, y_edges


def calculate_rmse_frequency_new(generated_trajectories, human_trajectories, model_name, grid_size=1.0):
    """
    计算模型轨迹空间分布和人类轨迹空间分布之间的RMSE_frequency差异指标（新统计方式）
    
    统计方式：对于每条轨迹，如果该轨迹经过了某个方格，则该方格的计数加1
    （如果一条轨迹的很多点都在某个方格内，该方格依旧只加1的计数）
    
    公式: RMSE_frequency = sqrt( (1/n) * sum_{i=1 to n} (f_sim,i - f_obs,i)^2 )
    其中:
    - f_sim,i: 经过第i个方格的模型轨迹数量
    - f_obs,i: 经过第i个方格的人类轨迹数量
    - n: 方格总数（只考虑存在模型轨迹或人类轨迹的方格，如果两个都是0则不考虑）
    
    Args:
        generated_trajectories: 模型生成轨迹列表，每个元素是 [N, 4] 的状态序列 [x, y, theta, v]
        human_trajectories: 人类轨迹列表，每个元素是 [N, 3] 的 [x, y, t]
        model_name: 模型名称
        grid_size: 网格大小（米），默认1.0m
        
    Returns:
        rmse_frequency: RMSE_frequency值
    """
    # 计算模型轨迹的统计矩阵（每条轨迹经过的方格）
    if len(generated_trajectories) > 0:
        H_sim, x_edges, y_edges = _count_trajectories_per_grid(generated_trajectories, model_name, grid_size)
    else:
        x_edges, y_edges = _get_grid_edges(model_name, grid_size)
        H_sim = np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    
    # 计算人类轨迹的统计矩阵（每条轨迹经过的方格）
    if len(human_trajectories) > 0:
        H_obs, _, _ = _count_trajectories_per_grid(human_trajectories, model_name, grid_size)
    else:
        H_obs = np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    
    # 展平为一维数组
    f_sim = H_sim.flatten()
    f_obs = H_obs.flatten()
    
    # 只考虑存在模型轨迹或人类轨迹的方格（如果两个都是0则不考虑）
    mask = (f_sim > 0) | (f_obs > 0)
    f_sim_filtered = f_sim[mask]
    f_obs_filtered = f_obs[mask]
    
    # 计算方格总数n（只考虑至少有一个分布非零的方格）
    n = len(f_sim_filtered)

    if n == 0:
        print("Warning: No valid grids found (both distributions are zero in all grids)")
        return 0.0
    
    # 计算RMSE_frequency = sqrt( (1/n) * sum_{i=1 to n} (f_sim,i - f_obs,i)^2 )
    squared_diff = (f_sim_filtered - f_obs_filtered) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmse_frequency = np.sqrt(mean_squared_diff)
    
    print(f"\nSpatial Distribution RMSE_frequency (new method): {rmse_frequency:.6f}")
    print(f"  Number of valid grids (n): {n}")
    
    return rmse_frequency


def calculate_unified_colorbar_range_new(trajectories_list, model_name, grid_size=1.0):
    """
    计算多个轨迹分布的统一colorbar范围（新统计方式）
    
    统计方式：对于每条轨迹，如果该轨迹经过了某个方格，则该方格的计数加1
    
    Args:
        trajectories_list: 轨迹列表的列表，每个元素是轨迹列表
        model_name: 模型名称
        grid_size: 网格大小（米）
        
    Returns:
        vmin: 统一的最小值
        vmax: 统一的最大值
    """
    x_edges, y_edges = _get_grid_edges(model_name, grid_size)
    
    # 计算所有分布的最大最小值
    all_max_counts = []
    all_min_counts = []
    
    for trajectories in trajectories_list:
        if len(trajectories) == 0:
            continue
        
        # 计算统计矩阵（每条轨迹经过的方格）
        H, _, _ = _count_trajectories_per_grid(trajectories, model_name, grid_size)
        
        # 记录非零值的最小值和最大值
        non_zero_H = H[H > 0]
        if len(non_zero_H) > 0:
            all_max_counts.append(np.max(H))
            all_min_counts.append(np.min(non_zero_H))
    
    # 返回全局最大值和最小值
    if len(all_max_counts) > 0:
        vmax = max(all_max_counts)
        vmin = min(all_min_counts) if len(all_min_counts) > 0 else 0
    else:
        vmax = 1
        vmin = 0
    
    return vmin, vmax


def plot_spatial_distribution_new(trajectories, title, model_name, save_path=None, grid_size=1.0,
                                  cmap=None, vmin=None, vmax=None):
    """
    绘制坐标在xOy坐标系中的分布热力图（新统计方式）
    
    统计方式：对于每条轨迹，如果该轨迹经过了某个方格，则该方格的计数加1
    （如果一条轨迹的很多点都在某个方格内，该方格依旧只加1的计数）
    
    Args:
        trajectories: 轨迹列表，每个元素是 [N, 4] 的状态序列 [x, y, theta, v] 或 [N, 3] 的 [x, y, t]
        title: 图标题
        model_name: 模型名称
        save_path: 保存路径（可选）
        grid_size: 网格大小（米），默认1.0m
        cmap: 可选的自定义颜色映射
        vmin: 可选的颜色归一化范围
        vmax: 可选的颜色归一化范围
    """
    if len(trajectories) == 0:
        print(f"Warning: No trajectories to plot for {title}")
        return None, None

    if "sce1" in model_name:
        # 坐标轴范围
        xlim = (-198, -188)
        ylim = (40, 80)
        base_size = 20  # 图窗基准尺寸
    elif "sce2" in model_name:
        xlim = (-200, -120)
        ylim = (-8, 6)
        base_size = 20
    elif "sce4" in model_name:
        xlim = (-45, 65)
        ylim = (-10, 100)
        base_size = 30
    else:
        xlim = (0, 20)
        ylim = (-20, 100)
        base_size = 40

    x_range = xlim[1] - xlim[0]  # x轴总长度
    y_range = ylim[1] - ylim[0]  # y轴总长度
    # 自动计算figsize（保证1:1比例下完整显示坐标轴范围）
    aspect_ratio = x_range / y_range  # x/y长度比
    if aspect_ratio >= 1:
        # x轴更长：宽=基准尺寸，高=基准尺寸/比例
        fig_width = base_size
        fig_height = base_size / aspect_ratio
    else:
        # y轴更长：宽=基准尺寸*比例，高=基准尺寸
        fig_width = base_size * aspect_ratio
        fig_height = base_size
    figsize = (fig_width, fig_height)

    # 计算每条轨迹经过的方格统计
    H, x_edges, y_edges = _count_trajectories_per_grid(trajectories, model_name, grid_size)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    
    # 绘制热力图（使用pcolormesh以更精确地显示1m×1m网格）
    im = ax.pcolormesh(
        x_edges, y_edges, H,
        cmap=(cmap if cmap is not None else LinearSegmentedColormap.from_list(
            'sky_to_darkred', ['#87CEEB', '#FFA07A', '#FF4500', '#8B0000'])
        ),
        vmin=vmin, vmax=vmax,
        shading='flat',
        edgecolors='none'
    )
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Trajectories', fontsize=16)
    
    # 绘制网格边框（白色实线）
    for x in x_edges:
        ax.axvline(x, color='white', linewidth=0.5, linestyle='-', alpha=0.8)
    for y in y_edges:
        ax.axhline(y, color='white', linewidth=0.5, linestyle='-', alpha=0.8)

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
    
    # 设置标签和标题
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title(title, fontsize=16)
    if "sce1" in model_name or "sce2" in model_name:
        ax.invert_xaxis()
    elif "sce3" in model_name or "sce4" in model_name:
        ax.invert_yaxis()
    # ax.axis('off')

    # 添加统计信息
    total_trajectories = len(trajectories)
    max_count = np.max(H)
    stats_text = f'Total trajectories: {total_trajectories}\nMax trajectories per grid: {int(max_count)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Spatial distribution plot saved to: {save_path}")
    
    # plt.show()
    
    return x_edges, y_edges


def _get_model_time_step(model_name):
    """
    根据模型名称推断时间步长（与Distribution.py中process_single_trajectory保持一致）
    
    Args:
        model_name: 模型文件名字符串，例如 'vae_offset_sce4_cond_ld8_epoch3000.pth'
        
    Returns:
        dt: 推断的时间步长
    """
    if "sce1" in model_name:
        return 0.02
    elif "sce2" in model_name:
        return 0.025
    elif "sce3" in model_name:
        return 0.015
    elif "sce4" in model_name:
        return 0.02
    else:
        return 0.02


def _prepare_model_stv_data(trajectories, model_name, axis="x"):
    """
    从模型轨迹中提取坐标-时间-速度三元组
    
    Args:
        trajectories: 模型轨迹列表，每个元素为 [N, 4] 的数组 [x, y, theta, v]
        model_name: 模型名称，用于推断时间步长
        axis: 'x' 或 'y'，决定使用哪一维坐标
        
    Returns:
        coords_list: 各条轨迹的坐标序列列表
        times_list: 各条轨迹的时间序列列表
        v_list: 各条轨迹的速度序列列表
    """
    dt = _get_model_time_step(model_name)
    coord_index = 0 if axis == "x" else 1
    
    coords_list = []
    times_list = []
    v_list = []
    
    for traj in trajectories:
        if traj.shape[1] < 4:
            continue
        n = traj.shape[0]
        times = np.arange(n) * dt
        coords = traj[:, coord_index]
        v = traj[:, 3]
        coords_list.append(coords)
        times_list.append(times)
        v_list.append(v)
    
    return coords_list, times_list, v_list


def _prepare_human_stv_data(human_trajectories, axis="x"):
    """
    从人类轨迹中提取坐标-时间-速度三元组
    
    人类轨迹格式参考 Distribution.py 中 calculate_human_velocities：
        traj: [x, y, t]
    
    Args:
        human_trajectories: 人类轨迹列表，每个元素为 [N, 3] 的数组 [x, y, t]
        axis: 'x' 或 'y'，决定使用哪一维坐标
        
    Returns:
        coords_list: 各条轨迹的坐标序列列表
        times_list: 各条轨迹的时间序列列表
        v_list: 各条轨迹的速度序列列表（按照 calculate_human_velocities 的方式计算）
    """
    coord_index = 0 if axis == "x" else 1
    
    coords_list = []
    times_list = []
    v_list = []
    
    for traj in human_trajectories:
        if traj.shape[1] < 3 or traj.shape[0] < 2:
            continue
        times = traj[:, 2]
        coords = traj[:, coord_index]
        
        x_coords = traj[:, 0]
        y_coords = traj[:, 1]
        velocities = []
        for i in range(len(traj) - 1):
            dt = times[i + 1] - times[i]
            if dt > 1e-6:
                dx = x_coords[i + 1] - x_coords[i]
                dy = y_coords[i + 1] - y_coords[i]
                v = np.sqrt(dx ** 2 + dy ** 2) / dt
                velocities.append(v)
            else:
                if len(velocities) > 0:
                    velocities.append(velocities[-1])
                else:
                    velocities.append(0.0)
        # 最后一个点使用前一时刻的速度
        if len(velocities) > 0:
            velocities.append(velocities[-1])
        else:
            velocities.append(0.0)
        
        velocities = np.array(velocities)
        coords_list.append(coords)
        times_list.append(times)
        v_list.append(velocities)
    
    return coords_list, times_list, v_list


def _calculate_unified_axes_ranges(model_coords_list, model_times_list, human_coords_list, human_times_list):
    """
    计算模型和人类轨迹的统一坐标轴范围
    
    Args:
        model_coords_list: 模型轨迹的坐标序列列表
        model_times_list: 模型轨迹的时间序列列表
        human_coords_list: 人类轨迹的坐标序列列表
        human_times_list: 人类轨迹的时间序列列表
        
    Returns:
        coord_range: (coord_min, coord_max)
        time_range: (time_min, time_max)
    """
    all_model_coords = np.concatenate(model_coords_list) if len(model_coords_list) > 0 else np.array([])
    all_model_times = np.concatenate(model_times_list) if len(model_times_list) > 0 else np.array([])
    all_human_coords = np.concatenate(human_coords_list) if len(human_coords_list) > 0 else np.array([])
    all_human_times = np.concatenate(human_times_list) if len(human_times_list) > 0 else np.array([])
    
    all_coords = np.concatenate([all_model_coords, all_human_coords]) if len(all_model_coords) > 0 or len(all_human_coords) > 0 else np.array([])
    all_times = np.concatenate([all_model_times, all_human_times]) if len(all_model_times) > 0 or len(all_human_times) > 0 else np.array([])
    
    if len(all_coords) > 0:
        coord_min, coord_max = all_coords.min(), all_coords.max()
    else:
        coord_min, coord_max = 0.0, 1.0
    
    if len(all_times) > 0:
        time_min, time_max = all_times.min(), all_times.max()
    else:
        time_min, time_max = 0.0, 1.0
    
    return (coord_min, coord_max), (time_min, time_max)


def _calculate_max_velocity_from_trajectories(model_v_list, human_v_list):
    """
    从模型和人类轨迹的速度列表中计算最大速度
    
    Args:
        model_v_list: 模型轨迹的速度序列列表
        human_v_list: 人类轨迹的速度序列列表
        
    Returns:
        vmax: 最大速度值
    """
    all_v = []
    if len(model_v_list) > 0:
        all_model_v = np.concatenate(model_v_list)
        all_v.append(all_model_v)
    if len(human_v_list) > 0:
        all_human_v = np.concatenate(human_v_list)
        all_v.append(all_human_v)
    
    if len(all_v) > 0:
        all_v = np.concatenate(all_v)
        vmax = all_v.max()
    else:
        vmax = 1.0
    
    return vmax


def _build_surface_from_stv(coords_list, times_list, v_list, num_coord_bins=40, num_time_bins=40,
                           coord_range=None, time_range=None):
    """
    根据若干条轨迹的 (coord, time, velocity) 离散点构建曲面数据。
    
    曲面生成方式（在此说明）：
    - 将所有轨迹的 (coord, time, velocity) 点汇总；
    - 在坐标轴和时间轴上分别划分均匀网格；
    - 对于每个 (coord_bin, time_bin) 网格，收集落入该网格的所有速度值，并取其平均值；
    - 如果某个网格没有任何轨迹经过，则该网格保持为空（用 NaN 表示，没有“伪造”的速度值）；
    - 这样得到的速度矩阵 v_surface 即为“在有轨迹经过的时空点上计算平均速度”的离散曲面。
    
    Args:
        coords_list: 各条轨迹的坐标序列列表
        times_list: 各条轨迹的时间序列列表
        v_list: 各条轨迹的速度序列列表
        num_coord_bins: 坐标方向网格数
        num_time_bins: 时间方向网格数
        
    Returns:
        coord_grid, time_grid, v_surface: 用于绘制曲面的二维网格与速度矩阵
    """
    all_coords = np.concatenate(coords_list)
    all_times = np.concatenate(times_list)
    all_v = np.concatenate(v_list)
    
    # 如果指定了范围，使用指定范围；否则使用数据范围
    if coord_range is not None:
        coord_min, coord_max = coord_range
    else:
        coord_min, coord_max = all_coords.min(), all_coords.max()
    
    if time_range is not None:
        time_min, time_max = time_range
    else:
        time_min, time_max = all_times.min(), all_times.max()
    
    coord_edges = np.linspace(coord_min, coord_max, num_coord_bins + 1)
    time_edges = np.linspace(time_min, time_max, num_time_bins + 1)
    
    v_surface = np.zeros((num_time_bins, num_coord_bins), dtype=float)
    count_surface = np.zeros((num_time_bins, num_coord_bins), dtype=int)
    
    coord_indices = np.digitize(all_coords, coord_edges) - 1
    time_indices = np.digitize(all_times, time_edges) - 1
    
    coord_indices = np.clip(coord_indices, 0, num_coord_bins - 1)
    time_indices = np.clip(time_indices, 0, num_time_bins - 1)
    
    for c_idx, t_idx, v in zip(coord_indices, time_indices, all_v):
        v_surface[t_idx, c_idx] += v
        count_surface[t_idx, c_idx] += 1
    
    mask = count_surface > 0
    # ===== 计算每个时空点的平均速度 =====
    # 对有数据的网格取平均速度
    v_surface[mask] /= count_surface[mask]
    # 对没有任何轨迹经过的时空网格，用 NaN 表示“无数据”，从而保持曲面只反映真实经过的时空点
    v_surface[~mask] = 0.0
    
    # ===== 使用高斯滤波平滑曲面，使曲面更加光滑（类似参考图的连续平滑样式） =====
    # 使用较大的 sigma 进行高斯滤波，使曲面更平滑连续
    v_surface = ndimage.gaussian_filter(v_surface, sigma=2.0, mode='nearest')
    
    coord_centers = 0.5 * (coord_edges[:-1] + coord_edges[1:])
    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])
    coord_grid, time_grid = np.meshgrid(coord_centers, time_centers)
    
    return coord_grid, time_grid, v_surface


def plot_space_time_velocity_model(loaded_trajectories, model_name, axis="x",
                                   save_path_lines=None, save_path_surface=None,
                                   num_coord_bins=40, num_time_bins=40,
                                   coord_range=None, time_range=None, vmin=None, vmax=None):
    """
    绘制模型轨迹的“坐标-时间-速度”三维图（曲线 + 曲面）
    
    Args:
        loaded_trajectories: 模型轨迹列表，每个元素为 [N, 4] 的状态序列 [x, y, theta, v]
        model_name: 模型名称，用于推断时间步长和场景信息
        axis: 'x' 或 'y'，选择使用x坐标或y坐标作为空间轴
        save_path_lines: 多条3D曲线图的保存路径（可选）
        save_path_surface: 3D曲面图的保存路径（可选）
        num_coord_bins: 曲面在坐标方向的网格数
        num_time_bins: 曲面在时间方向的网格数
    """
    if len(loaded_trajectories) == 0:
        print("Warning: No model trajectories provided for space-time-velocity plotting.")
        return
    
    axis_label = "X" if axis == "x" else "Y"
    coords_list, times_list, v_list = _prepare_model_stv_data(loaded_trajectories, model_name, axis=axis)
    
    # ===== 图1：若干条轨迹的3D曲线 =====
    fig_lines = plt.figure(figsize=(10, 10))
    ax_lines = fig_lines.add_subplot(111, projection='3d')
    for coords, times, v in zip(coords_list, times_list, v_list):
        ax_lines.plot(coords, times, v, alpha=0.8)
    
    # 设置统一的坐标轴范围
    if coord_range is not None:
        ax_lines.set_xlim(coord_range)
    if time_range is not None:
        ax_lines.set_ylim(time_range)
    if vmin is not None and vmax is not None:
        ax_lines.set_zlim(vmin, vmax)
    
    # 设置坐标轴和网格颜色：坐标系平面为白色/无色，坐标网格为黑色
    ax_lines.xaxis.pane.fill = False
    ax_lines.yaxis.pane.fill = False
    ax_lines.zaxis.pane.fill = False
    ax_lines.xaxis.pane.set_edgecolor('black')
    ax_lines.yaxis.pane.set_edgecolor('black')
    ax_lines.zaxis.pane.set_edgecolor('black')
    ax_lines.xaxis.pane.set_alpha(1.0)
    ax_lines.yaxis.pane.set_alpha(1.0)
    ax_lines.zaxis.pane.set_alpha(1.0)
    ax_lines.grid(True, color='black', linestyle='-', linewidth=0.5)
    
    ax_lines.set_xlabel(f'{axis_label} (m)', fontsize=14)
    ax_lines.set_ylabel('Time (s)', fontsize=14)
    ax_lines.set_zlabel('Velocity (m/s)', fontsize=14)
    ax_lines.set_title(f'Model Trajectories: {axis_label}-Time-Velocity', fontsize=16)
    
    # 根据model_name决定是否翻转空间轴（x轴）
    if "sce1" not in model_name:
        ax_lines.invert_xaxis()
    
    plt.tight_layout()
    if save_path_lines is not None:
        plt.savefig(save_path_lines, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Model space-time-velocity (lines) plot saved to: {save_path_lines}")
    # plt.show()
    
    # ===== 图2：通过网格平均形成的3D曲面 =====
    coord_grid, time_grid, v_surface = _build_surface_from_stv(
        coords_list, times_list, v_list,
        num_coord_bins=num_coord_bins,
        num_time_bins=num_time_bins,
        coord_range=coord_range,
        time_range=time_range
    )
    
    fig_surf = plt.figure(figsize=(10, 10))
    ax_surf = fig_surf.add_subplot(111, projection='3d')
    # 绘制曲面：曲面上不绘制网格线，使用平滑着色（类似参考图样式）
    plot_kwargs = {
        'cmap': 'viridis',
        'edgecolor': 'none',   # 曲面上不绘制网格线
        'linewidth': 0,
        'antialiased': True,
        'shade': True,
        'alpha': 0.95
    }
    if vmin is not None and vmax is not None:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax
    surf = ax_surf.plot_surface(coord_grid, time_grid, v_surface, **plot_kwargs)
    
    # 根据曲面绘制两条边际最高（最大值）投影曲线，投影到 xOz（坐标-速度）和 yOz（时间-速度）平面
    coord_centers_1d = coord_grid[0, :]
    time_centers_1d = time_grid[:, 0]
    max_v_over_time = v_surface.max(axis=0)   # 对时间取最大 → 坐标-速度
    max_v_over_coord = v_surface.max(axis=1)  # 对坐标取最大 → 时间-速度
    time_max = time_grid.max()
    coord_min = coord_grid.min()
    # xOz 平面上的边际最高曲线（放在 y=time_max 的背面）
    ax_surf.plot(coord_centers_1d, np.full_like(coord_centers_1d, time_max), max_v_over_time,
                 color='#C41E3A', linewidth=2, label='Max (coord-vel)')
    # yOz 平面上的边际最高曲线（放在 x=coord_min 的侧面）
    ax_surf.plot(np.full_like(time_centers_1d, coord_min), time_centers_1d, max_v_over_coord,
                 color='#E67E22', linewidth=2, label='Max (time-vel)')
    
    # colorbar放在右侧
    cbar = fig_surf.colorbar(surf, ax=ax_surf, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Velocity (m/s)', fontsize=14)
    
    # 设置统一的坐标轴范围
    if coord_range is not None:
        ax_surf.set_xlim(coord_range)
    if time_range is not None:
        ax_surf.set_ylim(time_range)
    if vmin is not None and vmax is not None:
        ax_surf.set_zlim(vmin, vmax)
    
    # 设置坐标轴和网格颜色：坐标系平面为白色/无色，坐标网格为黑色
    ax_surf.xaxis.pane.fill = False
    ax_surf.yaxis.pane.fill = False
    ax_surf.zaxis.pane.fill = False
    ax_surf.xaxis.pane.set_edgecolor('black')
    ax_surf.yaxis.pane.set_edgecolor('black')
    ax_surf.zaxis.pane.set_edgecolor('black')
    ax_surf.xaxis.pane.set_alpha(1.0)
    ax_surf.yaxis.pane.set_alpha(1.0)
    ax_surf.zaxis.pane.set_alpha(1.0)
    ax_surf.grid(True, color='black', linestyle='-', linewidth=0.5)
    
    ax_surf.set_xlabel(f'{axis_label} (m)', fontsize=14)
    ax_surf.set_ylabel('Time (s)', fontsize=14)
    ax_surf.set_zlabel('Velocity (m/s)', fontsize=14)
    ax_surf.set_title(f'Model Trajectories Surface: {axis_label}-Time-Velocity', fontsize=16)
    
    # 根据model_name决定是否翻转空间轴（x轴）
    if "sce1" not in model_name:
        ax_surf.invert_xaxis()
    
    plt.tight_layout()
    if save_path_surface is not None:
        plt.savefig(save_path_surface, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Model space-time-velocity (surface) plot saved to: {save_path_surface}")
    # plt.show()
    
    return v_surface  # 返回速度曲面


def plot_space_time_velocity_human(human_trajectories, model_name, axis="x",
                                   save_path_lines=None, save_path_surface=None,
                                   num_coord_bins=40, num_time_bins=40,
                                   coord_range=None, time_range=None, vmin=None, vmax=None):
    """
    绘制人类轨迹的“坐标-时间-速度”三维图（曲线 + 曲面）
    
    Args:
        human_trajectories: 人类轨迹列表，每个元素为 [N, 3] 的状态序列 [x, y, t]
        model_name: 模型名称，仅用于命名和保持接口一致
        axis: 'x' 或 'y'，选择使用x坐标或y坐标作为空间轴
        save_path_lines: 多条3D曲线图的保存路径（可选）
        save_path_surface: 3D曲面图的保存路径（可选）
        num_coord_bins: 曲面在坐标方向的网格数
        num_time_bins: 曲面在时间方向的网格数
    """
    if len(human_trajectories) == 0:
        print("Warning: No human trajectories provided for space-time-velocity plotting.")
        return
    
    axis_label = "X" if axis == "x" else "Y"
    coords_list, times_list, v_list = _prepare_human_stv_data(human_trajectories, axis=axis)
    
    # ===== 图1：若干条轨迹的3D曲线 =====
    fig_lines = plt.figure(figsize=(10, 10))
    ax_lines = fig_lines.add_subplot(111, projection='3d')
    for coords, times, v in zip(coords_list, times_list, v_list):
        ax_lines.plot(coords, times, v, alpha=0.8)
    
    # 设置统一的坐标轴范围
    if coord_range is not None:
        ax_lines.set_xlim(coord_range)
    if time_range is not None:
        ax_lines.set_ylim(time_range)
    if vmin is not None and vmax is not None:
        ax_lines.set_zlim(vmin, vmax)
    
    # 设置坐标轴和网格颜色：坐标系平面为白色/无色，坐标网格为黑色
    ax_lines.xaxis.pane.fill = False
    ax_lines.yaxis.pane.fill = False
    ax_lines.zaxis.pane.fill = False
    ax_lines.xaxis.pane.set_edgecolor('black')
    ax_lines.yaxis.pane.set_edgecolor('black')
    ax_lines.zaxis.pane.set_edgecolor('black')
    ax_lines.xaxis.pane.set_alpha(1.0)
    ax_lines.yaxis.pane.set_alpha(1.0)
    ax_lines.zaxis.pane.set_alpha(1.0)
    ax_lines.grid(True, color='black', linestyle='-', linewidth=0.5)
    
    ax_lines.set_xlabel(f'{axis_label} (m)', fontsize=14)
    ax_lines.set_ylabel('Time (s)', fontsize=14)
    ax_lines.set_zlabel('Velocity (m/s)', fontsize=14)
    ax_lines.set_title(f'Human Trajectories: {axis_label}-Time-Velocity', fontsize=16)
    
    # 根据model_name决定是否翻转空间轴（x轴）
    if "sce1" not in model_name:
        ax_lines.invert_xaxis()
    
    plt.tight_layout()
    if save_path_lines is not None:
        plt.savefig(save_path_lines, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Human space-time-velocity (lines) plot saved to: {save_path_lines}")
    # plt.show()
    
    # ===== 图2：通过网格平均形成的3D曲面 =====
    coord_grid, time_grid, v_surface = _build_surface_from_stv(
        coords_list, times_list, v_list,
        num_coord_bins=num_coord_bins,
        num_time_bins=num_time_bins,
        coord_range=coord_range,
        time_range=time_range
    )
    
    fig_surf = plt.figure(figsize=(10, 10))
    ax_surf = fig_surf.add_subplot(111, projection='3d')
    # 绘制曲面：曲面上不绘制网格线，使用平滑着色（类似参考图样式）
    plot_kwargs = {
        'cmap': 'viridis',
        'edgecolor': 'none',   # 曲面上不绘制网格线
        'linewidth': 0,
        'antialiased': True,
        'shade': True,
        'alpha': 0.95
    }
    if vmin is not None and vmax is not None:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax
    surf = ax_surf.plot_surface(coord_grid, time_grid, v_surface, **plot_kwargs)
    
    # 根据曲面绘制两条边际最高（最大值）投影曲线，投影到 xOz（坐标-速度）和 yOz（时间-速度）平面
    coord_centers_1d = coord_grid[0, :]
    time_centers_1d = time_grid[:, 0]
    max_v_over_time = v_surface.max(axis=0)
    max_v_over_coord = v_surface.max(axis=1)
    time_max = time_grid.max()
    coord_min = coord_grid.min()
    ax_surf.plot(coord_centers_1d, np.full_like(coord_centers_1d, time_max), max_v_over_time,
                 color='#C41E3A', linewidth=2, label='Max (coord-vel)')
    ax_surf.plot(np.full_like(time_centers_1d, coord_min), time_centers_1d, max_v_over_coord,
                 color='#E67E22', linewidth=2, label='Max (time-vel)')
    
    # colorbar放在右侧
    cbar = fig_surf.colorbar(surf, ax=ax_surf, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Velocity (m/s)', fontsize=14)
    
    # 设置统一的坐标轴范围
    if coord_range is not None:
        ax_surf.set_xlim(coord_range)
    if time_range is not None:
        ax_surf.set_ylim(time_range)
    if vmin is not None and vmax is not None:
        ax_surf.set_zlim(vmin, vmax)
    
    # 设置坐标轴和网格颜色：坐标系平面为白色/无色，坐标网格为黑色
    ax_surf.xaxis.pane.fill = False
    ax_surf.yaxis.pane.fill = False
    ax_surf.zaxis.pane.fill = False
    ax_surf.xaxis.pane.set_edgecolor('black')
    ax_surf.yaxis.pane.set_edgecolor('black')
    ax_surf.zaxis.pane.set_edgecolor('black')
    ax_surf.xaxis.pane.set_alpha(1.0)
    ax_surf.yaxis.pane.set_alpha(1.0)
    ax_surf.zaxis.pane.set_alpha(1.0)
    ax_surf.grid(True, color='black', linestyle='-', linewidth=0.5)
    
    ax_surf.set_xlabel(f'{axis_label} (m)', fontsize=14)
    ax_surf.set_ylabel('Time (s)', fontsize=14)
    ax_surf.set_zlabel('Velocity (m/s)', fontsize=14)
    ax_surf.set_title(f'Human Trajectories Surface: {axis_label}-Time-Velocity', fontsize=16)
    
    # 根据model_name决定是否翻转空间轴（x轴）
    if "sce1" not in model_name:
        ax_surf.invert_xaxis()
    
    plt.tight_layout()
    if save_path_surface is not None:
        plt.savefig(save_path_surface, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Human space-time-velocity (surface) plot saved to: {save_path_surface}")
    # plt.show()
    
    return v_surface  # 返回速度曲面


def calculate_surface_rmse(model_v_surface, human_v_surface, include_zero_velocity=True):
    """
    计算模型轨迹曲面和人类轨迹曲面的点对点均方误差（各个时空点平均速度的均方误差）
    
    Args:
        model_v_surface: 模型轨迹的速度曲面矩阵 [num_time_bins, num_coord_bins]
        human_v_surface: 人类轨迹的速度曲面矩阵 [num_time_bins, num_coord_bins]
        include_zero_velocity: bool变量，控制计算范围
            - True: 计算整个坐标范围内的所有点
            - False: 只计算z轴方向值不为0的点（即速度不为0的点）
        
    Returns:
        rmse: 均方根误差值
        num_points: 参与计算的点的数量
    """
    if model_v_surface.shape != human_v_surface.shape:
        raise ValueError(f"Surface shapes do not match: model {model_v_surface.shape} vs human {human_v_surface.shape}")
    
    # 计算差值
    diff = model_v_surface - human_v_surface
    
    # 根据include_zero_velocity决定计算范围
    if include_zero_velocity:
        # 计算整个坐标范围内的所有点
        mask = np.ones_like(model_v_surface, dtype=bool)
    else:
        # 只计算z轴方向值不为0的点（即速度不为0的点）
        # 如果模型或人类轨迹中至少有一个在该点的速度不为0，则包含该点
        mask = (model_v_surface != 0.0) | (human_v_surface != 0.0)
    
    # 提取有效点的差值
    valid_diff = diff[mask]
    
    if len(valid_diff) == 0:
        print("Warning: No valid points found for RMSE calculation.")
        return 0.0, 0
    
    # 计算均方误差
    mse = np.mean(valid_diff ** 2)
    rmse = np.sqrt(mse)
    num_points = len(valid_diff)
    print(f"{num_points} valid points, RMSE: {rmse:.8f}")
    
    return rmse, num_points

