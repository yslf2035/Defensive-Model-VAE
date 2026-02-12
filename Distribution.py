"""
批量处理轨迹生成和MPC跟踪，并绘制速度分布对比图
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from MPC.MPC_Tracking import PathTracker
from Tools import get_start_conditions_from_csv, load_model_and_generate_trajectory, get_human_and_bv_trajectories
import math
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import entropy
from Spatial_Distribution import *

# 全局字体配置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


def collect_csv_files(base_folder):
    """
    收集指定文件夹下所有子文件夹中的CSV文件
    
    Args:
        base_folder: 基础文件夹路径
        
    Returns:
        csv_files: CSV文件路径列表
    """
    csv_files = []
    subfolders = ['减速', '减速+转向', '转向']
    # subfolders = ['减速+转向']

    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        if os.path.exists(folder_path):
            csv_pattern = os.path.join(folder_path, '*.csv')
            files = glob.glob(csv_pattern)
            csv_files.extend(files)

    if len(csv_files) == 0:
        print("No CSV files found!")
        return None

    print(f"Total CSV files: {len(csv_files)}")
    return csv_files


def process_single_trajectory(csv_path, model_path, seq_len=10, dim=3, latent_dim=8, device='cpu'):
    """
    处理单个CSV文件：生成轨迹并使用MPC跟踪
    
    Args:
        csv_path: CSV文件路径
        model_path: VAE模型路径
        seq_len: 轨迹长度
        dim: 每个点的维度
        latent_dim: 潜在空间维度
        device: 计算设备
        
    Returns:
        tracked_trajectory: 跟踪后的轨迹状态序列 [N, 4] - [x, y, theta, v]
        times: 时间序列 [N]
    """
    try:
        model_name = os.path.basename(model_path)
        # 获取起始条件
        start_x, start_y, start_angle, start_vx, start_vy = get_start_conditions_from_csv(csv_path, model_name)
        
        # VAE模型生成轨迹点 [t, x, y]
        waypoints = load_model_and_generate_trajectory(
            model_path, start_x, start_y, seq_len, dim, latent_dim, device
        )
        # 转换为 [x, y, t] 格式
        waypoints = waypoints[:, [1, 2, 0]]
        waypoints[0, 2] = 0.0

        # 初始状态 [x, y, theta, vx, vy]
        initial_state = np.array([start_x, start_y, start_angle, start_vx, start_vy])

        if "sce1" in model_name:
            time_step = 0.02
        elif "sce2" in model_name:
            time_step = 0.025
        elif "sce3" in model_name:
            time_step = 0.015
        elif "sce4" in model_name:
            time_step = 0.02
        else:
            time_step = 0.02
        # 创建路径跟踪器
        tracker = PathTracker(
            waypoints=waypoints,
            initial_state=initial_state,
            wheelbase=2.8,
            prediction_horizon=30,
            control_horizon=20,
            dt=time_step
        )
        
        # 运行仿真
        total_time = waypoints[-1, -1]
        times, states, controls = tracker.run_simulation(total_time=total_time)
        
        return states, times
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None, None


def batch_process_trajectories(csv_files, model_path, seq_len=10, dim=3, latent_dim=8, device='cpu'):
    """
    使用模型批量为所有CSV文件生成轨迹，并保存每条轨迹为单独的npy文件
    
    Args:
        csv_files: CSV文件路径列表
        model_path: VAE模型路径
        seq_len: 轨迹长度
        dim: 每个点的维度
        latent_dim: 潜在空间维度
        device: 计算设备
        
    Returns:
        all_trajectories: 所有跟踪后的轨迹列表，每个元素是 [N, 4] 的状态序列
        all_times: 所有时间序列列表
        saved_files: 保存的文件路径列表
    """
    model_name = os.path.basename(model_path)
    model_name_parts = model_name.split('_')

    all_trajectories = []
    all_times = []
    saved_files = []
    
    # 创建保存目录
    save_dir = 'results/GeneratedData'
    
    print(f"\n开始批量处理 {len(csv_files)} 个CSV文件...")
    
    for i, csv_path in enumerate(csv_files):
        print(f"\n处理文件 {i+1}/{len(csv_files)}: {os.path.basename(csv_path)}")
        csv_name = os.path.basename(csv_path)
        csv_name_parts = csv_name.split('_')

        states, times = process_single_trajectory(
            csv_path, model_path, seq_len, dim, latent_dim, device
        )
        
        if states is not None and times is not None:
            all_trajectories.append(states)
            all_times.append(times)
            
            # 保存单条轨迹为npy文件
            npy_filename = f"tracked_trajectory_{model_name_parts[2]}_exp{csv_name_parts[1]}_{csv_name_parts[-1].split('.')[0]}.npy"
            npy_filepath = os.path.join(save_dir, npy_filename)
            np.save(npy_filepath, states)
            saved_files.append(npy_filepath)
            print(f"  已保存轨迹到: {npy_filepath}")
        else:
            print(f"跳过文件 {csv_path}（处理失败）")
    
    print(f"\n成功处理 {len(all_trajectories)} 条轨迹")
    return all_trajectories, all_times, saved_files


def load_tracked_trajectories_from_files(saved_files):
    """
    从保存的npy文件中加载所有轨迹
    
    Args:
        saved_files: 保存的npy文件路径列表
        
    Returns:
        trajectories: 所有轨迹列表，每个元素是 [N, 4] 的状态序列 [x, y, theta, v]
    """
    trajectories = []
    
    print(f"Loading {len(saved_files)} trajectory files...")
    for i, filepath in enumerate(saved_files):
        if os.path.exists(filepath):
            traj = np.load(filepath)
            trajectories.append(traj)
            if (i + 1) % 10 == 0 or i == len(saved_files) - 1:
                print(f"Loaded {i + 1}/{len(saved_files)} files")
        else:
            print(f"Warning: File not found: {filepath}")
    
    print(f"Successfully loaded {len(trajectories)} trajectories")
    return trajectories


def extract_velocities_from_trajectories(trajectories):
    """
    从轨迹状态序列中提取所有时刻的速度
    
    Args:
        trajectories: 轨迹列表，每个元素是 [N, 4] 的状态序列 [x, y, theta, v]
        
    Returns:
        velocities: 所有时刻的速度值（一维数组）
    """
    velocities = []
    for traj in trajectories:
        # 状态序列的第4列是速度
        v = traj[:, 3]
        velocities.extend(v.tolist())
    return np.array(velocities)


def load_human_trajectories(csv_files, model_name):
    """
    加载人类轨迹数据
    
    Args:
        csv_files: csv文件列
        model_name: 模型名称
        
    Returns:
        human_trajectories: 轨迹数组 (num_samples, target_points, 3) - [时间, x, y]
    """
    human_trajectories = []
    for i, csv_path in enumerate(csv_files):
        print(f"\n处理文件 {i + 1}/{len(csv_files)}: {os.path.basename(csv_path)}")
        human_trajectory, _, _ = get_human_and_bv_trajectories(csv_path, model_name)
        if "sce1" in model_name:
            mask = human_trajectory[:, 1] >= 40
        elif "sce2" in model_name:
            mask = human_trajectory[:, 0] >= 40
        elif "sce4" in model_name:
            mask = human_trajectory[:, 0] < 9
        else:
            mask = human_trajectory[:, 1] <= 40
        first_index = np.argmax(mask) if np.any(mask) else 0
        human_trajectory_1 = human_trajectory[first_index:]
        human_trajectories.append(human_trajectory_1)
    return human_trajectories


def calculate_human_velocities(human_trajectories):
    """
    从人类轨迹数据计算速度
    
    Args:
        human_trajectories: 轨迹数组 (num_samples, target_points, 3) - [时间, x, y]
        
    Returns:
        velocities: 所有时刻的速度值（一维数组）
    """
    velocities = []
    
    for traj in human_trajectories:
        # traj shape: [x, y, t]
        times = traj[:, 2]
        x_coords = traj[:, 0]
        y_coords = traj[:, 1]
        
        # 计算速度（通过相邻点之间的距离和时间差）
        for i in range(len(traj) - 1):
            dt = times[i + 1] - times[i]
            if dt > 1e-6:  # 避免除零错误
                dx = x_coords[i + 1] - x_coords[i]
                dy = y_coords[i + 1] - y_coords[i]
                v = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(v)
            else:
                # 如果时间差为0，使用前一个速度或0
                if len(velocities) > 0:
                    velocities.append(velocities[-1])
                else:
                    velocities.append(0.0)
        
        # 对于最后一点，使用前一个速度值
        if len(traj) > 1:
            dt = times[-1] - times[-2]
            if dt > 1e-6:
                dx = x_coords[-1] - x_coords[-2]
                dy = y_coords[-1] - y_coords[-2]
                v = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(v)
            else:
                # 如果时间差为0，使用前一个速度或0
                if len(velocities) > 0:
                    velocities.append(velocities[-1])
                else:
                    velocities.append(0.0)
    
    return np.array(velocities)


def plot_velocity_distribution(generated_velocities, human_velocities, save_path=None):
    """
    绘制速度分布对比图
    
    Args:
        generated_velocities: 生成轨迹的速度数组
        human_velocities: 人类轨迹的速度数组
        save_path: 保存路径（可选）
    """
    # 计算JS散度
    # 创建统一的bins用于计算分布
    v_min = min(np.min(generated_velocities), np.min(human_velocities))
    v_max = max(np.max(generated_velocities), np.max(human_velocities))
    bins_js = np.linspace(v_min, v_max, 50)
    
    # 计算直方图（使用counts，不归一化）
    hist_gen, _ = np.histogram(generated_velocities, bins=bins_js)
    hist_human, _ = np.histogram(human_velocities, bins=bins_js)
    
    # 归一化为概率分布（确保和为1）
    hist_gen = hist_gen / (hist_gen.sum() + 1e-10)
    hist_human = hist_human / (hist_human.sum() + 1e-10)
    
    # 计算中间分布 M = 0.5 * (P + Q)
    M = 0.5 * (hist_gen + hist_human)
    
    # 计算JS散度 = 0.5 * (KL(P||M) + KL(Q||M))
    # 使用base=2的对数，使JS散度范围在[0,1]
    # 添加小的epsilon避免log(0)
    epsilon = 1e-10
    kl_pm = entropy(hist_gen + epsilon, M + epsilon, base=2)
    kl_qm = entropy(hist_human + epsilon, M + epsilon, base=2)
    js_divergence = 0.5 * (kl_pm + kl_qm)
    
    print(f"\nVelocity Jensen-Shannon Divergence: {js_divergence:.6f} (range: [0, 1])")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图对比
    ax1 = axes[0]
    bins = np.linspace(0, max(np.max(generated_velocities), np.max(human_velocities)), 50)
    
    ax1.hist(generated_velocities, bins=bins, alpha=0.6, label=f'Model (n={len(generated_velocities)})',
             color=(0, 0.4470, 0.7410), density=True)
    ax1.hist(human_velocities, bins=bins, alpha=0.6, label=f'Human (n={len(human_velocities)})', 
             color=(0.7961, 0.1255, 0.1765), density=True)
    
    ax1.set_xlabel('Velocity (m/s)', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.set_title('Velocity Distribution Comparison', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 右图：箱线图对比
    ax2 = axes[1]
    data_to_plot = [generated_velocities, human_velocities]
    bp = ax2.boxplot(data_to_plot, labels=['Model', 'Human'], patch_artist=True)
    
    # 设置箱线图颜色
    colors = [(0, 0.4470, 0.7410), (0.7961, 0.1255, 0.1765)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
    
    ax2.set_ylabel('Velocity (m/s)', fontsize=14)
    ax2.set_title('Velocity Distribution Statistics', fontsize=16)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    gen_mean = np.mean(generated_velocities)
    gen_std = np.std(generated_velocities)
    human_mean = np.mean(human_velocities)
    human_std = np.std(human_velocities)
    
    stats_text = f'Model: μ={gen_mean:.2f}, σ={gen_std:.2f}\nHuman: μ={human_mean:.2f}, σ={human_std:.2f}'
    ax2.text(0.5, 0.95, stats_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Velocity distribution plot saved to: {save_path}")
    
    # plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("Batch Trajectory Generation and MPC Tracking")
    print("=" * 60)
    
    # 配置参数
    base_folder = 'DefensiveData/PredictableMovementTown05'
    model_path = 'training/models/vae_offset_sce3_cond_ld8_epoch3000.pth'
    model_name = os.path.basename(model_path)
    model_name_parts = model_name.split('_')
    seq_len = 10  # seq_len=10
    dim = 3
    latent_dim = 8
    device = 'cpu'
    human_trajectory_path = 'training/DefensiveDataProcessed/trajectory_' + model_name_parts[2] + '.npy'
    
    # # 收集所有CSV文件
    # print("\n[Step 1] Collecting CSV files...")
    # csv_files = collect_csv_files(base_folder)
    #
    # # 批量处理轨迹生成和MPC跟踪（每条轨迹会自动保存为单独的npy文件）
    # print("\n[Step 2] Batch processing trajectories...")
    # all_trajectories, all_times, saved_files = batch_process_trajectories(
    #     csv_files, model_path, seq_len, dim, latent_dim, device
    # )

    # 已保存模型生成轨迹数据npy文件时启用
    # # sce1
    # csv_files = ["DefensiveData/StaticBlindTown05/减速+转向/exp_60_control_StaticBlindTown05_3.csv",
    #              "DefensiveData/StaticBlindTown05/减速/exp_11_control_StaticBlindTown05_3.csv",
    #              "DefensiveData/StaticBlindTown05/减速/exp_13_control_StaticBlindTown05_3.csv",
    #              "DefensiveData/StaticBlindTown05/减速/exp_17_control_StaticBlindTown05_2.csv",
    #              "DefensiveData/StaticBlindTown05/减速/exp_17_control_StaticBlindTown05_3.csv"]
    # saved_files = ["results/GeneratedData/tracked_trajectory_sce1_exp60_3.npy",
    #                "results/GeneratedData/tracked_trajectory_sce1_exp11_3.npy",
    #                "results/GeneratedData/tracked_trajectory_sce1_exp13_3.npy",
    #                "results/GeneratedData/tracked_trajectory_sce1_exp17_2.npy",
    #                "results/GeneratedData/tracked_trajectory_sce1_exp17_3.npy"]
    # # sce3
    # csv_files = ["DefensiveData/PredictableMovementTown05/减速+转向/exp_34_control_PredictableMovementTown05_3.csv",
    #              "DefensiveData/PredictableMovementTown05/减速+转向/exp_44_control_PredictableMovementTown05_2.csv",
    #              "DefensiveData/PredictableMovementTown05/减速+转向/exp_58_control_PredictableMovementTown05_1.csv",
    #              "DefensiveData/PredictableMovementTown05/减速+转向/exp_62_control_PredictableMovementTown05_3.csv"]
    # saved_files = ["results/GeneratedData/tracked_trajectory_sce3_exp34_3.npy",
    #                "results/GeneratedData/tracked_trajectory_sce3_exp44_2.npy",
    #                "results/GeneratedData/tracked_trajectory_sce3_exp58_1.npy",
    #                "results/GeneratedData/tracked_trajectory_sce3_exp62_3.npy"]

    # 已保存模型生成轨迹数据npy文件时启用
    csv_folders = [
        "DefensiveData/PredictableMovementTown05/减速/",
        "DefensiveData/PredictableMovementTown05/转向/",
        "DefensiveData/PredictableMovementTown05/减速+转向/"
    ]
    csv_files = []
    # 遍历每个目标文件夹
    for folder_path in csv_folders:
        # 检查文件夹是否存在
        if os.path.isdir(folder_path):
            # 遍历文件夹内所有文件/文件夹
            for filename in os.listdir(folder_path):
                # 筛选出.csv后缀的文件
                if filename.lower().endswith(".csv"):  # lower()避免大小写问题（如.CSV）
                    # 拼接完整路径
                    full_path = os.path.join(folder_path, filename)
                    # 替换Windows下的\为/（保证路径格式和示例一致）
                    full_path = full_path.replace("\\", "/")
                    csv_files.append(full_path)
        else:
            print(f"警告：文件夹 {folder_path} 不存在或不是有效目录")

    saved_folder = "results/GeneratedData/"
    saved_files = []
    # 检查文件夹是否存在
    if os.path.isdir(saved_folder):
        # 遍历文件夹内所有文件
        for filename in os.listdir(saved_folder):
            # 筛选条件：文件名包含"sceX" + 后缀是.npy
            if "sce3" in filename and filename.lower().endswith(".npy"):
                # 拼接完整路径并统一分隔符为/
                full_path = os.path.join(saved_folder, filename)
                full_path = full_path.replace("\\", "/")
                saved_files.append(full_path)
    else:
        print(f"警告：文件夹 {saved_folder} 不存在或不是有效目录")

    # 从保存的npy文件中加载轨迹数据
    print("\n[Step 3] Loading tracked trajectories from saved files...")
    loaded_trajectories = load_tracked_trajectories_from_files(saved_files)

    # 提取速度数据
    print("\n[Step 4] Extracting velocities...")
    generated_velocities = extract_velocities_from_trajectories(loaded_trajectories)
    print(f"Generated trajectories: {len(generated_velocities)} velocity samples")
    print(f"  Mean: {np.mean(generated_velocities):.2f} m/s")
    print(f"  Std: {np.std(generated_velocities):.2f} m/s")

    # 加载人类轨迹并计算速度
    print("\n[Step 5] Loading human trajectories...")
    human_trajectories = load_human_trajectories(csv_files, model_name)
    human_velocities = calculate_human_velocities(human_trajectories)
    print(f"Human trajectories: {len(human_velocities)} samples")
    print(f"  Mean: {np.mean(human_velocities):.2f} m/s")
    print(f"  Std: {np.std(human_velocities):.2f} m/s")
    
    # 绘制速度分布对比图
    print("\n[Step 6] Plotting velocity distribution...")
    plot_save_path = 'results/ModelValidation/velocity_distribution_comparison_' + model_name_parts[2] + '.png'
    # plot_velocity_distribution(generated_velocities, human_velocities, plot_save_path)
    
    # 提取坐标点并绘制空间分布热力图
    print("\n[Step 7] Extracting coordinates and plotting spatial distribution...")
    
    # 提取生成轨迹的坐标
    generated_coordinates = extract_coordinates_from_trajectories(loaded_trajectories)
    print(f"Generated trajectories: {len(generated_coordinates)} coordinate points")
    
    # 提取人类轨迹的坐标
    human_coordinates = extract_human_coordinates(human_trajectories)
    print(f"Human trajectories: {len(human_coordinates)} coordinate points")
    
    # 网格边长
    grid_size = 0.5
    # 自定义颜色映射：值小为天蓝色，值大为深红色
    custom_cmap = LinearSegmentedColormap.from_list(
        'sky_to_darkred', ['#87CEEB', '#D4E8F0', '#FFFFE0', '#FFD260', '#FF6E30', '#F04228', '#CD2626']
    )
    
    # # ========== 统计轨迹在方格中所占频次 ==========
    # # 计算统一的colorbar范围
    # print("\nCalculating unified colorbar range...")
    # vmin, vmax = calculate_unified_colorbar_range(
    #     [generated_coordinates, human_coordinates],
    #     model_name,
    #     grid_size=grid_size
    # )
    # print(f"Unified colorbar range: [{vmin:.0f}, {vmax:.0f}]")
    #
    # # 计算空间分布的RMSE_frequency差异指标
    # print("\nCalculating spatial distribution RMSE_frequency...")
    # rmse_frequency = calculate_rmse_frequency(
    #     generated_coordinates,
    #     human_coordinates,
    #     model_name,
    #     grid_size=grid_size
    # )
    #
    # # 绘制生成轨迹的空间分布
    # print("\nPlotting generated trajectories spatial distribution...")
    # gen_spatial_save_path = 'results/ModelValidation/generated_trajectories_spatial_distribution_' + model_name_parts[2] + '.png'
    # _ = plot_spatial_distribution(
    #     generated_coordinates,
    #     'Model Trajectories Spatial Distribution',
    #     model_name,
    #     gen_spatial_save_path,
    #     grid_size=grid_size,
    #     cmap=custom_cmap,
    #     vmin=vmin,
    #     vmax=vmax
    # )
    #
    # # 绘制人类轨迹的空间分布
    # print("\nPlotting human trajectories spatial distribution...")
    # human_spatial_save_path = 'results/ModelValidation/human_trajectories_spatial_distribution_' + model_name_parts[2] + '.png'
    # _ = plot_spatial_distribution(
    #     human_coordinates,
    #     'Human Trajectories Spatial Distribution',
    #     model_name,
    #     human_spatial_save_path,
    #     grid_size=grid_size,
    #     cmap=custom_cmap,
    #     vmin=vmin,
    #     vmax=vmax
    # )
    
    # # ========== 新统计方式：按轨迹统计（每条轨迹经过某个方格只计数1次） ==========
    # print("\n[Step 8] Processing spatial distribution with new method (trajectory-based counting)...")
    #
    # # 计算统一的colorbar范围（新统计方式）
    # print("\nCalculating unified colorbar range (new method)...")
    # vmin_new, vmax_new = calculate_unified_colorbar_range_new(
    #     [loaded_trajectories, human_trajectories],
    #     model_name,
    #     grid_size=grid_size
    # )
    # print(f"Unified colorbar range (new method): [{vmin_new:.0f}, {vmax_new:.0f}]")
    #
    # # 计算空间分布的RMSE_frequency差异指标（新统计方式）
    # print("\nCalculating spatial distribution RMSE_frequency (new method)...")
    # rmse_frequency_new = calculate_rmse_frequency_new(
    #     loaded_trajectories,
    #     human_trajectories,
    #     model_name,
    #     grid_size=grid_size
    # )
    #
    # # 绘制生成轨迹的空间分布（新统计方式）
    # print("\nPlotting generated trajectories spatial distribution (new method)...")
    # gen_spatial_save_path_new = 'results/ModelValidation/generated_trajectories_spatial_distribution_new_' + model_name_parts[2] + '.png'
    # _ = plot_spatial_distribution_new(
    #     loaded_trajectories,
    #     'Model Trajectories Spatial Distribution (Trajectory-based)',
    #     model_name,
    #     gen_spatial_save_path_new,
    #     grid_size=grid_size,
    #     cmap=custom_cmap,
    #     vmin=vmin_new,
    #     vmax=vmax_new
    # )
    #
    # # 绘制人类轨迹的空间分布（新统计方式）
    # print("\nPlotting human trajectories spatial distribution (new method)...")
    # human_spatial_save_path_new = 'results/ModelValidation/human_trajectories_spatial_distribution_new_' + model_name_parts[2] + '.png'
    # _ = plot_spatial_distribution_new(
    #     human_trajectories,
    #     'Human Trajectories Spatial Distribution (Trajectory-based)',
    #     model_name,
    #     human_spatial_save_path_new,
    #     grid_size=grid_size,
    #     cmap=custom_cmap,
    #     vmin=vmin_new,
    #     vmax=vmax_new
    # )

    # ========== 新增：绘制模型/人类轨迹的三维坐标-时间-速度图 ==========
    print("\n[Step 9] Plotting 3D space-time-velocity figures...")
    axis = "y"
    model_stv_lines_path = 'results/ModelValidation/model_space_time_velocity_lines_' + model_name_parts[2] + '.png'
    model_stv_surface_path = 'results/ModelValidation/model_space_time_velocity_projection_' + model_name_parts[2] + '.png'
    human_stv_lines_path = 'results/ModelValidation/human_space_time_velocity_lines_' + model_name_parts[2] + '.png'
    human_stv_surface_path = 'results/ModelValidation/human_space_time_velocity_projection_' + model_name_parts[2] + '.png'

    # 准备数据以计算统一的坐标轴范围和速度范围
    from Spatial_Distribution import _prepare_model_stv_data, _prepare_human_stv_data, _calculate_unified_axes_ranges, _calculate_max_velocity_from_trajectories
    
    model_coords_list, model_times_list, model_v_list = _prepare_model_stv_data(loaded_trajectories, model_name, axis=axis)
    human_coords_list, human_times_list, human_v_list = _prepare_human_stv_data(human_trajectories, axis=axis)
    
    # 计算统一的坐标轴范围
    coord_range, time_range = _calculate_unified_axes_ranges(
        model_coords_list, model_times_list,
        human_coords_list, human_times_list
    )
    
    # 计算统一的速度范围：vmin=0，vmax从轨迹中获取最大速度
    vmin = 0.0
    vmax = _calculate_max_velocity_from_trajectories(model_v_list, human_v_list)
    
    # 绘制模型轨迹
    model_v_surface = plot_space_time_velocity_model(
        loaded_trajectories,
        model_name,
        axis=axis,
        save_path_lines=model_stv_lines_path,
        save_path_surface=model_stv_surface_path,
        num_coord_bins=40,
        num_time_bins=40,
        coord_range=coord_range,
        time_range=time_range,
        vmin=vmin,
        vmax=vmax
    )
    
    # 绘制人类轨迹
    human_v_surface = plot_space_time_velocity_human(
        human_trajectories,
        model_name,
        axis=axis,
        save_path_lines=human_stv_lines_path,
        save_path_surface=human_stv_surface_path,
        num_coord_bins=40,
        num_time_bins=40,
        coord_range=coord_range,
        time_range=time_range,
        vmin=vmin,
        vmax=vmax
    )

    calculate_surface_rmse(model_v_surface, human_v_surface, include_zero_velocity=True)
    calculate_surface_rmse(model_v_surface, human_v_surface, include_zero_velocity=False)
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

