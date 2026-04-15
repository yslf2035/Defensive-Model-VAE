import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Tools import get_start_conditions_from_csv
from Distribution import collect_csv_files

# 全局字体配置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


def analyze_and_plot_start_velocity(csv_files, model_name, save_dir="results/Trigger_THW", bin_width=0.5, x_max=18):
    """
    批量计算初始速度并绘制分布直方图
    :return: start_velocities(DataFrame), save_path(图片保存路径)
    """
    os.makedirs(save_dir, exist_ok=True)
    velocity_data = []

    # 遍历处理每个CSV
    for i, csv_path in enumerate(csv_files):
        print(f"\n处理文件 {i + 1}/{len(csv_files)}: {os.path.basename(csv_path)}")
        csv_name = os.path.basename(csv_path)

        # 获取初始条件
        start_x, start_y, start_angle, start_vx, start_vy = get_start_conditions_from_csv(csv_path, model_name)

        # 计算合速度
        start_v = math.sqrt(start_vx ** 2 + start_vy ** 2)

        velocity_data.append({
            'csv_name': csv_name,
            'start_v': start_v
        })

    # 构建DataFrame
    start_velocities = pd.DataFrame(velocity_data)

    # 统计信息
    v_min = start_velocities['start_v'].min()
    v_max = start_velocities['start_v'].max()
    v_mean = start_velocities['start_v'].mean()
    print(f"\n速度统计：最小值 {v_min:.2f} m/s | 最大值 {v_max:.2f} m/s | 均值 {v_mean:.2f} m/s")

    # ===================== 原图：start_v 直方图 =====================
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, x_max, bin_width)
    plt.hist(start_velocities['start_v'], bins=bins, alpha=0.8, color='skyblue', edgecolor='black', density=True)
    plt.title('Start Velocity Distribution', fontsize=14)
    plt.xlim(0, x_max)
    plt.xlabel('Start Velocity (m/s)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(alpha=0.3)
    save_path = os.path.join(save_dir, 'start_velocity_distribution.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"原始速度分布图已保存：{save_path}")

    return start_velocities, save_path


def plot_thw(df, model_name, save_dir="results/Trigger_THW", bin_width=0.5, x_max=10):
    """
    新增函数：计算 60/start_v 并绘制直方图
    """
    os.makedirs(save_dir, exist_ok=True)

    if "sce1" in model_name:
        df['THW'] = 60 / df['start_v']
    elif "sce3" in model_name:
        df['THW'] = 80 / df['start_v']
    elif "sce4" in model_name:
        # 提取 csv_name 最后一段（_后 .csv前）
        suffix = df['csv_name'].str.replace('.csv', '').str.split('_').str[-1]
        # 逐行判断：后缀=3 → 用40，否则用30，再除以start_v
        df['THW'] = df['start_v'].copy()
        df.loc[suffix == '3', 'THW'] = 40 / df.loc[suffix == '3', 'start_v']
        df.loc[suffix != '3', 'THW'] = 30 / df.loc[suffix != '3', 'start_v']

    # 统计
    t_min = df['THW'].min()
    t_max = df['THW'].max()
    t_mean = df['THW'].mean()
    print(f"\nTHW统计：最小值 {t_min:.2f} s | 最大值 {t_max:.2f} s | 均值 {t_mean:.2f} s")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, x_max, bin_width)
    plt.hist(df['THW'], bins=bins, alpha=0.8, color='lightgreen', edgecolor='black', density=True)
    plt.title('THW Distribution', fontsize=14)
    plt.xlim(0, x_max)
    plt.xlabel('THW (s)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, 'THW_distribution.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"60/速度分布图已保存：{save_path}")


def main():
    # 配置参数
    base_folder = 'DefensiveData/UnpredictableMovementTown04'
    model_path = 'training/models/vae_offset_sce4_cond_ld8_epoch3000.pth'

    # 读取文件
    csv_files = collect_csv_files(base_folder)
    model_name = os.path.basename(model_path)

    # 1. 绘制原始速度直方图
    start_velocities_df, img_save_path = analyze_and_plot_start_velocity(
        csv_files=csv_files,
        model_name=model_name,
        bin_width=0.5,
        x_max=22  # sce1: 18, sce2: 20, sce3: 16, sce4: 22
    )

    # 2. 如果模型名包含 sce1 → 额外画 60/start_v 直方图
    if "sce1" in model_name:
        plot_thw(
            df=start_velocities_df,
            model_name=model_name,
            save_dir="results/Trigger_THW",
            bin_width=0.2,
            x_max=7
        )
    elif "sce3" in model_name:
        plot_thw(
            df=start_velocities_df,
            model_name=model_name,
            save_dir="results/Trigger_THW",
            bin_width=0.2,
            x_max=13
        )
    elif "sce4" in model_name:
        plot_thw(
            df=start_velocities_df,
            model_name=model_name,
            save_dir="results/Trigger_THW",
            bin_width=0.2,
            x_max=4
        )

if __name__ == "__main__":
    main()
