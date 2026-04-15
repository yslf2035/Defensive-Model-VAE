"""
绘制静止轨迹点，见ITSC2026小论文图12、13
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import PathCollection
from matplotlib.ticker import FuncFormatter
from Tools import get_human_and_bv_trajectories
from Distribution import load_tracked_trajectories_from_files


def _subsample_at_interval(traj_xyt, interval_sec=1.0):
    """
    从轨迹 (x, y, t) 中每隔 interval_sec 秒取一个点（取最接近该时刻的点）。

    Args:
        traj_xyt: shape (N, 3)，列为 x, y, t
        interval_sec: 时间间隔（秒）

    Returns:
        (x, y) 两个一维数组，用于散点图
    """
    if traj_xyt is None or len(traj_xyt) == 0:
        return np.array([]), np.array([])
    t = traj_xyt[:, 2]
    t_max = t.max()
    targets = np.arange(0, t_max + 1e-6, interval_sec)
    x_pts, y_pts = [], []
    for t_target in targets:
        idx = np.argmin(np.abs(t - t_target))
        x_pts.append(traj_xyt[idx, 0])
        y_pts.append(traj_xyt[idx, 1])
    return np.array(x_pts), np.array(y_pts)


def plot_static_trajectories(
    human_traj,
    bv1_traj,
    bv2_traj,
    model_traj,
    model_name,
    save_path=None,
    human_traj_no_def=None,
    plot_human_no_def=False,
    legend_three_points=True,
):
    """
    绘制静态轨迹散点图：人类、背景车、模型轨迹每隔一定间隔一个点，不绘制车辆矩形。

    Args:
        human_traj: 人类轨迹 (来自 csv_path)，(N, 3) 为 (x,y,t)
        bv1_traj: 背景车1轨迹，格式同 human_traj
        bv2_traj: 背景车2轨迹，格式同 human_traj
        model_traj: 模型轨迹，(N, 3) 或 (N, 4)，会按 model_name 的 time_step 补 t
        model_name: 模型名称，用于 xlim/ylim/车道线/time_step/坐标轴翻转
        save_path: 可选，保存图片路径
        human_traj_no_def: 可选，来自 csv_path_no_def 的人类轨迹（仅人类，无 BV）
        plot_human_no_def: 是否绘制 human_traj_no_def
        legend_three_points: 图例中散点是否用三个点表示（默认 True）
    """
    if "sce1" in model_name:
        xlim = (-204, -184)
        ylim = (20, 88)
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
        xlim = (140, 170)
        ylim = (-82, 40)
        time_step = 0.015

    # 将 model_traj 统一为 (x, y, t)
    if model_traj is not None and model_traj.ndim == 2:
        if model_traj.shape[1] == 4:
            n = len(model_traj)
            t_col = np.arange(n) * time_step
            model_traj = np.column_stack((model_traj[:, 0], model_traj[:, 1], t_col))
        elif model_traj.shape[1] == 2:
            n = len(model_traj)
            model_traj = np.column_stack((model_traj[:, 0], model_traj[:, 1], np.arange(n) * time_step))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Y (m)', rotation=180)
    ax.set_ylabel('X (m)')
    ax.grid(False, axis='both')
    ax.set_aspect('equal')

    # 车道线（与 Tools.plot_gif_human_vs_model 一致）
    if "sce1" in model_name:
        y_range = np.linspace(0, 73.2, 100)
        ax.plot([-196.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
        ax.plot([-193.3] * len(y_range), y_range, 'k--', linewidth=1.5, alpha=0.7)
        ax.plot([-189.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
    elif "sce2" in model_name:
        x_range = np.linspace(-177, -50, 200)
        ax.plot(x_range, [-5.8] * len(x_range), 'k-', linewidth=1.5, alpha=0.7)
        ax.plot(x_range, [-2.3] * len(x_range), 'k--', linewidth=1.5, alpha=0.7)
        ax.plot(x_range, [1.2] * len(x_range), 'k-', linewidth=1.5, alpha=0.7)
    elif "sce4" in model_name:
        y_range = np.linspace(-40, 120, 100)
        for xv in [3.5, 7, 10.5, 14, 17.5]:
            ax.plot([xv] * len(y_range), y_range, 'k-' if xv in (3.5, 17.5) else 'k--', linewidth=1.5, alpha=0.7)
    else:
        y_range = np.linspace(-100, 60, 100)
        ax.plot([153.3] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
        ax.plot([156.8] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)
        ax.plot([149.7] * len(y_range), y_range, 'k-', linewidth=1.5, alpha=0.7)

    model_color = (0, 0.4470, 0.7410)
    human_color_from_csv = (0.4660, 0.6740, 0.1880)   # 绿色
    bv_color = (1.0, 0.584, 0.0)                        # 橙色
    human_no_def_color = (0.7961, 0.1255, 0.1765)     # 红色

    interval_sec = 0.2

    if model_traj is not None and len(model_traj) > 0:
        mx, my = _subsample_at_interval(model_traj, interval_sec)
        if len(mx) > 0:
            ax.scatter(mx, my, color=model_color, s=20, alpha=1, label='Model', zorder=8)

    if human_traj is not None and len(human_traj) > 0:
        hx, hy = _subsample_at_interval(human_traj, interval_sec)
        if len(hx) > 0:
            ax.scatter(hx, hy, color=human_color_from_csv, s=20, alpha=1, label='Human (Defensive)', zorder=6)

    if bv1_traj is not None and len(bv1_traj) > 0:
        b1x, b1y = _subsample_at_interval(bv1_traj, interval_sec)
        if len(b1x) > 0:
            ax.scatter(b1x, b1y, color=bv_color, s=20, alpha=1, label='Two-wheeler', zorder=5)

    if bv2_traj is not None and len(bv2_traj) > 0 and "sce3" not in model_name:
        b2x, b2y = _subsample_at_interval(bv2_traj, interval_sec)
        if len(b2x) > 0:
            ax.scatter(b2x, b2y, color=bv_color, s=20, alpha=1, zorder=5)

    if plot_human_no_def and human_traj_no_def is not None and len(human_traj_no_def) > 0:
        hnd_x, hnd_y = _subsample_at_interval(human_traj_no_def, interval_sec)
        if len(hnd_x) > 0:
            ax.scatter(hnd_x, hnd_y, color=human_no_def_color, s=20, alpha=1, label='Human (Non-defensive)', zorder=5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    handles, labels = list(by_label.values()), list(by_label.keys())

    if legend_three_points:
        # 将散点图例由单点改为三个点
        new_handles = []
        for h in handles:
            if isinstance(h, PathCollection):
                fc = h.get_facecolors()
                c = fc[0] if fc.size > 0 else (0, 0, 0, 1)
                if c.shape[0] == 4:
                    c = c[:3]
                sz = h.get_sizes()
                s = int(sz[0]) if sz.size > 0 else 20
                markersize = np.sqrt(s / np.pi) * 2
                proxy = mlines.Line2D(
                    [0, 1, 2], [0, 0, 0],
                    linestyle='',
                    marker='o',
                    color=c,
                    markersize=markersize,
                )
                new_handles.append(proxy)
            else:
                new_handles.append(h)
        handles = new_handles

    # ax.legend(handles, labels)
    ax.set_title(f'Human Trajectory VS Model Trajectory (static, every {interval_sec}s)')

    if "sce1" in model_name or "sce2" in model_name:
        ax.invert_xaxis()
    elif "sce3" in model_name or "sce4" in model_name:
        ax.invert_yaxis()

    # sce3：坐标轴刻度重标定——x 以 150 为 0、正方向不变；y 以 40 为 0、正方向翻转
    if "sce1" in model_name:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{-190 - x:.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y - 20:.0f}"))
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=90)
    elif "sce3" in model_name:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x - 150:.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{40 - y:.0f}"))
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=90)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Static trajectory plot saved: {save_path}")

    return fig, ax


def _get_npy_path_for_csv(csv_path, scenario_key, saved_folder):
    """
    根据 scenario_key 与 csv_path 中的 "exp_" 后代号、".csv" 前代号，定位对应的 npy 文件。
    例如：scenario_key="sce3", csv_path='.../exp_1_control_PredictableMovementTown05_2.csv'
         → 目标文件名为 "tracked_trajectory_sce3_exp1_2.npy"

    Args:
        csv_path: CSV 文件路径
        scenario_key: 场景代号，如 sce1/sce2/sce3/sce4
        saved_folder: 存放 npy 的目录

    Returns:
        匹配的 npy 文件完整路径，若不存在则返回 None
    """
    if not os.path.isdir(saved_folder):
        return None
    basename = os.path.basename(csv_path)
    name_no_ext = os.path.splitext(basename)[0]  # e.g. "exp_1_control_PredictableMovementTown05_2"
    # "exp_" 后的代号：取 exp_ 后第一个下划线前的数字
    exp_code = None
    if "exp_" in name_no_ext:
        after_exp = name_no_ext.split("exp_", 1)[1]
        exp_code = after_exp.split("_")[0]  # e.g. "1"
    # ".csv" 前的代号：即文件名（去扩展名）按 '_' 分割的最后一段
    suffix = name_no_ext.split("_")[-1] if "_" in name_no_ext else name_no_ext
    if exp_code is None:
        return None
    expected_name = f"tracked_trajectory_{scenario_key}_exp{exp_code}_{suffix}.npy"
    full_path = os.path.join(saved_folder, expected_name).replace("\\", "/")
    return full_path if os.path.isfile(full_path) else None


if __name__ == '__main__':
    # 与 Traj_Tracking_Intact.py 一致的参数
    model_path = 'training/models/vae_offset_sce1_cond_ld8_epoch3000.pth'
    # sce3: exp_1_control_PredictableMovementTown05_2.csv
    csv_path = 'DefensiveData/StaticBlindTown05/减速/exp_28_control_StaticBlindTown05_3.csv'
    # sce3: Data-0315\exp_control\exp_50_control_PredictableMovementTown05_1.csv
    csv_path_no_def = 'D:\枫\同济\研0\课题组\本科毕业设计（论文）\驾驶模拟器实验\实验轨迹数据\Defensive_Data_Collected\Data-0315\exp_control\exp_51_control_StaticBlindTown05_1.csv'
    if csv_path_no_def is None:
        plot_human_no_def = False  # 是否绘制从 csv_path_no_def 提取的人类轨迹
    else:
        plot_human_no_def = True

    model_name = os.path.basename(model_path)
    csv_name = os.path.basename(csv_path)

    human_trajectory, bv1_trajectory, bv2_trajectory = get_human_and_bv_trajectories(csv_path, model_name)
    if human_trajectory is None:
        print("Failed to load human/BV trajectories from CSV. Exit.")
        exit(1)

    human_trajectory_no_def = None
    if csv_path_no_def is not None and os.path.isfile(csv_path_no_def):
        out = get_human_and_bv_trajectories(csv_path_no_def, model_name)
        if out[0] is not None:
            human_trajectory_no_def = out[0]
        else:
            print("Warning: could not load human trajectory from csv_path_no_def.")

    # 从 npy 中取一条模型轨迹：按 scenario + csv 的 exp_ 代号与 .csv 前代号精确匹配 npy 文件名
    saved_folder = 'results/GeneratedData'
    for part in ['sce1', 'sce2', 'sce3', 'sce4']:
        if part in model_name:
            scenario_key = part
            break
        else:
            scenario_key = 'sce3'
    npy_path = _get_npy_path_for_csv(csv_path, scenario_key, saved_folder)
    if npy_path is None:
        print(f"No matching .npy file in {saved_folder} for scenario '{scenario_key}' and csv (exp_* / * before .csv). Using dummy model trajectory.")
        n_pts = min(100, len(human_trajectory))
        model_trajectory = human_trajectory[:n_pts].copy()
        model_trajectory[:, :2] += np.random.randn(n_pts, 2) * 2
    else:
        trajectories = load_tracked_trajectories_from_files([npy_path])
        if not trajectories:
            print("load_tracked_trajectories_from_files returned empty. Exit.")
            exit(1)
        model_trajectory = trajectories[0]

    model_name_parts = model_name.split('_')
    csv_name_parts = csv_name.split('_')
    save_path = (
        'results/ModelValidation/Cases/static_trajectories_'
        + model_name_parts[2]
        + '_cond_'
        + csv_name_parts[1]
        + '_'
        + csv_name_parts[-1].split('.')[0]
        + '.png'
    )

    plot_static_trajectories(
        human_trajectory,
        bv1_trajectory,
        bv2_trajectory,
        model_trajectory,
        model_name,
        save_path=save_path,
        human_traj_no_def=human_trajectory_no_def,
        plot_human_no_def=plot_human_no_def,
        legend_three_points=True,
    )
    plt.show()
