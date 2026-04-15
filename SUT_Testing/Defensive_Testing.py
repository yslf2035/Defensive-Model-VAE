"""
将 results/GeneratedData 中的跟踪轨迹 npy 写回对应的 DefensiveData CSV。

npy 每行为 [x, y, theta, v]：theta 为航向角（弧度），v 为合成速度（m/s）。
除 ego_x、ego_y 外，同步写入 ego_vx、ego_vy、ego_ax、ego_ay（由 v、theta 与时间序列计算）
及 ego_yaw（theta 转为角度制）。

要求：Python 3.8+
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

# sce 代号 -> CSV 文件名中的城镇后缀
SCE_TO_TOWN = {
    "sce1": "StaticBlindTown05",
    "sce2": "DynamicBlindTown05",
    "sce3": "PredictableMovementTown05",
    "sce4": "UnpredictableMovementTown04",
}


def _project_root() -> Path:
    # SUT_Testing/Defensive_Testing.py -> 项目根 DefensiveModel
    return Path(__file__).resolve().parent.parent


def parse_tracked_npy_name(filename: Union[str, Path]) -> Tuple[str, str, str]:
    """
    tracked_trajectory_sce1_exp1_3.npy -> (sce1, 1, 3)
    对应 CSV: exp_1_control_StaticBlindTown05_3.csv
    """
    name = os.path.basename(str(filename))
    m = re.match(
        r"tracked_trajectory_(sce[1-4])_exp(\d+)_(\d+)\.npy$",
        name,
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(
            f"无法解析 npy 文件名（期望 tracked_trajectory_sce*_exp*_*.npy）: {name}"
        )
    sce_key, exp_num, suffix_num = m.group(1).lower(), m.group(2), m.group(3)
    if sce_key not in SCE_TO_TOWN:
        raise ValueError(f"不支持的 scenario 代号: {sce_key}")
    return sce_key, exp_num, suffix_num


def expected_csv_name(sce_key: str, exp_num: str, suffix_num: str) -> str:
    town = SCE_TO_TOWN[sce_key]
    return f"exp_{exp_num}_control_{town}_{suffix_num}.csv"


def find_csv_under_defensive_data(
    defensive_data_root: Path, csv_basename: str
) -> Path:
    """在 DefensiveData/<城镇>/<二级>/<三级>/ 下按文件名查找 CSV。"""
    town_dirs = [d for d in defensive_data_root.iterdir() if d.is_dir()]
    matches: List[Path] = []
    for town in town_dirs:
        for p in town.rglob(csv_basename):
            if p.is_file():
                matches.append(p)
    if not matches:
        raise FileNotFoundError(
            f"在 {defensive_data_root} 下未找到文件: {csv_basename}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"找到多个同名 CSV，请手动指定路径: {csv_basename}\n"
            + "\n".join(str(x) for x in matches)
        )
    return matches[0]


_EGO_COLS_FULL = (
    "ego_x",
    "ego_y",
    "ego_vx",
    "ego_vy",
    "ego_ax",
    "ego_ay",
    "ego_yaw",
)


def parse_tracked_npy_state(arr: np.ndarray) -> np.ndarray:
    """
    解析 npy 为 [x, y, theta, v]，形状 (N, 4)。
    列数不足 4 时无法写入全部 ego 量，将报错。
    """
    if arr.ndim != 2:
        raise ValueError(f"轨迹应为二维数组，得到 shape={arr.shape}")
    if arr.shape[1] < 4:
        raise ValueError(
            f"npy 需至少 4 列 [x, y, theta, v]，当前列数为 {arr.shape[1]}"
        )
    return np.asarray(arr[:, :4], dtype=float)


def row_times_seconds(df: pd.DataFrame, start_row: int, L: int) -> np.ndarray:
    """
    取与替换段对齐的时间序列，用于 d(vx)/dt、d(vy)/dt。
    优先列名 ``frame``，否则 ``time``；若均不存在则用 0,1,...,L-1（等间隔 1）。
    列值应为单调时间（与采集一致）；若为纯帧序号，加速度量纲需自行理解。
    """
    if L <= 0:
        return np.array([], dtype=float)
    sub = df.iloc[start_row : start_row + L]
    if "frame" in df.columns:
        t = pd.to_numeric(sub["frame"], errors="coerce").to_numpy()
    elif "time" in df.columns:
        t = pd.to_numeric(sub["time"], errors="coerce").to_numpy()
    else:
        t = np.arange(L, dtype=float)

    if np.any(np.isnan(t)):
        s = pd.Series(t)
        t = s.interpolate(limit_direction="both").bfill().ffill().to_numpy()

    return np.asarray(t, dtype=float)


def compute_ego_kinematics(
    theta: np.ndarray, v: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    theta（弧度）、v（m/s）、t（秒，与样本对齐）。
    返回 vx, vy, ax, ay, ego_yaw（度）。
    vx = v*cos(theta), vy = v*sin(theta)；ax、ay 为 vx、vy 对时间的导数，
    使用 numpy.gradient 与 t 做非均匀采样二阶精度近似。
    """
    theta = np.asarray(theta, dtype=float)
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    L = len(theta)
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    yaw_deg = np.rad2deg(theta)

    if L == 1:
        z = np.zeros(1, dtype=float)
        return vx, vy, z, z, yaw_deg

    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    return vx, vy, ax, ay, yaw_deg


def find_best_start_row(df: pd.DataFrame, x0: float, y0: float) -> int:
    if "ego_x" not in df.columns or "ego_y" not in df.columns:
        raise KeyError('CSV 中需要包含列 "ego_x" 与 "ego_y"')
    ex = pd.to_numeric(df["ego_x"], errors="coerce").to_numpy()
    ey = pd.to_numeric(df["ego_y"], errors="coerce").to_numpy()
    d2 = (ex - x0) ** 2 + (ey - y0) ** 2
    i = int(np.nanargmin(d2))
    return i


def merge_trajectory_into_csv(
    df: pd.DataFrame, traj_xytv: np.ndarray, start_row: int
) -> pd.DataFrame:
    """
    从 start_row 起写入 npy 段 [x,y,theta,v] 推导的 ego 列；长度取 min，截断较长一侧。
    """
    for col in _EGO_COLS_FULL:
        if col not in df.columns:
            raise KeyError(f'CSV 中缺少列 "{col}"')

    n_csv = len(df)
    remain = n_csv - start_row
    if remain <= 0:
        raise ValueError("起始行超出 CSV 行数")
    n_traj = len(traj_xytv)
    L = min(n_traj, remain)
    if L <= 0:
        raise ValueError("轨迹长度为 0")

    seg = traj_xytv[:L]
    x, y, theta, v = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    t = row_times_seconds(df, start_row, L)
    vx, vy, ax, ay, yaw_deg = compute_ego_kinematics(theta, v, t)

    out = df.copy()
    sl = slice(start_row, start_row + L)
    loc = out.columns.get_loc
    out.iloc[sl, loc("ego_x")] = x
    out.iloc[sl, loc("ego_y")] = y
    out.iloc[sl, loc("ego_vx")] = vx
    out.iloc[sl, loc("ego_vy")] = vy
    out.iloc[sl, loc("ego_ax")] = ax
    out.iloc[sl, loc("ego_ay")] = ay
    out.iloc[sl, loc("ego_yaw")] = yaw_deg

    # npy 较短：截掉 CSV 从 start_row+L 之后的行
    if L < remain:
        keep_idx = list(range(start_row + L))
        out = out.iloc[keep_idx].reset_index(drop=True)
    return out


def main() -> None:
    root = _project_root()
    # 可在 main 中修改：相对项目根的 npy 路径
    npy_relative = os.path.join(
        "results", "GeneratedData", "tracked_trajectory_sce4_exp13_2.npy"
    )
    npy_path = root / npy_relative

    defensive_data_root = root / "DefensiveData"
    out_dir = Path(__file__).resolve().parent / "collected_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(npy_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raise ValueError("不支持的 npy 内容（object 数组），请保存为数值型轨迹数组")

    traj = np.asarray(raw)
    traj_xytv = parse_tracked_npy_state(traj)
    x0, y0 = float(traj_xytv[0, 0]), float(traj_xytv[0, 1])
    print(f"x0={x0}, y0={y0}")

    sce_key, exp_num, suffix_num = parse_tracked_npy_name(npy_path.name)
    csv_name = expected_csv_name(sce_key, exp_num, suffix_num)
    csv_path = find_csv_under_defensive_data(defensive_data_root, csv_name)

    df = pd.read_csv(csv_path)
    start_row = find_best_start_row(df, x0, y0)
    merged = merge_trajectory_into_csv(df, traj_xytv, start_row)

    stem = Path(csv_path.name).stem
    out_path = out_dir / f"{stem}_def.csv"
    merged.to_csv(out_path, index=False)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
