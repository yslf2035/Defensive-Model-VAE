"""
指标计算：从 collected_data 读取 CSV，按场景截取片段并计算 TTC 或 PET。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

# 默认数据目录：本文件位于 SUT_Testing/tools/，数据在 SUT_Testing/collected_data/
_DEFAULT_COLLECTED_DIR = Path(__file__).resolve().parent.parent / "collected_data"

EPS_V = 1e-9
EPS_DET = 1e-12


def _pet_two_rays(
    px1: np.ndarray,
    py1: np.ndarray,
    vx1: np.ndarray,
    vy1: np.ndarray,
    yaw1_deg: np.ndarray,
    px2: np.ndarray,
    py2: np.ndarray,
    vx2: np.ndarray,
    vy2: np.ndarray,
    yaw2_deg: np.ndarray,
) -> np.ndarray:
    """
    PET：两车以当前速率大小、航向 yaw（度）确定的匀速直线（射线）运动，
    若两射线交于一点且两车沿前进方向到达该点的时间为 t1、t2（均 >= 0），
    则 PET = |t1 - t2|；否则为 NaN（平行、重合无唯一交点，或交点在行驶反方向）。
    """
    sp1 = np.hypot(vx1, vy1)
    sp2 = np.hypot(vx2, vy2)
    th1 = np.deg2rad(yaw1_deg)
    th2 = np.deg2rad(yaw2_deg)
    V1x = sp1 * np.cos(th1)
    V1y = sp1 * np.sin(th1)
    V2x = sp2 * np.cos(th2)
    V2y = sp2 * np.sin(th2)
    dpx = px2 - px1
    dpy = py2 - py1
    # [V1x -V2x; V1y -V2y] [t1;t2] = [dpx;dpy]
    det = V1x * (-V2y) - (-V2x) * V1y
    with np.errstate(all="ignore"):
        t1 = (dpx * (-V2y) - dpy * (-V2x)) / det
        t2 = (V1x * dpy - V1y * dpx) / det
        pet = np.abs(t1 - t2)
    invalid = (
        (np.abs(det) < EPS_DET)
        | (sp1 < EPS_V)
        | (sp2 < EPS_V)
        | (t1 < 0)
        | (t2 < 0)
        | ~np.isfinite(t1)
        | ~np.isfinite(t2)
    )
    pet = np.asarray(pet, dtype=float)
    pet[invalid] = np.nan
    return pet


def pet_ego_sv2(df: pd.DataFrame) -> pd.Series:
    """ego 与 sv2 的 PET（列：ego_*, sv2_*）。"""
    ex = np.asarray(df["ego_x"], dtype=float)
    ey = np.asarray(df["ego_y"], dtype=float)
    evx = np.asarray(df["ego_vx"], dtype=float)
    evy = np.asarray(df["ego_vy"], dtype=float)
    eyaw = np.asarray(df["ego_yaw"], dtype=float)
    sx = np.asarray(df["sv2_x"], dtype=float)
    sy = np.asarray(df["sv2_y"], dtype=float)
    svx = np.asarray(df["sv2_vx"], dtype=float)
    svy = np.asarray(df["sv2_vy"], dtype=float)
    syaw = np.asarray(df["sv2_yaw"], dtype=float)
    pet = _pet_two_rays(ex, ey, evx, evy, eyaw, sx, sy, svx, svy, syaw)
    return pd.Series(pet, index=df.index)


def pet_ego_sv1(df: pd.DataFrame) -> pd.Series:
    """ego 与 sv1 的 PET。"""
    ex = np.asarray(df["ego_x"], dtype=float)
    ey = np.asarray(df["ego_y"], dtype=float)
    evx = np.asarray(df["ego_vx"], dtype=float)
    evy = np.asarray(df["ego_vy"], dtype=float)
    eyaw = np.asarray(df["ego_yaw"], dtype=float)
    sx = np.asarray(df["sv1_x"], dtype=float)
    sy = np.asarray(df["sv1_y"], dtype=float)
    svx = np.asarray(df["sv1_vx"], dtype=float)
    svy = np.asarray(df["sv1_vy"], dtype=float)
    syaw = np.asarray(df["sv1_yaw"], dtype=float)
    pet = _pet_two_rays(ex, ey, evx, evy, eyaw, sx, sy, svx, svy, syaw)
    return pd.Series(pet, index=df.index)


def parse_filename(stem: str) -> Tuple[str, str, Optional[int]]:
    """
    从文件名（不含扩展名）解析：模型名、场景名、测试次序号（无则为 None）。
    规则：第一个 '_' 前为模型；第一个与第二个 '_' 之间为场景；若存在第二个 '_'，其后为测试序号。
    """
    parts = stem.split("_", 2)
    if len(parts) < 2:
        raise ValueError(f"文件名格式无效: {stem}")
    model = parts[0]
    if len(parts) == 2:
        return model, parts[1], None
    # len == 3: model, scenario, tail (可能为纯数字序号)
    scenario = parts[1]
    tail = parts[2]
    if tail.isdigit():
        return model, scenario, int(tail)
    # 场景名中可能含额外下划线时，split('_',2) 会把剩余合并到第三部分
    return model, f"{scenario}_{tail}", None


def resolve_csv_path(
    model: str,
    scenario: str,
    test_run: Optional[Union[int, str]] = None,
    collected_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """根据模型、场景、测试次序号解析 CSV 路径。"""
    base = Path(collected_dir) if collected_dir is not None else _DEFAULT_COLLECTED_DIR
    if test_run is not None and str(test_run).strip() != "":
        name = f"{model}_{scenario}_{int(test_run)}.csv"
    else:
        name = f"{model}_{scenario}.csv"
    path = base / name
    if not path.is_file():
        raise FileNotFoundError(f"未找到 CSV: {path}")
    return path


def _first_index(mask: pd.Series) -> Optional[int]:
    idx = mask.to_numpy().nonzero()[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def filter_static_blind_town05(df: pd.DataFrame) -> pd.DataFrame:
    """StaticBlindTown05：开始 ego_y>0 且 sv2 速度非零；结束后第一个 ego_y>=80（含）或文件末尾。"""
    m_start = (df["ego_y"] > 0) & (df["sv2_vx"].astype(float) != 0) & (df["sv2_vy"].astype(float) != 0)
    i0 = _first_index(m_start)
    if i0 is None:
        raise ValueError("StaticBlindTown05: 未找到满足条件的开始行")
    sub = df.iloc[i0:].reset_index(drop=True)
    m_end = sub["ego_y"] >= 80
    i1 = _first_index(m_end)
    if i1 is None:
        return sub
    return sub.iloc[: i1 + 1].reset_index(drop=True)


def filter_dynamic_blind_town05(df: pd.DataFrame) -> pd.DataFrame:
    """DynamicBlindTown05：开始 sv1_yaw<-150；结束第一个 ego_x<-186.8897。"""
    m_start = df["sv1_yaw"].astype(float) < -150
    i0 = _first_index(m_start)
    if i0 is None:
        raise ValueError("DynamicBlindTown05: 未找到满足条件的开始行")
    sub = df.iloc[i0:].reset_index(drop=True)
    m_end = sub["ego_x"].astype(float) < -186.8897
    i1 = _first_index(m_end)
    if i1 is None:
        return sub
    return sub.iloc[: i1 + 1].reset_index(drop=True)


def filter_predictable_movement_town05(df: pd.DataFrame) -> pd.DataFrame:
    """PredictableMovementTown05：开始 ego_y<40 且 ego_y!=0 且 sv1 速度非零；结束 ego_y<-78。"""
    ey = df["ego_y"].astype(float)
    m_start = (
        (ey < 40)
        & (ey != 0)
        & (df["sv1_vx"].astype(float) != 0)
        & (df["sv1_vy"].astype(float) != 0)
    )
    i0 = _first_index(m_start)
    if i0 is None:
        raise ValueError("PredictableMovementTown05: 未找到满足条件的开始行")
    sub = df.iloc[i0:].reset_index(drop=True)
    m_end = sub["ego_y"].astype(float) < -78
    i1 = _first_index(m_end)
    if i1 is None:
        return sub
    return sub.iloc[: i1 + 1].reset_index(drop=True)


def filter_unpredictable_movement_town04(df: pd.DataFrame) -> pd.DataFrame:
    """UnpredictableMovementTown04：开始 距离<=30 且 |sv1_ax|>=0.1；结束 0<|sv1_ax|<0.1 且 sv1_yaw<-90 且 sv1_x>15。"""
    ex, ey = df["ego_x"].astype(float), df["ego_y"].astype(float)
    sv1_x, sv1_y = df["sv1_x"].astype(float), df["sv1_y"].astype(float)
    dist = np.sqrt((ex - sv1_x) ** 2 + (ey - sv1_y) ** 2)
    sv1_ax = df["sv1_ax"].astype(float)
    sv1_yaw = df["sv1_yaw"].astype(float)
    m_start = (dist <= 30) & (np.abs(sv1_ax) >= 0.1)
    i0 = _first_index(m_start)
    if i0 is None:
        raise ValueError("UnpredictableMovementTown04: 未找到满足条件的开始行")
    sub = df.iloc[i0:].reset_index(drop=True)
    sv1_x_s = sub["sv1_x"].astype(float)
    sv1_ax_s = sub["sv1_ax"].astype(float)
    sv1_yaw_s = sub["sv1_yaw"].astype(float)
    m_end = (np.abs(sv1_ax_s) < 0.1) & (sv1_yaw_s < -90) & (sv1_ax_s != 0) & (sv1_x_s > 15)
    i1 = _first_index(m_end)
    if i1 is None:
        return sub
    return sub.iloc[: i1 + 1].reset_index(drop=True)


def ttc_axis_positive_y(df: pd.DataFrame) -> pd.Series:
    """ego 与 sv2 沿 y 轴（正方向相对运动）的一维 TTC: (sv2_y - ego_y) / (ego_vy - sv2_vy)。"""
    ey = df["ego_y"].astype(float)
    sy = df["sv2_y"].astype(float)
    evy = df["ego_vy"].astype(float)
    svy = df["sv2_vy"].astype(float)
    denom = evy - svy
    num = sy - ey
    out = num / denom.replace(0, np.nan)
    out = out.where(denom.abs() > EPS_V, np.nan)
    return out


def ttc_axis_negative_x(df: pd.DataFrame) -> pd.Series:
    """ego 与 sv2 沿 x 轴负方向的一维 TTC: (sv2_x - ego_x) / (ego_vx - sv2_vx)。"""
    ex = df["ego_x"].astype(float)
    sx = df["sv2_x"].astype(float)
    evx = df["ego_vx"].astype(float)
    svx = df["sv2_vx"].astype(float)
    denom = evx - svx
    num = sx - ex
    out = num / denom.replace(0, np.nan)
    out = out.where(denom.abs() > EPS_V, np.nan)
    return out


def ttc_axis_negative_y_ego_sv1(df: pd.DataFrame) -> pd.Series:
    """ego 与 sv1 沿 y 轴负方向的一维 TTC: (sv1_y - ego_y) / (ego_vy - sv1_vy)。"""
    ey = df["ego_y"].astype(float)
    sy = df["sv1_y"].astype(float)
    evy = df["ego_vy"].astype(float)
    svy = df["sv1_vy"].astype(float)
    denom = evy - svy
    num = sy - ey
    out = num / denom.replace(0, np.nan)
    out = out.where(denom.abs() > EPS_V, np.nan)
    return out


def filter_dataframe_by_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "StaticBlindTown05":
        return filter_static_blind_town05(df)
    if scenario == "DynamicBlindTown05":
        return filter_dynamic_blind_town05(df)
    if scenario == "PredictableMovementTown05":
        return filter_predictable_movement_town05(df)
    if scenario == "UnpredictableMovementTown04":
        return filter_unpredictable_movement_town04(df)
    raise ValueError(f"未知场景: {scenario}")


def add_ttc_column(filtered: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = filtered.copy()
    if scenario == "StaticBlindTown05":
        out["TTC"] = ttc_axis_positive_y(out)
    elif scenario == "DynamicBlindTown05":
        out["TTC"] = ttc_axis_negative_x(out)
    elif scenario in ("PredictableMovementTown05", "UnpredictableMovementTown04"):
        out["TTC"] = ttc_axis_negative_y_ego_sv1(out)
    else:
        raise ValueError(f"未知场景: {scenario}")
    return out


def add_pet_column(filtered: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """按场景在截取后的数据上添加 PET 列（对象与 TTC 相同：Static/Dynamic 为 ego–sv2，其余为 ego–sv1）。"""
    out = filtered.copy()
    if scenario in ("StaticBlindTown05", "DynamicBlindTown05"):
        out["PET"] = pet_ego_sv2(out)
    elif scenario in ("PredictableMovementTown05", "UnpredictableMovementTown04"):
        out["PET"] = pet_ego_sv1(out)
    else:
        raise ValueError(f"未知场景: {scenario}")
    return out


def _default_dt_jerk(scenario: str) -> float:
    """无 sim_time 时各场景默认采样步长（秒）。"""
    if scenario in ("StaticBlindTown05", "UnpredictableMovementTown04"):
        return 0.02
    if scenario == "DynamicBlindTown05":
        return 0.025
    if scenario == "PredictableMovementTown05":
        return 0.015
    raise ValueError(f"未知场景: {scenario}")


def add_jerk_column(filtered: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    添加自车 JERK 列（加速度变化率，单位为 加速度单位/秒）。

    - DynamicBlindTown05: 计算 ego_ax 的变化率
    - 其余三个场景: 计算 ego_ay 的变化率

    时间间隔：优先用 sim_time 的相邻差分；若无 sim_time 列，则按场景使用固定步长：
    StaticBlindTown05 / UnpredictableMovementTown04 为 0.02s，DynamicBlindTown05 为 0.025s，
    PredictableMovementTown05 为 0.015s。
    """
    out = filtered.copy()

    if scenario == "DynamicBlindTown05":
        a = out["ego_ax"].astype(float)
    else:
        a = out["ego_ay"].astype(float)

    if "sim_time" in out.columns:
        t = out["sim_time"].astype(float)
        dt = t.diff()
        jerk = a.diff() / dt
        jerk = jerk.where(dt.abs() > EPS_V, np.nan)
    else:
        dt = _default_dt_jerk(scenario)
        jerk = a.diff() / dt

    out["JERK"] = jerk
    return out


def compute_metric_from_csv(
    model: str,
    scenario: str,
    test_run: Optional[Union[int, str]] = None,
    collected_dir: Optional[Union[str, Path]] = None,
    metric: str = "TTC",
) -> pd.DataFrame:
    """
    读取对应 CSV，按场景截取行，并添加 TTC / PET / JERK 列。

    - "TTC": 一维轴上 TTC
    - "PET": 航向+速率射线交点到达时间差
    - "JERK": ego 自车 jerk（加速度变化率），Dynamic 用 ego_ax，其余用 ego_ay；
      无 sim_time 时按场景默认步长（0.02 / 0.025 / 0.015 s）
    """
    m = metric.strip().upper()
    if m not in ("TTC", "PET", "JERK"):
        raise ValueError(f"metric 须为 TTC / PET / JERK，收到: {metric!r}")
    path = resolve_csv_path(model, scenario, test_run, collected_dir)
    df = pd.read_csv(path)
    filtered = filter_dataframe_by_scenario(df, scenario)
    if m == "TTC":
        return add_ttc_column(filtered, scenario)
    if m == "PET":
        return add_pet_column(filtered, scenario)
    return add_jerk_column(filtered, scenario)


def compute_ttc_from_csv(
    model: str,
    scenario: str,
    test_run: Optional[Union[int, str]] = None,
    collected_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """读取对应 CSV，按场景截取行，并添加 TTC 列（等价于 metric=\"TTC\"）。"""
    return compute_metric_from_csv(model, scenario, test_run, collected_dir, metric="TTC")


def _sub_valid_for_metric(sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    """按指标筛选用于统计均/最值的行：TTC 沿用 >0；PET 为有限非负；JERK 为有限值。"""
    col = metric.strip().upper()
    if col == "TTC":
        return sub[sub["TTC"] > 0]
    if col == "PET":
        v = sub["PET"]
        return sub[v.notna() & np.isfinite(v) & (v >= 0)]
    if col == "JERK":
        v = sub["JERK"]
        return sub[v.notna() & np.isfinite(v)]
    return sub


def _print_metric_stats(sub_valid: pd.DataFrame, mcol: str) -> None:
    """打印指标统计：TTC/PET 打印均值+最小值；JERK 打印 |JERK| 均值 + |JERK| 最大值。"""
    if len(sub_valid) == 0:
        print(f"无有效行，无法统计 {mcol}。")
        return
    if mcol == "JERK":
        vabs = sub_valid["JERK"].abs()
        print("JERK(绝对值) 均值：", vabs.mean())
        print("JERK(绝对值) 最大值：", vabs.max())
        return
    print(f"{mcol} 均值：", sub_valid[mcol].mean())
    print(f"{mcol} 最小值：", sub_valid[mcol].min())


def main():
    # ---在此选择模型、场景、测试次序号---
    # IDM, BEHAVIOR, TCP, DEF
    model = "DEF"
    # StaticBlindTown05, DynamicBlindTown05, PredictableMovementTown05, UnpredictableMovementTown04
    scenario = "UnpredictableMovementTown04"
    test_run: Optional[int] = 1  # 例如 1; 不需要序号时改为 None
    # TTC, PET, JERK
    metric = "TTC"

    result = compute_metric_from_csv(model, scenario, test_run, metric=metric)
    mcol = metric.strip().upper()
    print(f"模型={model}, 场景={scenario}, 测试序号={test_run}, metric={mcol}")
    print(f"有效行数: {len(result)}")

    if scenario == "StaticBlindTown05" and mcol in result.columns:
        mask = (result["sv2_x"] >= -196.81) & (result["sv2_x"] <= -193.31)
        sub = result.loc[mask]
        sub_valid = _sub_valid_for_metric(sub, mcol)
        _print_metric_stats(sub_valid, mcol)
        print("最大纵向减速度：", sub_valid["ego_ay"].min())
        print("最大转向角：", sub_valid["ego_yaw"].max())
        return
    if scenario == "DynamicBlindTown05" and mcol in result.columns:
        # 去除发生碰撞后
        positive_mask = result["ego_ax"] >= 100.0
        if positive_mask.any():
            # 获取第一个满足 ego_vx > 0 的行索引
            first_positive_idx = result[positive_mask].index[0]
            sub = result.loc[:first_positive_idx - 1]
        else:
            # 如果没有 ego_vx > 0 的行，保留全部数据
            sub = result.copy()
        sub_valid = _sub_valid_for_metric(sub, mcol)
        _print_metric_stats(sub_valid, mcol)
        print("最大纵向减速度：", sub_valid["ego_ax"].max())
        return
    if scenario == "PredictableMovementTown05" and mcol in result.columns:
        # 自行车进入自车车道
        mask = result["sv1_x"] <= 156.76
        sub = result.loc[mask]
        sub_valid = _sub_valid_for_metric(sub, mcol)
        _print_metric_stats(sub_valid, mcol)
        print("最大纵向减速度：", sub_valid["ego_ay"].max())
        return
    elif scenario == "UnpredictableMovementTown04" and mcol in result.columns:
        # 车道线端点坐标
        x1, y1 = 13.06, -160
        x2, y2 = 14.77, 220
        # 交互车进入自车车道
        right_side_mask = (result["sv1_x"] - x1) * (y2 - y1) - (result["sv1_y"] - y1) * (x2 - x1) > 0
        if right_side_mask.any():
            first_right_idx = result[right_side_mask].index[0]
            sub = result.loc[first_right_idx:]
        else:
            sub = result.iloc[0:0]
        sub_valid = _sub_valid_for_metric(sub, mcol)
        _print_metric_stats(sub_valid, mcol)
        print("最大纵向减速度：", sub_valid["ego_ay"].max())
        return


if __name__ == "__main__":
    main()
