import re
from pathlib import Path
import pandas as pd
import numpy as np
from dnda_functions import DrivableArea_RiskLevel_Calculation
from DNDA import Road, Vehicle

# 文件夹路径
UNPROCESSED_DIR = Path(__file__).parent.parent / "DefensiveData_Unprocessed"
OUTPUT_FILE = Path(__file__).parent / "max_dnda_summary.csv"

# 文件名正则
# FILENAME_RE = re.compile(
#     r"^exp_(?P<participant>\d+)_control_(?P<scenario>[^_]+)_(?P<trial>\d+)\.csv$",
#     re.IGNORECASE,
# )
FILENAME_RE = re.compile(
    r"^(?P<model>[^_]+)_(?P<scenario>[^_]+)_(?P<trial>\d+)\.csv$",
    re.IGNORECASE,
)

def parse_filename(file_name: str):
    m = FILENAME_RE.match(file_name)
    if not m:
        return None
    d = m.groupdict()
    # return {
    #     "participant_id": int(d["participant"]),
    #     "scenario": d["scenario"],
    #     "trial": int(d["trial"]),
    # }
    return {
        "model": d["model"],
        "scenario": d["scenario"],
        "trial": int(d["trial"]),
    }

def generate_baseline_for_static_blind_town():
    """
    生成 StaticBlindTown 场景的参考路径点。
    参考路径为经过 (-189.81, 0) 和 (-189.81, 100) 的直线，y 值范围为 [-80, 200]，点间距为 0.5。
    """
    x = -189.81
    y_values = np.arange(-80, 300, 0.25)  # y 值范围 [-80, 200]，间距 0.25
    baseline = np.array([[x, y] for y in y_values]).flatten()
    return baseline

def generate_baseline_for_dynamic_blind_town05():
    """
    生成 DynamicBlindTown05 场景的参考路径点。
    参考路径为经过 (-50, 0.92) 和 (-177.3, 1.23) 的直线，x 值范围为 [-300, 20]，点间距为 0.5。
    """
    x_values = np.arange(20.5,-350, -0.25)  # x 值范围 [-300, 20]，间距 0.25
    slope = (1.23 - 0.92) / (-177.3 - (-50))  # 计算直线斜率
    intercept = 0.92 - slope * (-50)  # 计算直线截距
    baseline = np.array([[x, slope * x + intercept] for x in x_values]).flatten()
    return baseline

def generate_baseline_for_predictable_movement_town05():
    """
    生成 PredictableMovementTown05 场景的参考路径点。
    参考路径为经过 (153.33, 60) 和 (153.19, -100) 的直线，y 值范围为 [-200, 100]，点间距为 0.5。
    """
    y_values = np.arange(100.5, -300, -0.25)  # y 值范围 [-200, 100]，间距 0.25
    x = 153.33  # x 坐标恒定
    baseline = np.array([[x, y] for y in y_values]).flatten()
    return baseline

def generate_baseline_for_unpredictable_movement_town04():
    """
    生成 UnpredictableMovementTown04 场景的参考路径点。
    参考路径为经过 (7.77, 220) 和 (6.06, -160) 的直线，y 值范围为 [-450, 210]，点间距为 0.5。
    """
    y_values = np.arange(210.5, -450, -0.25)  # y 值范围 [-450, 210]，间距 0.25
    slope = (220 - (-160)) / (7.77 - 6.06)  # 计算直线斜率0
    intercept = 220 - slope * 7.77  # 计算直线截距
    baseline = np.array([[y / slope - intercept / slope, y] for y in y_values]).flatten()
    return baseline

def calculate_relative_theta_and_init_q(ego_x, ego_y, ego_yaw, baseline):
    """
    计算自车与参考路径的相对夹角 (relative_theta_) 和垂直距离 (init_q_)。
    """
    baseline = baseline.reshape(-1, 2)  # 重塑为二维数组，每行一个参考路径点 (x, y)
    # 找到距离自车最近的参考路径点
    distances = np.linalg.norm(baseline - np.array([ego_x, ego_y]), axis=1)
    nearest_idx = np.argmin(distances)
    nearest_point = baseline[nearest_idx]

    # 计算参考路径的方向向量
    if nearest_idx < len(baseline) - 1:
        next_point = baseline[nearest_idx + 1]
    else:
        next_point = baseline[nearest_idx - 1]
    path_vector = next_point - nearest_point
    path_yaw = np.arctan2(path_vector[1], path_vector[0])  # 参考路径的方向角（弧度）

     # 计算相对夹角和垂直距离
    relative_theta = ego_yaw - path_yaw  # 保持单位为弧度
    relative_theta = (relative_theta + np.pi) % (2 * np.pi) - np.pi  # 将角度归一化到 [-π, π]
    init_q = np.dot(np.array([ego_x, ego_y]) - nearest_point, [-path_vector[1], path_vector[0]]) / np.linalg.norm(path_vector)

    return relative_theta, init_q

def calculate_dnda(row, baseline, Mrow_b, surVhe_input, Mrow_s, frame, log_flag, test_data_log, basepoint_num, polygon_folder, surRect_folder, scenario):
    
    # 根据场景生成参考路径    
    relative_theta, init_q = calculate_relative_theta_and_init_q(row['ego_x'], row['ego_y'], row['ego_yaw'], baseline)

    # 转换相对角度为弧度
    relative_theta = np.deg2rad(relative_theta)
    
    lane_num=1
    lane_egodirect=1
    if scenario == "StaticBlindTown05":
        lane_num = 2
        lane_egodirect = 2
    elif scenario == "DynamicBlindTown05":
        lane_num = 2
        lane_egodirect = 2
    elif scenario == "PredictableMovementTown05":
        lane_num = 1
        lane_egodirect = 1
    elif scenario == "UnpredictableMovementTown04":
        lane_num = 3
        lane_egodirect = 3

    # 创建道路对象
    road = Road(
        cross_centerline= False,
        lane_num=lane_num,
        lane_egodirect=lane_egodirect,
        lane_width=3.5,
        maxAbsoluteAcc=9.8
    )

    # 创建自车对象
    egoVeh = Vehicle(
        x=row['ego_x'],
        y=row['ego_y'],
        length=7.2,
        width=2.3,
        speed_x=row['ego_vx'],
        speed_y=row['ego_vy'],
        speed=np.sqrt(row['ego_vx']**2 + row['ego_vy']**2),
        absolute_theta=row['ego_yaw'],
        acc=np.cos(row['ego_yaw'])*row['ego_ax'] + np.sin(row['ego_yaw'])*row['ego_ay'],
        init_q=init_q,
        lane_posi=1,
        
        relative_theta=relative_theta
    )

    # 调用 DrivableArea_RiskLevel_Calculation 计算 DA 和 RL
    result = DrivableArea_RiskLevel_Calculation(
        timeHorizon=3.0,
        timeStep=0.1,
        road=road,
        egoVeh=egoVeh,
        baseline=baseline,
        Mrow_b=Mrow_b,
        surVhe_input=surVhe_input,
        Mrow_s=Mrow_s,
        frame=frame,
        log_flag=log_flag,
        test_data_log = test_data_log,
        basepoint_num=basepoint_num,
        polygon_folder=polygon_folder,
        surRect_folder=surRect_folder
    )

    # DNDA = RL
    dnda = result[1]
    return dnda

def generate_surVhe_input(row, max_sv, scenario):
    """
    从当前帧的 CSV 数据中生成 surVhe_input 数组。
    数组排列顺序为 [sv1_x, sv1_y, sv1_length, sv1_width, sv1_vx, sv1_vy, sv1_ax, sv1_ay, sv1_yaw, ...]。
    """
    surVhe_input = []
    for i in range(1, max_sv + 1):
        sv_x = row.get(f'sv{i}_x', np.nan)
        sv_y = row.get(f'sv{i}_y', np.nan)

        # 根据场景和背景车辆编号设置尺寸
        if (scenario == "StaticBlindTown05" and i == 2) or (scenario == "PredictableMovementTown05" and i == 1):
            sv_length = 2.0  # 自行车长度
            sv_width = 0.7   # 自行车宽度
        else:
            sv_length = 4.0  # 轿车长度
            sv_width = 2.0   # 轿车宽度

        sv_vx = row.get(f'sv{i}_vx', np.nan)
        sv_vy = row.get(f'sv{i}_vy', np.nan)
        sv_ax = row.get(f'sv{i}_ax', 0.0)          # 默认加速度为 0.0
        sv_ay = row.get(f'sv{i}_ay', 0.0)          # 默认加速度为 0.0
        sv_yaw = row.get(f'sv{i}_yaw', 0.0)        # 默认朝向为 0.0

        # 如果背景车辆的位置信息缺失，则跳过该车辆
        if np.isnan(sv_x) or np.isnan(sv_y):
            continue

        # 按顺序添加到 surVhe_input 数组
        surVhe_input.extend([sv_x, sv_y, sv_length, sv_width, sv_vx, sv_vy, sv_ax, sv_ay, sv_yaw])

    return surVhe_input

def rotate_point(x, y, angle):
    """
    旋转点 (x, y) 绕原点逆时针旋转指定角度。
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    x_rot = x * cos_theta + y * sin_theta
    y_rot = -x * sin_theta + y * cos_theta
    return x_rot, y_rot

def rotate_baseline_to_x_direction(baseline):
    """
    将参考路径旋转到 x 方向。
    """
    baseline = baseline.reshape(-1, 2)  # 将一维数组重塑为二维数组
    x1, y1 = baseline[0]  # 起点
    x2, y2 = baseline[-1]  # 终点
    angle = np.arctan2(y2 - y1, x2 - x1)  # 计算旋转角度

    # 对所有点进行旋转
    rotated_baseline = np.array([rotate_point(x, y, -angle) for x, y in baseline])
    return rotated_baseline.flatten(), angle  # 返回旋转后的路径和旋转角度

def rotate_vehicle_positions(row, angle):
    """
    旋转场景中所有车辆（包括 ego 和 sv）的坐标、朝向、速度和加速度。
    
    参数:
        row: 包含车辆状态的字典或 Pandas 行。
        angle: 旋转角度（弧度），逆时针为正。
    
    返回:
        row: 旋转后的车辆状态。
    """
    # 旋转 ego 车辆
    ego_x, ego_y = row['ego_x'], row['ego_y']
    ego_vx, ego_vy = row['ego_vx'], row['ego_vy']
    ego_ax, ego_ay = row['ego_ax'], row['ego_ay']
    ego_yaw = row['ego_yaw']

    # 旋转 ego 的位置
    row['ego_x'], row['ego_y'] = rotate_point(ego_x, ego_y, -angle)
    # 旋转 ego 的速度
    row['ego_vx'], row['ego_vy'] = rotate_point(ego_vx, ego_vy, -angle)
    # 旋转 ego 的加速度
    row['ego_ax'], row['ego_ay'] = rotate_point(ego_ax, ego_ay, -angle)
    # 旋转 ego 的朝向
    row['ego_yaw'] = (ego_yaw - np.rad2deg(angle)) % 360
    # 转换为与 y 轴的夹角，单位为弧度
    row['ego_yaw'] = np.deg2rad(90 - row['ego_yaw'])

    # 旋转所有 sv 车辆
    for i in range(1, 21):  # 假设最多有 20 辆车
        if f'sv{i}_x' in row and f'sv{i}_y' in row:
            sv_x = row[f'sv{i}_x']
            sv_y = row[f'sv{i}_y']
            sv_vx = row[f'sv{i}_vx']
            sv_vy = row[f'sv{i}_vy']
            sv_ax = row[f'sv{i}_ax']
            sv_ay = row[f'sv{i}_ay']
            sv_yaw = row[f'sv{i}_yaw']

            if not np.isnan(sv_x) and not np.isnan(sv_y):
                # 旋转 sv 的位置
                row[f'sv{i}_x'], row[f'sv{i}_y'] = rotate_point(sv_x, sv_y, -angle)
                # 旋转 sv 的速度
                row[f'sv{i}_vx'], row[f'sv{i}_vy'] = rotate_point(sv_vx, sv_vy, -angle)
                # 旋转 sv 的加速度
                row[f'sv{i}_ax'], row[f'sv{i}_ay'] = rotate_point(sv_ax, sv_ay, -angle)
                # 旋转 sv 的朝向
                row[f'sv{i}_yaw'] = (sv_yaw - np.rad2deg(angle)) % 360
                # 转换为与 y 轴的夹角，单位为弧度
                row[f'sv{i}_yaw'] = np.deg2rad(90 - row[f'sv{i}_yaw'])

    return row


def process_file(filepath: Path, baseline, Mrow_b, log_flag, test_data_log, basepoint_num, polygon_folder, surRect_folder, scenario):
    df = pd.read_csv(filepath)
    max_dnda = -np.inf

    # 创建输出文件路径
    output_file = f"DrivableArea_to_python/DNDA_detailed_res/dnda_results_{filepath.stem}.csv"

    # 将参考路径旋转到 x 方向
    rotated_baseline, angle = rotate_baseline_to_x_direction(baseline)

    # 统计背景车辆数量
    sv_columns = [col for col in df.columns if col.startswith('sv') and col.endswith('_x')]
    sv_nums = [int(re.search(r'sv(\d+)_x', col).group(1)) for col in sv_columns if re.search(r'sv(\d+)_x', col)]
    max_sv = max(sv_nums) if sv_nums else 0

    # 准备保存每帧结果的列表
    frame_results = []
    # 找到 ego 开始运动的帧
    start_frame = df[(df['ego_vx'] != 0) | (df['ego_vy'] != 0)].index[0]

    for idx, row in df.iterrows():
        frame = idx + 1  # 假设帧号从 1 开始

        # 每 10 个帧计算一次
        if frame % 10 != 0:
            continue

        # 跳过静止的帧
        if frame < start_frame:
            continue

        # 检查停止条件
        if scenario == "StaticBlindTown05" and row['ego_y'] >= 80:
            print(f"Stopping calculation for {scenario} at frame {frame} (ego_y >= 80).")
            break
        elif scenario == "DynamicBlindTown05" and row['ego_x'] <= -186.8897:
            print(f"Stopping calculation for {scenario} at frame {frame} (ego_x <= -186.8897).")
            break
        elif scenario == "PredictableMovementTown05" and row['ego_y'] <= -78:
            print(f"Stopping calculation for {scenario} at frame {frame} (ego_y <= -78).")
            break
        elif scenario == "UnpredictableMovementTown04":
            sv1_x = row.get('sv1_x', np.nan)
            sv1_yaw = row.get('sv1_yaw', np.nan)
            ego_y = row['ego_y']
            if not np.isnan(sv1_x) and not np.isnan(sv1_yaw):
                # 检查停止条件：sv1_x > 14 且 sv1_yaw ≈ -90°
                if sv1_x > 14 and abs(sv1_yaw - (-90)) < 3:  # 允许一定的误差范围
                    print(f"Stopping calculation for {scenario} at frame {frame} (sv1_x > 14 and sv1_yaw ≈ -90°).")
                    break
            if ego_y <= -160:
                print(f"Stopping calculation for {scenario} at frame {frame} (ego_y <= -160).")
                break

        # 旋转车辆坐标和朝向
        row = rotate_vehicle_positions(row, angle)

        # 生成 surVhe_input 数组
        surVhe_input = generate_surVhe_input(row, max_sv, scenario)
        Mrow_s = max_sv # 每辆车有 9 个参数

        # 计算 DNDA
        dnda = calculate_dnda(row, rotated_baseline, Mrow_b, surVhe_input, Mrow_s, frame, log_flag, test_data_log, basepoint_num, polygon_folder, surRect_folder, scenario)
        max_dnda = max(max_dnda, dnda)

        # 保存当前帧的结果
        frame_results.append({"frame": frame, "dnda": dnda})

        # 检查 DNDA 值是否为 1
        if dnda == 1:
            print(f"Stopping calculation for {scenario} at frame {frame} (DNDA=1).")
            break

    # input()
    # 将每帧结果保存到单独的 CSV 文件
    pd.DataFrame(frame_results).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved DNDA results for {filepath.name} to {output_file}")

    return max_dnda

def find_csv_files(root: Path):
    return sorted([p for p in root.rglob("*.csv") if FILENAME_RE.match(p.name)])



def main():
    # 示例输入
    baseline_static_blind_town = generate_baseline_for_static_blind_town()
    baseline_dynamic_blind_town05 = generate_baseline_for_dynamic_blind_town05()
    baseline_predictable_movement_town05 = generate_baseline_for_predictable_movement_town05()
    baseline_unpredictable_movement_town04 = generate_baseline_for_unpredictable_movement_town04()

    log_flag = False            # 记录背景车和自车的包围盒
    test_data_log = False       # 记录计算DNDA每一步的处理后的参考路径信息等
    basepoint_num = 400
    polygon_folder = "VisulizationResult/polygon/"
    surRect_folder = "VisulizationResult/surRect/"

    # 指定从哪个文件开始计算
    start_file = "DEF_UnpredictableMovementTown04_1.csv"  # 设置为文件名（如 "exp_1_control_StaticBlindTown_1.csv"），或 None 表示从头开始

    rows = []
    csv_files = find_csv_files(UNPROCESSED_DIR)
    print(f"Found {len(csv_files)} CSV files to process.")
    # 跳过 start_file 之前的文件
    start_processing = start_file is None
    for fp in csv_files:
        if not start_processing:
            if fp.name == start_file:
                start_processing = True
            else:
                continue
        
        print(f"----------Processing file: {fp.name}----------")
        meta = parse_filename(fp.name)
        if not meta:
            continue
        scenario = meta["scenario"]
        
        # 跳过 FreeDriveTown05 场景
        if scenario == "FreeDriveTown05":
            print(f"Skipping calculation for {fp.name} (scenario: FreeDriveTown05).")
            continue
        
        # # 补充计算：这里只计算 StaticBlindTown05 场景的 DNDA
        # if scenario != "StaticBlindTown05":
        #     print(f"Skipping calculation for {fp.name} (scenario: {scenario}).")
        #     continue

        if scenario == "StaticBlindTown05":
            baseline = baseline_static_blind_town
        elif scenario == "DynamicBlindTown05":
            baseline = baseline_dynamic_blind_town05
        elif scenario == "PredictableMovementTown05":
            baseline = baseline_predictable_movement_town05
        elif scenario == "UnpredictableMovementTown04":
            baseline = baseline_unpredictable_movement_town04
        else:
            # 其他场景的参考路径生成逻辑（预留位置）
            baseline = [...]  # 请补充具体逻辑

        Mrow_b = len(baseline) //2  # 每个参考路径点有 x 和 y 两个参数
        # print(len(baseline))
        # print(Mrow_b)

        max_dnda = process_file(fp, baseline, Mrow_b, log_flag, test_data_log, basepoint_num, polygon_folder, surRect_folder, scenario)
        row = {**meta, "max_dnda": max_dnda, "file_path": str(fp.relative_to(UNPROCESSED_DIR))}
        rows.append(row)
        print(f"Processed file: {fp.name}, max_dnda: {max_dnda:.2f}")

         # 保存结果到文件
        # df_out = pd.DataFrame([row])
        # if not OUTPUT_FILE.exists():
        #     df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        # else:
        #     df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig", mode='a', header=False)

        # 保存结果到文件，替换已有记录
        if OUTPUT_FILE.exists():
            df_existing = pd.read_csv(OUTPUT_FILE)
            df_existing = df_existing[df_existing["file_path"] != row["file_path"]]  # 删除旧记录
            df_out = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
        else:
            df_out = pd.DataFrame([row])
        
        df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Finished processing {len(rows)} files. Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()