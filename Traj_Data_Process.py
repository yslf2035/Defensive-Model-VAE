import os
import numpy as np
import pandas as pd
import random

# 人类轨迹数据处理
# ===================== 场景与动作配置 =====================
SCENE_CONFIG = {
    'StaticBlindTown05': {
        'start_cond': lambda df: (df['ego_y'] >= 18) & (df['sv2_vx'] != 0) & (df['sv2_vy'] != 0),
        'end_cond': lambda row: row['ego_y'] >= 95
    },
    'DynamicBlindTown05': {
        'start_cond': lambda df: df['sv1_yaw'] < -170,
        'end_cond': lambda row: row['ego_x'] < -186
    },
    'PredictableMovementTown05': {
        'start_cond': lambda df: (df['sv1_vx'] != 0) & (df['sv1_vy'] != 0) & (df['ego_y'] <= 40) & (df['ego_y'] != 0),
        'end_cond': lambda row: row['ego_y'] <= -80
    },
    'UnpredictableMovementTown04': {
        'start_cond': lambda df: ((df['ego_x'] - df['sv1_x']) ** 2 + (df['ego_y'] - df['sv1_y']) ** 2 <= 40 ** 2)
                                 & (df['sv1_yaw'] >= -89.9),
        'end_cond': lambda row: (row['sv1_x'] > 15) and (row['sv1_yaw'] < -85)
    },
}

ACTIONS = ['减速', '减速+转向', '转向']

# ===================== 随机轨迹生成函数 =====================
def generate_random_trajectories(num_trajs, traj_length, max_angle_deviation=5.0):
    """
    生成随机轨迹
    
    Args:
        num_trajs: 轨迹数量
        traj_length: 每条轨迹的长度
        max_angle_deviation: 最大角度偏差（度），默认5度
    
    Returns:
        trajectories: 形状为(num_trajs, traj_length, 2)的numpy数组
    """
    trajectories = []
    max_angle_rad = np.radians(max_angle_deviation)
    
    for _ in range(num_trajs):
        # 从原点开始
        traj = np.zeros((traj_length, 2))
        
        # 初始方向
        current_angle = 0.0
        
        for i in range(1, traj_length):
            # 随机生成角度变化，范围在[-max_angle_deviation, max_angle_deviation]度之间
            angle_change = random.uniform(-max_angle_rad, max_angle_rad)
            current_angle += angle_change
            
            # 计算下一步位置（假设每步长度为1）
            step_length = 1.0
            dx = step_length * np.cos(current_angle)
            dy = step_length * np.sin(current_angle)
            
            # 更新位置
            traj[i, 0] = traj[i-1, 0] + dx
            traj[i, 1] = traj[i-1, 1] + dy
        
        trajectories.append(traj)
    
    return np.array(trajectories)

# ===================== 主处理函数 =====================
def process_csv(csv_path, scene, action, target_points=5, point_mode='normal', time_interval=0.015):
    df = pd.read_csv(csv_path)
    config = SCENE_CONFIG[scene]
    # 找到起始行
    start_idx = None
    start_mask = config['start_cond'](df)
    for idx, val in enumerate(start_mask):
        if val:
            start_idx = idx
            break
    if start_idx is None:
        return None  # 没有满足起始条件
    # 提取起始行及其之后的所有行
    sub_df = df.iloc[start_idx:].copy()
    # 查找结束行
    end_idx = None
    for i, row in sub_df.iterrows():
        if i == sub_df.index[0]:
            continue  # 起始行之后才判断结束条件
        if config['end_cond'](row):
            end_idx = i
            break
    if end_idx is not None:
        sub_df = sub_df.loc[:end_idx-1]  # 不包含结束行
    # 只保留ego_x和ego_y
    if 'ego_x' not in sub_df.columns or 'ego_y' not in sub_df.columns:
        return None
    traj = sub_df[['ego_x', 'ego_y']].values
    
    # 根据目标点数采样
    if len(traj) < target_points:
        return None  # 轨迹点数不足
    
    # 等间距采样，包括起点和终点
    indices = np.linspace(0, len(traj) - 1, target_points, dtype=int)
    print(((len(traj) - 1) * time_interval) / (target_points - 1))
    if point_mode == 'normal':
        traj = traj[indices]
    elif point_mode == 'extend_mid':
        part1 = indices[:-1]
        part2 = indices[1:]
        indices1 = np.ceil((part1 + part2) / 2).astype(int)
        indices_new = np.append(np.insert(indices1[:-1], 0, indices[0]), indices[-1])
        traj = traj[indices_new]
    
    # 计算时间列：从起点开始，每个采样点对应的时间
    times = np.arange(target_points) * time_interval * ((len(sub_df) - 1) / (target_points - 1))
    
    # 组合时间列和轨迹列
    result = np.column_stack((times, traj))
    return result


def collect_trajectories(data_root, scenes, actions, target_points=5, point_mode='normal', time_interval=0.015):
    all_trajs = []
    for scene in scenes:
        scene_path = os.path.join(data_root, scene)
        for action in actions:
            action_path = os.path.join(scene_path, action)
            if not os.path.exists(action_path):
                continue
            for fname in os.listdir(action_path):
                if fname.endswith('.csv'):
                    csv_path = os.path.join(action_path, fname)
                    traj = process_csv(csv_path, scene, action, target_points, point_mode, time_interval)
                    if traj is not None and len(traj) == target_points:
                        all_trajs.append(traj)
                    else:
                        print(f"No trajectory found for {scene}, {action}, {fname}")
    return all_trajs


def pad_and_save(trajs, save_path):
    if save_path is not None:
        # 所有轨迹长度已经统一为target_points，直接保存
        trajs_array = np.array(trajs)  # (num_samples, target_points, 3) - 时间+轨迹
        np.save(save_path, trajs_array)
        print(f"已保存 {trajs_array.shape[0]} 条轨迹，每条轨迹 {trajs_array.shape[1]} 个点，保存路径: {save_path}")
    else:
        print("No saving path, mode error")

# ===================== 主程序 =====================
if __name__ == "__main__":
    mode = 'dataset'  # 'dataset', 'random'
    
    if mode == 'dataset':
        # 数据集模式参数
        data_root = 'DefensiveData'  # 数据根目录
        scenes = ['UnpredictableMovementTown04']  # 选择要处理的场景文件夹，可多选
        # ['StaticBlindTown05', 'DynamicBlindTown05', 'PredictableMovementTown05', 'UnpredictableMovementTown04']
        actions = ['减速', '转向', '减速+转向']  # 选择要处理的动作文件夹，可多选：'减速'、'减速+转向'、'转向'
        target_points = 10  # 每条轨迹的目标点数
        time_interval = 0.02  # 时间间隔（秒）
        # sce1:38条, timetick=0.02s, 目标点数10, 点间隔0.52~2.19s; sce2:16条, timetick=0.025s, 目标点数10, 点间隔0.33~1.34s;
        # sce3:66条, timetick=0.015s, 目标点数10, 点间隔0.78~2.20s; sce4:135条, timetick=0.02s, 目标点数10, 点间隔0.56~1.84s
        point_mode = 'normal'  # 'normal', 'extend_mid'
        if point_mode == 'normal':
            save_path = 'training/DefensiveDataProcessed/trajectory_sce4_cond.npy'  # 保存的npy文件名
            trajs = collect_trajectories(data_root, scenes, actions, target_points, point_mode, time_interval)
        elif point_mode == 'extend_mid':
            save_path = 'training/DefensiveDataProcessed/trajectory_sce2_extend1.npy'  # 保存的npy文件名
            trajs1 = collect_trajectories(data_root, scenes, actions, target_points, 'normal', time_interval)
            trajs2 = collect_trajectories(data_root, scenes, actions, target_points, point_mode, time_interval)
            trajs = trajs1 + trajs2
        else:
            trajs = []
            save_path = None
            print("Unknown point mode...")

        if len(trajs) == 0:
            print('未提取到任何轨迹，请检查参数和数据目录。')
        else:
            pad_and_save(trajs, save_path)
    
    elif mode == 'random':
        # 随机生成模式参数
        num_trajs = 100  # 轨迹数量
        traj_length = 50  # 每条轨迹的长度
        max_angle_deviation = 0.0  # 最大角度偏差（度）
        save_path = 'training/DefensiveDataProcessed/straight_trajectory.npy'  # 保存的npy文件名
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 生成随机轨迹
        trajectories = generate_random_trajectories(num_trajs, traj_length, max_angle_deviation)
        np.save(save_path, trajectories)
        print(f"已生成并保存 {trajectories.shape[0]} 条随机轨迹，长度为 {trajectories.shape[1]}，保存路径: {save_path}")
    
    else:
        print("错误：mode参数必须是 'dataset' 或 'random'")
