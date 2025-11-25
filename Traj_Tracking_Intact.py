import math
import numpy as np
import os
from MPC.MPC_Tracking import PathTracker
from Tools import load_model_and_generate_trajectory, get_start_conditions_from_csv, get_human_and_bv_trajectories
from Tools import plot_gif_human_vs_model, save_animation_as_gif, process_model_trajectory

# 完整轨迹：潜在风险点前人类轨迹 + 潜在风险点后VAE模型生成轨迹
# 模型参数
model_path = 'training/models/vae_sce2_ld8_epoch2000.pth'  # 模型文件路径
seq_len = 10                  # 轨迹长度（sce3=12，其他=10）
dim = 3                       # 每个点的维度
latent_dim = 8                # 潜在空间维度
device = 'cpu'                # 计算设备
# CSV文件路径
csv_path = 'DefensiveData/DynamicBlindTown05/减速+转向/exp_12_control_DynamicBlindTown05_2.csv'

model_name = os.path.basename(model_path)
csv_name = os.path.basename(csv_path)
# 提取人类轨迹和背景车轨迹
human_trajectory, bv1_trajectory, bv2_trajectory = get_human_and_bv_trajectories(csv_path, model_name)

# 模型生成轨迹起始点
if "sce1" in model_name or "sce2" in model_name or "sce3" in model_name or "sce4" in model_name:
    print("使用CSV文件获取起始条件")
    start_x, start_y, start_angle, start_v = get_start_conditions_from_csv(csv_path, model_name)
else:
    print("模型名称不合规，使用默认起始条件")
    start_x, start_y = 155.0, -15.0
    start_angle = -90 * math.pi / 180
    start_v = 10.0

# VAE模型生成轨迹点 [x, y, t]
waypoints = load_model_and_generate_trajectory(model_path, start_x, start_y, seq_len, dim, latent_dim, device)
waypoints = waypoints[:, [1, 2, 0]]
waypoints[0, 2] = 0.0

# 初始状态 [x, y, theta, v]
initial_state = np.array([start_x, start_y, start_angle, start_v])

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
    wheelbase=2.8,  # 轴距
    prediction_horizon=30,  # MPC预测时域
    control_horizon=20,     # MPC控制时域
    dt=time_step            # 时间步长
)

total_time = waypoints[-1, -1]
# 运行仿真
times, states, controls = tracker.run_simulation(total_time=total_time)
# MPC跟踪后的模型生成轨迹
vae_trajectory = process_model_trajectory(human_trajectory, start_x, start_y, states, time_step)

model_name_parts = model_name.split('_')
csv_name_parts = csv_name.split('_')

# 绘制结果
pic_output_path = ("MPC/pics/vae_" + model_name_parts[1] + "_exp" + csv_name_parts[1] + "_" +
                   csv_name_parts[-1].split('.')[0] + ".png")
tracker.plot_results(pic_output_path, 'x')  # sce1,sce2-->'x',sce3,sce4-->'y'

# 绘制gif
output_filename = ('animation_' + model_name_parts[1] + '_' + csv_name_parts[1] + '_' +
                   csv_name_parts[-1].split('.')[0] + '.gif')  # 输出文件名
output_path = f'MPC/gifs/{output_filename}'  # 输出路径
print("创建gif...")
animation, fig = plot_gif_human_vs_model(human_trajectory, bv1_trajectory, bv2_trajectory, vae_trajectory, model_name)
fps = 1 / time_step  # 帧/秒
save_animation_as_gif(animation, fig, output_path, fps)
print("轨迹动图生成完成！")
