import numpy as np
import os
from MPC.MPC_Tracking import PathTracker
from MPC.Drawing_Tools import create_path_tracking_gif, create_path_tracking_gif_with_reference
from Tools import *

# VAE模型生成轨迹的MPC跟踪测试
# 模型参数
model_path = 'training/models/vae_offset_sce3_ld8_epoch3000.pth'  # 模型文件路径
seq_len = 10                  # 轨迹长度（sce3=12，其他=10）
dim = 3                       # 每个点的维度
latent_dim = 8                # 潜在空间维度
device = 'cpu'                # 计算设备
# CSV文件路径
csv_path = 'DefensiveData/PredictableMovementTown05/减速+转向/exp_62_control_PredictableMovementTown05_3.csv'
# 起始点
model_name = os.path.basename(model_path)
if "sce1" in model_name or "sce2" in model_name or "sce3" in model_name or "sce4" in model_name:
    print("使用CSV文件获取起始条件")
    start_x, start_y, start_angle, start_vx, start_vy = get_start_conditions_from_csv(csv_path, model_name)
else:
    print("模型名称不合规，使用默认起始条件")
    start_x, start_y = 155.0, -15.0
    start_angle = -90 * math.pi / 180
    start_vx = 0.0
    start_vy = 10.0

# VAE模型生成轨迹点 [x, y, t]
waypoints = load_model_and_generate_trajectory(model_path, start_x, start_y, seq_len, dim, latent_dim, device)
waypoints = waypoints[:, [1, 2, 0]]
waypoints[0, 2] = 0.0

# 初始状态 [x, y, theta, vx, vy]
initial_state = np.array([start_x, start_y, start_angle, start_vx, start_vy])

# 创建路径跟踪器
tracker = PathTracker(
    waypoints=waypoints,
    initial_state=initial_state,
    wheelbase=2.8,  # 轴距
    prediction_horizon=30,  # MPC预测时域
    control_horizon=20,     # MPC控制时域
    dt=0.015                 # 时间步长
)

total_time = waypoints[-1, -1]
# 运行仿真
times, states, controls = tracker.run_simulation(total_time=total_time)
# 绘制结果
tracker.plot_results("MPC/pics/vae_path_tracking_results.png", 'y')  # sce1,sce2-->'x',sce3,sce4-->'y'
