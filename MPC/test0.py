import numpy as np
from MPC_Tracking import PathTracker
from Drawing_Tools import create_path_tracking_gif, create_path_tracking_gif_with_reference

# 指定路径点和初始状态的MPC跟踪测试
# 定义路径点 [x, y, t]
waypoints = np.array([
    [0.0, 0.0, 0.0],    # 起点
    [10.0, 0.0, 1.0],
    [20.0, 0.0, 2.0],
    [30.0, 0.0, 3.0],
    [38.0, 0.5, 4.0],
    [45.0, 1.0, 5.0],
    [50.5, 1.5, 6.0],
    [55.0, 1.5, 7.0],
    [59.0, 1.0, 8.0],
    [63.0, 0.5, 9.0]    # 终点
])

# 初始状态 [x, y, theta, v]
initial_state = np.array([0.0, 0.0, 0.0, 12.0])

# 创建路径跟踪器
tracker = PathTracker(
    waypoints=waypoints,
    initial_state=initial_state,
    wheelbase=2.8,  # 轴距
    prediction_horizon=30,  # MPC预测时域
    control_horizon=20,     # MPC控制时域
    dt=0.05                 # 时间步长
)

total_time = waypoints[-1, -1]
# 运行仿真
times, states, controls = tracker.run_simulation(total_time=total_time)

# 绘制结果
tracker.plot_results("pics/path_tracking_results.png")

# 提取参考路径数据 (x, y, t)
reference_path = []
for t in times:
    x_ref, y_ref, vx_ref, vy_ref = tracker.path_interp.get_reference(t)
    reference_path.append([x_ref, y_ref, t])

reference_path = np.array(reference_path)

# 提取实际路径数据 (x, y, t)
actual_path = []
for i, t in enumerate(times):
    x_actual = states[i, 0]  # 实际x坐标
    y_actual = states[i, 1]  # 实际y坐标
    actual_path.append([x_actual, y_actual, t])

actual_path = np.array(actual_path)

# 创建GIF动画
print("\n开始创建路径跟踪GIF动画...")

# 创建仅包含实际路径的GIF
gif_path1 = create_path_tracking_gif(
    actual_path=actual_path,
    output_path="gifs/path_tracking_actual.gif",
    fps=20,  # frame/second
    dpi=100,
    figsize=(12, 12)
)

# # 创建包含参考路径和实际路径的GIF
# gif_path2 = create_path_tracking_gif_with_reference(
#     actual_path=actual_path,
#     reference_path=reference_path,
#     output_path="gifs/path_tracking_with_reference.gif",
#     fps=20,
#     dpi=100,
#     figsize=(12, 8)
# )

print(f"\nGIF动画创建完成！")
