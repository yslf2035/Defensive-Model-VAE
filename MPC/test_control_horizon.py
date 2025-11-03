"""
测试控制时域功能的脚本
验证预测时域和控制时域的不同设置
"""

from PathTracking import PathTracker
import numpy as np
import matplotlib.pyplot as plt

def test_control_horizon():
    """测试不同控制时域设置的效果"""
    
    # 定义路径点 [x, y, t]
    waypoints = np.array([
        [0.0, 0.0, 0.0],    # 起点
        [10.0, 5.0, 2.0],   # 中间点
        [20.0, 0.0, 4.0],   # 终点
    ])
    
    # 初始状态 [x, y, theta, v]
    initial_state = np.array([0.0, 0.0, 0.0, 3.0])
    
    # 测试不同的控制时域设置
    test_cases = [
        {"name": "pred_horizon=10, ctrl_horizon=10", "pred_horizon": 10, "ctrl_horizon": 10},
        {"name": "pred_horizon=10, ctrl_horizon=5", "pred_horizon": 10, "ctrl_horizon": 5},
        {"name": "pred_horizon=15, ctrl_horizon=5", "pred_horizon": 15, "ctrl_horizon": 5},
        {"name": "pred_horizon=20, ctrl_horizon=8", "pred_horizon": 20, "ctrl_horizon": 8},
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"\n=== 测试案例: {case['name']} ===")
        
        # 创建路径跟踪器
        tracker = PathTracker(
            waypoints=waypoints,
            initial_state=initial_state,
            wheelbase=2.8,
            prediction_horizon=case['pred_horizon'],
            control_horizon=case['ctrl_horizon'],
            dt=0.01
        )
        
        # 运行仿真
        total_time = waypoints[-1, 2] + 1.0
        times, states, controls = tracker.run_simulation(total_time)
        
        # 计算跟踪误差
        ref_x = np.array([tracker.path_interp.get_reference(t)[0] for t in times])
        ref_y = np.array([tracker.path_interp.get_reference(t)[1] for t in times])
        pos_error = np.sqrt((states[:, 0] - ref_x)**2 + (states[:, 1] - ref_y)**2)
        
        # 存储结果
        results[case['name']] = {
            'times': times,
            'states': states,
            'controls': controls,
            'pos_error': pos_error,
            'max_error': np.max(pos_error),
            'mean_error': np.mean(pos_error),
            'final_error': pos_error[-1]
        }
        
        print(f"最大位置误差: {np.max(pos_error):.3f}m")
        print(f"平均位置误差: {np.mean(pos_error):.3f}m")
        print(f"最终位置误差: {pos_error[-1]:.3f}m")
    
    # 绘制对比图
    plot_comparison(results, waypoints)
    
    return results

def plot_comparison(results, waypoints):
    """绘制不同控制时域设置的对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 轨迹对比
    ax1 = axes[0, 0]
    ax1.plot(waypoints[:, 0], waypoints[:, 1], 'ko-', label='Reference Path', markersize=6)
    
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(data['states'][:, 0], data['states'][:, 1], 
                color=colors[i], linewidth=2, label=name)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Path Tracking Comparison')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # 位置误差对比
    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        ax2.plot(data['times'], data['pos_error'], 
                color=colors[i], linewidth=2, label=name)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Position Error Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # 速度对比
    ax3 = axes[1, 0]
    for i, (name, data) in enumerate(results.items()):
        ax3.plot(data['times'], data['states'][:, 3], 
                color=colors[i], linewidth=2, label=name)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Comparison')
    ax3.legend()
    ax3.grid(True)
    
    # 控制输入对比
    ax4 = axes[1, 1]
    for i, (name, data) in enumerate(results.items()):
        if len(data['controls']) > 0:
            ax4.plot(data['times'][:-1], data['controls'][:, 0], 
                    color=colors[i], linewidth=2, label=f"{name} (Acc)")
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Acceleration (m/s²)')
    ax4.set_title('Control Input Comparison')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('pics/control_horizon_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*60)
    print("控制时域测试结果摘要")
    print("="*60)
    print(f"{'配置':<25} {'最大误差(m)':<12} {'平均误差(m)':<12} {'最终误差(m)':<12}")
    print("-"*60)
    
    for name, data in results.items():
        print(f"{name:<25} {data['max_error']:<12.3f} {data['mean_error']:<12.3f} {data['final_error']:<12.3f}")

if __name__ == "__main__":
    print("=== 控制时域功能测试 ===")
    print("测试不同预测时域和控制时域设置对路径跟踪性能的影响")
    
    results = test_control_horizon()
    print_summary(results)
    
    print("\n测试完成！结果图已保存到 pics/control_horizon_comparison.png")
