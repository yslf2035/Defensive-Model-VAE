"""
路径跟踪系统测试脚本
简化版本，用于验证核心功能
"""

import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from MPC_Tracking import PathTracker, create_test_path
    print("✓ 成功导入PathTracking模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)


def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    try:
        # 创建简单测试路径
        waypoints = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 2.0, 2.0],
            [10.0, 2.0, 4.0],
            [15.0, 3.0, 6.0]
        ])
        
        # 初始状态
        initial_state = np.array([0.0, 0.0, 0.0, 1.5])
        
        # 创建跟踪器
        tracker = PathTracker(
            waypoints=waypoints,
            initial_state=initial_state,
            wheelbase=2.8,
            horizon=5,  # 减少预测时域以加快测试
            dt=0.01
        )
        
        print("✓ 成功创建PathTracker")
        
        # 测试单步控制
        state, control = tracker.step(0.0)
        print(f"✓ 单步控制成功: 位置=({state[0]:.2f}, {state[1]:.2f}), 速度={state[3]:.2f}")
        
        # 测试路径插值
        ref = tracker.path_interp.get_reference(1.0)
        print(f"✓ 路径插值成功: 参考点=({ref[0]:.2f}, {ref[1]:.2f})")
        
        # 运行短时间仿真
        times, states, controls = tracker.run_simulation(2.0)
        print(f"✓ 仿真成功: {len(times)}步, 最终位置=({states[-1,0]:.2f}, {states[-1,1]:.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mpc_optimization():
    """测试MPC优化"""
    print("\n=== 测试MPC优化 ===")
    
    try:
        from MPC_Tracking import VehicleModel, MPCController
        
        # 创建车辆模型
        vehicle = VehicleModel()
        
        # 创建MPC控制器
        mpc = MPCController(vehicle, horizon=5, dt=0.01)
        
        # 测试状态
        current_state = np.array([0.0, 0.0, 0.0, 2.0])
        
        # 创建参考轨迹
        ref_trajectory = np.array([
            [0.0, 0.0, 0.0, 2.0],
            [0.02, 0.0, 0.0, 2.0],
            [0.04, 0.0, 0.0, 2.0],
            [0.06, 0.0, 0.0, 2.0],
            [0.08, 0.0, 0.0, 2.0],
            [0.10, 0.0, 0.0, 2.0]
        ])
        
        # 求解MPC
        control_sequence = mpc.solve_mpc(current_state, ref_trajectory)
        
        print(f"✓ MPC求解成功: 控制序列形状={control_sequence.shape}")
        print(f"  第一个控制: 加速度={control_sequence[0,0]:.3f}, 转向角={control_sequence[0,1]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ MPC测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vehicle_dynamics():
    """测试车辆动力学"""
    print("\n=== 测试车辆动力学 ===")
    
    try:
        from MPC_Tracking import VehicleModel
        
        vehicle = VehicleModel()
        
        # 测试状态
        state = np.array([0.0, 0.0, 0.0, 2.0])
        control = np.array([1.0, 0.1])  # 加速度1m/s², 转向角0.1rad
        
        # 计算动力学
        state_derivative = vehicle.dynamics(state, control, 0.01)
        
        print(f"✓ 车辆动力学计算成功: 状态导数={state_derivative}")
        
        # 测试轨迹预测
        controls = np.array([[1.0, 0.1], [0.5, 0.05], [0.0, 0.0]])
        trajectory = vehicle.predict_trajectory(state, controls, 0.01)
        
        print(f"✓ 轨迹预测成功: 轨迹形状={trajectory.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 车辆动力学测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始测试路径跟踪系统...")
    
    tests = [
        test_vehicle_dynamics,
        test_mpc_optimization,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有测试通过！系统可以正常使用。")
    else:
        print("✗ 部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()
