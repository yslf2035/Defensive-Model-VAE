"""
自动驾驶轨迹跟踪系统
基于模型预测控制(MPC)的轨迹跟踪控制器
功能: 实现平滑的路径跟踪，输出每帧的坐标和速度
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 全局字体配置
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体族
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei']  # 中文字体
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class VehicleModel:
    """车辆动力学模型 - 自行车模型"""
    
    def __init__(self, wheelbase: float = 2.8, max_steer: float = 0.5, max_accel: float = 7.0):
        """
        初始化车辆模型参数
        
        Args:
            wheelbase: 轴距 (m)
            max_steer: 最大转向角 (rad)
            max_accel: 最大加速度 (m/s²)
        """
        self.L = wheelbase  # 轴距
        self.max_steer = max_steer
        self.max_accel = max_accel
        
    def dynamics(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        车辆动力学方程
        
        Args:
            state: [x, y, theta, v] - 位置x, 位置y, 航向角, 速度
            control: [a, delta] - 加速度, 转向角
            dt: 时间步长
            
        Returns:
            状态导数
        """
        x, y, theta, v = state
        a, delta = control
        
        # 限制控制输入
        a = np.clip(a, -self.max_accel, self.max_accel)
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        
        # 自行车模型动力学方程
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * np.tan(delta) / self.L
        dv = a
        
        return np.array([dx, dy, dtheta, dv])
    
    def predict_trajectory(self, initial_state: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
        """
        预测轨迹
        
        Args:
            initial_state: 初始状态
            controls: 控制序列 [N, 2]
            dt: 时间步长
            
        Returns:
            预测状态序列 [N+1, 4]
        """
        N = len(controls)
        states = np.zeros((N + 1, 4))
        states[0] = initial_state
        
        for i in range(N):
            state_derivative = self.dynamics(states[i], controls[i], dt)
            states[i + 1] = states[i] + state_derivative * dt
            
        return states


class PathInterpolator:
    """路径插值器"""
    
    def __init__(self, waypoints: np.ndarray, initial_state: np.ndarray):
        """
        初始化路径插值器
        
        Args:
            waypoints: 路径点 [N, 3] - [x, y, t]
        """
        self.waypoints = waypoints
        self.initial_state = initial_state
        self._create_interpolators()
    
    def _create_interpolators(self):
        """创建插值函数"""
        t = self.waypoints[:, 2]
        x = self.waypoints[:, 0]
        y = self.waypoints[:, 1]
        
        # 检查数据有效性
        if len(t) < 2:
            raise ValueError("至少需要2个路径点才能创建插值器")
        
        # 检查时间是否单调递增
        if not np.all(np.diff(t) > 0):
            raise ValueError("路径点的时间必须单调递增")
        
        # 根据数据点数量选择合适的插值方法
        n_points = len(t)
        
        # 位置插值
        try:
            if n_points >= 4:
                # 4个或更多点，尝试三次样条插值
                self.x_interp = interp1d(t, x, kind='cubic', bounds_error=False, fill_value='extrapolate')
                self.y_interp = interp1d(t, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
            elif n_points >= 3:
                # 3个点，使用二次插值
                self.x_interp = interp1d(t, x, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                self.y_interp = interp1d(t, y, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            else:
                # 2个点，使用线性插值
                self.x_interp = interp1d(t, x, kind='linear', bounds_error=False, fill_value='extrapolate')
                self.y_interp = interp1d(t, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        except Exception as e:
            # 如果插值失败，回退到线性插值
            print(f"警告: 插值失败，使用线性插值: {e}")
            self.x_interp = interp1d(t, x, kind='linear', bounds_error=False, fill_value='extrapolate')
            self.y_interp = interp1d(t, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # 计算速度插值
        if n_points >= 2:
            dt = np.diff(t)
            # 使用位置插值器获得在原始时间点上的平滑轨迹
            x_smooth = self.x_interp(t)
            y_smooth = self.y_interp(t)
            dx = np.diff(x_smooth)
            dy = np.diff(y_smooth)
            
            # 避免除零错误
            dt = np.where(dt == 0, 1e-6, dt)
            
            vx = dx / dt
            vy = dy / dt

            # 添加初始速度
            vx0 = self.initial_state[-2]
            vy0 = self.initial_state[-1]
            vx = np.concatenate((np.array([vx0]), vx))
            vy = np.concatenate((np.array([vy0]), vy))
            
            # 速度插值时间点
            t_vel = t[:-1] + dt/2
            t0 = 0.0
            t_vel = np.concatenate((np.array([t0]), t_vel))
            
            try:
                # 根据速度点数量选择插值方法
                n_vel_points = len(t_vel)
                if n_vel_points >= 4:
                    vel_interp_kind = 'cubic'
                elif n_vel_points >= 3:
                    vel_interp_kind = 'quadratic'
                else:
                    vel_interp_kind = 'linear'
                
                self.vx_interp = interp1d(t_vel, vx, kind=vel_interp_kind, bounds_error=False, fill_value='extrapolate')
                self.vy_interp = interp1d(t_vel, vy, kind=vel_interp_kind, bounds_error=False, fill_value='extrapolate')
            except Exception as e:
                # 如果速度插值失败，使用线性插值
                print(f"警告: 速度插值失败，使用线性插值: {e}")
                self.vx_interp = interp1d(t_vel, vx, kind='linear', bounds_error=False, fill_value='extrapolate')
                self.vy_interp = interp1d(t_vel, vy, kind='linear', bounds_error=False, fill_value='extrapolate')
        else:
            # 只有一个点，创建常值插值
            self.vx_interp = lambda t_val: 0.0
            self.vy_interp = lambda t_val: 0.0
    
    def get_reference(self, t: float) -> Tuple[float, float, float, float]:
        """
        获取参考轨迹点
        
        Args:
            t: 时间
            
        Returns:
            (x_ref, y_ref, vx_ref, vy_ref)
        """
        x_ref = float(self.x_interp(t))
        y_ref = float(self.y_interp(t))
        vx_ref = float(self.vx_interp(t))
        vy_ref = float(self.vy_interp(t))
        
        return x_ref, y_ref, vx_ref, vy_ref
    
    def get_reference_heading(self, t: float) -> float:
        """获取参考航向角"""
        vx_ref, vy_ref = self.get_reference(t)[2:4]
        return np.arctan2(vy_ref, vx_ref)


class MPCController:
    """模型预测控制器"""
    
    def __init__(self, vehicle_model: VehicleModel, prediction_horizon: int = 10, 
                 control_horizon: int = 5, dt: float = 0.01):
        """
        初始化MPC控制器
        
        Args:
            vehicle_model: 车辆模型
            prediction_horizon: 预测时域长度
            control_horizon: 控制时域长度
            dt: 时间步长
        """
        self.vehicle = vehicle_model
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.dt = dt
        
        # 验证参数
        if control_horizon > prediction_horizon:
            raise ValueError("控制时域不能大于预测时域")
        
        # 权重参数（仅跟踪航向角theta与速度v）
        self.Q = np.diag([20.0, 5.0])  # 状态权重 [theta, v]
        self.R = np.diag([1.0, 50.0])  # 控制增量权重 [Δa, Δdelta]（控制量变化量的权重）
        self.Qf = np.diag([20.0, 5.0])  # 终端状态权重
        
        # 保存上一次的控制输入，用于计算控制增量
        self.last_control = None
        
    def solve_mpc(self, current_state: np.ndarray, reference_trajectory: np.ndarray) -> np.ndarray:
        """
        求解MPC优化问题
        
        Args:
            current_state: 当前状态 [x, y, theta, v]
            reference_trajectory: 参考轨迹 [prediction_horizon+1, 2]，仅包含 [theta, v]
            
        Returns:
            最优控制序列 [control_horizon, 2]
        """
        # 初始控制猜测 - 只优化控制时域内的控制输入
        u0 = np.zeros((self.control_horizon, 2))
        # 如果有上一次的控制输入，将其作为第一个控制输入的初始猜测
        if self.last_control is not None:
            u0[0] = self.last_control.copy()
        
        # 定义目标函数
        def objective(u_flat):
            u = u_flat.reshape(self.control_horizon, 2)
            
            # 构建完整的控制序列
            # 控制时域内的控制输入 + 控制时域外的零控制输入
            full_u = np.zeros((self.prediction_horizon, 2))
            full_u[:self.control_horizon] = u
            # 控制时域外的控制输入保持为0（或最后一个控制输入）
            if self.control_horizon < self.prediction_horizon:
                # 使用最后一个控制输入填充剩余时域
                full_u[self.control_horizon:] = u[-1] if len(u) > 0 else np.zeros(2)
            
            # 预测轨迹
            states = self.vehicle.predict_trajectory(current_state, full_u, self.dt)
            
            # 计算代价（仅基于theta与v）
            cost = 0.0
            
            # 跟踪误差代价
            for i in range(self.prediction_horizon + 1):
                state_slice = states[i, 2:4]  # 提取theta与v
                state_error = state_slice - reference_trajectory[i]
                if i < self.prediction_horizon:
                    cost += state_error.T @ self.Q @ state_error
                else:
                    cost += state_error.T @ self.Qf @ state_error
            
            # 控制增量代价 - 惩罚控制量变化量 ||Δu||^2
            for i in range(self.control_horizon):
                if i == 0:
                    # 第一个控制输入：与前一个控制输入比较
                    if self.last_control is None:
                        # 如果没有前一个控制量，认为Δu=0（不增加代价）
                        delta_u = np.zeros(2)
                    else:
                        # 计算与前一个控制输入的差值
                        delta_u = u[i] - self.last_control
                else:
                    # 后续控制输入：与上一个控制输入比较
                    delta_u = u[i] - u[i-1]
                
                # 控制增量代价：||Δu||^2 = Δu^T @ R @ Δu
                cost += delta_u.T @ self.R @ delta_u
            
            return cost
        
        # 定义约束
        def constraint(u_flat):
            u = u_flat.reshape(self.control_horizon, 2)
            constraints = []
            
            # 控制约束 - 只对控制时域内的控制输入施加约束
            for i in range(self.control_horizon):
                constraints.append(self.vehicle.max_accel - u[i, 0])  # 加速度上限
                constraints.append(u[i, 0] + self.vehicle.max_accel)  # 加速度下限
                constraints.append(self.vehicle.max_steer - u[i, 1])  # 转向角上限
                constraints.append(u[i, 1] + self.vehicle.max_steer)  # 转向角下限
            
            return np.array(constraints)
        
        # 优化求解 - 只优化控制时域内的控制输入
        bounds = [
            (-self.vehicle.max_accel, self.vehicle.max_accel) for _ in range(self.control_horizon)
        ] + [
            (-self.vehicle.max_steer, self.vehicle.max_steer) for _ in range(self.control_horizon)
        ]
        
        result = minimize(
            objective, 
            u0.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint},
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if result.success:
            control_sequence = result.x.reshape(self.control_horizon, 2)
            # 更新上一次的控制输入，用于下次计算控制增量
            self.last_control = control_sequence[0].copy()
            return control_sequence
        else:
            print(f"MPC优化失败: {result.message}")
            # 优化失败时也更新last_control（如果之前有值）
            if self.last_control is not None:
                self.last_control = u0[0].copy()
            return u0


class PathTracker:
    """路径跟踪主控制器"""
    
    def __init__(self, waypoints: np.ndarray, initial_state: np.ndarray, 
                 wheelbase: float = 2.8, prediction_horizon: int = 10, 
                 control_horizon: int = 5, dt: float = 0.01):
        """
        初始化路径跟踪器
        
        Args:
            waypoints: 路径点 [N, 3] - [x, y, t]
            initial_state: 初始状态 [x, y, theta, v]
            wheelbase: 轴距
            prediction_horizon: MPC预测时域
            control_horizon: MPC控制时域
            dt: 时间步长
        """
        initial_state_copy = initial_state.copy()
        initial_state_copy[-2:] = np.sqrt(np.sum(initial_state_copy[-2:] ** 2))
        initial_state_copy = initial_state_copy[:-1]
        self.waypoints = waypoints
        self.current_state = initial_state_copy.copy()
        self.dt = dt
        
        # 初始化组件
        self.vehicle = VehicleModel(wheelbase=wheelbase)
        self.path_interp = PathInterpolator(waypoints, initial_state)
        self.mpc = MPCController(self.vehicle, prediction_horizon, control_horizon, dt)
        
        # 记录数据
        self.trajectory = [initial_state_copy.copy()]
        self.controls = []
        self.times = [0.0]
        
    def step(self, current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行一步路径跟踪
        
        Args:
            current_time: 当前时间
            
        Returns:
            (current_state, control) - 当前状态和控制输入
        """
        # 生成参考轨迹（仅航向角与速度）
        ref_trajectory = np.zeros((self.mpc.prediction_horizon + 1, 2))
        for i in range(self.mpc.prediction_horizon + 1):
            t_ref = current_time + i * self.dt
            x_ref, y_ref, vx_ref, vy_ref = self.path_interp.get_reference(t_ref)
            theta_ref = self.path_interp.get_reference_heading(t_ref)
            v_ref = np.sqrt(vx_ref**2 + vy_ref**2)
            ref_trajectory[i] = [theta_ref, v_ref]
        
        # 求解MPC
        control_sequence = self.mpc.solve_mpc(self.current_state, ref_trajectory)
        control = control_sequence[0]  # 使用第一个控制输入
        
        # 应用控制
        state_derivative = self.vehicle.dynamics(self.current_state, control, self.dt)
        self.current_state += state_derivative * self.dt
        
        # 记录数据
        self.trajectory.append(self.current_state.copy())
        self.controls.append(control.copy())
        self.times.append(current_time + self.dt)
        
        return self.current_state.copy(), control
    
    def run_simulation(self, total_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        运行完整仿真
        
        Args:
            total_time: 总仿真时间
            
        Returns:
            (times, states, controls) - 时间、状态、控制序列
        """
        num_steps = int(total_time / self.dt)
        
        print(f"开始路径跟踪仿真...")
        print(f"仿真时间: {total_time:.2f}s, 步数: {num_steps}, 时间步长: {self.dt:.3f}s")
        
        start_time = time.time()
        
        for i in range(num_steps):
            current_time = i * self.dt
            state, control = self.step(current_time)
            
            if i % 100 == 0:  # 每1秒打印一次进度
                print(f"时间: {current_time:.2f}s, 位置: ({state[0]:.2f}, {state[1]:.2f}), 速度: {state[3]:.2f}m/s")
        
        end_time = time.time()
        print(f"仿真完成! 用时: {end_time - start_time:.2f}s")
        
        return np.array(self.times), np.array(self.trajectory), np.array(self.controls)
    
    def plot_results(self, save_path: Optional[str] = None, axis_flip: str = 'none'):
        """绘制结果"""
        times = np.array(self.times)
        states = np.array(self.trajectory)
        controls = np.array(self.controls)
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 设置坐标轴刻度字体为Times New Roman（用于数字）
        def set_tick_font(ax):
            for label in ax.get_xticklabels():
                label.set_fontname('Times New Roman')
            for label in ax.get_yticklabels():
                label.set_fontname('Times New Roman')
        
        # 轨迹图
        axes[0, 0].plot(self.waypoints[:, 0], self.waypoints[:, 1], 'ro-', label='Reference Path', markersize=4)
        axes[0, 0].plot(states[:, 0], states[:, 1], 'b-', label='Actual Path', linewidth=2)
        axes[0, 0].set_xlabel('X (m)', fontsize=16)
        axes[0, 0].set_ylabel('Y (m)', fontsize=16)
        axes[0, 0].set_title('Path Tracking', fontsize=18)
        axes[0, 0].legend(fontsize=16)
        axes[0, 0].grid(True)
        axes[0, 0].axis('equal')
        # 根据axis_flip参数翻转坐标轴
        if axis_flip == 'x':
            axes[0, 0].invert_xaxis()
        elif axis_flip == 'y':
            axes[0, 0].invert_yaxis()
        elif axis_flip == 'xy':
            axes[0, 0].invert_xaxis()
            axes[0, 0].invert_yaxis()
        
        # 位置误差
        ref_x = np.array([self.path_interp.get_reference(t)[0] for t in times])
        ref_y = np.array([self.path_interp.get_reference(t)[1] for t in times])
        pos_error = np.sqrt((states[:, 0] - ref_x)**2 + (states[:, 1] - ref_y)**2)
        axes[0, 1].plot(times, pos_error, 'r-', linewidth=2)
        axes[0, 1].set_ylim(0, 5.0)
        axes[0, 1].set_xlabel('Time (s)', fontsize=16)
        axes[0, 1].set_ylabel('Δs (m)', fontsize=16)
        axes[0, 1].set_title('Position Error', fontsize=18)
        axes[0, 1].grid(True)
        
        # 速度曲线
        axes[0, 2].plot(times, states[:, 3], 'b-', label='Actual Velocity', linewidth=2)
        ref_v = np.array([np.sqrt(self.path_interp.get_reference(t)[2]**2 + 
                                  self.path_interp.get_reference(t)[3]**2) for t in times])
        axes[0, 2].plot(times, ref_v, 'r--', label='Reference Velocity', linewidth=2)
        axes[0, 2].set_xlabel('Time (s)', fontsize=16)
        axes[0, 2].set_ylabel('V (m/s)', fontsize=16)
        axes[0, 2].set_title('Velocity Tracking', fontsize=18)
        axes[0, 2].legend(fontsize=16)
        axes[0, 2].grid(True)
        
        # 航向角
        axes[1, 0].plot(times, np.degrees(states[:, 2]), 'b-', label='Actual Heading', linewidth=2)
        ref_theta = np.array([self.path_interp.get_reference_heading(t) for t in times])
        axes[1, 0].plot(times, np.degrees(ref_theta), 'r--', label='Reference Velocity', linewidth=2)
        # axes[1, 0].set_ylim(-0.1, 0.1)
        axes[1, 0].set_xlabel('Time (s)', fontsize=16)
        axes[1, 0].set_ylabel('Heading (°)', fontsize=16)
        axes[1, 0].set_title('Heading Tracking', fontsize=18)
        axes[1, 0].grid(True)
        
        # 控制输入 - 加速度
        axes[1, 1].plot(times[:-1], controls[:, 0], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)', fontsize=16)
        axes[1, 1].set_ylabel('Acc (m/s²)', fontsize=16)
        axes[1, 1].set_title('Acceleration', fontsize=18)
        axes[1, 1].grid(True)
        
        # 控制输入 - 转向角
        axes[1, 2].plot(times[:-1], np.degrees(controls[:, 1]), 'g-', linewidth=2)
        axes[1, 2].set_xlabel('Time (s)', fontsize=16)
        axes[1, 2].set_ylabel('Steering Angle (°)', fontsize=16)
        axes[1, 2].set_title('Steering Angle', fontsize=18)
        axes[1, 2].grid(True)
        
        # 为所有坐标轴设置Times New Roman字体（用于数字刻度）
        for ax in axes.flat:
            set_tick_font(ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"结果图已保存到: {save_path}")
        
        plt.show()


def create_test_path() -> np.ndarray:
    """创建测试路径"""
    # 创建一个S形路径
    t = np.linspace(0, 10, 50)
    x = t * 2
    y = 5 * np.sin(0.5 * t)
    
    waypoints = np.column_stack([x, y, t])
    return waypoints


def main():
    """主函数 - 演示路径跟踪"""
    print("=== 自动驾驶路径跟踪系统 ===")
    print("基于模型预测控制(MPC)的路径跟踪")
    
    # 创建测试路径
    waypoints = create_test_path()
    print(f"路径点数量: {len(waypoints)}")
    print(f"路径总时长: {waypoints[-1, 2]:.2f}s")
    
    # 初始状态 [x, y, theta, vx, vy]
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 2.0])  # 初始位置(0,0), 航向角0, 速度2m/s
    
    # 创建路径跟踪器
    tracker = PathTracker(
        waypoints=waypoints,
        initial_state=initial_state,
        wheelbase=2.8,  # 轴距2.8m
        prediction_horizon=10,  # 预测时域10步
        control_horizon=5,      # 控制时域5步
        dt=0.01                 # 时间步长0.01s
    )
    
    # 运行仿真
    total_time = waypoints[-1, 2] + 2.0  # 比路径时间长2秒
    times, states, controls = tracker.run_simulation(float(total_time))
    
    # 输出结果统计
    print("\n=== 仿真结果统计 ===")
    print(f"总仿真时间: {times[-1]:.2f}s")
    print(f"总步数: {len(times)}")
    print(f"平均计算时间: {(time.time() - time.time()) / len(times) * 1000:.2f}ms/步")
    
    # 计算跟踪误差
    ref_x = np.array([tracker.path_interp.get_reference(t)[0] for t in times])
    ref_y = np.array([tracker.path_interp.get_reference(t)[1] for t in times])
    pos_error = np.sqrt((states[:, 0] - ref_x)**2 + (states[:, 1] - ref_y)**2)
    
    print(f"最大位置误差: {np.max(pos_error):.3f}m")
    print(f"平均位置误差: {np.mean(pos_error):.3f}m")
    print(f"最终位置误差: {pos_error[-1]:.3f}m")
    
    # 绘制结果
    tracker.plot_results("pics/path_tracking_results.png")
    
    # 输出每帧数据示例（前10帧）
    print("\n=== 前10帧数据示例 ===")
    print("时间(s) | X(m) | Y(m) | 航向角(度) | 速度(m/s) | 加速度(m/s²) | 转向角(度)")
    print("-" * 80)
    
    for i in range(min(10, len(times))):
        t = times[i]
        state = states[i]
        if i < len(controls):
            control = controls[i]
            accel, steer = control
        else:
            accel, steer = 0.0, 0.0
        
        print(f"{t:6.3f} | {state[0]:5.2f} | {state[1]:5.2f} | {np.degrees(state[2]):8.1f} | {state[3]:7.2f} | {accel:9.2f} | {np.degrees(steer):9.1f}")


if __name__ == "__main__":
    main()
