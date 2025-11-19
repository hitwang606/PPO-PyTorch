# integrator_body_env.py
import gym
from gym import spaces
import numpy as np
from typing import Callable, Tuple, Optional


class FirstOrderBodyFrameEnv(gym.Env):
    """
    一阶积分器：x_{k+1} = x_k + u_k * dt, |u_k| <= v_max

    - 世界系 W 中给出参考轨迹 x_ref(t), v_ref(t)，要求 |v_ref| < v_max
    - 本体系 B 与世界系 W 只有平移差异：p_B = p_W - x_agent
    - 强化学习的观测在本体系下表示（即“相对量”）
    - 动作 u 视为本体系下的速度控制（与世界系平移等价）
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dt: float = 0.05,
        horizon: int = 200,
        v_max: float = 2.0,
        x0_range: Tuple[float, float] = (-1.0, 1.0),
        traj_func: Optional[Callable[[float], Tuple[float, float]]] = None,
    ):
        """
        参数：
        - dt:      仿真步长
        - horizon: 每个 episode 的最大步数
        - v_max:   最大速度（动作/轨迹速度都不超过这个值）
        - x0_range: 初始位置范围（世界系）
        - traj_func: 参考轨迹生成函数 f(t) -> (x_ref(t), v_ref(t))
                     若为 None，则使用默认正弦轨迹
        """
        super().__init__()

        self.dt = dt
        self.horizon = horizon
        self.v_max = v_max
        self.x0_range = x0_range

        if traj_func is None:
            self.traj_func = self.default_traj
        else:
            self.traj_func = traj_func

        # -------- 观测空间（本体系）--------
        # 这里给一个比较通用的选择：
        # obs = [e, v_ref, u_prev]
        #   e     = x_ref^B = x_ref^W - x_agent^W  （位置误差 / 相对位置）
        #   v_ref = 参考轨迹速度（本体系 = 世界系，因为只有平移）
        #   u_prev= 上一时刻控制（便于策略利用“加速度”信息，可选）
        obs_low = np.array([-np.inf, -self.v_max, -self.v_max], dtype=np.float32)
        obs_high = np.array([np.inf, self.v_max, self.v_max], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # -------- 动作空间（本体系速度控制）--------
        # 动作为标量速度 u ∈ [-v_max, v_max]
        self.action_space = spaces.Box(
            low=np.array([-self.v_max], dtype=np.float32),
            high=np.array([self.v_max], dtype=np.float32),
            dtype=np.float32
        )

        # 内部状态：世界系下的位置 x_W、时间 t、本体系下上一时刻动作 u_prev
        self.x_W = 0.0
        self.t = 0.0
        self.u_prev = 0.0
        self.step_count = 0

    # ================= 世界系下的默认参考轨迹 =================
    def default_traj(self, t: float) -> Tuple[float, float]:
        """
        默认正弦轨迹 (世界系)：
            x_ref(t) = sin(ωt), v_ref(t) = ω cos(ωt)
        |v_ref| = |ω cos(ωt)| < 2 通过选择合适 ω 保证
        """
        omega = 0.5  # => max |v_ref| = 0.5 < 2
        x_ref = np.sin(omega * t)
        v_ref = omega * np.cos(omega * t)
        return x_ref, v_ref

    # ================== 世界系 -> 本体系观测变换 ==================
    def _get_obs(self) -> np.ndarray:
        """
        计算当前观测（本体系 B）：
            e = x_ref^B = x_ref^W - x_agent^W
            v_ref^B = v_ref^W   （只有平移，速度在两系中相同）
        """
        x_ref_W, v_ref_W = self.traj_func(self.t)
        e = x_ref_W - self.x_W  # 参考轨迹在本体系的坐标

        obs = np.array([e, v_ref_W, self.u_prev], dtype=np.float32)
        return obs

    # =================== Gym 接口 ===================
    def reset(self):
        """
        重置环境：
        - 随机初始位置（世界系）
        - 时间归零
        - 上一时刻控制设为 0
        """
        self.t = 0.0
        self.step_count = 0
        self.x_W = np.random.uniform(self.x0_range[0], self.x0_range[1])
        self.u_prev = 0.0
        return self._get_obs()

    def step(self, action):
        """
        输入：
        - action: PPO 输出的本体系速度控制 u_B
                  由于只有平移，本体系速度 = 世界系速度
        动力学：
            x_W^+ = x_W + dt * u
        奖励：
            r = - e^2 - λ_u u^2
        """
        # 处理动作形状并限幅
        action = np.asarray(action, dtype=np.float32)
        if action.ndim > 0:
            u = float(action[0])
        else:
            u = float(action)
        u = np.clip(u, -self.v_max, self.v_max)

        # 一阶积分器更新（世界系）
        self.x_W = self.x_W + self.dt * u
        self.t += self.dt
        self.step_count += 1

        obs = self._get_obs()
        e = float(obs[0])          # 本体系下位置误差
        v_ref = float(obs[1])      # 参考速度（可用于调试）

        # 奖励：跟踪误差 + 控制惩罚
        w_e = 1.0
        w_u = 0.01
        reward = - w_e * (e ** 2) - w_u * (u ** 2)

        # 更新上一时刻控制
        self.u_prev = u

        done = self.step_count >= self.horizon
        info = {
            "x_W": self.x_W,
            "x_ref_W": self.traj_func(self.t)[0],
            "e": e,
            "v_ref": v_ref,
            "u": u,
        }
        return obs, float(reward), done, info

    def render(self, mode="human"):
        # 简单打印，真正可视化建议用外部脚本画轨迹
        x_ref_W, v_ref_W = self.traj_func(self.t)
        print(f"t={self.t:.2f}, x_W={self.x_W:.3f}, x_ref_W={x_ref_W:.3f}")
