# integrator_env.py
import gym
from gym import spaces
import numpy as np


class FirstOrderIntegratorEnv(gym.Env):
    """
    一阶积分器 x_{k+1} = x_k + u_k * dt
    任务：轨迹跟踪 x_ref(t)

    obs = [x, x_ref, v_ref, e]   e = x_ref - x
    act = u (标量), 物理含义是速度命令
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 dt: float = 0.05,
                 horizon: int = 200,
                 x0_range=(-1.0, 1.0),
                 u_max: float = 1.0):
        super().__init__()

        self.dt = dt
        self.horizon = horizon
        self.x0_range = x0_range
        self.u_max = u_max

        # 观测：x, x_ref, v_ref, e
        obs_low = np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 动作：标量 u ∈ [-u_max, u_max]
        self.action_space = spaces.Box(low=np.array([-u_max], dtype=np.float32),
                                       high=np.array([u_max], dtype=np.float32),
                                       dtype=np.float32)

        self.x = 0.0
        self.t = 0.0
        self.step_count = 0

    # 参考轨迹，可以以后换成别的
    def ref_traj(self, t: float):
        # 正弦参考轨迹
        # x_ref = sin(ω t), v_ref = ω cos(ω t)
        omega = 0.5
        x_ref = np.sin(omega * t)
        v_ref = omega * np.cos(omega * t)
        return x_ref, v_ref

    def _get_obs(self):
        x_ref, v_ref = self.ref_traj(self.t)
        e = x_ref - self.x
        obs = np.array([self.x, x_ref, v_ref, e], dtype=np.float32)
        return obs

    def reset(self):
        self.t = 0.0
        self.step_count = 0
        self.x = np.random.uniform(self.x0_range[0], self.x0_range[1])
        return self._get_obs()

    def step(self, action):
        # 处理动作形状并限幅
        action = np.asarray(action, dtype=np.float32)
        if action.ndim > 0:
            u = float(action[0])
        else:
            u = float(action)
        u = np.clip(u, -self.u_max, self.u_max)

        # 一阶积分器步进
        self.t += self.dt
        self.step_count += 1
        self.x = self.x + self.dt * u

        obs = self._get_obs()
        x_ref = obs[1]
        e = obs[3]

        # 奖励：跟踪误差 + 控制惩罚
        w_e = 1.0
        w_u = 0.01
        reward = - w_e * (e ** 2) - w_u * (u ** 2)

        done = self.step_count >= self.horizon
        info = {}

        return obs, float(reward), done, info

    def render(self, mode="human"):
        # 简单打印，可视化你可以自己用 matplotlib 画
        print(f"t={self.t:.2f}, x={self.x:.3f}")
