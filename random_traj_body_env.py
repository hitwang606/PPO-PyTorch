# random_traj_2d_body_env.py
import gym
from gym import spaces
import numpy as np
from typing import Tuple


class RandomTraj2DBodyFrameEnv(gym.Env):
    """
    2D 一阶积分器：
        p_{k+1} = p_k + u_k * dt,  p,u ∈ R^2,  ||u|| ≲ v_max

    - 世界系 W 下有 2D 参考轨迹 p_ref^W(t) = [x_ref, y_ref]^T
    - 本体系 B 与 W 只有平移差异：
        e = p_ref^W - p_agent^W   (机体系下参考位置)
    - 观测在本体系下表示：
        obs = [e_x, e_y, v_ref_x, v_ref_y, u_prev_x, u_prev_y]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dt: float = 0.05,
        horizon: int = 200,
        v_max: float = 2.0,
        x0_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()

        self.dt = dt
        self.horizon = horizon
        self.v_max = v_max
        self.x0_range = x0_range

        # 观测：e(2) + v_ref(2) + u_prev(2) = 6 维
        obs_low = np.array(
            [-np.inf, -np.inf, -v_max, -v_max, -v_max, -v_max],
            dtype=np.float32,
        )
        obs_high = np.array(
            [np.inf, np.inf, v_max, v_max, v_max, v_max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # 动作：2 维速度，逐分量限幅 [-v_max, v_max]
        self.action_space = spaces.Box(
            low=np.array([-v_max, -v_max], dtype=np.float32),
            high=np.array([v_max, v_max], dtype=np.float32),
            dtype=np.float32,
        )

        # 内部状态：世界系下位置 p_W ∈ R^2
        self.p_W = np.zeros(2, dtype=np.float32)
        self.t = 0.0
        self.u_prev = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        # 当前 episode 的随机轨迹参数
        # 分别给 x、y 各搞一套多正弦参数
        self.traj_params_x = None
        self.traj_params_y = None

    # ============= 随机生成一条 2D 轨迹（世界系） =============
    def _sample_random_traj_1d(self):
        """
        生成 1D 轨迹参数：
            s(t) = b + A1 sin(w1 t + phi1) + A2 sin(w2 t + phi2)
        速度：
            v(t) = A1 w1 cos(...) + A2 w2 cos(...)
        """
        A1 = np.random.uniform(0.3, 0.8)
        A2 = np.random.uniform(0.0, 0.4)
        b = np.random.uniform(-0.5, 0.5)
        # 频率适当限制，保证速度不会太大
        w1 = np.random.uniform(0.2, 0.7)
        w2 = np.random.uniform(0.2, 1.0)
        phi1 = np.random.uniform(0.0, 2 * np.pi)
        phi2 = np.random.uniform(0.0, 2 * np.pi)
        return (A1, w1, A2, w2, b, phi1, phi2)

    def _sample_random_traj_2d(self):
        """
        为 x 和 y 各生成一套轨迹参数
        """
        self.traj_params_x = self._sample_random_traj_1d()
        self.traj_params_y = self._sample_random_traj_1d()

    def _traj_1d(self, t: float, params):
        A1, w1, A2, w2, b, phi1, phi2 = params
        s = b + A1 * np.sin(w1 * t + phi1) + A2 * np.sin(w2 * t + phi2)
        v = A1 * w1 * np.cos(w1 * t + phi1) + A2 * w2 * np.cos(w2 * t + phi2)
        return s, v

    def _traj_2d(self, t: float):
        """
        世界系下 2D 参考轨迹:
            p_ref^W(t) = [x_ref, y_ref]
            v_ref^W(t) = [vx_ref, vy_ref]
        """
        x_ref, vx_ref = self._traj_1d(t, self.traj_params_x)
        y_ref, vy_ref = self._traj_1d(t, self.traj_params_y)
        v_ref = np.array([vx_ref, vy_ref], dtype=np.float32)
        # 速度向量 clip 到 ||v|| <= v_max
        v_norm = np.linalg.norm(v_ref)
        if v_norm > self.v_max:
            v_ref = v_ref * (self.v_max / (v_norm + 1e-8))

        p_ref = np.array([x_ref, y_ref], dtype=np.float32)
        return p_ref, v_ref

    # ============= 世界系 -> 本体系观测 =============
    def _get_obs(self):
        p_ref_W, v_ref_W = self._traj_2d(self.t)
        e = p_ref_W - self.p_W  # 本体系下参考位置

        obs = np.concatenate([e, v_ref_W, self.u_prev]).astype(np.float32)
        return obs

    # ================= Gym 接口 =================
    def reset(self):
        # 每个 episode 随机一条 2D 参考轨迹
        self._sample_random_traj_2d()

        self.t = 0.0
        self.step_count = 0

        # 初始智能体位置：在 x0_range × x0_range 内
        # 也可以改成“在参考轨迹初始点附近”：
        # p_ref0, _ = self._traj_2d(0.0)
        # self.p_W = p_ref0 + np.random.uniform(-0.5, 0.5, size=2)
        self.p_W = np.random.uniform(
            low=self.x0_range[0], high=self.x0_range[1], size=2
        ).astype(np.float32)

        self.u_prev = np.zeros(2, dtype=np.float32)
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 2:
            raise ValueError(f"action dim should be 2, got {action.shape}")
        # 逐分量限幅
        u = np.clip(action, -self.v_max, self.v_max)

        # 一阶积分器更新（世界系）
        self.p_W = self.p_W + self.dt * u
        self.t += self.dt
        self.step_count += 1

        obs = self._get_obs()
        e = obs[0:2]
        p_ref_W, v_ref_W = self._traj_2d(self.t)

        # reward：位置误差 + 控制惩罚
        w_e = 2.0
        w_u = 0.005
        reward = - w_e * float(e @ e) - w_u * float(u @ u)

        self.u_prev = u
        done = self.step_count >= self.horizon

        info = {
            "p_W": self.p_W.copy(),
            "p_ref_W": p_ref_W.copy(),
            "e": e.copy(),
            "v_ref_W": v_ref_W.copy(),
            "u": u.copy(),
        }
        return obs, reward, done, info

    def render(self, mode="human"):
        p_ref_W, v_ref_W = self._traj_2d(self.t)
        print(f"t={self.t:.2f}, p_W={self.p_W}, p_ref_W={p_ref_W}, v_ref_W={v_ref_W}")
