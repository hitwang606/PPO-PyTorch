# # deploy_sim_3agents_2d_body.py
# import numpy as np
# import matplotlib.pyplot as plt

# from PPO import PPO

# dt = 0.05
# T_total = 10.0
# horizon = int(T_total / dt)
# v_max = 2.0

# # 训练好的 2D 策略模型
# model_path = "PPO_preTrained/RandomTraj2DBodyFrame/PPO_RandomTraj2DBodyFrame_100000.pth"

# NUM_AGENTS = 3


# # ========== 世界系 2D 参考轨迹生成（3 条不同轨迹） ==========
# def ref_traj_world_2d(agent_id: int, t: float):
#     """
#     为第 agent_id 个智能体生成世界系下的 2D 参考轨迹:
#         p_ref^W_i(t) = [x_ref, y_ref]
#         v_ref^W_i(t) = [vx_ref, vy_ref]

#     这里用：每个智能体一个相位/偏置不同的 Lissajous 风格轨迹。
#     部署时你可以改成任意规划出的轨迹（只要保证速度不过快）。
#     """
#     # 为不同 agent 设置不同的偏置和相位
#     # 方便肉眼区分三条轨迹
#     phase_shift = agent_id * (np.pi / 3.0)
#     bias_x = (agent_id - 1) * 0.5      # -0.5, 0, 0.5
#     bias_y = (agent_id - 1) * 0.5

#     # x 方向
#     A1x, w1x = 0.8, 0.5
#     A2x, w2x = 0.3, 1.0
#     x_ref = bias_x + A1x * np.sin(w1x * t + phase_shift) + \
#             A2x * np.sin(w2x * t + phase_shift + 0.5)
#     vx_ref = A1x * w1x * np.cos(w1x * t + phase_shift) + \
#              A2x * w2x * np.cos(w2x * t + phase_shift + 0.5)

#     # y 方向
#     A1y, w1y = 0.6, 0.4
#     A2y, w2y = 0.2, 0.9
#     y_ref = bias_y + A1y * np.sin(w1y * t + phase_shift + 1.0) + \
#             A2y * np.sin(w2y * t + phase_shift + 2.0)
#     vy_ref = A1y * w1y * np.cos(w1y * t + phase_shift + 1.0) + \
#              A2y * w2y * np.cos(w2y * t + phase_shift + 2.0)

#     p_ref = np.array([x_ref, y_ref], dtype=np.float32)
#     v_ref = np.array([vx_ref, vy_ref], dtype=np.float32)

#     # clip 速度模长，保证 ||v_ref|| <= v_max
#     v_norm = np.linalg.norm(v_ref)
#     if v_norm > v_max:
#         v_ref = v_ref * (v_max / (v_norm + 1e-8))

#     return p_ref, v_ref


# # ========== 加载 2D PPO 策略（共享给三个智能体） ==========
# def load_2d_velocity_tracker(model_path: str):
#     state_dim = 6   # [e_x,e_y, vref_x,vref_y, u_prev_x,u_prev_y]
#     action_dim = 2  # [u_x,u_y]

#     agent = PPO(
#         state_dim=state_dim,
#         action_dim=action_dim,
#         lr_actor=3e-4,
#         lr_critic=1e-3,
#         gamma=0.99,
#         K_epochs=10,
#         eps_clip=0.2,
#         has_continuous_action_space=True,
#         action_std_init=0.5,
#     )
#     agent.load(model_path)
#     return agent


# def run_deploy_3agents():
#     print("加载模型:", model_path)
#     ppo_agent = load_2d_velocity_tracker(model_path)

#     # ---------- 初始状态：3 个智能体 ----------
#     # 对每个 agent，让其初始位置在各自参考轨迹初始点附近
#     p_agents = np.zeros((NUM_AGENTS, 2), dtype=np.float32)
#     u_prev = np.zeros((NUM_AGENTS, 2), dtype=np.float32)

#     t0 = 0.0
#     for i in range(NUM_AGENTS):
#         p_ref0, v_ref0 = ref_traj_world_2d(i, t0)
#         # 初始位置在参考点附近 ± 0.1m
#         p_agents[i] = p_ref0 + np.random.uniform(-0.1, 0.1, size=2).astype(np.float32)

#     # ---------- 记录轨迹 ----------
#     t_list = []
#     pref_hist = []    # (T, NUM_AGENTS, 2)
#     pagent_hist = []  # (T, NUM_AGENTS, 2)
#     err_hist = []     # (T, NUM_AGENTS, 2)

#     # ---------- 仿真循环 ----------
#     for k in range(horizon):
#         t = k * dt

#         p_refs = np.zeros((NUM_AGENTS, 2), dtype=np.float32)
#         v_refs = np.zeros((NUM_AGENTS, 2), dtype=np.float32)
#         e_all = np.zeros((NUM_AGENTS, 2), dtype=np.float32)

#         # 1. 为每个智能体计算世界系参考轨迹 & 误差
#         for i in range(NUM_AGENTS):
#             p_ref_W, v_ref_W = ref_traj_world_2d(i, t)
#             p_refs[i] = p_ref_W
#             v_refs[i] = v_ref_W

#             # 世界系 -> 各自机体系：只保留平移
#             # e_i = p_ref_i^W - p_agent_i^W
#             e = p_ref_W - p_agents[i]
#             e_all[i] = e

#         # 2. 每个智能体独立构造观测 -> 调用同一个策略网络
#         u_cmds = np.zeros((NUM_AGENTS, 2), dtype=np.float32)
#         for i in range(NUM_AGENTS):
#             e = e_all[i]
#             v_ref = v_refs[i]
#             u_prev_i = u_prev[i]

#             # obs_i = [e_x, e_y, vref_x, vref_y, u_prev_x, u_prev_y]
#             obs = np.concatenate([e, v_ref, u_prev_i]).astype(np.float32)

#             # 部署时用无噪声策略（你需要在 PPO 中实现 select_action_eval）
#             action = ppo_agent.select_action_eval(obs)
#             u = np.array(action, dtype=np.float32)  # [u_x, u_y]
#             # 限幅
#             u = np.clip(u, -v_max, v_max)

#             u_cmds[i] = u

#         # 3. 用 2D 一阶动力学更新三个智能体的世界系位置
#         p_agents = p_agents + dt * u_cmds

#         # 4. 记录
#         t_list.append(t)
#         pref_hist.append(p_refs.copy())
#         pagent_hist.append(p_agents.copy())
#         err_hist.append(e_all.copy())
#         u_prev = u_cmds

#     # ---------- 转成数组 ----------
#     t_arr = np.array(t_list)                         # (T,)
#     pref_arr = np.stack(pref_hist, axis=0)          # (T, N, 2)
#     pagent_arr = np.stack(pagent_hist, axis=0)      # (T, N, 2)
#     err_arr = np.stack(err_hist, axis=0)            # (T, N, 2)

#     # ---------- 画 X/Y 随时间轨迹（每个智能体一条曲线） ----------
#     plt.figure(figsize=(10, 4))

#     # X(t)
#     plt.subplot(1, 2, 1)
#     for i in range(NUM_AGENTS):
#         plt.plot(t_arr, pref_arr[:, i, 0], linestyle="--", label=f"agent{i} x_ref")
#         plt.plot(t_arr, pagent_arr[:, i, 0], label=f"agent{i} x", alpha=0.8)
#     plt.xlabel("Time [s]")
#     plt.ylabel("X [m]")
#     plt.grid(True)
#     plt.legend(fontsize=8)

#     # Y(t)
#     plt.subplot(1, 2, 2)
#     for i in range(NUM_AGENTS):
#         plt.plot(t_arr, pref_arr[:, i, 1], linestyle="--", label=f"agent{i} y_ref")
#         plt.plot(t_arr, pagent_arr[:, i, 1], label=f"agent{i} y", alpha=0.8)
#     plt.xlabel("Time [s]")
#     plt.ylabel("Y [m]")
#     plt.grid(True)
#     plt.legend(fontsize=8)

#     plt.suptitle("3-Agent 2D Deployment (World Frame, X/Y vs Time)")
#     plt.tight_layout()
#     plt.show()

#     # ---------- 画 XY 平面轨迹 ----------
#     plt.figure()
#     for i in range(NUM_AGENTS):
#         plt.plot(pref_arr[:, i, 0], pref_arr[:, i, 1],
#                  linestyle="--", label=f"agent{i} p_ref")
#         plt.plot(pagent_arr[:, i, 0], pagent_arr[:, i, 1],
#                  label=f"agent{i} p", alpha=0.8)
#     plt.xlabel("X [m]")
#     plt.ylabel("Y [m]")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.legend(fontsize=8)
#     plt.title("3-Agent Position Trajectories in World Frame")
#     plt.tight_layout()
#     plt.show()

#     # ---------- 误差指标 ----------
#     # 对每个智能体分别统计 MSE 和最大误差
#     for i in range(NUM_AGENTS):
#         e_i = err_arr[:, i, :]           # (T,2)
#         e_norm = np.linalg.norm(e_i, axis=1)
#         mse_i = np.mean(e_norm**2)
#         max_e_i = np.max(e_norm)
#         print(f"[Agent {i}] MSE(||e||^2) = {mse_i:.4f}, max||e|| = {max_e_i:.4f}")


# if __name__ == "__main__":
#     run_deploy_3agents()


import numpy as np
import matplotlib.pyplot as plt
from PPO import PPO
import os

dt = 0.05
T_total = 20.0
horizon = int(T_total / dt)
v_max = 2.0
NUM_AGENTS = 3

model_path = "PPO_preTrained/RandomTraj2DBodyFrame/PPO_RandomTraj2DBodyFrame_300000.pth"


def equilateral_formation(L=1.0):
    p1 = np.array([0.0, 0.0], dtype=np.float32)
    p2 = np.array([L,   0.0], dtype=np.float32)
    p3 = np.array([0.5 * L, np.sqrt(3) / 2 * L], dtype=np.float32)
    return np.stack([p1, p2, p3], axis=0)


def compute_rel_desired(p_form):
    ref = np.zeros((NUM_AGENTS, NUM_AGENTS, 2), dtype=np.float32)
    for i in range(NUM_AGENTS):
        for j in range(NUM_AGENTS):
            if i == j:
                continue
            ref[i, j] = p_form[j] - p_form[i]
    return ref


def load_2d_velocity_tracker(path):
    state_dim = 6
    action_dim = 2
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        has_continuous_action_space=True,
        action_std_init=0.5,
    )
    agent.load(path)
    return agent


def run_deploy_formation():
    print("加载模型:", model_path)
    ppo_agent = load_2d_velocity_tracker(model_path)

    # —— 正三角队形相对位移 ref_ij —— 
    L = 2.0
    p_form = equilateral_formation(L=L)   # 只是为了算 ref_ij
    ref_ij = compute_rel_desired(p_form)  # ref_ij[i,j] = p_i* - p_j*

    # —— 初始化真实智能体位置 p_agents(0) —— 
    # 随机给一个初始形状，这里简单放在原点附近
    x0 = 0.0  # 这条直线的 x 坐标
    y_coords = np.arange(NUM_AGENTS, dtype=np.float32) * 1.5  # [0, 2, 4]，间隔为 2

    p_agents = np.stack([
        np.full(NUM_AGENTS, x0, dtype=np.float32),  # 全是 x0
        y_coords                                   # 不同的 y
    ], axis=1).astype(np.float32)
    # —— 初始化参考轨迹 p_ref(0)：从当前真实位置出发 —— 
    p_ref = p_agents.copy()

    u_prev = np.zeros((NUM_AGENTS, 2), dtype=np.float32)

    # 记录
    t_list = []
    pref_hist = []
    pagent_hist = []
    err_hist = []

    K_f = 0.2  # 队形控制律增益

    for k in range(horizon):
        t = k * dt

        # ====== (A) 用真实位置 p_agents 计算编队参考速度 v_ref ======
        v_ref = np.zeros((NUM_AGENTS, 2), dtype=np.float32)

        # 1号：v1_ref = (x1 - x2 - ref12) + (x1 - x3 - ref13)
        v_ref[0] = (p_agents[0] - p_agents[1] - ref_ij[0, 1]) + \
                   (p_agents[0] - p_agents[2] - ref_ij[0, 2])

        # 2号：v2_ref = (x2 - x1 - ref21) + (x2 - x3 - ref23)
        v_ref[1] = (p_agents[1] - p_agents[0] - ref_ij[1, 0]) + \
                   (p_agents[1] - p_agents[2] - ref_ij[1, 2])

        # 3号：v3_ref = (x3 - x1 - ref31) + (x3 - x2 - ref32)
        v_ref[2] = (p_agents[2] - p_agents[0] - ref_ij[2, 0]) + \
                   (p_agents[2] - p_agents[1] - ref_ij[2, 1])

        v_ref = np.array([0.1, 0.1]) - K_f * v_ref

        # 限制参考速度模长
        for i in range(NUM_AGENTS):
            v_norm = np.linalg.norm(v_ref[i])
            if v_norm > v_max:
                v_ref[i] = v_ref[i] * (v_max / (v_norm + 1e-8))

        # ====== (B) 对 v_ref 积分，得到参考轨迹 p_ref ======
        p_ref = p_ref + dt * v_ref

        # ====== (C) RL 智能体跟踪各自参考轨迹 ======
        u_cmds = np.zeros((NUM_AGENTS, 2), dtype=np.float32)
        e_all = np.zeros((NUM_AGENTS, 2), dtype=np.float32)

        for i in range(NUM_AGENTS):
            # 世界系 -> 机体系：只平移
            e = p_ref[i] - p_agents[i]
            e_all[i] = e

            obs = np.concatenate([e, v_ref[i], u_prev[i]]).astype(np.float32)

            action = ppo_agent.select_action_eval(obs)
            u = np.array(action, dtype=np.float32)
            u = np.clip(u, -v_max, v_max)

            u_cmds[i] = u

        # 一阶 2D 动力学更新真实智能体
        p_agents = p_agents + dt * u_cmds
        u_prev = u_cmds

        # 记录
        t_list.append(t)
        pref_hist.append(p_ref.copy())
        pagent_hist.append(p_agents.copy())
        err_hist.append(e_all.copy())


    # ====== 可视化 & 误差统计 ======
    t_arr = np.array(t_list)
    pref_arr = np.stack(pref_hist, axis=0)      # (T,3,2)
    pagent_arr = np.stack(pagent_hist, axis=0)  # (T,3,2)
    err_arr = np.stack(err_hist, axis=0)        # (T,3,2)
    np.save("pagent_arr.npy", pagent_arr)

    # X/Y vs time
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for i in range(NUM_AGENTS):
        plt.plot(t_arr, pref_arr[:, i, 0], '--', label=f"agent{i} x_ref")
        plt.plot(t_arr, pagent_arr[:, i, 0], label=f"agent{i} x", alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("X [m]")
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for i in range(NUM_AGENTS):
        plt.plot(t_arr, pref_arr[:, i, 1], '--', label=f"agent{i} y_ref")
        plt.plot(t_arr, pagent_arr[:, i, 1], label=f"agent{i} y", alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.suptitle("3-Agent Formation Reference Tracking (v_ref from p_agents)")
    plt.tight_layout()
    plt.show()

    # XY 平面轨迹
    plt.figure()
    for i in range(NUM_AGENTS):
        # plt.plot(pref_arr[:, i, 0], pref_arr[:, i, 1], '--', label=f"agent{i} p_ref")
        plt.plot(pagent_arr[:, i, 0], pagent_arr[:, i, 1], label=f"agent{i} p", alpha=0.8)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.title("3-Agent Position Trajectories (World Frame)")
    plt.tight_layout()
    plt.show()

    # 误差
    for i in range(NUM_AGENTS):
        e_i = err_arr[:, i, :]
        e_norm = np.linalg.norm(e_i, axis=1)
        mse_i = np.mean(e_norm**2)
        max_e_i = np.max(e_norm)
        print(f"[Agent {i}] MSE(||e||^2) = {mse_i:.4f}, max||e|| = {max_e_i:.4f}")


if __name__ == "__main__":
    run_deploy_formation()
