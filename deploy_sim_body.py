# deploy_sim_2d_body.py
import numpy as np
import matplotlib.pyplot as plt

from PPO import PPO

dt = 0.05
T_total = 10.0
horizon = int(T_total / dt)
v_max = 2.0

model_path = "PPO_preTrained/RandomTraj2DBodyFrame/PPO_RandomTraj2DBodyFrame_100000.pth"


# =========== 世界系 2D 参考轨迹生成 ===========
def ref_traj_world_2d(t: float):
    """
    世界系下 2D 参考轨迹:
        p_ref^W(t) = [x_ref, y_ref]
        v_ref^W(t) = [vx_ref, vy_ref]
    这里给一个固定的 2D 复合正弦轨迹作为示例，
    部署时你可以换成任意规划出来的轨迹。
    """
    # x 方向
    A1x, w1x = 0.8, 0.5
    A2x, w2x = 0.3, 1.0
    bx, phix1, phix2 = 0.0, 0.0, 1.0

    x_ref = bx + A1x * np.sin(w1x * t + phix1) + A2x * np.sin(w2x * t + phix2)
    vx_ref = A1x * w1x * np.cos(w1x * t + phix1) + A2x * w2x * np.cos(w2x * t + phix2)

    # y 方向
    A1y, w1y = 0.6, 0.4
    A2y, w2y = 0.2, 0.9
    by, phiy1, phiy2 = 0.0, 0.5, 2.0

    y_ref = by + A1y * np.sin(w1y * t + phiy1) + A2y * np.sin(w2y * t + phiy2)
    vy_ref = A1y * w1y * np.cos(w1y * t + phiy1) + A2y * w2y * np.cos(w2y * t + phiy2)

    p_ref = np.array([x_ref, y_ref], dtype=np.float32)
    v_ref = np.array([vx_ref, vy_ref], dtype=np.float32)

    # clip 速度模长
    v_norm = np.linalg.norm(v_ref)
    if v_norm > v_max:
        v_ref = v_ref * (v_max / (v_norm + 1e-8))

    return p_ref, v_ref


# =========== 加载 PPO 策略（部署用） ===========
def load_2d_velocity_tracker(model_path: str):
    state_dim = 6   # [e_x,e_y,vx_ref,vy_ref,ux_prev,uy_prev]
    action_dim = 2  # [u_x,u_y]

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
    agent.load(model_path)
    return agent


def run_deploy_sim_2d():
    print("加载模型:", model_path)
    ppo_agent = load_2d_velocity_tracker(model_path)

    # 初始时刻
    t0 = 0.0
    p_ref0, v_ref0 = ref_traj_world_2d(t0)

    # 初始智能体位置设在参考轨迹初始位置附近（保持接近）
    p_agent = p_ref0 + np.random.uniform(-0.1, 0.1, size=2).astype(np.float32)
    u_prev = np.zeros(2, dtype=np.float32)

    # 记录
    t_list = []
    pref_list = []
    pagent_list = []
    u_list = []
    e_list = []

    for k in range(horizon):
        t = k * dt

        # 世界系参考轨迹
        p_ref_W, v_ref_W = ref_traj_world_2d(t)

        # 世界系 -> 机体系：仅平移
        # e = p_ref^W - p_agent^W
        e = p_ref_W - p_agent

        # 构造观测：obs = [e_x, e_y, vx_ref, vy_ref, u_prev_x, u_prev_y]
        obs = np.concatenate([e, v_ref_W, u_prev]).astype(np.float32)

        # 策略推理（部署阶段用无噪声接口）
        action = ppo_agent.select_action_eval(obs)  # 返回 list 长度 2
        u = np.array(action, dtype=np.float32)
        # 限幅
        u = np.clip(u, -v_max, v_max)

        # 一阶 2D 动力学更新
        p_agent = p_agent + dt * u

        # 记录
        t_list.append(t)
        pref_list.append(p_ref_W.copy())
        pagent_list.append(p_agent.copy())
        u_list.append(u.copy())
        e_list.append(e.copy())

        u_prev = u

    # 转成数组
    t_arr = np.array(t_list)
    pref_arr = np.stack(pref_list, axis=0)      # (N,2)
    pagent_arr = np.stack(pagent_list, axis=0)  # (N,2)
    e_arr = np.stack(e_list, axis=0)

    # ==== 画 X/Y 随时间轨迹 ====
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_arr, pref_arr[:, 0], label="x_ref")
    plt.plot(t_arr, pagent_arr[:, 0], label="x_agent", alpha=0.9)
    plt.xlabel("Time [s]")
    plt.ylabel("X [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_arr, pref_arr[:, 1], label="y_ref")
    plt.plot(t_arr, pagent_arr[:, 1], label="y_agent", alpha=0.9)
    plt.xlabel("Time [s]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.legend()

    plt.suptitle("2D Deployment (World Frame)")
    plt.tight_layout()
    plt.show()

    # ==== 也可以画 XY 平面轨迹 ====
    plt.figure()
    plt.plot(pref_arr[:, 0], pref_arr[:, 1], label="p_ref (world)")
    plt.plot(pagent_arr[:, 0], pagent_arr[:, 1], label="p_agent", alpha=0.9)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("2D Position Trajectory in World Frame")
    plt.tight_layout()
    plt.show()

    # 打印误差指标
    mse = np.mean(np.sum(e_arr**2, axis=1))
    max_err = np.max(np.linalg.norm(e_arr, axis=1))
    print(f"Deployment finished. MSE(||e||^2) = {mse:.4f}, max||e|| = {max_err:.4f}")


if __name__ == "__main__":
    run_deploy_sim_2d()
