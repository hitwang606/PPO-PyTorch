# plot_bodyframe_traj.py
import numpy as np
import matplotlib.pyplot as plt
from PPO import PPO
from integrator_body_env import FirstOrderBodyFrameEnv

model_path = "PPO_preTrained/FirstOrderBodyFrame/PPO_FirstOrderBodyFrame_100000.pth"

env = FirstOrderBodyFrameEnv(
    dt=0.05,
    horizon=200,
    v_max=2.0,
    x0_range=(-0.5, 0.5),  # 可以收紧一点方便观察
    # traj_func=...
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ppo_agent = PPO(
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
ppo_agent.load(model_path)

# === 收集一条轨迹 ===
state = env.reset()
done = False

t_list, x_list, xref_list = [], [], []

step_idx = 0
while not done:
    # 记录当前世界系位置和参考位置
    info_dummy = {}
    # 用 env 内部的信息（更保险一点，我们从 step 返回的 info 里读）
    action = ppo_agent.select_action(state)
    next_state, reward, done, info = env.step(action)

    t = step_idx * env.dt
    t_list.append(t)
    x_list.append(info["x_W"])
    xref_list.append(info["x_ref_W"])

    state = next_state
    step_idx += 1

# === 画图 ===
t_arr = np.array(t_list)
x_arr = np.array(x_list)
xref_arr = np.array(xref_list)

plt.figure()
plt.plot(t_arr, xref_arr, label="x_ref (world)")
plt.plot(t_arr, x_arr, label="x (agent)", alpha=0.9)
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
