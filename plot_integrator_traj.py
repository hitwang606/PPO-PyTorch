import matplotlib.pyplot as plt
import numpy as np
from PPO import PPO
from integrator_env import FirstOrderIntegratorEnv

env = FirstOrderIntegratorEnv(dt=0.05, horizon=200, x0_range=(-0.5, 0.5), u_max=1.0)

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
    action_std_init=0.5
)

ppo_agent.load("PPO_preTrained/FirstOrderIntegrator/PPO_FirstOrderIntegrator_100000.pth")

x_list, xref_list, t_list = [], [], []

state = env.reset()
done = False
while not done:
    # 当前状态
    x = state[0]
    x_ref = state[1]
    t = len(t_list) * env.dt

    x_list.append(x)
    xref_list.append(x_ref)
    t_list.append(t)

    action = ppo_agent.select_action(state)
    state, reward, done, _ = env.step(action)

# 画图
t_arr = np.array(t_list)
plt.figure()
plt.plot(t_arr, xref_list, label="x_ref")
plt.plot(t_arr, x_list,  label="x")
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.legend()
plt.grid(True)
plt.show()
