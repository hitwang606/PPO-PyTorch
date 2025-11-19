# test_integrator.py
import torch
from PPO import PPO
from integrator_env import FirstOrderIntegratorEnv

env_name = "FirstOrderIntegrator"
model_path = "PPO_preTrained/FirstOrderIntegrator/PPO_FirstOrderIntegrator_100000.pth"

env = FirstOrderIntegratorEnv(dt=0.05, horizon=200, x0_range=(-1.0, 1.0), u_max=1.0)

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

ppo_agent.load(model_path)  # 用你原来的 PPO.save / load

# 测几条轨迹的平均回报
num_episodes = 10
for ep in range(num_episodes):
    state = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        action = ppo_agent.select_action(state)  # 这里 用的还是带噪声的策略
        state, reward, done, info = env.step(action)
        ep_reward += reward
    print(f"[Test] Episode {ep+1}, reward = {ep_reward:.3f}")
