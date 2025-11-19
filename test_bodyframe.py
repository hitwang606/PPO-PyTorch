# test_bodyframe.py
import numpy as np
from PPO import PPO
from integrator_body_env import FirstOrderBodyFrameEnv

env_name = "FirstOrderBodyFrame"
model_path = "PPO_preTrained/FirstOrderBodyFrame/PPO_FirstOrderBodyFrame_100000.pth"

# === 创建环境（参数要和训练时保持一致）===
env = FirstOrderBodyFrameEnv(
    dt=0.05,
    horizon=200,
    v_max=2.0,
    x0_range=(-1.0, 1.0),
    # traj_func=None   # 如果你训练时传了自定义轨迹函数，这里也要一起传
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# === 创建 PPO，超参数数值和训练时保持一致即可 ===
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

# 加载训练好的模型
ppo_agent.load(model_path)

# === 测试若干回合 ===
num_episodes = 10
rewards = []

for ep in range(num_episodes):
    state = env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        # 这里用的是带噪声的 select_action，如果你已经实现了
        # select_action_eval，可以替换为 eval 版本减少随机性
        action = ppo_agent.select_action_eval(state)

        state, reward, done, info = env.step(action)
        ep_reward += reward

    rewards.append(ep_reward)
    print(f"[Test] Episode {ep+1}, reward = {ep_reward:.3f}")

print("Mean reward:", np.mean(rewards), "Std:", np.std(rewards))
