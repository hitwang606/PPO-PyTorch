# train_random_2d_body.py
import os
from random_traj_body_env import RandomTraj2DBodyFrameEnv
from PPO import PPO

env_name = "RandomTraj2DBodyFrame"

log_dir = f"PPO_logs/{env_name}/"
os.makedirs(log_dir, exist_ok=True)
ckpt_dir = f"PPO_preTrained/{env_name}/"
os.makedirs(ckpt_dir, exist_ok=True)

env = RandomTraj2DBodyFrameEnv(
    dt=0.05,
    horizon=200,
    v_max=2.0,
    x0_range=(-1.0, 1.0),
)

state_dim = env.observation_space.shape[0]    # 6
action_dim = env.action_space.shape[0]       # 2
has_continuous_action_space = True

print(f"State dim: {state_dim}, Action dim: {action_dim}")

# PPO 超参数
max_training_timesteps = int(3e5)     # 先跑 1e5 看效果
max_ep_len = env.horizon

update_timestep = max_ep_len * 2
K_epochs = 10
eps_clip = 0.2
gamma = 0.99
lr_actor = 3e-4
lr_critic = 1e-3
action_std_init = 0.5

print_freq = max_ep_len * 5
save_model_freq = int(2e4)

ppo_agent = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    gamma=gamma,
    K_epochs=K_epochs,
    eps_clip=eps_clip,
    has_continuous_action_space=has_continuous_action_space,
    action_std_init=action_std_init,
)

print(f"开始训练 2D 一阶动力学智能体：{env_name} ...")
time_step = 0
i_episode = 0
log_running_reward = 0.0
log_running_episodes = 0

while time_step <= max_training_timesteps:
    i_episode += 1
    state = env.reset()
    current_ep_reward = 0.0

    for t in range(1, max_ep_len + 1):
        # 策略采样动作（训练阶段可以用带噪声的 select_action）
        action = ppo_agent.select_action(state)

        next_state, reward, done, info = env.step(action)

        # 写入 buffer
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        time_step += 1
        current_ep_reward += reward
        state = next_state

        # 更新 PPO
        if time_step % update_timestep == 0:
            ppo_agent.update()

        # 打印日志
        if time_step % print_freq == 0:
            avg_reward = log_running_reward / max(log_running_episodes, 1)
            print(
                f"Timesteps: {time_step:7d}  Episodes: {i_episode:5d}  "
                f"AvgReward: {avg_reward:8.3f}"
            )
            log_running_reward = 0.0
            log_running_episodes = 0

        # 保存模型
        if time_step % save_model_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"PPO_{env_name}_{time_step}.pth")
            print("保存模型到:", ckpt_path)
            ppo_agent.save(ckpt_path)

        if done:
            break

    log_running_reward += current_ep_reward
    log_running_episodes += 1

print("训练结束。")
