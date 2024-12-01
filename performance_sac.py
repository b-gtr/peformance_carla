# test_algorithm.py

import gym
import numpy as np
import torch
from torch import nn, optim
from collections import deque
import matplotlib.pyplot as plt

# Import the modules you've provided
from preprocess import Preprocessor
from soft_actor_critic import ParamsPool
from replay_buffer import ReplayBuffer, Transition

def main():
    # Set up the environment
    env = gym.make('CarRacing-v2')
    env = gym.wrappers.GrayScaleObservation(env)  # Convert to grayscale
    env = gym.wrappers.ResizeObservation(env, (480, 640))  # Resize to match the preprocessor input

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    feature_dim = 256
    scalar_dim = 0  # Since we don't have additional scalar observations
    action_dim = env.action_space.shape[0]  # Should be 3 for CarRacing-v2
    input_height = 480
    input_width = 640
    total_episodes = 100
    max_steps_per_episode = 1000
    batch_size = 64
    start_training_after = 1000  # Number of steps before starting training
    update_every = 50  # Number of steps between updates
    num_updates = 50  # Number of updates per update step

    # Initialize the agent
    agent = ParamsPool(
        feature_dim=feature_dim,
        scalar_dim=scalar_dim,
        action_dim=action_dim,
        input_height=input_height,
        input_width=input_width,
        activate_scale=False,
        device=device
    )

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(capacity=100000, device=device)

    # For tracking rewards
    episode_rewards = []
    moving_average_rewards = []
    moving_average_window = 10  # Window size for moving average
    reward_deque = deque(maxlen=moving_average_window)
    total_steps = 0

    for episode in range(total_episodes):
        obs, info = env.reset()  # Updated for CarRacing-v2
        obs = obs / 255.0  # Normalize pixel values
        scalar_obs = np.array([])  # Empty since scalar_dim=0
        total_reward = 0
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < max_steps_per_episode:
            # Uncomment the following line to render the environment
            # env.render()

            # Select action from the agent
            action = agent.act(obs, scalar_obs)

            # Add exploration noise
            action = np.clip(action + np.random.normal(0, 0.1, size=action_dim), -1, 1)

            # Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)  # Updated for CarRacing-v2
            next_obs = next_obs / 255.0  # Normalize pixel values
            next_scalar_obs = np.array([])  # Empty since scalar_dim=0

            # Determine done flag
            done = terminated or truncated

            # Store the transition in the replay buffer
            transition = Transition(
                img=obs,
                scalars=scalar_obs,
                a=action,
                r=reward,
                n_img=next_obs,
                n_scalars=next_scalar_obs,
                d=float(done)
            )
            replay_buffer.push(transition)

            # Move to the next state
            obs = next_obs
            scalar_obs = next_scalar_obs
            total_reward += reward
            step += 1
            total_steps += 1

            # Update the networks after enough steps
            if len(replay_buffer.memory) > start_training_after and total_steps % update_every == 0:
                for _ in range(num_updates):
                    batch = replay_buffer.sample(batch_size)
                    agent.update_networks(batch)

        episode_rewards.append(total_reward)
        reward_deque.append(total_reward)
        moving_average = np.mean(reward_deque)
        moving_average_rewards.append(moving_average)

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, "
              f"Moving Average Reward = {moving_average:.2f}")

    env.close()

    # Plot the rewards
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(moving_average_rewards, label=f'{moving_average_window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance of SAC Agent')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
