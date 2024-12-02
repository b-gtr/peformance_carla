import gym
import gym_carla
import numpy as np
import torch
import cv2

from preprocess import Preprocessor
from soft_actor_critic import ParamsPool
from replay_buffer import ReplayBuffer, Transition

# Set up environment configuration
env_config = {
    'host': 'localhost',
    'port': 2000,
    'town': 'Town01',
    'verbose': False,
    'server_map': '/Game/Carla/Maps/Town01',
    'number_of_vehicles': 0,
    'number_of_pedestrians': 0,
    'display_size': 640,  # Width and height of the output image
    'obs_size': [640, 480],  # Observation image size [width, height]
    'window_size': 640,
    'max_past_step': 1,
    'dt': 0.05,
    'discrete': False,
    'continuous': True,
    'max_time_episode': 1000,
    'reward_fn': None,
    'encode_state_fn': None,
    'vehicles_list': None,
    'task_mode': 'straight',  # Task mode can be 'straight', 'left', 'right', etc.
    'random_seed': 0,
}

# Create environment
env = gym.make('carla-v0', params=env_config)

# Create agent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = ParamsPool(
    feature_dim=256,
    scalar_dim=1,  # Use speed as scalar input
    action_dim=2,  # Steering and throttle
    input_height=480,
    input_width=640,
    activate_scale=False,
    device=device
)

# Create replay buffer
buffer_capacity = 100000
replay_buffer = ReplayBuffer(capacity=buffer_capacity, device=device)

# Training loop parameters
num_episodes = 1000
max_steps_per_episode = 1000
batch_size = 64
update_every = 1  # Update agent every step

for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps_per_episode:
        # Extract image and scalars from observation
        image = obs[0]  # Assuming obs[0] contains the image
        speed = obs[1]['speed']  # Assuming obs[1] contains state info with 'speed'
        scalars = np.array([speed], dtype=np.float32)

        # Preprocess image: convert to grayscale, resize, normalize
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize to 640x480
        resized_image = cv2.resize(gray_image, (640, 480))
        # Normalize pixel values to [0, 1]
        resized_image = resized_image.astype(np.float32) / 255.0

        # Agent acts
        action = agent.act(resized_image, scalars)

        # Send action to environment
        # Environment expects actions in [-1, 1], which matches the agent's output
        env_action = action  # No scaling needed

        # Step in environment
        next_obs, reward, done, _ = env.step(env_action)

        # Extract next image and next scalars
        next_image = next_obs[0]
        next_speed = next_obs[1]['speed']
        next_scalars = np.array([next_speed], dtype=np.float32)

        # Preprocess next image
        next_gray_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
        next_resized_image = cv2.resize(next_gray_image, (640, 480))
        next_resized_image = next_resized_image.astype(np.float32) / 255.0

        # Store transition in replay buffer
        transition = Transition(
            img=resized_image,
            scalars=scalars,
            a=action,
            r=reward,
            n_img=next_resized_image,
            n_scalars=next_scalars,
            d=done
        )
        replay_buffer.push(transition)

        # Update agent
        if replay_buffer.ready_for(batch_size):
            batch = replay_buffer.sample(batch_size)
            agent.update_networks(batch)

        # Move to next time step
        obs = next_obs
        total_reward += reward
        step += 1

    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
