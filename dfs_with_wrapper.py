# main.py

import gym
import gym_carla
from soft_actor_critic import ParamsPool
from replay_buffer import ReplayBuffer, Transition
import torch
import numpy as np

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the environment
# Set the parameters for gym_carla
env_params = {
    'number_of_vehicles': 30,
    'number_of_walkers': 10,
    'display_size': 640,  # Screen size of bird-eye render
    'max_past_step': 1,   # The number of past steps to draw
    'dt': 0.1,  # Time interval between two frames
    'discrete': False,  # Whether to use discrete control space
    'discrete_acc': [1.0],  # Discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # Discrete value of steering angles
    'continuous_accel_range': [-1.0, 1.0],  # Continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # Continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # Filter for defining ego vehicle
    'port': 2000,  # Connection port
    'town': 'Town03',  # Which town to simulate
    'task_mode': 'random',  # Mode of the task
    'max_time_episode': 500,  # Maximum timesteps per episode
    'max_waypt': 12,  # Maximum number of waypoints
    'obs_range': 32,  # Observation range (meter)
    'lidar_bin': 0.125,  # Bin size of lidar sensor (meter)
    'd_behind': 12,  # Distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # Threshold for out of lane
    'desired_speed': 8,  # Desired speed (m/s)
    'max_ego_spawn_times': 200,  # Maximum times to spawn ego vehicle
    'display_route': True,  # Whether to display the route
    'pixor_size': 64,  # Size of the PIXOR image
}

env = gym.make('carla-v0', params=env_params)

# Reset environment to get initial observation
obs = env.reset()

# Extract image and scalars from observation
image = obs['camera']
scalars = obs['state']

# Get image dimensions
input_height = image.shape[0]
input_width = image.shape[1]
input_channels = image.shape[2]  # Should be 3 for RGB images

scalar_dim = len(scalars)
action_dim = env.action_space.shape[0]

feature_dim = 256  # Output size of Preprocessor

agent = ParamsPool(
    feature_dim=feature_dim,
    scalar_dim=scalar_dim,
    action_dim=action_dim,
    input_height=input_height,
    input_width=input_width,
    device=device
)

# Initialize replay buffer
buffer_capacity = 1000000
replay_buffer = ReplayBuffer(capacity=buffer_capacity, device=device)

# Training loop parameters
num_episodes = 1000
max_steps = 1000
batch_size = 256

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        # Extract image and scalars from observation
        image = obs['camera']
        scalars = obs['state']

        # Get action from agent
        action = agent.act(image, scalars)

        # Take step in the environment
        next_obs, reward, done, info = env.step(action)

        # Store transition in replay buffer
        next_image = next_obs['camera']
        next_scalars = next_obs['state']

        transition = Transition(
            img=image,
            scalars=scalars,
            a=action,
            r=reward,
            n_img=next_image,
            n_scalars=next_scalars,
            d=done
        )
        replay_buffer.push(transition)

        # Update agent if enough samples in buffer
        if replay_buffer.ready_for(batch_size):
            batch = replay_buffer.sample(batch_size)
            agent.update_networks(batch)

        # Prepare for next step
        obs = next_obs
        step += 1

    # Optionally save model after each episode
    agent.save_model('model_checkpoint.pth')
