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
    'continuous_steer_range': [-1, 1],  # Continuous steering angle range
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

# Inspect the observation to understand its structure
print("Observation keys:", obs.keys())

# Extract image and scalars from observation
image = obs['camera']  # Assuming 'camera' is the key for the image data

# Since 'obs['state']' may contain complex data structures, we need to process it accordingly
# For example, if it contains the vehicle's speed and location, extract those values

# Initialize scalars as an empty list
scalars = []

# Check and extract the necessary scalar information
if 'state' in obs:
    state_info = obs['state']
    # Example: If state_info is a dictionary with keys 'speed', 'location', etc.
    if isinstance(state_info, dict):
        speed = state_info.get('speed', 0.0)
        # Assuming 'location' is a dictionary with 'x', 'y' coordinates
        location = state_info.get('location', {'x': 0.0, 'y': 0.0})
        if isinstance(location, dict):
            loc_x = location.get('x', 0.0)
            loc_y = location.get('y', 0.0)
        else:
            # Handle the case where location is not a dict
            loc_x, loc_y = 0.0, 0.0

        # Append the scalars to the list
        scalars.extend([speed, loc_x, loc_y])
    else:
        # If state_info is not a dict, handle accordingly
        pass
else:
    print("State information not found in observation.")
    scalars = [0.0, 0.0, 0.0]  # Default values

# Convert scalars to a NumPy array
scalars = np.array(scalars, dtype=np.float32)

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
    total_reward = 0.0

    while not done and step < max_steps:
        # Extract image and scalars from observation
        image = obs['camera']

        # Process the 'state' data to extract scalars
        scalars = []
        if 'state' in obs:
            state_info = obs['state']
            if isinstance(state_info, dict):
                speed = state_info.get('speed', 0.0)
                location = state_info.get('location', {'x': 0.0, 'y': 0.0})
                if isinstance(location, dict):
                    loc_x = location.get('x', 0.0)
                    loc_y = location.get('y', 0.0)
                else:
                    loc_x, loc_y = 0.0, 0.0

                scalars.extend([speed, loc_x, loc_y])
            else:
                pass  # Handle other types if necessary
        else:
            scalars = [0.0, 0.0, 0.0]  # Default values

        scalars = np.array(scalars, dtype=np.float32)

        # Ensure image is in the correct format (C, H, W)
        if image.shape[2] == 3:
            # Convert image from (H, W, C) to (C, H, W)
            image = np.transpose(image, (2, 0, 1))
        else:
            print("Unexpected number of image channels:", image.shape[2])
            break  # Exit the loop if image format is unexpected

        # Get action from agent
        action = agent.act(image, scalars)

        # Take step in the environment
        next_obs, reward, done, info = env.step(action)
        total_reward += reward

        # Extract next image and scalars
        next_image = next_obs['camera']

        # Process next 'state' data
        next_scalars = []
        if 'state' in next_obs:
            next_state_info = next_obs['state']
            if isinstance(next_state_info, dict):
                next_speed = next_state_info.get('speed', 0.0)
                next_location = next_state_info.get('location', {'x': 0.0, 'y': 0.0})
                if isinstance(next_location, dict):
                    next_loc_x = next_location.get('x', 0.0)
                    next_loc_y = next_location.get('y', 0.0)
                else:
                    next_loc_x, next_loc_y = 0.0, 0.0

                next_scalars.extend([next_speed, next_loc_x, next_loc_y])
            else:
                pass  # Handle other types if necessary
        else:
            next_scalars = [0.0, 0.0, 0.0]  # Default values

        next_scalars = np.array(next_scalars, dtype=np.float32)

        # Ensure next_image is in the correct format (C, H, W)
        if next_image.shape[2] == 3:
            next_image = np.transpose(next_image, (2, 0, 1))
        else:
            print("Unexpected number of next image channels:", next_image.shape[2])
            break  # Exit the loop if image format is unexpected

        # Store transition in replay buffer
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

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Optionally save model after each episode
    agent.save_model('model_checkpoint.pth')
