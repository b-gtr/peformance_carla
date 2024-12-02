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
    'client_timeout': 10.0,
    'traffic_manager_port': 8000,
    'traffic_manager_seed': 0,
    'traffic_manager_parameters': {'global_distance_to_leading_vehicle': 2.5},
    'sync_mode': True,
    'no_rendering_mode': False,
    'render': False,
    'verbose': False,
    'seed': 0,
    'weather': {
        'cloudiness': 0.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 0.0,
        'sun_azimuth_angle': 0.0,
        'sun_altitude_angle': 90.0,
        'fog_density': 0.0,
        'fog_distance': 0.0,
        'wetness': 0.0,
    },
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'disable_two_wheels': True,
    'task_mode': 'straight',  # Options: 'random', 'straight', 'left', 'right'
    'max_time_episode': 1000,
    'max_waypt': 12,
    'obs_range': 32.0,
    'lidar_bin': 0.125,
    'd_behind': 12.0,
    'out_lane_thres': 2.0,
    'desired_speed': 7.0,  # km/h
    'max_ego_spawn_times': 200,
    'display_size': 640,  # Screen size of bird-eye render
    'obs_size': [640, 480],  # Observation size (width, height)
    'window_size': 800,  # Screen size of the pygame window
    'max_past_step': 1,
    'dt': 0.05,
    'discrete': False,
    'discrete_acc': [1.0],  # Not used when 'discrete' is False
    'discrete_steer': [-0.2, 0.0, 0.2],  # Not used when 'discrete' is False
    'continuous_accel_range': [-1.0, 1.0],  # For continuous actions
    'continuous_steer_range': [-1.0, 1.0],
    'ego_vehicle_filter': 'vehicle.lincoln*',
    'traffic_vehicle_filter': 'vehicle.*',
    'walker_filter': 'walker.pedestrian.*',
    'collision_sensor': True,
    'lane_invasion_sensor': True,
    'manual_control': False,
    'rgb_cam': {
        'x': 1.5,  # Relative position in meters
        'y': 0.0,
        'z': 2.4,
        'roll': 0.0,  # In degrees
        'pitch': 0.0,
        'yaw': 0.0,
        'width': 640,
        'height': 480,
        'fov': 100,
        'sensor_tick': 0.05,
    },
    'use_image': True,
    'use_depth_camera': False,
    'use_semantic_segmentation': False,
    'early_termination': True,
    'reward_type': 'speed',  # Options: 'speed', 'distance', 'custom'
    'reward_weights': {
        'collision': -100.0,
        'speed': 1.0,
        'steer': -0.05,
        'out_of_lane': -10.0,
        'distance': 1.0,
    },
    'continuous': True,
    'random_spawn': False,
    'dynamic': False,
    'traffic_light': False,
    'hybrid': False,
    'behavior': 'normal',  # Options: 'cautious', 'normal', 'aggressive'
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
        # Assuming obs is a dictionary with keys 'camera' and 'state'
        image = obs['camera']  # Shape: (480, 640, 3)
        speed = obs['state']['speed']  # Speed in km/h
        scalars = np.array([speed], dtype=np.float32)

        # Preprocess image: convert to grayscale, resize, normalize
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize to 640x480 if necessary (already 640x480 in this case)
        resized_image = cv2.resize(gray_image, (640, 480))
        # Normalize pixel values to [0, 1]
        resized_image = resized_image.astype(np.float32) / 255.0

        # Agent acts
        action = agent.act(resized_image, scalars)

        # Send action to environment
        # Map agent's action to environment action space
        # Agent's action: [steering (-1 to 1), throttle (-1 to 1)]
        # Environment expects:
        #   steering: -1 to 1
        #   throttle: 0 to 1
        #   brake: 0 or 1
        # Map throttle to [0, 1]
        throttle = np.clip(action[1], 0.0, 1.0)
        env_action = {
            'steer': float(action[0]),  # Ensure it's a Python float
            'throttle': float(throttle),
            'brake': 0.0
        }

        # Step in environment
        next_obs, reward, done, info = env.step(env_action)

        # Extract next image and next scalars
        next_image = next_obs['camera']
        next_speed = next_obs['state']['speed']
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

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
