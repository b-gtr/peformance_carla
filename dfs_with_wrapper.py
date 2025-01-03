import gym
import gym_carla
import numpy as np
import torch
import cv2
import pygame  # Ensure pygame is imported if used elsewhere

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
    'random_seed': 0,  # Ensures reproducibility
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
    'number_of_vehicles': 0,  # No traffic vehicles
    'number_of_walkers': 0,   # No pedestrians
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
    
    # Display and Window Configuration
    'display_size': (640, 480),  # Changed from list to tuple
    'obs_size': (640, 480),      # Changed from list to tuple
    'window_size': (640, 480),   # Changed from list to tuple
    
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
    
    # Camera Configuration
    'rgb_cam': {  # This will be disabled since we are using semantic segmentation
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
    'semantic_segmentation_cam': {  # Added for semantic segmentation
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
    
    'use_image': False,  # Disable RGB image
    'use_depth_camera': False,
    'use_semantic_segmentation': True,  # Enable semantic segmentation
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
    'display_route': True,  # **Added Parameter** to display the route
    
    # Disable Additional Sensors
    'use_lidar': False,  # Disable LIDAR
    'lidar_params': {
        'range': 50.0,
        'channels': 32,
        'points_per_second': 100000,
        'rotation_frequency': 10.0,
        'upper_fov': 10.0,
        'lower_fov': -30.0,
        'dropoff_general_rate': 0.5,
        'dropoff_intensity_limit': 0.8,
    },
    'use_gps': False,  # Disable GPS
    'gps_params': {
        'sensor_tick': 0.05,
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
    },
    'use_imu': False,  # Disable IMU
    'imu_params': {
        'sensor_tick': 0.05,
        'accelerometer_noise_stddev': 0.02,
        'gyroscope_noise_stddev': 0.02,
    },
    'additional_sensors': [],  # No additional sensors
    'rendering_resolution': (640, 480),  # Changed from list to tuple
    'sensor_tick': 0.05,  # Global sensor tick rate
    'physics_tick': 50,  # Physics simulation tick rate
    'vehicle_mass': 1500,  # Mass of the ego vehicle in kg
    'vehicle_max_speed': 30.0,  # Max speed in m/s
    'spawn_points': [],  # Custom spawn points
    'town_map_path': '',  # Path to custom town map if any
    'traffic_density': 0.0,  # No traffic
    'pedestrian_density': 0.0,  # No pedestrians
    'recording': False,  # Disable recording
    'recording_path': 'recordings/',  # Path to save recordings
}

# Create environment
env = gym.make('carla-v0', params=env_config)

# Initialize Pygame Display (Ensure sizes are tuples)
try:
    pygame.init()
    self.display = pygame.display.set_mode(env_config['window_size'])  # Ensure 'window_size' is a tuple
except TypeError as e:
    print(f"Error initializing pygame display: {e}")
    pygame.quit()
    raise

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
        # Extract segmentation image and scalars from observation
        # Assuming obs is a dictionary with keys 'semantic_segmentation_cam' and 'state'
        segmentation_image = obs['semantic_segmentation_cam']  # Shape: (480, 640, 3)
        speed = obs['state']['speed']  # Speed in km/h
        scalars = np.array([speed], dtype=np.float32)

        # Preprocess segmentation image
        # **Important:** Semantic segmentation images use different colors for different classes.
        # Instead of converting to grayscale, it's better to map colors to class indices.

        # Define a color to class mapping (example, adjust based on your segmentation scheme)
        color_to_class = {
            (0, 0, 0): 0,            # Unknown
            (70, 70, 70): 1,         # Building
            (190, 153, 153): 2,      # Fence
            (72, 0, 90): 3,          # Other
            (220, 20, 60): 4,        # Pedestrian
            (153, 153, 153): 5,      # Pole
            (157, 234, 50): 6,       # Road
            (128, 64, 128): 7,       # Road markings
            (244, 35, 232): 8,        # Sidewalk
            (107, 142, 35): 9,       # Vegetation
            (0, 0, 142): 10,         # Vehicles
            (0, 0, 70): 11,          # Walls
            (0, 60, 100): 12,         # Traffic sign
            (0, 80, 100): 13,         # Traffic light
            (0, 0, 230): 14,          # Sky
            (119, 11, 32): 15,        # Ground
            # Add more mappings as per your semantic segmentation scheme
        }

        def color_to_label(segmentation_image, color_to_class_map):
            """
            Convert a color segmentation image to a label image.
            Args:
                segmentation_image (np.ndarray): HxWx3 image.
                color_to_class_map (dict): Mapping from RGB tuples to class indices.
            Returns:
                label_image (np.ndarray): HxW image with class indices.
            """
            label_image = np.zeros(segmentation_image.shape[:2], dtype=np.int32)
            for color, label in color_to_class_map.items():
                matches = np.all(segmentation_image == color, axis=-1)
                label_image[matches] = label
            return label_image

        # Convert segmentation image to label image
        segmentation_label = color_to_label(segmentation_image, color_to_class)
        # Resize to 640x480 if necessary (already 640x480 in this case)
        resized_label = cv2.resize(segmentation_label, env_config['obs_size'], interpolation=cv2.INTER_NEAREST)
        # Normalize label values if necessary (depends on your network)
        # For example, if you have N classes, you might leave them as integers [0, N-1]
        # Or normalize to [0, 1] by dividing by (N-1)
        num_classes = len(color_to_class)
        normalized_label = resized_label.astype(np.float32) / (num_classes - 1)

        # Agent acts
        action = agent.act(normalized_label, scalars)

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

        # Extract next segmentation image and next scalars
        next_segmentation_image = next_obs['semantic_segmentation_cam']
        next_speed = next_obs['state']['speed']
        next_scalars = np.array([next_speed], dtype=np.float32)

        # Preprocess next segmentation image
        next_segmentation_label = color_to_label(next_segmentation_image, color_to_class)
        next_resized_label = cv2.resize(next_segmentation_label, env_config['obs_size'], interpolation=cv2.INTER_NEAREST)
        next_normalized_label = next_resized_label.astype(np.float32) / (num_classes - 1)

        # Store transition in replay buffer
        transition = Transition(
            img=normalized_label,
            scalars=scalars,
            a=action,
            r=reward,
            n_img=next_normalized_label,
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
pygame.quit()
