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
env_params = {
    # ... your existing parameters ...
}
env = gym.make('carla-v0', params=env_params)

# Reset environment to get initial observation
obs = env.reset()

# Inspect the observation
print("Observation keys:", obs.keys())
print("Type of obs['state']:", type(obs['state']))
print("Contents of obs['state']:", obs['state'])

# Extract image and scalars from observation
image = obs['camera']
# Adjust extraction of scalars
try:
    speed = obs['state'].speed
    location = obs['state'].location
    scalars = np.array([speed, location.x, location.y])
except AttributeError as e:
    print("Error extracting scalars:", e)
    scalars = np.array([])  # Handle the error appropriately

# Proceed only if scalars are correctly extracted
if scalars.size == 0:
    print("Scalars could not be extracted. Exiting.")
    exit()

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

# ... rest of your training loop ...
