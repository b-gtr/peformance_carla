import gym
from gym import spaces
import numpy as np
import math
import random
import cv2
import carla
import threading

# Helper Functions
def vector_2d(vec_carla):
    return np.array([vec_carla.x, vec_carla.y], dtype=np.float32)

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def compute_lane_offset(seg_img):
    if seg_img is None or seg_img.sum() == 0:
        return 0.0
    bottom_row = seg_img[-50:, :, 0]
    road_pixels = np.where(bottom_row > 0.5)
    if len(road_pixels[1]) == 0:
        return 0.0
    left = np.min(road_pixels[1])
    right = np.max(road_pixels[1])
    midpoint = (left + right) / 2
    offset_pixels = midpoint - 320  # 640 width, center at 320
    return offset_pixels / 320.0  # Normalized to [-1, 1]

# Sensors
class CollisionSensor:
    # ... (same as original)

class CameraSensor:
    def __init__(self, vehicle, blueprint_library, world, callback):
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),  # Front-facing
            carla.Rotation(pitch=0.0)
        )
        self.sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        self.callback = callback
        self._listen()
    # ... (rest same as original)

# Environment
class CarlaGymEnv(gym.Env):
    def __init__(self, host='localhost', port=2000, display=True):
        # ... (initial setup same as original)
        self.observation_space = spaces.Dict({
            "segmentation": spaces.Box(0.0, 1.0, (480, 640, 1), dtype=np.float32),
            "lane_offset": spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
            "slip_angle": spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32),
            "speed": spaces.Box(0.0, np.inf, (1,), dtype=np.float32),
        })
        # ... (rest same as original)

    def get_slip_angle(self):
        vel = self.vehicle.get_velocity()
        yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        vx = vel.x * math.cos(yaw) + vel.y * math.sin(yaw)
        vy = -vel.x * math.sin(yaw) + vel.y * math.cos(yaw)
        return math.atan2(vy, vx) if (vx, vy) != (0, 0) else 0.0

    def _get_obs(self):
        with self.image_lock:
            seg_img = self.camera_image.copy() if self.camera_image is not None else np.zeros((480, 640, 1))
            lane_offset = compute_lane_offset(seg_img)
        return {
            "segmentation": seg_img,
            "lane_offset": np.array([lane_offset], dtype=np.float32),
            "slip_angle": np.array([self.get_slip_angle()], dtype=np.float32),
            "speed": np.array([self.get_vehicle_speed()], dtype=np.float32),
        }

    def _compute_reward_done_info(self):
        # ... (collision and off-lane checks same as original)
        obs = self._get_obs()
        lane_offset = obs["lane_offset"][0]
        slip_angle = abs(obs["slip_angle"][0])
        speed = obs["speed"][0]

        # Reward components
        lane_reward = math.exp(-0.5 * abs(lane_offset))  # k1=0.5
        slip_reward = math.exp(-0.1 * slip_angle)        # k2=0.1
        speed_factor = min(speed / 10.0, 1.0)            # Max speed ~36 km/h

        total_reward = speed_factor * (40 * lane_reward + 20 * slip_reward)
        if speed < 6.0:
            total_reward *= 0.5

        # Terminate if too far from lane center
        if abs(lane_offset) > 0.8:
            return -1.0, True, {"off_center": True}

        return total_reward, False, {}

    # ... (rest of the methods same as original)
