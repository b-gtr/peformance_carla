import gym
from gym import spaces
import numpy as np
import math
import random
import cv2
import carla
import threading

# ---------------------------------
# Helper Functions
# ---------------------------------
def vector_2d(vec_carla):
    return np.array([vec_carla.x, vec_carla.y], dtype=np.float32)

def distance_2d(a, b):
    return float(np.linalg.norm(a - b))

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def compute_lateral_offset(vehicle_transform, waypoint_transform):
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    dx = veh_loc.x - wp_loc.x
    dy = veh_loc.y - wp_loc.y

    forward = waypoint_transform.get_forward_vector()
    fx, fy = forward.x, forward.y

    cross_val = dx * fy - dy * fx
    return cross_val

def is_waypoint_behind(vehicle_transform, waypoint_transform):
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    forward = vehicle_transform.get_forward_vector()
    to_waypoint = wp_loc - veh_loc

    dot = (forward.x * to_waypoint.x + forward.y * to_waypoint.y + forward.z * to_waypoint.z)
    return dot < 0.0

# ---------------------------------
# Sensors
# ---------------------------------
class CollisionSensor:
    def __init__(self, vehicle, blueprint_library, world):
        self.vehicle = vehicle
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.history = []
        self._listen()

    def _listen(self):
        self.sensor.listen(lambda event: self.history.append(event))

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()

    def get_history(self):
        return self.history

class CameraSensor:
    def __init__(self, vehicle, blueprint_library, world, callback):
        self.vehicle = vehicle
        
        # Front-facing semantic camera
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')

        transform = carla.Transform(
            carla.Location(x=2.0, y=0.0, z=1.0),  # Front-facing
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

        self.sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        self.callback = callback
        self._listen()

    def _listen(self):
        self.sensor.listen(self.callback)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()

# ---------------------------------
# Main Environment
# ---------------------------------
class CarlaGymEnv(gym.Env):
    def __init__(self, host='localhost', port=2000, display=True):
        super().__init__()
        self.host = host
        self.port = port
        self.display = display

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        self.world = self.client.load_world('Town01')
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.image_lock = threading.Lock()
        self.camera_image = None

        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None
        self.next_waypoint = None

        self.wait_steps = 0
        self.wait_steps_total = int(3.0 / 0.05)

        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "segmentation": spaces.Box(low=0.0, high=1.0, shape=(480, 640, 1), dtype=np.float32),
            "dist_center": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "slip_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "speed": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })

        self.reset()

    def _init_vehicle_sensors(self):
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        spawn_point = random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)

        def camera_callback(image):
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            labels = array[:, :, 2].astype(np.float32) / 22.0
            labels = np.expand_dims(labels, axis=-1)
            with self.image_lock:
                self.camera_image = labels

        self.camera_sensor = CameraSensor(
            self.vehicle, self.blueprint_library, self.world, camera_callback
        )

        for _ in range(10):
            self.world.tick()

        self._pick_next_waypoint()

    def _clear_actors(self):
        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        self.camera_image = None

    def _pick_next_waypoint(self):
        if not self.vehicle:
            return

        veh_transform = self.vehicle.get_transform()
        current_wp = self.map.get_waypoint(veh_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        possible_next = current_wp.next(5.0) if current_wp else []

        if not possible_next:
            self.next_waypoint = None
            return

        self.next_waypoint = random.choice(possible_next)

    def get_vehicle_speed(self):
        if not self.vehicle:
            return 0.0
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def _get_slip_angle(self):
        if not self.vehicle:
            return 0.0
        velocity = self.vehicle.get_velocity()
        vx_world = velocity.x
        vy_world = velocity.y
        yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        vx = vx_world * math.cos(yaw) + vy_world * math.sin(yaw)
        vy = -vx_world * math.sin(yaw) + vy_world * math.cos(yaw)
        if abs(vx) < 1e-4:
            return math.copysign(math.pi/2, vy)
        return math.atan(vy / vx)

    def reset(self):
        self._clear_actors()
        valid_spawn_found = False
        while not valid_spawn_found:
            self._init_vehicle_sensors()
            if self.next_waypoint and not is_waypoint_behind(self.vehicle.get_transform(), self.next_waypoint.transform):
                valid_spawn_found = True
            else:
                self._clear_actors()
        self.wait_steps = self.wait_steps_total
        return self._get_obs()

    def step(self, action):
        if self.wait_steps > 0:
            self.wait_steps -= 1
            self.vehicle.apply_control(carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0))
            self.world.tick()
            return self._get_obs(), 0.0, False, {}

        steer = clamp(action[0], -0.5, 0.5)
        throttle = clamp((action[1] + 0.5), 0.0, 0.5)
        self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle))
        self.world.tick()

        reward, done, info = self._compute_reward_done_info()
        return self._get_obs(), reward, done, info

    def _compute_reward_done_info(self):
        info = {}
        done = False
        reward = 0.0

        if len(self.collision_sensor.get_history()) > 0:
            reward, done = -1.0, True
            info["collision"] = True
            return reward, done, info

        current_wp = self.map.get_waypoint(self.vehicle.get_transform().location, lane_type=carla.LaneType.Any)
        if not current_wp or current_wp.lane_type != carla.LaneType.Driving:
            reward, done = -1.0, True
            info["off_lane"] = True
            return reward, done, info

        lateral_offset = compute_lateral_offset(self.vehicle.get_transform(), current_wp.transform)
        if abs(lateral_offset) >= 1.0:
            reward, done = -1.0, True
            info["off_center"] = True
            return reward, done, info

        waypoint_yaw = current_wp.transform.rotation.yaw
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        eψ_deg = (vehicle_yaw - waypoint_yaw + 180) % 360 - 180
        eψ_deg_abs = abs(eψ_deg)

        slip_angle = self._get_slip_angle()
        eβ_deg_abs = abs(math.degrees(slip_angle))

        speed = self.get_vehicle_speed()

        # Reward components
        k1, k2 = 0.5, 0.1
        rey = math.exp(-k1 * abs(lateral_offset))

        if eψ_deg_abs < 90:
            reψ = math.exp(-k2 * eψ_deg_abs)
        else:
            reψ = -math.exp(-k2 * (180 - eψ_deg_abs))

        if eβ_deg_abs < 90:
            reβ = math.exp(-k2 * eβ_deg_abs)
        else:
            reβ = -math.exp(-k2 * (180 - eβ_deg_abs))

        reward = speed * (40 * rey + 40 * reψ + 20 * reβ)
        if speed < 6.0:
            reward *= 0.5

        reward = clamp(reward, -1.0, 1.0)
        return reward, done, info

    def _get_obs(self):
        with self.image_lock:
            seg_img = self.camera_image.copy() if self.camera_image is not None else np.zeros((480, 640, 1), dtype=np.float32)

        current_wp = self.map.get_waypoint(self.vehicle.get_transform().location, lane_type=carla.LaneType.Driving)
        lateral_offset = compute_lateral_offset(self.vehicle.get_transform(), current_wp.transform) if current_wp else 0.0
        slip_angle = self._get_slip_angle()
        speed = self.get_vehicle_speed()

        if self.display:
            self._show_image(seg_img)

        return {
            "segmentation": seg_img,
            "dist_center": np.array([lateral_offset], dtype=np.float32),
            "slip_angle": np.array([slip_angle], dtype=np.float32),
            "speed": np.array([speed], dtype=np.float32)
        }

    def _show_image(self, seg_img):
        gray = (seg_img[..., 0] * 255).astype(np.uint8)
        cv2.imshow("CARLA Front Camera", gray)
        cv2.waitKey(1)

    def close(self):
        self._clear_actors()
        self.world.apply_settings(self.original_settings)
        cv2.destroyAllWindows()
