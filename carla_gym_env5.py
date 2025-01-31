import gym
from gym import spaces
import numpy as np
import math
import random
import cv2
import carla
import threading
import json

# ---------------------------------
# Hilfsfunktionen
# ---------------------------------
def vector_2d(vec_carla):
    """Konvertiere carla.Vector3D -> (x, y) in float."""
    return np.array([vec_carla.x, vec_carla.y], dtype=np.float32)

def distance_2d(a, b):
    """Euklidischer 2D-Abstand."""
    return float(np.linalg.norm(a - b))

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def compute_lateral_offset(vehicle_transform, waypoint_transform):
    """
    Berechnet die laterale Distanz (seitlicher Versatz) des Fahrzeugs
    zum Center des Waypoint. Ignoriert z-Höhenunterschiede.
    """
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    dx = veh_loc.x - wp_loc.x
    dy = veh_loc.y - wp_loc.y

    forward = waypoint_transform.get_forward_vector()
    fx, fy = forward.x, forward.y

    # 2D-Kreuzprodukt => dx*fy - dy*fx
    cross_val = dx * fy - dy * fx
    return cross_val  # positive oder negative Werte möglich

def is_waypoint_behind(vehicle_transform, waypoint_transform):
    """
    Prüft, ob der Waypoint "hinter" dem Fahrzeug liegt, via Skalarprodukt
    zwischen Vorwärtsrichtung und Vektor Fahrzeug->Waypoint (< 0 => hinten).
    """
    veh_loc = vehicle_transform.location
    wp_loc = waypoint_transform.location

    forward = vehicle_transform.get_forward_vector()
    to_waypoint = wp_loc - veh_loc

    dot = (forward.x * to_waypoint.x
           + forward.y * to_waypoint.y
           + forward.z * to_waypoint.z)
    return dot < 0.0

# ---------------------------------
# Sensoren
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
        
        # Semantische Kamera mit Birdview (Top-Down),
        # aber deutlich näher (z=12.0 statt 20).
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')

        # Hier wird der Transform geändert, damit wir eine Vogelperspektive (Top-Down) haben
        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=12.0),  # etwas näher zum Fahrzeug
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
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
# Haupt-Env
# ---------------------------------
class CarlaGymEnv(gym.Env):
    """
    Environment, das:
      - semantische Kamera (480x640 Birdview),
      - Distanz zur Fahrbahnmitte,
      - GPS-Koordinate zum nächsten Waypoint,
      - eigene GPS-Koordinate,
      - Geschwindigkeit
    als Dictionary-Observation zurückgibt.
    """
    def __init__(self, host='localhost', port=2000, display=True, route_file='recorded_routes/recorded_route.json'):
        super().__init__()
        self.host = host
        self.port = port
        self.display = display  # Zeige Kamerabild per OpenCV an
        self.route_file = route_file

        # Verbinde mit CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        # Lade die Town10HD_Opt Karte
        self.world = self.client.load_world('Town01')
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # Lade die generierte Route
        with open(self.route_file, 'r') as f:
            route_data = json.load(f)
        self.route_transforms = []
        for wp in route_data['waypoints']:
            location = carla.Location(x=wp['x'], y=wp['y'], z=wp['z'])
            rotation = carla.Rotation(yaw=wp['yaw'])
            transform = carla.Transform(location, rotation)
            self.route_transforms.append(transform)
        self.current_target_idx = 1  # Starte mit dem ersten Ziel nach dem Spawn-Punkt

        # Synchronous Mode
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # Lock für die Kamera
        self.image_lock = threading.Lock()
        self.camera_image = None

        # Actoren
        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None

        # Nächster Waypoint
        self.next_waypoint_transform = None

        # 3 Sekunden Warten nach Reset (bei 20 FPS => 3 / 0.05 = 60 Ticks)
        self.wait_steps = 0
        self.wait_steps_total = int(3.0 / settings.fixed_delta_seconds)

        # Action Space: [steer, throttle] in [-0.5, 0.5]
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(2,), dtype=np.float32)

        # Observation Space
        self.observation_space = spaces.Dict({
            "segmentation": spaces.Box(
                low=0.0, high=1.0, shape=(480, 640, 1), dtype=np.float32
            ),
            "dist_center": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "gps_next_waypoint": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            "gps_own": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            "speed": spaces.Box(
                low=0.0, high=np.inf, shape=(1,), dtype=np.float32
            ),
        })

        # Zu Beginn einmal reset
        self.reset()

    # ---------------------------------
    # Hilfsfunktionen
    # ---------------------------------
    def _init_vehicle_sensors(self):
        # Fahrzeug spawnen
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        if len(self.route_transforms) == 0:
            raise ValueError("No route waypoints available for spawning.")
        # Use the first spawn point from the map
        spawn_transform = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)

        # Kollision
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)

        # Kamera (Birdview semantisch)
        def camera_callback(image):
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # => (480,640,4)

            # Red-Kanal hat die semantischen Labels
            labels = array[:, :, 2].astype(np.float32)
            labels /= 22.0  # Normierung auf [0,1]

            labels = np.expand_dims(labels, axis=-1)  # (480,640)->(480,640,1)
            with self.image_lock:
                self.camera_image = labels

        self.camera_sensor = CameraSensor(
            self.vehicle, self.blueprint_library, self.world, camera_callback
        )

        # Warmlaufen lassen
        for _ in range(10):
            self.world.tick()

        # Setze den nächsten Waypoint
        self._pick_next_waypoint()

    def _clear_actors(self):
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        self.camera_image = None

    def _pick_next_waypoint(self):
        """Setzt self.next_waypoint_transform auf den nächsten Wegpunkt in der Route."""
        if self.current_target_idx < len(self.route_transforms):
            self.next_waypoint_transform = self.route_transforms[self.current_target_idx]
            self.current_target_idx += 1
        else:
            self.next_waypoint_transform = None

    def get_vehicle_speed(self):
        """Geschwindigkeit in m/s."""
        if not self.vehicle:
            return 0.0
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    # ---------------------------------
    # Gym-Methoden
    # ---------------------------------
    def reset(self):
        """
        Reset-Methode, die so lange neu spawnt, bis Waypoint NICHT hinter dem Fahrzeug liegt.
        Dann warten wir 3s (self.wait_steps).
        """
        self._clear_actors()
        self.current_target_idx = 1  # Reset the waypoint index for the new episode
        valid_spawn_found = False
        while not valid_spawn_found:
            # Init (spawnt Fahrzeug + Sensoren und ruft _pick_next_waypoint() auf)
            self._init_vehicle_sensors()

            if self.next_waypoint_transform is None:
                # Safety: Kein Waypoint gefunden => nochmal neu
                self._clear_actors()
                continue

            # Check: Liegt der Waypoint hinter dem Fahrzeug?
            if is_waypoint_behind(self.vehicle.get_transform(), self.next_waypoint_transform):
                print("Spawn ungünstig (Waypoint direkt hinter Fahrzeug). Versuche neuen Spawn...")
                self._clear_actors()
            else:
                # Alles gut
                valid_spawn_found = True

        # Nach erfolgreichem Spawn: Wartezeit für 3 Sekunden
        self.wait_steps = self.wait_steps_total

        # Geben wir zunächst die Observation zurück
        return self._get_obs()

    def step(self, action):
        """
        Während self.wait_steps > 0 ignorieren wir das Action-Handling
        und geben reward=0 zurück (vehicle bleibt stehen).
        """
        # 1. Falls wir noch warten: setze Fahrzeugcontrol = 0, verringere wait_steps, reward=0
        if self.wait_steps > 0:
            self.wait_steps -= 1
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.vehicle.apply_control(control)

            self.world.tick()
            obs = self._get_obs()
            # No reward, not done
            return obs, 0.0, False, {}

        # 2. Normale Aktion ausführen
        steer = float(clamp(action[0], -0.5, 0.5))
        throttle = float(clamp(action[1], -0.5, 0.5))
        # Skaliere Throttle von -0.5..+0.5 => 0..1
        throttle = (throttle + 0.5)
        throttle = clamp(throttle, 0.0, 0.5)  # Safety

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        self.vehicle.apply_control(control)

        # Einen Tick simulieren
        self.world.tick()

        # Reward, Done, Info berechnen
        reward, done, info = self._compute_reward_done_info()

        # Update Waypoint nur, wenn nicht done
        if not done and self.next_waypoint_transform is not None:
            dist = distance_2d(
                vector_2d(self.vehicle.get_transform().location),
                vector_2d(self.next_waypoint_transform.location)
            )
            if dist < 2.0:
                print(f"Fahrzeug ist nah genug am Waypoint (dist={dist:.2f}). Neuen Waypoint setzen.")
                self._pick_next_waypoint()
            elif is_waypoint_behind(self.vehicle.get_transform(), self.next_waypoint_transform):
                print("Waypoint bereits hinter uns. Nächsten Waypoint wählen.")
                self._pick_next_waypoint()

        obs = self._get_obs()
        return obs, reward, done, info

    def _compute_reward_done_info(self):
        """Berechnet Reward und Done."""
        info = {}
        done = False
        reward = 0.0

        # Check if route is completed
        if self.current_target_idx >= len(self.route_transforms) and self.next_waypoint_transform is None:
            print("Route completed successfully!")
            reward = 1.0
            done = True
            info["route_completed"] = True
            return reward, done, info

        # 1) Kollision => sofort terminiert + starker Malus
        if len(self.collision_sensor.get_history()) > 0:
            print(">>> Kollision erkannt, Episode terminiert!")
            reward = -1.0
            done = True
            info["collision"] = True
            return reward, done, info

        # 2) Hole aktuelles Waypoint und prüfe LaneType
        current_wp = self.map.get_waypoint(
            self.vehicle.get_transform().location,
            lane_type=carla.LaneType.Any
        )

        # Falls man auf Bordstein, Gehweg o.Ä. gerät => sofortiger Abbruch + starker Malus
        if current_wp.lane_type != carla.LaneType.Driving:
            print(">>> Fahrzeug auf falscher Lane (z.B. Bordstein/Gehweg). Episode terminiert!")
            reward = -1.0
            done = True
            info["off_lane"] = True
            return reward, done, info

        # 3) Distanz zur Fahrbahnmitte => Reward
        lateral_offset = compute_lateral_offset(
            self.vehicle.get_transform(),
            current_wp.transform
        )
        offset_magnitude = abs(lateral_offset)
        max_offset = 1.0  # verschärfter Threshold

        # Wenn das Fahrzeug den Bordstein berührt/weiter als 1m weg -> sofort DONE + Straf-Reward
        if offset_magnitude >= max_offset:
            print(">>> Zu weit von Fahrbahnmitte (Bordstein berührt?). Episode terminiert!")
            reward = -1.0
            done = True
            info["off_center"] = True
            return reward, done, info

        # Sonst normaler Spurhaltungsreward
        dist_center_reward = 0.5 * (1.0 - offset_magnitude / max_offset)

        # 4) Geschwindigkeit (m/s)
        speed = self.get_vehicle_speed()
        if speed < 0.1:
            speed_reward = -0.3
        else:
            capped_speed = min(speed, 10.0)
            speed_reward = 0.5 * (capped_speed / 10.0)

        # Gesamtreward
        reward = dist_center_reward + speed_reward
        # clamp auf [-1,1]
        reward = clamp(reward, -1.0, 1.0)

        print(f"reward: {reward}")

        return reward, done, info

    def _get_obs(self):
        """
        Dictionary-Observation:
          {
             "segmentation": (480,640,1),
             "dist_center": (1,),
             "gps_next_waypoint": (2,),
             "gps_own": (2,),
             "speed": (1,),
          }
        """
        with self.image_lock:
            if self.camera_image is None:
                seg_img = np.zeros((480, 640, 1), dtype=np.float32)
            else:
                seg_img = self.camera_image.copy()

        # Bild anzeigen (falls display=True)
        if self.display:
            self._show_image(seg_img)

        # Distanz zur Mitte
        current_wp = self.map.get_waypoint(
            self.vehicle.get_transform().location,
            lane_type=carla.LaneType.Driving
        )
        if current_wp is not None:
            lateral_offset = compute_lateral_offset(
                self.vehicle.get_transform(),
                current_wp.transform
            )
        else:
            lateral_offset = 0.0

        dist_center = np.array([lateral_offset], dtype=np.float32)

        # GPS (x,y) Fahrzeug
        veh_loc = self.vehicle.get_transform().location
        gps_own = np.array([veh_loc.x, veh_loc.y], dtype=np.float32)

        # GPS (x,y) nächster Waypoint
        if self.next_waypoint_transform is None:
            wp_xy = np.array([0, 0], dtype=np.float32)
        else:
            wp_loc = self.next_waypoint_transform.location
            wp_xy = np.array([wp_loc.x, wp_loc.y], dtype=np.float32)

        # Speed
        speed = np.array([self.get_vehicle_speed()], dtype=np.float32)

        return {
            "segmentation": seg_img,
            "dist_center": dist_center,
            "gps_next_waypoint": wp_xy,
            "gps_own": gps_own,
            "speed": speed
        }

    def _show_image(self, seg_img):
        """
        Zeigt das Segmentationsbild per OpenCV-Fenster.
        seg_img: shape (480,640,1) in [0,1].
        """
        gray = (seg_img[..., 0] * 255).astype(np.uint8)  # (480,640)
        cv2.imshow("CARLA Semantic Segmentation (BirdView)", gray)
        cv2.waitKey(1)

    def render(self, mode="human"):
        """Nichts extra, da wir _show_image() schon in _get_obs() aufrufen."""
        pass

    def close(self):
        self._clear_actors()
        self.world.apply_settings(self.original_settings)
        cv2.destroyAllWindows()