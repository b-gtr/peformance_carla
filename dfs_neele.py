from abc import ABC, abstractmethod
import math
import threading
import carla
import cv2
import numpy as np
import torch
import pygame
import random
import time
import hashlib  # Hinzugefügt für Bild-Hashing
import os       # Hinzugefügt für Dateisystemoperationen
from params_pool2 import ParamsPool
from replay_buffer import ReplayBuffer, Transition

# ===========================
# =        CONFIGURATIONS    =
# ===========================

# Trainingsparameter
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
BATCH_SIZE = 64
LOAD_MODEL = False  # Auf True setzen, um ein gespeichertes Modell zu laden

# Debugging-Parameter
DEBUG = True  # Auf True setzen, um detaillierte Logs zu aktivieren

# Hashing-Parameter
HASH_CHECK_INTERVAL = 70  # Intervall für Bildüberprüfung
HASH_HISTORY_LIMIT = HASH_CHECK_INTERVAL  # Anzahl der gespeicherten Hashes

# Bildspeicher-Parameter
SAVE_IMAGES = True  # Auf True setzen, um Bilder zu speichern
RGB_IMAGE_DIR = 'saved_images/rgb'
LABEL_IMAGE_DIR = 'saved_images/labels'

# Initialisiere Gerät
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===========================
# =        SENSOR KLASSEN    =
# ===========================

class Sensor(ABC):
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.sensor = None
        self.history = []

    @abstractmethod
    def listen(self):
        pass

    def clear_history(self):
        self.history.clear()

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except RuntimeError as e:
                print(f"Error destroying sensor: {e}")

    def get_history(self):
        return self.history

class CollisionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        
    def _on_collision(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_collision)

class LaneInvasionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)

    def _on_lane_invasion(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_lane_invasion)        

class GnssSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        self.sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        self.current_gnss = None

    def _on_gnss_event(self, event):
        self.current_gnss = event

    def listen(self):
        self.sensor.listen(self._on_gnss_event)
    
    def get_current_gnss(self):
        return self.current_gnss

class CameraSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world, image_processor_callback):
        super().__init__(vehicle)
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.image_processor_callback = image_processor_callback

    def listen(self):
        self.sensor.listen(self.image_processor_callback)

# ===========================
# =        UMGEBUNG         =
# ===========================

class CarlaEnv:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.client.load_world('Town01')
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None

        self.image_lock = threading.Lock()
        self.running = True

        # Initialize Pygame und Display einrichten
        pygame.init()
        self.display = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("CARLA Semantic Segmentation")

        # Synchronous Modus
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        self.latest_image = None
        self.agent_image = None
        self.display_surface = None

        # Bild-Update Tracking
        self.image_timestamp = 0.0
        self.image_counter = 0  # Zähler für Bild-Updates

        # Hashing-Tracking
        self.image_hashes = []  # Liste zur Speicherung von Bild-Hashes

        # Bildspeicher initialisieren
        if SAVE_IMAGES:
            self.setup_image_saving()

        self.reset_environment()

    def setup_image_saving(self):
        """
        Erstellt die Verzeichnisse für das Speichern von Bildern, falls sie nicht existieren.
        """
        os.makedirs(RGB_IMAGE_DIR, exist_ok=True)
        os.makedirs(LABEL_IMAGE_DIR, exist_ok=True)
        if DEBUG:
            print(f"[DEBUG] Bildspeicher-Verzeichnisse erstellt: '{RGB_IMAGE_DIR}' und '{LABEL_IMAGE_DIR}'")

    def reset_environment(self):
        self._clear_sensors()

        # Fahrzeug zerstören, wenn es existiert
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_sensor = None

        # Fahrzeug an zufälligem Spawn-Punkt spawnen
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        self.spawn_point = random.choice(self.spawn_points)  # Zufälligen Spawn-Punkt wählen
        self.spawn_rotation = self.spawn_point.rotation
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        # Sensoren anhängen
        self.setup_sensors()

        # Warten, bis Sensoren initialisiert sind
        for _ in range(10):
            self.world.tick()

    def setup_sensors(self):
        # Kamera-Sensor
        self.camera_sensor = CameraSensor(self.vehicle, self.blueprint_library, self.world, self.process_image)
        self.camera_sensor.listen()

        # Kollisionssensor
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)
        self.collision_sensor.listen()

        # Spur-Invasionssensor
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.blueprint_library, self.world)
        self.lane_invasion_sensor.listen()

        # GNSS-Sensor
        self.gnss_sensor = GnssSensor(self.vehicle, self.blueprint_library, self.world)
        self.gnss_sensor.listen()

    def _clear_sensors(self):
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.gnss_sensor is not None:
            self.gnss_sensor.destroy()
        self.latest_image = None
        self.agent_image = None
        self.display_surface = None

    def process_image(self, image):
        try:
            # Bild für die Anzeige konvertieren
            image.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            rgb_array = array[:, :, :3]  # RGB-Kanäle extrahieren

            with self.image_lock:
                self.latest_image = rgb_array.copy()
                self.image_counter += 1  # Bildzähler erhöhen
                self.image_timestamp = time.time()  # Zeitstempel aktualisieren

                # Labels für den Agenten erhalten
                image.convert(carla.ColorConverter.Raw)
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                labels = array[:, :, 2]  # Labels aus dem roten Kanal extrahieren

                # Labels normalisieren
                self.agent_image = labels / 22.0  # Normalisieren auf [0, 1]

                # Pygame-Oberfläche für die Anzeige erstellen
                self.display_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))

                # Bild-Hash berechnen und speichern
                image_hash = self.hash_image(self.agent_image)
                self.image_hashes.append(image_hash)
                if len(self.image_hashes) > HASH_HISTORY_LIMIT:
                    self.image_hashes.pop(0)  # Ältesten Hash entfernen

                # Bilddaten speichern
                if SAVE_IMAGES:
                    self.save_images(rgb_array, labels)

            if DEBUG:
                print(f"[DEBUG] Bild verarbeitet um {self.image_timestamp}, Gesamtbilder: {self.image_counter}, Hash: {image_hash}")

            # Alle 70 Bilder die Hashes überprüfen
            if self.image_counter % HASH_CHECK_INTERVAL == 0:
                self.check_image_hashes()

        except Exception as e:
            print(f"Fehler bei der Bildverarbeitung: {e}")

    def hash_image(self, image_array):
        """
        Erstellt einen SHA256-Hash des gegebenen Bildarrays.
        """
        # Stellen Sie sicher, dass das Array in einem konsistenten Format ist
        image_bytes = image_array.tobytes()
        return hashlib.sha256(image_bytes).hexdigest()

    def check_image_hashes(self):
        """
        Überprüft, ob alle gespeicherten Bild-Hashes gleich sind.
        """
        if len(self.image_hashes) != HASH_CHECK_INTERVAL:
            # Noch nicht genug Bilder gesammelt
            return

        first_hash = self.image_hashes[0]
        all_same = all(h == first_hash for h in self.image_hashes)

        if all_same:
            print(f"[WARNING] Alle {HASH_CHECK_INTERVAL} Bilder sind identisch! Mögliche Bildwiederverwendung erkannt.")
        else:
            if DEBUG:
                print(f"[DEBUG] Bild-Hash-Überprüfung erfolgreich: {HASH_CHECK_INTERVAL} unterschiedliche Bilder vorhanden.")

    def save_images(self, rgb_array, labels):
        """
        Speichert die RGB- und Label-Bilder in externen Dateien.
        """
        try:
            # Formatieren des Dateinamens basierend auf dem Bildzähler
            rgb_filename = os.path.join(RGB_IMAGE_DIR, f"rgb_{self.image_counter:06d}.png")
            label_filename = os.path.join(LABEL_IMAGE_DIR, f"labels_{self.image_counter:06d}.png")

            # Speichern des RGB-Bildes
            cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))

            # Labels wieder in den ursprünglichen Bereich bringen und speichern
            labels_uint8 = (labels).astype(np.uint8)  # Annahme: Labels sind bereits im Bereich [0, 22]
            cv2.imwrite(label_filename, labels_uint8)

            if DEBUG and self.image_counter % 100 == 0:
                print(f"[DEBUG] Bilder gespeichert: {rgb_filename}, {label_filename}")
        except Exception as e:
            print(f"Fehler beim Speichern der Bilder: {e}")

    def get_vehicle_speed(self):
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def process_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def render_display(self):
        with self.image_lock:
            if self.display_surface is not None:
                self.display.blit(self.display_surface, (0, 0))
                pygame.display.flip()

    def destroy(self):
        # Ursprüngliche Einstellungen wiederherstellen
        self.world.apply_settings(self.original_settings)

        # Akteure bereinigen
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.gnss_sensor is not None:
            self.gnss_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
        pygame.quit()  # Pygame beenden

# ===========================
# =        TRAINING LOOP     =
# ===========================

def train_agent(env, agent, replay_buffer, num_episodes=1000, max_steps_per_episode=1000, batch_size=64):
    try:
        previous_image_counter = env.image_counter  # Initialisiere vorherigen Bildzähler

        for episode in range(num_episodes):
            if not env.running:
                break
            env.reset_environment()

            direction_vector = env.spawn_rotation.get_forward_vector()

            destination = env.spawn_point.location + direction_vector * 40

            previous_distance = None
            episode_reward = 0
            termination_reason = None

            for step in range(max_steps_per_episode):
                if not env.running:
                    break

                with env.image_lock:
                    current_agent_image = env.agent_image.copy() if env.agent_image is not None else None
                    current_image_timestamp = env.image_timestamp

                current_gnss = env.gnss_sensor.get_current_gnss()
                if current_agent_image is None or current_gnss is None:
                    env.world.tick()
                    continue

                # Log Bild-Update
                if DEBUG:
                    print(f"[DEBUG] Episode {episode+1}, Schritt {step}: Bildzähler = {env.image_counter}")

                # Überprüfen, ob ein neues Bild empfangen wurde
                assert env.image_counter > previous_image_counter, "Bild wurde wiederverwendet! Zähler hat sich nicht erhöht."
                previous_image_counter = env.image_counter  # Für den nächsten Schritt aktualisieren

                transform = env.vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation
                yaw = math.radians(rotation.yaw)

                map = env.world.get_map()
                waypoint = map.get_waypoint(location)
                next_waypoints = waypoint.next(2.0)
                if next_waypoints:
                    next_waypoint = next_waypoints[0]
                else:
                    next_waypoint = waypoint

                wp_location = next_waypoint.transform.location
                dx = wp_location.x - location.x
                dy = wp_location.y - location.y
                desired_yaw = math.atan2(dy, dx)
                epsilon = desired_yaw - yaw
                epsilon = (epsilon + math.pi) % (2 * math.pi) - math.pi

                lateral_deviation = -math.sin(yaw) * dx + math.cos(yaw) * dy

                speed = env.get_vehicle_speed()
                distance_to_destination = location.distance(destination)

                if previous_distance is None:
                    previous_distance = distance_to_destination

                scalars = np.array([distance_to_destination, lateral_deviation])

                action = agent.act(current_agent_image, scalars)

                steer = float(action[0])
                throttle = float(action[1])
                control = carla.VehicleControl(
                    steer=np.clip(steer, -1.0, 1.0),
                    throttle=np.clip(throttle, 0.0, 1.0)
                )
                env.vehicle.apply_control(control)

                env.world.tick()

                with env.image_lock:
                    next_agent_image = env.agent_image.copy() if env.agent_image is not None else None
                    next_image_timestamp = env.image_timestamp

                transform = env.vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation
                yaw = math.radians(rotation.yaw)

                speed = env.get_vehicle_speed()

                deviation_threshold = 0.7
                deviation_penalty_scale = 4.0

                if len(env.collision_sensor.get_history()) > 0:
                    reward = -30
                    done = True
                    termination_reason = 'collision'
                elif step >= max_steps_per_episode - 1:
                    reward = -1
                    done = True
                    termination_reason = 'timeout'
                else:
                    if abs(lateral_deviation) <= deviation_threshold:
                        r_lane_centering = 1.0 / (abs(lateral_deviation) + 0.1)
                    else:
                        r_lane_centering = -deviation_penalty_scale * (abs(lateral_deviation) - deviation_threshold)

                    v = speed * 3.6
                    v_target = 20
                    r_speed = 1 - min(1, abs(v - v_target) / 5)

                    r_heading = - (abs(epsilon) / 3) ** 2

                    if distance_to_destination < previous_distance:
                        r_traveled = 1
                    else:
                        r_traveled = -0.1

                    r_overspeed = -5 if v > 25 else 0

                    reward = r_lane_centering + r_speed + r_heading + r_traveled + r_overspeed
                    done = False

                episode_reward += reward

                if DEBUG:
                    print(f"[DEBUG] Episode {episode+1}, Schritt {step}, Belohnung: {reward:.2f}, Gesamte Belohnung: {episode_reward:.2f}")
                    print(f"[DEBUG] r_travel: {r_traveled}")

                transition = Transition(
                    img=current_agent_image,
                    scalars=scalars,
                    a=action,
                    r=reward,
                    n_img=next_agent_image,
                    n_scalars=scalars,
                    d=done
                )
                replay_buffer.push(transition)

                if step % 20 == 0 and replay_buffer.ready_for(batch_size):
                    batch = replay_buffer.sample(batch_size)
                    agent.update_networks(batch)

                env.process_pygame_events()
                env.render_display()

                if done:
                    if DEBUG:
                        print(f"[DEBUG] Episode {episode+1} beendet aufgrund von {termination_reason}.")
                    break

                previous_distance = distance_to_destination

            print(f'Episode {episode+1}, Gesamte Belohnung: {episode_reward:.2f}')
            if (episode + 1) % 50 == 0:
                agent.save_model('model_params.pth')
                print(f'Modellparameter nach Episode {episode+1} gespeichert.')

        print('Training abgeschlossen.')

    finally:
        env.destroy()

# ===========================
# =          MAIN            =
# ===========================

def main():
    # Konfigurationsparameter ausgeben
    print("===== Konfigurationsparameter =====")
    print(f"Anzahl der Episoden: {NUM_EPISODES}")
    print(f"Maximale Schritte pro Episode: {MAX_STEPS_PER_EPISODE}")
    print(f"Batch-Größe: {BATCH_SIZE}")
    print(f"Modell laden: {LOAD_MODEL}")
    print(f"Gerät: {device}")
    print(f"Debug-Modus: {DEBUG}")
    print(f"Hash-Überprüfungsintervall: {HASH_CHECK_INTERVAL}")
    print(f"Bildspeicherung aktiviert: {SAVE_IMAGES}")
    print("===================================\n")

    env = CarlaEnv()
    input_height = 480
    input_width = 640
    feature_dim = 64
    scalar_dim = 2
    action_dim = 2

    agent = ParamsPool(feature_dim=feature_dim,
                       scalar_dim=scalar_dim,
                       action_dim=action_dim,
                       input_height=input_height,
                       input_width=input_width,
                       device=device)
    replay_buffer = ReplayBuffer(capacity=25000, device=device)

    if LOAD_MODEL:
        try:
            agent.load_model('model_params.pth')
            print('Modellparameter geladen.')
        except Exception as e:
            print(f"Laden der Modellparameter fehlgeschlagen: {e}")

    train_agent(env, agent, replay_buffer,
                num_episodes=NUM_EPISODES,
                max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()
