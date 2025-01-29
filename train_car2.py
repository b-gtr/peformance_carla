# train_car.py
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from carla_gym_env4 import CarlaGymEnv

# For Recurrent PPO and the LSTM-based policy
from sb3_contrib import RecurrentPPO
# This policy can handle Dict observation spaces with an LSTM
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy


def main():
    env = CarlaGymEnv(display=True)
    # In Stable Baselines 3 müssen wir das Env vectorisieren:
    env = DummyVecEnv([lambda: env])

    # PPO-Instanz
    model = PPO("MultiInputPolicy", env, verbose=1, device="cuda", n_steps=2048, gamma=0.999, learning_rate=1e-3)  # oder "cpu"

    model.learn(total_timesteps=1_000_000)

    obs = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()