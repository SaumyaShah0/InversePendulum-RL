"""
evaluate_ppo.py

Load the trained PPO model and watch it balance the inverted pendulum
in real time with rendering.
"""

import os
import time

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    # -----------------------------
    # 1. Load environment with rendering
    # -----------------------------
    env = gym.make("CartPole-v1", render_mode="human")

    # -----------------------------
    # 2. Load trained PPO model
    # -----------------------------
    model_path = os.path.join("models", "pendulum_ppo")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.zip\n"
            "Run train_ppo.py first."
        )

    model = PPO.load(model_path)

    # -----------------------------
    # 3. Run some episodes
    # -----------------------------
    num_episodes = 5

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Small delay so we can see motion smoothly
            time.sleep(0.02)

            # Predict an action from the current observation
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward

        print(f"Episode {ep + 1}: total reward = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()