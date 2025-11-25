"""
train_ppo.py

Train a PPO (Proximal Policy Optimization) agent to balance
the inverted pendulum in the classic CartPole-v1 environment.

We use:
- gymnasium for the environment
- stable-baselines3 for the RL algorithm
"""

import os

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    # -----------------------------
    # 1. Create training environment
    # -----------------------------
    # CartPole-v1 is a classic benchmark for inverted pendulum.
    # render_mode=None because we don't need to see the training visually.
    env = gym.make("CartPole-v1")

    # -----------------------------
    # 2. Create PPO model
    # -----------------------------
    # "MlpPolicy" = simple neural network for state-based observations
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,          # show training info in console
        n_steps=2048,       # steps per update
        batch_size=64,      # minibatch size
        learning_rate=3e-4, # standard PPO LR
        gamma=0.99,         # discount factor
    )

    # -----------------------------
    # 3. Learn / Train
    # -----------------------------
    # total_timesteps: you can increase this for better performance.
    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps)

    # -----------------------------
    # 4. Save trained model
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "pendulum_ppo")
    model.save(model_path)
    print(f"✅ Training finished. Model saved to: {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()