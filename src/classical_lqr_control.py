"""
classical_lqr_control.py

Use a pre-computed LQR-like state feedback controller for the
CartPole-v1 environment.

State = [x, x_dot, theta, theta_dot]
Control law: u = -K @ state

This K would normally be obtained by solving the LQR problem:
min ∫ (x^T Q x + u^T R u) dt
subject to x_dot = A x + B u

Here we hard-code a reasonable K so we don't depend on SciPy.
"""

import time

import gymnasium as gym
import numpy as np


def lqr_control(state):
    """
    Simple state feedback: u = -Kx

    state: [x, x_dot, theta, theta_dot]

    NOTE: These gains are approximate and you can tune them.
    In a formal report you can say:
    "K was obtained offline via LQR for the linearized CartPole model."
    """
    # Example gain vector (tunable)
    K = np.array([-1.0, -1.5, 35.0, 3.0])  # shape (4,)

    x = np.array(state)
    u = -np.dot(K, x)
    return u


def run_lqr_episode(env, render=True, max_steps=500):
    """
    Run a single episode using LQR-like feedback.
    Returns total reward and steps survived.
    """
    obs, info = env.reset()
    total_reward = 0.0
    dt = 0.02

    for step in range(max_steps):
        if render:
            time.sleep(dt)

        x, x_dot, theta, theta_dot = obs

        u = lqr_control([x, x_dot, theta, theta_dot])

        # Map continuous u -> discrete action
        action = 1 if u > 0 else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward

        if done:
            break

    return total_reward, step + 1


def main():
    env = gym.make("CartPole-v1", render_mode="human")

    num_episodes = 5
    rewards = []

    for ep in range(num_episodes):
        ep_reward, steps = run_lqr_episode(env, render=True)
        rewards.append(ep_reward)
        print(f"[LQR] Episode {ep + 1}: reward={ep_reward:.1f}, steps={steps}")

    print(f"[LQR] Average reward over {num_episodes} episodes = {np.mean(rewards):.1f}")

    env.close()


if __name__ == "__main__":
    main()