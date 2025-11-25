"""
classical_pid_control.py

Use a simple PID controller to keep the inverted pendulum upright
in the CartPole-v1 environment.

State = [x, x_dot, theta, theta_dot]
We control the horizontal force by choosing discrete actions:
- action 0: push cart to the LEFT
- action 1: push cart to the RIGHT

We design PID on the angle theta (target = 0 rad).
"""

import time

import gymnasium as gym
import numpy as np


def pid_controller(theta, theta_dot, integral_error, dt,
                   kp=50.0, ki=1.0, kd=5.0):
    """
    Very simple PID on the pendulum angle.

    theta: current angle (rad)
    theta_dot: angular velocity (rad/s)
    integral_error: accumulated error over time
    dt: time step (s)
    kp, ki, kd: PID gains
    """
    # Error: we want theta -> 0 (upright)
    error = 0.0 - theta

    # Integrate error
    integral_error += error * dt

    # Derivative (using theta_dot directly as approx)
    derivative_error = -theta_dot  # because d(error)/dt = -theta_dot

    # PID control output (continuous "force" style signal)
    u = kp * error + ki * integral_error + kd * derivative_error
    return u, integral_error


def run_pid_episode(env, render=True, max_steps=500):
    """
    Run a single episode with PID control.
    Returns total reward and number of steps survived.
    """
    # Reset env: Gymnasium returns (obs, info)
    obs, info = env.reset()
    total_reward = 0.0
    dt = 0.02  # CartPole internal time step
    integral_error = 0.0

    for step in range(max_steps):
        if render:
            time.sleep(dt)

        # Unpack observation
        x, x_dot, theta, theta_dot = obs

        # Compute PID control signal
        u, integral_error = pid_controller(theta, theta_dot, integral_error, dt)

        # Map continuous u -> discrete action
        # If control wants to push right (u>0) -> action 1, else 0
        action = 1 if u > 0 else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward

        if done:
            break

    return total_reward, step + 1


def main():
    # Create environment with rendering
    env = gym.make("CartPole-v1", render_mode="human")

    num_episodes = 5
    rewards = []

    for ep in range(num_episodes):
        ep_reward, steps = run_pid_episode(env, render=True)
        rewards.append(ep_reward)
        print(f"[PID] Episode {ep + 1}: reward={ep_reward:.1f}, steps={steps}")

    print(f"[PID] Average reward over {num_episodes} episodes = {np.mean(rewards):.1f}")

    env.close()


if __name__ == "__main__":
    main()