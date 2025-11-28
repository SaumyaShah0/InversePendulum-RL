# 🤖 Inverse Pendulum Control using Reinforcement Learning & Classical Control

This project explores balancing an inverted pendulum (CartPole) using three control strategies:

- **PPO Reinforcement Learning (Stable-Baselines3)**
- **PID Controller**
- **LQR-like State Feedback Controller**

It compares results from AI-based control versus traditional control theory.

---

## 🎯 Project Goal

- Keep the pole upright (\(\theta \approx 0\))
- Maintain the cart near the track center
- Optimize stability and handle disturbances

Applications in real-world systems include:
- Humanoid robot balance
- Segway stabilization
- Robotic manipulators

---

## 📌 Problem Definition

### System: *Inverted Pendulum on a Cart*

State variables:
- `x` (cart position)
- `x_dot` (cart velocity)
- `theta` (pendulum angle in radians)
- `theta_dot` (pendulum angular velocity)

### Control Input:
Discrete force applied on the cart:
- `0` → Push Left
- `1` → Push Right

---

## 🚀 Solution Approaches

| Controller | Type             | Advantages               | Limitations                |
|------------|------------------|--------------------------|----------------------------|
| PID        | Feedback Control | Simple, quick setup      | Hard to tune, fails under large disturbances |
| LQR        | Optimal Control  | Good stability           | Requires full system model |
| PPO        | Deep RL          | Learns/adapts, robust    | Needs training time, less predictable        |

---

## 🧠 Reinforcement Learning (PPO)

PPO (Proximal Policy Optimization) trains a neural network to map observations to actions to maximize the time the pole stays balanced.

Training loop:
- Observe system state
- Decide push direction
- Get reward for balance
- Improve behavior iteratively

Saved model: `models/pendulum_ppo.zip`

---

## 🔧 Classical Controllers

### PID
Uses error: `target_angle (0) – theta`  
Control law:

$$u = K_p \cdot \text{error} + K_i \int{\text{error}\,dt} + K_d \frac{d(\text{error})}{dt}$$

Simple controller; works for small deviations, struggles with large disturbances.

### LQR-like Controller
State-feedback law:

$$u = -Kx$$

Where `K` is tuned for the linearized system. Offers stability but needs correct modeling.

---

## 🧪 How to Run

### 1. Setup (Windows)
```
.\setup.bat
```
Creates `venv/` and installs modules.

Activate:
```
.\venv\Scripts\activate
```

### 2. Train RL Model
```
python src/train_ppo.py
```

### 3. Watch Trained Agent
```
python src/evaluate_ppo.py
```

### 4. Use Classical Controllers
```
python src/classical_pid_control.py
```
```
python src/classical_lqr_control.py
```

---

## 📂 Project Structure

```Text
InversePendulum-RL/
├── src/
│ ├── train_ppo.py           # Train PPO RL agent
│ ├── evaluate_ppo.py        # Test trained model
│ ├── classical_pid.py       # PID controller
│ └── classical_lqr.py       # LQR state feedback
├── models/                  # Saved RL models (*.zip)
├── results/                 # Plots, logs, videos
├── requirements.txt         # Dependencies
├── setup.bat                # Windows setup
└── README.md
```
---

## 📊 Results

Control Method | Avg Balance Time | Notes
---------------|------------------|-----------------------------------
PID            | ⭐⭐☆☆☆        | Works only for small angles
LQR            | ⭐⭐⭐⭐☆      | Stable & quick response
PPO            | ⭐⭐⭐⭐⭐     | Best after training; most robust

RL achieves the most robust balancing under disturbances.

---

## 📈 Future Improvements

- Add noisy sensors (realistic conditions)
- Use continuous action space (SAC / TD3)
- Add result graphs to README
- Export controller to Arduino robot

---

## 🧑‍💻 Technologies

- Python 3.10
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy

---

## 🏆 Credits

Project by: Saumya Shah & Astha  
For learning and research in control systems + AI.

---

⭐ If you like this project, star the repo!

---
