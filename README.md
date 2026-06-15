# Q-Learning vs PID for DC Motor Speed Control

MCTA4362 Machine Learning Mini Project

Semester 2, 2025/2026

This project develops and compares a classical PID controller and a
Q-Learning reinforcement-learning controller for DC motor speed regulation.
Both controllers are evaluated under the same operating conditions, including
a temporary load disturbance.

## Final Deliverables

- [Final project report](REPORT%20MINI%20PROJECT%20ML.pdf)
- [Presentation slides](AI_versus_PID_Motor_Control.pdf)
- [Jupyter notebook](notebooks/dc_motor_q_learning_vs_pid.ipynb)
- [Google Colab notebook](https://colab.research.google.com/drive/1ykMG-HuH2YoBbmu5XkA7YxPAAqGCVeEM?usp=sharing)

## Project Objectives

- Model and simulate a DC motor speed-control system.
- Design a PID controller for speed regulation.
- Implement an intelligent controller using Q-Learning.
- Compare both controllers using standard control-performance metrics.
- Evaluate robustness under a load disturbance.

## Methodology

The simulated DC motor uses a target speed of `50 rad/s`, a maximum control
voltage of `100 V`, and a sampling time of `0.005 s`. A load disturbance of
`-15 rad/s` is applied between `3.5 s` and `4.0 s`.

The PID controller uses:

- `Kp = 5.0`
- `Ki = 8.0`
- `Kd = 0.05`

The Q-Learning controller uses:

- 61 discretized error states
- 21 voltage actions from `0 V` to `100 V`
- Learning rate `alpha = 0.4`
- Discount factor `gamma = 0.95`
- 400 training episodes

## Final Results

| Metric | PID | Q-Learning |
| --- | ---: | ---: |
| Rise time | 0.275 s | 0.055 s |
| Overshoot | 0% | 1.84% |
| Settling time | 1.47 s | 0.075 s |
| Mean squared error | 0.2247 rad/s² | 0.0661 rad/s² |
| Disturbance recovery time | 0.15 s | 0.01 s |

The Q-Learning controller responds and recovers faster, with lower tracking
error. The PID controller produces no overshoot and has a smoother,
more interpretable control action.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- REPORT MINI PROJECT ML.pdf
|-- AI_versus_PID_Motor_Control.pdf
`-- notebooks/
    `-- dc_motor_q_learning_vs_pid.ipynb
```

## Running the Notebook

The recommended method is to open the linked Google Colab notebook and select
**Runtime > Run all**.

For a local Jupyter environment:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/dc_motor_q_learning_vs_pid.ipynb
```

The notebook trains the Q-Learning controller, runs both controller
simulations, prints the performance metrics, and generates comparison plots.
Q-Learning uses random exploration, so results may vary slightly between runs.

## Group Members

| Name | Matric Number | Programme |
| --- | --- | --- |
| Amir Farhan Bin Ghaffar | 2115617 | BMCT |
| Muhammad Amin Bin Mohamad Rizal | 2217535 | BMCT |
| Muhammad Irsyad Ilham Bin Azizan | 2217555 | BMCT |
| Muhammad Haikal Hanif Bin Abdul Razak | 2213297 | BMCT |
