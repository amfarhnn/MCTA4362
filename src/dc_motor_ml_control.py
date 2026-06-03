from __future__ import annotations

import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class MotorParameters:
    inertia: float = 0.01
    damping: float = 0.1
    motor_constant: float = 0.01
    resistance: float = 1.0
    inductance: float = 0.5


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.01
    total_time: float = 10.0
    voltage_limit: float = 24.0
    reference_speed: float = 1.0
    disturbance_start: float = 5.0
    disturbance_end: float = 6.2
    disturbance_torque: float = 0.002


@dataclass(frozen=True)
class PIDGains:
    kp: float = 18.0
    ki: float = 35.0
    kd: float = 0.8


@dataclass
class NeuralControllerModel:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


class PIDController:
    def __init__(self, gains: PIDGains, voltage_limit: float, integral_limit: float = 2.0):
        self.gains = gains
        self.voltage_limit = voltage_limit
        self.integral_limit = integral_limit
        self.integral_error = 0.0
        self.previous_error = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0
        self.previous_error = 0.0

    def step(self, reference: float, speed: float, dt: float) -> tuple[float, np.ndarray]:
        error = reference - speed
        self.integral_error += error * dt
        self.integral_error = float(
            np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        )
        derivative_error = (error - self.previous_error) / dt
        self.previous_error = error

        voltage = (
            self.gains.kp * error
            + self.gains.ki * self.integral_error
            + self.gains.kd * derivative_error
        )
        voltage = float(np.clip(voltage, -self.voltage_limit, self.voltage_limit))

        features = np.array(
            [reference, speed, error, self.integral_error, derivative_error],
            dtype=float,
        )
        return voltage, features


class NeuralNetworkController:
    def __init__(
        self,
        model: NeuralControllerModel,
        voltage_limit: float,
        integral_limit: float = 2.0,
    ):
        self.model = model
        self.voltage_limit = voltage_limit
        self.integral_limit = integral_limit
        self.integral_error = 0.0
        self.previous_error = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0
        self.previous_error = 0.0

    def step(self, reference: float, speed: float, dt: float) -> tuple[float, np.ndarray]:
        error = reference - speed
        self.integral_error += error * dt
        self.integral_error = float(
            np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        )
        derivative_error = (error - self.previous_error) / dt
        self.previous_error = error

        features = np.array(
            [reference, speed, error, self.integral_error, derivative_error],
            dtype=float,
        )
        voltage = neural_predict(self.model, features[None, :])[0]
        voltage = float(np.clip(voltage, -self.voltage_limit, self.voltage_limit))
        return voltage, features


def dc_motor_derivative(
    state: np.ndarray,
    voltage: float,
    load_torque: float,
    params: MotorParameters,
) -> np.ndarray:
    speed, current = state
    d_speed = (
        params.motor_constant * current
        - params.damping * speed
        - load_torque
    ) / params.inertia
    d_current = (
        voltage
        - params.resistance * current
        - params.motor_constant * speed
    ) / params.inductance
    return np.array([d_speed, d_current], dtype=float)


def rk4_step(
    state: np.ndarray,
    voltage: float,
    load_torque: float,
    dt: float,
    params: MotorParameters,
) -> np.ndarray:
    k1 = dc_motor_derivative(state, voltage, load_torque, params)
    k2 = dc_motor_derivative(state + 0.5 * dt * k1, voltage, load_torque, params)
    k3 = dc_motor_derivative(state + 0.5 * dt * k2, voltage, load_torque, params)
    k4 = dc_motor_derivative(state + dt * k3, voltage, load_torque, params)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def default_reference(_: float, cfg: SimulationConfig) -> float:
    return cfg.reference_speed


def default_disturbance(t: float, cfg: SimulationConfig) -> float:
    if cfg.disturbance_start <= t <= cfg.disturbance_end:
        return cfg.disturbance_torque
    return 0.0


def simulate_controller(
    controller,
    params: MotorParameters,
    cfg: SimulationConfig,
    reference_function=default_reference,
    disturbance_function=default_disturbance,
) -> dict[str, np.ndarray]:
    controller.reset()
    steps = int(cfg.total_time / cfg.dt) + 1
    time = np.linspace(0.0, cfg.total_time, steps)
    state = np.array([0.0, 0.0], dtype=float)

    speed = np.zeros(steps)
    current = np.zeros(steps)
    voltage = np.zeros(steps)
    reference = np.zeros(steps)
    disturbance = np.zeros(steps)

    for index, t in enumerate(time):
        reference[index] = reference_function(float(t), cfg)
        disturbance[index] = disturbance_function(float(t), cfg)
        speed[index] = state[0]
        current[index] = state[1]

        command, _ = controller.step(reference[index], speed[index], cfg.dt)
        voltage[index] = command

        if index < steps - 1:
            state = rk4_step(state, command, disturbance[index], cfg.dt, params)

    return {
        "time": time,
        "reference": reference,
        "speed": speed,
        "current": current,
        "voltage": voltage,
        "disturbance": disturbance,
    }


def generate_reference_sequence(
    rng: np.random.Generator,
    total_time: float,
) -> list[tuple[float, float]]:
    change_times = [0.0, 1.5, 3.0, 4.5]
    references = rng.uniform(0.35, 1.25, size=len(change_times))
    return list(zip(change_times, references))


def sequence_value(sequence: list[tuple[float, float]], t: float) -> float:
    value = sequence[0][1]
    for start_time, candidate in sequence:
        if t >= start_time:
            value = candidate
        else:
            break
    return float(value)


def collect_training_data(
    params: MotorParameters,
    cfg: SimulationConfig,
    samples_per_episode: int = 600,
    episodes: int = 34,
    seed: int = 4362,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[float] = []
    training_cfg = SimulationConfig(
        dt=cfg.dt,
        total_time=samples_per_episode * cfg.dt,
        voltage_limit=cfg.voltage_limit,
        reference_speed=cfg.reference_speed,
        disturbance_start=cfg.disturbance_start,
        disturbance_end=cfg.disturbance_end,
        disturbance_torque=cfg.disturbance_torque,
    )

    for _ in range(episodes):
        controller = PIDController(PIDGains(), cfg.voltage_limit)
        state = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)])
        references = generate_reference_sequence(rng, training_cfg.total_time)
        load_start = rng.uniform(2.0, 4.8)
        load_end = load_start + rng.uniform(0.6, 1.4)
        load_torque = rng.uniform(0.0, 0.003)

        for index in range(samples_per_episode):
            t = index * training_cfg.dt
            reference = sequence_value(references, t)
            load = load_torque if load_start <= t <= load_end else 0.0
            command, row = controller.step(reference, float(state[0]), training_cfg.dt)

            features.append(row)
            labels.append(command)
            state = rk4_step(state, command, load, training_cfg.dt, params)

    return np.vstack(features), np.array(labels, dtype=float)


def train_neural_controller(
    features: np.ndarray,
    labels: np.ndarray,
    hidden_units: int = 24,
    epochs: int = 180,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    seed: int = 4362,
) -> tuple[NeuralControllerModel, list[float]]:
    rng = np.random.default_rng(seed)

    x_mean = features.mean(axis=0)
    x_std = features.std(axis=0)
    x_std[x_std < 1e-9] = 1.0

    y_mean = float(labels.mean())
    y_std = float(labels.std())
    if y_std < 1e-9:
        y_std = 1.0

    x = (features - x_mean) / x_std
    y = ((labels - y_mean) / y_std)[:, None]

    input_units = x.shape[1]
    w1 = rng.normal(0.0, 0.25, size=(input_units, hidden_units))
    b1 = np.zeros((1, hidden_units))
    w2 = rng.normal(0.0, 0.25, size=(hidden_units, 1))
    b2 = np.zeros((1, 1))

    params = [w1, b1, w2, b2]
    moments = [np.zeros_like(param) for param in params]
    velocities = [np.zeros_like(param) for param in params]
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    step = 0
    losses: list[float] = []

    for _ in range(epochs):
        order = rng.permutation(x.shape[0])
        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            xb = x[indices]
            yb = y[indices]

            z1 = xb @ w1 + b1
            h1 = np.tanh(z1)
            prediction = h1 @ w2 + b2
            error = prediction - yb
            loss = float(np.mean(error**2))
            epoch_loss += loss
            batch_count += 1

            dpred = 2.0 * error / xb.shape[0]
            dw2 = h1.T @ dpred
            db2 = dpred.sum(axis=0, keepdims=True)
            dh1 = dpred @ w2.T
            dz1 = dh1 * (1.0 - h1**2)
            dw1 = xb.T @ dz1
            db1 = dz1.sum(axis=0, keepdims=True)
            grads = [dw1, db1, dw2, db2]

            step += 1
            for i, grad in enumerate(grads):
                moments[i] = beta1 * moments[i] + (1.0 - beta1) * grad
                velocities[i] = beta2 * velocities[i] + (1.0 - beta2) * (grad**2)
                m_hat = moments[i] / (1.0 - beta1**step)
                v_hat = velocities[i] / (1.0 - beta2**step)
                params[i][...] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        losses.append(epoch_loss / max(batch_count, 1))

    model = NeuralControllerModel(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )
    return model, losses


def neural_predict(model: NeuralControllerModel, features: np.ndarray) -> np.ndarray:
    x = (features - model.x_mean) / model.x_std
    hidden = np.tanh(x @ model.w1 + model.b1)
    y = hidden @ model.w2 + model.b2
    return (y[:, 0] * model.y_std) + model.y_mean


def calculate_metrics(result: dict[str, np.ndarray], cfg: SimulationConfig) -> dict[str, float]:
    time = result["time"]
    reference = result["reference"]
    speed = result["speed"]
    voltage = result["voltage"]
    error = reference - speed
    target = float(reference[-1])
    target_abs = max(abs(target), 1e-9)

    idx_10 = np.where(speed >= 0.1 * target)[0]
    idx_90 = np.where(speed >= 0.9 * target)[0]
    rise_time = math.nan
    if len(idx_10) and len(idx_90):
        rise_time = float(max(0.0, time[idx_90[0]] - time[idx_10[0]]))

    overshoot = max(0.0, (float(np.max(speed)) - target) / target_abs * 100.0)

    tolerance = 0.02 * target_abs
    outside = np.where(np.abs(error) > tolerance)[0]
    if len(outside) == 0:
        settling_time = 0.0
    elif outside[-1] >= len(time) - 1:
        settling_time = math.nan
    else:
        settling_time = float(time[outside[-1] + 1])

    disturbance_mask = (
        (time >= cfg.disturbance_start)
        & (time <= cfg.disturbance_end + 1.0)
    )
    disturbance_error = float(np.max(np.abs(error[disturbance_mask])))

    absolute_error = np.abs(error)
    integrated_absolute_error = float(
        np.sum(0.5 * (absolute_error[:-1] + absolute_error[1:]) * np.diff(time))
    )

    return {
        "rise_time_s": rise_time,
        "overshoot_percent": float(overshoot),
        "settling_time_s": settling_time,
        "mse": float(np.mean(error**2)),
        "iae": integrated_absolute_error,
        "mean_abs_voltage_v": float(np.mean(np.abs(voltage))),
        "disturbance_max_abs_error": disturbance_error,
    }


def format_number(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def write_time_history(
    pid: dict[str, np.ndarray],
    ml: dict[str, np.ndarray],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "reference_rad_s",
                "pid_speed_rad_s",
                "ml_speed_rad_s",
                "pid_voltage_v",
                "ml_voltage_v",
                "disturbance_torque_nm",
            ]
        )
        for row in zip(
            pid["time"],
            pid["reference"],
            pid["speed"],
            ml["speed"],
            pid["voltage"],
            ml["voltage"],
            pid["disturbance"],
        ):
            writer.writerow([format_number(float(value)) for value in row])


def write_metrics(
    metrics: dict[str, dict[str, float]],
    path: Path,
) -> None:
    metric_names = list(next(iter(metrics.values())).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["controller", *metric_names])
        for controller_name, values in metrics.items():
            writer.writerow(
                [controller_name]
                + [format_number(values[metric]) for metric in metric_names]
            )


def write_training_loss(losses: list[float], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "mse_loss"])
        for index, loss in enumerate(losses, start=1):
            writer.writerow([index, format_number(loss)])


def scale_points(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    points = []
    for xi, yi in zip(x, y):
        px = left + (float(xi) - x_min) / x_span * width
        py = top + height - (float(yi) - y_min) / y_span * height
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def line_plot_svg(
    path: Path,
    title: str,
    y_label: str,
    series: list[tuple[str, np.ndarray, np.ndarray, str]],
    x_label: str = "Time (s)",
) -> None:
    width = 920
    height = 520
    left = 78
    right = 28
    top = 54
    bottom = 76
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_x = np.concatenate([item[1] for item in series])
    all_y = np.concatenate([item[2] for item in series])
    x_min = float(np.min(all_x))
    x_max = float(np.max(all_x))
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    y_margin = max((y_max - y_min) * 0.08, 0.05)
    y_min -= y_margin
    y_max += y_margin

    grid = []
    for index in range(6):
        gx = left + index * plot_width / 5
        x_value = x_min + index * (x_max - x_min) / 5
        grid.append(
            f'<line x1="{gx:.2f}" y1="{top}" x2="{gx:.2f}" y2="{top + plot_height}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{gx:.2f}" y="{top + plot_height + 28}" text-anchor="middle" '
            'font-size="12" fill="#374151">'
            f"{x_value:.1f}</text>"
        )

    for index in range(6):
        gy = top + plot_height - index * plot_height / 5
        y_value = y_min + index * (y_max - y_min) / 5
        grid.append(
            f'<line x1="{left}" y1="{gy:.2f}" x2="{left + plot_width}" y2="{gy:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 12}" y="{gy + 4:.2f}" text-anchor="end" '
            'font-size="12" fill="#374151">'
            f"{y_value:.2f}</text>"
        )

    lines = []
    legend = []
    for index, (label, x, y, color) in enumerate(series):
        points = scale_points(
            x,
            y,
            x_min,
            x_max,
            y_min,
            y_max,
            left,
            top,
            plot_width,
            plot_height,
        )
        safe_label = html.escape(label)
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_x = left + index * 170
        legend_y = height - 24
        legend.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        legend.append(
            f'<text x="{legend_x + 36}" y="{legend_y + 4}" '
            f'font-size="13" fill="#111827">{safe_label}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>
{''.join(grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>
<text x="{width / 2}" y="{height - 42}" text-anchor="middle" font-size="14" fill="#111827">{html.escape(x_label)}</text>
<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle" font-size="14" fill="#111827">{html.escape(y_label)}</text>
{''.join(lines)}
{''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_summary(metrics: dict[str, dict[str, float]], path: Path) -> None:
    pid = metrics["PID"]
    ml = metrics["ML Neural Network"]
    better_mse = "PID" if pid["mse"] <= ml["mse"] else "ML Neural Network"
    better_disturbance = (
        "PID"
        if pid["disturbance_max_abs_error"] <= ml["disturbance_max_abs_error"]
        else "ML Neural Network"
    )

    lines = [
        "# Results Summary",
        "",
        "This file is generated by `src/dc_motor_ml_control.py`.",
        "",
        "## Key Observations",
        "",
        f"- Lower MSE controller: {better_mse}",
        f"- Better disturbance error controller: {better_disturbance}",
        "- PID is interpretable and easy to tune.",
        "- The ML controller learns the expert-control behaviour from data.",
        "- The ML controller should be tested carefully outside its training range.",
        "",
        "## Metrics",
        "",
        "| Controller | Rise time (s) | Overshoot (%) | Settling time (s) | MSE | Disturbance max error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in metrics.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    format_number(values["rise_time_s"]),
                    format_number(values["overshoot_percent"]),
                    format_number(values["settling_time_s"]),
                    format_number(values["mse"]),
                    format_number(values["disturbance_max_abs_error"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_metrics(metrics: dict[str, dict[str, float]]) -> None:
    headers = [
        "controller",
        "rise_time_s",
        "overshoot_percent",
        "settling_time_s",
        "mse",
        "disturbance_max_abs_error",
    ]
    print(" | ".join(headers))
    print("-" * 96)
    for name, values in metrics.items():
        row = [
            name,
            format_number(values["rise_time_s"]),
            format_number(values["overshoot_percent"]),
            format_number(values["settling_time_s"]),
            format_number(values["mse"]),
            format_number(values["disturbance_max_abs_error"]),
        ]
        print(" | ".join(row))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    params = MotorParameters()
    cfg = SimulationConfig()

    features, labels = collect_training_data(params, cfg)
    model, losses = train_neural_controller(features, labels)

    pid_controller = PIDController(PIDGains(), cfg.voltage_limit)
    ml_controller = NeuralNetworkController(model, cfg.voltage_limit)

    pid_result = simulate_controller(pid_controller, params, cfg)
    ml_result = simulate_controller(ml_controller, params, cfg)

    metrics = {
        "PID": calculate_metrics(pid_result, cfg),
        "ML Neural Network": calculate_metrics(ml_result, cfg),
    }

    write_time_history(pid_result, ml_result, OUTPUT_DIR / "time_history.csv")
    write_metrics(metrics, OUTPUT_DIR / "metrics.csv")
    write_training_loss(losses, OUTPUT_DIR / "training_loss.csv")
    write_summary(metrics, OUTPUT_DIR / "results_summary.md")

    line_plot_svg(
        OUTPUT_DIR / "speed_response.svg",
        "DC Motor Speed Response",
        "Speed (rad/s)",
        [
            ("Reference", pid_result["time"], pid_result["reference"], "#111827"),
            ("PID", pid_result["time"], pid_result["speed"], "#2563eb"),
            ("ML Neural Network", ml_result["time"], ml_result["speed"], "#dc2626"),
        ],
    )
    line_plot_svg(
        OUTPUT_DIR / "control_signal.svg",
        "Control Voltage Signal",
        "Voltage (V)",
        [
            ("PID", pid_result["time"], pid_result["voltage"], "#2563eb"),
            ("ML Neural Network", ml_result["time"], ml_result["voltage"], "#dc2626"),
        ],
    )
    line_plot_svg(
        OUTPUT_DIR / "training_loss.svg",
        "Neural Network Training Loss",
        "MSE Loss",
        [
            (
                "Training loss",
                np.arange(1, len(losses) + 1, dtype=float),
                np.array(losses, dtype=float),
                "#059669",
            )
        ],
        x_label="Epoch",
    )

    print_metrics(metrics)
    print(f"\nGenerated outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
