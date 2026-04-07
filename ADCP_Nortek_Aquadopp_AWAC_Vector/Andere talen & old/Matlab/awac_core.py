from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None


WAVE_COLUMNS = {
    "signif_height": 7,
    "wave_h3": 8,
    "wave_max": 10,
    "wave_mean": 11,
    "mean_period": 12,
    "peak_period": 13,
    "period_t3": 15,
    "max_period_tmax": 17,
    "peak_dir": 18,
    "mean_direction": 20,
    "mean_pressure": 22,
    "current_speed": 28,
    "current_direction": 29,
}


def load_wave_file(path: str | Path) -> dict[str, np.ndarray]:
    wave = np.loadtxt(path, ndmin=2)
    out = {"wave": wave, "date_campaign": wave[:, :6]}
    for key, idx in WAVE_COLUMNS.items():
        out[key] = wave[:, idx]
    out["sea_level"] = 1.40 + out["mean_pressure"] * 1.019716
    return out


def load_named_matrix(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if loadmat is not None:
        try:
            data = loadmat(path)
            return {key: value for key, value in data.items() if not key.startswith("__")}
        except Exception:
            pass
    return {path.stem: np.loadtxt(path, ndmin=2)}


def load_awac_currents(path: str | Path, sample_stride: int = 21) -> dict[str, np.ndarray]:
    raw = load_named_matrix(path)
    matrix = next(iter(raw.values()))
    temperature_pressure = matrix[::sample_stride].copy()
    current = np.delete(matrix.copy(), np.s_[::sample_stride], axis=0)
    current = np.delete(current, np.s_[10:19], axis=1)
    output = {
        "temperature_pressure": temperature_pressure,
        "pressure": temperature_pressure[:, 13],
        "current": current,
    }
    for cell in range(1, 9):
        output[f"cell{cell}"] = current[current[:, 0] == cell]
    return output


def select_wave_day(wave_data: dict[str, np.ndarray], month: int, day: int) -> dict[str, np.ndarray]:
    mask = (wave_data["wave"][:, 0] == month) & (wave_data["wave"][:, 1] == day)
    return {key: value[mask] if isinstance(value, np.ndarray) and value.shape[0] == wave_data["wave"].shape[0] else value for key, value in wave_data.items()}


def expand_series_to_length(values: np.ndarray, target_length: int = 14400, step: int = 1200) -> np.ndarray:
    series = np.full(target_length, np.nan, dtype=float)
    series[np.arange(0, min(target_length, len(values) * step), step)] = values[: len(series[::step])]
    return series


def wave_statistics(path: str | Path) -> dict[str, dict[str, float]]:
    wave_data = load_wave_file(path)
    result = {}
    for key in ("signif_height", "wave_max", "peak_period", "mean_direction"):
        values = wave_data[key]
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "kurtosis_proxy": float(np.mean(((values - np.mean(values)) / np.std(values, ddof=1)) ** 4)),
            "skewness_proxy": float(np.mean(((values - np.mean(values)) / np.std(values, ddof=1)) ** 3)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }
    return result


def plot_temperature_pressure_time_series(path: str | Path) -> tuple[plt.Figure, plt.Figure]:
    data = np.loadtxt(path, ndmin=2)
    temperature = data[:, 14]
    pressure = data[:, 13]
    time = data[:, [2, 0, 1, 3, 4]]

    fig_temp, ax_temp = plt.subplots(figsize=(12, 4))
    ax_temp.plot(temperature, "k")
    ax_temp.set_ylabel("Temperature (degC)")

    fig_press, ax_press = plt.subplots(figsize=(12, 4))
    ax_press.plot(pressure, "k")
    ax_press.set_ylabel("Pressure (dbar)")
    return fig_temp, fig_press


def plot_wave_day_summary(path: str | Path, month: int, day: int) -> plt.Figure:
    wave_data = select_wave_day(load_wave_file(path), month, day)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    series = [
        ("wave_max", "max wave"),
        ("current_direction", "current direction"),
        ("current_speed", "current speed"),
        ("mean_direction", "wave direction"),
        ("mean_period", "mean period"),
        ("signif_height", "significant height"),
    ]
    for axis, (key, label) in zip(axes.ravel(), series):
        axis.plot(wave_data[key])
        axis.set_title(label)
    return fig
