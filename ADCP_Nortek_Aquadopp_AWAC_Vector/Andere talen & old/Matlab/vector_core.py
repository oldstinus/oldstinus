from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from awac_core import expand_series_to_length, load_wave_file


def load_vector_dat_files(folder: str | Path, pattern: str = "VEC*.dat") -> list[tuple[str, np.ndarray]]:
    folder = Path(folder)
    return [(path.name, np.loadtxt(path, ndmin=2)) for path in sorted(folder.glob(pattern))]


def block_average(values: np.ndarray, block_size: int = 32) -> np.ndarray:
    full_blocks = len(values) // block_size
    remainder = len(values) % block_size
    chunks = values[: full_blocks * block_size].reshape(full_blocks, block_size).mean(axis=1)
    if remainder:
        chunks = np.concatenate([chunks, [values[-remainder:].mean()]])
    return chunks


def compute_direction(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    return np.mod(np.degrees(np.arctan2(north, east)), 360.0)


def aggregate_vector_dataset(folder: str | Path, pattern: str = "VEC*.dat") -> dict[str, np.ndarray]:
    datasets = load_vector_dat_files(folder, pattern)
    if not datasets:
        raise FileNotFoundError(f"No vector files matching {pattern!r} found in {folder}")

    east = []
    north = []
    up = []
    pressure = []
    for _, dataset in datasets:
        east.append(block_average(dataset[:, 2]))
        north.append(block_average(dataset[:, 3]))
        up.append(block_average(dataset[:, 4]))
        pressure.append(block_average(dataset[:, 14]))

    vec_e = np.concatenate(east)
    vec_n = np.concatenate(north)
    vec_u = np.concatenate(up)
    press = np.concatenate(pressure)
    velocity_norm = np.sqrt(vec_e**2 + vec_n**2 + vec_u**2)
    direction = compute_direction(vec_e, vec_n)
    ut = vec_e - np.mean(vec_e)
    vt = vec_n - np.mean(vec_n)
    wt = vec_u - np.mean(vec_u)

    return {
        "vec_e_date": vec_e,
        "vec_n_date": vec_n,
        "vec_u_date": vec_u,
        "press_date": press,
        "surface_vec": press * 1.019716 + 0.56 + 0.217,
        "vel3_date": velocity_norm,
        "turb_tke_date": 0.5 * (ut**2 + vt**2 + wt**2),
        "r_st_date": ut * wt,
        "alfa_dir_date": direction,
    }


def build_awac_sea_level_overlay(wave_path: str | Path, month: int, day: int, target_length: int) -> np.ndarray:
    wave_data = load_wave_file(wave_path)
    mask = (wave_data["wave"][:, 0] == month) & (wave_data["wave"][:, 1] == day)
    return expand_series_to_length(wave_data["sea_level"][mask], target_length=target_length)


def plot_vector_overview(result: dict[str, np.ndarray], awac_overlay: np.ndarray | None = None) -> plt.Figure:
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    plot_specs = [
        ("vel3_date", "vel (m/s)"),
        ("surface_vec", "depth (m)"),
        ("alfa_dir_date", "vel dir (deg)"),
        ("turb_tke_date", "TKE (m^2/s^2)"),
        ("vec_e_date", "u (m/s)"),
        ("r_st_date", "u'w' (m^2/s^2)"),
        ("vec_n_date", "v (m/s)"),
        (None, "SSC (mg/l)"),
        ("vec_u_date", "w (m/s)"),
        (None, "SSC (mg/l)"),
    ]
    for axis, (key, label) in zip(axes, plot_specs):
        if key is not None:
            axis.plot(result[key])
        if key == "surface_vec" and awac_overlay is not None:
            axis.plot(awac_overlay, "r:+")
        axis.set_ylabel(label)
    return fig
