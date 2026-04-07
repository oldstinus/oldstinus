#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reken Aquadopp-profieldata terug van ENU naar instrumentassen XYZ
en optioneel verder terug naar beam-ruimte.

Waarom:
- als de Aquadopp tijdens de meting in ENU stond, zijn .v1/.v2/.v3
  al geroteerd met heading/pitch/roll
- voor analyse in instrumentassen wil je juist XYZ (of zelfs beam 1..3)

Model:
    ENU = H(heading) @ Tilt(pitch, roll) @ XYZ
dus:
    XYZ = inv(H @ Tilt) @ ENU

en als de HDR-transformation matrix beam -> XYZ bevat:
    XYZ = T @ BEAM
dus:
    BEAM = inv(T) @ XYZ

Gebruik:
    python terugrekenen_enu_naar_xyz_aquadopp.py --hdr pad\\naar\\bestand.hdr

Als --hdr ontbreekt, opent een bestandsdialoog.
"""

from __future__ import annotations

import argparse
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, simpledialog

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def find_one(text: str, pattern: str, cast=str, default=None):
    match = re.search(pattern, text, flags=re.I | re.S)
    if not match:
        return default
    value = match.group(1).strip()
    return cast(value) if cast else value


def resolve_companion_path(hdr_path: Path, ext: str) -> Path:
    candidate = hdr_path.with_suffix(ext)
    if candidate.exists():
        return candidate
    upper_candidate = hdr_path.with_suffix(ext.upper())
    if upper_candidate.exists():
        return upper_candidate
    raise FileNotFoundError(f"Kon bij {hdr_path.name} geen bestand {ext} vinden.")


def extract_sen_columns(hdr_text: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    block = re.search(r"\[.*?\.sen\](.*?)(?:\r?\n\s*\r?\n|---------------------------------------------------------------------)", hdr_text, flags=re.S | re.I)
    if not block:
        return mapping
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"(\d+)\s+(.+?)\s+\(", line)
        if not match:
            continue
        idx = int(match.group(1))
        name = match.group(2).strip().lower().replace(" ", "_")
        mapping[name] = idx
    return mapping


def parse_hdr(hdr_path: Path) -> dict:
    text = read_text(hdr_path)
    n_meas = find_one(text, r"Number of measurements\s+(\d+)", int)
    n_cells = find_one(text, r"Number of cells\s+(\d+)", int)
    dt_sec = find_one(text, r"Profile interval\s+([0-9.]+)\s*sec", float)
    coord = find_one(text, r"Coordinate system\s+([A-Z]+)", str, default="UNKNOWN")
    t0 = pd.to_datetime(
        find_one(text, r"Time of first measurement\s+(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2})"),
        dayfirst=True,
    )

    cell_distances: list[float] = []
    block = re.search(
        r"Current profile cell center distance from head \(m\)\s*\n[-]+\s*\n(.*?)(?:\n\s*\n|Data file format)",
        text,
        flags=re.S | re.I,
    )
    if block:
        for line in block.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    cell_distances.append(float(parts[1]))
                except ValueError:
                    pass

    mat_match = re.search(
        r"Transformation matrix\s+([0-9\.\-\s]+)\n\s*([0-9\.\-\s]+)\n\s*([0-9\.\-\s]+)",
        text,
        flags=re.I,
    )
    if not mat_match:
        raise ValueError("Transformation matrix niet gevonden in HDR.")
    transform_rows = []
    for row_index in range(1, 4):
        row = [float(value) for value in mat_match.group(row_index).split()]
        if len(row) != 3:
            raise ValueError("Transformation matrix in HDR is niet 3x3.")
        transform_rows.append(row)

    return {
        "n_meas": n_meas,
        "n_cells": n_cells,
        "dt_sec": dt_sec,
        "t0": t0,
        "coordinate_system": coord,
        "cell_distances_m": cell_distances,
        "beam_to_xyz": np.array(transform_rows, dtype=float),
        "sen_columns": extract_sen_columns(text),
    }


def build_time_index(t0: pd.Timestamp, n_meas: int, dt_sec: float) -> pd.DatetimeIndex:
    return pd.date_range(start=t0, periods=n_meas, freq=pd.to_timedelta(dt_sec, unit="s"))


def load_matrix(path: Path, n_meas: int, n_cells: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=float, ndmin=2)
    if data.shape[1] != n_cells:
        raise ValueError(f"{path.name}: verwacht {n_cells} kolommen, kreeg {data.shape[1]}.")
    return data[:n_meas, :]


def load_sen_orientation(sen_path: Path, sen_columns: dict[str, int], n_meas: int) -> pd.DataFrame:
    data = np.loadtxt(sen_path, dtype=float, ndmin=2)[:n_meas, :]
    if not sen_columns:
        sen_columns = {
            "heading": 11,
            "pitch": 12,
            "roll": 13,
            "pressure": 14,
            "temperature": 15,
        }
    get_col = lambda name, default: sen_columns.get(name, default) - 1
    return pd.DataFrame(
        {
            "heading": data[:, get_col("heading", 11)],
            "pitch": data[:, get_col("pitch", 12)],
            "roll": data[:, get_col("roll", 13)],
            "pressure": data[:, get_col("pressure", 14)] if data.shape[1] >= get_col("pressure", 14) + 1 else np.nan,
            "temperature": data[:, get_col("temperature", 15)] if data.shape[1] >= get_col("temperature", 15) + 1 else np.nan,
        }
    )


def parse_cell_selection(selection: str | None, n_cells: int) -> tuple[int, int]:
    if not selection:
        return 0, n_cells

    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", selection)
    if not match:
        raise ValueError("Gebruik voor --cells het formaat start:end, bijvoorbeeld 1:5.")

    start_cell = int(match.group(1))
    end_cell = int(match.group(2))
    if start_cell < 1 or end_cell < 1:
        raise ValueError("Celnummers starten bij 1.")
    if start_cell > end_cell:
        raise ValueError("Bij --cells moet start <= end zijn.")
    if end_cell > n_cells:
        raise ValueError(f"Gevraagde eindcel {end_cell} valt buiten het bereik 1..{n_cells}.")

    return start_cell - 1, end_cell


def choose_cells_interactively(n_cells: int) -> tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    selection = simpledialog.askstring(
        "Cellen kiezen",
        f"Kies celbereik voor uitmiddelen (start:end, 1..{n_cells}).\nLaat leeg voor alle cellen.",
        initialvalue=f"1:{n_cells}",
        parent=root,
    )
    root.destroy()
    return parse_cell_selection(selection, n_cells)


def parse_avg_points(avg_points: int | None, n_points: int) -> int:
    if avg_points is None:
        return 1
    if avg_points < 1:
        raise ValueError("Aantal meetpunten voor uitmiddelen moet minstens 1 zijn.")
    if avg_points > n_points:
        raise ValueError(f"Aantal meetpunten voor uitmiddelen ({avg_points}) is groter dan het aantal metingen ({n_points}).")
    return avg_points


def choose_avg_points_interactively(n_points: int) -> int:
    root = tk.Tk()
    root.withdraw()
    avg_points = simpledialog.askinteger(
        "Uitmiddelen",
        f"Kies aantal meetpunten voor uitmiddelen in de tijd (1..{n_points}).\n1 = geen extra uitmiddeling.",
        initialvalue=1,
        minvalue=1,
        maxvalue=n_points,
        parent=root,
    )
    root.destroy()
    return parse_avg_points(avg_points, n_points)


def mean_over_cells(data: np.ndarray, cell_slice: slice) -> np.ndarray:
    return np.nanmean(data[:, cell_slice], axis=1)


def smooth_series(data: np.ndarray, avg_points: int) -> np.ndarray:
    if avg_points <= 1:
        return np.asarray(data, dtype=float)
    return (
        pd.Series(np.asarray(data, dtype=float))
        .rolling(window=avg_points, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def heading_matrix_deg(heading_deg: float) -> np.ndarray:
    heading = np.deg2rad(heading_deg - 90.0)
    return np.array(
        [
            [np.cos(heading), np.sin(heading), 0.0],
            [-np.sin(heading), np.cos(heading), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def tilt_matrix_deg(pitch_deg: float, roll_deg: float) -> np.ndarray:
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    return np.array(
        [
            [np.cos(pitch), -np.sin(pitch) * np.sin(roll), -np.cos(roll) * np.sin(pitch)],
            [0.0, np.cos(roll), -np.sin(roll)],
            [np.sin(pitch), np.sin(roll) * np.cos(pitch), np.cos(pitch) * np.cos(roll)],
        ],
        dtype=float,
    )


def recover_xyz_from_enu(
    enu_east: np.ndarray,
    enu_north: np.ndarray,
    enu_up: np.ndarray,
    heading: np.ndarray,
    pitch: np.ndarray,
    roll: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_time, n_cells = enu_east.shape
    x = np.full((n_time, n_cells), np.nan, dtype=float)
    y = np.full((n_time, n_cells), np.nan, dtype=float)
    z = np.full((n_time, n_cells), np.nan, dtype=float)

    for t in range(n_time):
        rotation = heading_matrix_deg(float(heading[t])) @ tilt_matrix_deg(float(pitch[t]), float(roll[t]))
        inverse_rotation = np.linalg.inv(rotation)
        enu_stack = np.stack([enu_east[t, :], enu_north[t, :], enu_up[t, :]], axis=0)
        xyz_stack = inverse_rotation @ enu_stack
        x[t, :] = xyz_stack[0, :]
        y[t, :] = xyz_stack[1, :]
        z[t, :] = xyz_stack[2, :]
    return x, y, z


def recover_beams_from_xyz(beam_to_xyz: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_to_beam = np.linalg.inv(beam_to_xyz)
    xyz_stack = np.stack([x, y, z], axis=1)
    beam_stack = np.einsum("ij,tjk->tik", xyz_to_beam, xyz_stack)
    return beam_stack[:, 0, :], beam_stack[:, 1, :], beam_stack[:, 2, :]


def wide_component_frame(time_index: pd.DatetimeIndex, prefix: str, data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(data, index=time_index, columns=[f"{prefix}_cell{idx+1}" for idx in range(data.shape[1])])


def export_outputs(
    out_prefix: Path,
    time_index: pd.DatetimeIndex,
    orientation: pd.DataFrame,
    enu: tuple[np.ndarray, np.ndarray, np.ndarray],
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    beam: tuple[np.ndarray, np.ndarray, np.ndarray],
    cell_distances: list[float],
) -> tuple[Path, Path]:
    e, n, u = enu
    x, y, z = xyz
    b1, b2, b3 = beam

    df = pd.concat(
        [
            pd.DataFrame({"DateTime": time_index}),
            orientation.reset_index(drop=True),
            wide_component_frame(time_index, "east", e).reset_index(drop=True),
            wide_component_frame(time_index, "north", n).reset_index(drop=True),
            wide_component_frame(time_index, "up", u).reset_index(drop=True),
            wide_component_frame(time_index, "x_recovered", x).reset_index(drop=True),
            wide_component_frame(time_index, "y_recovered", y).reset_index(drop=True),
            wide_component_frame(time_index, "z_recovered", z).reset_index(drop=True),
            wide_component_frame(time_index, "beam1_recovered", b1).reset_index(drop=True),
            wide_component_frame(time_index, "beam2_recovered", b2).reset_index(drop=True),
            wide_component_frame(time_index, "beam3_recovered", b3).reset_index(drop=True),
        ],
        axis=1,
    )

    csv_path = out_prefix.with_name(out_prefix.name + "_teruggerekend_xyz_beam.csv")
    npz_path = out_prefix.with_name(out_prefix.name + "_teruggerekend_xyz_beam.npz")
    df.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        time=time_index.astype("datetime64[ns]").to_numpy(),
        cell_distances_m=np.asarray(cell_distances, dtype=float),
        east=e,
        north=n,
        up=u,
        x=x,
        y=y,
        z=z,
        beam1=b1,
        beam2=b2,
        beam3=b3,
        heading=orientation["heading"].to_numpy(dtype=float),
        pitch=orientation["pitch"].to_numpy(dtype=float),
        roll=orientation["roll"].to_numpy(dtype=float),
    )
    return csv_path, npz_path


def classify_tide_from_pressure(pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.asarray(pressure, dtype=float)
    dpressure = np.gradient(pressure)
    threshold = max(np.nanstd(dpressure) * 0.05, 1e-6)
    tide_phase = np.full(pressure.shape, "kentering", dtype=object)
    tide_phase[dpressure > threshold] = "opkomend getij"
    tide_phase[dpressure < -threshold] = "afgaand getij"
    return dpressure, tide_phase


def compute_flow_metrics(east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(np.square(east) + np.square(north))
    direction = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    return speed, direction


def classify_velocity_direction(velocity: np.ndarray) -> tuple[np.ndarray, float]:
    velocity = np.asarray(velocity, dtype=float)
    threshold = max(np.nanstd(velocity) * 0.05, 1e-6)
    direction = np.full(velocity.shape, "stilstand", dtype=object)
    direction[velocity > threshold] = "inwatering (+X)"
    direction[velocity < -threshold] = "uitwatering (-X)"
    return direction, threshold


def add_velocity_band(
    ax: plt.Axes,
    time_index: pd.DatetimeIndex,
    velocity: np.ndarray,
) -> plt.Axes:
    velocity_ax = ax.twinx()
    velocity_ax.spines["right"].set_position(("axes", 1.10))
    velocity_ax.spines["right"].set_visible(True)
    velocity_ax.set_frame_on(True)
    velocity_ax.patch.set_visible(False)
    velocity_direction, threshold = classify_velocity_direction(velocity)
    vel_limit = max(float(np.nanmax(np.abs(velocity))), threshold) * 1.1
    if vel_limit <= 0:
        vel_limit = 1.0

    velocity_ax.fill_between(
        time_index,
        velocity,
        0.0,
        where=velocity_direction == "inwatering (+X)",
        color="tab:green",
        alpha=0.18,
    )
    velocity_ax.fill_between(
        time_index,
        velocity,
        0.0,
        where=velocity_direction == "uitwatering (-X)",
        color="tab:red",
        alpha=0.18,
    )
    velocity_ax.fill_between(
        time_index,
        velocity,
        0.0,
        where=velocity_direction == "stilstand",
        color="tab:gray",
        alpha=0.12,
    )
    velocity_ax.set_ylim(-vel_limit, vel_limit)
    velocity_ax.set_ylabel("X-snelheid")
    velocity_ax.tick_params(axis="y", pad=8)
    return velocity_ax


def plot_x_tide_link_detail(
    time_index: pd.DatetimeIndex,
    x_mean: np.ndarray,
    pressure: np.ndarray,
    dpressure: np.ndarray,
    cell_label: str,
) -> None:
    fig, tide_ax = plt.subplots(1, 1, figsize=(16, 6), sharex=True)
    tide_ax.plot(time_index, pressure, color="black", linewidth=1.0, label="druk")
    tide_ax2 = tide_ax.twinx()
    tide_ax2.plot(time_index, dpressure, color="tab:orange", linewidth=1.0, alpha=0.8, label="d(druk)/dt")
    add_velocity_band(tide_ax, time_index, x_mean)
    tide_ax.set_title("Getij uit druk: stijgend = opkomend, dalend = afgaand")
    tide_ax.set_ylabel("pressure")
    tide_ax2.set_ylabel("dP/dt")
    tide_ax.grid(True, alpha=0.3)

    tide_handles, tide_labels = tide_ax.get_legend_handles_labels()
    rate_handles, rate_labels = tide_ax2.get_legend_handles_labels()
    tide_ax.legend(
        tide_handles
        + rate_handles
        + [
            Patch(facecolor="tab:green", alpha=0.18, label="inwatering (+X)"),
            Patch(facecolor="tab:red", alpha=0.18, label="uitwatering (-X)"),
            Patch(facecolor="tab:gray", alpha=0.12, label="stilstand"),
        ],
        tide_labels + rate_labels + ["inwatering (+X)", "uitwatering (-X)", "stilstand"],
        loc="upper left",
    )

    fig.autofmt_xdate()
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
    add_interaction(fig, np.array([tide_ax], dtype=object))


def summarize_x_vs_tide(x_mean: np.ndarray, tide_phase: np.ndarray) -> dict[str, int]:
    x_direction = np.full(x_mean.shape, "stilstand", dtype=object)
    x_direction[x_mean > 0] = "inwatering (+X)"
    x_direction[x_mean < 0] = "uitwatering (-X)"

    summary = {
        "inwatering_bij_opkomend": int(np.sum((x_direction == "inwatering (+X)") & (tide_phase == "opkomend getij"))),
        "inwatering_bij_afgaand": int(np.sum((x_direction == "inwatering (+X)") & (tide_phase == "afgaand getij"))),
        "uitwatering_bij_opkomend": int(np.sum((x_direction == "uitwatering (-X)") & (tide_phase == "opkomend getij"))),
        "uitwatering_bij_afgaand": int(np.sum((x_direction == "uitwatering (-X)") & (tide_phase == "afgaand getij"))),
        "kentering": int(np.sum(tide_phase == "kentering")),
    }
    return summary


def add_interaction(fig: plt.Figure, axes: np.ndarray) -> None:
    annotations = {}
    for ax in axes:
        annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.85},
            arrowprops={"arrowstyle": "->", "alpha": 0.5},
        )
        annotation.set_visible(False)
        annotations[ax] = annotation

    def on_scroll(event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        scale = 0.85 if event.button == "up" else 1.15
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        x_range = (x_right - x_left) * scale
        y_range = (y_top - y_bottom) * scale
        x_rel = (event.xdata - x_left) / (x_right - x_left) if x_right != x_left else 0.5
        y_rel = (event.ydata - y_bottom) / (y_top - y_bottom) if y_top != y_bottom else 0.5
        ax.set_xlim(event.xdata - x_range * x_rel, event.xdata + x_range * (1 - x_rel))
        ax.set_ylim(event.ydata - y_range * y_rel, event.ydata + y_range * (1 - y_rel))
        fig.canvas.draw_idle()

    def on_move(event) -> None:
        if event.inaxes is None or event.xdata is None:
            changed = False
            for annotation in annotations.values():
                if annotation.get_visible():
                    annotation.set_visible(False)
                    changed = True
            if changed:
                fig.canvas.draw_idle()
            return

        ax = event.inaxes
        annotation = annotations.get(ax)
        if annotation is None:
            return
        best_line = None
        best_idx = None
        best_dist = None
        for line in ax.lines:
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata(), dtype=float)
            if xdata.size == 0 or ydata.size == 0:
                continue
            try:
                distances = np.abs(xdata - event.xdata)
            except TypeError:
                continue
            idx = int(np.argmin(distances))
            if np.isnan(ydata[idx]):
                continue
            dist = float(distances[idx])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
                best_line = line

        if best_line is None or best_idx is None:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        xdata = np.asarray(best_line.get_xdata())
        ydata = np.asarray(best_line.get_ydata(), dtype=float)
        annotation.xy = (xdata[best_idx], ydata[best_idx])
        label = best_line.get_label()
        annotation.set_text(f"{label}\n{pd.to_datetime(xdata[best_idx])}\n{ydata[best_idx]:.3f}")
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("motion_notify_event", on_move)


def plot_quickcheck(
    time_index: pd.DatetimeIndex,
    enu: tuple[np.ndarray, np.ndarray, np.ndarray],
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    pressure: np.ndarray,
    cell_slice: slice,
    cell_label: str,
    avg_points: int,
    avg_label: str,
) -> None:
    e, n, u = enu
    x, y, z = xyz
    e_mean = smooth_series(mean_over_cells(e, cell_slice), avg_points)
    n_mean = smooth_series(mean_over_cells(n, cell_slice), avg_points)
    u_mean = smooth_series(mean_over_cells(u, cell_slice), avg_points)
    x_mean = smooth_series(mean_over_cells(x, cell_slice), avg_points)
    y_mean = smooth_series(mean_over_cells(y, cell_slice), avg_points)
    z_mean = smooth_series(mean_over_cells(z, cell_slice), avg_points)
    pressure_mean = smooth_series(pressure, avg_points)
    flow_speed, flow_direction = compute_flow_metrics(e_mean, n_mean)
    dpressure, tide_phase = classify_tide_from_pressure(pressure_mean)

    fig, axes = plt.subplots(8, 1, figsize=(12, 19), sharex=True)
    mean_pairs = [
        (e_mean, x_mean, f"East -> recovered X ({cell_label}, {avg_label})"),
        (n_mean, y_mean, f"North -> recovered Y ({cell_label}, {avg_label})"),
        (u_mean, z_mean, f"Up -> recovered Z ({cell_label}, {avg_label})"),
    ]
    for idx, (original, recovered, title) in enumerate(mean_pairs):
        velocity_ax = axes[idx * 2]
        pressure_ax = axes[idx * 2 + 1]

        velocity_ax.plot(time_index, original, label="origineel ENU", linewidth=1.0)
        velocity_ax.plot(time_index, recovered, label="teruggerekend XYZ", linewidth=1.0)
        velocity_ax.set_title(title)
        velocity_ax.grid(True, alpha=0.3)
        velocity_ax.legend()

        pressure_ax.plot(time_index, pressure_mean, color="black", label="druk", linewidth=1.0)
        pressure_ax.set_title(f"Druk bij {title}")
        pressure_ax.set_ylabel("pressure")
        pressure_ax.grid(True, alpha=0.3)
        pressure_ax.legend()

    flow_ax = axes[6]
    flow_ax.plot(time_index, flow_speed, color="tab:cyan", linewidth=1.0, label="grootte |U|")
    flow_ax.set_title(f"Dieptegemiddelde horizontale stroming: grootte en richting ({cell_label}, {avg_label})")
    flow_ax.set_ylabel("|U|")
    flow_ax.grid(True, alpha=0.3)
    flow_ax.legend(loc="upper left")

    direction_ax = flow_ax.twinx()
    direction_ax.plot(time_index, flow_direction, color="tab:purple", linewidth=1.0, alpha=0.85, label="richting")
    direction_ax.set_ylabel("richting [deg]")
    direction_ax.set_ylim(0.0, 360.0)
    direction_ax.set_yticks([0.0, 90.0, 180.0, 270.0, 360.0])
    direction_ax.set_yticklabels(["N", "O", "Z", "W", "N"])
    speed_handles, speed_labels = flow_ax.get_legend_handles_labels()
    direction_handles, direction_labels = direction_ax.get_legend_handles_labels()
    flow_ax.legend(speed_handles + direction_handles, speed_labels + direction_labels, loc="upper left")

    tide_ax = axes[7]
    tide_ax.plot(time_index, pressure_mean, color="black", linewidth=1.0, label="druk")
    tide_ax2 = tide_ax.twinx()
    tide_ax2.plot(time_index, dpressure, color="tab:orange", linewidth=1.0, alpha=0.8, label="d(druk)/dt")
    velocity_band_ax = add_velocity_band(tide_ax, time_index, x_mean)
    tide_ax.set_title("Getij uit druk: stijgend = opkomend, dalend = afgaand")
    tide_ax.set_ylabel("pressure")
    tide_ax2.set_ylabel("dP/dt")
    tide_ax.grid(True, alpha=0.3)

    tide_handles, tide_labels = tide_ax.get_legend_handles_labels()
    rate_handles, rate_labels = tide_ax2.get_legend_handles_labels()
    tide_ax.legend(
        tide_handles
        + rate_handles
        + [
            Patch(facecolor="tab:green", alpha=0.18, label="inwatering (+X)"),
            Patch(facecolor="tab:red", alpha=0.18, label="uitwatering (-X)"),
            Patch(facecolor="tab:gray", alpha=0.12, label="stilstand"),
        ],
        tide_labels + rate_labels + ["inwatering (+X)", "uitwatering (-X)", "stilstand"],
        loc="upper left",
    )

    fig.autofmt_xdate()
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
    add_interaction(fig, axes)
    plot_x_tide_link_detail(time_index, x_mean, pressure_mean, dpressure, f"{cell_label}, {avg_label}")
    plt.show()


def choose_hdr_interactively() -> Path:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Kies Aquadopp .hdr bestand",
        filetypes=[("HDR files", "*.hdr *.HDR"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        raise SystemExit("Geen .hdr bestand gekozen.")
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reken Aquadopp ENU terug naar XYZ en beam.")
    parser.add_argument("--hdr", help="Pad naar .hdr bestand")
    parser.add_argument("--cells", help="Celbereik voor uitmiddelen, formaat start:end (1-gebaseerd), bijvoorbeeld 1:5")
    parser.add_argument("--avg-points", type=int, help="Aantal meetpunten voor rollend uitmiddelen in de tijd")
    parser.add_argument("--no-plot", action="store_true", help="Geen controleplot tonen")
    args = parser.parse_args()

    hdr_path = Path(args.hdr) if args.hdr else choose_hdr_interactively()
    meta = parse_hdr(hdr_path)

    if str(meta["coordinate_system"]).upper() != "ENU":
        print(f"Waarschuwing: HDR zegt coordinate system = {meta['coordinate_system']}. Script rekent uit van ENU naar XYZ.")

    v1_path = resolve_companion_path(hdr_path, ".v1")
    v2_path = resolve_companion_path(hdr_path, ".v2")
    v3_path = resolve_companion_path(hdr_path, ".v3")
    sen_path = resolve_companion_path(hdr_path, ".sen")
    if args.cells:
        cell_start, cell_end = parse_cell_selection(args.cells, meta["n_cells"])
    else:
        cell_start, cell_end = choose_cells_interactively(meta["n_cells"])
    cell_slice = slice(cell_start, cell_end)
    cell_label = f"cellen {cell_start + 1}-{cell_end}"

    time_index = build_time_index(meta["t0"], meta["n_meas"], meta["dt_sec"])
    east = load_matrix(v1_path, meta["n_meas"], meta["n_cells"])
    north = load_matrix(v2_path, meta["n_meas"], meta["n_cells"])
    up = load_matrix(v3_path, meta["n_meas"], meta["n_cells"])
    n_eff = min(len(time_index), east.shape[0], north.shape[0], up.shape[0])

    time_index = time_index[:n_eff]
    east = east[:n_eff, :]
    north = north[:n_eff, :]
    up = up[:n_eff, :]
    if args.avg_points is not None:
        avg_points = parse_avg_points(args.avg_points, n_eff)
    else:
        avg_points = choose_avg_points_interactively(n_eff)
    avg_label = "geen tijdsuitmiddeling" if avg_points == 1 else f"{avg_points}-puntsgemiddelde"
    orientation = load_sen_orientation(sen_path, meta["sen_columns"], n_eff)
    orientation.index = time_index

    x, y, z = recover_xyz_from_enu(
        east,
        north,
        up,
        orientation["heading"].to_numpy(dtype=float),
        orientation["pitch"].to_numpy(dtype=float),
        orientation["roll"].to_numpy(dtype=float),
    )
    beam1, beam2, beam3 = recover_beams_from_xyz(meta["beam_to_xyz"], x, y, z)

    csv_path, npz_path = export_outputs(
        hdr_path.with_suffix(""),
        time_index,
        orientation,
        (east, north, up),
        (x, y, z),
        (beam1, beam2, beam3),
        meta["cell_distances_m"],
    )

    print(f"Klaar.\nCSV: {csv_path}\nNPZ: {npz_path}")
    print("Belangrijk: dit is een inverse rotatie uit ENU op basis van heading/pitch/roll per ensemble.")
    print(f"Gemiddelden in samenvatting en grafieken gebruiken {cell_label}.")
    print(f"Tijdsuitmiddeling: {avg_label}.")
    x_mean = smooth_series(mean_over_cells(x, cell_slice), avg_points)
    pressure_mean = smooth_series(orientation["pressure"].to_numpy(dtype=float), avg_points)
    _, tide_phase = classify_tide_from_pressure(pressure_mean)
    summary = summarize_x_vs_tide(x_mean, tide_phase)
    print("Koppeling X-stroming vs getij (aanname: +X = inwatering, -X = uitwatering):")
    print(f"  Inwatering bij opkomend getij : {summary['inwatering_bij_opkomend']}")
    print(f"  Inwatering bij afgaand getij  : {summary['inwatering_bij_afgaand']}")
    print(f"  Uitwatering bij opkomend getij: {summary['uitwatering_bij_opkomend']}")
    print(f"  Uitwatering bij afgaand getij : {summary['uitwatering_bij_afgaand']}")
    print(f"  Kentering                     : {summary['kentering']}")

    if not args.no_plot:
        plot_quickcheck(
            time_index,
            (east, north, up),
            (x, y, z),
            orientation["pressure"].to_numpy(dtype=float),
            cell_slice,
            cell_label,
            avg_points,
            avg_label,
        )


if __name__ == "__main__":
    main()
