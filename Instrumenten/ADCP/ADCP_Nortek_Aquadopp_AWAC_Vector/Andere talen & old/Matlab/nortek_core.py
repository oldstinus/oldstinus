from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MATLAB_EPOCH_OFFSET = 366


def matlab_datenum(dt: datetime) -> float:
    day_fraction = (
        dt.hour / 24
        + dt.minute / 1440
        + dt.second / 86400
        + dt.microsecond / 86400_000000
    )
    return dt.toordinal() + MATLAB_EPOCH_OFFSET + day_fraction


def matlab_datenum_to_datetime(value: float) -> datetime:
    ordinal = int(np.floor(value))
    fraction = float(value) - ordinal
    return datetime.fromordinal(ordinal - MATLAB_EPOCH_OFFSET) + timedelta(days=fraction)


def normalize_date_string(value: str) -> str:
    return (
        value.strip()
        .replace("_", " ")
        .replace("Sept", "Sep")
        .replace("  ", " ")
    )


def parse_matlab_date(value: Any) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)

    text = normalize_date_string(str(value))
    formats = (
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            return matlab_datenum(datetime.strptime(text, fmt))
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value!r}")


def save_figure(fig: plt.Figure, out: "OutputData", stem: str) -> None:
    if out.save_format:
        out.figure_folder_name.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.figure_folder_name / f"{stem}.{out.save_format}", dpi=200, bbox_inches="tight")


def add_super_label(fig: plt.Figure, text: str, which_label: str = "t") -> None:
    if which_label == "t":
        fig.suptitle(text)
    elif which_label == "x":
        fig.supxlabel(text)
    elif which_label in {"y", "yy"}:
        fig.supylabel(text)
    else:
        raise ValueError(f"Unsupported label selector: {which_label}")


@dataclass
class DeploymentData:
    folder_name: Path
    dpl_name: str
    z0: float = 0.0
    vertical_dir: str = "upward"
    reference_height: str = "TAW"
    data_treatment: str = "raw"
    time_gap: float = 0.0
    nb_measurements: int = 0
    nb_cells: int = 0
    coordinate_system: str = "ENU"
    z: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sen_columns: dict[str, int] = field(default_factory=dict)
    v_columns: dict[str, int] = field(default_factory=dict)
    time: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    pressure: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    heading: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    pitch: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    roll: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    temperature: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    v: np.ndarray = field(default_factory=lambda: np.empty((3, 0, 0), dtype=float))


@dataclass
class OutputData:
    time_start: float
    time_stop: float
    time_start_index: int
    time_stop_index: int
    time: np.ndarray
    cell_start: int
    cell_stop: int
    cells: np.ndarray
    z: np.ndarray
    save_format: str = ""
    figure_folder_name: Path = Path("figures")


def read_deployment_parameters(dpl_param: dict[str, Any]) -> DeploymentData:
    if "folder_name" not in dpl_param:
        raise ValueError("dpl_param.folder_name must be defined")
    if "dpl_name" not in dpl_param:
        raise ValueError("dpl_param.dpl_name must be defined")

    vertical_dir = dpl_param.get("vertical_dir", "upward")
    if vertical_dir not in {"upward", "downward"}:
        raise ValueError(f'vertical_dir should be "upward" or "downward", not "{vertical_dir}"')

    return DeploymentData(
        folder_name=Path(dpl_param["folder_name"]),
        dpl_name=dpl_param["dpl_name"],
        z0=float(dpl_param.get("z0", 0.0)),
        vertical_dir=vertical_dir,
        reference_height=dpl_param.get("reference_height", "TAW"),
        data_treatment=dpl_param.get("data_treatment", "raw"),
        time_gap=float(dpl_param.get("time_gap", 0.0)),
    )


def _split_double_space(line: str) -> list[str]:
    return [part for part in line.split("  ") if part.strip()]


def load_hdr_file(dpl: DeploymentData) -> dict[str, Any]:
    path = dpl.folder_name / f"{dpl.dpl_name}.hdr"
    hdr: dict[str, Any] = {"sen_columns": {}, "v_columns": {}, "z": {}}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if stripped.startswith("Number of measurements"):
            hdr["nb_measurements"] = int(_split_double_space(line)[-1])
        elif stripped.startswith("Time of first measurement"):
            hdr["time_start"] = parse_matlab_date(_split_double_space(line)[-1])
        elif stripped.startswith("Time of last measurement"):
            hdr["time_stop"] = parse_matlab_date(_split_double_space(line)[-1])
        elif stripped.startswith("Profile interval") or stripped.startswith("Measurement/Burst interval"):
            parts = _split_double_space(line)[-1].split()
            if parts[-1] != "sec":
                raise NotImplementedError("Only second-based profile intervals are supported")
            hdr["dt"] = float(parts[0])
        elif stripped.startswith("Number of cells"):
            hdr["nb_cells"] = int(_split_double_space(line)[-1])
        elif stripped.startswith("Coordinate system"):
            hdr["coordinate_system"] = _split_double_space(line)[-1].split()[-1]
        elif stripped.startswith("Current profile cell center distance from head (m)"):
            idx += 1
            while idx < len(lines):
                next_line = lines[idx].strip()
                if not next_line:
                    break
                cell_parts = [part for part in lines[idx].split("   ") if part.strip()]
                cell_index = int(cell_parts[0])
                distance = float(cell_parts[-1])
                hdr["z"][cell_index] = dpl.z0 + distance if dpl.vertical_dir == "upward" else dpl.z0 - distance
                idx += 1
        elif stripped.startswith("Current profile cell center distances from transducer head."):
            idx += 3
            while idx < len(lines):
                next_line = lines[idx].strip()
                if not next_line:
                    break
                cell_parts = [part for part in lines[idx].split("   ") if part.strip()]
                cell_index = int(cell_parts[0])
                distance = float(cell_parts[-1])
                hdr["z"][cell_index] = dpl.z0 + distance if dpl.vertical_dir == "upward" else dpl.z0 - distance
                idx += 1
        elif stripped.endswith(".sen"):
            idx += 1
            while idx < len(lines) and lines[idx].strip():
                parts = _split_double_space(lines[idx])
                if len(parts) >= 2:
                    hdr["sen_columns"][parts[1].strip().lower().replace(" ", "_")] = int(parts[0])
                idx += 1
        elif stripped.endswith(".v1"):
            idx += 1
            while idx < len(lines) and lines[idx].strip():
                parts = _split_double_space(lines[idx])
                if len(parts) >= 2 and parts[1].strip() == "Velocity Cell 1 (Beam1|X|East)":
                    hdr["v_columns"]["cell_1"] = int(parts[0])
                idx += 1
        idx += 1

    max_cell = max(hdr["z"].keys()) if hdr["z"] else 0
    hdr["z"] = np.array([hdr["z"][i] for i in range(1, max_cell + 1)], dtype=float)
    return hdr


def _load_numeric_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)


def load_sen_file(dpl: DeploymentData) -> dict[str, np.ndarray]:
    suffix = ".sen" if dpl.data_treatment == "raw" else "_p.sen"
    if dpl.data_treatment not in {"raw", "storm"}:
        raise ValueError('load_sen_file only supports data_treatment "raw" or "storm"')

    mat = _load_numeric_matrix(dpl.folder_name / f"{dpl.dpl_name}{suffix}")
    col = lambda key: dpl.sen_columns[key] - 1

    dt_list = []
    for row in mat:
        dt = datetime(
            int(row[col("year")]),
            int(row[col("month")]),
            int(row[col("day")]),
            int(row[col("hour")]),
            int(row[col("minute")]),
            int(row[col("second")]),
        ) - timedelta(seconds=dpl.time_gap)
        dt_list.append(matlab_datenum(dt))

    return {
        "time": np.array(dt_list, dtype=float),
        "heading": mat[:, col("heading")],
        "pitch": mat[:, col("pitch")],
        "roll": mat[:, col("roll")],
        "pressure": mat[:, col("pressure")],
        "temperature": mat[:, col("temperature")],
    }


def load_v_files(dpl: DeploymentData) -> np.ndarray:
    if dpl.data_treatment not in {"raw", "storm"}:
        raise ValueError('load_v_files only supports data_treatment "raw" or "storm"')

    velocities = np.zeros((3, len(dpl.time), len(dpl.z)), dtype=float)
    first_cell = dpl.v_columns["cell_1"] - 1

    for component in range(3):
        suffix = f".v{component + 1}" if dpl.data_treatment == "raw" else f"_p.v{component + 1}"
        mat = _load_numeric_matrix(dpl.folder_name / f"{dpl.dpl_name}{suffix}")
        velocities[component] = mat[:, first_cell : first_cell + len(dpl.z)]

    if dpl.data_treatment == "storm":
        velocities[velocities == -99] = np.nan

    if dpl.vertical_dir == "downward":
        velocities[1] *= -1
        velocities[2] *= -1

    if dpl.coordinate_system == "ENU":
        return velocities
    if dpl.coordinate_system != "XYZ":
        raise NotImplementedError(f'Coordinate system transformation from "{dpl.coordinate_system}" is not implemented')

    v_enu = np.full_like(velocities, np.nan)
    for j in range(dpl.nb_measurements):
        heading = np.deg2rad(dpl.heading[j] - 90.0)
        pitch = np.deg2rad(dpl.pitch[j])
        roll = np.deg2rad(dpl.roll[j])
        heading_matrix = np.array(
            [
                [np.cos(heading), np.sin(heading), 0.0],
                [-np.sin(heading), np.cos(heading), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        tilt_matrix = np.array(
            [
                [np.cos(pitch), -np.sin(pitch) * np.sin(roll), -np.cos(roll) * np.sin(pitch)],
                [0.0, np.cos(roll), -np.sin(roll)],
                [np.sin(pitch), np.sin(roll) * np.cos(pitch), np.cos(pitch) * np.cos(roll)],
            ]
        )
        transform = heading_matrix @ tilt_matrix
        for k in range(dpl.nb_cells):
            column = velocities[:, j, k]
            if np.isnan(column).any():
                continue
            v_enu[:, j, k] = transform @ column
    return v_enu


def load_deployment_files(dpl_param: dict[str, Any]) -> DeploymentData:
    dpl = read_deployment_parameters(dpl_param)
    hdr = load_hdr_file(dpl)
    dpl.nb_measurements = int(hdr["nb_measurements"])
    dpl.nb_cells = int(hdr["nb_cells"])
    dpl.coordinate_system = hdr["coordinate_system"]
    dpl.z = hdr["z"]
    dpl.sen_columns = hdr["sen_columns"]
    dpl.v_columns = hdr["v_columns"]

    sen = load_sen_file(dpl)
    dpl.time = sen["time"]
    dpl.pressure = sen["pressure"]
    dpl.heading = sen["heading"]
    dpl.pitch = sen["pitch"]
    dpl.roll = sen["roll"]
    dpl.temperature = sen["temperature"]
    dpl.v = load_v_files(dpl)
    return dpl


def read_output_parameters(dpl: DeploymentData, out_param: dict[str, Any]) -> OutputData:
    time_start = parse_matlab_date(out_param.get("time_start", dpl.time[0]))
    time_stop = parse_matlab_date(out_param.get("time_stop", dpl.time[-1]))

    start_indices = np.where(dpl.time >= time_start)[0]
    if not len(start_indices):
        raise ValueError("time_start is later than the end of the deployment")
    stop_indices = np.where(dpl.time <= time_stop)[0]
    if not len(stop_indices):
        raise ValueError("time_stop is earlier than the start of the deployment")

    time_start_index = int(start_indices[0])
    time_stop_index = int(stop_indices[-1])
    cell_start = int(out_param.get("cell_start", 1))
    cell_stop = int(out_param.get("cell_stop", dpl.nb_cells))
    if cell_start < 1:
        raise ValueError("cell_start should be >= 1")
    if cell_stop > dpl.nb_cells:
        raise ValueError("cell_stop should be <= deployment.nb_cells")
    if cell_start > cell_stop:
        raise ValueError("cell_start should be <= cell_stop")

    cells = np.arange(cell_start - 1, cell_stop)
    valid_cells = [cell for cell in cells if np.isnan(dpl.v[0, :, cell]).sum() < dpl.v.shape[1]]
    figure_folder_name = Path(out_param.get("figure_folder_name", dpl.folder_name / "figures"))
    figure_folder_name.mkdir(parents=True, exist_ok=True)

    return OutputData(
        time_start=time_start,
        time_stop=time_stop,
        time_start_index=time_start_index,
        time_stop_index=time_stop_index,
        time=dpl.time[time_start_index : time_stop_index + 1],
        cell_start=cell_start,
        cell_stop=cell_stop,
        cells=np.array(valid_cells, dtype=int),
        z=dpl.z[valid_cells],
        save_format=out_param.get("save_format", ""),
        figure_folder_name=figure_folder_name,
    )


def compute_depth_averaged_velocity(dpl: DeploymentData, out: OutputData) -> np.ndarray:
    v = dpl.v[:, out.time_start_index : out.time_stop_index + 1, :][:, :, out.cells]
    return np.nanmean(v, axis=2)


def compute_velocity_norm(dpl: DeploymentData, out: OutputData) -> np.ndarray:
    v = dpl.v[:, out.time_start_index : out.time_stop_index + 1, :][:, :, out.cells]
    return np.sqrt(np.nansum(v**2, axis=0))


def compute_ebb_flood(dpl: DeploymentData, out: OutputData) -> tuple[np.ndarray, np.ndarray]:
    v = dpl.v[:, out.time_start_index : out.time_stop_index + 1, :][:, :, out.cells]
    vh = compute_depth_averaged_velocity(dpl, out)
    uh = vh[:2]
    speed = np.sqrt(np.sum(uh**2, axis=0))

    min_indices = [0]
    for idx in range(len(speed)):
        start = max(0, idx - 24)
        stop = min(len(speed), idx + 25)
        if speed[idx] == np.nanmin(speed[start:stop]):
            min_indices.append(idx)
    min_indices.append(len(speed) - 1)

    nan_cells = np.isnan(v[0]).sum(axis=1)
    mean_nan_cells = float(np.nanmean(nan_cells))
    ebb = np.ones(len(out.time), dtype=float)
    flood = np.ones(len(out.time), dtype=float)
    for start, stop in zip(min_indices[:-1], min_indices[1:]):
        if stop - start <= 1:
            continue
        if np.nanmean(nan_cells[start : stop + 1]) > mean_nan_cells:
            flood[start + 1 : stop] = np.nan
        else:
            ebb[start + 1 : stop] = np.nan
    return ebb, flood


def figure_ts_instrument_orientation(dpl: DeploymentData, out: OutputData) -> plt.Figure:
    time_hours = (out.time - out.time[0]) * 24
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_hours, dpl.heading[out.time_start_index : out.time_stop_index + 1], "k", linewidth=1.5)
    axes[0].set_ylabel("heading (deg)")
    axes[1].plot(time_hours, dpl.pitch[out.time_start_index : out.time_stop_index + 1], "k", linewidth=1.5)
    axes[1].set_ylabel("pitch (deg)")
    axes[2].plot(time_hours, dpl.roll[out.time_start_index : out.time_stop_index + 1], "k", linewidth=1.5)
    axes[2].set_ylabel("roll (deg)")
    axes[2].set_xlabel(f"time since {matlab_datenum_to_datetime(out.time[0])} [hours]")
    add_super_label(fig, "Instrument orientation", "t")
    save_figure(fig, out, "ts_intrument_orientation")
    return fig


def figure_ts_depth_averaged_velocity_components(dpl: DeploymentData, out: OutputData) -> plt.Figure:
    vh = compute_depth_averaged_velocity(dpl, out)
    time_hours = (out.time - out.time[0]) * 24
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ("Eastward component", "Northward component", "Upward component")
    for idx, axis in enumerate(axes):
        axis.plot(time_hours, vh[idx], "k", linewidth=1.5)
        axis.set_title(labels[idx])
        axis.set_ylabel("(m/s)")
    axes[-1].set_xlabel(f"time since {matlab_datenum_to_datetime(out.time[0])} [hours]")
    add_super_label(fig, "Depth-averaged velocity", "t")
    save_figure(fig, out, "ts_depth_averaged_velocity_components")
    return fig


def figure_ts_velocity_norm_profile(
    dpl: DeploymentData,
    out: OutputData,
    days_per_subplot: int,
    min_depth_display: float,
    max_depth_display: float,
    max_velocity_display: float,
) -> plt.Figure:
    velocity_norm = compute_velocity_norm(dpl, out)
    time_datetimes = [matlab_datenum_to_datetime(value) for value in out.time]
    nb_subplot = max(1, int(np.ceil((np.ceil(out.time[-1]) - np.floor(out.time[0])) / days_per_subplot)))
    min_depth = np.nanmin(out.z) if np.isnan(min_depth_display) else min_depth_display
    max_depth = np.nanmax(out.z) if np.isnan(max_depth_display) else max_depth_display
    max_velocity = np.nanmax(velocity_norm) if np.isnan(max_velocity_display) else max_velocity_display

    fig, axes = plt.subplots(nb_subplot, 1, figsize=(14, 4 * nb_subplot), squeeze=False)
    axes = axes.ravel()
    for axis in axes:
        mesh = axis.pcolormesh(time_datetimes, out.z, velocity_norm.T, shading="auto", vmin=0 if max_velocity > 0 else None, vmax=max_velocity if max_velocity > 0 else None)
        axis.set_ylim(min_depth, max_depth)
        axis.set_ylabel("height above sensor (m)")
        fig.colorbar(mesh, ax=axis, label="m/s")
    add_super_label(fig, "Velocity norm [m/s]", "t")
    save_figure(fig, out, "ts_velocity_norm_profile")
    return fig


def figure_depth_averaged_horizontal_velocity(
    dpl: DeploymentData,
    out: OutputData,
    ebb_flood: bool,
    max_velocity_display: float,
    ellipse_bounding_box_display: list[float] | tuple[float, float, float, float],
) -> plt.Figure:
    vh = compute_depth_averaged_velocity(dpl, out)
    uh = vh[:2]
    speed = np.sqrt(np.sum(uh**2, axis=0))
    time_hours = (out.time - out.time[0]) * 24
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    if ebb_flood:
        ebb, flood = compute_ebb_flood(dpl, out)
        axes[0].plot(time_hours, speed * ebb, "b", linewidth=1.5, label="ebb current")
        axes[0].plot(time_hours, speed * flood, "r", linewidth=1.5, label="flood current")
        axes[0].legend()
        axes[1].plot(uh[0] * ebb, uh[1] * ebb, "b.", markersize=3)
        axes[1].plot(uh[0] * flood, uh[1] * flood, "r.", markersize=3)
    else:
        axes[0].plot(time_hours, speed, "k", linewidth=1.5)
        axes[1].plot(uh[0], uh[1], "k.", markersize=3)

    axes[0].set_xlabel(f"time since {matlab_datenum_to_datetime(out.time[0])} [hours]")
    axes[0].set_ylabel("norm (m/s)")
    axes[0].set_ylim(0, np.ceil(np.nanmax(speed) * 10) / 10 if np.isnan(max_velocity_display) else max_velocity_display)
    axes[1].set_xlabel("eastward component (m/s)")
    axes[1].set_ylabel("northward component (m/s)")
    if any(np.isnan(ellipse_bounding_box_display)):
        axes[1].set_aspect("equal", adjustable="datalim")
    else:
        left, right, bottom, top = ellipse_bounding_box_display
        axes[1].set_xlim(left, right)
        axes[1].set_ylim(bottom, top)
        axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True)
    add_super_label(fig, "Depth-averaged horizontal velocity", "t")
    save_figure(fig, out, "depth_averaged_horizontal_velocity")
    return fig
