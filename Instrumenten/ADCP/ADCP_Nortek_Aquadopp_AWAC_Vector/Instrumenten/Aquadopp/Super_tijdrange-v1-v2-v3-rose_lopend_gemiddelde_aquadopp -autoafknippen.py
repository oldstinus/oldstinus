#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aquadopp (AQP) / Aquadopp-profiel verwerking:
- .hdr bevat metadata (starttijd, interval, cell distances, transformation matrix)
- .v1 .v2 .v3 bevatten beam-snelheden per cel (whitespace separated, geen tijdkolom)
- .csv bevat o.a. Speed#i / Dir#i per cel (controle/verwerkte data)

GUI:
- kies .hdr (basisnaam), vink cellen aan
- kies begin/eindmoment (tijdvenster) voor visualisatie + export
- herbereken en teken:
  * beam1/beam2/beam3 uit .dat: time series per geselecteerde cel + gemiddelde (alleen binnen tijdvenster)
  * resultante speed+dir uit .csv over geselecteerde cellen (alleen binnen tijdvenster)
  * X/Y/Z uit .dat (Beam1|X|East, Beam2|Y|North, Beam3|Z|Up): per geselecteerde cel + gemiddelde (alleen binnen tijdvenster)
- export: tab-gescheiden, decimaal ',' en duizendtallen '.'
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tempfile
import shutil
import warnings

try:
    from docx import Document
    from docx.shared import Inches
except ImportError:  # pragma: no cover
    Document = None
    Inches = None


# ----------------------------
# Parsing helpers
# ----------------------------
def _parse_hdr_text(hdr_path: str) -> str:
    with open(hdr_path, "r", errors="ignore") as f:
        return f.read()


def _find_one(text: str, pattern: str, cast=None, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    val = m.group(1).strip()
    return cast(val) if cast else val


def _extract_sen_pressure_column(text: str) -> int:
    block = re.search(r"\[.*?\.sen\](.*?)(?:\r?\n\s*\r?\n|$)", text, flags=re.S | re.I)
    if not block:
        return 14
    block_text = block.group(1)
    line = re.search(r"^\s*(\d+)\s+Pressure\b", block_text, flags=re.M)
    if not line:
        return 14
    try:
        return int(line.group(1))
    except ValueError:
        return 14


def _extract_sen_column_map(text: str) -> dict:
    block = re.search(r"\[.*?\.sen\](.*?)(?:\r?\n\s*\r?\n|---------------------------------------------------------------------)", text, flags=re.S | re.I)
    if not block:
        return {}
    mapping = {}
    for line in block.group(1).splitlines():
        m = re.match(r"^\s*(\d+)\s+(.+?)\s*(?:\(|$)", line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2).strip().lower().replace(" ", "_")
        mapping[name] = idx
    return mapping


def resolve_companion_path(hdr_path: str, ext: str) -> str:
    """
    Zoek bijbehorend bestand bij gekozen .hdr/.hrd:
    1) zelfde basisnaam + extensie
    2) case-variant
    3) expliciete verwijzing in hdr/HRD-tekst
    4) unieke match in dezelfde map op extensie
    """
    base_prefix = os.path.splitext(hdr_path)[0]
    primary = base_prefix + ext
    if os.path.exists(primary):
        return primary

    alt_case = base_prefix + ext.upper()
    if os.path.exists(alt_case):
        return alt_case

    folder = os.path.dirname(hdr_path)
    text = _parse_hdr_text(hdr_path)
    tokens = re.findall(r"[^\s,;\"']+", text)
    ext_l = ext.lower()
    for tok in tokens:
        if not tok.lower().endswith(ext_l):
            continue
        cand = tok.strip().strip("\"'")
        if os.path.isabs(cand) and os.path.exists(cand):
            return cand
        cand_local = os.path.join(folder, os.path.basename(cand))
        if os.path.exists(cand_local):
            return cand_local

    try:
        matches = [
            os.path.join(folder, fn)
            for fn in os.listdir(folder)
            if fn.lower().endswith(ext_l)
        ]
    except OSError:
        matches = []

    if len(matches) == 1:
        return matches[0]
    return primary


def parse_hdr(hdr_path: str) -> dict:
    """
    Leest essentiÃ«le metadata uit .hdr:
    - n_meas, n_cells, profile_interval_sec, t0 (datetime)
    - cell_distances_m (list)
    - transformation_matrix (3x3)
    """
    text = _parse_hdr_text(hdr_path)

    n_meas = _find_one(text, r"Number of measurements\s+(\d+)", int)
    n_cells = _find_one(text, r"Number of cells\s+(\d+)", int)
    dt_sec = _find_one(text, r"Profile interval\s+([0-9.]+)\s*sec", float)

    t0_str = _find_one(
        text,
        r"Time of first measurement\s+(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2})",
        str
    )
    if not t0_str:
        raise ValueError("Kon 'Time of first measurement' niet vinden in .hdr.")
    try:
        t0 = pd.to_datetime(t0_str, dayfirst=True)
    except (ValueError, pd.errors.OutOfBoundsDatetime):
        raise ValueError("Ongeldige starttijd in .hdr: {0}".format(t0_str))

    # cell center distances block
    cell_distances = []
    block = re.search(
        r"Current profile cell center distance from head \(m\)\s*\n[-]+\s*\n(.*?)(?:\n\s*\n|Data file format)",
        text,
        flags=re.S
    )
    if block:
        for line in block.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                try:
                    cell_distances.append(float(parts[1]))
                except Exception:
                    pass

    # Transformation matrix (3x3)
    mat_m = re.search(
        r"Transformation matrix\s+([0-9\.\-\s]+)\n\s*([0-9\.\-\s]+)\n\s*([0-9\.\-\s]+)",
        text
    )
    if not mat_m:
        raise ValueError("Kon 'Transformation matrix' niet vinden in .hdr.")
    rows = []
    for i in range(1, 4):
        row = [float(x) for x in mat_m.group(i).split()]
        if len(row) != 3:
            raise ValueError("Transformation matrix rij is niet 3 waarden breed.")
        rows.append(row)
    T = np.array(rows, dtype=float)

    pressure_column = _extract_sen_pressure_column(text)
    sen_columns = _extract_sen_column_map(text)
    if n_meas is None or n_cells is None or dt_sec is None:
        raise ValueError("Kon n_meas / n_cells / Profile interval niet volledig vinden in .hdr.")

    return {
        "n_meas": n_meas,
        "n_cells": n_cells,
        "dt_sec": dt_sec,
        "t0": t0,
        "cell_distances_m": cell_distances,
        "T": T,
        "sen_pressure_column": pressure_column,
        "sen_columns": sen_columns,
    }


def build_time_index(t0: pd.Timestamp, n_meas: int, dt_sec: float) -> pd.DatetimeIndex:
    freq = pd.to_timedelta(dt_sec, unit="s")
    return pd.date_range(start=t0, periods=n_meas, freq=freq)


def load_v_file(path: str, n_meas: int, n_cells: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != n_cells:
        raise ValueError(f"{os.path.basename(path)}: verwacht {n_cells} kolommen, kreeg {arr.shape[1]}.")
    if arr.shape[0] != n_meas:
        min_n = min(arr.shape[0], n_meas)
        arr = arr[:min_n, :]
    return arr


def load_dat_velocity_cells(dat_path: str, n_meas: int, n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Leest .dat profielblokken:
    - ensemble headerregel (timestamp + status + n_beams + n_cells)
    - gevolgd door n_cells regels met o.a. vel1/vel2/vel3.
    """
    b1 = np.full((n_meas, n_cells), np.nan, dtype=float)
    b2 = np.full((n_meas, n_cells), np.nan, dtype=float)
    b3 = np.full((n_meas, n_cells), np.nan, dtype=float)

    with open(dat_path, "r", errors="ignore") as f:
        lines = f.readlines()

    t = 0
    i = 0
    n_lines = len(lines)
    while i < n_lines and t < n_meas:
        parts = re.split(r"\s+", lines[i].strip())
        is_header = False
        if len(parts) >= 19:
            try:
                _ = list(map(int, parts[:6]))
                _ = int(parts[-2])  # number of beams
                n_cells_line = int(parts[-1])
                is_header = (n_cells_line == n_cells)
            except ValueError:
                is_header = False

        if not is_header:
            i += 1
            continue

        ok_cells = 0
        for c in range(n_cells):
            if i + 1 + c >= n_lines:
                break
            cp = re.split(r"\s+", lines[i + 1 + c].strip())
            if len(cp) < 5:
                continue
            try:
                _ = int(cp[0])  # cellnr
                b1[t, c] = float(cp[2])  # Velocity (Beam1|X|East)
                b2[t, c] = float(cp[3])  # Velocity (Beam2|Y|North)
                b3[t, c] = float(cp[4])  # Velocity (Beam3|Z|Up)
                ok_cells += 1
            except ValueError:
                continue

        if ok_cells > 0:
            t += 1
        i += (1 + n_cells)

    if t == 0:
        raise ValueError(f"Geen geldige profielen gevonden in {os.path.basename(dat_path)}.")
    return b1[:t, :], b2[:t, :], b3[:t, :]


def smooth_over_cells(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1:
        return arr.copy()
    k = np.ones(kernel_size, dtype=float) / float(kernel_size)
    out = np.empty_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        out[i, :] = np.convolve(arr[i, :], k, mode="same")
    return out


def detect_wall_for_beam(
    amp: np.ndarray,
    corr: np.ndarray,
    amp_jump_min: float,
    corr_max: float,
    corr_drop_min: float,
    smooth_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
      - hit_idx: eerste wandcel-index per ensemble (-1 = niet gevonden)
      - wall_mask: bool [n_time, n_cells], True = wandbeinvloed
    """
    a = smooth_over_cells(amp, smooth_k)
    c = smooth_over_cells(corr, smooth_k)
    dA = np.diff(a, axis=1)
    dC = np.diff(c, axis=1)

    n_time, n_cells = a.shape
    hit_idx = np.full(n_time, -1, dtype=int)
    wall_mask = np.zeros((n_time, n_cells), dtype=bool)

    for t in range(n_time):
        cond = (
            (dA[t, :] >= amp_jump_min)
            & ((c[t, 1:] <= corr_max) | (dC[t, :] <= -corr_drop_min))
        )
        idx = np.flatnonzero(cond)
        if idx.size == 0:
            continue
        hit = int(idx[0] + 1)
        hit_idx[t] = hit
        wall_mask[t, hit:] = True

    return hit_idx, wall_mask


def detect_wall_for_beam_amp_only(
    amp: np.ndarray,
    amp_jump_min: float,
    smooth_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    a = smooth_over_cells(amp, smooth_k)
    dA = np.diff(a, axis=1)
    n_time, n_cells = a.shape
    hit_idx = np.full(n_time, -1, dtype=int)
    wall_mask = np.zeros((n_time, n_cells), dtype=bool)

    for t in range(n_time):
        idx = np.flatnonzero(dA[t, :] >= amp_jump_min)
        if idx.size == 0:
            continue
        hit = int(idx[0] + 1)
        hit_idx[t] = hit
        wall_mask[t, hit:] = True
    return hit_idx, wall_mask


def detect_wall_mask_from_amp_corr(
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    amp_jump_min: float,
    corr_max: float,
    corr_drop_min: float,
    smooth_k: int,
    min_beams: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
      - wall_any: bool [n_time, n_cells], wand indien >= min_beams beams dit aangeven
      - beam_hits: int [3, n_time], eerste wandcel-index per beam/ensemble
    """
    hit1, m1 = detect_wall_for_beam(a1, c1, amp_jump_min, corr_max, corr_drop_min, smooth_k)
    hit2, m2 = detect_wall_for_beam(a2, c2, amp_jump_min, corr_max, corr_drop_min, smooth_k)
    hit3, m3 = detect_wall_for_beam(a3, c3, amp_jump_min, corr_max, corr_drop_min, smooth_k)
    votes = m1.astype(int) + m2.astype(int) + m3.astype(int)
    wall_any = votes >= int(min_beams)
    beam_hits = np.vstack([hit1, hit2, hit3])
    return wall_any, beam_hits


def detect_wall_mask_from_amp_only(
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    amp_jump_min: float,
    smooth_k: int,
    min_beams: int,
) -> tuple[np.ndarray, np.ndarray]:
    hit1, m1 = detect_wall_for_beam_amp_only(a1, amp_jump_min, smooth_k)
    hit2, m2 = detect_wall_for_beam_amp_only(a2, amp_jump_min, smooth_k)
    hit3, m3 = detect_wall_for_beam_amp_only(a3, amp_jump_min, smooth_k)
    votes = m1.astype(int) + m2.astype(int) + m3.astype(int)
    wall_any = votes >= int(min_beams)
    beam_hits = np.vstack([hit1, hit2, hit3])
    return wall_any, beam_hits


def find_earliest_jump_distance(
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    cell_distances: list[float],
    amp_jump_min: float,
    smooth_k: int,
) -> float | None:
    if not cell_distances:
        return None
    n_cells = a1.shape[1]
    if len(cell_distances) < n_cells:
        return None

    candidates = []
    for arr in (a1, a2, a3):
        prof = np.nanmean(arr, axis=0)
        # Rolling mean zonder zero-padding randartefacten.
        prof_sm = pd.Series(prof).rolling(
            window=max(1, int(smooth_k)),
            center=True,
            min_periods=1
        ).mean().to_numpy()
        dA = np.diff(prof_sm)
        cond = dA >= amp_jump_min
        # Eerste overgang is vaak near-head artefact: niet gebruiken als "echte" wand-sprong.
        if cond.size > 0:
            cond[0] = False
        # Vereis dat de sprong niet meteen volledig terugvalt (single-bin spike).
        if cond.size > 1:
            next_not_collapse = np.r_[dA[1:] >= (-0.25 * amp_jump_min), True]
            cond &= next_not_collapse
        idx = np.flatnonzero(cond)
        if idx.size:
            cell_idx = int(idx[0] + 1)
            candidates.append(float(cell_distances[cell_idx]))
    if not candidates:
        return None
    return float(min(candidates))


def build_distance_mask(
    n_time: int,
    n_cells: int,
    cell_distances: list[float],
    max_distance_m: float,
) -> np.ndarray:
    mask_1d = np.zeros(n_cells, dtype=bool)
    if cell_distances and len(cell_distances) >= n_cells:
        d = np.asarray(cell_distances[:n_cells], dtype=float)
        # Strikt: alleen cellen met afstand < cutoff blijven geldig.
        mask_1d = d >= max_distance_m
    else:
        # Zonder afstandsinformatie geen harde geometrische cutoff.
        mask_1d[:] = False
    return np.repeat(mask_1d.reshape(1, -1), n_time, axis=0)


def load_csv_processed(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", engine="python")
    if df.columns.size > 0 and (df.columns[-1].strip() == "" or "Unnamed" in df.columns[-1]):
        df = df.iloc[:, :-1]
    df.columns = [c.strip() for c in df.columns]

    if "DateTime" not in df.columns:
        raise ValueError("CSV mist kolom 'DateTime'.")

    dt_txt = df["DateTime"].astype(str).str.strip()
    dt_main = pd.to_datetime(
        dt_txt,
        format="%d/%m/%Y %H:%M:%S",
        dayfirst=True,
        errors="coerce"
    )
    missing = dt_main.isna()
    if missing.any():
        dt_fallback = pd.to_datetime(
            dt_txt[missing],
            format="%d/%m/%Y %H:%M",
            dayfirst=True,
            errors="coerce"
        )
        dt_main.loc[missing] = dt_fallback
    if dt_main.isna().all():
        raise ValueError("CSV DateTime kon niet geparsed worden (verwacht dd/mm/yyyy hh:mm[:ss]).")
    if dt_main.isna().any():
        dropped = int(dt_main.isna().sum())
        warnings.warn(
            f"{os.path.basename(csv_path)}: {dropped} rij(en) met ongeldige DateTime verwijderd.",
            RuntimeWarning,
        )
        df = df.loc[~dt_main.isna()].copy()
        dt_main = dt_main.loc[~dt_main.isna()]
    df["DateTime"] = dt_main

    for c in df.columns:
        if c == "DateTime":
            continue
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )

    return df.sort_values("DateTime").reset_index(drop=True)


def load_sen_pressure(sen_path: str, pressure_column: int) -> pd.Series:
    if pressure_column < 8:
        raise ValueError("Pressure column index moet >= 8 zijn.")
    times = []
    values = []
    with open(sen_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 7:
                continue
            try:
                month, day, year, hour, minute, second = map(int, parts[:6])
            except ValueError:
                continue
            try:
                dt = pd.Timestamp(year, month, day, hour, minute, second)
            except ValueError:
                continue
            data_values = parts[6:]
            idx = pressure_column - 7
            val = np.nan
            if 0 <= idx < len(data_values):
                raw = data_values[idx].strip().replace(",", ".")
                if raw:
                    try:
                        val = float(raw)
                    except ValueError:
                        val = np.nan
            times.append(dt)
            values.append(val)
    if not times:
        return pd.Series([], dtype=float)
    series = pd.Series(values, index=pd.DatetimeIndex(times))
    return series


def load_sen_orientation(sen_path: str, sen_columns: dict) -> pd.DataFrame:
    heading_col = int(sen_columns.get("heading", 11))
    pitch_col = int(sen_columns.get("pitch", 12))
    roll_col = int(sen_columns.get("roll", 13))
    pressure_col = int(sen_columns.get("pressure", 14))
    temperature_col = int(sen_columns.get("temperature", 15))

    records = []
    with open(sen_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 6:
                continue
            try:
                month, day, year, hour, minute, second = map(int, parts[:6])
                dt = pd.Timestamp(year, month, day, hour, minute, second)
            except ValueError:
                continue
            values = parts[6:]

            def pick(col_1based: int):
                idx = col_1based - 7
                if 0 <= idx < len(values):
                    try:
                        return float(values[idx].replace(",", "."))
                    except ValueError:
                        return np.nan
                return np.nan

            records.append(
                {
                    "DateTime": dt,
                    "heading": pick(heading_col),
                    "pitch": pick(pitch_col),
                    "roll": pick(roll_col),
                    "pressure": pick(pressure_col),
                    "temperature": pick(temperature_col),
                }
            )
    if not records:
        return pd.DataFrame(columns=["heading", "pitch", "roll", "pressure", "temperature"])
    return pd.DataFrame.from_records(records).set_index("DateTime")


def speed_dir_to_uv(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(direction_deg)
    u = speed * np.sin(th)
    v = speed * np.cos(th)
    return u, v


def uv_to_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u*u + v*v)
    direction = (np.rad2deg(np.arctan2(u, v)) + 360.0) % 360.0
    return speed, direction


def _find_speed_dir_column_maps(df_csv: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    speed_map: dict[int, str] = {}
    dir_map: dict[int, str] = {}
    for col in df_csv.columns:
        name = str(col).strip()
        m_speed = re.match(r"(?i)^speed\s*#\s*(\d+)\b", name) or re.match(r"(?i)^speed\s*cell\s*(\d+)\b", name)
        if m_speed:
            speed_map[int(m_speed.group(1))] = col
            continue
        m_dir = re.match(r"(?i)^dir\s*#\s*(\d+)\b", name) or re.match(r"(?i)^direction\s*cell\s*(\d+)\b", name)
        if m_dir:
            dir_map[int(m_dir.group(1))] = col
    return speed_map, dir_map


def _compute_csv_avg_speed_dir_to_max_cell(
    df_csv: pd.DataFrame,
    max_cell_one_based: int,
) -> tuple[pd.Series, np.ndarray, np.ndarray, list[int]]:
    t = pd.to_datetime(df_csv["DateTime"], errors="coerce")
    if max_cell_one_based < 1:
        return t, np.full(len(df_csv), np.nan), np.full(len(df_csv), np.nan), []

    speed_map, dir_map = _find_speed_dir_column_maps(df_csv)
    used_cells = [k for k in range(1, max_cell_one_based + 1) if k in speed_map and k in dir_map]
    if not used_cells:
        return t, np.full(len(df_csv), np.nan), np.full(len(df_csv), np.nan), []

    u_list = []
    v_list = []
    for k in used_cells:
        sp = df_csv[speed_map[k]].to_numpy(dtype=float)
        dr = df_csv[dir_map[k]].to_numpy(dtype=float)
        u, v = speed_dir_to_uv(sp, dr)
        u_list.append(u)
        v_list.append(v)

    U = np.nanmean(np.vstack(u_list), axis=0)
    V = np.nanmean(np.vstack(v_list), axis=0)
    spd, direc = uv_to_speed_dir(U, V)
    return t, spd, direc, used_cells


def compute_xyz_from_beams(T: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B = np.stack([b1, b2, b3], axis=-1)
    XYZ = np.einsum("ij,tcj->tci", T, B)
    x = XYZ[..., 0]
    y = XYZ[..., 1]
    z = XYZ[..., 2]
    return x, y, z


def _heading_matrix_deg(heading_deg: float) -> np.ndarray:
    heading = np.deg2rad(heading_deg - 90.0)
    return np.array(
        [
            [np.cos(heading), np.sin(heading), 0.0],
            [-np.sin(heading), np.cos(heading), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _tilt_matrix_deg(pitch_deg: float, roll_deg: float) -> np.ndarray:
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


def recover_xyz_from_enu(enu_x: np.ndarray, enu_y: np.ndarray, enu_z: np.ndarray, heading: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_time, n_cells = enu_x.shape
    x = np.full((n_time, n_cells), np.nan, dtype=float)
    y = np.full((n_time, n_cells), np.nan, dtype=float)
    z = np.full((n_time, n_cells), np.nan, dtype=float)
    for t in range(n_time):
        rot = _heading_matrix_deg(float(heading[t])) @ _tilt_matrix_deg(float(pitch[t]), float(roll[t]))
        inv_rot = np.linalg.inv(rot)
        enu_stack = np.stack([enu_x[t, :], enu_y[t, :], enu_z[t, :]], axis=0)
        xyz_stack = inv_rot @ enu_stack
        x[t, :] = xyz_stack[0, :]
        y[t, :] = xyz_stack[1, :]
        z[t, :] = xyz_stack[2, :]
    return x, y, z


def recover_beams_from_xyz(T_beam_to_xyz: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_to_beam = np.linalg.inv(T_beam_to_xyz)
    xyz_stack = np.stack([x, y, z], axis=1)
    beam_stack = np.einsum("ij,tjk->tik", xyz_to_beam, xyz_stack)
    return beam_stack[:, 0, :], beam_stack[:, 1, :], beam_stack[:, 2, :]


# ----------------------------
# Plot helpers
# ----------------------------
def _new_or_clear_figure(fig_key: str, registry: dict, title: str):
    old_fig = registry.get(fig_key)
    if old_fig is not None and plt.fignum_exists(old_fig.number):
        try:
            plt.close(old_fig)
        except Exception:
            pass

    fig = plt.figure()
    registry[fig_key] = fig

    def _drop_closed_figure(event, key=fig_key, reg=registry):
        if reg.get(key) is event.canvas.figure:
            reg.pop(key, None)

    fig.canvas.mpl_connect("close_event", _drop_closed_figure)
    fig.suptitle(title)
    return fig


def _format_time_axis(ax):
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=90)


def _apply_tight_layout(fig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()


def _stack_windows_vertical(fig_top, fig_bottom, pad_px: int = 40) -> None:
    """
    Probeer figuur-vensters onder elkaar te plaatsen (snelheid boven richting).
    Werkt best-effort voor Tk / Qt backends; faalt stil als niet ondersteund.
    """
    try:
        mgr_top = getattr(getattr(fig_top, "canvas", None), "manager", None)
        mgr_bottom = getattr(getattr(fig_bottom, "canvas", None), "manager", None)
        win_top = getattr(mgr_top, "window", None)
        win_bottom = getattr(mgr_bottom, "window", None)
        if win_top is None or win_bottom is None:
            return

        # TkAgg
        if hasattr(win_top, "winfo_x") and hasattr(win_top, "winfo_y") and hasattr(win_top, "winfo_width"):
            win_top.update_idletasks()
            x = int(win_top.winfo_x())
            y = int(win_top.winfo_y())
            w = int(win_top.winfo_width())
            h = int(win_top.winfo_height())
            if w <= 1 or h <= 1:
                return
            win_bottom.update_idletasks()
            bw = int(win_bottom.winfo_width())
            bh = int(win_bottom.winfo_height())
            if bw <= 1 or bh <= 1:
                bw, bh = w, h
            win_bottom.wm_geometry(f"{bw}x{bh}+{x}+{y + h + int(pad_px)}")
            return

        # Qt (Qt5/Qt6)
        if hasattr(win_top, "geometry") and hasattr(win_top, "setGeometry"):
            g = win_top.geometry()
            x, y, w, h = int(g.x()), int(g.y()), int(g.width()), int(g.height())
            if w <= 1 or h <= 1:
                return
            gb = win_bottom.geometry()
            bw, bh = int(gb.width()), int(gb.height())
            if bw <= 1 or bh <= 1:
                bw, bh = w, h
            win_bottom.setGeometry(x, y + h + int(pad_px), bw, bh)
            return
    except Exception:
        return


def _plot_pressure_panel(ax, time_index, pressure_arr):
    """
    Tekent de druk (pressure) onder een tijdreeks.
    """
    ax.set_ylabel("Pressure (dbar)")
    ax.grid(True, alpha=0.2)
    if pressure_arr is None or len(pressure_arr) == 0 or (isinstance(pressure_arr, np.ndarray) and np.all(np.isnan(pressure_arr))):
        ax.text(0.5, 0.5, "Geen drukgegevens beschikbaar.", ha="center", va="center", color="gray",
                transform=ax.transAxes)
        ax.set_yticks([])
        _format_time_axis(ax)
        return

    y = pressure_arr
    x = time_index
    if isinstance(pressure_arr, pd.Series):
        y = pressure_arr.to_numpy()
        x = pressure_arr.index
    ax.plot(x, y, color="tab:cyan", linewidth=1.25)
    _format_time_axis(ax)
    ax.set_xlabel("Tijd")


def plot_beam(
    time_index,
    beam_arr,
    selected_cells,
    cell_distances,
    beam_name,
    fig_registry,
    pressure_arr=None,
    show_cells: bool = True,
    show_mean: bool = True,
):
    fig = _new_or_clear_figure(f"beam_{beam_name}", fig_registry, f"{beam_name} - snelheden per cel + gemiddelde")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_pressure = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.tick_params(labelbottom=False)

    if not selected_cells:
        ax.text(0.5, 0.5, "Geen cellen geselecteerd.", ha="center", va="center", transform=ax.transAxes)
        _apply_tight_layout(fig)
        fig.show()
        return

    plotted_any = False
    if show_cells:
        for ci in selected_cells:
            label = f"Cel {ci+1}"
            if cell_distances and ci < len(cell_distances):
                label += f" ({cell_distances[ci]:.2f} m)"
            y = beam_arr[:, ci]
            if np.isfinite(y).any():
                ax.plot(time_index, y, label=label, linewidth=0.9)
                plotted_any = True

    if show_mean:
        avg = np.nanmean(beam_arr[:, selected_cells], axis=1)
        if np.isfinite(avg).any():
            ax.plot(time_index, avg, label="Gemiddelde (geselecteerde cellen)", linewidth=2.0)
            plotted_any = True

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "Geen zichtbare lijnen (alles gemaskeerd/NaN in dit venster).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
        )

    ax.set_ylabel("Snelheid langs beam (m/s)")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)
    _plot_pressure_panel(ax_pressure, time_index, pressure_arr)
    _apply_tight_layout(fig)
    fig.show()


def plot_resultant_from_csv(df_csv, selected_cells, fig_registry, pressure_arr=None):
    """
    Berekent resultante speed+dir vanuit CSV over cel 1 t/m hoogste geselecteerde cel.
    """
    # Opruimen van legacy-figuurkey indien die nog bestaat.
    if "resultant_csv" in fig_registry and plt.fignum_exists(fig_registry["resultant_csv"].number):
        try:
            plt.close(fig_registry["resultant_csv"])
        except Exception:
            pass
    fig_registry.pop("resultant_csv", None)

    fig_speed = _new_or_clear_figure(
        "resultant_speed_csv",
        fig_registry,
        "Resultante snelheid (uit CSV, cel 1 t/m max selectie)",
    )
    gs_spd = fig_speed.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax_spd = fig_speed.add_subplot(gs_spd[0, 0])
    ax_pressure_spd = fig_speed.add_subplot(gs_spd[1, 0], sharex=ax_spd)
    ax_spd.tick_params(labelbottom=False)

    fig_dir = _new_or_clear_figure(
        "resultant_dir_csv",
        fig_registry,
        "Resultante richting (uit CSV, cel 1 t/m max selectie)",
    )
    gs_dir = fig_dir.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax_dir = fig_dir.add_subplot(gs_dir[0, 0])
    ax_pressure_dir = fig_dir.add_subplot(gs_dir[1, 0], sharex=ax_dir)
    ax_dir.tick_params(labelbottom=False)

    if df_csv is None or df_csv.empty:
        ax_spd.text(0.5, 0.5, "CSV niet ingelezen.", ha="center", va="center", transform=ax_spd.transAxes)
        ax_dir.text(0.5, 0.5, "CSV niet ingelezen.", ha="center", va="center", transform=ax_dir.transAxes)
        _plot_pressure_panel(ax_pressure_spd, pd.DatetimeIndex([]), None)
        _plot_pressure_panel(ax_pressure_dir, pd.DatetimeIndex([]), None)
        _apply_tight_layout(fig_speed)
        _apply_tight_layout(fig_dir)
        fig_speed.show()
        fig_dir.show()
        _stack_windows_vertical(fig_speed, fig_dir)
        return

    if not selected_cells:
        ax_spd.text(0.5, 0.5, "Geen cellen geselecteerd.", ha="center", va="center", transform=ax_spd.transAxes)
        ax_dir.text(0.5, 0.5, "Geen cellen geselecteerd.", ha="center", va="center", transform=ax_dir.transAxes)
        _plot_pressure_panel(ax_pressure_spd, pd.DatetimeIndex([]), None)
        _plot_pressure_panel(ax_pressure_dir, pd.DatetimeIndex([]), None)
        _apply_tight_layout(fig_speed)
        _apply_tight_layout(fig_dir)
        fig_speed.show()
        fig_dir.show()
        _stack_windows_vertical(fig_speed, fig_dir)
        return

    max_cell = max(selected_cells) + 1
    t, spd, direc, used_cells = _compute_csv_avg_speed_dir_to_max_cell(df_csv, max_cell)
    if not used_cells:
        ax_spd.text(
            0.5,
            0.5,
            "Geen Speed/Direction-kolommen gevonden voor celbereik 1..max.",
            ha="center",
            va="center",
            transform=ax_spd.transAxes,
        )
        ax_dir.text(
            0.5,
            0.5,
            "Geen Speed/Direction-kolommen gevonden voor celbereik 1..max.",
            ha="center",
            va="center",
            transform=ax_dir.transAxes,
        )
        _plot_pressure_panel(ax_pressure_spd, pd.DatetimeIndex([]), None)
        _plot_pressure_panel(ax_pressure_dir, pd.DatetimeIndex([]), None)
        _apply_tight_layout(fig_speed)
        _apply_tight_layout(fig_dir)
        fig_speed.show()
        fig_dir.show()
        _stack_windows_vertical(fig_speed, fig_dir)
        return

    ax_spd.plot(t, spd, linewidth=1.5)
    ax_spd.set_ylabel("Resultante snelheid (m/s)")
    ax_spd.grid(True, alpha=0.3)

    ax_dir.plot(t, direc, linewidth=1.0, color="tab:orange")
    ax_dir.set_ylabel("Richting (deg)")
    ax_dir.grid(True, alpha=0.3)

    _plot_pressure_panel(ax_pressure_spd, t, pressure_arr)
    _plot_pressure_panel(ax_pressure_dir, t, pressure_arr)
    _apply_tight_layout(fig_speed)
    _apply_tight_layout(fig_dir)
    fig_speed.show()
    fig_dir.show()
    _stack_windows_vertical(fig_speed, fig_dir)


def plot_hrp_from_csv(df_csv, fig_registry, pressure_arr=None):
    """
    Plot Heading/Roll/Pitch uit CSV in hetzelfde tijdvenster.
    """
    fig = _new_or_clear_figure("hrp_csv", fig_registry, "Heading / Roll / Pitch (uit CSV)")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_pressure = fig.add_subplot(gs[1, 0], sharex=ax_h)
    ax_h.tick_params(labelbottom=False)

    if df_csv is None or df_csv.empty:
        ax_h.text(0.5, 0.5, "CSV niet ingelezen.", ha="center", va="center", transform=ax_h.transAxes)
        _apply_tight_layout(fig)
        _plot_pressure_panel(ax_pressure, pd.DatetimeIndex([]), None)
        fig.show()
        return

    col_map = {str(c).strip().lower(): c for c in df_csv.columns}
    heading_col = col_map.get("heading")
    roll_col = col_map.get("roll")
    pitch_col = col_map.get("pitch")

    if heading_col is None and roll_col is None and pitch_col is None:
        ax_h.text(0.5, 0.5, "Geen Heading/Pitch/Roll kolommen in CSV.", ha="center", va="center", transform=ax_h.transAxes)
        _apply_tight_layout(fig)
        _plot_pressure_panel(ax_pressure, pd.DatetimeIndex([]), None)
        fig.show()
        return

    t = pd.to_datetime(df_csv["DateTime"], errors="coerce")
    ax_rp = ax_h.twinx()
    handles = []
    labels = []

    if heading_col is not None:
        ln = ax_h.plot(t, df_csv[heading_col].to_numpy(dtype=float), color="tab:blue", linewidth=1.2, label="Heading")[0]
        handles.append(ln)
        labels.append("Heading")
        ax_h.set_ylabel("Heading (deg)")
    else:
        ax_h.set_ylabel("Heading (deg)")

    if roll_col is not None:
        ln = ax_rp.plot(t, df_csv[roll_col].to_numpy(dtype=float), color="tab:orange", linewidth=1.1, label="Roll")[0]
        handles.append(ln)
        labels.append("Roll")
    if pitch_col is not None:
        ln = ax_rp.plot(t, df_csv[pitch_col].to_numpy(dtype=float), color="tab:green", linewidth=1.1, label="Pitch")[0]
        handles.append(ln)
        labels.append("Pitch")
    ax_rp.set_ylabel("Roll/Pitch (deg)")

    ax_h.grid(True, alpha=0.3)
    if handles:
        ax_h.legend(handles, labels, loc="upper right", fontsize=8)

    _plot_pressure_panel(ax_pressure, t, pressure_arr)
    _apply_tight_layout(fig)
    fig.show()


def plot_component(
    time_index,
    comp_arr,
    selected_cells,
    cell_distances,
    title,
    ylabel,
    fig_key,
    fig_registry,
    pressure_arr=None,
    show_cells: bool = True,
    show_mean: bool = True,
):
    fig = _new_or_clear_figure(fig_key, fig_registry, title)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_pressure = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.tick_params(labelbottom=False)

    if not selected_cells:
        ax.text(0.5, 0.5, "Geen cellen geselecteerd.", ha="center", va="center", transform=ax.transAxes)
        _apply_tight_layout(fig)
        fig.show()
        return

    plotted_any = False
    if show_cells:
        for ci in selected_cells:
            label = f"Cel {ci+1}"
            if cell_distances and ci < len(cell_distances):
                label += f" ({cell_distances[ci]:.2f} m)"
            y = comp_arr[:, ci]
            if np.isfinite(y).any():
                ax.plot(time_index, y, label=label, linewidth=0.9)
                plotted_any = True

    if show_mean:
        avg = np.nanmean(comp_arr[:, selected_cells], axis=1)
        if np.isfinite(avg).any():
            ax.plot(time_index, avg, label="Gemiddelde (geselecteerde cellen)", linewidth=2.0)
            plotted_any = True

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "Geen zichtbare lijnen (alles gemaskeerd/NaN in dit venster).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
        )

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)
    _plot_pressure_panel(ax_pressure, time_index, pressure_arr)
    _apply_tight_layout(fig)
    fig.show()


def plot_amplitude_profile(
    a1_window: np.ndarray,
    a2_window: np.ndarray,
    a3_window: np.ndarray,
    cell_distances: list[float],
    fig_registry: dict,
    max_distance_m: float | None = None,
    earliest_jump_m: float | None = None,
):
    fig = _new_or_clear_figure(
        "amplitude_profile",
        fig_registry,
        "Gemiddelde amplitude per hoogte (tijdvenster)",
    )
    ax = fig.add_subplot(111)

    n_cells = a1_window.shape[1]
    if cell_distances and len(cell_distances) >= n_cells:
        x = np.asarray(cell_distances[:n_cells], dtype=float)
        xlabel = "Afstand vanaf sensor (m)"
    else:
        x = np.arange(1, n_cells + 1, dtype=float)
        xlabel = "Celnummer"

    a1_mean = np.nanmean(a1_window, axis=0)
    a2_mean = np.nanmean(a2_window, axis=0)
    a3_mean = np.nanmean(a3_window, axis=0)

    ax.plot(x, a1_mean, marker="o", linewidth=1.5, label="Beam 1 amplitude")
    ax.plot(x, a2_mean, marker="o", linewidth=1.5, label="Beam 2 amplitude")
    ax.plot(x, a3_mean, marker="o", linewidth=1.5, label="Beam 3 amplitude")

    if earliest_jump_m is not None:
        ax.axvline(
            earliest_jump_m,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"Vroegste sprong ({earliest_jump_m:.2f} m)",
        )
    if max_distance_m is not None:
        ax.axvline(
            max_distance_m,
            color="red",
            linewidth=1.7,
            linestyle="-",
            label=f"Max afstand ({max_distance_m:.2f} m)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude (counts)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _apply_tight_layout(fig)
    fig.show()
def apply_running_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Maakt een lopend gemiddelde over tijd (axis=0) voor alle cellen.
    """
    if window <= 1:
        return arr
    arr_np = np.asarray(arr)
    if arr_np.ndim == 1:
        sr = pd.Series(arr_np)
        return sr.rolling(window=window, min_periods=1, center=True).mean().to_numpy()
    df = pd.DataFrame(arr_np)
    return df.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def plot_current_rose(u: np.ndarray,
                      v: np.ndarray,
                      title: str,
                      fig_key: str,
                      fig_registry: dict,
                      n_dir_bins: int = 36):
    """
    Stroomroos (current rose) op basis van u,v componenten.
    Richting = waar de stroming NAARTOE gaat (oceanografisch).
    """

    speed = np.sqrt(u**2 + v**2)
    direction = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0

    bins = np.linspace(0, 360, n_dir_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = 2 * np.pi / n_dir_bins

    mean_speed = np.zeros(n_dir_bins)
    for i in range(n_dir_bins):
        mask = (direction >= bins[i]) & (direction < bins[i + 1])
        if np.any(mask):
            mean_speed[i] = np.nanmean(speed[mask])

    theta = np.deg2rad(bin_centers)

    fig = _new_or_clear_figure(fig_key, fig_registry, title)
    ax = fig.add_subplot(111, polar=True)
    ax.bar(theta, mean_speed, width=bin_width, bottom=0.0)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    _apply_tight_layout(fig)
    fig.show()


# ----------------------------
# Export helper (EU formatting)
# ----------------------------
def export_tsv_eu(df: pd.DataFrame, out_path: str, decimals: int = 6):
    """
    Exporteer tab-separated met:
      - decimaal = ','
      - duizendtallen = '.'
    """
    df_out = df.copy()

    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            df_out[col] = df_out[col].map(
                lambda x: (
                    f"{x:,.{decimals}f}"
                    .replace(",", "TMP")
                    .replace(".", ",")
                    .replace("TMP", ".")
                ) if pd.notnull(x) else ""
            )

    df_out.to_csv(out_path, sep="\t", index=False)


def filter_time_window(time_index: pd.DatetimeIndex, start_dt: pd.Timestamp | None, end_dt: pd.Timestamp | None):
    """
    Returns boolean mask for time_index within [start_dt, end_dt] (inclusive).
    """
    mask = np.ones(len(time_index), dtype=bool)
    if start_dt is not None:
        mask &= (time_index >= start_dt)
    if end_dt is not None:
        mask &= (time_index <= end_dt)
    return mask


# ----------------------------
# GUI App
# ----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aquadopp - beams (.dat) + CSV controle - GUI")
        self.geometry("1040x700")

        self.meta = None
        self.base_hdr = None
        self.base_prefix = None

        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.df_csv = None
        self.time_index = None
        self.pressure_series = None
        self.orientation_df = None
        self.x_backcalc = None
        self.y_backcalc = None
        self.z_backcalc = None
        self.beam1_backcalc = None
        self.beam2_backcalc = None
        self.beam3_backcalc = None

        self.fig_registry = {}

        # UI state
        self.cell_vars = []
        self.export_var = tk.BooleanVar(value=False)
        self.auto_backcalc_on_load_var = tk.BooleanVar(value=False)
        self.running_mean_var = tk.StringVar(value="1")
        self.rose_mode_var = tk.StringVar(value="XY")
        self.show_amplitude_profile_var = tk.BooleanVar(value=True)
        self.show_hrp_var = tk.BooleanVar(value=True)
        self.show_cell_lines_var = tk.BooleanVar(value=True)
        self.show_mean_line_var = tk.BooleanVar(value=True)
        self.wall_filter_var = tk.BooleanVar(value=True)
        self.wall_amp_jump_var = tk.StringVar(value="2.5")
        self.wall_corr_max_var = tk.StringVar(value="60")
        self.wall_corr_drop_var = tk.StringVar(value="8")
        self.wall_smooth_k_var = tk.StringVar(value="3")
        self.wall_min_beams_var = tk.StringVar(value="2")
        self.wall_hard_max_dist_var = tk.StringVar(value="0.40")

        self.start_var = tk.StringVar(value="")
        self.end_var = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        btn_pick = ttk.Button(top, text="Kies .hdr bestand...", command=self.pick_hdr)
        btn_pick.pack(side=tk.LEFT)

        chk_auto_backcalc = ttk.Checkbutton(
            top,
            text="Bij laden ENU -> XYZ terugrekenen en tonen",
            variable=self.auto_backcalc_on_load_var,
        )
        chk_auto_backcalc.pack(side=tk.LEFT, padx=(10, 0))

        self.lbl_file = ttk.Label(top, text="(geen bestand gekozen)")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        mid = ttk.Frame(self, padding=10)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # left: cell selection (scrollable)
        left = ttk.LabelFrame(mid, text="Cellen (aanvinken)", padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas = tk.Canvas(left, width=320, height=500)
        self.scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # right: actions/info
        right = ttk.Frame(mid, padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Time window frame
        tw = ttk.LabelFrame(right, text="Tijdvenster (zoom / export)", padding=8)
        tw.pack(fill=tk.X)

        ttk.Label(tw, text="Begin (dd/mm/yyyy hh:mm:ss) leeg = start").grid(row=0, column=0, sticky="w")
        e1 = ttk.Entry(tw, textvariable=self.start_var, width=28)
        e1.grid(row=0, column=1, padx=6, pady=2, sticky="w")

        ttk.Label(tw, text="Einde (dd/mm/yyyy hh:mm:ss) leeg = eind").grid(row=1, column=0, sticky="w")
        e2 = ttk.Entry(tw, textvariable=self.end_var, width=28)
        e2.grid(row=1, column=1, padx=6, pady=2, sticky="w")

        btn_set_full = ttk.Button(tw, text="Vul volledig bereik in", command=self.fill_full_range)
        btn_set_full.grid(row=0, column=2, rowspan=2, padx=10, sticky="ns")

        tw.columnconfigure(3, weight=1)

        self.txt_info = tk.Text(right, height=12, wrap="word")
        self.txt_info.pack(fill=tk.X, pady=(10, 0))
        self.txt_info.insert("end", "Kies een .hdr bestand om te starten.\n")
        self.txt_info.configure(state="disabled")

        action = ttk.Frame(right)
        action.pack(fill=tk.X, pady=10)

        chk_export = ttk.Checkbutton(action, text="Export bij herbereken", variable=self.export_var)
        chk_export.pack(side=tk.LEFT)

        btn_run = ttk.Button(action, text="Herbereken en teken", command=self.recompute_and_plot)
        btn_run.pack(side=tk.LEFT, padx=10)

        btn_save_npz = ttk.Button(action, text="Schrijf NPZ weg (beams+xyz)", command=self.save_npz)
        btn_save_npz.pack(side=tk.LEFT)

        btn_backcalc_xyz = ttk.Button(action, text="ENU -> XYZ terugrekenen", command=self.export_enu_to_xyz_backcalculation)
        btn_backcalc_xyz.pack(side=tk.LEFT, padx=(6, 0))

        btn_export_word = ttk.Button(action, text="Exporteer figuren -> Word", command=self.export_figures_to_word)
        btn_export_word.pack(side=tk.LEFT, padx=(6, 0))

        opts = ttk.Frame(right)
        opts.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(opts, text="Stroomroos uit:").pack(side=tk.LEFT)
        cmb = ttk.Combobox(
            opts,
            textvariable=self.rose_mode_var,
            state="readonly",
            values=["XY", "Alleen X"],
            width=10
        )
        cmb.pack(side=tk.LEFT, padx=6)

        ttk.Label(opts, text="Lopend gemiddelde (N):").pack(side=tk.LEFT, padx=(16, 0))
        ent_rm = ttk.Entry(opts, textvariable=self.running_mean_var, width=6)
        ent_rm.pack(side=tk.LEFT, padx=(4, 0))

        chk_amp_profile = ttk.Checkbutton(
            opts,
            text="Amplitude-profielvenster",
            variable=self.show_amplitude_profile_var
        )
        chk_amp_profile.pack(side=tk.LEFT, padx=(16, 0))

        chk_hrp = ttk.Checkbutton(
            opts,
            text="HRP (uit .csv)",
            variable=self.show_hrp_var
        )
        chk_hrp.pack(side=tk.LEFT, padx=(12, 0))

        chk_cells = ttk.Checkbutton(
            opts,
            text="Toon cellen",
            variable=self.show_cell_lines_var,
        )
        chk_cells.pack(side=tk.LEFT, padx=(16, 0))

        chk_mean = ttk.Checkbutton(
            opts,
            text="Toon gemiddelde",
            variable=self.show_mean_line_var,
        )
        chk_mean.pack(side=tk.LEFT, padx=(12, 0))

        wall = ttk.LabelFrame(right, text="Wanddetectie (amplitude + correlatie)", padding=8)
        wall.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(
            wall,
            text="Activeer wandfilter (zet wandcellen op NaN)",
            variable=self.wall_filter_var
        ).grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 4))
        ttk.Label(wall, text="dA min").grid(row=1, column=0, sticky="w")
        ttk.Entry(wall, textvariable=self.wall_amp_jump_var, width=7).grid(row=1, column=1, padx=(4, 10), sticky="w")
        ttk.Label(wall, text="Corr max").grid(row=1, column=2, sticky="w")
        ttk.Entry(wall, textvariable=self.wall_corr_max_var, width=7).grid(row=1, column=3, padx=(4, 10), sticky="w")
        ttk.Label(wall, text="dCorr min").grid(row=1, column=4, sticky="w")
        ttk.Entry(wall, textvariable=self.wall_corr_drop_var, width=7).grid(row=1, column=5, padx=(4, 10), sticky="w")
        ttk.Label(wall, text="Smooth k").grid(row=1, column=6, sticky="w")
        ttk.Entry(wall, textvariable=self.wall_smooth_k_var, width=5).grid(row=1, column=7, padx=(4, 10), sticky="w")
        ttk.Label(wall, text="Min beams").grid(row=1, column=8, sticky="w")
        cmb_wall_beams = ttk.Combobox(
            wall,
            textvariable=self.wall_min_beams_var,
            state="readonly",
            values=["1", "2", "3"],
            width=4
        )
        cmb_wall_beams.grid(row=1, column=9, padx=(4, 0), sticky="w")
        ttk.Label(wall, text="Hard max m").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(wall, textvariable=self.wall_hard_max_dist_var, width=7).grid(
            row=2, column=1, padx=(4, 10), sticky="w", pady=(6, 0)
        )
        ttk.Label(
            wall,
            text="Effectieve cutoff = min(Hard max m, vroegste amplitudesprong)",
        ).grid(row=2, column=2, columnspan=8, sticky="w", pady=(6, 0))

        tips = ttk.Label(right, text=(
            "Opmerking:\n"
            "* Tijdas wordt opgebouwd uit starttijd + Profile interval.\n"
            "* Uit .sen worden pressure en eventueel heading/pitch/roll gelezen.\n"
            "* Vink bovenaan ENU -> XYZ aan om bij het laden teruggerekende instrumentassen te berekenen.\n"
            "* Zonder dat vinkje komen X/Y/Z rechtstreeks uit .dat Velocity profile (kolommen 3/4/5 per cel).\n"
            "* Resultante speed/dir uit CSV over cel 1 t/m hoogste geselecteerde cel.\n"
            "* HRP-plot gebruikt Heading/Pitch/Roll uit CSV.\n"
            "* Tijdvenster geldt voor plots en export.\n"
            "* Kies stroomroos-bron en eventueel lopend gemiddelde in dezelfde actie-regel."
        ))
        tips.pack(fill=tk.X, pady=10)

    def log(self, msg: str):
        self.txt_info.configure(state="normal")
        self.txt_info.insert("end", msg.rstrip() + "\n")
        self.txt_info.see("end")
        self.txt_info.configure(state="disabled")

    def parse_time_entry(self, s: str) -> pd.Timestamp | None:
        s = (s or "").strip()
        if not s:
            return None
        try:
            return pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", dayfirst=True)
        except Exception:
            raise ValueError(f"Ongeldig tijdformaat: '{s}'. Verwacht dd/mm/yyyy hh:mm:ss")

    def _align_pressure_series(self, pressure_series: pd.Series, time_index: pd.DatetimeIndex, dt_sec: float) -> pd.Series | None:
        if pressure_series is None or pressure_series.empty:
            return None
        df_base = pd.DataFrame({"DateTime": time_index})
        df_pressure = pressure_series.dropna().sort_index().reset_index()
        df_pressure.columns = ["DateTime", "pressure"]
        tolerance = pd.Timedelta(seconds=max(dt_sec, 1.0))
        merged = pd.merge_asof(df_base, df_pressure, on="DateTime", direction="nearest", tolerance=tolerance)
        merged = merged.sort_values("DateTime")
        return pd.Series(merged["pressure"].to_numpy(), index=time_index)

    def _align_orientation_df(self, orientation_df: pd.DataFrame, time_index: pd.DatetimeIndex, dt_sec: float) -> pd.DataFrame | None:
        if orientation_df is None or orientation_df.empty:
            return None
        df_base = pd.DataFrame({"DateTime": time_index})
        df_ori = orientation_df.sort_index().reset_index()
        df_ori.columns = ["DateTime"] + [c for c in orientation_df.columns]
        tolerance = pd.Timedelta(seconds=max(dt_sec, 1.0))
        merged = pd.merge_asof(df_base, df_ori, on="DateTime", direction="nearest", tolerance=tolerance)
        merged = merged.sort_values("DateTime").set_index("DateTime")
        return merged

    def _log_time_axis_consistency(self, df_csv: pd.DataFrame, time_index: pd.DatetimeIndex, dt_sec: float) -> None:
        if df_csv is None or df_csv.empty or "DateTime" not in df_csv.columns:
            self.log("Tijdcontrole CSV vs HDR-index: CSV ontbreekt of is leeg.")
            return
        if time_index is None or len(time_index) == 0:
            self.log("Tijdcontrole CSV vs HDR-index: HDR-tijdindex ontbreekt.")
            return

        t_csv = pd.to_datetime(df_csv["DateTime"], errors="coerce").dropna().sort_values().reset_index(drop=True)
        if t_csv.empty:
            self.log("Tijdcontrole CSV vs HDR-index: geen geldige CSV DateTime waarden.")
            return

        start_delta = float((t_csv.iloc[0] - time_index[0]).total_seconds())
        end_delta = float((t_csv.iloc[-1] - time_index[-1]).total_seconds())
        self.log(
            "Tijdcontrole CSV vs HDR-index: "
            f"csv_rows={len(t_csv)}, beam_rows={len(time_index)}, "
            f"start_delta={start_delta:.1f}s, end_delta={end_delta:.1f}s"
        )

        if len(t_csv) < 2:
            return
        dt_steps = t_csv.diff().dt.total_seconds().dropna()
        if dt_steps.empty:
            return

        median_dt = float(dt_steps.median())
        p95_dt = float(dt_steps.quantile(0.95))
        max_dt = float(dt_steps.max())
        tol = 1e-9
        n_gt = int((dt_steps > (dt_sec + tol)).sum())
        n_lt = int((dt_steps < (dt_sec - tol)).sum())
        n_neq = n_gt + n_lt
        self.log(
            "CSV tijdstapstatistiek: "
            f"verwacht={dt_sec:g}s, median={median_dt:.3f}s, p95={p95_dt:.3f}s, max={max_dt:.3f}s, "
            f"afwijkend={n_neq}/{len(dt_steps)}"
        )
        if (
            abs(start_delta) > max(dt_sec, 1.0)
            or abs(end_delta) > max(dt_sec, 1.0)
            or n_neq > 0
        ):
            self.log(
                "WAARSCHUWING: CSV DateTime wijkt af van HDR-index; "
                "dit kan zichtbare tijd-shift/uitrekking geven t.o.v. beamplots."
            )


    def fill_full_range(self):
        if self.time_index is None or len(self.time_index) == 0:
            return
        self.start_var.set(self.time_index[0].strftime("%d/%m/%Y %H:%M:%S"))
        self.end_var.set(self.time_index[-1].strftime("%d/%m/%Y %H:%M:%S"))

    def _get_running_mean_window(self) -> int:
        value = (self.running_mean_var.get() or "").strip()
        if not value:
            return 1
        try:
            window = int(value)
        except ValueError:
            raise ValueError("Lopend gemiddelde moet een positief geheel getal zijn.")
        if window < 1:
            raise ValueError("Lopend gemiddelde moet minimaal 1 zijn.")
        return window

    def _get_wall_filter_settings(self) -> tuple[float, float, float, int, int, float]:
        try:
            amp_jump_min = float((self.wall_amp_jump_var.get() or "").strip())
            corr_max = float((self.wall_corr_max_var.get() or "").strip())
            corr_drop_min = float((self.wall_corr_drop_var.get() or "").strip())
            smooth_k = int((self.wall_smooth_k_var.get() or "").strip())
            min_beams = int((self.wall_min_beams_var.get() or "").strip())
            hard_max_dist_m = float((self.wall_hard_max_dist_var.get() or "").strip())
        except ValueError:
            raise ValueError("Wanddetectie-instellingen zijn ongeldig (controleer getalvelden).")
        if smooth_k < 1:
            raise ValueError("Smooth k moet minimaal 1 zijn.")
        if min_beams not in (1, 2, 3):
            raise ValueError("Min beams moet 1, 2 of 3 zijn.")
        if hard_max_dist_m <= 0:
            raise ValueError("Hard max m moet groter zijn dan 0.")
        return amp_jump_min, corr_max, corr_drop_min, smooth_k, min_beams, hard_max_dist_m

    def _clear_backcalculated_data(self) -> None:
        self.x_backcalc = None
        self.y_backcalc = None
        self.z_backcalc = None
        self.beam1_backcalc = None
        self.beam2_backcalc = None
        self.beam3_backcalc = None

    def _has_backcalculated_xyz(self) -> bool:
        return all(arr is not None for arr in [self.x_backcalc, self.y_backcalc, self.z_backcalc])

    def _show_backcalculated_overview(
        self,
        time_index: pd.DatetimeIndex,
        enu_x: np.ndarray,
        enu_y: np.ndarray,
        enu_z: np.ndarray,
        x_rec: np.ndarray,
        y_rec: np.ndarray,
        z_rec: np.ndarray,
    ) -> None:
        fig = _new_or_clear_figure(
            "enu_to_xyz_backcalc",
            self.fig_registry,
            "ENU -> XYZ terugrekening (gemiddelde over cellen)",
        )
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
        ax1.plot(time_index, np.nanmean(enu_x, axis=1), label="East (origineel ENU)", linewidth=1.0)
        ax1.plot(time_index, np.nanmean(x_rec, axis=1), label="X (teruggerekend)", linewidth=1.0)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("m/s")
        ax2.plot(time_index, np.nanmean(enu_y, axis=1), label="North (origineel ENU)", linewidth=1.0)
        ax2.plot(time_index, np.nanmean(y_rec, axis=1), label="Y (teruggerekend)", linewidth=1.0)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel("m/s")
        ax3.plot(time_index, np.nanmean(enu_z, axis=1), label="Up (origineel ENU)", linewidth=1.0)
        ax3.plot(time_index, np.nanmean(z_rec, axis=1), label="Z (teruggerekend)", linewidth=1.0)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylabel("m/s")
        ax3.set_xlabel("Tijd")
        _format_time_axis(ax3)
        _apply_tight_layout(fig)
        fig.show()

    def _prepare_backcalculated_data(self, show_plot: bool = False) -> bool:
        self._clear_backcalculated_data()
        if not self.auto_backcalc_on_load_var.get():
            return False
        if self.v1 is None or self.v2 is None or self.v3 is None:
            self.log("ENU -> XYZ terugrekening overgeslagen: .dat profielsnelheden ontbreken.")
            return False
        if self.orientation_df is None or self.orientation_df.empty:
            self.log("ENU -> XYZ terugrekening aangevinkt, maar .sen heading/pitch/roll ontbreken.")
            return False

        try:
            heading = self.orientation_df["heading"].to_numpy(dtype=float)
            pitch = self.orientation_df["pitch"].to_numpy(dtype=float)
            roll = self.orientation_df["roll"].to_numpy(dtype=float)
            x_rec, y_rec, z_rec = recover_xyz_from_enu(self.v1, self.v2, self.v3, heading, pitch, roll)
            b1_rec, b2_rec, b3_rec = recover_beams_from_xyz(self.meta["T"], x_rec, y_rec, z_rec)
            self.x_backcalc = x_rec
            self.y_backcalc = y_rec
            self.z_backcalc = z_rec
            self.beam1_backcalc = b1_rec
            self.beam2_backcalc = b2_rec
            self.beam3_backcalc = b3_rec
            self.log("ENU -> XYZ terugrekening automatisch berekend.")
            if show_plot:
                self._show_backcalculated_overview(self.time_index, self.v1, self.v2, self.v3, x_rec, y_rec, z_rec)
                self.log("Controlefiguur geopend: gemiddelde ENU vs teruggerekende XYZ.")
            return True
        except Exception as exc:
            self.log(f"ENU -> XYZ terugrekening mislukt: {exc}")
            self._clear_backcalculated_data()
            return False

    def _get_xyz_plot_source(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        if self.auto_backcalc_on_load_var.get() and self._has_backcalculated_xyz():
            return self.x_backcalc, self.y_backcalc, self.z_backcalc, "backcalc"
        return self.v1, self.v2, self.v3, "raw"

    def pick_hdr(self):
        path = filedialog.askopenfilename(
            title="Kies .hdr bestand",
            filetypes=[("HDR files", "*.hdr"), ("All files", "*.*")]
        )
        if not path:
            return

        self._clear_backcalculated_data()

        try:
            meta = parse_hdr(path)
        except Exception as e:
            messagebox.showerror("Fout", f"Kan .hdr niet parsen:\n{e}")
            return

        base_prefix = os.path.splitext(path)[0]
        dat_path = resolve_companion_path(path, ".dat")
        a1_path = resolve_companion_path(path, ".a1")
        a2_path = resolve_companion_path(path, ".a2")
        a3_path = resolve_companion_path(path, ".a3")
        c1_path = resolve_companion_path(path, ".c1")
        c2_path = resolve_companion_path(path, ".c2")
        c3_path = resolve_companion_path(path, ".c3")
        csv_path = resolve_companion_path(path, ".csv")

        missing = [p for p in [dat_path, csv_path] if not os.path.exists(p)]
        if missing:
            messagebox.showwarning(
                "Bestanden ontbreken",
                "Volgende bestanden ontbreken:\n\n" + "\n".join(missing) +
                "\n\nIk ga verder met wat beschikbaar is."
            )

        # load .dat velocity profile (bron voor beam/X/Y/Z)
        try:
            if os.path.exists(dat_path):
                self.v1, self.v2, self.v3 = load_dat_velocity_cells(dat_path, meta["n_meas"], meta["n_cells"])
                self.log(f".dat profielsnelheden geladen: {os.path.basename(dat_path)}")
            else:
                self.v1 = self.v2 = self.v3 = None
        except Exception as e:
            messagebox.showerror("Fout", f"Kan .dat profielsnelheden niet inlezen:\n{e}")
            return

        # load amplitude + correlatie (optioneel, voor wandfilter)
        self.a1 = self.a2 = self.a3 = None
        self.c1 = self.c2 = self.c3 = None
        try:
            if all(os.path.exists(p) for p in [a1_path, a2_path, a3_path]):
                self.a1 = load_v_file(a1_path, meta["n_meas"], meta["n_cells"])
                self.a2 = load_v_file(a2_path, meta["n_meas"], meta["n_cells"])
                self.a3 = load_v_file(a3_path, meta["n_meas"], meta["n_cells"])
                self.log("Amplitudebestanden .a1/.a2/.a3 geladen.")
            else:
                self.log("Amplitudebestanden (.a1/.a2/.a3) niet volledig aanwezig -> wandfilter niet beschikbaar.")

            if all(os.path.exists(p) for p in [c1_path, c2_path, c3_path]):
                self.c1 = load_v_file(c1_path, meta["n_meas"], meta["n_cells"])
                self.c2 = load_v_file(c2_path, meta["n_meas"], meta["n_cells"])
                self.c3 = load_v_file(c3_path, meta["n_meas"], meta["n_cells"])
                self.log("Correlatiebestanden .c1/.c2/.c3 geladen.")
            else:
                if all(x is not None for x in [self.a1, self.a2, self.a3]):
                    self.log("Correlatiebestanden (.c1/.c2/.c3) ontbreken -> wandfilter schakelt automatisch naar amplitude-only.")
                else:
                    self.log("Correlatiebestanden (.c1/.c2/.c3) niet volledig aanwezig.")
        except Exception as e:
            self.log(f"Kon amplitude/correlatie niet inlezen: {e}")
            self.a1 = self.a2 = self.a3 = None
            self.c1 = self.c2 = self.c3 = None

        # time index length = min beschikbare arrays
        n_meas_eff = meta["n_meas"]
        for arr in [self.v1, self.v2, self.v3]:
            if arr is not None:
                n_meas_eff = min(n_meas_eff, arr.shape[0])

        # trim v arrays
        if self.v1 is not None:
            self.v1 = self.v1[:n_meas_eff, :]
        if self.v2 is not None:
            self.v2 = self.v2[:n_meas_eff, :]
        if self.v3 is not None:
            self.v3 = self.v3[:n_meas_eff, :]
        if self.a1 is not None:
            self.a1 = self.a1[:n_meas_eff, :]
        if self.a2 is not None:
            self.a2 = self.a2[:n_meas_eff, :]
        if self.a3 is not None:
            self.a3 = self.a3[:n_meas_eff, :]
        if self.c1 is not None:
            self.c1 = self.c1[:n_meas_eff, :]
        if self.c2 is not None:
            self.c2 = self.c2[:n_meas_eff, :]
        if self.c3 is not None:
            self.c3 = self.c3[:n_meas_eff, :]

        self.time_index = build_time_index(meta["t0"], n_meas_eff, meta["dt_sec"])
        self.pressure_series = None
        self.orientation_df = None
        sen_path = base_prefix + '.sen'
        if os.path.exists(sen_path):
            try:
                raw_pressure = load_sen_pressure(sen_path, meta.get("sen_pressure_column", 14))
                self.pressure_series = self._align_pressure_series(raw_pressure, self.time_index, meta["dt_sec"])
                self.orientation_df = load_sen_orientation(sen_path, meta.get("sen_columns", {}))
                if self.orientation_df is not None and not self.orientation_df.empty:
                    self.orientation_df = self._align_orientation_df(self.orientation_df, self.time_index, meta["dt_sec"])
                if self.pressure_series is not None:
                    self.log(f"Druk uit .sen geladen: {len(self.pressure_series)} punten")
                else:
                    self.log("Drukgegevens konden niet op tijdsindex worden afgestemd.")
                if self.orientation_df is not None and not self.orientation_df.empty:
                    self.log("Heading/pitch/roll uit .sen geladen voor ENU -> XYZ terugrekening.")
                else:
                    self.log("Heading/pitch/roll uit .sen niet bruikbaar voor ENU -> XYZ terugrekening.")
            except Exception as exc:
                self.log(f"Kon drukwaarden uit .sen niet inlezen: {exc}")
                self.orientation_df = None
        else:
            self.log(".sen bestand niet gevonden -> drukplots worden overgeslagen.")
        # load csv
        self.df_csv = None
        if os.path.exists(csv_path):
            try:
                self.df_csv = load_csv_processed(csv_path)
                self._log_time_axis_consistency(self.df_csv, self.time_index, meta["dt_sec"])
            except Exception as e:
                messagebox.showwarning("CSV probleem", f"CSV kon niet goed ingelezen worden:\n{e}")

        self.meta = meta
        self.base_hdr = path
        self.base_prefix = base_prefix

        self.lbl_file.configure(text=os.path.basename(path))
        self.log(f"Gekozen: {path}")
        self.log(f"n_meas={meta['n_meas']} | n_cells={meta['n_cells']} | dt={meta['dt_sec']} s | start={meta['t0']}")
        if meta["cell_distances_m"]:
            self.log("Cell center distances (m): " + ", ".join(f"{x:.2f}" for x in meta["cell_distances_m"]))
        self.log("Transformation matrix T:\n" + "\n".join("  " + "  ".join(f"{v:8.4f}" for v in r) for r in meta["T"]))
        if self.auto_backcalc_on_load_var.get():
            auto_backcalc_ok = self._prepare_backcalculated_data(show_plot=True)
            if not auto_backcalc_ok:
                messagebox.showwarning(
                    "ENU -> XYZ niet berekend",
                    "De optie voor automatische ENU -> XYZ terugrekening stond aan,\n"
                    "maar de nodige .sen heading/pitch/roll gegevens ontbreken of zijn ongeldig."
                )

        # rebuild cell checkboxes
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self.cell_vars = []
        n_cells = meta["n_cells"]
        dists = meta["cell_distances_m"]

        for i in range(n_cells):
            v = tk.BooleanVar(value=True)
            self.cell_vars.append(v)
            label = f"Cel {i+1}"
            if dists and i < len(dists):
                label += f"  ({dists[i]:.2f} m)"
            cb = ttk.Checkbutton(self.scroll_frame, text=label, variable=v)
            cb.pack(anchor="w")

        # fill time range defaults
        self.fill_full_range()

        self.log("Cellen aangevinkt (standaard: allemaal). Tijdvenster ingevuld met volledig bereik.")

    def selected_cells(self):
        return [i for i, v in enumerate(self.cell_vars) if v.get()]

    def recompute_and_plot(self):
        if self.meta is None or self.time_index is None:
            messagebox.showinfo("Info", "Kies eerst een .hdr bestand.")
            return
        if self.v1 is None or self.v2 is None or self.v3 is None:
            messagebox.showerror("Fout", "Profielsnelheden uit .dat ontbreken. Beam/XYZ-plots vereisen .dat.")
            return

        try:
            rm_window = self._get_running_mean_window()
        except ValueError as exc:
            messagebox.showerror("Lopend gemiddelde fout", str(exc))
            return
        try:
            amp_jump_min, corr_max, corr_drop_min, smooth_k, min_beams, hard_max_dist_m = self._get_wall_filter_settings()
        except ValueError as exc:
            messagebox.showerror("Wanddetectie fout", str(exc))
            return

        # parse window
        try:
            start_dt = self.parse_time_entry(self.start_var.get())
            end_dt = self.parse_time_entry(self.end_var.get())
        except Exception as e:
            messagebox.showerror("Tijdvenster fout", str(e))
            return

        if start_dt and end_dt and end_dt < start_dt:
            messagebox.showerror("Tijdvenster fout", "Eindmoment ligt vÃ³Ã³r beginmoment.")
            return

        mask = filter_time_window(self.time_index, start_dt, end_dt)
        if not mask.any():
            messagebox.showwarning("Tijdvenster", "Geen data binnen dit tijdvenster.")
            return

        ti = self.time_index[mask]
        pressure_plot = None
        if self.pressure_series is not None:
            pressure_window = np.asarray(self.pressure_series.iloc[mask])
            if pressure_window.size > 0:
                pressure_plot = apply_running_mean(pressure_window, rm_window)
        v1w = self.v1[mask, :]
        v2w = self.v2[mask, :]
        v3w = self.v3[mask, :]
        x_src, y_src, z_src, xyz_mode = self._get_xyz_plot_source()
        xw = x_src[mask, :]
        yw = y_src[mask, :]
        zw = z_src[mask, :]
        dists = self.meta["cell_distances_m"]

        if self.wall_filter_var.get():
            earliest_jump_m = None
            effective_max_m = hard_max_dist_m
            if all(x is not None for x in [self.a1, self.a2, self.a3]):
                earliest_jump_m = find_earliest_jump_distance(
                    self.a1[mask, :],
                    self.a2[mask, :],
                    self.a3[mask, :],
                    dists,
                    amp_jump_min=amp_jump_min,
                    smooth_k=smooth_k,
                )
                if earliest_jump_m is not None:
                    effective_max_m = min(hard_max_dist_m, earliest_jump_m)

            if all(x is not None for x in [self.a1, self.a2, self.a3, self.c1, self.c2, self.c3]):
                a1w = self.a1[mask, :]
                a2w = self.a2[mask, :]
                a3w = self.a3[mask, :]
                c1w = self.c1[mask, :]
                c2w = self.c2[mask, :]
                c3w = self.c3[mask, :]
                wall_mask, beam_hits = detect_wall_mask_from_amp_corr(
                    a1w, a2w, a3w, c1w, c2w, c3w,
                    amp_jump_min=amp_jump_min,
                    corr_max=corr_max,
                    corr_drop_min=corr_drop_min,
                    smooth_k=smooth_k,
                    min_beams=min_beams,
                )
                dist_mask = build_distance_mask(v1w.shape[0], v1w.shape[1], dists, effective_max_m)
                # Laat wall-detectie alleen ingrijpen in de zone op/na de cutoff.
                wall_mask = wall_mask & dist_mask
                v1w = np.where(wall_mask, np.nan, v1w)
                v2w = np.where(wall_mask, np.nan, v2w)
                v3w = np.where(wall_mask, np.nan, v3w)
                xw = np.where(wall_mask, np.nan, xw)
                yw = np.where(wall_mask, np.nan, yw)
                zw = np.where(wall_mask, np.nan, zw)
                v1w = np.where(dist_mask, np.nan, v1w)
                v2w = np.where(dist_mask, np.nan, v2w)
                v3w = np.where(dist_mask, np.nan, v3w)
                xw = np.where(dist_mask, np.nan, xw)
                yw = np.where(dist_mask, np.nan, yw)
                zw = np.where(dist_mask, np.nan, zw)
                wall_fraction = float(wall_mask.mean())
                hit_frac = [float((beam_hits[i, :] >= 0).mean()) for i in range(3)]
                self.log(
                    "Wandfilter actief: "
                    f"wall_fraction={wall_fraction:.3f}, "
                    f"hit_frac_beams=({hit_frac[0]:.3f}, {hit_frac[1]:.3f}, {hit_frac[2]:.3f})"
                )
                self.log(
                    f"Afstandscutoff toegepast: hard_max={hard_max_dist_m:.2f} m, "
                    f"vroegste_sprong={earliest_jump_m:.2f} m" if earliest_jump_m is not None
                    else f"Afstandscutoff toegepast: hard_max={hard_max_dist_m:.2f} m (geen sprong gevonden)."
                )
            elif all(x is not None for x in [self.a1, self.a2, self.a3]):
                wall_mask, beam_hits = detect_wall_mask_from_amp_only(
                    a1=self.a1[mask, :],
                    a2=self.a2[mask, :],
                    a3=self.a3[mask, :],
                    amp_jump_min=amp_jump_min,
                    smooth_k=smooth_k,
                    min_beams=min_beams,
                )
                dist_mask = build_distance_mask(v1w.shape[0], v1w.shape[1], dists, effective_max_m)
                # Laat wall-detectie alleen ingrijpen in de zone op/na de cutoff.
                wall_mask = wall_mask & dist_mask
                v1w = np.where(wall_mask, np.nan, v1w)
                v2w = np.where(wall_mask, np.nan, v2w)
                v3w = np.where(wall_mask, np.nan, v3w)
                xw = np.where(wall_mask, np.nan, xw)
                yw = np.where(wall_mask, np.nan, yw)
                zw = np.where(wall_mask, np.nan, zw)
                v1w = np.where(dist_mask, np.nan, v1w)
                v2w = np.where(dist_mask, np.nan, v2w)
                v3w = np.where(dist_mask, np.nan, v3w)
                xw = np.where(dist_mask, np.nan, xw)
                yw = np.where(dist_mask, np.nan, yw)
                zw = np.where(dist_mask, np.nan, zw)
                wall_fraction = float(wall_mask.mean())
                hit_frac = [float((beam_hits[i, :] >= 0).mean()) for i in range(3)]
                self.log(
                    "Wandfilter actief (amplitude-only, corr ontbreekt): "
                    f"wall_fraction={wall_fraction:.3f}, "
                    f"hit_frac_beams=({hit_frac[0]:.3f}, {hit_frac[1]:.3f}, {hit_frac[2]:.3f})"
                )
                self.log(
                    f"Afstandscutoff toegepast: hard_max={hard_max_dist_m:.2f} m, "
                    f"vroegste_sprong={earliest_jump_m:.2f} m" if earliest_jump_m is not None
                    else f"Afstandscutoff toegepast: hard_max={hard_max_dist_m:.2f} m (geen sprong gevonden)."
                )
            else:
                self.log("Wandfilter aangevinkt maar amp/corr bestanden ontbreken -> filter overgeslagen.")

        v1w_plot = apply_running_mean(v1w, rm_window)
        v2w_plot = apply_running_mean(v2w, rm_window)
        v3w_plot = apply_running_mean(v3w, rm_window)

        sel = self.selected_cells()
        sel_valid = [
            ci for ci in sel
            if (
                np.isfinite(v1w_plot[:, ci]).any()
                or np.isfinite(v2w_plot[:, ci]).any()
                or np.isfinite(v3w_plot[:, ci]).any()
            )
        ]
        removed = len(sel) - len(sel_valid)
        if removed > 0:
            self.log(f"Valid-only selectie: {removed} cel(len) verwijderd door cutoff/filter; {len(sel_valid)} over.")
        if not sel_valid:
            self.log("Geen geldige cellen over na cutoff/filter in dit tijdvenster.")

        sel = sel_valid

        # Beam figures (windowed)
        plot_beam(
            ti, v1w_plot, sel, dists, "East (.dat ENU)" if xyz_mode == "backcalc" else "Beam 1 / X (.dat)", self.fig_registry,
            pressure_arr=pressure_plot,
            show_cells=bool(self.show_cell_lines_var.get()),
            show_mean=bool(self.show_mean_line_var.get()),
        )
        plot_beam(
            ti, v2w_plot, sel, dists, "North (.dat ENU)" if xyz_mode == "backcalc" else "Beam 2 / Y (.dat)", self.fig_registry,
            pressure_arr=pressure_plot,
            show_cells=bool(self.show_cell_lines_var.get()),
            show_mean=bool(self.show_mean_line_var.get()),
        )
        plot_beam(
            ti, v3w_plot, sel, dists, "Up (.dat ENU)" if xyz_mode == "backcalc" else "Beam 3 / Z (.dat)", self.fig_registry,
            pressure_arr=pressure_plot,
            show_cells=bool(self.show_cell_lines_var.get()),
            show_mean=bool(self.show_mean_line_var.get()),
        )

        if self.show_amplitude_profile_var.get():
            if all(x is not None for x in [self.a1, self.a2, self.a3]):
                earliest_jump_plot = find_earliest_jump_distance(
                    self.a1[mask, :],
                    self.a2[mask, :],
                    self.a3[mask, :],
                    dists,
                    amp_jump_min=amp_jump_min,
                    smooth_k=smooth_k,
                )
                effective_max_plot = hard_max_dist_m
                if earliest_jump_plot is not None:
                    effective_max_plot = min(hard_max_dist_m, earliest_jump_plot)
                plot_amplitude_profile(
                    self.a1[mask, :],
                    self.a2[mask, :],
                    self.a3[mask, :],
                    dists,
                    self.fig_registry,
                    max_distance_m=effective_max_plot,
                    earliest_jump_m=earliest_jump_plot,
                )
            else:
                self.log("Amplitudeprofiel over hoogte niet getoond: .a1/.a2/.a3 ontbreken.")

        # Resultant from CSV within window (filter by DateTime)
        if self.df_csv is not None:
            dfw = self.df_csv.copy().sort_values("DateTime")
            if start_dt is not None:
                dfw = dfw[dfw["DateTime"] >= start_dt]
            if end_dt is not None:
                dfw = dfw[dfw["DateTime"] <= end_dt]
            pressure_for_csv = None
            if self.pressure_series is not None and not dfw.empty:
                pressure_for_csv = self._align_pressure_series(
                    self.pressure_series,
                    pd.DatetimeIndex(dfw["DateTime"].values),
                    self.meta["dt_sec"]
                )
            plot_resultant_from_csv(dfw, self.selected_cells(), self.fig_registry, pressure_arr=pressure_for_csv)
            if self.show_hrp_var.get():
                plot_hrp_from_csv(dfw, self.fig_registry, pressure_arr=pressure_for_csv)
        else:
            self.log("Geen CSV geladen -> resultante speed/dir figuur wordt overgeslagen.")

        x_plot = apply_running_mean(xw, rm_window)
        y_plot = apply_running_mean(yw, rm_window)
        z_plot = apply_running_mean(zw, rm_window)
        x_title = "X (ENU -> XYZ terugrekening)" if xyz_mode == "backcalc" else "X (uit .dat)"
        y_title = "Y (ENU -> XYZ terugrekening)" if xyz_mode == "backcalc" else "Y (uit .dat)"
        z_title = "Z (ENU -> XYZ terugrekening)" if xyz_mode == "backcalc" else "Z (uit .dat)"

        plot_component(
            ti,
            x_plot,
            sel,
            dists,
            x_title,
            "X (m/s)",
            "comp_x",
            self.fig_registry,
            pressure_arr=pressure_plot,
            show_cells=bool(self.show_cell_lines_var.get()),
            show_mean=bool(self.show_mean_line_var.get()),
        )
        plot_component(
            ti,
            y_plot,
            sel,
            dists,
            y_title,
            "Y (m/s)",
            "comp_y",
            self.fig_registry,
            pressure_arr=pressure_plot,
            show_cells=bool(self.show_cell_lines_var.get()),
            show_mean=bool(self.show_mean_line_var.get()),
        )
        plot_component(
            ti,
            z_plot,
            sel,
            dists,
            z_title,
            "Z (m/s)",
            "comp_z",
            self.fig_registry,
            pressure_arr=pressure_plot,
            show_cells=bool(self.show_cell_lines_var.get()),
            show_mean=bool(self.show_mean_line_var.get()),
        )

        if sel:
            u_sel = np.nanmean(x_plot[:, sel], axis=1)
            if self.rose_mode_var.get() == "Alleen X":
                v_sel = np.zeros_like(u_sel)
            else:
                v_sel = np.nanmean(y_plot[:, sel], axis=1)

            plot_current_rose(
                u=u_sel,
                v=v_sel,
                title="Stroomroos - gemiddelde stroming (gekozen cellen, tijdvenster)",
                fig_key="current_rose",
                fig_registry=self.fig_registry,
            )

        # Optional export (windowed)
        if self.export_var.get():
            try:
                self._export_current_windowed(sel, ti, v1w_plot, v2w_plot, v3w_plot, x_plot, y_plot, z_plot, start_dt, end_dt)
            except Exception as e:
                messagebox.showwarning("Export fout", f"Export faalde:\n{e}")

    def _export_current_windowed(self, sel, ti, v1w, v2w, v3w, x, y, z, start_dt, end_dt):
        out_default = self.base_prefix + "_export_window.tsv"
        out_path = filedialog.asksaveasfilename(
            title="Kies exportbestand",
            defaultextension=".tsv",
            initialfile=os.path.basename(out_default),
            filetypes=[("TSV (tab separated)", "*.tsv"), ("All files", "*.*")]
        )
        if not out_path:
            return

        b1_avg = np.nanmean(v1w[:, sel], axis=1) if sel else np.full(len(ti), np.nan)
        b2_avg = np.nanmean(v2w[:, sel], axis=1) if sel else np.full(len(ti), np.nan)
        b3_avg = np.nanmean(v3w[:, sel], axis=1) if sel else np.full(len(ti), np.nan)

        x_avg = np.nanmean(x[:, sel], axis=1) if sel else np.full(len(ti), np.nan)
        y_avg = np.nanmean(y[:, sel], axis=1) if sel else np.full(len(ti), np.nan)
        z_avg = np.nanmean(z[:, sel], axis=1) if sel else np.full(len(ti), np.nan)

        df_out = pd.DataFrame({
            "DateTime": ti,
            "beam1_avg": b1_avg,
            "beam2_avg": b2_avg,
            "beam3_avg": b3_avg,
            "x_avg": x_avg,
            "y_avg": y_avg,
            "z_avg": z_avg,
        })

        # CSV toevoegen indien beschikbaar (filtered window)
        if self.df_csv is not None and "DateTime" in self.df_csv.columns:
            df_csv = self.df_csv.copy().sort_values("DateTime")
            if start_dt is not None:
                df_csv = df_csv[df_csv["DateTime"] >= start_dt]
            if end_dt is not None:
                df_csv = df_csv[df_csv["DateTime"] <= end_dt]

            # Resultante uit CSV over cel 1 t/m hoogste ingestelde cel.
            sel_for_csv = self.selected_cells()
            max_cell = (max(sel_for_csv) + 1) if sel_for_csv else 0
            _, spd, direc, used_cells = _compute_csv_avg_speed_dir_to_max_cell(df_csv, max_cell)
            if used_cells:
                df_csv = df_csv.copy()
                df_csv["resultant_speed_sel"] = spd
                df_csv["resultant_dir_sel"] = direc

            keep_cols = [c for c in ["Battery", "Heading", "Pitch", "Roll", "Pressure", "Temperature",
                                     "resultant_speed_sel", "resultant_dir_sel"] if c in df_csv.columns]
            df_csv_small = df_csv[["DateTime"] + keep_cols].copy()

            df_out = df_out.sort_values("DateTime")
            df_csv_small = df_csv_small.sort_values("DateTime")
            merge_tol = pd.Timedelta(seconds=max(float(self.meta.get("dt_sec", 1.0)), 1.0))
            df_out = pd.merge_asof(
                df_out,
                df_csv_small,
                on="DateTime",
                direction="nearest",
                tolerance=merge_tol,
            )

        export_tsv_eu(df_out, out_path)
        self.log(f"Export geschreven: {out_path}")

    def export_figures_to_word(self):
        if Document is None or Inches is None:
            messagebox.showerror(
                "Fout",
                "De module 'python-docx' ontbreekt. Installeer deze via 'pip install python-docx' om grafieken naar Word te exporteren."
            )
            return

        figs = [
            (name, fig)
            for name, fig in self.fig_registry.items()
            if plt.fignum_exists(fig.number)
        ]
        if not figs:
            messagebox.showinfo("Info", "Er zijn geen actieve figuren om te exporteren.")
            return

        temp_dir = tempfile.mkdtemp(prefix="aqp_figs_")
        saved = []
        try:
            for name, fig in figs:
                sanitized = name.replace(" ", "_")
                path = os.path.join(temp_dir, f"{sanitized}.png")
                fig.savefig(path, bbox_inches="tight")
                saved.append((name, path))

            doc = Document()
            doc.add_heading("Aquadopp - Openstaande grafieken", level=1)
            start_str = (self.start_var.get() or "").strip()
            end_str = (self.end_var.get() or "").strip()
            if not start_str:
                start_str = "begin (volledig bereik)"
            if not end_str:
                end_str = "einde (volledig bereik)"
            window_desc = (
                f"Tijdvenster: {start_str} -> {end_str}\n"
                f"Lopend gemiddelde: {self.running_mean_var.get() or '1'}\n"
                f"Stroomroos-bron: {self.rose_mode_var.get()}\n"
                f"HRP-plot: {bool(self.show_hrp_var.get())}\n"
                f"Geselecteerde cellen: {len(self.selected_cells())}"
            )
            doc.add_paragraph("Instellingen", style="Heading 2")
            for line in window_desc.splitlines():
                doc.add_paragraph(line, style="List Bullet")
            for idx, (name, path) in enumerate(saved):
                if idx:
                    doc.add_page_break()
                doc.add_paragraph(name)
                doc.add_picture(path, width=Inches(6))

            initialdir = os.path.dirname(self.base_prefix) if self.base_prefix else os.getcwd()
            initialfile = (
                f"{os.path.basename(self.base_prefix) if self.base_prefix else 'aquadopp'}_figuren.docx"
            )
            out_path = filedialog.asksaveasfilename(
                title="Kies Word-bestand",
                initialdir=initialdir,
                initialfile=initialfile,
                defaultextension=".docx",
                filetypes=[("Word document", "*.docx"), ("All files", "*.*")]
            )
            if not out_path:
                return
            doc.save(out_path)
            self.log(f"Figuren in Word geschreven: {out_path}")
        except Exception as exc:
            messagebox.showerror("Fout", f"Kon Word-export niet uitvoeren:\n{exc}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def export_enu_to_xyz_backcalculation(self):
        if self.meta is None or self.time_index is None:
            messagebox.showinfo("Info", "Kies eerst een .hdr bestand.")
            return
        if self.v1 is None or self.v2 is None or self.v3 is None:
            messagebox.showerror("Fout", "Profielsnelheden uit .dat ontbreken.")
            return
        if self.orientation_df is None or self.orientation_df.empty:
            messagebox.showerror("Fout", "Heading/pitch/roll uit .sen ontbreken. Terugrekening ENU -> XYZ is niet mogelijk.")
            return

        out_default = self.base_prefix + "_teruggerekend_xyz_beam.csv"
        out_csv = filedialog.asksaveasfilename(
            title="Kies CSV voor ENU -> XYZ terugrekening",
            defaultextension=".csv",
            initialfile=os.path.basename(out_default),
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not out_csv:
            return

        try:
            heading = self.orientation_df["heading"].to_numpy(dtype=float)
            pitch = self.orientation_df["pitch"].to_numpy(dtype=float)
            roll = self.orientation_df["roll"].to_numpy(dtype=float)
            if self._has_backcalculated_xyz():
                x_rec, y_rec, z_rec = self.x_backcalc, self.y_backcalc, self.z_backcalc
                b1_rec, b2_rec, b3_rec = self.beam1_backcalc, self.beam2_backcalc, self.beam3_backcalc
            else:
                x_rec, y_rec, z_rec = recover_xyz_from_enu(self.v1, self.v2, self.v3, heading, pitch, roll)
                b1_rec, b2_rec, b3_rec = recover_beams_from_xyz(self.meta["T"], x_rec, y_rec, z_rec)

            df_export = pd.DataFrame({"DateTime": self.time_index})
            for name, values in (
                ("heading", heading),
                ("pitch", pitch),
                ("roll", roll),
            ):
                df_export[name] = values

            component_blocks = [
                ("east", self.v1),
                ("north", self.v2),
                ("up", self.v3),
                ("x_recovered", x_rec),
                ("y_recovered", y_rec),
                ("z_recovered", z_rec),
                ("beam1_recovered", b1_rec),
                ("beam2_recovered", b2_rec),
                ("beam3_recovered", b3_rec),
            ]
            for prefix, arr in component_blocks:
                for ci in range(arr.shape[1]):
                    df_export[f"{prefix}_cell{ci+1}"] = arr[:, ci]

            df_export.to_csv(out_csv, index=False)

            out_npz = os.path.splitext(out_csv)[0] + ".npz"
            np.savez_compressed(
                out_npz,
                time=self.time_index.astype("datetime64[ns]").values,
                cell_distances_m=np.array(self.meta["cell_distances_m"], dtype=float) if self.meta["cell_distances_m"] else np.array([]),
                T=self.meta["T"],
                east=self.v1,
                north=self.v2,
                up=self.v3,
                x=x_rec,
                y=y_rec,
                z=z_rec,
                beam1=b1_rec,
                beam2=b2_rec,
                beam3=b3_rec,
                heading=heading,
                pitch=pitch,
                roll=roll,
            )

            self._show_backcalculated_overview(self.time_index, self.v1, self.v2, self.v3, x_rec, y_rec, z_rec)
            self.log(f"ENU -> XYZ terugrekening geschreven: {out_csv}")
            self.log(f"ENU -> XYZ terugrekening NPZ: {out_npz}")
            self.log("Controlefiguur geopend: gemiddelde ENU vs teruggerekende XYZ.")
        except Exception as exc:
            messagebox.showerror("Fout", f"Terugrekening ENU -> XYZ faalde:\n{exc}")

    def save_npz(self):
        if self.meta is None or self.time_index is None:
            messagebox.showinfo("Info", "Kies eerst een .hdr bestand.")
            return
        if self.v1 is None or self.v2 is None or self.v3 is None:
            messagebox.showerror("Fout", ".dat profielsnelheden vereist.")
            return

        out_default = self.base_prefix + "_beams_xyz.npz"
        out_path = filedialog.asksaveasfilename(
            title="Kies NPZ opslagbestand",
            defaultextension=".npz",
            initialfile=os.path.basename(out_default),
            filetypes=[("NPZ", "*.npz"), ("All files", "*.*")]
        )
        if not out_path:
            return

        T = self.meta["T"]
        x, y, z, xyz_mode = self._get_xyz_plot_source()

        np.savez_compressed(
            out_path,
            time=self.time_index.astype("datetime64[ns]").values,
            cell_distances_m=np.array(self.meta["cell_distances_m"], dtype=float) if self.meta["cell_distances_m"] else np.array([]),
            T=T,
            v1=self.v1,
            v2=self.v2,
            v3=self.v3,
            xyz_source=np.array([xyz_mode]),
            x=x, y=y, z=z
        )
        self.log(f"NPZ geschreven: {out_path}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
