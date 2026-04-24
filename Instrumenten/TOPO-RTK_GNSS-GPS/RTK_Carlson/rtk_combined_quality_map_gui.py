#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import html
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pyproj import Transformer
from tkinter import filedialog, messagebox, ttk

try:
    import folium
except Exception:
    folium = None

NONE = "(none)"

DEFAULT_SIGMA_XY_A = 0.005
DEFAULT_SIGMA_Z_A = 0.010
DEFAULT_SIGMA_XY_B = 0.010
DEFAULT_SIGMA_Z_B = 0.020
DEFAULT_OUTLIER_SIGMA = 2.0
DEFAULT_MAP_TILES = "CartoDB positron"


@dataclass
class Guess:
    name: Optional[str] = None
    group: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    z: Optional[str] = None
    sigma_xy: Optional[str] = None
    sigma_z: Optional[str] = None
    time: Optional[str] = None
    date: Optional[str] = None
    profile: str = "Onbekend"
    note: str = ""


@dataclass
class DaeOpts:
    cube: float = 0.75
    ribbon: bool = True
    ribbon_w: float = 0.30
    use_h: bool = False
    every: int = 1


@dataclass
class ProcessOptions:
    output_base: Path
    crs: str
    name_col: Optional[str]
    group_col: Optional[str]
    x_col: str
    y_col: str
    z_col: str
    sigma_xy_col: str
    sigma_z_col: str
    time_col: Optional[str]
    date_col: Optional[str]
    sigma_xy_a: float
    sigma_z_a: float
    sigma_xy_b: float
    sigma_z_b: float
    outlier_sigma: float
    worst_allowed_quality: str
    max_sigma_xy: Optional[float]
    max_sigma_z: Optional[float]
    apply_outlier_filter: bool
    map_every: int
    dae_every: int
    cube: float
    ribbon: bool
    ribbon_w: float
    use_h: bool
    lift: float


def s(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() == "nan" else text


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def is_num(value: object) -> bool:
    try:
        float(s(value).replace(",", "."))
        return bool(s(value))
    except ValueError:
        return False


def sniff_delim(path: str | Path, default: str = ";") -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()][:50]
    candidates = ["\t", ";", "|", ","]
    best_delim: Optional[str] = None
    best_score: tuple[float, float, float] | None = None

    for delim in candidates:
        counts = [line.count(delim) for line in lines]
        positive = [count for count in counts if count > 0]
        if len(positive) < 2:
            continue
        mean_count = float(np.mean(positive))
        std_count = float(np.std(positive))
        consistency = len(positive) / max(1, len(lines))
        score = (consistency, mean_count, -std_count)
        if best_score is None or score > best_score:
            best_delim, best_score = delim, score

    if best_delim is not None:
        return best_delim

    sample = text[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        return default


def detect_header(path: str | Path, delim: str) -> bool:
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        for row in csv.reader(fh, delimiter=delim):
            if row and any(str(x).strip() for x in row):
                rows.append([str(x).strip() for x in row])
            if len(rows) >= 3:
                break
    if len(rows) < 2:
        return True
    hdr = {
        "pt id", "point id", "id", "name", "desc", "description", "omschrijving", "groep", "setup",
        "x", "y", "z", "e", "n", "sigmaxy", "sigma z", "sigmaz", "time", "date", "datum",
    }
    if {x.lower() for x in rows[0]} & hdr:
        return True
    a = sum(is_num(x) for x in rows[0]) / max(1, len(rows[0]))
    b = sum(is_num(x) for x in rows[1]) / max(1, len(rows[1]))
    return a < 0.35 and b >= 0.35


def read_table(path: str | Path, delim: str, header_mode: str) -> tuple[pd.DataFrame, bool, str]:
    if delim == "auto":
        delim = sniff_delim(path)
    has_header = detect_header(path, delim) if header_mode == "auto" else header_mode == "yes"
    df = pd.read_csv(path, sep=delim, header=0 if has_header else None, dtype=str, keep_default_na=False, engine="python")
    if has_header:
        df.columns = [s(col) or f"col{i+1}" for i, col in enumerate(df.columns)]
    else:
        df.columns = [f"col{i+1}" for i in range(df.shape[1])]
    for col in df.columns:
        df[col] = df[col].map(s)
    return df, has_header, delim


def guess_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def num_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if float(numeric(df[col]).notna().mean()) >= 0.75]


def guess_cols(df: pd.DataFrame, has_header: bool) -> Guess:
    g = Guess(
        name=guess_col(df, ["pt id", "point id", "id", "name"]),
        group=guess_col(df, ["desc", "description", "omschrijving", "groep", "setup"]),
        x=guess_col(df, ["x", "e", "east", "easting"]),
        y=guess_col(df, ["y", "n", "north", "northing"]),
        z=guess_col(df, ["z", "h", "height", "hoogte"]),
        sigma_xy=guess_col(df, ["sigmaxy", "sigma xy", "hz precision", "precisionxy", "rmsxy"]),
        sigma_z=guess_col(df, ["sigmaz", "sigma z", "sigmah", "vertical precision", "precisionz", "rmsz"]),
        time=guess_col(df, ["time", "tijd", "timestamp"]),
        date=guess_col(df, ["date", "datum"]),
        profile="Header-profiel",
    )
    if g.x or g.y or g.name:
        return g

    cols = list(df.columns)
    nums = num_cols(df)
    if not has_header and len(cols) >= 9:
        return Guess(
            name=cols[0],
            group=cols[1],
            x=cols[2],
            y=cols[3],
            z=cols[4],
            sigma_xy=cols[5],
            sigma_z=cols[6],
            time=cols[7],
            date=cols[8],
            profile="Carlson TXT zonder header",
            note="Kolomvolgorde 1-9 gebruikt als naam, setup, X, Y, Z, SigmaXY, SigmaZ, tijd en datum.",
        )
    if len(nums) >= 5:
        return Guess(
            name=cols[0] if cols else None,
            group=cols[1] if len(cols) > 1 else None,
            x=nums[0],
            y=nums[1],
            z=nums[2],
            sigma_xy=nums[3],
            sigma_z=nums[4],
            profile="Generieke detectie",
            note="Eerste vijf numerieke kolommen gekozen als X, Y, Z, SigmaXY en SigmaZ.",
        )
    return g


def quality_flag(sigma_xy: float, sigma_z: float, opts: ProcessOptions) -> str:
    if pd.isna(sigma_xy) or pd.isna(sigma_z):
        return "?"
    if sigma_xy <= opts.sigma_xy_a and sigma_z <= opts.sigma_z_a:
        return "A"
    if sigma_xy <= opts.sigma_xy_b and sigma_z <= opts.sigma_z_b:
        return "B"
    return "C"


def parse_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    stamp = (date_series.map(s) + " " + time_series.map(s)).str.strip()
    stamp = stamp.mask(stamp.eq(""))
    year_first_mask = stamp.fillna("").str.match(r"^\d{4}[/\-]")
    if bool(year_first_mask.any()):
        dt = pd.to_datetime(stamp, errors="coerce", yearfirst=True)
    else:
        dt = pd.to_datetime(stamp, errors="coerce", dayfirst=True)
    if dt.notna().any():
        return dt
    return pd.to_datetime(stamp, errors="coerce", yearfirst=True)


def derive_base_name(name_series: pd.Series, group_series: pd.Series) -> pd.Series:
    name = name_series.map(s)
    group = group_series.map(s)
    base = name.where(name.ne(""), group)
    base = base.str.replace(r"[-_\s]*\d+$", "", regex=True).str.rstrip("-_ ").str.strip()
    return base.where(base.ne(""), "Onbekend punt")


def add_reason(series: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    out = series.astype(str).copy()
    out.loc[mask] = np.where(out.loc[mask].eq(""), reason, out.loc[mask] + "; " + reason)
    return out


def apply_filters(df: pd.DataFrame, opts: ProcessOptions) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = {"A": 0, "B": 1, "C": 2}
    raw = df.copy()
    raw["FilterReason"] = ""

    if opts.max_sigma_xy is not None:
        mask = raw["SigmaXY"] > opts.max_sigma_xy
        raw["FilterReason"] = add_reason(raw["FilterReason"], mask, f"SigmaXY>{opts.max_sigma_xy:.4f}")

    if opts.max_sigma_z is not None:
        mask = raw["SigmaZ"] > opts.max_sigma_z
        raw["FilterReason"] = add_reason(raw["FilterReason"], mask, f"SigmaZ>{opts.max_sigma_z:.4f}")

    allowed_rank = ranks.get(opts.worst_allowed_quality, 2)
    quality_rank = raw["Quality"].map(ranks).fillna(99)
    mask = quality_rank > allowed_rank
    raw["FilterReason"] = add_reason(raw["FilterReason"], mask, f"Kwaliteit>{opts.worst_allowed_quality}")
    raw["PassInitialFilter"] = raw["FilterReason"].eq("")

    filtered = raw.loc[raw["PassInitialFilter"]].copy()
    filtered["PassOutlierFilter"] = True
    filtered["OutlierReason"] = ""

    if not opts.apply_outlier_filter or filtered.empty:
        return raw, filtered

    keep_index: list[int] = []
    for _, group in filtered.groupby("BaseName"):
        z_std = group["Z"].std()
        if pd.isna(z_std) or z_std == 0:
            keep_index.extend(group.index.tolist())
            continue
        z_mean = group["Z"].mean()
        ok_mask = (group["Z"] - z_mean).abs() <= opts.outlier_sigma * z_std
        keep_index.extend(group.index[ok_mask].tolist())
        bad_index = group.index[~ok_mask]
        filtered.loc[bad_index, "PassOutlierFilter"] = False
        filtered.loc[bad_index, "OutlierReason"] = f"|Z-mean|>{opts.outlier_sigma:.2f}*std"

    filtered = filtered.loc[filtered.index.isin(keep_index)].copy()
    return raw, filtered


def summarise_quality(series: pd.Series) -> str:
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else "?"


def project_xy(crs: str, x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if crs == "EPSG:4326":
        return x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return tr.transform(x.to_numpy(dtype=float), y.to_numpy(dtype=float))


def color_for_quality(quality: str) -> str:
    return {"A": "#1a9850", "B": "#fdae61", "C": "#d73027", "?": "#7f7f7f"}.get(quality, "#7f7f7f")


def label_for_track(row: pd.Series, idx: int) -> str:
    name = s(row.get("PointName", ""))
    group = s(row.get("GroupName", ""))
    if name and group and name != group:
        return f"{name} | {group}"
    return name or group or f"Punt {idx + 1}"


def details_for_track(row: pd.Series, idx: int) -> list[tuple[str, str]]:
    out = [("Index", str(idx + 1))]
    for key, label in (("PointName", "Naam"), ("GroupName", "Setup"), ("BaseName", "Basisnaam"), ("Quality", "Kwaliteit")):
        if s(row.get(key, "")):
            out.append((label, s(row[key])))
    for key, label in (("X", "X"), ("Y", "Y"), ("Z", "Z"), ("SigmaXY", "SigmaXY"), ("SigmaZ", "SigmaZ"), ("Latitude", "Lat"), ("Longitude", "Lon")):
        val = row.get(key, np.nan)
        if not pd.isna(val):
            out.append((label, f"{float(val):.6f}" if label in {"Lat", "Lon"} else f"{float(val):.4f}"))
    if s(row.get("Time", "")):
        out.append(("Tijd", s(row["Time"])))
    if s(row.get("Date", "")):
        out.append(("Datum", s(row["Date"])))
    return out


def popup_html_track(row: pd.Series, idx: int) -> str:
    return "<br>".join(f"<b>{html.escape(k)}:</b> {html.escape(v)}" for k, v in details_for_track(row, idx))


def hover_text_for_measurement(row: pd.Series, idx: int) -> str:
    return "\n".join(f"{k}: {v}" for k, v in details_for_track(row, idx))


def hover_text_for_agg(row: pd.Series) -> str:
    parts = [
        f"Punt: {s(row.get('BaseName', ''))}",
        f"Kwaliteit: {s(row.get('Quality', ''))}",
        f"Aantal: {int(row.get('N', 0))}",
        f"X: {float(row['X']):.4f}",
        f"Y: {float(row['Y']):.4f}",
        f"Z: {float(row['Z']):.4f}",
        f"SigmaXY: {float(row['SigmaXY']):.4f}",
        f"SigmaZ: {float(row['SigmaZ']):.4f}",
    ]
    return "\n".join(parts)


def make_track_map(df: pd.DataFrame, out_html: Path, every: int) -> bool:
    if folium is None or df.empty:
        return False
    center = [float(df["Latitude"].median()), float(df["Longitude"].median())]
    m = folium.Map(location=center, zoom_start=17, tiles=DEFAULT_MAP_TILES)
    pts = df[["Latitude", "Longitude"]].astype(float).values.tolist()
    if len(pts) >= 2:
        folium.PolyLine(pts, weight=4, color="#d95f02", opacity=0.9).add_to(m)
    for idx, row in df.iloc[:: max(1, int(every))].reset_index(drop=True).iterrows():
        folium.CircleMarker(
            location=[float(row["Latitude"]), float(row["Longitude"])],
            radius=5,
            color=color_for_quality(str(row["Quality"])),
            fill=True,
            fill_color=color_for_quality(str(row["Quality"])),
            fill_opacity=0.95,
            tooltip=f"{label_for_track(row, idx)} | Q={row['Quality']}",
            popup=folium.Popup(popup_html_track(row, idx), max_width=460),
        ).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return True


def make_points_map(agg: pd.DataFrame, out_html: Path) -> bool:
    if folium is None or agg.empty:
        return False
    center = [float(agg["Latitude"].median()), float(agg["Longitude"].median())]
    m = folium.Map(location=center, zoom_start=16, tiles=DEFAULT_MAP_TILES)
    for _, row in agg.sort_values("BaseName").iterrows():
        popup = "<br>".join(
            [
                f"<b>Punt:</b> {html.escape(str(row['BaseName']))}",
                f"<b>Kwaliteit:</b> {html.escape(str(row['Quality']))}",
                f"<b>Aantal:</b> {int(row['N'])}",
                f"<b>X:</b> {float(row['X']):.4f}",
                f"<b>Y:</b> {float(row['Y']):.4f}",
                f"<b>Z:</b> {float(row['Z']):.4f}",
                f"<b>SigmaXY:</b> {float(row['SigmaXY']):.4f}",
                f"<b>SigmaZ:</b> {float(row['SigmaZ']):.4f}",
                f"<b>Lat:</b> {float(row['Latitude']):.8f}",
                f"<b>Lon:</b> {float(row['Longitude']):.8f}",
            ]
        )
        folium.CircleMarker(
            location=[float(row["Latitude"]), float(row["Longitude"])],
            radius=7,
            color=color_for_quality(str(row["Quality"])),
            fill=True,
            fill_color=color_for_quality(str(row["Quality"])),
            fill_opacity=0.9,
            tooltip=f"{row['BaseName']} | kwaliteit {row['Quality']}",
            popup=folium.Popup(popup, max_width=420),
        ).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return True


def render_quality_plot(ax, df: pd.DataFrame, opts: ProcessOptions) -> list[dict[str, object]]:
    ax.clear()
    if df.empty:
        ax.set_title("Kwaliteit")
        return []
    colors = df["Quality"].map({"A": "#1a9850", "B": "#fee08b", "C": "#d73027", "?": "#7f7f7f"}).fillna("#7f7f7f")
    ax.scatter(df["SigmaXY"], df["SigmaZ"], c=colors, s=28, edgecolors="none")
    ax.axvline(opts.sigma_xy_a, color="#1a9850", linestyle="--", linewidth=1)
    ax.axhline(opts.sigma_z_a, color="#1a9850", linestyle="--", linewidth=1)
    ax.axvline(opts.sigma_xy_b, color="#d95f02", linestyle="--", linewidth=1)
    ax.axhline(opts.sigma_z_b, color="#d95f02", linestyle="--", linewidth=1)
    if opts.max_sigma_xy is not None:
        ax.axvline(opts.max_sigma_xy, color="#2c7fb8", linestyle=":", linewidth=1)
    if opts.max_sigma_z is not None:
        ax.axhline(opts.max_sigma_z, color="#2c7fb8", linestyle=":", linewidth=1)
    ax.set_xlabel("Sigma XY (m)")
    ax.set_ylabel("Sigma Z (m)")
    ax.set_title("RTK kwaliteit")
    ax.grid(alpha=0.25)
    hover_points: list[dict[str, object]] = []
    for idx, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        hover_points.append(
            {
                "ax": ax,
                "x": float(row["SigmaXY"]),
                "y": float(row["SigmaZ"]),
                "text": hover_text_for_measurement(row, idx),
            }
        )
    return hover_points


def render_z_plot(ax, df: pd.DataFrame) -> list[dict[str, object]]:
    ax.clear()
    ax.set_title("Z stabiliteit")
    if df.empty:
        return []
    hover_points: list[dict[str, object]] = []
    for base_name, group in df.groupby("BaseName"):
        g = group.sort_values(["Datetime", "_RowOrder"], na_position="last")
        if g["Datetime"].notna().any():
            x = g["Datetime"]
            ax.set_xlabel("Tijd")
            x_hover = mdates.date2num(pd.to_datetime(x).to_numpy())
        else:
            x = np.arange(1, len(g) + 1)
            ax.set_xlabel("Meting")
            x_hover = np.asarray(x, dtype=float)
        ax.plot(x, g["Z"], marker="o", linewidth=1.2, label=base_name)
        for local_idx, ((_, row), x_val) in enumerate(zip(g.iterrows(), x_hover)):
            hover_points.append(
                {
                    "ax": ax,
                    "x": float(x_val),
                    "y": float(row["Z"]),
                    "text": hover_text_for_measurement(row, local_idx),
                }
            )
    ax.set_ylabel("Z (m)")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    if df["BaseName"].nunique() <= 12:
        ax.legend(loc="best", fontsize=8)
    return hover_points


def save_quality_plot(df: pd.DataFrame, out_path: Path, opts: ProcessOptions) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    render_quality_plot(ax, df, opts)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_z_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    render_z_plot(ax, df)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_sigma_plot(agg: pd.DataFrame, out_path: Path) -> None:
    if agg.empty:
        return
    plt.figure(figsize=(10, 5))
    agg_sorted = agg.sort_values("SigmaZ")
    plt.plot(agg_sorted["BaseName"], agg_sorted["SigmaZ"], marker="o")
    plt.xticks(rotation=90)
    plt.ylabel("Sigma Z (m)")
    plt.title("Sigma Z per punt")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def dae_material() -> str:
    return """
  <library_effects>
    <effect id="track-effect"><profile_COMMON><technique sid="common"><lambert>
      <ambient><color>0.20 0.12 0.04 1</color></ambient>
      <diffuse><color>0.96 0.49 0.08 1</color></diffuse>
    </lambert></technique></profile_COMMON></effect>
  </library_effects>
  <library_materials>
    <material id="track-material" name="track-material"><instance_effect url="#track-effect"/></material>
  </library_materials>
"""


def write_dae(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray], out_dae: Path, opts: DaeOpts) -> tuple[float, float, float]:
    step = max(1, int(opts.every))
    x, y = x[::step], y[::step]
    if z is not None:
        z = z[::step]
    x0, y0 = float(x[0]), float(y[0])
    z0 = float(z[0]) if z is not None else 0.0
    rel_x, rel_y = x - x0, y - y0
    rel_z = (z - z0) if (opts.use_h and z is not None) else np.zeros_like(rel_x)
    s2 = float(opts.cube) / 2.0
    cube = np.array(
        [[-s2, -s2, -s2], [s2, -s2, -s2], [s2, s2, -s2], [-s2, s2, -s2], [-s2, -s2, s2], [s2, -s2, s2], [s2, s2, s2], [-s2, s2, s2]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=int)
    verts, tris, off = [], [], 0
    for xi, yi, zi in zip(rel_x, rel_y, rel_z):
        verts.append(cube + np.array([xi, yi, zi], dtype=float))
        tris.append(faces + off)
        off += 8
    verts = np.vstack(verts) if verts else np.zeros((0, 3), dtype=float)
    tris = np.vstack(tris) if tris else np.zeros((0, 3), dtype=int)
    if opts.ribbon and len(rel_x) >= 2:
        rverts, rtris = [], []
        halfw = float(opts.ribbon_w) / 2.0
        for i in range(len(rel_x) - 1):
            p0 = np.array([rel_x[i], rel_y[i], rel_z[i]])
            p1 = np.array([rel_x[i + 1], rel_y[i + 1], rel_z[i + 1]])
            dx, dy = (p1 - p0)[0], (p1 - p0)[1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            offv = np.array([-dy / length * halfw, dx / length * halfw, 0.0], dtype=float)
            rverts.append(np.vstack([p0 - offv, p0 + offv, p1 + offv, p1 - offv]))
            rtris.append(np.array([[0, 1, 2], [0, 2, 3]], dtype=int) + verts.shape[0] + len(rverts[:-1]) * 4)
        if rverts:
            verts = np.vstack([verts, np.vstack(rverts)])
            tris = np.vstack([tris, np.vstack(rtris)])
    pos = " ".join(f"{v:.6f}" for v in verts.reshape(-1))
    tri = " ".join(" ".join(str(int(v)) for v in t) for t in tris)
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><contributor><authoring_tool>rtk_combined_quality_map_gui.py</authoring_tool></contributor><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
{dae_material()}
  <library_geometries><geometry id="trackGeom" name="Track"><mesh>
    <source id="trackGeom-positions"><float_array id="trackGeom-positions-array" count="{verts.size}">{pos}</float_array><technique_common><accessor source="#trackGeom-positions-array" count="{verts.shape[0]}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
    <vertices id="trackGeom-vertices"><input semantic="POSITION" source="#trackGeom-positions"/></vertices>
    <triangles material="track-symbol" count="{tris.shape[0]}"><input semantic="VERTEX" source="#trackGeom-vertices" offset="0"/><p>{tri}</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="Scene" name="Scene"><node id="TrackNode" name="TrackNode"><instance_geometry url="#trackGeom"><bind_material><technique_common><instance_material symbol="track-symbol" target="#track-material"/></technique_common></bind_material></instance_geometry></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
"""
    out_dae.parent.mkdir(parents=True, exist_ok=True)
    out_dae.write_text(xml, encoding="utf-8")
    return x0, y0, z0


def cdata(text: str) -> str:
    return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def kml_desc(row: pd.Series, idx: int) -> str:
    return cdata(popup_html_track(row, idx))


def point_coord(row: pd.Series, use_h: bool) -> str:
    alt = float(row["Z"]) if use_h and not pd.isna(row.get("Z", np.nan)) else 0.0
    return f"{float(row['Longitude']):.10f},{float(row['Latitude']):.10f},{alt:.3f}"


def write_kml(df: pd.DataFrame, dae_name: str, out_kml: Path, every: int, use_h: bool, lift: float) -> None:
    line = " ".join(point_coord(row, use_h) for _, row in df.iterrows())
    points = []
    for idx, (_, row) in enumerate(df.iloc[:: max(1, int(every))].iterrows()):
        real = idx * max(1, int(every))
        points.append(
            f"""
    <Placemark><name>{html.escape(label_for_track(row, real))}</name><description>{kml_desc(row, real)}</description>
      <Point><altitudeMode>{"absolute" if use_h else "clampToGround"}</altitudeMode><coordinates>{point_coord(row, use_h)}</coordinates></Point>
    </Placemark>"""
        )
    first = df.iloc[0]
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Style id="track-line"><LineStyle><color>ff1478ff</color><width>4</width></LineStyle></Style>
  <Folder><name>Tracklijn</name><Placemark><name>Track</name><styleUrl>#track-line</styleUrl>
    <LineString><tessellate>1</tessellate><altitudeMode>{"absolute" if use_h else "clampToGround"}</altitudeMode><coordinates>{line}</coordinates></LineString>
  </Placemark></Folder>
  <Folder><name>Punten</name>{''.join(points)}</Folder>
  <Folder><name>Model</name><Placemark><name>DAE-model</name>
    <Model><altitudeMode>relativeToGround</altitudeMode><Location><longitude>{float(first['Longitude']):.10f}</longitude><latitude>{float(first['Latitude']):.10f}</latitude><altitude>{float(lift):.3f}</altitude></Location>
    <Orientation><heading>0</heading><tilt>0</tilt><roll>0</roll></Orientation><Scale><x>1</x><y>1</y><z>1</z></Scale><Link><href>{html.escape(dae_name)}</href></Link></Model>
  </Placemark></Folder>
</Document></kml>"""
    out_kml.parent.mkdir(parents=True, exist_ok=True)
    out_kml.write_text(kml, encoding="utf-8")


def export_kmz(kmz_path: Path, kml_path: Path, dae_path: Path) -> None:
    kmz_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(kml_path, arcname="doc.kml")
        zf.write(dae_path, arcname=dae_path.name)


def paths_for(base: Path) -> dict[str, Path]:
    parent = base.parent if str(base.parent) != "" else Path(".")
    stem = base.name
    return {
        "raw_csv": parent / f"{stem}_raw_with_quality.csv",
        "filtered_csv": parent / f"{stem}_filtered.csv",
        "agg_csv": parent / f"{stem}_aggregated_with_WGS84.csv",
        "quality_png": parent / f"{stem}_quality.png",
        "z_png": parent / f"{stem}_z_stability.png",
        "sigma_png": parent / f"{stem}_sigmaZ.png",
        "track_map_html": parent / f"{stem}_track_map.html",
        "points_map_html": parent / f"{stem}_quality_points.html",
        "dae": parent / f"{stem}_track.dae",
        "kml": parent / f"{stem}_track.kml",
        "kmz": parent / f"{stem}_track.kmz",
        "summary": parent / f"{stem}_summary.txt",
    }


def analyse_dataframe(df_in: pd.DataFrame, opts: ProcessOptions) -> dict[str, object]:
    df = df_in.copy()
    df["_RowOrder"] = np.arange(len(df))
    df["PointName"] = df[opts.name_col].map(s) if opts.name_col else ""
    df["GroupName"] = df[opts.group_col].map(s) if opts.group_col else ""
    df["X"] = numeric(df[opts.x_col])
    df["Y"] = numeric(df[opts.y_col])
    df["Z"] = numeric(df[opts.z_col])
    df["SigmaXY"] = numeric(df[opts.sigma_xy_col])
    df["SigmaZ"] = numeric(df[opts.sigma_z_col])
    df["Time"] = df[opts.time_col].map(s) if opts.time_col else ""
    df["Date"] = df[opts.date_col].map(s) if opts.date_col else ""
    df["Datetime"] = parse_datetime(df["Date"], df["Time"])
    df["BaseName"] = derive_base_name(df["PointName"], df["GroupName"])

    df = df.dropna(subset=["X", "Y", "Z", "SigmaXY", "SigmaZ"]).copy()
    if df.empty:
        raise ValueError("Geen geldige X, Y, Z, SigmaXY en SigmaZ waarden gevonden.")

    df["Quality"] = [quality_flag(xy, z, opts) for xy, z in zip(df["SigmaXY"], df["SigmaZ"])]
    lon, lat = project_xy(opts.crs, df["X"], df["Y"])
    df["Longitude"], df["Latitude"] = lon, lat

    raw, filtered = apply_filters(df, opts)
    if filtered.empty:
        raise ValueError("Na filtering blijven geen punten over. Versoepel de filterinstellingen.")

    filtered = filtered.sort_values(["Datetime", "_RowOrder"], na_position="last").reset_index(drop=True)
    agg = (
        filtered.groupby("BaseName", as_index=False)
        .agg(
            X=("X", "mean"),
            Y=("Y", "mean"),
            Z=("Z", "mean"),
            SigmaXY=("SigmaXY", "mean"),
            SigmaZ=("SigmaZ", "mean"),
            N=("BaseName", "size"),
            Quality=("Quality", summarise_quality),
            FirstTime=("Datetime", "min"),
            LastTime=("Datetime", "max"),
        )
    )
    agg["SigmaZ_reduced"] = agg["SigmaZ"] / np.sqrt(agg["N"].clip(lower=1))
    lon_agg, lat_agg = project_xy(opts.crs, agg["X"], agg["Y"])
    agg["Longitude"], agg["Latitude"] = lon_agg, lat_agg

    quality_counts = raw["Quality"].value_counts().sort_index()
    summary_lines = [
        f"Input rijen: {len(df_in)}",
        f"Geldige meetrijen: {len(raw)}",
        f"Na filtering: {len(filtered)}",
        f"Geaggregeerde punten: {len(agg)}",
        f"Kwaliteitsverdeling: {quality_counts.to_dict()}",
        f"Initiale filter OK: {int(raw['PassInitialFilter'].sum())}",
        f"Outlier filter toegepast: {'ja' if opts.apply_outlier_filter else 'nee'}",
        f"CRS naar WGS84: {opts.crs} -> EPSG:4326",
    ]
    return {"raw": raw, "filtered": filtered, "agg": agg, "summary_lines": summary_lines}


def write_analysis_outputs(results: dict[str, object], opts: ProcessOptions) -> dict[str, Path]:
    raw = results["raw"]
    filtered = results["filtered"]
    agg = results["agg"]
    paths = paths_for(opts.output_base)

    raw.to_csv(paths["raw_csv"], sep=";", decimal=",", index=False)
    filtered.to_csv(paths["filtered_csv"], sep=";", decimal=",", index=False)
    agg.to_csv(paths["agg_csv"], sep=";", decimal=",", index=False)

    save_quality_plot(raw, paths["quality_png"], opts)
    save_z_plot(filtered, paths["z_png"])
    save_sigma_plot(agg, paths["sigma_png"])

    track_map_ok = make_track_map(filtered, paths["track_map_html"], opts.map_every)
    points_map_ok = make_points_map(agg, paths["points_map_html"])

    summary_lines = list(results["summary_lines"])
    summary_lines.append(f"Trackkaart aangemaakt: {'ja' if track_map_ok else 'nee'}")
    summary_lines.append(f"Puntenkaart aangemaakt: {'ja' if points_map_ok else 'nee'}")
    paths["summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return paths


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RTK gecombineerde analyse + WGS84/DAE export")
        self.geometry("1450x980")
        self.df: Optional[pd.DataFrame] = None
        self.results: Optional[dict[str, object]] = None

        self.v_in = tk.StringVar()
        self.v_out = tk.StringVar()
        self.v_delim = tk.StringVar(value="auto")
        self.v_header = tk.StringVar(value="auto")
        self.v_crs = tk.StringVar(value="EPSG:31370")

        self.v_name = tk.StringVar(value=NONE)
        self.v_group = tk.StringVar(value=NONE)
        self.v_x = tk.StringVar()
        self.v_y = tk.StringVar()
        self.v_z = tk.StringVar()
        self.v_sigma_xy = tk.StringVar()
        self.v_sigma_z = tk.StringVar()
        self.v_time = tk.StringVar(value=NONE)
        self.v_date = tk.StringVar(value=NONE)

        self.v_sigma_xy_a = tk.StringVar(value=f"{DEFAULT_SIGMA_XY_A:.4f}")
        self.v_sigma_z_a = tk.StringVar(value=f"{DEFAULT_SIGMA_Z_A:.4f}")
        self.v_sigma_xy_b = tk.StringVar(value=f"{DEFAULT_SIGMA_XY_B:.4f}")
        self.v_sigma_z_b = tk.StringVar(value=f"{DEFAULT_SIGMA_Z_B:.4f}")
        self.v_outlier_sigma = tk.StringVar(value=f"{DEFAULT_OUTLIER_SIGMA:.2f}")
        self.v_worst_quality = tk.StringVar(value="C")
        self.v_max_sigma_xy = tk.StringVar(value="")
        self.v_max_sigma_z = tk.StringVar(value="")
        self.v_apply_outlier = tk.BooleanVar(value=True)

        self.v_map_every = tk.IntVar(value=1)
        self.v_dae_every = tk.IntVar(value=1)
        self.v_cube = tk.DoubleVar(value=0.75)
        self.v_ribbon = tk.BooleanVar(value=True)
        self.v_ribbon_w = tk.DoubleVar(value=0.30)
        self.v_use_h = tk.BooleanVar(value=False)
        self.v_lift = tk.DoubleVar(value=1.5)
        self._hover_points: list[dict[str, object]] = []
        self._hover_annotations: dict[object, object] = {}

        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(8, weight=1)
        root.grid_rowconfigure(9, weight=1)

        ttk.Label(root, text="Inputbestand").grid(row=0, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.v_in, width=120).grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(root, text="Bladeren...", command=self._pick_in).grid(row=1, column=1, sticky="w")

        ttk.Label(root, text="Output basisnaam").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(root, textvariable=self.v_out, width=120).grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(root, text="Kies...", command=self._pick_out).grid(row=3, column=1, sticky="w")

        read_box = ttk.LabelFrame(root, text="Inlezen", padding=10)
        read_box.grid(row=4, column=0, columnspan=2, sticky="we", pady=(12, 0))
        ttk.Label(read_box, text="Delimiter:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(read_box, textvariable=self.v_delim, values=["auto", ";", ",", "\\t", "|"], width=8, state="readonly").grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(read_box, text="Header:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        ttk.Combobox(read_box, textvariable=self.v_header, values=["auto", "yes", "no"], width=8, state="readonly").grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(read_box, text="Input CRS:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(read_box, textvariable=self.v_crs, values=["EPSG:31370", "EPSG:25831", "EPSG:32631", "EPSG:4326"], width=18, state="readonly").grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Button(read_box, text="Lees en detecteer kolommen", command=self._load).grid(row=2, column=0, sticky="w", pady=(10, 0))

        cols = ttk.LabelFrame(root, text="Kolommen", padding=10)
        cols.grid(row=5, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(cols, text="Puntnaam:").grid(row=0, column=0, sticky="w")
        ttk.Label(cols, text="Setup/groep:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Label(cols, text="X / Easting:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(cols, text="Y / Northing:").grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Label(cols, text="Z / Hoogte:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(cols, text="SigmaXY:").grid(row=2, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Label(cols, text="SigmaZ:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Label(cols, text="Tijd:").grid(row=3, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Label(cols, text="Datum:").grid(row=4, column=0, sticky="w", pady=(8, 0))

        self.cb_name = ttk.Combobox(cols, textvariable=self.v_name, values=[NONE], width=28, state="readonly")
        self.cb_group = ttk.Combobox(cols, textvariable=self.v_group, values=[NONE], width=28, state="readonly")
        self.cb_x = ttk.Combobox(cols, textvariable=self.v_x, values=[], width=28, state="readonly")
        self.cb_y = ttk.Combobox(cols, textvariable=self.v_y, values=[], width=28, state="readonly")
        self.cb_z = ttk.Combobox(cols, textvariable=self.v_z, values=[], width=28, state="readonly")
        self.cb_sigma_xy = ttk.Combobox(cols, textvariable=self.v_sigma_xy, values=[], width=28, state="readonly")
        self.cb_sigma_z = ttk.Combobox(cols, textvariable=self.v_sigma_z, values=[], width=28, state="readonly")
        self.cb_time = ttk.Combobox(cols, textvariable=self.v_time, values=[NONE], width=28, state="readonly")
        self.cb_date = ttk.Combobox(cols, textvariable=self.v_date, values=[NONE], width=28, state="readonly")

        self.cb_name.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.cb_group.grid(row=0, column=3, sticky="w", padx=(6, 0))
        self.cb_x.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_y.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_z.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_sigma_xy.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_sigma_z.grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_time.grid(row=3, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_date.grid(row=4, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        quality = ttk.LabelFrame(root, text="Kwaliteit en filtering", padding=10)
        quality.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(quality, text="A: max SigmaXY").grid(row=0, column=0, sticky="w")
        ttk.Entry(quality, textvariable=self.v_sigma_xy_a, width=10).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(quality, text="A: max SigmaZ").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Entry(quality, textvariable=self.v_sigma_z_a, width=10).grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(quality, text="B: max SigmaXY").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_sigma_xy_b, width=10).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="B: max SigmaZ").grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_sigma_z_b, width=10).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="Slechtste toegestane klasse").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(quality, textvariable=self.v_worst_quality, values=["A", "B", "C"], width=8, state="readonly").grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="Max SigmaXY filter").grid(row=2, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_max_sigma_xy, width=10).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="Max SigmaZ filter").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_max_sigma_z, width=10).grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="Outlier sigma op Z").grid(row=3, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_outlier_sigma, width=10).grid(row=3, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Checkbutton(quality, text="Outlierfilter op Z toepassen", variable=self.v_apply_outlier).grid(row=4, column=0, sticky="w", pady=(10, 0))

        export = ttk.LabelFrame(root, text="Kaart en DAE/KMZ", padding=10)
        export.grid(row=7, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(export, text="Kaart: toon elke n punten").grid(row=0, column=0, sticky="w")
        ttk.Entry(export, textvariable=self.v_map_every, width=8).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(export, text="DAE sample elke n punten").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Entry(export, textvariable=self.v_dae_every, width=8).grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(export, text="Cube size (m)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(export, textvariable=self.v_cube, width=8).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Checkbutton(export, text="Pad-ribbon toevoegen", variable=self.v_ribbon).grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Label(export, text="Ribbon breedte (m)").grid(row=1, column=3, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Entry(export, textvariable=self.v_ribbon_w, width=8).grid(row=1, column=4, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Checkbutton(export, text="Gebruik hoogte in model-Z", variable=self.v_use_h).grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(export, text="Model boven maaiveld (m)").grid(row=2, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Entry(export, textvariable=self.v_lift, width=8).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        actions = ttk.Frame(root)
        actions.grid(row=8, column=0, columnspan=2, sticky="we", pady=(12, 0))
        ttk.Button(actions, text="Preview analyse", command=self._preview).pack(side="left")
        ttk.Button(actions, text="Export rapport + kaarten", command=self._export_all).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Export DAE", command=self._export_dae).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Export KMZ", command=self._export_kmz).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Sluiten", command=self.destroy).pack(side="right")

        lower = ttk.Panedwindow(root, orient="horizontal")
        lower.grid(row=9, column=0, columnspan=2, sticky="nsew", pady=(12, 0))

        plot_wrap = ttk.Frame(lower)
        lower.add(plot_wrap, weight=3)
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax_quality = self.fig.add_subplot(121)
        self.ax_z = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_wrap)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
        self._reset_hover_annotations()

        log_wrap = ttk.Frame(lower)
        lower.add(log_wrap, weight=2)
        self.txt = tk.Text(log_wrap, height=20, wrap="word")
        self.txt.pack(fill="both", expand=True)

    def _log(self, msg: str) -> None:
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _pick_in(self) -> None:
        p = filedialog.askopenfilename(title="Selecteer RTK TXT/CSV", filetypes=[("Text/CSV", "*.txt *.csv *.dat *.log"), ("All files", "*.*")])
        if p:
            self.v_in.set(p)
            if not self.v_out.get():
                self.v_out.set(str(Path(p).with_suffix("")) + "_combined")

    def _pick_out(self) -> None:
        p = filedialog.asksaveasfilename(title="Kies output basisnaam", defaultextension="")
        if p:
            self.v_out.set(str(Path(p).with_suffix("")))

    def _sel(self, val: str) -> Optional[str]:
        val = val.strip()
        return None if not val or val == NONE else val

    def _parse_optional_float(self, value: str) -> Optional[float]:
        value = value.strip().replace(",", ".")
        return None if not value else float(value)

    def _hide_hover_annotations(self) -> None:
        changed = False
        for ann in self._hover_annotations.values():
            if ann.get_visible():
                ann.set_visible(False)
                changed = True
        if changed:
            self.canvas.draw_idle()

    def _reset_hover_annotations(self) -> None:
        self._hover_annotations = {}
        for ax in (self.ax_quality, self.ax_z):
            ann = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(14, 14),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.4", "fc": "#fffdf8", "ec": "#bca27f", "alpha": 0.96},
                arrowprops={"arrowstyle": "->", "color": "#8f3f00"},
            )
            ann.set_visible(False)
            self._hover_annotations[ax] = ann

    def _nearest_hover_point(self, event) -> Optional[dict[str, object]]:
        if event.inaxes is None or event.x is None or event.y is None:
            return None
        best: Optional[dict[str, object]] = None
        best_dist = 12.0
        for item in self._hover_points:
            if item["ax"] is not event.inaxes:
                continue
            x_disp, y_disp = event.inaxes.transData.transform((item["x"], item["y"]))
            dist = math.hypot(event.x - x_disp, event.y - y_disp)
            if dist <= best_dist:
                best = item
                best_dist = dist
        return best

    def _on_plot_hover(self, event) -> None:
        target = self._nearest_hover_point(event)
        if target is None:
            self._hide_hover_annotations()
            return
        changed = False
        for ax, ann in self._hover_annotations.items():
            if ax is target["ax"]:
                ann.xy = (target["x"], target["y"])
                ann.set_text(str(target["text"]))
                if not ann.get_visible():
                    ann.set_visible(True)
                changed = True
            elif ann.get_visible():
                ann.set_visible(False)
                changed = True
        if changed:
            self.canvas.draw_idle()

    def _build_opts(self) -> ProcessOptions:
        out = self.v_out.get().strip()
        if not out:
            raise ValueError("Kies eerst een output basisnaam.")
        x_col = self._sel(self.v_x.get())
        y_col = self._sel(self.v_y.get())
        z_col = self._sel(self.v_z.get())
        sigma_xy_col = self._sel(self.v_sigma_xy.get())
        sigma_z_col = self._sel(self.v_sigma_z.get())
        if not x_col or not y_col or not z_col or not sigma_xy_col or not sigma_z_col:
            raise ValueError("Selecteer kolommen voor X, Y, Z, SigmaXY en SigmaZ.")
        return ProcessOptions(
            output_base=Path(out),
            crs=self.v_crs.get().strip(),
            name_col=self._sel(self.v_name.get()),
            group_col=self._sel(self.v_group.get()),
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            sigma_xy_col=sigma_xy_col,
            sigma_z_col=sigma_z_col,
            time_col=self._sel(self.v_time.get()),
            date_col=self._sel(self.v_date.get()),
            sigma_xy_a=float(self.v_sigma_xy_a.get().replace(",", ".")),
            sigma_z_a=float(self.v_sigma_z_a.get().replace(",", ".")),
            sigma_xy_b=float(self.v_sigma_xy_b.get().replace(",", ".")),
            sigma_z_b=float(self.v_sigma_z_b.get().replace(",", ".")),
            outlier_sigma=float(self.v_outlier_sigma.get().replace(",", ".")),
            worst_allowed_quality=self.v_worst_quality.get().strip() or "C",
            max_sigma_xy=self._parse_optional_float(self.v_max_sigma_xy.get()),
            max_sigma_z=self._parse_optional_float(self.v_max_sigma_z.get()),
            apply_outlier_filter=bool(self.v_apply_outlier.get()),
            map_every=max(1, int(self.v_map_every.get())),
            dae_every=max(1, int(self.v_dae_every.get())),
            cube=float(self.v_cube.get()),
            ribbon=bool(self.v_ribbon.get()),
            ribbon_w=float(self.v_ribbon_w.get()),
            use_h=bool(self.v_use_h.get()),
            lift=float(self.v_lift.get()),
        )

    def _load(self) -> None:
        inp = self.v_in.get().strip()
        if not inp:
            messagebox.showerror("Fout", "Kies eerst een inputbestand.")
            return
        delim = "\t" if self.v_delim.get() == "\\t" else self.v_delim.get()
        try:
            self._log("Lezen bestand...")
            self.df, has_header, used = read_table(inp, "auto" if self.v_delim.get() == "auto" else delim, self.v_header.get().strip())
            self.results = None
            self._log(f"  Rijen: {len(self.df)}, kolommen: {len(self.df.columns)}")
            self._log(f"  Delimiter: {repr(used)} | Header: {'ja' if has_header else 'nee'}")
            self._log(f"  Kolommen: {list(self.df.columns)}")
            g = guess_cols(self.df, has_header)
            values = list(self.df.columns)
            meta_values = [NONE] + values
            for cb, vals in (
                (self.cb_name, meta_values),
                (self.cb_group, meta_values),
                (self.cb_x, values),
                (self.cb_y, values),
                (self.cb_z, values),
                (self.cb_sigma_xy, values),
                (self.cb_sigma_z, values),
                (self.cb_time, meta_values),
                (self.cb_date, meta_values),
            ):
                cb["values"] = vals
            self.v_name.set(g.name or NONE)
            self.v_group.set(g.group or NONE)
            self.v_x.set(g.x or "")
            self.v_y.set(g.y or "")
            self.v_z.set(g.z or "")
            self.v_sigma_xy.set(g.sigma_xy or "")
            self.v_sigma_z.set(g.sigma_z or "")
            self.v_time.set(g.time or NONE)
            self.v_date.set(g.date or NONE)
            self._log(f"  Profiel: {g.profile}")
            if g.note:
                self._log(f"  {g.note}")
            if len(self.df):
                self._log("  Eerste rij: " + " | ".join(f"{c}={s(self.df.iloc[0][c])}" for c in self.df.columns[:9]))
            messagebox.showinfo("OK", "Bestand ingelezen. Controleer de kolommen en klik daarna op 'Preview analyse'.")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc))
            self._log(f"ERROR: {exc}")

    def _run_analysis(self) -> tuple[dict[str, object], ProcessOptions]:
        if self.df is None:
            self._load()
            if self.df is None:
                raise ValueError("Geen inputdata beschikbaar.")
        opts = self._build_opts()
        results = analyse_dataframe(self.df, opts)
        self.results = results
        return results, opts

    def _preview(self) -> None:
        try:
            results, opts = self._run_analysis()
            self._hover_points = []
            self._hover_points.extend(render_quality_plot(self.ax_quality, results["raw"], opts))
            self._hover_points.extend(render_z_plot(self.ax_z, results["filtered"]))
            self._reset_hover_annotations()
            self.fig.tight_layout()
            self.canvas.draw()
            for line in results["summary_lines"]:
                self._log(line)
            self._log(f"Preview klaar. Punten in track: {len(results['filtered'])} | unieke basispunten: {len(results['agg'])}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc))
            self._log(f"ERROR: {exc}")

    def _export_all(self) -> None:
        try:
            results, opts = self._run_analysis()
            paths = write_analysis_outputs(results, opts)
            self._log("Bestanden weggeschreven:")
            for key in ("raw_csv", "filtered_csv", "agg_csv", "quality_png", "z_png", "sigma_png", "track_map_html", "points_map_html", "summary"):
                self._log(f"  {key}: {paths[key]}")
            messagebox.showinfo("OK", f"Analysebestanden opgeslagen:\n{paths['summary']}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc))
            self._log(f"ERROR: {exc}")

    def _export_dae(self) -> None:
        try:
            results, opts = self._run_analysis()
            filtered = results["filtered"]
            paths = paths_for(opts.output_base)
            z = filtered["Z"].to_numpy(dtype=float) if not filtered["Z"].isna().all() else None
            e0, n0, h0 = write_dae(filtered["X"].to_numpy(dtype=float), filtered["Y"].to_numpy(dtype=float), z, paths["dae"], DaeOpts(opts.cube, opts.ribbon, opts.ribbon_w, opts.use_h, opts.dae_every))
            self._log(f"DAE opgeslagen: {paths['dae']}")
            self._log(f"  Origin input CRS: X0={e0:.3f}, Y0={n0:.3f}, Z0={h0:.3f}")
            messagebox.showinfo("OK", f"DAE opgeslagen:\n{paths['dae']}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc))
            self._log(f"ERROR: {exc}")

    def _export_kmz(self) -> None:
        try:
            results, opts = self._run_analysis()
            filtered = results["filtered"]
            paths = paths_for(opts.output_base)
            z = filtered["Z"].to_numpy(dtype=float) if not filtered["Z"].isna().all() else None
            write_dae(filtered["X"].to_numpy(dtype=float), filtered["Y"].to_numpy(dtype=float), z, paths["dae"], DaeOpts(opts.cube, opts.ribbon, opts.ribbon_w, opts.use_h, opts.dae_every))
            write_kml(filtered, paths["dae"].name, paths["kml"], opts.map_every, opts.use_h, opts.lift)
            export_kmz(paths["kmz"], paths["kml"], paths["dae"])
            self._log(f"KML opgeslagen: {paths['kml']}")
            self._log(f"KMZ opgeslagen: {paths['kmz']}")
            messagebox.showinfo("OK", f"KML en KMZ opgeslagen:\n{paths['kml']}\n{paths['kmz']}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc))
            self._log(f"ERROR: {exc}")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
