#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QRev / SonTek MAT -> meerdere tracks + dieptegemiddelde vectoren op 1 kaart (georef).

Wat dit script doet:
- Laat je meerdere .mat bestanden kiezen.
- Berekent per meetpunt de dieptegemiddelde snelheidsvector (u, v, snelheid, richting).
- Zet alle tracks samen op 1 absolute kaart (lon/lat) in 1 interactieve HTML.
- Kaartstijl kan in de HTML geschakeld worden: OpenStreetMap / Carto Positron / Carto Dark.
"""

from __future__ import annotations

import os
import numpy as np

from scipy.io import loadmat

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


def load_qrev_mat(path: str):
    return loadmat(path, struct_as_record=False, squeeze_me=True)


def _as_float_1d(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(x, dtype=float).ravel()
    except Exception:
        try:
            return np.asarray(np.squeeze(x), dtype=float).ravel()
        except Exception:
            return np.array([], dtype=float)


def _align_to_len(arr, n: int) -> np.ndarray:
    out = np.full(int(n), np.nan, dtype=float)
    a = _as_float_1d(arr)
    if a.size == 0:
        return out
    m = min(int(n), int(a.size))
    out[:m] = a[:m]
    return out


def _scalar_from_field(x, default=np.nan) -> float:
    arr = _as_float_1d(x)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanmedian(arr))


def compute_cell_depths(system, setup, n_cells: int) -> np.ndarray:
    cell_size = np.nan
    for cand in ["CellSize", "Cell_Size", "WT_CellSize", "Cell_Sz"]:
        if hasattr(system, cand):
            cell_size = _scalar_from_field(getattr(system, cand), default=np.nan)
            if np.isfinite(cell_size) and cell_size > 0:
                break
    if not (np.isfinite(cell_size) and cell_size > 0) and hasattr(setup, "WT_CellSize"):
        cell_size = _scalar_from_field(getattr(setup, "WT_CellSize"), default=np.nan)
    if not (np.isfinite(cell_size) and cell_size > 0):
        cell_size = 0.1

    cell_start = np.nan
    for cand in ["Cell_Start", "CellStart", "Cell_Start_m", "CellStart_m"]:
        if hasattr(system, cand):
            cell_start = _scalar_from_field(getattr(system, cand), default=np.nan)
            if np.isfinite(cell_start):
                break
    if not np.isfinite(cell_start):
        cell_start = 0.0

    i = np.arange(int(n_cells), dtype=float)
    return cell_start + (i + 0.5) * cell_size


def _standardize_velocity(velocity, n_ens_hint: int | None = None) -> np.ndarray:
    """
    Breng velocity naar shape (n_cells, n_comp, n_ens).
    Verwachte component-as bevat minstens 2 componenten (east/north).
    """
    arr = np.asarray(np.squeeze(velocity), dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Onverwachte Velocity ndim: {arr.ndim}, shape: {arr.shape}")

    best = None
    best_score = -1
    axes = [0, 1, 2]
    for comp_axis in axes:
        if arr.shape[comp_axis] < 2:
            continue
        for ens_axis in axes:
            if ens_axis == comp_axis:
                continue
            cell_axis = ({0, 1, 2} - {comp_axis, ens_axis}).pop()
            cand = np.moveaxis(arr, [cell_axis, comp_axis, ens_axis], [0, 1, 2])
            score = 0
            if cand.shape[1] == 4:
                score += 2
            if n_ens_hint is not None and cand.shape[2] == int(n_ens_hint):
                score += 3
            if cand.shape[0] > 3:
                score += 1
            if score > best_score:
                best_score = score
                best = cand

    if best is None or best.shape[1] < 2:
        raise ValueError(f"Kon Velocity niet standaardiseren, shape={arr.shape}")

    return best


def compute_depth_mean_vectors(md: dict):
    try:
        system = md["System"]
        setup = md["Setup"]
        water_track = md["WaterTrack"]
        bottom_track = md["BottomTrack"]
    except Exception as e:
        raise RuntimeError("Vereiste velden ontbreken (System/Setup/WaterTrack/BottomTrack).") from e

    vb_depth_raw = _as_float_1d(getattr(bottom_track, "VB_Depth", None))
    n_hint = int(vb_depth_raw.size) if vb_depth_raw.size > 0 else None
    vel = _standardize_velocity(getattr(water_track, "Velocity"), n_ens_hint=n_hint)

    east = vel[:, 0, :].astype(float)
    north = vel[:, 1, :].astype(float)

    n_cells, n_ens = east.shape
    vb_depth = _align_to_len(vb_depth_raw, n_ens)
    depths = compute_cell_depths(system, setup, n_cells).astype(float).ravel()

    east_m = east.copy()
    north_m = north.copy()
    for j in range(n_ens):
        bd = vb_depth[j]
        if np.isfinite(bd):
            mask = depths > bd
            east_m[mask, j] = np.nan
            north_m[mask, j] = np.nan

    valid_u = np.isfinite(east_m)
    valid_v = np.isfinite(north_m)
    cnt_u = np.sum(valid_u, axis=0)
    cnt_v = np.sum(valid_v, axis=0)

    u_mean = np.full(n_ens, np.nan, dtype=float)
    v_mean = np.full(n_ens, np.nan, dtype=float)
    good_u = cnt_u > 0
    good_v = cnt_v > 0
    if np.any(good_u):
        u_mean[good_u] = np.nansum(east_m[:, good_u], axis=0) / cnt_u[good_u]
    if np.any(good_v):
        v_mean[good_v] = np.nansum(north_m[:, good_v], axis=0) / cnt_v[good_v]

    spd_mean = np.sqrt(u_mean ** 2 + v_mean ** 2)
    dir_mean = (np.degrees(np.arctan2(u_mean, v_mean)) + 360.0) % 360.0
    dir_mean[~np.isfinite(spd_mean)] = np.nan

    return n_ens, u_mean, v_mean, spd_mean, dir_mean


def extract_lonlat(md: dict, n_ens: int):
    gps = md.get("GPS", None)
    if gps is None:
        return np.full(n_ens, np.nan), np.full(n_ens, np.nan)

    lon = _align_to_len(getattr(gps, "Longitude", None), n_ens)
    lat = _align_to_len(getattr(gps, "Latitude", None), n_ens)

    lon[(lon < -180.0) | (lon > 180.0)] = np.nan
    lat[(lat < -90.0) | (lat > 90.0)] = np.nan
    return lon, lat


def _uv_to_latlon_delta(lat_deg, u_ms, v_ms, meters_per_ms=25.0):
    lat = np.asarray(lat_deg, dtype=float)
    u = np.asarray(u_ms, dtype=float)
    v = np.asarray(v_ms, dtype=float)
    dy_m = v * float(meters_per_ms)
    dx_m = u * float(meters_per_ms)
    dlat = dy_m / 111320.0
    coslat = np.cos(np.radians(lat))
    coslat = np.where(np.abs(coslat) < 1e-6, np.nan, coslat)
    dlon = dx_m / (111320.0 * coslat)
    return dlon, dlat


def _estimate_zoom(lon_all, lat_all):
    lon = np.asarray(lon_all, dtype=float)
    lat = np.asarray(lat_all, dtype=float)
    m = np.isfinite(lon) & np.isfinite(lat)
    if np.sum(m) < 2:
        return 15
    dlon = float(np.nanmax(lon[m]) - np.nanmin(lon[m]))
    dlat = float(np.nanmax(lat[m]) - np.nanmin(lat[m]))
    span = max(dlon, dlat)
    if span <= 0.0005:
        return 18
    if span <= 0.001:
        return 17
    if span <= 0.002:
        return 16
    if span <= 0.005:
        return 15
    if span <= 0.01:
        return 14
    if span <= 0.02:
        return 13
    if span <= 0.05:
        return 12
    if span <= 0.1:
        return 11
    return 10


def load_track_record(mat_path: str):
    md = load_qrev_mat(mat_path)
    n_ens, u_mean, v_mean, spd_mean, dir_mean = compute_depth_mean_vectors(md)
    lon, lat = extract_lonlat(md, n_ens)
    valid = np.isfinite(lon) & np.isfinite(lat)

    return {
        "path": mat_path,
        "name": os.path.basename(mat_path),
        "lon": lon,
        "lat": lat,
        "u_mean": u_mean,
        "v_mean": v_mean,
        "spd_mean": spd_mean,
        "dir_mean": dir_mean,
        "n_valid": int(np.sum(valid)),
    }


def build_combined_map(records, vector_step=3, vector_m_per_ms=25.0, map_style="open-street-map"):
    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative
    except Exception as e:
        raise RuntimeError("Plotly ontbreekt. Installeer met: pip install plotly") from e

    step = max(1, int(vector_step))
    vec_scale = float(vector_m_per_ms)
    colors = qualitative.Plotly + qualitative.D3 + qualitative.Dark24

    fig = go.Figure()
    all_lon = []
    all_lat = []

    for i, rec in enumerate(records):
        color = colors[i % len(colors)]
        lon = np.asarray(rec["lon"], dtype=float).ravel()
        lat = np.asarray(rec["lat"], dtype=float).ravel()
        u = np.asarray(rec["u_mean"], dtype=float).ravel()
        v = np.asarray(rec["v_mean"], dtype=float).ravel()
        s = np.asarray(rec["spd_mean"], dtype=float).ravel()
        d = np.asarray(rec["dir_mean"], dtype=float).ravel()

        m = np.isfinite(lon) & np.isfinite(lat)
        if np.sum(m) < 2:
            continue

        all_lon.append(lon[m])
        all_lat.append(lat[m])

        text = [
            (
                f"{rec['name']}<br>"
                f"Punt: {k + 1}<br>"
                f"Vgem: {sp:.3f} m/s<br>"
                f"Richting: {dr:.1f} deg"
            )
            if np.isfinite(sp) and np.isfinite(dr)
            else f"{rec['name']}<br>Punt: {k + 1}<br>Vgem: NaN"
            for k, (sp, dr) in enumerate(zip(s, d))
        ]

        fig.add_trace(
            go.Scattermapbox(
                lon=lon[m],
                lat=lat[m],
                mode="lines+markers",
                marker=dict(size=6, color=color),
                line=dict(width=2, color=color),
                text=np.asarray(text, dtype=object)[m],
                hovertemplate="%{text}<extra></extra>",
                name=rec["name"],
            )
        )

        idx = np.arange(0, lon.size, step, dtype=int)
        good = (
            np.isfinite(lon[idx])
            & np.isfinite(lat[idx])
            & np.isfinite(u[idx])
            & np.isfinite(v[idx])
        )
        idx = idx[good]
        if idx.size:
            dlon, dlat = _uv_to_latlon_delta(lat[idx], u[idx], v[idx], meters_per_ms=vec_scale)
            lon2 = lon[idx] + dlon
            lat2 = lat[idx] + dlat
            seg_lon, seg_lat = [], []
            for xa, ya, xb, yb in zip(lon[idx], lat[idx], lon2, lat2):
                if np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb):
                    seg_lon.extend([xa, xb, None])
                    seg_lat.extend([ya, yb, None])
            if seg_lon:
                fig.add_trace(
                    go.Scattermapbox(
                        lon=seg_lon,
                        lat=seg_lat,
                        mode="lines",
                        line=dict(width=2, color=color),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    if not all_lon or not all_lat:
        raise RuntimeError("Geen geldige gegeorefereerde punten gevonden (lon/lat).")

    lon_all = np.concatenate(all_lon)
    lat_all = np.concatenate(all_lat)
    center_lon = float(np.nanmean(lon_all))
    center_lat = float(np.nanmean(lat_all))
    zoom = _estimate_zoom(lon_all, lat_all)

    fig.update_layout(
        title=f"Gecombineerde tracks + vectoren ({len(records)} bestanden)",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        mapbox=dict(
            style=map_style,
            center=dict(lon=center_lon, lat=center_lat),
            zoom=zoom,
        ),
    )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(label="OpenStreetMap", method="relayout", args=[{"mapbox.style": "open-street-map"}]),
                    dict(label="Carto Positron", method="relayout", args=[{"mapbox.style": "carto-positron"}]),
                    dict(label="Carto Dark", method="relayout", args=[{"mapbox.style": "carto-darkmatter"}]),
                ],
            )
        ]
    )

    return fig


def main():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "Selecteer MAT files",
        "Kies 1 of meerdere QRev/SonTek .mat bestanden.\n"
        "Het script zet alle gegeorefereerde tracks + vectoren op 1 kaart."
    )

    mat_paths = filedialog.askopenfilenames(
        title="Selecteer QRev/SonTek .mat bestanden",
        filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
    )
    if not mat_paths:
        return

    first_dir = os.path.dirname(mat_paths[0])
    out_default = os.path.join(first_dir, "combined_tracks_vectors_map.html")
    out_html = filedialog.asksaveasfilename(
        title="Bewaar gecombineerde kaart als HTML",
        defaultextension=".html",
        initialfile=os.path.basename(out_default),
        initialdir=os.path.dirname(out_default),
        filetypes=[("HTML", "*.html")],
    )
    if not out_html:
        return

    vector_step = simpledialog.askinteger(
        "Vector stap",
        "Toon elke Nde vector (1=alle punten):",
        initialvalue=3,
        minvalue=1,
        maxvalue=100,
    )
    if vector_step is None:
        vector_step = 3

    vector_scale = simpledialog.askfloat(
        "Vector schaal",
        "Meters pijllengte per 1 m/s:",
        initialvalue=25.0,
        minvalue=0.1,
        maxvalue=2000.0,
    )
    if vector_scale is None:
        vector_scale = 25.0

    style_in = simpledialog.askstring(
        "Kaartstijl",
        "Startstijl: osm / positron / dark",
        initialvalue="osm",
    )
    style_lookup = {
        "osm": "open-street-map",
        "openstreetmap": "open-street-map",
        "positron": "carto-positron",
        "dark": "carto-darkmatter",
    }
    map_style = style_lookup.get((style_in or "osm").strip().lower(), "open-street-map")

    records = []
    skipped = []

    for p in mat_paths:
        try:
            rec = load_track_record(p)
            if rec["n_valid"] < 2:
                skipped.append((p, "te weinig geldige lon/lat punten"))
                continue
            records.append(rec)
        except Exception as e:
            skipped.append((p, str(e)))

    if not records:
        msg = "Geen bruikbare bestanden gevonden.\n\n"
        if skipped:
            msg += "\n".join([f"- {os.path.basename(p)}: {err}" for p, err in skipped[:12]])
        messagebox.showerror("Fout", msg)
        return

    fig = build_combined_map(
        records,
        vector_step=vector_step,
        vector_m_per_ms=vector_scale,
        map_style=map_style,
    )
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)

    msg = (
        f"HTML opgeslagen:\n{out_html}\n\n"
        f"Ingeladen: {len(records)} bestanden\n"
        f"Overgeslagen: {len(skipped)} bestanden"
    )
    if skipped:
        msg += "\n\nEerste overslagen:\n"
        msg += "\n".join([f"- {os.path.basename(p)}: {err}" for p, err in skipped[:10]])
    messagebox.showinfo("Klaar", msg)


if __name__ == "__main__":
    main()

