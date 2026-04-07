#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
M9 / SonTek QRev (.mat) â€“ interactieve snelheidsdoorsnede + boottrack (Matplotlib)

V6 â€“ fixes op basis van jouw laatste feedback
--------------------------------------------
1) **Vert. x slider verwijderd** (zoals gevraagd).
2) **Max diepte knipt echt af**:
   - Alle cellen dieper dan "Max diepte (m)" worden **gemaskeerd (NaN)** Ã©n de y-lim wordt aangepast.
   - Dus niet enkel andere as-waarden: de doorsnede verandert effectief.
3) **Tekst overlap opgelost** door ruimere layout + kortere labels + kleinere slider-fonts.
4) **Mouse-over (hover)** op snelheidsdoorsnede:
   - toont Afstand (m), Diepte (m) en Snelheid (m/s) van de cel onder de cursor.
5) **Interactief wegschrijven**:
   - Knop **HTML** exporteert een **Plotly HTML** (zoom/pan/hover) van de huidige view.
     (vereist: `pip install plotly`)
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np

# ---- backend for echte interactie (sliders/knoppen) ----
def _force_interactive_backend():
    """
    Forceer een GUI-backend vÃ³Ã³r pyplot import.
    TkAgg is het meest robuust op Windows; fallback naar QtAgg indien beschikbaar.
    """
    try:
        import matplotlib
        current = matplotlib.get_backend().lower()
        if "tkagg" in current or "qtagg" in current:
            return
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            try:
                matplotlib.use("QtAgg", force=True)
            except Exception:
                pass
    except Exception:
        pass

_force_interactive_backend()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.ticker import FuncFormatter

# scipy loadmat
from scipy.io import loadmat

# tkinter file dialogs
import tkinter as tk
from tkinter import filedialog, messagebox


# ------------------ IO: QRev MAT laden ------------------

def load_qrev_mat(path: str):
    """Laad QRev/SonTek .mat bestand."""
    # struct_as_record=False + squeeze_me=True maakt nested structs handiger.
    md = loadmat(path, struct_as_record=False, squeeze_me=True)
    # QRev exports zetten meestal 'Site', 'System', 'WaterTrack', 'BottomTrack', 'Setup'
    return md


def _as_float_1d(x) -> np.ndarray:
    """Zet onbekend MATLAB veld om naar 1D float array (kan leeg zijn)."""
    if x is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(x).astype(float).ravel()
        return arr
    except Exception:
        try:
            arr = np.asarray(np.squeeze(x)).astype(float).ravel()
            return arr
        except Exception:
            return np.array([], dtype=float)


def _scalar_from_field(x, *, default=np.nan) -> float:
    """Pak een bruikbare scalar uit veld dat soms (1xN) array is."""
    arr = _as_float_1d(x)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanmedian(arr))


# ------------------ Afgeleiden / berekeningen ------------------

def compute_boat_track(bottom_track, system):
    """
    Benader boottrack XY door integratie van bottom-track velocities.
    QRev struct varianten:
      - BT_Vel: (n_ens, 4) of (4, n_ens)
      - Ensemble duration: System.SampleTime of System.EnsembleTime of BottomTrack.EnsDur
    Output: x, y (meter), t index
    """
    bt_vel = np.asarray(np.squeeze(bottom_track.BT_Vel))
    if bt_vel.ndim != 2:
        raise ValueError(f"Onverwachte BT_Vel ndim: {bt_vel.ndim}, shape: {bt_vel.shape}")
    # normaliseer naar (n_ens, 4)
    if bt_vel.shape[0] == 4 and bt_vel.shape[1] != 4:
        bt_vel = bt_vel.T
    if bt_vel.shape[1] < 2:
        raise ValueError(f"Onverwachte BT_Vel shape: {bt_vel.shape}")

    vx = bt_vel[:, 0].astype(float)  # east
    vy = bt_vel[:, 1].astype(float)  # north

    # dt
    dt = None
    for cand in ["SampleTime", "EnsembleTime", "EnsDur"]:
        if hasattr(system, cand):
            dt = _scalar_from_field(getattr(system, cand), default=np.nan)
            if np.isfinite(dt) and dt > 0:
                break
    if not (dt and np.isfinite(dt) and dt > 0):
        # fallback: 1s
        dt = 1.0

    # integreer â€“ eenvoudige dead-reckoning
    x = np.cumsum(np.nan_to_num(vx) * dt)
    y = np.cumsum(np.nan_to_num(vy) * dt)
    t = np.arange(x.size)
    return x, y, t


def compute_transect_distance(x, y):
    """Cumulatieve afstand langs traject uit XY (meter)."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    ds = np.sqrt(dx * dx + dy * dy)
    return np.cumsum(ds)


def _align_to_len(arr, n: int):
    """Breng 1D array op lengte n (trim of pad met NaN)."""
    a = _as_float_1d(arr)
    out = np.full(int(n), np.nan, dtype=float)
    if a.size == 0:
        return out
    m = min(int(n), int(a.size))
    out[:m] = a[:m]
    return out


def extract_absolute_track(md: dict, n_ens: int):
    """
    Zoek absolute trackcoordinaten in GPS velden.
    Return dict met optionele keys: utm_x, utm_y, lon, lat.
    """
    out = {}
    gps = md.get("GPS", None)
    if gps is None:
        return out

    # UTM: vaak (n,2) [E,N], soms getransponeerd.
    if hasattr(gps, "UTM"):
        try:
            utm = np.asarray(np.squeeze(gps.UTM)).astype(float)
            if utm.ndim == 2:
                if utm.shape[1] == 2:
                    out["utm_x"] = _align_to_len(utm[:, 0], n_ens)
                    out["utm_y"] = _align_to_len(utm[:, 1], n_ens)
                elif utm.shape[0] == 2:
                    out["utm_x"] = _align_to_len(utm[0, :], n_ens)
                    out["utm_y"] = _align_to_len(utm[1, :], n_ens)
        except Exception:
            pass

    if hasattr(gps, "Longitude") and hasattr(gps, "Latitude"):
        out["lon"] = _align_to_len(getattr(gps, "Longitude"), n_ens)
        out["lat"] = _align_to_len(getattr(gps, "Latitude"), n_ens)

    return out


def compute_cell_depths(system, setup, n_cells: int):
    """
    Bepaal cell depths (centers) in meter.
    QRev varianten: System.Cell_Start / System.CellSize / Setup.WT_CellSize etc.
    We nemen:
      depth_i = cell_start + (i + 0.5) * cell_size
    """
    # cell size
    cell_size = None
    for cand in ["CellSize", "Cell_Size", "WT_CellSize", "Cell_Sz"]:
        if hasattr(system, cand):
            cell_size = _scalar_from_field(getattr(system, cand), default=np.nan)
            if np.isfinite(cell_size) and cell_size > 0:
                break
    if not (cell_size and np.isfinite(cell_size) and cell_size > 0):
        if hasattr(setup, "WT_CellSize"):
            cell_size = _scalar_from_field(setup.WT_CellSize, default=np.nan)
    if not (cell_size and np.isfinite(cell_size) and cell_size > 0):
        cell_size = 0.1  # fallback

    # cell start (van transducer tot eerste cell center offset)
    cell_start = None
    for cand in ["Cell_Start", "CellStart", "Cell_Start_m", "CellStart_m"]:
        if hasattr(system, cand):
            cell_start = _scalar_from_field(getattr(system, cand), default=np.nan)
            if np.isfinite(cell_start):
                break
    if not (cell_start and np.isfinite(cell_start)):
        cell_start = 0.0

    i = np.arange(n_cells, dtype=float)
    depths = cell_start + (i + 0.5) * cell_size
    return depths


def compute_speed_section_and_track_vectors(water_track, bottom_track, system, setup):
    """
    Bouw snelheidsdoorsnede + dieptegemiddelde vectoren per ensemble.

    Output:
      depths (n_cells,)
      speed_masked (n_cells, n_ens)
      vb_depth (n_ens,)
      u_mean (n_ens,)  east component (m/s)
      v_mean (n_ens,)  north component (m/s)
      spd_mean (n_ens,) grootte (m/s)
      dir_mean (n_ens,) richting NAAR, graden t.o.v. Noord
    """
    vel = np.squeeze(water_track.Velocity)  # (n_cells, 4, n_ens)
    if vel.ndim != 3 or vel.shape[1] < 2:
        raise ValueError(f"Onverwachte WaterTrack.Velocity shape: {vel.shape}")

    east = vel[:, 0, :].astype(float)
    north = vel[:, 1, :].astype(float)
    speed = np.sqrt(east ** 2 + north ** 2)

    n_cells, n_ens = speed.shape
    vb_depth = np.squeeze(bottom_track.VB_Depth).astype(float).ravel()
    if vb_depth.size != n_ens:
        raise ValueError(f"VB_Depth lengte ({vb_depth.size}) != ensembles ({n_ens})")

    depths = compute_cell_depths(system, setup, n_cells).astype(float).ravel()
    speed_m = speed.copy()
    east_m = east.copy()
    north_m = north.copy()

    # maskeer onder de bodem
    for j in range(n_ens):
        bd = vb_depth[j]
        if np.isfinite(bd):
            mask = depths > bd
            speed_m[mask, j] = np.nan
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

    return depths, speed_m, vb_depth, u_mean, v_mean, spd_mean, dir_mean


def _percentile_limits(z: np.ndarray, lo=2.0, hi=98.0):
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


# ------------------ Plotly export helpers ------------------

def _mpl_cmap_to_plotly(cmapname: str, n=256):
    from matplotlib import cm
    cmap = cm.get_cmap(cmapname, n)
    colorscale = []
    for i in range(n):
        r, g, b, a = cmap(i)
        colorscale.append([i / (n - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"])
    return colorscale


def _export_speed_html(dist, depths, speed, vb_depth, vmin, vmax, cmapname, max_depth, out_html):
    try:
        import plotly.graph_objects as go
    except Exception:
        messagebox.showerror(
            "Plotly ontbreekt",
            "Voor HTML-export is plotly nodig.\nInstalleer met:\n\npip install plotly\n"
        )
        return

    dist = np.asarray(dist).ravel()
    depths = np.asarray(depths).ravel()
    speed = np.asarray(speed)
    vb_depth = np.asarray(vb_depth).ravel()

    # knip tot max_depth
    m = depths <= float(max_depth)
    depths_t = depths[m]
    speed_t = speed[m, :]

    colorscale = _mpl_cmap_to_plotly(cmapname)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=speed_t,
            x=dist,
            y=depths_t,
            colorscale=colorscale,
            zmin=float(vmin),
            zmax=float(vmax),
            hovertemplate="Afstand: %{x:.1f} m<br>Diepte: %{y:.2f} m<br>Snelheid: %{z:.3f} m/s<extra></extra>",
            colorbar=dict(title="m/s")
        )
    )
    # bodemlijn (optioneel)
    if vb_depth.size == dist.size:
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=np.clip(vb_depth, 0, float(max_depth)),
                mode="lines",
                name="Bodem",
                hoverinfo="skip"
            )
        )

    fig.update_layout(
        title="Snelheidsdoorsnede (afstand vs diepte)",
        xaxis_title="Afstand langs traject (m)",
        yaxis_title="Diepte (m)",
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(autorange="reversed", range=[float(max_depth), 0])

    fig.write_html(out_html, include_plotlyjs=True, full_html=True)


def _uv_to_latlon_delta(lat_deg, u_ms, v_ms, meters_per_ms=25.0):
    """Converteer u/v (m/s) naar lon/lat delta voor pijltjes op kaart."""
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


def _export_track_html(
    x,
    y,
    out_html,
    title="Boottrack",
    u=None,
    v=None,
    spd=None,
    direc=None,
    mode="xy",
    lat=None,
    lon=None,
    map_style="open-street-map",
    vec_step=3,
    vec_m_per_ms=25.0,
):
    try:
        import plotly.graph_objects as go
    except Exception:
        messagebox.showerror(
            "Plotly ontbreekt",
            "Voor HTML-export is plotly nodig.\nInstalleer met:\n\npip install plotly\n"
        )
        return

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    u = _align_to_len(u, n) if u is not None else np.full(n, np.nan, dtype=float)
    v = _align_to_len(v, n) if v is not None else np.full(n, np.nan, dtype=float)
    spd = _align_to_len(spd, n) if spd is not None else np.sqrt(u ** 2 + v ** 2)
    direc = _align_to_len(direc, n) if direc is not None else (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
    step = max(1, int(vec_step))

    fig = go.Figure()

    if mode == "map" and lat is not None and lon is not None:
        lat = _align_to_len(lat, n)
        lon = _align_to_len(lon, n)
        m = np.isfinite(lat) & np.isfinite(lon)
        if not np.any(m):
            raise RuntimeError("Geen geldige lon/lat gevonden voor kaartexport.")

        hover_txt = [
            f"Punt {i}<br>Vgem: {s:.3f} m/s<br>Richting: {d:.1f} deg"
            if np.isfinite(s) and np.isfinite(d) else f"Punt {i}<br>Vgem: NaN"
            for i, (s, d) in enumerate(zip(spd, direc), start=1)
        ]

        fig.add_trace(
            go.Scattermapbox(
                lat=lat[m],
                lon=lon[m],
                mode="lines+markers",
                marker=dict(size=7),
                text=np.asarray(hover_txt, dtype=object)[m],
                hovertemplate="%{text}<extra></extra>",
                name="Track"
            )
        )

        idx = np.arange(0, n, step, dtype=int)
        mv = m[idx] & np.isfinite(u[idx]) & np.isfinite(v[idx])
        idx = idx[mv]
        if idx.size:
            dlon, dlat = _uv_to_latlon_delta(lat[idx], u[idx], v[idx], meters_per_ms=vec_m_per_ms)
            lon2 = lon[idx] + dlon
            lat2 = lat[idx] + dlat
            lons, lats = [], []
            for xa, ya, xb, yb in zip(lon[idx], lat[idx], lon2, lat2):
                if np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb):
                    lons.extend([xa, xb, None])
                    lats.extend([ya, yb, None])
            if lons:
                fig.add_trace(
                    go.Scattermapbox(
                        lat=lats, lon=lons, mode="lines",
                        line=dict(width=2),
                        name="Vectoren"
                    )
                )

        fig.update_layout(
            title=title,
            margin=dict(l=20, r=20, t=60, b=20),
            mapbox=dict(
                style=map_style,
                center=dict(lat=float(np.nanmean(lat[m])), lon=float(np.nanmean(lon[m]))),
                zoom=16
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    else:
        hover_txt = [
            f"Punt {i}<br>X: {xx:.2f}<br>Y: {yy:.2f}<br>Vgem: {s:.3f} m/s<br>Richting: {d:.1f} deg"
            if np.isfinite(s) and np.isfinite(d) else f"Punt {i}<br>X: {xx:.2f}<br>Y: {yy:.2f}<br>Vgem: NaN"
            for i, (xx, yy, s, d) in enumerate(zip(x, y, spd, direc), start=1)
        ]
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines+markers",
                text=hover_txt, hovertemplate="%{text}<extra></extra>",
                name="Track"
            )
        )

        idx = np.arange(0, n, step, dtype=int)
        mv = np.isfinite(u[idx]) & np.isfinite(v[idx]) & np.isfinite(x[idx]) & np.isfinite(y[idx])
        idx = idx[mv]
        if idx.size:
            x2 = x[idx] + u[idx] * float(vec_m_per_ms)
            y2 = y[idx] + v[idx] * float(vec_m_per_ms)
            xs, ys = [], []
            for xa, ya, xb, yb in zip(x[idx], y[idx], x2, y2):
                xs.extend([xa, xb, None])
                ys.extend([ya, yb, None])
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    name="Vectoren"
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            margin=dict(l=60, r=30, t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.write_html(out_html, include_plotlyjs=True, full_html=True)


# ------------------ Figuur 1: snelheidsdoorsnede ------------------

def interactive_velocity_section(dist, depths, speed, vb_depth, basepath: str):
    dist = np.asarray(dist).ravel()
    depths = np.asarray(depths).ravel()
    vb_depth = np.asarray(vb_depth).ravel()
    speed = np.asarray(speed)

    # sorteer op afstand
    order = np.argsort(dist)
    dist = dist[order]
    if vb_depth.size == dist.size:
        vb_depth = vb_depth[order]
    if speed.ndim == 2 and speed.shape[1] == order.size:
        speed = speed[:, order]

    finite = speed[np.isfinite(speed)]
    vmax_data = float(np.nanmax(finite)) if finite.size else 1.0

    # init waarden
    vmin0, vmax0 = _percentile_limits(speed, 2, 98)
    vmax0 = max(vmin0 + 1e-6, vmax0)
    max_depth0 = float(np.nanmax(depths))
    alpha0 = 0.90
    cmap0 = "viridis"
    interp0 = "nearest"

    # --- fig layout (ruimer, geen overlap) ---
    fig = plt.figure(figsize=(15, 8), dpi=110)
    fig.canvas.manager.set_window_title("Snelheidsdoorsnede")

    ax = fig.add_axes([0.07, 0.28, 0.80, 0.68])
    ax.set_title("Snelheidsdoorsnede (afstand vs diepte)")
    ax.set_xlabel("Afstand langs traject (m)")
    ax.set_ylabel("Diepte (m)")
    ax.grid(True, alpha=0.2)

    # colorbar rechts
    cax = fig.add_axes([0.89, 0.30, 0.02, 0.64])

    # status tekst in plot (niet in controls)
    status = ax.text(
        0.01, 0.02,
        f"Backend: {matplotlib.get_backend()}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.7)
    )

    # hover label
    hover_txt = ax.text(
        0.99, 0.02,
        "",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.85),
        visible=True
    )

    # controls axes
    ax_vmin = fig.add_axes([0.07, 0.20, 0.36, 0.03])
    ax_vmax = fig.add_axes([0.51, 0.20, 0.36, 0.03])
    ax_maxd = fig.add_axes([0.07, 0.15, 0.54, 0.03])
    ax_alpha = fig.add_axes([0.68, 0.15, 0.19, 0.03])

    s_vmin = Slider(ax_vmin, "vmin", 0.0, max(1e-6, vmax_data), valinit=vmin0, valfmt="%.3f")
    s_vmax = Slider(ax_vmax, "vmax", 0.0, max(1e-6, vmax_data), valinit=vmax0, valfmt="%.3f")
    s_maxd = Slider(ax_maxd, "Max diepte (m)", 0.05, float(np.nanmax(depths)), valinit=max_depth0, valfmt="%.2f")
    s_alpha = Slider(ax_alpha, "Bodem Î±", 0.0, 1.0, valinit=alpha0, valfmt="%.2f")

    for s in (s_vmin, s_vmax, s_maxd, s_alpha):
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # radio's
    ax_cmap = fig.add_axes([0.07, 0.03, 0.18, 0.10])
    ax_interp = fig.add_axes([0.28, 0.03, 0.18, 0.10])
    cmaps = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "RdBu_r"]
    interps = ["nearest", "bilinear", "bicubic"]
    r_cmap = RadioButtons(ax_cmap, cmaps, active=0)
    r_interp = RadioButtons(ax_interp, interps, active=0)
    for lab in r_cmap.labels + r_interp.labels:
        lab.set_fontsize(9)

    # checks
    ax_checks = fig.add_axes([0.48, 0.03, 0.18, 0.10])
    c_checks = CheckButtons(ax_checks, ["Bodemvulling", "Raster", "Hover"], [True, True, True])
    for lab in c_checks.labels:
        lab.set_fontsize(9)

    # buttons
    ax_auto = fig.add_axes([0.68, 0.08, 0.07, 0.05])
    ax_reset = fig.add_axes([0.76, 0.08, 0.07, 0.05])
    ax_html = fig.add_axes([0.84, 0.08, 0.07, 0.05])
    ax_png = fig.add_axes([0.68, 0.02, 0.10, 0.05])
    ax_pdf = fig.add_axes([0.80, 0.02, 0.11, 0.05])

    b_auto = Button(ax_auto, "Auto")
    b_reset = Button(ax_reset, "Reset")
    b_html = Button(ax_html, "HTML")
    b_png = Button(ax_png, "PNG")
    b_pdf = Button(ax_pdf, "PDF")

    # file outputs
    out_png = basepath + "_speed.png"
    out_pdf = basepath + "_speed.pdf"

    def _save_png(_evt=None):
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Opgeslagen", f"PNG opgeslagen:\n{out_png}")

    def _save_pdf(_evt=None):
        fig.savefig(out_pdf, bbox_inches="tight")
        messagebox.showinfo("Opgeslagen", f"PDF opgeslagen:\n{out_pdf}")

    # internal state holders
    state = {
        "mesh": None,
        "cb": None,
        "fill": None,
        "bottom_line": None,
        "cmap": cmap0,
        "interp": interp0,
        "max_depth": max_depth0,
        "vmin": vmin0,
        "vmax": vmax0,
    }

    def _apply_grid(on: bool):
        ax.grid(on, alpha=0.2)

    def _clip_by_depth(md: float):
        """Maskeer data dieper dan md."""
        md = float(md)
        m = depths <= md
        if not np.any(m):
            # minimum: toon 1 cell
            m = np.zeros_like(depths, dtype=bool)
            m[0] = True
        return m

    def _redraw():
        # huidige instellingen
        md = float(s_maxd.val)
        vmin = float(s_vmin.val)
        vmax = float(s_vmax.val)
        if vmax <= vmin:
            vmax = vmin + 1e-6

        cmapname = r_cmap.value_selected
        interp = r_interp.value_selected
        alpha = float(s_alpha.val)

        show_fill, show_grid, show_hover = c_checks.get_status()

        # mask op max diepte
        m = _clip_by_depth(md)
        depths_t = depths[m]
        speed_t = speed[m, :]

        # verwijder oude mesh
        if state["mesh"] is not None:
            try:
                state["mesh"].remove()
            except Exception:
                pass
            state["mesh"] = None

        # pcolormesh (centers => edges)
        # edges voor afstand: midpunten
        dist_edges = np.r_[dist[0] - (dist[1] - dist[0]) / 2.0,
                           0.5 * (dist[1:] + dist[:-1]),
                           dist[-1] + (dist[-1] - dist[-2]) / 2.0] if dist.size > 1 else np.array([dist[0]-0.5, dist[0]+0.5])
        # edges voor diepte
        if depths_t.size > 1:
            mids = 0.5 * (depths_t[1:] + depths_t[:-1])
            first = depths_t[0] - (mids[0] - depths_t[0])
            last = depths_t[-1] + (depths_t[-1] - mids[-1])
            depth_edges = np.r_[first, mids, last]
        else:
            depth_edges = np.array([max(0.0, depths_t[0]-0.05), depths_t[0]+0.05])

        mesh = ax.pcolormesh(
            dist_edges, depth_edges, speed_t,
            shading="auto",
            cmap=cmapname,
            vmin=vmin,
            vmax=vmax
        )
        # interpolation effect voor pcolormesh: via rasterization/antialiasing beperken
        if interp in ("bilinear", "bicubic"):
            mesh.set_rasterized(True)

        state["mesh"] = mesh

        # colorbar
        if state["cb"] is None:
            state["cb"] = plt.colorbar(mesh, cax=cax)
            state["cb"].set_label("Snelheid (m/s)")
        else:
            state["cb"].update_normal(mesh)

        # bodem: lijn + (optioneel) vulling
        if state["bottom_line"] is None:
            (ln,) = ax.plot(dist, vb_depth, lw=1.0, alpha=0.7)
            state["bottom_line"] = ln
        else:
            state["bottom_line"].set_data(dist, vb_depth)

        # vulling onder bodem tot md
        if state["fill"] is not None:
            try:
                state["fill"].remove()
            except Exception:
                pass
            state["fill"] = None

        if show_fill:
            vb_clip = np.clip(vb_depth, 0.0, md)
            state["fill"] = ax.fill_between(dist, vb_clip, md, alpha=alpha)

        # y-lim: knip echt af (en invert)
        ax.set_ylim(md, 0.0)

        _apply_grid(show_grid)
        hover_txt.set_visible(show_hover)

        fig.canvas.draw_idle()

    # init draw
    _redraw()

    def _auto(_evt=None):
        lo, hi = _percentile_limits(speed, 2, 98)
        s_vmin.set_val(lo)
        s_vmax.set_val(hi)

    def _reset(_evt=None):
        s_vmin.set_val(vmin0)
        s_vmax.set_val(vmax0)
        s_maxd.set_val(max_depth0)
        s_alpha.set_val(alpha0)
        # radios & checks reset
        r_cmap.set_active(0)
        r_interp.set_active(0)
        # checkbuttons: zet naar default True/True/True
        current = c_checks.get_status()
        desired = [True, True, True]
        for i, (c, d) in enumerate(zip(current, desired)):
            if c != d:
                c_checks.set_active(i)
        _redraw()

    def _save_html(_evt=None):
        # vraag pad
        init = basepath + "_speed.html"
        out = filedialog.asksaveasfilename(
            title="Bewaar interactieve HTML (Plotly)",
            defaultextension=".html",
            initialfile=os.path.basename(init),
            initialdir=os.path.dirname(init),
            filetypes=[("HTML", "*.html")]
        )
        if not out:
            return
        # export met huidige state
        md = float(s_maxd.val)
        vmin = float(s_vmin.val)
        vmax = float(s_vmax.val)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        cmapname = r_cmap.value_selected
        _export_speed_html(dist, depths, speed, vb_depth, vmin, vmax, cmapname, md, out)
        messagebox.showinfo("Opgeslagen", f"Interactieve HTML opgeslagen:\n{out}")

    # callbacks
    s_vmin.on_changed(lambda _v: _redraw())
    s_vmax.on_changed(lambda _v: _redraw())
    s_maxd.on_changed(lambda _v: _redraw())
    s_alpha.on_changed(lambda _v: _redraw())
    r_cmap.on_clicked(lambda _l: _redraw())
    r_interp.on_clicked(lambda _l: _redraw())
    c_checks.on_clicked(lambda _l: _redraw())

    b_auto.on_clicked(_auto)
    b_reset.on_clicked(_reset)
    b_html.on_clicked(_save_html)
    b_png.on_clicked(_save_png)
    b_pdf.on_clicked(_save_pdf)

    # hover event
    def _on_move(event):
        if not hover_txt.get_visible():
            return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        md = float(s_maxd.val)
        if y < 0 or y > md:
            hover_txt.set_text("")
            fig.canvas.draw_idle()
            return

        # dichtstbijzijnde indices (centers)
        i = int(np.clip(np.searchsorted(dist, x), 0, dist.size - 1))
        # diepte index: in truncated set
        m = _clip_by_depth(md)
        depths_t = depths[m]
        j = int(np.clip(np.searchsorted(depths_t, y), 0, depths_t.size - 1))
        val = speed[m, :][j, i]
        if not np.isfinite(val):
            hover_txt.set_text(f"Afst: {dist[i]:.1f} m | Diepte: {depths_t[j]:.2f} m | Snelheid: NaN")
        else:
            hover_txt.set_text(f"Afst: {dist[i]:.1f} m | Diepte: {depths_t[j]:.2f} m | Snelheid: {val:.3f} m/s")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)

    # hou widgets vast (voorkomt 'dode' sliders)
    fig._widgets = dict(
        s_vmin=s_vmin, s_vmax=s_vmax, s_maxd=s_maxd, s_alpha=s_alpha,
        r_cmap=r_cmap, r_interp=r_interp, c_checks=c_checks,
        b_auto=b_auto, b_reset=b_reset, b_html=b_html, b_png=b_png, b_pdf=b_pdf
    )

    return fig


# ------------------ Figuur 2: boottrack ------------------

def _rotate_xy(x, y, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return xr, yr


def interactive_boat_track(
    x_rel,
    y_rel,
    basepath: str,
    u_mean,
    v_mean,
    spd_mean,
    dir_mean,
    absolute_track: dict | None = None,
):
    x_rel = np.asarray(x_rel, dtype=float).ravel()
    y_rel = np.asarray(y_rel, dtype=float).ravel()
    n = x_rel.size

    u_mean = _align_to_len(u_mean, n)
    v_mean = _align_to_len(v_mean, n)
    spd_mean = _align_to_len(spd_mean, n)
    dir_mean = _align_to_len(dir_mean, n)

    absolute_track = absolute_track or {}
    utm_x = _align_to_len(absolute_track.get("utm_x"), n) if "utm_x" in absolute_track else None
    utm_y = _align_to_len(absolute_track.get("utm_y"), n) if "utm_y" in absolute_track else None
    lon = _align_to_len(absolute_track.get("lon"), n) if "lon" in absolute_track else None
    lat = _align_to_len(absolute_track.get("lat"), n) if "lat" in absolute_track else None

    modes = ["Relatief XY"]
    if utm_x is not None and utm_y is not None and np.any(np.isfinite(utm_x) & np.isfinite(utm_y)):
        modes.append("Absoluut UTM")
    if lon is not None and lat is not None and np.any(np.isfinite(lon) & np.isfinite(lat)):
        modes.append("Absoluut lon/lat")

    map_styles = ["OpenStreetMap", "Carto Positron", "Carto Dark"]
    map_style_map = {
        "OpenStreetMap": "open-street-map",
        "Carto Positron": "carto-positron",
        "Carto Dark": "carto-darkmatter",
    }

    fig = plt.figure(figsize=(12, 8), dpi=110)
    fig.canvas.manager.set_window_title("Boottrack + dieptegemiddelde vectoren")

    ax = fig.add_axes([0.08, 0.24, 0.82, 0.70])
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    status_txt = ax.text(
        0.01, 0.01, "", transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.8),
    )

    angle0 = 0.0
    lw0 = 1.5
    ms0 = 3.0
    vec_m0 = 25.0
    step0 = 3
    show_pts0 = True

    (ln,) = ax.plot([], [], lw=lw0)
    sc = None
    qv = None

    ax_ang = fig.add_axes([0.10, 0.16, 0.55, 0.03])
    ax_lw = fig.add_axes([0.10, 0.12, 0.55, 0.03])
    ax_ms = fig.add_axes([0.10, 0.08, 0.55, 0.03])
    ax_vecm = fig.add_axes([0.10, 0.04, 0.55, 0.03])
    ax_step = fig.add_axes([0.10, 0.00, 0.55, 0.03])

    s_ang = Slider(ax_ang, "Rot (deg)", -180, 180, valinit=angle0, valfmt="%.0f")
    s_lw = Slider(ax_lw, "Lijn", 0.5, 6.0, valinit=lw0, valfmt="%.1f")
    s_ms = Slider(ax_ms, "Marker", 1.0, 10.0, valinit=ms0, valfmt="%.1f")
    s_vecm = Slider(ax_vecm, "Vector m/(m/s)", 1.0, 200.0, valinit=vec_m0, valfmt="%.0f")
    s_step = Slider(ax_step, "Vector stap", 1.0, 20.0, valinit=step0, valstep=1, valfmt="%.0f")

    for s in (s_ang, s_lw, s_ms, s_vecm, s_step):
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    ax_mode = fig.add_axes([0.68, 0.07, 0.16, 0.13])
    r_mode = RadioButtons(ax_mode, modes, active=0)
    for lab in r_mode.labels:
        lab.set_fontsize(9)

    ax_style = fig.add_axes([0.85, 0.07, 0.13, 0.13])
    r_style = RadioButtons(ax_style, map_styles, active=0)
    for lab in r_style.labels:
        lab.set_fontsize(8)

    ax_chk = fig.add_axes([0.68, 0.00, 0.30, 0.06])
    c_chk = CheckButtons(ax_chk, ["Punten", "Aspect=1", "Vectoren"], [show_pts0, True, True])
    for lab in c_chk.labels:
        lab.set_fontsize(9)

    ax_reset = fig.add_axes([0.91, 0.23, 0.08, 0.05])
    ax_html = fig.add_axes([0.91, 0.17, 0.08, 0.05])
    ax_csv = fig.add_axes([0.91, 0.11, 0.08, 0.05])
    ax_png = fig.add_axes([0.91, 0.05, 0.08, 0.05])
    ax_pdf = fig.add_axes([0.91, -0.01, 0.08, 0.05])

    b_reset = Button(ax_reset, "Reset")
    b_html = Button(ax_html, "HTML")
    b_csv = Button(ax_csv, "CSV")
    b_png = Button(ax_png, "PNG")
    b_pdf = Button(ax_pdf, "PDF")

    out_png = basepath + "_track_vectors.png"
    out_pdf = basepath + "_track_vectors.pdf"

    def _current_xyuv():
        mode = r_mode.value_selected
        ang = float(s_ang.val)
        xx, yy = x_rel, y_rel
        uu, vv = u_mean.copy(), v_mean.copy()

        if mode == "Absoluut UTM" and utm_x is not None and utm_y is not None:
            xx, yy = utm_x, utm_y
        elif mode == "Absoluut lon/lat" and lon is not None and lat is not None:
            xx, yy = lon, lat
            ang = 0.0

        if mode != "Absoluut lon/lat":
            xx, yy = _rotate_xy(xx, yy, ang)
            uu, vv = _rotate_xy(uu, vv, ang)

        return mode, xx, yy, uu, vv

    def _update():
        nonlocal sc, qv
        mode, xx, yy, uu, vv = _current_xyuv()
        lw = float(s_lw.val)
        ms = float(s_ms.val)
        vec_m = float(s_vecm.val)
        vec_step = max(1, int(round(s_step.val)))
        show_pts, aspect1, show_vec = c_chk.get_status()

        ln.set_data(xx, yy)
        ln.set_linewidth(lw)

        if sc is not None:
            sc.remove()
            sc = None
        if show_pts:
            mxy = np.isfinite(xx) & np.isfinite(yy)
            sc = ax.scatter(xx[mxy], yy[mxy], s=ms**2, alpha=0.7)

        if qv is not None:
            qv.remove()
            qv = None
        if show_vec:
            idx = np.arange(0, n, vec_step, dtype=int)
            if mode == "Absoluut lon/lat":
                if lon is not None and lat is not None:
                    dlon, dlat = _uv_to_latlon_delta(lat[idx], uu[idx], vv[idx], meters_per_ms=vec_m)
                    good = np.isfinite(xx[idx]) & np.isfinite(yy[idx]) & np.isfinite(dlon) & np.isfinite(dlat)
                    if np.any(good):
                        qv = ax.quiver(xx[idx][good], yy[idx][good], dlon[good], dlat[good], angles="xy", scale_units="xy", scale=1.0, width=0.0025, alpha=0.8)
            else:
                good = np.isfinite(xx[idx]) & np.isfinite(yy[idx]) & np.isfinite(uu[idx]) & np.isfinite(vv[idx])
                if np.any(good):
                    qv = ax.quiver(xx[idx][good], yy[idx][good], uu[idx][good] * vec_m, vv[idx][good] * vec_m, angles="xy", scale_units="xy", scale=1.0, width=0.0025, alpha=0.8)

        if mode == "Relatief XY":
            ax.set_title("Boottrack relatief (XY) + dieptegemiddelde vectoren")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
        elif mode == "Absoluut UTM":
            ax.set_title("Boottrack absoluut (UTM) + dieptegemiddelde vectoren")
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
        else:
            ax.set_title("Boottrack absoluut (lon/lat) + dieptegemiddelde vectoren")
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")

        if aspect1:
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("auto")

        finite_spd = spd_mean[np.isfinite(spd_mean)]
        if finite_spd.size:
            status_txt.set_text(
                f"Mode: {mode} | Vgem median: {np.nanmedian(finite_spd):.3f} m/s | Vectorschaal: {vec_m:.0f} m/(m/s)"
            )
        else:
            status_txt.set_text(f"Mode: {mode} | Geen geldige vectoren")

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def _reset(_evt=None):
        s_ang.set_val(angle0)
        s_lw.set_val(lw0)
        s_ms.set_val(ms0)
        s_vecm.set_val(vec_m0)
        s_step.set_val(step0)

        cur = c_chk.get_status()
        des = [show_pts0, True, True]
        for i, (c, d) in enumerate(zip(cur, des)):
            if c != d:
                c_chk.set_active(i)

        if r_mode.value_selected != modes[0]:
            r_mode.set_active(0)
        if r_style.value_selected != map_styles[0]:
            r_style.set_active(0)
        _update()

    def _save_png(_evt=None):
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Opgeslagen", f"PNG opgeslagen:\n{out_png}")

    def _save_pdf(_evt=None):
        fig.savefig(out_pdf, bbox_inches="tight")
        messagebox.showinfo("Opgeslagen", f"PDF opgeslagen:\n{out_pdf}")

    def _save_csv(_evt=None):
        import csv
        init = basepath + "_track_vectors.csv"
        out = filedialog.asksaveasfilename(
            title="Bewaar vectoren als CSV",
            defaultextension=".csv",
            initialfile=os.path.basename(init),
            initialdir=os.path.dirname(init),
            filetypes=[("CSV", "*.csv")]
        )
        if not out:
            return
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "x_rel_m", "y_rel_m", "utm_x_m", "utm_y_m", "lon_deg", "lat_deg", "u_mean_mps", "v_mean_mps", "speed_mean_mps", "dir_mean_deg"])
            for i in range(n):
                w.writerow([
                    i + 1,
                    x_rel[i], y_rel[i],
                    utm_x[i] if utm_x is not None else np.nan,
                    utm_y[i] if utm_y is not None else np.nan,
                    lon[i] if lon is not None else np.nan,
                    lat[i] if lat is not None else np.nan,
                    u_mean[i], v_mean[i], spd_mean[i], dir_mean[i]
                ])
        messagebox.showinfo("Opgeslagen", f"CSV opgeslagen:\n{out}")

    def _save_html(_evt=None):
        mode, xx, yy, uu, vv = _current_xyuv()
        vec_step = max(1, int(round(s_step.val)))
        vec_m = float(s_vecm.val)
        init = basepath + "_track_vectors.html"
        out = filedialog.asksaveasfilename(
            title="Bewaar interactieve HTML (Plotly)",
            defaultextension=".html",
            initialfile=os.path.basename(init),
            initialdir=os.path.dirname(init),
            filetypes=[("HTML", "*.html")]
        )
        if not out:
            return

        if mode == "Absoluut lon/lat" and lon is not None and lat is not None:
            style = map_style_map.get(r_style.value_selected, "open-street-map")
            _export_track_html(
                xx, yy, out, title="Boottrack op kaart + vectoren",
                u=uu, v=vv, spd=spd_mean, direc=dir_mean,
                mode="map", lat=lat, lon=lon, map_style=style,
                vec_step=vec_step, vec_m_per_ms=vec_m
            )
        else:
            _export_track_html(
                xx, yy, out, title=f"Boottrack ({mode}) + vectoren",
                u=uu, v=vv, spd=spd_mean, direc=dir_mean,
                mode="xy", vec_step=vec_step, vec_m_per_ms=vec_m
            )
        messagebox.showinfo("Opgeslagen", f"Interactieve HTML opgeslagen:\n{out}")

    s_ang.on_changed(lambda _v: _update())
    s_lw.on_changed(lambda _v: _update())
    s_ms.on_changed(lambda _v: _update())
    s_vecm.on_changed(lambda _v: _update())
    s_step.on_changed(lambda _v: _update())
    c_chk.on_clicked(lambda _l: _update())
    r_mode.on_clicked(lambda _l: _update())
    r_style.on_clicked(lambda _l: _update())

    b_reset.on_clicked(_reset)
    b_html.on_clicked(_save_html)
    b_csv.on_clicked(_save_csv)
    b_png.on_clicked(_save_png)
    b_pdf.on_clicked(_save_pdf)

    fig._widgets = dict(
        s_ang=s_ang, s_lw=s_lw, s_ms=s_ms, s_vecm=s_vecm, s_step=s_step,
        c_chk=c_chk, r_mode=r_mode, r_style=r_style,
        b_reset=b_reset, b_html=b_html, b_csv=b_csv, b_png=b_png, b_pdf=b_pdf
    )

    _update()
    return fig


# ------------------ Pipeline ------------------

def process_mat_file(mat_path: str):
    md = load_qrev_mat(mat_path)

    # haal structs
    try:
        system = md["System"]
        setup = md["Setup"]
        water_track = md["WaterTrack"]
        bottom_track = md["BottomTrack"]
    except Exception as e:
        raise RuntimeError("Kan vereiste velden niet vinden in .mat (System/Setup/WaterTrack/BottomTrack)") from e

    # track
    x, y, _t = compute_boat_track(bottom_track, system)
    dist = compute_transect_distance(x, y)

    # speed section + dieptegemiddelde vectoren
    depths, speed_masked, vb_depth, u_mean, v_mean, spd_mean, dir_mean = compute_speed_section_and_track_vectors(
        water_track, bottom_track, system, setup
    )

    # absolute coordinaten (optioneel)
    abs_track = extract_absolute_track(md, n_ens=x.size)

    base = os.path.splitext(mat_path)[0]

    fig1 = interactive_velocity_section(dist, depths, speed_masked, vb_depth, basepath=base)
    fig2 = interactive_boat_track(
        x, y, basepath=base,
        u_mean=u_mean, v_mean=v_mean, spd_mean=spd_mean, dir_mean=dir_mean,
        absolute_track=abs_track
    )

    plt.show()


def main():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "Selecteer .mat",
        "Kies een QRev/SonTek .mat bestand.\n\nTip: run als script in terminal voor echte interactie."
    )

    mat_path = filedialog.askopenfilename(
        title="Selecteer QRev/SonTek .mat",
        filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
    )
    if not mat_path:
        return

    try:
        process_mat_file(mat_path)
    except Exception as e:
        messagebox.showerror("Fout", f"Er is een fout opgetreden:\n\n{e}")
        raise


if __name__ == "__main__":
    main()

