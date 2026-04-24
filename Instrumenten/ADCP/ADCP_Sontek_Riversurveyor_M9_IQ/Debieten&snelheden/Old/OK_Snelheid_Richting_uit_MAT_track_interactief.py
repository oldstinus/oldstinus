#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
M9 / SonTek QRev (.mat) – interactieve doorsneden + boottrack (Matplotlib) + interactieve HTML export (Plotly)

V7 – uitbreidingen op basis van jouw laatste feedback
----------------------------------------------------
1) Vert. x slider blijft weg.
2) Max diepte knipt effectief af:
   - doorsnede wordt op diepte gemaskeerd (NaN) + y-lim wordt aangepast.
3) Hover (mouse-over) op doorsneden:
   - toont Afstand (m), Diepte (m) en de waarde (snelheid / richting / backscatter).
4) Extra figuren:
   - Snelheidsrichting (° t.o.v. Noord, richting NAAR)
   - Backscatter (indien veld gevonden in .mat)
5) Interactief wegschrijven:
   - per figuur: knop HTML -> Plotly HTML (zoom/pan/hover).

Opmerking:
- Run als script in een terminal (VS Code: "Run Python File in Terminal") voor echte interactie.
- Voor HTML-export: `pip install plotly`
"""

from __future__ import annotations

import os
import sys
import math
import importlib.util
import numpy as np
from pathlib import Path


_HTML_STATE_HELPER_ROOT = Path(__file__).resolve().parents[1]
if not (_HTML_STATE_HELPER_ROOT / "html_state_saver.py").exists():
    _HTML_STATE_HELPER_ROOT = Path(__file__).resolve().parents[2]
_HTML_STATE_HELPER_SPEC = importlib.util.spec_from_file_location(
    "html_state_saver", _HTML_STATE_HELPER_ROOT / "html_state_saver.py"
)
if _HTML_STATE_HELPER_SPEC is None or _HTML_STATE_HELPER_SPEC.loader is None:
    raise ImportError("html_state_saver.py kon niet worden geladen.")
_html_state_saver = importlib.util.module_from_spec(_HTML_STATE_HELPER_SPEC)
_HTML_STATE_HELPER_SPEC.loader.exec_module(_html_state_saver)
add_interactive_html_saver = _html_state_saver.add_interactive_html_saver

# ---- backend for echte interactie (sliders/knoppen) ----
def _force_interactive_backend():
    """
    Forceer een GUI-backend vóór pyplot import.
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
from scipy.io import loadmat

# tkinter file dialogs
import tkinter as tk
from tkinter import filedialog, messagebox


# ------------------ IO: QRev MAT laden ------------------

def load_qrev_mat(path: str):
    """Laad QRev/SonTek .mat bestand."""
    return loadmat(path, struct_as_record=False, squeeze_me=True)


def _as_float_1d(x) -> np.ndarray:
    """Zet onbekend MATLAB veld om naar 1D float array (kan leeg zijn)."""
    if x is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(x).astype(float).ravel()
    except Exception:
        try:
            return np.asarray(np.squeeze(x)).astype(float).ravel()
        except Exception:
            return np.array([], dtype=float)


def _scalar_from_field(x, *, default=np.nan) -> float:
    """Pak een bruikbare scalar uit veld dat soms (1xN) array is."""
    arr = _as_float_1d(x)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanmedian(arr))


# ------------------ Track / afstand ------------------

def compute_boat_track(bottom_track, system):
    """
    Benader boottrack XY door integratie van bottom-track velocities.
    QRev struct varianten:
      - BT_Vel: (n_ens, 4) of (4, n_ens)
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
        dt = 1.0

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


# ------------------ Diepte-as (cell centers) ------------------

def compute_cell_depths(system, setup, n_cells: int):
    """
    Bepaal cell depths (centers) in meter.
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

    # cell start
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


# ------------------ Doorsneden ------------------

def _mask_below_bottom(z2d: np.ndarray, depths: np.ndarray, vb_depth: np.ndarray) -> np.ndarray:
    """Maskeer cellen dieper dan bodemdiepte."""
    z = np.asarray(z2d, dtype=float).copy()
    depths = np.asarray(depths, dtype=float).ravel()
    vb_depth = np.asarray(vb_depth, dtype=float).ravel()
    if z.ndim != 2:
        return z
    n_cells, n_ens = z.shape
    if vb_depth.size != n_ens:
        return z
    for j in range(n_ens):
        bd = vb_depth[j]
        if np.isfinite(bd):
            z[depths > bd, j] = np.nan
    return z


def compute_speed_and_direction_sections(water_track, bottom_track, system, setup):
    """
    Uit WaterTrack.Velocity:
      - speed = sqrt(east^2 + north^2)
      - direction = bearing (deg) clockwise from North, richting NAAR (atan2(east, north))
    """
    vel = np.squeeze(water_track.Velocity)  # (n_cells, 4, n_ens)
    if vel.ndim != 3 or vel.shape[1] < 2:
        raise ValueError(f"Onverwachte WaterTrack.Velocity shape: {vel.shape}")

    east = vel[:, 0, :].astype(float)
    north = vel[:, 1, :].astype(float)

    speed = np.sqrt(east ** 2 + north ** 2)

    # richting (0=N, 90=E, 180=S, 270=W)
    direction = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0

    n_cells, n_ens = speed.shape
    bt_depth = np.squeeze(bottom_track.BT_Depth).astype(float).ravel()
    if bt_depth.size != n_ens:
        raise ValueError(f"BT_Depth lengte ({bt_depth.size}) != ensembles ({n_ens})")

    depths = compute_cell_depths(system, setup, n_cells).astype(float).ravel()

    # maskeer onder bodem + neem NaN-mask van speed over naar direction
    speed_m = _mask_below_bottom(speed, depths, bt_depth)
    direction_m = _mask_below_bottom(direction, depths, bt_depth)
    direction_m[~np.isfinite(speed_m)] = np.nan

    return depths, speed_m, direction_m, bt_depth


def _match_2d_to_cells_ens(arr, n_cells, n_ens):
    """Probeer arr om te vormen naar (n_cells, n_ens)."""
    a = np.asarray(arr)
    if a.ndim != 2:
        return None
    if a.shape == (n_cells, n_ens):
        return a.astype(float)
    if a.shape == (n_ens, n_cells):
        return a.T.astype(float)
    return None


def _match_3d_to_cells_ens(arr, n_cells, n_ens):
    """
    Probeer arr om te vormen naar (n_cells, n_ens) door over beam/extra as te middelen.
    Verwachte vormen:
      - (n_cells, n_beams, n_ens)
      - (n_cells, n_ens, n_beams)
      - (n_beams, n_cells, n_ens)
      - (n_ens, n_cells, n_beams) ...
    """
    a = np.asarray(arr)
    if a.ndim != 3:
        return None

    shp = a.shape
    # vind assen die cells/ens matchen
    axes_cells = [i for i, s in enumerate(shp) if s == n_cells]
    axes_ens = [i for i, s in enumerate(shp) if s == n_ens]
    if not axes_cells or not axes_ens:
        return None

    # kies eerste match
    ax_c = axes_cells[0]
    ax_e = axes_ens[0]
    # bepaal derde as = beam/extra
    ax_b = [0, 1, 2]
    ax_b.remove(ax_c)
    if ax_e in ax_b:
        ax_b.remove(ax_e)
    if len(ax_b) != 1:
        return None
    ax_b = ax_b[0]

    # permute naar (cells, beams, ens) en average beams
    a_perm = np.moveaxis(a, (ax_c, ax_b, ax_e), (0, 1, 2))
    # a_perm: (n_cells, n_beams, n_ens)
    bs = np.nanmean(a_perm.astype(float), axis=1)
    if bs.shape != (n_cells, n_ens):
        return None
    return bs


def extract_backscatter_section(water_track, bottom_track, depths, vb_depth, n_cells, n_ens):
    """
    Zoek backscatter/intensity veld in de .mat.
    We proberen eerst WaterTrack, dan BottomTrack.
    Returns: backscatter_2d (n_cells,n_ens) of None, en naam van het veld.
    """
    candidates = [
        "Backscatter", "BackScatter", "BS", "BackScat",
        "Intensity", "Intens", "Amplitude", "Amp",
        "RSSI", "SNR", "Signal", "Echo"
    ]

    # 1) WaterTrack
    for name in candidates:
        if hasattr(water_track, name):
            arr = np.squeeze(getattr(water_track, name))
            z2 = _match_2d_to_cells_ens(arr, n_cells, n_ens)
            if z2 is None:
                z2 = _match_3d_to_cells_ens(arr, n_cells, n_ens)
            if z2 is not None:
                z2 = _mask_below_bottom(z2, depths, vb_depth)
                return z2, f"WaterTrack.{name}"

    # 2) BottomTrack (soms VB-profiel/echo per cell)
    for name in candidates:
        if hasattr(bottom_track, name):
            arr = np.squeeze(getattr(bottom_track, name))
            z2 = _match_2d_to_cells_ens(arr, n_cells, n_ens)
            if z2 is None:
                z2 = _match_3d_to_cells_ens(arr, n_cells, n_ens)
            if z2 is not None:
                z2 = _mask_below_bottom(z2, depths, vb_depth)
                return z2, f"BottomTrack.{name}"

    return None, None


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


def _export_section_html(dist, depths, z2d, vb_depth, vmin, vmax, cmapname, max_depth, out_html,
                        title, cbar_title, hover_label, z_fmt="{z:.3f}"):
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
    z2d = np.asarray(z2d, dtype=float)
    vb_depth = np.asarray(vb_depth).ravel()

    md = float(max_depth)
    m = depths <= md
    if not np.any(m):
        m = np.zeros_like(depths, dtype=bool)
        m[0] = True

    depths_t = depths[m]
    zt = z2d[m, :]

    colorscale = _mpl_cmap_to_plotly(cmapname)

    hovertemplate = (
        "Afstand: %{x:.1f} m<br>"
        "Diepte: %{y:.2f} m<br>"
        f"{hover_label}: %{z}<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=zt,
            x=dist,
            y=depths_t,
            colorscale=colorscale,
            zmin=float(vmin),
            zmax=float(vmax),
            hovertemplate=hovertemplate,
            colorbar=dict(title=cbar_title)
        )
    )

    if vb_depth.size == dist.size:
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=np.clip(vb_depth, 0, md),
                mode="lines",
                name="Bodem",
                hoverinfo="skip"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Afstand langs traject (m)",
        yaxis_title="Diepte (m)",
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(autorange="reversed", range=[md, 0])
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    add_interactive_html_saver(out_html)


def _export_track_html(x, y, out_html, title="Boottrack"):
    try:
        import plotly.graph_objects as go
    except Exception:
        messagebox.showerror(
            "Plotly ontbreekt",
            "Voor HTML-export is plotly nodig.\nInstalleer met:\n\npip install plotly\n"
        )
        return

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines+markers",
            hovertemplate="X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>",
            name="track"
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    add_interactive_html_saver(out_html)


# ------------------ Generieke interactieve doorsnede ------------------

def interactive_section(dist, depths, z2d, vb_depth, basepath: str, suffix: str,
                        title: str, cbar_label: str, hover_label: str,
                        cmaps: list[str], cmap_default: str,
                        vmin_fixed=None, vmax_fixed=None):
    """
    Interactieve doorsnede met:
      - vmin/vmax sliders
      - Max diepte (m) slider (knipt effectief)
      - Bodemvulling + alpha
      - Raster + Hover toggle
      - HTML/PNG/PDF export
    """
    dist = np.asarray(dist).ravel()
    depths = np.asarray(depths).ravel()
    vb_depth = np.asarray(vb_depth).ravel()
    z2d = np.asarray(z2d, dtype=float)

    # sorteer op afstand
    order = np.argsort(dist)
    dist = dist[order]
    if vb_depth.size == dist.size:
        vb_depth = vb_depth[order]
    if z2d.ndim == 2 and z2d.shape[1] == order.size:
        z2d = z2d[:, order]

    # init vmin/vmax
    finite = z2d[np.isfinite(z2d)]
    max_data = float(np.nanmax(finite)) if finite.size else 1.0

    if vmin_fixed is not None and vmax_fixed is not None:
        vmin0, vmax0 = float(vmin_fixed), float(vmax_fixed)
        slider_max = float(vmax_fixed)
    else:
        vmin0, vmax0 = _percentile_limits(z2d, 2, 98)
        vmax0 = max(vmin0 + 1e-6, vmax0)
        slider_max = max(1e-6, max_data)

    max_depth0 = float(np.nanmax(depths))
    alpha0 = 0.90

    # fig layout
    fig = plt.figure(figsize=(15, 8), dpi=110)
    fig.canvas.manager.set_window_title(title)

    ax = fig.add_axes([0.07, 0.28, 0.78, 0.68])
    ax.set_title(title)
    ax.set_xlabel("Afstand langs traject (m)")
    ax.set_ylabel("Diepte (m)")
    ax.grid(True, alpha=0.2)

    # colorbar rechts
    cax = fig.add_axes([0.87, 0.30, 0.02, 0.64])

    # status & hover in plot
    ax.text(
        0.01, 0.02,
        f"Backend: {matplotlib.get_backend()}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.7)
    )

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
    ax_vmax = fig.add_axes([0.49, 0.20, 0.36, 0.03])
    ax_maxd = fig.add_axes([0.07, 0.15, 0.56, 0.03])
    ax_alpha = fig.add_axes([0.69, 0.15, 0.16, 0.03])

    s_vmin = Slider(ax_vmin, "vmin", float(vmin0 if vmin_fixed is not None else 0.0), float(slider_max),
                    valinit=float(vmin0), valfmt="%.3f")
    s_vmax = Slider(ax_vmax, "vmax", float(vmin0 if vmin_fixed is not None else 0.0), float(slider_max),
                    valinit=float(vmax0), valfmt="%.3f")
    s_maxd = Slider(ax_maxd, "Max diepte (m)", 0.05, float(np.nanmax(depths)), valinit=max_depth0, valfmt="%.2f")
    s_alpha = Slider(ax_alpha, "Bodem α", 0.0, 1.0, valinit=alpha0, valfmt="%.2f")

    for s in (s_vmin, s_vmax, s_maxd, s_alpha):
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # radio's
    ax_cmap = fig.add_axes([0.07, 0.03, 0.18, 0.10])
    ax_interp = fig.add_axes([0.28, 0.03, 0.18, 0.10])
    interps = ["nearest", "bilinear", "bicubic"]
    r_cmap = RadioButtons(ax_cmap, cmaps, active=max(0, cmaps.index(cmap_default) if cmap_default in cmaps else 0))
    r_interp = RadioButtons(ax_interp, interps, active=0)
    for lab in r_cmap.labels + r_interp.labels:
        lab.set_fontsize(9)

    # checks
    ax_checks = fig.add_axes([0.48, 0.03, 0.18, 0.10])
    c_checks = CheckButtons(ax_checks, ["Bodemvulling", "Raster", "Hover"], [True, True, True])
    for lab in c_checks.labels:
        lab.set_fontsize(9)

    # buttons
    ax_auto = fig.add_axes([0.69, 0.08, 0.07, 0.05])
    ax_reset = fig.add_axes([0.77, 0.08, 0.07, 0.05])
    ax_html = fig.add_axes([0.85, 0.08, 0.07, 0.05])
    ax_png = fig.add_axes([0.69, 0.02, 0.10, 0.05])
    ax_pdf = fig.add_axes([0.81, 0.02, 0.11, 0.05])

    b_auto = Button(ax_auto, "Auto")
    b_reset = Button(ax_reset, "Reset")
    b_html = Button(ax_html, "HTML")
    b_png = Button(ax_png, "PNG")
    b_pdf = Button(ax_pdf, "PDF")

    out_png = basepath + f"_{suffix}.png"
    out_pdf = basepath + f"_{suffix}.pdf"

    state = {"mesh": None, "cb": None, "fill": None, "bottom_line": None}

    def _clip_mask(md: float):
        md = float(md)
        m = depths <= md
        if not np.any(m):
            m = np.zeros_like(depths, dtype=bool)
            m[0] = True
        return m

    def _redraw():
        md = float(s_maxd.val)
        vmin = float(s_vmin.val)
        vmax = float(s_vmax.val)
        if vmax <= vmin:
            vmax = vmin + 1e-6

        cmapname = r_cmap.value_selected
        interp = r_interp.value_selected
        alpha = float(s_alpha.val)

        show_fill, show_grid, show_hover = c_checks.get_status()

        m = _clip_mask(md)
        depths_t = depths[m]
        zt = z2d[m, :]

        # verwijder oude mesh
        if state["mesh"] is not None:
            try:
                state["mesh"].remove()
            except Exception:
                pass
            state["mesh"] = None

        # edges voor afstand
        if dist.size > 1:
            dist_edges = np.r_[dist[0] - (dist[1] - dist[0]) / 2.0,
                               0.5 * (dist[1:] + dist[:-1]),
                               dist[-1] + (dist[-1] - dist[-2]) / 2.0]
        else:
            dist_edges = np.array([dist[0] - 0.5, dist[0] + 0.5])

        # edges voor diepte
        if depths_t.size > 1:
            mids = 0.5 * (depths_t[1:] + depths_t[:-1])
            first = depths_t[0] - (mids[0] - depths_t[0])
            last = depths_t[-1] + (depths_t[-1] - mids[-1])
            depth_edges = np.r_[first, mids, last]
        else:
            depth_edges = np.array([max(0.0, depths_t[0] - 0.05), depths_t[0] + 0.05])

        mesh = ax.pcolormesh(
            dist_edges, depth_edges, zt,
            shading="auto",
            cmap=cmapname,
            vmin=vmin, vmax=vmax
        )
        if interp in ("bilinear", "bicubic"):
            mesh.set_rasterized(True)

        state["mesh"] = mesh

        # colorbar
        if state["cb"] is None:
            state["cb"] = plt.colorbar(mesh, cax=cax)
            state["cb"].set_label(cbar_label)
        else:
            state["cb"].update_normal(mesh)
            state["cb"].set_label(cbar_label)

        # bodemlijn + vulling
        if vb_depth.size == dist.size:
            if state["bottom_line"] is None:
                (ln,) = ax.plot(dist, vb_depth, lw=1.0, alpha=0.7)
                state["bottom_line"] = ln
            else:
                state["bottom_line"].set_data(dist, vb_depth)

            if state["fill"] is not None:
                try:
                    state["fill"].remove()
                except Exception:
                    pass
                state["fill"] = None

            if show_fill:
                vb_clip = np.clip(vb_depth, 0.0, md)
                state["fill"] = ax.fill_between(dist, vb_clip, md, alpha=alpha)
        else:
            # als er geen bodemprofiel is, verwijder eventuele oude dingen
            if state["bottom_line"] is not None:
                try:
                    state["bottom_line"].remove()
                except Exception:
                    pass
                state["bottom_line"] = None
            if state["fill"] is not None:
                try:
                    state["fill"].remove()
                except Exception:
                    pass
                state["fill"] = None

        # y-lim: knip af en invert
        ax.set_ylim(md, 0.0)
        ax.grid(show_grid, alpha=0.2)
        hover_txt.set_visible(show_hover)

        fig.canvas.draw_idle()

    # initial draw
    _redraw()

    def _auto(_evt=None):
        if vmin_fixed is not None and vmax_fixed is not None:
            s_vmin.set_val(float(vmin_fixed))
            s_vmax.set_val(float(vmax_fixed))
        else:
            lo, hi = _percentile_limits(z2d, 2, 98)
            s_vmin.set_val(lo)
            s_vmax.set_val(hi)

    def _reset(_evt=None):
        s_vmin.set_val(float(vmin0))
        s_vmax.set_val(float(vmax0))
        s_maxd.set_val(float(max_depth0))
        s_alpha.set_val(float(alpha0))
        # radios reset
        r_cmap.set_active(max(0, cmaps.index(cmap_default) if cmap_default in cmaps else 0))
        r_interp.set_active(0)
        # checks reset True/True/True
        current = c_checks.get_status()
        desired = [True, True, True]
        for i, (c, d) in enumerate(zip(current, desired)):
            if c != d:
                c_checks.set_active(i)
        _redraw()

    def _save_png(_evt=None):
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Opgeslagen", f"PNG opgeslagen:\n{out_png}")

    def _save_pdf(_evt=None):
        fig.savefig(out_pdf, bbox_inches="tight")
        messagebox.showinfo("Opgeslagen", f"PDF opgeslagen:\n{out_pdf}")

    def _save_html(_evt=None):
        init = basepath + f"_{suffix}.html"
        out = filedialog.asksaveasfilename(
            title="Bewaar interactieve HTML (Plotly)",
            defaultextension=".html",
            initialfile=os.path.basename(init),
            initialdir=os.path.dirname(init),
            filetypes=[("HTML", "*.html")]
        )
        if not out:
            return
        md = float(s_maxd.val)
        vmin = float(s_vmin.val)
        vmax = float(s_vmax.val)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        cmapname = r_cmap.value_selected
        _export_section_html(dist, depths, z2d, vb_depth, vmin, vmax, cmapname, md, out,
                            title=title, cbar_title=cbar_label, hover_label=hover_label)
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

    # hover
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

        i = int(np.clip(np.searchsorted(dist, x), 0, dist.size - 1))
        m = _clip_mask(md)
        depths_t = depths[m]
        j = int(np.clip(np.searchsorted(depths_t, y), 0, depths_t.size - 1))
        val = z2d[m, :][j, i]
        if np.isfinite(val):
            if "°" in cbar_label:
                hover_txt.set_text(f"Afst: {dist[i]:.1f} m | Diepte: {depths_t[j]:.2f} m | {hover_label}: {val:.1f}°")
            else:
                hover_txt.set_text(f"Afst: {dist[i]:.1f} m | Diepte: {depths_t[j]:.2f} m | {hover_label}: {val:.3f}")
        else:
            hover_txt.set_text(f"Afst: {dist[i]:.1f} m | Diepte: {depths_t[j]:.2f} m | {hover_label}: NaN")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)

    # hou widgets vast (voorkomt dode sliders)
    fig._widgets = dict(
        s_vmin=s_vmin, s_vmax=s_vmax, s_maxd=s_maxd, s_alpha=s_alpha,
        r_cmap=r_cmap, r_interp=r_interp, c_checks=c_checks,
        b_auto=b_auto, b_reset=b_reset, b_html=b_html, b_png=b_png, b_pdf=b_pdf
    )
    return fig


# ------------------ Boottrack (zoals v6) ------------------

def _rotate_xy(x, y, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return xr, yr


def interactive_boat_track(x, y, basepath: str):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    fig = plt.figure(figsize=(9, 7), dpi=110)
    fig.canvas.manager.set_window_title("Boottrack")

    ax = fig.add_axes([0.08, 0.20, 0.86, 0.74])
    ax.set_title("Boottrack (relatief, uit BT-vel integratie)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    angle0 = 0.0
    lw0 = 1.5
    ms0 = 3.0
    show_pts0 = True

    (ln,) = ax.plot(x, y, lw=lw0)
    sc = ax.scatter(x, y, s=ms0**2, alpha=0.7) if show_pts0 else None

    ax_ang = fig.add_axes([0.10, 0.12, 0.60, 0.03])
    ax_lw = fig.add_axes([0.10, 0.08, 0.60, 0.03])
    ax_ms = fig.add_axes([0.10, 0.04, 0.60, 0.03])

    s_ang = Slider(ax_ang, "Rot (°)", -180, 180, valinit=angle0, valfmt="%.0f")
    s_lw = Slider(ax_lw, "Lijn", 0.5, 6.0, valinit=lw0, valfmt="%.1f")
    s_ms = Slider(ax_ms, "Marker", 1.0, 10.0, valinit=ms0, valfmt="%.1f")
    for s in (s_ang, s_lw, s_ms):
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    ax_chk = fig.add_axes([0.73, 0.04, 0.18, 0.11])
    c_chk = CheckButtons(ax_chk, ["Punten", "Aspect=1"], [show_pts0, True])
    for lab in c_chk.labels:
        lab.set_fontsize(9)

    ax_reset = fig.add_axes([0.73, 0.15, 0.10, 0.05])
    ax_html = fig.add_axes([0.84, 0.15, 0.10, 0.05])
    ax_png = fig.add_axes([0.73, 0.10, 0.10, 0.05])
    ax_pdf = fig.add_axes([0.84, 0.10, 0.10, 0.05])

    b_reset = Button(ax_reset, "Reset")
    b_html = Button(ax_html, "HTML")
    b_png = Button(ax_png, "PNG")
    b_pdf = Button(ax_pdf, "PDF")

    out_png = basepath + "_track.png"
    out_pdf = basepath + "_track.pdf"

    def _update():
        ang = float(s_ang.val)
        lw = float(s_lw.val)
        ms = float(s_ms.val)
        show_pts, aspect1 = c_chk.get_status()

        xr, yr = _rotate_xy(x, y, ang)
        ln.set_data(xr, yr)
        ln.set_linewidth(lw)

        nonlocal sc
        if sc is not None:
            sc.remove()
            sc = None
        if show_pts:
            sc = ax.scatter(xr, yr, s=ms**2, alpha=0.7)

        ax.set_aspect("equal" if aspect1 else "auto", adjustable="box" if aspect1 else None)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def _reset(_evt=None):
        s_ang.set_val(angle0)
        s_lw.set_val(lw0)
        s_ms.set_val(ms0)
        cur = c_chk.get_status()
        des = [show_pts0, True]
        for i, (c, d) in enumerate(zip(cur, des)):
            if c != d:
                c_chk.set_active(i)
        _update()

    def _save_png(_evt=None):
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Opgeslagen", f"PNG opgeslagen:\n{out_png}")

    def _save_pdf(_evt=None):
        fig.savefig(out_pdf, bbox_inches="tight")
        messagebox.showinfo("Opgeslagen", f"PDF opgeslagen:\n{out_pdf}")

    def _save_html(_evt=None):
        init = basepath + "_track.html"
        out = filedialog.asksaveasfilename(
            title="Bewaar interactieve HTML (Plotly)",
            defaultextension=".html",
            initialfile=os.path.basename(init),
            initialdir=os.path.dirname(init),
            filetypes=[("HTML", "*.html")]
        )
        if not out:
            return
        ang = float(s_ang.val)
        xr, yr = _rotate_xy(x, y, ang)
        _export_track_html(xr, yr, out, title="Boottrack")
        messagebox.showinfo("Opgeslagen", f"Interactieve HTML opgeslagen:\n{out}")

    s_ang.on_changed(lambda _v: _update())
    s_lw.on_changed(lambda _v: _update())
    s_ms.on_changed(lambda _v: _update())
    c_chk.on_clicked(lambda _l: _update())

    b_reset.on_clicked(_reset)
    b_html.on_clicked(_save_html)
    b_png.on_clicked(_save_png)
    b_pdf.on_clicked(_save_pdf)

    fig._widgets = dict(
        s_ang=s_ang, s_lw=s_lw, s_ms=s_ms,
        c_chk=c_chk, b_reset=b_reset, b_html=b_html, b_png=b_png, b_pdf=b_pdf
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

    # track + afstand
    x, y, _t = compute_boat_track(bottom_track, system)
    dist = compute_transect_distance(x, y)

    # speed + direction
    depths, speed_m, dir_m, vb_depth = compute_speed_and_direction_sections(water_track, bottom_track, system, setup)
    n_cells, n_ens = speed_m.shape

    # backscatter (optioneel)
    bs_m, bs_src = extract_backscatter_section(water_track, bottom_track, depths, vb_depth, n_cells, n_ens)
    if bs_m is None:
        messagebox.showwarning(
            "Backscatter niet gevonden",
            "Ik vond geen duidelijk backscatter/intensity veld in dit .mat bestand.\n\n"
            "Snelheid + richting worden wél geplot.\n\n"
            "Als je wil: stuur de .mat-structuur (veld-namen), dan voeg ik de juiste mapping toe."
        )

    base = os.path.splitext(mat_path)[0]

    # figuren
    cmaps_speed = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "RdBu_r"]
    cmaps_dir = ["twilight", "hsv", "twilight_shifted", "viridis", "turbo"]
    cmaps_bs = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "gray", "bone"]

    fig_speed = interactive_section(
        dist, depths, speed_m, vb_depth, basepath=base, suffix="speed",
        title="Snelheidsdoorsnede (afstand vs diepte)",
        cbar_label="Snelheid (m/s)", hover_label="Snelheid (m/s)",
        cmaps=cmaps_speed, cmap_default="viridis"
    )

    fig_dir = interactive_section(
        dist, depths, dir_m, vb_depth, basepath=base, suffix="direction",
        title="Snelheidsrichting (° t.o.v. Noord, richting NAAR)",
        cbar_label="Richting (°)", hover_label="Richting",
        cmaps=cmaps_dir, cmap_default="twilight",
        vmin_fixed=0.0, vmax_fixed=360.0
    )

    fig_track = interactive_boat_track(x, y, basepath=base)

    if bs_m is not None:
        fig_bs = interactive_section(
            dist, depths, bs_m, vb_depth, basepath=base, suffix="backscatter",
            title=f"Backscatter doorsnede (bron: {bs_src})",
            cbar_label="Backscatter (a.u.)", hover_label="Backscatter",
            cmaps=cmaps_bs, cmap_default="cividis"
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
