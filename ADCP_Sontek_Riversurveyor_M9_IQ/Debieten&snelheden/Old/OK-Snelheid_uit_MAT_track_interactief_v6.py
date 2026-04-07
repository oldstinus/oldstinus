#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
M9 / SonTek QRev (.mat) – interactieve snelheidsdoorsnede + boottrack (Matplotlib)

V6 – fixes op basis van jouw laatste feedback
--------------------------------------------
1) **Vert. x slider verwijderd** (zoals gevraagd).
2) **Max diepte knipt echt af**:
   - Alle cellen dieper dan "Max diepte (m)" worden **gemaskeerd (NaN)** én de y-lim wordt aangepast.
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

    # integreer – eenvoudige dead-reckoning
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


def compute_speed_section(water_track, bottom_track, system, setup):
    """
    Bouw snelheidsdoorsnede:
      - speed = sqrt(east^2 + north^2) per cell & ensemble
      - maskeer cellen onder VB_Depth
    Output:
      depths (n_cells,), speed_masked (n_cells, n_ens), vb_depth (n_ens,)
    """
    vel = np.squeeze(water_track.Velocity)  # (n_cells, 4, n_ens)
    if vel.ndim != 3 or vel.shape[1] < 2:
        raise ValueError(f"Onverwachte WaterTrack.Velocity shape: {vel.shape}")
    east = vel[:, 0, :]
    north = vel[:, 1, :]
    speed = np.sqrt(east ** 2 + north ** 2)

    n_cells, n_ens = speed.shape
    vb_depth = np.squeeze(bottom_track.VB_Depth).astype(float).ravel()
    if vb_depth.size != n_ens:
        raise ValueError(f"VB_Depth lengte ({vb_depth.size}) != ensembles ({n_ens})")

    depths = compute_cell_depths(system, setup, n_cells).astype(float).ravel()
    speed_masked = speed.astype(float).copy()

    # maskeer onder de bodem
    for j in range(n_ens):
        bd = vb_depth[j]
        if np.isfinite(bd):
            speed_masked[depths > bd, j] = np.nan

    return depths, speed_masked, vb_depth


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
    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # aspect gelijk
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
    s_alpha = Slider(ax_alpha, "Bodem α", 0.0, 1.0, valinit=alpha0, valfmt="%.2f")

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

    # initial state
    angle0 = 0.0
    lw0 = 1.5
    ms0 = 3.0
    show_pts0 = True

    # plot
    (ln,) = ax.plot(x, y, lw=lw0)
    sc = ax.scatter(x, y, s=ms0**2, alpha=0.7) if show_pts0 else None

    # controls
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

        if aspect1:
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("auto")

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def _reset(_evt=None):
        s_ang.set_val(angle0)
        s_lw.set_val(lw0)
        s_ms.set_val(ms0)
        # reset checks naar default
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

    # track
    x, y, _t = compute_boat_track(bottom_track, system)
    dist = compute_transect_distance(x, y)

    # speed section
    depths, speed_masked, vb_depth = compute_speed_section(water_track, bottom_track, system, setup)

    base = os.path.splitext(mat_path)[0]

    fig1 = interactive_velocity_section(dist, depths, speed_masked, vb_depth, basepath=base)
    fig2 = interactive_boat_track(x, y, basepath=base)

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
