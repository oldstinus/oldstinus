#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI: selecteer meerdere M9 MAT-bestanden van (ongeveer) dezelfde vaarrichting
en visualiseer ze parallel in 3D op basis van tijdsverschil.

Concept:
- de vroegste track is referentie (offset = 0)
- elke volgende track wordt parallel verschoven met:
    offset_m = delta_dagen_tov_start * schaal_m_per_dag
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

import numpy as np
import pandas as pd
import scipy.io as sio

try:
    import plotly.graph_objects as go  # type: ignore

    _PLOTLY_OK = True
except Exception:
    go = None
    _PLOTLY_OK = False


@dataclass
class Track3DData:
    path: Path
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    speed: np.ndarray
    track_e: np.ndarray
    track_n: np.ndarray
    track_speed_surface: np.ndarray
    time_utc: pd.DatetimeIndex
    ensemble: np.ndarray
    start_time_utc: pd.Timestamp
    note: str = ""


def _m9_time_to_utc(seconds_since_2000: np.ndarray) -> pd.DatetimeIndex:
    base = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    td = pd.to_timedelta(np.asarray(seconds_since_2000, dtype=float).reshape(-1), unit="s")
    return (pd.Timestamp(base) + td).tz_convert("UTC")


def _as_1d(arr: object | None, n: int) -> np.ndarray:
    if arr is None:
        return np.full(n, np.nan, dtype=float)
    out = np.asarray(arr, dtype=float).reshape(-1)
    if out.size >= n:
        out = out[:n]
    else:
        out = np.pad(out, (0, n - out.size), mode="constant", constant_values=np.nan)
    out[~np.isfinite(out)] = np.nan
    return out


def _reorder_velocity(vel_raw: np.ndarray, ns_target: int) -> np.ndarray:
    vel = np.asarray(vel_raw, dtype=float)
    if vel.ndim != 3:
        raise ValueError(f"WaterTrack.Velocity moet 3D zijn, kreeg shape {vel.shape}")

    axes = [0, 1, 2]
    if 4 in vel.shape:
        comp_axis = int(np.where(np.array(vel.shape) == 4)[0][0])
    else:
        comp_axis = min(axes, key=lambda a: abs(vel.shape[a] - 4))

    rem = [a for a in axes if a != comp_axis]
    ens_axis = min(rem, key=lambda a: abs(vel.shape[a] - ns_target))
    cell_axis = [a for a in rem if a != ens_axis][0]
    return np.moveaxis(vel, [cell_axis, comp_axis, ens_axis], [0, 1, 2])


def _surface_speed_from_columns(speed: np.ndarray) -> np.ndarray:
    ncols = speed.shape[1]
    out = np.full(ncols, np.nan, dtype=float)
    for j in range(ncols):
        col = speed[:, j]
        idx = np.where(np.isfinite(col))[0]
        if idx.size > 0:
            out[j] = float(col[idx[0]])
    return out


def read_mat_3d(
    mat_path: str | Path,
    m9_hour_shift: int = -1,
    track_is_ne: bool = False,
    ensemble_step: int = 2,
    cell_step: int = 1,
) -> Track3DData:
    p = Path(mat_path)
    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)

    for key in ("System", "Summary", "BottomTrack", "WaterTrack"):
        if key not in mat:
            raise ValueError(f"{p.name}: structuur '{key}' ontbreekt.")

    sys_obj = mat["System"]
    sum_obj = mat["Summary"]
    bt_obj = mat["BottomTrack"]
    wt_obj = mat["WaterTrack"]

    if not hasattr(sys_obj, "Time"):
        raise ValueError(f"{p.name}: System.Time ontbreekt.")
    if not hasattr(sum_obj, "Track"):
        raise ValueError(f"{p.name}: Summary.Track ontbreekt.")
    if not hasattr(wt_obj, "Velocity"):
        raise ValueError(f"{p.name}: WaterTrack.Velocity ontbreekt.")

    t_sec = np.asarray(sys_obj.Time, dtype=float).reshape(-1)
    track = np.asarray(sum_obj.Track, dtype=float)
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError(f"{p.name}: Summary.Track vorm onverwacht {track.shape}")
    track = track[:, :2]
    if track_is_ne:
        track = track[:, [1, 0]]

    ns = min(len(t_sec), track.shape[0])
    if ns < 2:
        raise ValueError(f"{p.name}: te weinig ensembles ({ns}).")

    t_sec = t_sec[:ns]
    track_e = np.asarray(track[:ns, 0], dtype=float)
    track_n = np.asarray(track[:ns, 1], dtype=float)

    cstart = _as_1d(getattr(sys_obj, "Cell_Start", None), ns)
    csize = _as_1d(getattr(sys_obj, "Cell_Size", None), ns)
    bed = _as_1d(getattr(bt_obj, "BT_Depth", None), ns)

    vel = _reorder_velocity(np.asarray(getattr(wt_obj, "Velocity")), ns_target=ns)
    nc, ncomp, ns_vel = vel.shape
    if ncomp < 2:
        raise ValueError(f"{p.name}: Velocity heeft minder dan 2 componenten: {vel.shape}")
    ns = min(ns, ns_vel)

    track_e = track_e[:ns]
    track_n = track_n[:ns]
    cstart = cstart[:ns]
    csize = csize[:ns]
    bed = bed[:ns]
    t_sec = t_sec[:ns]
    vel = vel[:, :, :ns]

    u = np.asarray(vel[:, 0, :], dtype=float)
    v = np.asarray(vel[:, 1, :], dtype=float)
    speed = np.sqrt(u * u + v * v)

    row_idx = np.arange(nc, dtype=float).reshape(-1, 1)
    depth_abs = cstart.reshape(1, -1) + (row_idx + 0.5) * csize.reshape(1, -1)

    valid = np.isfinite(depth_abs) & np.isfinite(speed) & np.isfinite(track_e.reshape(1, -1)) & np.isfinite(
        track_n.reshape(1, -1)
    )
    valid &= depth_abs > 0.0
    bed_ok = np.isfinite(bed).reshape(1, -1)
    valid &= (~bed_ok) | (depth_abs < bed.reshape(1, -1))

    x = np.tile(track_e.reshape(1, -1), (nc, 1))
    y = np.tile(track_n.reshape(1, -1), (nc, 1))
    z = -depth_abs.copy()

    x[~valid] = np.nan
    y[~valid] = np.nan
    z[~valid] = np.nan
    speed[~valid] = np.nan

    ens_step = max(1, int(ensemble_step))
    c_step = max(1, int(cell_step))
    csel = np.arange(0, ns, ens_step, dtype=int)
    rsel = np.arange(0, nc, c_step, dtype=int)

    x = x[np.ix_(rsel, csel)]
    y = y[np.ix_(rsel, csel)]
    z = z[np.ix_(rsel, csel)]
    speed = speed[np.ix_(rsel, csel)]

    track_e = track_e[csel]
    track_n = track_n[csel]
    t_utc = _m9_time_to_utc(t_sec)
    t_utc = (t_utc + pd.Timedelta(hours=int(m9_hour_shift)))[csel]
    if len(t_utc) == 0:
        raise ValueError(f"{p.name}: geen geldige tijdstempels.")
    start_time = pd.Timestamp(t_utc[0]).tz_convert("UTC")
    ensemble = (csel + 1).astype(int)
    speed_surface = _surface_speed_from_columns(speed)

    if np.count_nonzero(np.isfinite(speed)) < 10:
        raise ValueError(f"{p.name}: onvoldoende geldige snelheidspunten.")

    main_sel = _primary_track_slice(track_e, track_n)
    trimmed_cols = int(track_e.size - len(range(*main_sel.indices(track_e.size))))
    if trimmed_cols > 0:
        x = x[:, main_sel]
        y = y[:, main_sel]
        z = z[:, main_sel]
        speed = speed[:, main_sel]
        track_e = track_e[main_sel]
        track_n = track_n[main_sel]
        t_utc = t_utc[main_sel]
        ensemble = ensemble[main_sel]
        speed_surface = speed_surface[main_sel]
        if len(t_utc) == 0:
            raise ValueError(f"{p.name}: geen geldige transectpunten na trim.")
        start_time = pd.Timestamp(t_utc[0]).tz_convert("UTC")

    note = ""
    med_abs = max(float(np.nanmedian(np.abs(track_e))), float(np.nanmedian(np.abs(track_n))))
    if med_abs < 5000:
        note = "Track lijkt relatief (mogelijk niet gegeorefereerd)."
    if trimmed_cols > 0:
        extra = f"Post-transect staart verwijderd ({trimmed_cols} ensembles)."
        note = f"{note} {extra}".strip() if note else extra

    return Track3DData(
        path=p,
        x=x,
        y=y,
        z=z,
        speed=speed,
        track_e=track_e,
        track_n=track_n,
        track_speed_surface=speed_surface,
        time_utc=t_utc,
        ensemble=ensemble,
        start_time_utc=start_time,
        note=note,
    )


def _fill_1d_linear(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).reshape(-1).copy()
    idx = np.where(np.isfinite(out))[0]
    if idx.size == 0:
        return out
    if idx.size == 1:
        out[:] = out[idx[0]]
        return out
    xx = np.arange(out.size, dtype=float)
    out[:] = np.interp(xx, idx.astype(float), out[idx])
    return out


def _chainage_from_track(track_e: np.ndarray, track_n: np.ndarray) -> np.ndarray:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    valid = np.isfinite(te) & np.isfinite(tn)
    if np.count_nonzero(valid) == 0:
        return np.full(te.size, np.nan, dtype=float)

    te_f = _fill_1d_linear(te)
    tn_f = _fill_1d_linear(tn)
    d = np.hypot(np.diff(te_f), np.diff(tn_f))
    d[~np.isfinite(d)] = 0.0
    s = np.concatenate([[0.0], np.cumsum(d)])

    first = np.where(valid)[0][0]
    s = s - s[first]
    s[~valid] = np.nan
    return s


def _primary_track_slice(
    track_e: np.ndarray,
    track_n: np.ndarray,
    tail_return_frac: float = 0.35,
    peak_keep_frac: float = 0.85,
) -> slice:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    valid = np.isfinite(te) & np.isfinite(tn)
    idx = np.where(valid)[0]
    if idx.size < 3:
        return slice(0, te.size)

    p0 = np.array([float(te[idx[0]]), float(tn[idx[0]])], dtype=float)
    pts = np.column_stack([te[idx], tn[idx]])
    dist = np.hypot(pts[:, 0] - p0[0], pts[:, 1] - p0[1])
    peak_local = int(np.nanargmax(dist))
    peak_dist = float(dist[peak_local])
    if peak_dist < 1e-6 or peak_local >= idx.size - 1:
        return slice(0, te.size)

    tail = dist[peak_local + 1 :]
    if tail.size == 0 or float(np.nanmin(tail)) > tail_return_frac * peak_dist:
        return slice(0, te.size)

    keep_local = np.where(dist >= peak_keep_frac * peak_dist)[0]
    keep_local = keep_local[keep_local >= peak_local]
    if keep_local.size == 0:
        end_local = peak_local
    else:
        end_local = int(keep_local[-1])
    end_idx = int(idx[end_local]) + 1
    return slice(0, end_idx)


def _track_direction_unit(track_e: np.ndarray, track_n: np.ndarray) -> np.ndarray:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    valid = np.isfinite(te) & np.isfinite(tn)
    idx = np.where(valid)[0]
    if idx.size < 2:
        raise ValueError("Track heeft te weinig geldige punten.")

    i0, i1 = int(idx[0]), int(idx[-1])
    vec = np.array([float(te[i1] - te[i0]), float(tn[i1] - tn[i0])], dtype=float)
    nrm = float(np.hypot(vec[0], vec[1]))
    if nrm < 1e-9:
        pts = np.column_stack([te[valid], tn[valid]])
        ctr = np.nanmean(pts, axis=0)
        pts0 = pts - ctr.reshape(1, 2)
        _, _, vh = np.linalg.svd(pts0, full_matrices=False)
        vec = vh[0, :]
        nrm = float(np.hypot(vec[0], vec[1]))
    if nrm < 1e-9:
        vec = np.array([1.0, 0.0], dtype=float)
        nrm = 1.0
    return vec / nrm


def _reference_axis(track_e: np.ndarray, track_n: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    valid = np.isfinite(te) & np.isfinite(tn)
    idx = np.where(valid)[0]
    if idx.size < 2:
        raise ValueError("Referentietrack heeft te weinig geldige punten.")

    i0 = int(idx[0])
    origin_e = float(te[i0])
    origin_n = float(tn[i0])
    d = _track_direction_unit(te, tn)
    # Conventie: positieve normale richting = linkeroever -> rechteroever.
    n = np.array([d[1], -d[0]], dtype=float)
    return origin_e, origin_n, d, n


def _should_reverse_profile(ref_dir_vec: np.ndarray, data: Track3DData) -> bool:
    try:
        data_dir_vec = _track_direction_unit(data.track_e, data.track_n)
    except ValueError:
        return False
    return float(np.dot(ref_dir_vec, data_dir_vec)) < 0.0


def _first_valid_track_point(track_e: np.ndarray, track_n: np.ndarray) -> np.ndarray | None:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    idx = np.where(np.isfinite(te) & np.isfinite(tn))[0]
    if idx.size == 0:
        return None
    i0 = int(idx[0])
    return np.array([float(te[i0]), float(tn[i0])], dtype=float)


def _reverse_profile_data(data: Track3DData) -> Track3DData:
    return replace(
        data,
        x=np.asarray(data.x, dtype=float)[:, ::-1].copy(),
        y=np.asarray(data.y, dtype=float)[:, ::-1].copy(),
        z=np.asarray(data.z, dtype=float)[:, ::-1].copy(),
        speed=np.asarray(data.speed, dtype=float)[:, ::-1].copy(),
        track_e=np.asarray(data.track_e, dtype=float)[::-1].copy(),
        track_n=np.asarray(data.track_n, dtype=float)[::-1].copy(),
        track_speed_surface=np.asarray(data.track_speed_surface, dtype=float)[::-1].copy(),
        time_utc=pd.DatetimeIndex(np.asarray(data.time_utc)[::-1]),
        ensemble=np.asarray(data.ensemble, dtype=int)[::-1].copy(),
    )


def _parallel_xy_for_dataset(
    data: Track3DData,
    origin_e: float,
    origin_n: float,
    normal_vec: np.ndarray,
    offset_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref0 = np.array([float(origin_e), float(origin_n)], dtype=float)
    p0 = _first_valid_track_point(data.track_e, data.track_n)
    if p0 is None:
        shift_e = 0.0
        shift_n = 0.0
    else:
        shift_e = float(ref0[0] - p0[0])
        shift_n = float(ref0[1] - p0[1])

    off_e = float(offset_m) * float(normal_vec[0])
    off_n = float(offset_m) * float(normal_vec[1])

    tx = np.asarray(data.track_e, dtype=float) + shift_e + off_e
    ty = np.asarray(data.track_n, dtype=float) + shift_n + off_n
    xx = np.asarray(data.x, dtype=float) + shift_e + off_e
    yy = np.asarray(data.y, dtype=float) + shift_n + off_n

    invalid = ~np.isfinite(data.speed)
    xx[invalid] = np.nan
    yy[invalid] = np.nan
    return xx, yy, tx, ty


def _fmt_delta(delta: pd.Timedelta) -> str:
    sec = float(delta.total_seconds())
    sign = "-" if sec < 0 else "+"
    sec = abs(sec)
    d = int(sec // 86400)
    h = int((sec % 86400) // 3600)
    m = int((sec % 3600) // 60)
    return f"{sign}{d}d {h:02d}h {m:02d}m"


def _plotly_parallel_controls_post_script(default_scale: str = "Turbo") -> str:
    script = r"""
(function() {
  const gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) return;

  const flowIdx = [];
  for (let i = 0; i < gd.data.length; i++) {
    const t = gd.data[i];
    if (!t || !t.meta || !t.meta.layer) continue;
    const layer = String(t.meta.layer).toLowerCase();
    if (layer === 'flow') {
      flowIdx.push(i);
    }
  }
  if (!gd.layout || !gd.layout.scene) return;

  const scales = ["Turbo", "Viridis", "Plasma", "Cividis", "Jet", "Portland", "RdBu", "Bluered", "YlGnBu"];
  const meta = (gd.layout && gd.layout.meta) ? gd.layout.meta : {};
  const sMin0 = Number(meta.speed_min);
  const sMax0 = Number(meta.speed_max);
  const defaultScale = String(meta.color_scale || __DEFAULT_SCALE__);

  function numOrNan(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }
  const scene0 = gd.layout.scene || {};
  const mode0 = String(scene0.aspectmode || 'data');
  const ratio0 = (scene0.aspectratio && Number.isFinite(Number(scene0.aspectratio.x)) && Number.isFinite(Number(scene0.aspectratio.y)) && Number.isFinite(Number(scene0.aspectratio.z)))
    ? { x: Number(scene0.aspectratio.x), y: Number(scene0.aspectratio.y), z: Number(scene0.aspectratio.z) }
    : { x: 1.0, y: 1.0, z: 1.0 };

  const root = gd.parentElement || gd;
  if (window.getComputedStyle(root).position === 'static') {
    root.style.position = 'relative';
  }

  const panel = document.createElement('div');
  panel.style.cssText = [
    "position:absolute",
    "top:12px",
    "left:12px",
    "z-index:35",
    "background:rgba(255,255,255,0.95)",
    "border:1px solid #a0a0a0",
    "border-radius:8px",
    "padding:8px 10px",
    "font:12px/1.2 Arial,sans-serif",
    "box-shadow:0 2px 8px rgba(0,0,0,0.18)"
  ].join(';');

  const rowStyle = "display:grid;grid-template-columns:140px 120px;gap:8px;align-items:center;margin:4px 0;";
  panel.innerHTML = ''
    + '<div style="font-weight:600;margin-bottom:6px;">Schaal en kleur (HTML)</div>'
    + '<div style="' + rowStyle + '"><label for="m9-xscale">X schaal (track-as)</label><input id="m9-xscale" type="number" step="0.1" min="0.1" value="1"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-zscale">Hoogte overdrijving</label><input id="m9-zscale" type="number" step="0.1" min="0.1" value="1"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-vscale">Snelheid schaal</label><input id="m9-vscale" type="number" step="0.1" min="0.1" value="1"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-scale">Kleurschaal</label><select id="m9-scale"></select></div>'
    + '<div style="margin-top:6px;"><button id="m9-apply" type="button">Toepassen</button> <button id="m9-reset" type="button">Reset</button></div>';
  root.appendChild(panel);

  const xIn = panel.querySelector('#m9-xscale');
  const zIn = panel.querySelector('#m9-zscale');
  const vIn = panel.querySelector('#m9-vscale');
  const scSel = panel.querySelector('#m9-scale');
  const applyBtn = panel.querySelector('#m9-apply');
  const resetBtn = panel.querySelector('#m9-reset');
  if (!xIn || !zIn || !vIn || !scSel || !applyBtn || !resetBtn) return;

  scales.forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    scSel.appendChild(opt);
  });
  scSel.value = scales.includes(defaultScale) ? defaultScale : scales[0];

  function apply() {
    const xScale = Math.max(0.1, numOrNan(xIn.value) || 1.0);
    const zScale = Math.max(0.1, numOrNan(zIn.value) || 1.0);
    const vScale = Math.max(0.1, numOrNan(vIn.value) || 1.0);

    if (Math.abs(xScale - 1.0) < 1e-9 && Math.abs(zScale - 1.0) < 1e-9) {
      if (mode0 === "manual") {
        Plotly.relayout(gd, {
          "scene.aspectmode": "manual",
          "scene.aspectratio.x": ratio0.x,
          "scene.aspectratio.y": ratio0.y,
          "scene.aspectratio.z": ratio0.z,
        });
      } else {
        Plotly.relayout(gd, { "scene.aspectmode": mode0 });
      }
    } else {
      Plotly.relayout(gd, {
        "scene.aspectmode": "manual",
        "scene.aspectratio.x": ratio0.x * xScale,
        "scene.aspectratio.y": ratio0.y,
        "scene.aspectratio.z": ratio0.z * zScale,
      });
    }

    for (const idx of flowIdx) {
      const upd = { colorscale: scSel.value };
      if (Number.isFinite(sMin0) && Number.isFinite(sMax0) && sMax0 > sMin0) {
        upd.cmin = sMin0 / vScale;
        upd.cmax = sMax0 / vScale;
      }
      Plotly.restyle(gd, upd, [idx]);
    }
  }

  function reset() {
    xIn.value = "1";
    zIn.value = "1";
    vIn.value = "1";
    scSel.value = scales.includes(defaultScale) ? defaultScale : scales[0];
    apply();
  }

  applyBtn.addEventListener('click', apply);
  resetBtn.addEventListener('click', reset);
})();
"""
    return script.replace("__DEFAULT_SCALE__", json.dumps(default_scale))


def build_parallel_figure(
    datasets: list[Track3DData],
    spacing_m_per_day: float = 25.0,
    opacity: float = 0.92,
    color_scale: str = "Turbo",
) -> go.Figure:
    if not datasets:
        raise ValueError("Geen datasets beschikbaar voor figuur.")

    speeds = [d.speed[np.isfinite(d.speed)] for d in datasets if np.count_nonzero(np.isfinite(d.speed)) > 0]
    if not speeds:
        raise ValueError("Geen geldige snelheden gevonden.")
    smin = min(float(np.nanpercentile(s, 2)) for s in speeds)
    smax = max(float(np.nanpercentile(s, 98)) for s in speeds)
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        smin = min(float(np.nanmin(s)) for s in speeds)
        smax = max(float(np.nanmax(s)) for s in speeds)

    ref = datasets[0]
    ref_start = ref.start_time_utc
    origin_e, origin_n, dir_vec, normal_vec = _reference_axis(ref.track_e, ref.track_n)

    fig = go.Figure()
    track_colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#ff7f00",
        "#984ea3",
        "#a65628",
        "#f781bf",
        "#999999",
    ]

    for i, d in enumerate(datasets):
        d_plot = d
        was_reversed = False
        if _should_reverse_profile(dir_vec, d):
            d_plot = _reverse_profile_data(d)
            was_reversed = True

        delta = pd.Timestamp(d.start_time_utc).tz_convert("UTC") - pd.Timestamp(ref_start).tz_convert("UTC")
        delta_days = float(delta.total_seconds()) / 86400.0
        offset_m = delta_days * float(spacing_m_per_day)
        xx, yy, tx, ty = _parallel_xy_for_dataset(
            d_plot,
            origin_e,
            origin_n,
            normal_vec,
            offset_m,
        )

        col = track_colors[i % len(track_colors)]
        grp = d.path.name
        t_iso = np.array([pd.Timestamp(t).tz_convert("UTC").isoformat() for t in d_plot.time_utc], dtype=object)

        nr, nc = d_plot.speed.shape
        surf_text = np.empty((nr, nc), dtype=object)
        for c in range(nc):
            ens = int(d_plot.ensemble[c]) if c < len(d_plot.ensemble) else (c + 1)
            ts = t_iso[c] if c < len(t_iso) else ""
            for r in range(nr):
                sp = d_plot.speed[r, c]
                dep = abs(float(d_plot.z[r, c])) if np.isfinite(d_plot.z[r, c]) else np.nan
                sp_txt = f"{float(sp):.3f}" if np.isfinite(sp) else "n/a"
                dep_txt = f"{dep:.3f}" if np.isfinite(dep) else "n/a"
                surf_text[r, c] = (
                    f"<b>{d.path.name}</b><br>"
                    f"ensemble={ens}<br>"
                    f"{ts}<br>"
                    f"Diepte={dep_txt} m<br>"
                    f"Snelheid={sp_txt} m/s<br>"
                    f"Profiel omgedraaid={'ja' if was_reversed else 'nee'}<br>"
                    f"Offset (L->R +)={offset_m:.2f} m<br>"
                    f"Delta t={_fmt_delta(delta)}"
                )

        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=d_plot.z,
                surfacecolor=d_plot.speed,
                hovertext=surf_text,
                hoverinfo="text",
                colorscale=color_scale,
                cmin=smin,
                cmax=smax,
                opacity=float(opacity),
                showscale=(i == 0),
                colorbar=dict(title="Snelheid (m/s)", x=1.03, len=0.74, thickness=18) if i == 0 else None,
                name=f"{d.path.stem} | dt={_fmt_delta(delta)}",
                legendgroup=grp,
                showlegend=True,
                connectgaps=False,
                meta=dict(layer="flow"),
            )
        )

        z_track = np.zeros_like(tx, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=tx,
                y=ty,
                z=z_track,
                mode="lines",
                line=dict(color=col, width=6),
                name=f"{d.path.stem} track",
                legendgroup=grp,
                showlegend=False,
                meta=dict(layer="track"),
                hovertemplate=(
                    f"{d.path.name}<br>"
                    f"Start UTC={pd.Timestamp(d.start_time_utc).tz_convert('UTC').isoformat()}<br>"
                    f"Profiel omgedraaid={'ja' if was_reversed else 'nee'}<br>"
                    f"Delta t={_fmt_delta(delta)}<br>"
                    f"Offset (L->R +)={offset_m:.2f} m<extra></extra>"
                ),
            )
        )

        finite_start = np.where(np.isfinite(tx) & np.isfinite(ty))[0]
        if finite_start.size > 0:
            k0 = int(finite_start[0])
            fig.add_trace(
                go.Scatter3d(
                    x=[tx[k0]],
                    y=[ty[k0]],
                    z=[0.0],
                    mode="markers+text",
                    marker=dict(size=5, color=col),
                    text=[f"T{i+1}"],
                    textposition="top center",
                    name=f"{d.path.stem} start",
                    legendgroup=grp,
                    showlegend=False,
                    meta=dict(layer="start"),
                    hovertemplate=(
                    f"{d.path.name}<br>"
                    f"Start UTC={pd.Timestamp(d.start_time_utc).tz_convert('UTC').isoformat()}<br>"
                    f"Profiel omgedraaid={'ja' if was_reversed else 'nee'}<br>"
                    f"Delta t={_fmt_delta(delta)}<br>"
                    f"Offset (L->R +)={offset_m:.2f} m<extra></extra>"
                ),
                )
            )

    fig.update_layout(
        title=(
            "3D waterkolom in parallelle tijdstracks "
            f"(referentie: {ref.path.name}, schaal: {float(spacing_m_per_day):.2f} m/dag, "
            "profielrichting genormaliseerd, offset + = L->R)"
        ),
        template="plotly_white",
        scene=dict(
            xaxis_title="Easting (parallel verschoven, m)",
            yaxis_title="Northing (parallel verschoven, m)",
            zaxis_title="Diepte (m, negatief)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.75)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=10, r=10, b=10, t=55),
        meta=dict(
            speed_min=float(smin),
            speed_max=float(smax),
            color_scale=str(color_scale),
            ref_x_origin=float(origin_e),
        ),
    )
    return fig


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M9 MAT multi 3D parallelle tijdtracks")
        self.geometry("1140x800")

        self.var_out = tk.StringVar(value=str(Path.cwd() / "m9_parallel_tijdtracks_3d.html"))
        self.var_m9_shift = tk.StringVar(value="-1")
        self.var_track_is_ne = tk.BooleanVar(value=False)
        self.var_ens_step = tk.IntVar(value=2)
        self.var_cell_step = tk.IntVar(value=1)
        self.var_opacity = tk.DoubleVar(value=0.92)
        self.var_spacing_m_per_day = tk.DoubleVar(value=25.0)
        self.var_open_after = tk.BooleanVar(value=True)

        self._mat_files: list[Path] = []
        self._build()

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(3, weight=1)

        files_box = ttk.LabelFrame(frm, text="MAT-bestanden", padding=10)
        files_box.grid(row=0, column=0, sticky="nsew")

        btn_row = ttk.Frame(files_box)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Toevoegen...", command=self._add_files).pack(side="left")
        ttk.Button(btn_row, text="Verwijder selectie", command=self._remove_selected).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Leegmaken", command=self._clear_files).pack(side="left", padx=(8, 0))

        list_wrap = ttk.Frame(files_box)
        list_wrap.pack(fill="both", expand=True, pady=(8, 0))
        self.listbox = tk.Listbox(list_wrap, height=13, selectmode=tk.EXTENDED)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        opt = ttk.LabelFrame(frm, text="3D opties", padding=10)
        opt.grid(row=1, column=0, sticky="we", pady=(10, 0))

        ttk.Label(opt, text="M9 tijdshift (uren):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            opt,
            textvariable=self.var_m9_shift,
            values=["-2", "-1", "0", "+1", "+2"],
            width=6,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Checkbutton(opt, text="Summary.Track is N,E (swap)", variable=self.var_track_is_ne).grid(
            row=0, column=2, sticky="w", padx=(14, 0)
        )

        ttk.Label(opt, text="Ensemble step:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_ens_step, width=8).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(opt, text="Cell step:").grid(row=1, column=2, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_cell_step, width=8).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(opt, text="Opacity (0-1):").grid(row=1, column=4, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_opacity, width=8).grid(row=1, column=5, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Parallelafstand per dag (m/dag):").grid(row=2, column=0, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_spacing_m_per_day, width=10).grid(
            row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        out_box = ttk.LabelFrame(frm, text="Output HTML", padding=10)
        out_box.grid(row=2, column=0, sticky="we", pady=(10, 0))
        out_box.grid_columnconfigure(0, weight=1)

        ttk.Entry(out_box, textvariable=self.var_out, width=110).grid(row=0, column=0, sticky="we")
        ttk.Button(out_box, text="Opslaan als...", command=self._pick_out).grid(row=0, column=1, padx=(8, 0))
        ttk.Checkbutton(out_box, text="Open na export", variable=self.var_open_after).grid(row=1, column=0, sticky="w", pady=(8, 0))

        bot = ttk.Frame(frm)
        bot.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        bot.grid_columnconfigure(0, weight=1)
        bot.grid_rowconfigure(2, weight=1)

        btns = ttk.Frame(bot)
        btns.grid(row=0, column=0, sticky="we")
        ttk.Button(btns, text="Maak 3D HTML", command=self._run).pack(side="left")
        ttk.Button(btns, text="Sluiten", command=self.destroy).pack(side="right")

        ttk.Label(bot, text="Log").grid(row=1, column=0, sticky="w")
        self.txt = tk.Text(bot, height=14, wrap="word")
        self.txt.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

    def _log(self, s: str) -> None:
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, "end")
        for p in self._mat_files:
            self.listbox.insert("end", str(p))

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Kies MAT-bestanden", filetypes=[("MAT", "*.mat"), ("All", "*.*")])
        if not paths:
            return
        for s in paths:
            p = Path(s)
            if p not in self._mat_files:
                self._mat_files.append(p)
        self._refresh_listbox()

    def _remove_selected(self) -> None:
        idx = sorted(self.listbox.curselection(), reverse=True)
        if not idx:
            return
        for i in idx:
            del self._mat_files[i]
        self._refresh_listbox()

    def _clear_files(self) -> None:
        self._mat_files = []
        self._refresh_listbox()

    def _pick_out(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Kies output HTML",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("All", "*.*")],
        )
        if p:
            self.var_out.set(p)

    def _run(self) -> None:
        try:
            if not _PLOTLY_OK:
                messagebox.showerror("Dependency", "plotly niet beschikbaar. Installeer: pip install plotly")
                return
            if not self._mat_files:
                messagebox.showerror("Input", "Voeg eerst een of meer MAT-bestanden toe.")
                return

            out_html = self.var_out.get().strip()
            if not out_html:
                messagebox.showerror("Output", "Geef een output HTML-bestand op.")
                return

            m9_shift = int(self.var_m9_shift.get())
            track_is_ne = bool(self.var_track_is_ne.get())
            ens_step = max(1, int(self.var_ens_step.get()))
            cell_step = max(1, int(self.var_cell_step.get()))
            opacity = min(1.0, max(0.05, float(self.var_opacity.get())))
            colorscale = "Turbo"
            spacing_m_per_day = float(self.var_spacing_m_per_day.get())
            open_after = bool(self.var_open_after.get())

            datasets: list[Track3DData] = []
            failed: list[str] = []

            self._log("Inlezen MAT-bestanden...")
            for p in self._mat_files:
                try:
                    d = read_mat_3d(
                        p,
                        m9_hour_shift=m9_shift,
                        track_is_ne=track_is_ne,
                        ensemble_step=ens_step,
                        cell_step=cell_step,
                    )
                    datasets.append(d)
                    self._log(
                        f"  OK: {p.name} | start={pd.Timestamp(d.start_time_utc).isoformat()} | "
                        f"ensembles={d.speed.shape[1]} | cells={d.speed.shape[0]}"
                    )
                    if d.note:
                        self._log(f"      note: {d.note}")
                except Exception as e:
                    failed.append(f"{p.name}: {e}")
                    self._log(f"  ERROR: {p.name} -> {e}")

            if not datasets:
                raise ValueError("Geen geldig MAT-bestand kunnen verwerken.")

            datasets.sort(key=lambda d: pd.Timestamp(d.start_time_utc).value)
            ref_t = pd.Timestamp(datasets[0].start_time_utc).tz_convert("UTC")
            self._log("")
            self._log("Trackvolgorde (op starttijd):")
            for i, d in enumerate(datasets, start=1):
                dt_i = pd.Timestamp(d.start_time_utc).tz_convert("UTC")
                delta = dt_i - ref_t
                delta_days = float(delta.total_seconds()) / 86400.0
                off = delta_days * spacing_m_per_day
                self._log(
                    f"  T{i:02d} | {d.path.name} | start={dt_i.isoformat()} | "
                    f"delta={_fmt_delta(delta)} | offset={off:.2f} m"
                )

            self._log("")
            self._log("3D figuur bouwen...")
            fig = build_parallel_figure(
                datasets=datasets,
                spacing_m_per_day=spacing_m_per_day,
                opacity=opacity,
                color_scale=colorscale,
            )

            outp = Path(out_html)
            outp.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(
                str(outp),
                include_plotlyjs="cdn",
                full_html=True,
                config={"responsive": True},
                post_script=_plotly_parallel_controls_post_script(default_scale=colorscale),
            )
            self._log(f"Klaar: {outp}")

            if failed:
                self._log("")
                self._log("Bestanden met fouten:")
                for s in failed:
                    self._log(f"  - {s}")

            if open_after:
                try:
                    webbrowser.open(outp.resolve().as_uri(), new=2)
                except Exception:
                    pass

            messagebox.showinfo("Klaar", f"3D HTML gemaakt met {len(datasets)} track(s).")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"FOUT: {e}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
