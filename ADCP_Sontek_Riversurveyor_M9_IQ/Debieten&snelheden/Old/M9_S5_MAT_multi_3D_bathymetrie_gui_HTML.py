#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI: selecteer meerdere M9 MAT-bestanden van hetzelfde profiel en visualiseer
enkel de bathymetrie parallel in 3D op basis van tijdsverschil.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
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
class BathyTrackData:
    path: Path
    track_e: np.ndarray
    track_n: np.ndarray
    bathy_depth: np.ndarray
    bathy_depth_raw: np.ndarray
    bathy_filled_mask: np.ndarray
    time_utc: pd.DatetimeIndex
    ensemble: np.ndarray
    start_time_utc: pd.Timestamp
    depth_source: str
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
    end_local = int(keep_local[-1]) if keep_local.size else peak_local
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
    n = np.array([d[1], -d[0]], dtype=float)
    return origin_e, origin_n, d, n


def _should_reverse_profile(ref_dir_vec: np.ndarray, data: BathyTrackData) -> bool:
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


def _reverse_profile_data(data: BathyTrackData) -> BathyTrackData:
    return replace(
        data,
        track_e=np.asarray(data.track_e, dtype=float)[::-1].copy(),
        track_n=np.asarray(data.track_n, dtype=float)[::-1].copy(),
        bathy_depth=np.asarray(data.bathy_depth, dtype=float)[::-1].copy(),
        bathy_depth_raw=np.asarray(data.bathy_depth_raw, dtype=float)[::-1].copy(),
        bathy_filled_mask=np.asarray(data.bathy_filled_mask, dtype=bool)[::-1].copy(),
        time_utc=pd.DatetimeIndex(np.asarray(data.time_utc)[::-1]),
        ensemble=np.asarray(data.ensemble, dtype=int)[::-1].copy(),
    )


def _parallel_xy_for_dataset(
    data: BathyTrackData,
    origin_e: float,
    origin_n: float,
    normal_vec: np.ndarray,
    offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    return tx, ty


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


def _fmt_delta(delta: pd.Timedelta) -> str:
    sec = float(delta.total_seconds())
    sign = "-" if sec < 0 else "+"
    sec = abs(sec)
    d = int(sec // 86400)
    h = int((sec % 86400) // 3600)
    m = int((sec % 3600) // 60)
    return f"{sign}{d}d {h:02d}h {m:02d}m"


def _source_label(depth_source: str) -> str:
    return "BottomTrack" if str(depth_source).lower() == "bt" else "Vertical Beam"


def _bridge_segments(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    raw_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    raw = np.asarray(raw_depth, dtype=float).reshape(-1)
    raw_valid = np.isfinite(raw)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    n = raw.size
    i = 0
    while i < n:
        if raw_valid[i] or not np.isfinite(z[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and (not raw_valid[j + 1]) and np.isfinite(z[j + 1]):
            j += 1
        i0 = i - 1 if i > 0 and np.isfinite(z[i - 1]) else i
        i1 = j + 1 if (j + 1) < n and np.isfinite(z[j + 1]) else j
        if i1 > i0:
            xs.extend(x[i0 : i1 + 1].tolist())
            ys.extend(y[i0 : i1 + 1].tolist())
            zs.extend(z[i0 : i1 + 1].tolist())
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)
        i = j + 1
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float)


def read_mat_bathy(
    mat_path: str | Path,
    depth_source: str = "bt",
    m9_hour_shift: int = -1,
    track_is_ne: bool = False,
    ensemble_step: int = 2,
) -> BathyTrackData:
    p = Path(mat_path)
    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)

    for key in ("System", "Summary", "BottomTrack"):
        if key not in mat:
            raise ValueError(f"{p.name}: structuur '{key}' ontbreekt.")

    sys_obj = mat["System"]
    sum_obj = mat["Summary"]
    bt_obj = mat["BottomTrack"]

    if not hasattr(sys_obj, "Time"):
        raise ValueError(f"{p.name}: System.Time ontbreekt.")
    if not hasattr(sum_obj, "Track"):
        raise ValueError(f"{p.name}: Summary.Track ontbreekt.")

    depth_source = str(depth_source).strip().lower()
    if depth_source not in {"bt", "vb"}:
        raise ValueError(f"Onbekende dieptebron: {depth_source}")
    depth_field = "BT_Depth" if depth_source == "bt" else "VB_Depth"

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

    bathy_raw = _as_1d(getattr(bt_obj, depth_field, None), ns)
    ns = min(ns, bathy_raw.size)
    if ns < 2:
        raise ValueError(f"{p.name}: dieptebron '{depth_field}' bevat te weinig data.")

    t_sec = t_sec[:ns]
    track_e = np.asarray(track[:ns, 0], dtype=float)
    track_n = np.asarray(track[:ns, 1], dtype=float)
    bathy_raw = np.asarray(bathy_raw[:ns], dtype=float)
    bathy_raw[~np.isfinite(bathy_raw) | (bathy_raw <= 0.0)] = np.nan

    ens_step = max(1, int(ensemble_step))
    csel = np.arange(0, ns, ens_step, dtype=int)
    track_e = track_e[csel]
    track_n = track_n[csel]
    bathy_raw = bathy_raw[csel]
    t_utc = _m9_time_to_utc(t_sec)
    t_utc = (t_utc + pd.Timedelta(hours=int(m9_hour_shift)))[csel]
    if len(t_utc) == 0:
        raise ValueError(f"{p.name}: geen geldige tijdstempels.")
    ensemble = (csel + 1).astype(int)

    main_sel = _primary_track_slice(track_e, track_n)
    trimmed_cols = int(track_e.size - len(range(*main_sel.indices(track_e.size))))
    if trimmed_cols > 0:
        track_e = track_e[main_sel]
        track_n = track_n[main_sel]
        bathy_raw = bathy_raw[main_sel]
        t_utc = t_utc[main_sel]
        ensemble = ensemble[main_sel]

    if np.count_nonzero(np.isfinite(bathy_raw)) < 2:
        raise ValueError(f"{p.name}: onvoldoende geldige dieptepunten in {depth_field}.")

    bathy_filled = _fill_1d_linear(bathy_raw)
    filled_mask = ~np.isfinite(bathy_raw) & np.isfinite(bathy_filled)
    if np.count_nonzero(np.isfinite(bathy_filled)) < 2:
        raise ValueError(f"{p.name}: bathymetrie kon niet worden aangevuld.")

    start_time = pd.Timestamp(t_utc[0]).tz_convert("UTC")
    note = ""
    med_abs = max(float(np.nanmedian(np.abs(track_e))), float(np.nanmedian(np.abs(track_n))))
    if med_abs < 5000:
        note = "Track lijkt relatief (mogelijk niet gegeorefereerd)."
    if trimmed_cols > 0:
        extra = f"Post-transect staart verwijderd ({trimmed_cols} ensembles)."
        note = f"{note} {extra}".strip() if note else extra
    nfill = int(np.count_nonzero(filled_mask))
    if nfill > 0:
        extra = f"Ontbrekende bodemdata aangevuld ({nfill} punten)."
        note = f"{note} {extra}".strip() if note else extra

    return BathyTrackData(
        path=p,
        track_e=track_e,
        track_n=track_n,
        bathy_depth=bathy_filled,
        bathy_depth_raw=bathy_raw,
        bathy_filled_mask=filled_mask,
        time_utc=t_utc,
        ensemble=ensemble,
        start_time_utc=start_time,
        depth_source=depth_source,
        note=note,
    )


def _plotly_bathy_controls_post_script() -> str:
    return r"""
(function() {
  const gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly || !gd.layout || !gd.layout.scene) return;
  const scene0 = gd.layout.scene || {};
  const mode0 = String(scene0.aspectmode || 'data');
  const ratio0 = (scene0.aspectratio &&
                  Number.isFinite(Number(scene0.aspectratio.x)) &&
                  Number.isFinite(Number(scene0.aspectratio.y)) &&
                  Number.isFinite(Number(scene0.aspectratio.z)))
    ? { x: Number(scene0.aspectratio.x), y: Number(scene0.aspectratio.y), z: Number(scene0.aspectratio.z) }
    : { x: 1.0, y: 1.0, z: 1.0 };
  const root = gd.parentElement || gd;
  if (window.getComputedStyle(root).position === 'static') root.style.position = 'relative';
  const panel = document.createElement('div');
  panel.style.cssText = 'position:absolute;top:12px;left:12px;z-index:35;background:rgba(255,255,255,0.95);'
    + 'border:1px solid #a0a0a0;border-radius:8px;padding:8px 10px;font:12px/1.2 Arial,sans-serif;'
    + 'box-shadow:0 2px 8px rgba(0,0,0,0.18)';
  const rowStyle = 'display:grid;grid-template-columns:140px 120px;gap:8px;align-items:center;margin:4px 0;';
  panel.innerHTML = ''
    + '<div style="font-weight:600;margin-bottom:6px;">Schaal (HTML)</div>'
    + '<div style="' + rowStyle + '"><label for="m9-xscale">X schaal (track-as)</label><input id="m9-xscale" type="number" step="0.1" min="0.1" value="1"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-zscale">Hoogte overdrijving</label><input id="m9-zscale" type="number" step="0.1" min="0.1" value="1"></div>'
    + '<div style="margin-top:6px;"><button id="m9-apply" type="button">Toepassen</button> <button id="m9-reset" type="button">Reset</button></div>';
  root.appendChild(panel);
  const xIn = panel.querySelector('#m9-xscale');
  const zIn = panel.querySelector('#m9-zscale');
  const applyBtn = panel.querySelector('#m9-apply');
  const resetBtn = panel.querySelector('#m9-reset');
  if (!xIn || !zIn || !applyBtn || !resetBtn) return;
  function numOrNan(v) { const n = Number(v); return Number.isFinite(n) ? n : NaN; }
  function apply() {
    const xScale = Math.max(0.1, numOrNan(xIn.value) || 1.0);
    const zScale = Math.max(0.1, numOrNan(zIn.value) || 1.0);
    if (Math.abs(xScale - 1.0) < 1e-9 && Math.abs(zScale - 1.0) < 1e-9) {
      if (mode0 === 'manual') {
        Plotly.relayout(gd, {'scene.aspectmode': 'manual', 'scene.aspectratio.x': ratio0.x, 'scene.aspectratio.y': ratio0.y, 'scene.aspectratio.z': ratio0.z});
      } else {
        Plotly.relayout(gd, {'scene.aspectmode': mode0});
      }
    } else {
      Plotly.relayout(gd, {'scene.aspectmode': 'manual', 'scene.aspectratio.x': ratio0.x * xScale, 'scene.aspectratio.y': ratio0.y, 'scene.aspectratio.z': ratio0.z * zScale});
    }
  }
  function reset() { xIn.value = '1'; zIn.value = '1'; apply(); }
  applyBtn.addEventListener('click', apply);
  resetBtn.addEventListener('click', reset);
})();
"""


def _normalized_datasets(
    datasets: list[BathyTrackData],
) -> tuple[list[BathyTrackData], np.ndarray]:
    if not datasets:
        return [], np.array([1.0, 0.0], dtype=float)
    ref_dir = _reference_axis(datasets[0].track_e, datasets[0].track_n)[2]
    out: list[BathyTrackData] = []
    for d in datasets:
        out.append(_reverse_profile_data(d) if _should_reverse_profile(ref_dir, d) else d)
    return out, ref_dir


def _profile_points(data: BathyTrackData) -> tuple[np.ndarray, np.ndarray]:
    s = _chainage_from_track(data.track_e, data.track_n)
    z = np.asarray(data.bathy_depth, dtype=float).reshape(-1)
    valid = np.isfinite(s) & np.isfinite(z)
    sv = s[valid]
    zv = z[valid]
    if sv.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    uq, idx = np.unique(sv, return_index=True)
    return uq.astype(float), zv[idx].astype(float)


def _profile_grid_step(profiles: list[tuple[np.ndarray, np.ndarray]]) -> float:
    steps: list[float] = []
    for sx, _ in profiles:
        if sx.size < 2:
            continue
        ds = np.diff(sx)
        ds = ds[np.isfinite(ds) & (ds > 1e-9)]
        if ds.size > 0:
            steps.append(float(np.nanmedian(ds)))
    if not steps:
        return 0.05
    return max(0.01, float(np.nanmedian(steps)))


def _profile_rmse_for_shift(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    shift_m: float,
    grid_step: float,
    min_overlap_points: int = 8,
) -> tuple[float, int]:
    if ref_x.size < 2 or test_x.size < 2:
        return float("inf"), 0
    lo = max(float(ref_x[0]), float(test_x[0] + shift_m))
    hi = min(float(ref_x[-1]), float(test_x[-1] + shift_m))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("inf"), 0
    n = int(np.floor((hi - lo) / grid_step)) + 1
    if n < min_overlap_points:
        return float("inf"), n
    xg = lo + np.arange(n, dtype=float) * grid_step
    ref_g = np.interp(xg, ref_x, ref_y)
    test_g = np.interp(xg - shift_m, test_x, test_y)
    diff = ref_g - test_g
    valid = np.isfinite(diff)
    nv = int(np.count_nonzero(valid))
    if nv < min_overlap_points:
        return float("inf"), nv
    rmse = float(np.sqrt(np.nanmean(diff[valid] * diff[valid])))
    return rmse, nv


def _best_profile_shift(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    grid_step: float,
) -> float:
    if ref_x.size < 2 or test_x.size < 2:
        return 0.0
    ref_len = float(ref_x[-1] - ref_x[0])
    test_len = float(test_x[-1] - test_x[0])
    max_shift = max(0.25, 0.75 * min(ref_len, test_len))
    coarse_step = max(grid_step, max_shift / 60.0)
    coarse_shifts = np.arange(-max_shift, max_shift + 0.5 * coarse_step, coarse_step, dtype=float)

    best_shift = 0.0
    best_rmse = float("inf")
    best_overlap = -1
    for shift in coarse_shifts:
        rmse, overlap = _profile_rmse_for_shift(ref_x, ref_y, test_x, test_y, float(shift), grid_step)
        if overlap > best_overlap or (overlap == best_overlap and rmse < best_rmse):
            best_shift = float(shift)
            best_rmse = rmse
            best_overlap = overlap

    fine_step = max(grid_step / 5.0, 0.01)
    fine_span = max(2.0 * coarse_step, fine_step)
    fine_shifts = np.arange(best_shift - fine_span, best_shift + fine_span + 0.5 * fine_step, fine_step, dtype=float)
    for shift in fine_shifts:
        rmse, overlap = _profile_rmse_for_shift(ref_x, ref_y, test_x, test_y, float(shift), grid_step)
        if overlap > best_overlap or (overlap == best_overlap and rmse < best_rmse):
            best_shift = float(shift)
            best_rmse = rmse
            best_overlap = overlap
    return best_shift


def _profile_matrix_from_shifts(
    profiles: list[tuple[str, np.ndarray, np.ndarray]],
    shifts: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    max_dist = 0.0
    for name, sx, _ in profiles:
        if sx.size == 0:
            continue
        max_dist = max(max_dist, float(np.nanmax(sx + float(shifts.get(name, 0.0)))))
    if max_dist <= 0.0:
        raise ValueError("Onvoldoende profielafstand voor gemiddeld profiel.")

    grid_step = _profile_grid_step([(sx, sy) for _, sx, sy in profiles])
    grid = np.arange(0.0, max_dist + 0.5 * grid_step, grid_step, dtype=float)
    cols: dict[str, np.ndarray] = {}
    for name, sx, sy in profiles:
        col = np.full(grid.size, np.nan, dtype=float)
        if sx.size == 1:
            col[:] = sy[0]
        elif sx.size >= 2:
            shift = float(shifts.get(name, 0.0))
            x_shift = sx + shift
            inside = (grid >= float(x_shift[0])) & (grid <= float(x_shift[-1]))
            if np.count_nonzero(inside) > 0:
                col[inside] = np.interp(grid[inside], x_shift, sy)
        cols[name] = col
    return grid, cols


def _columnwise_nanmean_std_count(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Matrix voor gemiddelde profiel moet 2D zijn.")
    valid = np.isfinite(arr)
    count = np.count_nonzero(valid, axis=1).astype(int)
    sumv = np.nansum(arr, axis=1)
    mean = np.full(arr.shape[0], np.nan, dtype=float)
    ok = count > 0
    mean[ok] = sumv[ok] / count[ok]

    std = np.full(arr.shape[0], np.nan, dtype=float)
    if np.any(ok):
        centered = np.where(valid, arr - mean.reshape(-1, 1), np.nan)
        var = np.nansum(centered * centered, axis=1)
        std[ok] = np.sqrt(var[ok] / count[ok])
    return mean, std, count


def build_mean_bathy_profile(
    datasets: list[BathyTrackData],
) -> tuple[pd.DataFrame, list[str]]:
    normed, _ = _normalized_datasets(datasets)
    if not normed:
        raise ValueError("Geen datasets voor gemiddeld profiel.")

    profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
    for d in normed:
        sx, sy = _profile_points(d)
        profiles.append((d.path.stem, sx, sy))

    valid_profiles = [(name, sx, sy) for name, sx, sy in profiles if sx.size >= 2]
    if not valid_profiles:
        raise ValueError("Onvoldoende geldige profielen voor gemiddeld profiel.")

    grid_step = _profile_grid_step([(sx, sy) for _, sx, sy in valid_profiles])
    anchor_name, anchor_x, anchor_y = max(valid_profiles, key=lambda item: float(item[1][-1] - item[1][0]))
    shifts: dict[str, float] = {name: 0.0 for name, _, _ in profiles}
    for name, sx, sy in valid_profiles:
        if name == anchor_name:
            continue
        shifts[name] = _best_profile_shift(anchor_x, anchor_y, sx, sy, grid_step)

    grid0, cols0 = _profile_matrix_from_shifts(profiles, shifts)
    mat0 = np.column_stack([cols0[name] for name, _, _ in profiles]) if profiles else np.empty((grid0.size, 0), dtype=float)
    mean0, _, _ = _columnwise_nanmean_std_count(mat0) if mat0.size else (
        np.full(grid0.size, np.nan, dtype=float),
        np.full(grid0.size, np.nan, dtype=float),
        np.zeros(grid0.size, dtype=int),
    )
    mean_valid = np.isfinite(grid0) & np.isfinite(mean0)
    mean_x = grid0[mean_valid]
    mean_y = mean0[mean_valid]
    if mean_x.size >= 2:
        for name, sx, sy in valid_profiles:
            if name == anchor_name:
                continue
            shifts[name] = _best_profile_shift(mean_x, mean_y, sx, sy, grid_step)

    grid, cols = _profile_matrix_from_shifts(profiles, shifts)
    order = [name for name, _, _ in profiles]

    mat = np.column_stack([cols[name] for name in order]) if order else np.empty((grid.size, 0), dtype=float)
    mean_depth, std_depth, n_tracks = _columnwise_nanmean_std_count(mat) if mat.size else (
        np.full(grid.size, np.nan, dtype=float),
        np.full(grid.size, np.nan, dtype=float),
        np.zeros(grid.size, dtype=int),
    )

    df = pd.DataFrame(
        {
            "distance_m": grid,
            "mean_depth_m": mean_depth,
            "std_depth_m": std_depth,
            "n_tracks": n_tracks.astype(int),
        }
    )
    for name in order:
        df[f"depth_{name}_m"] = cols[name]
        df[f"shift_{name}_m"] = float(shifts.get(name, 0.0))
    return df, order


def build_mean_bathy_figure(
    profile_df: pd.DataFrame,
    track_names: list[str],
    depth_source_label: str,
) -> go.Figure:
    fig = go.Figure()
    dist = np.asarray(profile_df["distance_m"], dtype=float)
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#a65628", "#f781bf", "#999999"]

    for i, name in enumerate(track_names):
        col_name = f"depth_{name}_m"
        if col_name not in profile_df:
            continue
        yy = np.asarray(profile_df[col_name], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=yy,
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=1.5),
                name=name,
                opacity=0.6,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dist,
            y=np.asarray(profile_df["mean_depth_m"], dtype=float),
            mode="lines",
            line=dict(color="#111111", width=4),
            name="Gemiddeld profiel",
        )
    )

    fig.update_layout(
        title=f"Gemiddeld bathymetrisch profiel ({depth_source_label})",
        template="plotly_white",
        xaxis_title="Profielafstand vanaf start (m)",
        yaxis_title="Diepte (m)",
        legend=dict(itemsizing="constant"),
        margin=dict(l=60, r=30, b=50, t=55),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def build_parallel_bathy_figure(
    datasets: list[BathyTrackData],
    spacing_m_per_day: float = 25.0,
    bridge_color: str = "#111111",
    depth_exaggeration: float = 12.0,
) -> go.Figure:
    if not datasets:
        raise ValueError("Geen datasets beschikbaar voor figuur.")

    normed, _ = _normalized_datasets(datasets)
    all_depths = [d.bathy_depth[np.isfinite(d.bathy_depth)] for d in normed]
    all_depths = [d for d in all_depths if d.size > 0]
    if not all_depths:
        raise ValueError("Geen geldige bathymetrie gevonden.")

    z_exag = max(0.1, float(depth_exaggeration))
    ref = normed[0]
    ref_start = ref.start_time_utc
    origin_e, origin_n, _, normal_vec = _reference_axis(ref.track_e, ref.track_n)

    fig = go.Figure()
    track_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#a65628", "#f781bf", "#999999"]
    show_bridge_legend = True
    for i, (d, d_plot) in enumerate(zip(datasets, normed)):
        was_reversed = d_plot is not d

        delta = pd.Timestamp(d.start_time_utc).tz_convert("UTC") - pd.Timestamp(ref_start).tz_convert("UTC")
        delta_days = float(delta.total_seconds()) / 86400.0
        offset_m = delta_days * float(spacing_m_per_day)
        tx, ty = _parallel_xy_for_dataset(d_plot, origin_e, origin_n, normal_vec, offset_m)

        col = track_colors[i % len(track_colors)]
        grp = d.path.name
        t_iso = np.array([pd.Timestamp(t).tz_convert("UTC").isoformat() for t in d_plot.time_utc], dtype=object)
        z_full = -np.asarray(d_plot.bathy_depth, dtype=float)
        z_raw = -np.asarray(d_plot.bathy_depth_raw, dtype=float)

        hover = np.empty(z_full.size, dtype=object)
        for j in range(z_full.size):
            depth_txt = f"{float(d_plot.bathy_depth[j]):.3f}" if np.isfinite(d_plot.bathy_depth[j]) else "n/a"
            raw_txt = f"{float(d_plot.bathy_depth_raw[j]):.3f}" if np.isfinite(d_plot.bathy_depth_raw[j]) else "n/a"
            ens = int(d_plot.ensemble[j]) if j < len(d_plot.ensemble) else (j + 1)
            ts = t_iso[j] if j < len(t_iso) else ""
            hover[j] = (
                f"<b>{d.path.name}</b><br>"
                f"ensemble={ens}<br>"
                f"{ts}<br>"
                f"Diepte bron={_source_label(d_plot.depth_source)}<br>"
                f"Diepte gevuld={depth_txt} m<br>"
                f"Ruwe diepte={raw_txt} m<br>"
                f"Aangevuld={'ja' if bool(d_plot.bathy_filled_mask[j]) else 'nee'}<br>"
                f"Profiel omgedraaid={'ja' if was_reversed else 'nee'}<br>"
                f"Offset (L->R +)={offset_m:.2f} m<br>"
                f"Delta t={_fmt_delta(delta)}"
            )

        fig.add_trace(
            go.Scatter3d(
                x=tx,
                y=ty,
                z=z_raw,
                mode="lines",
                line=dict(color=col, width=7),
                name=f"{d.path.stem} | dt={_fmt_delta(delta)}",
                legendgroup=grp,
                showlegend=True,
                hovertext=hover,
                hoverinfo="text",
                meta=dict(layer="bathy_raw"),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=tx,
                y=ty,
                z=np.zeros_like(tx, dtype=float),
                mode="lines",
                line=dict(color=col, width=4),
                name=f"{d.path.stem} track",
                legendgroup=grp,
                showlegend=False,
                hovertemplate=(
                    f"{d.path.name}<br>"
                    f"Track op wateroppervlak<br>"
                    f"Profiel omgedraaid={'ja' if was_reversed else 'nee'}<br>"
                    f"Delta t={_fmt_delta(delta)}<br>"
                    f"Offset (L->R +)={offset_m:.2f} m<extra></extra>"
                ),
                meta=dict(layer="track"),
            )
        )

        bx, by, bz = _bridge_segments(tx, ty, z_full, d_plot.bathy_depth_raw)
        if bx.size > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=bx,
                    y=by,
                    z=bz,
                    mode="lines",
                    line=dict(color=bridge_color, width=7),
                    name="Aangevulde delen",
                    legendgroup="filled_segments",
                    showlegend=show_bridge_legend,
                    hoverinfo="skip",
                    meta=dict(layer="bathy_fill"),
                )
            )
            show_bridge_legend = False

        finite_start = np.where(np.isfinite(tx) & np.isfinite(ty) & np.isfinite(z_full))[0]
        if finite_start.size > 0:
            k0 = int(finite_start[0])
            fig.add_trace(
                go.Scatter3d(
                    x=[tx[k0]],
                    y=[ty[k0]],
                    z=[z_full[k0]],
                    mode="markers+text",
                    marker=dict(size=5, color=col),
                    text=[f"T{i+1}"],
                    textposition="top center",
                    name=f"{d.path.stem} start",
                    legendgroup=grp,
                    showlegend=False,
                    hovertemplate=(
                        f"{d.path.name}<br>"
                        f"Start UTC={pd.Timestamp(d.start_time_utc).tz_convert('UTC').isoformat()}<br>"
                        f"Bron={_source_label(d_plot.depth_source)}<br>"
                        f"Profiel omgedraaid={'ja' if was_reversed else 'nee'}<br>"
                        f"Delta t={_fmt_delta(delta)}<br>"
                        f"Offset (L->R +)={offset_m:.2f} m<extra></extra>"
                    ),
                    meta=dict(layer="start"),
                )
            )

    max_depth = max(float(np.nanmax(d)) for d in all_depths)
    all_tx = [np.asarray(_parallel_xy_for_dataset(d, origin_e, origin_n, normal_vec, (
        (pd.Timestamp(d.start_time_utc).tz_convert("UTC") - pd.Timestamp(ref_start).tz_convert("UTC")).total_seconds()
        / 86400.0 * float(spacing_m_per_day)
    ))[0], dtype=float) for d in normed]
    all_ty = [np.asarray(_parallel_xy_for_dataset(d, origin_e, origin_n, normal_vec, (
        (pd.Timestamp(d.start_time_utc).tz_convert("UTC") - pd.Timestamp(ref_start).tz_convert("UTC")).total_seconds()
        / 86400.0 * float(spacing_m_per_day)
    ))[1], dtype=float) for d in normed]
    x_vals = np.concatenate([v[np.isfinite(v)] for v in all_tx if np.count_nonzero(np.isfinite(v)) > 0])
    y_vals = np.concatenate([v[np.isfinite(v)] for v in all_ty if np.count_nonzero(np.isfinite(v)) > 0])
    x_span = float(np.nanmax(x_vals) - np.nanmin(x_vals)) if x_vals.size else 1.0
    y_span = float(np.nanmax(y_vals) - np.nanmin(y_vals)) if y_vals.size else 1.0
    z_span = max_depth
    base_span = max(x_span, y_span, z_span, 1e-6)
    aspectratio = dict(
        x=max(x_span / base_span, 0.05),
        y=max(y_span / base_span, 0.05),
        z=max((z_span / base_span) * z_exag, 0.05),
    )
    fig.update_layout(
        title=(
            "3D bathymetrie in parallelle tijdstracks "
            f"(referentie: {ref.path.name}, bron: {_source_label(ref.depth_source)}, "
            f"schaal: {float(spacing_m_per_day):.2f} m/dag, visuele z x{z_exag:.1f})"
        ),
        template="plotly_white",
        scene=dict(
            xaxis_title="Easting (parallel verschoven, m)",
            yaxis_title="Northing (parallel verschoven, m)",
            zaxis_title="Bodemdiepte (m, negatief)",
            aspectmode="manual",
            aspectratio=aspectratio,
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.75)),
            zaxis=dict(range=[-max_depth * 1.05, 0.0]),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=10, r=10, b=10, t=55),
    )
    return fig


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M9 MAT multi 3D parallelle bathymetrie")
        self.geometry("1140x820")

        self.var_out = tk.StringVar(value=str(Path.cwd() / "m9_parallel_bathymetrie_3d.html"))
        self.var_m9_shift = tk.StringVar(value="-1")
        self.var_track_is_ne = tk.BooleanVar(value=False)
        self.var_ens_step = tk.IntVar(value=2)
        self.var_spacing_m_per_day = tk.DoubleVar(value=25.0)
        self.var_depth_exaggeration = tk.DoubleVar(value=12.0)
        self.var_open_after = tk.BooleanVar(value=True)
        self.var_depth_source = tk.StringVar(value="bt")

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
        self.listbox = tk.Listbox(list_wrap, height=10, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=sb.set)

        opt = ttk.LabelFrame(frm, text="Instellingen", padding=10)
        opt.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        opt.grid_columnconfigure(1, weight=1)

        ttk.Label(opt, text="Output HTML").grid(row=0, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.var_out).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(opt, text="Kies...", command=self._choose_output).grid(row=0, column=2, sticky="w")

        ttk.Label(opt, text="Dieptebron").grid(row=1, column=0, sticky="w", pady=(8, 0))
        src_row = ttk.Frame(opt)
        src_row.grid(row=1, column=1, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Radiobutton(src_row, text="BottomTrack diepte", value="bt", variable=self.var_depth_source).pack(side="left")
        ttk.Radiobutton(src_row, text="Vertical beam diepte", value="vb", variable=self.var_depth_source).pack(
            side="left", padx=(14, 0)
        )

        ttk.Label(opt, text="M9 uurshift").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_m9_shift, width=10).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Checkbutton(opt, text="Summary.Track is N,E (swap)", variable=self.var_track_is_ne).grid(
            row=3, column=1, sticky="w", pady=(8, 0)
        )

        ttk.Label(opt, text="Ensemble stap").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_ens_step, width=10).grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(opt, text="Schaal m per dag").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_spacing_m_per_day, width=10).grid(
            row=5, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(opt, text="Diepte overdrijving").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_depth_exaggeration, width=10).grid(
            row=6, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        ttk.Checkbutton(opt, text="Open HTML na maken", variable=self.var_open_after).grid(
            row=7, column=1, sticky="w", pady=(8, 0)
        )

        log_box = ttk.LabelFrame(frm, text="Log", padding=10)
        log_box.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        log_box.grid_columnconfigure(0, weight=1)
        log_box.grid_rowconfigure(0, weight=1)

        self.txt = tk.Text(log_box, height=16, wrap="word")
        self.txt.grid(row=0, column=0, sticky="nsew")
        sb2 = ttk.Scrollbar(log_box, orient="vertical", command=self.txt.yview)
        sb2.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=sb2.set)

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(btns, text="Maak 3D HTML", command=self._run).pack(side="left")
        ttk.Button(btns, text="Sluiten", command=self.destroy).pack(side="right")

    def _log(self, msg: str) -> None:
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Kies M9 MAT-bestanden",
            filetypes=[("MAT files", "*.mat"), ("Alle bestanden", "*.*")],
        )
        for s in paths:
            p = Path(s)
            if p not in self._mat_files:
                self._mat_files.append(p)
                self.listbox.insert("end", str(p))

    def _remove_selected(self) -> None:
        idx = sorted(self.listbox.curselection(), reverse=True)
        for i in idx:
            del self._mat_files[i]
            self.listbox.delete(i)

    def _clear_files(self) -> None:
        self._mat_files.clear()
        self.listbox.delete(0, "end")

    def _choose_output(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Kies output HTML",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("Alle bestanden", "*.*")],
            initialfile=Path(self.var_out.get()).name,
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
            spacing_m_per_day = float(self.var_spacing_m_per_day.get())
            depth_exaggeration = max(0.1, float(self.var_depth_exaggeration.get()))
            open_after = bool(self.var_open_after.get())
            depth_source = str(self.var_depth_source.get()).strip().lower()

            datasets: list[BathyTrackData] = []
            failed: list[str] = []
            self._log("Inlezen MAT-bestanden...")
            self._log(f"Dieptebron: {_source_label(depth_source)}")
            self._log(f"Diepte overdrijving: x{depth_exaggeration:.1f}")
            for p in self._mat_files:
                try:
                    d = read_mat_bathy(
                        p,
                        depth_source=depth_source,
                        m9_hour_shift=m9_shift,
                        track_is_ne=track_is_ne,
                        ensemble_step=ens_step,
                    )
                    datasets.append(d)
                    nfill = int(np.count_nonzero(d.bathy_filled_mask))
                    self._log(
                        f"  OK: {p.name} | start={pd.Timestamp(d.start_time_utc).isoformat()} | "
                        f"punten={d.bathy_depth.size} | aangevuld={nfill}"
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
            fig = build_parallel_bathy_figure(
                datasets=datasets,
                spacing_m_per_day=spacing_m_per_day,
                depth_exaggeration=depth_exaggeration,
            )

            self._log("Gemiddeld profiel berekenen...")
            mean_df, track_names = build_mean_bathy_profile(datasets)
            mean_fig = build_mean_bathy_figure(mean_df, track_names, _source_label(depth_source))
            self._log("Beste profiel-overlap (horizontale shift):")
            for name in track_names:
                shift_col = f"shift_{name}_m"
                shift_val = float(mean_df[shift_col].iloc[0]) if shift_col in mean_df.columns and len(mean_df) > 0 else 0.0
                self._log(f"  {name}: shift={shift_val:+.3f} m")

            outp = Path(out_html)
            outp.parent.mkdir(parents=True, exist_ok=True)
            mean_html = outp.with_name(outp.stem + "_gemiddeld_profiel.html")
            mean_csv = outp.with_name(outp.stem + "_gemiddeld_profiel.csv")
            fig.write_html(
                str(outp),
                include_plotlyjs="cdn",
                full_html=True,
                config={"responsive": True},
                post_script=_plotly_bathy_controls_post_script(),
            )
            mean_fig.write_html(
                str(mean_html),
                include_plotlyjs="cdn",
                full_html=True,
                config={"responsive": True},
            )
            mean_df.to_csv(mean_csv, index=False)
            self._log(f"Klaar 3D: {outp}")
            self._log(f"Klaar gemiddeld profiel HTML: {mean_html}")
            self._log(f"Klaar gemiddeld profiel CSV: {mean_csv}")

            if failed:
                self._log("")
                self._log("Bestanden met fouten:")
                for s in failed:
                    self._log(f"  - {s}")

            if open_after:
                webbrowser.open(outp.resolve().as_uri())
                webbrowser.open(mean_html.resolve().as_uri())
            messagebox.showinfo(
                "Klaar",
                f"3D HTML bathymetrie gemaakt met {len(datasets)} track(s).\n"
                f"Gemiddeld profiel HTML en CSV ook geëxporteerd.",
            )
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"FOUT: {e}")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
