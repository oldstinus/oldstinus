#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI script voor:
- 3D interpolatie van stroomsnelheid tussen nabije meetcellen (IDW in 3D)
- extrapolatie binnen vaargebied-domein, met optie voor strikte trackgrens (convex hull)
- optionele bodeminterpolatie (bathymetrie) + DTM export (.asc)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

import numpy as np
import scipy.io as sio
from scipy.spatial import Delaunay, QhullError, cKDTree

try:
    import plotly.graph_objects as go  # type: ignore

    _PLOTLY_OK = True
except Exception:
    go = None
    _PLOTLY_OK = False


@dataclass
class M9PointData:
    path: Path
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    speed: np.ndarray
    track_e: np.ndarray
    track_n: np.ndarray
    bed_depth: np.ndarray
    note: str = ""


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


def read_mat_points(
    mat_path: str | Path,
    track_is_ne: bool = False,
    ensemble_step: int = 1,
    cell_step: int = 1,
) -> M9PointData:
    p = Path(mat_path)
    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)

    for key in ("System", "Summary", "BottomTrack", "WaterTrack"):
        if key not in mat:
            raise ValueError(f"{p.name}: structuur '{key}' ontbreekt.")

    sys_obj = mat["System"]
    sum_obj = mat["Summary"]
    bt_obj = mat["BottomTrack"]
    wt_obj = mat["WaterTrack"]

    if not hasattr(sum_obj, "Track"):
        raise ValueError(f"{p.name}: Summary.Track ontbreekt.")
    if not hasattr(wt_obj, "Velocity"):
        raise ValueError(f"{p.name}: WaterTrack.Velocity ontbreekt.")

    track = np.asarray(sum_obj.Track, dtype=float)
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError(f"{p.name}: Summary.Track vorm onverwacht {track.shape}")
    track = track[:, :2]
    if track_is_ne:
        track = track[:, [1, 0]]

    ns_track = track.shape[0]
    if ns_track < 2:
        raise ValueError(f"{p.name}: te weinig ensembles ({ns_track}).")

    track_e = np.asarray(track[:, 0], dtype=float)
    track_n = np.asarray(track[:, 1], dtype=float)
    cstart = _as_1d(getattr(sys_obj, "Cell_Start", None), ns_track)
    csize = _as_1d(getattr(sys_obj, "Cell_Size", None), ns_track)
    bed = _as_1d(getattr(bt_obj, "BT_Depth", None), ns_track)

    vel = _reorder_velocity(np.asarray(getattr(wt_obj, "Velocity")), ns_target=ns_track)
    nc, ncomp, ns_vel = vel.shape
    if ncomp < 2:
        raise ValueError(f"{p.name}: Velocity heeft minder dan 2 componenten: {vel.shape}")
    ns = min(ns_track, ns_vel)

    track_e = track_e[:ns]
    track_n = track_n[:ns]
    cstart = cstart[:ns]
    csize = csize[:ns]
    bed = bed[:ns]
    vel = vel[:, :, :ns]

    u = np.asarray(vel[:, 0, :], dtype=float)
    v = np.asarray(vel[:, 1, :], dtype=float)
    speed = np.sqrt(u * u + v * v)

    row_idx = np.arange(nc, dtype=float).reshape(-1, 1)
    depth_abs = cstart.reshape(1, -1) + (row_idx + 0.5) * csize.reshape(1, -1)

    valid = (
        np.isfinite(depth_abs)
        & np.isfinite(speed)
        & np.isfinite(track_e.reshape(1, -1))
        & np.isfinite(track_n.reshape(1, -1))
    )
    valid &= depth_abs > 0.0
    bed_ok = np.isfinite(bed).reshape(1, -1)
    valid &= (~bed_ok) | (depth_abs < bed.reshape(1, -1))

    ens_step = max(1, int(ensemble_step))
    csel = np.arange(0, ns, ens_step, dtype=int)
    rsel = np.arange(0, nc, max(1, int(cell_step)), dtype=int)

    depth_abs = depth_abs[np.ix_(rsel, csel)]
    speed = speed[np.ix_(rsel, csel)]
    valid = valid[np.ix_(rsel, csel)]
    tr_e = track_e[csel]
    tr_n = track_n[csel]
    bed_sel = bed[csel]

    x = np.tile(tr_e.reshape(1, -1), (len(rsel), 1))
    y = np.tile(tr_n.reshape(1, -1), (len(rsel), 1))
    z = -depth_abs

    x = x[valid]
    y = y[valid]
    z = z[valid]
    sp = speed[valid]

    if np.count_nonzero(np.isfinite(sp)) < 20:
        raise ValueError(f"{p.name}: onvoldoende geldige snelheidspunten.")

    note = ""
    med_abs = max(float(np.nanmedian(np.abs(tr_e))), float(np.nanmedian(np.abs(tr_n))))
    if med_abs < 5000:
        note = "Track lijkt relatief (mogelijk niet gegeorefereerd)."

    return M9PointData(
        path=p,
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        z=np.asarray(z, dtype=float),
        speed=np.asarray(sp, dtype=float),
        track_e=np.asarray(tr_e, dtype=float),
        track_n=np.asarray(tr_n, dtype=float),
        bed_depth=np.asarray(bed_sel, dtype=float),
        note=note,
    )


def _idw_interpolate(
    src_xyz: np.ndarray,
    src_val: np.ndarray,
    dst_xyz: np.ndarray,
    k: int = 12,
    power: float = 2.0,
    radius: float = np.inf,
    extrapolate: bool = False,
) -> np.ndarray:
    src_xyz = np.asarray(src_xyz, dtype=float)
    src_val = np.asarray(src_val, dtype=float).reshape(-1)
    dst_xyz = np.asarray(dst_xyz, dtype=float)

    finite_src = np.all(np.isfinite(src_xyz), axis=1) & np.isfinite(src_val)
    src_xyz = src_xyz[finite_src]
    src_val = src_val[finite_src]
    if src_xyz.size == 0:
        return np.full(dst_xyz.shape[0], np.nan, dtype=float)

    kk = int(max(1, min(int(k), src_xyz.shape[0])))
    rr = float(radius) if np.isfinite(radius) and radius > 0 else np.inf

    tree = cKDTree(src_xyz)
    dist, idx = tree.query(dst_xyz, k=kk, distance_upper_bound=rr)

    if kk == 1:
        dist = dist.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    out = np.full(dst_xyz.shape[0], np.nan, dtype=float)
    valid = np.isfinite(dist) & (idx < src_val.size)
    if not np.any(valid):
        if extrapolate:
            d1, i1 = tree.query(dst_xyz, k=1)
            ok1 = np.isfinite(d1) & (i1 < src_val.size)
            out[ok1] = src_val[i1[ok1]]
        return out

    exact = valid & (dist <= 1e-12)
    has_exact = np.any(exact, axis=1)
    if np.any(has_exact):
        first_exact = np.argmax(exact, axis=1)
        ridx = np.where(has_exact)[0]
        cidx = first_exact[has_exact]
        out[ridx] = src_val[idx[ridx, cidx]]

    rem = ~has_exact
    if np.any(rem):
        d = dist[rem]
        ii = idx[rem]
        vv_mask = valid[rem]
        safe_idx = np.where(vv_mask, ii, 0)
        vv = src_val[safe_idx]
        w = np.where(vv_mask, 1.0 / np.power(np.maximum(d, 1e-12), float(power)), 0.0)
        w_sum = np.sum(w, axis=1)
        num = np.sum(w * vv, axis=1)
        good = w_sum > 0
        rem_idx = np.where(rem)[0]
        out[rem_idx[good]] = num[good] / w_sum[good]

    if extrapolate:
        miss = ~np.isfinite(out)
        if np.any(miss):
            d1, i1 = tree.query(dst_xyz[miss], k=1)
            ok1 = np.isfinite(d1) & (i1 < src_val.size)
            miss_idx = np.where(miss)[0]
            out[miss_idx[ok1]] = src_val[i1[ok1]]
    return out


def _build_domain_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    track_x: np.ndarray,
    track_y: np.ndarray,
    buffer_m: float,
) -> np.ndarray:
    tx = np.asarray(track_x, dtype=float).reshape(-1)
    ty = np.asarray(track_y, dtype=float).reshape(-1)
    ok = np.isfinite(tx) & np.isfinite(ty)
    tx = tx[ok]
    ty = ty[ok]
    if tx.size == 0:
        return np.zeros_like(grid_x, dtype=bool)

    pts = np.column_stack([tx, ty])
    tree = cKDTree(pts)
    q = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    dist, _ = tree.query(q, k=1)
    mask = np.isfinite(dist) & (dist <= float(max(0.5, buffer_m)))
    return mask.reshape(grid_x.shape)


def _build_track_hull_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    track_x: np.ndarray,
    track_y: np.ndarray,
) -> np.ndarray | None:
    tx = np.asarray(track_x, dtype=float).reshape(-1)
    ty = np.asarray(track_y, dtype=float).reshape(-1)
    ok = np.isfinite(tx) & np.isfinite(ty)
    pts = np.column_stack([tx[ok], ty[ok]])
    if pts.shape[0] < 3:
        return None

    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
        return None

    try:
        tri = Delaunay(pts)
    except QhullError:
        return None

    q = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    in_hull = tri.find_simplex(q) >= 0
    return in_hull.reshape(grid_x.shape)


def _interpolate_bathymetry_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    domain_mask: np.ndarray,
    track_x: np.ndarray,
    track_y: np.ndarray,
    bed_depth: np.ndarray,
    k: int,
    power: float,
    radius: float,
    extrapolate: bool,
) -> np.ndarray:
    tx = np.asarray(track_x, dtype=float).reshape(-1)
    ty = np.asarray(track_y, dtype=float).reshape(-1)
    bd = np.asarray(bed_depth, dtype=float).reshape(-1)
    valid = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(bd) & (bd > 0.0)
    tx = tx[valid]
    ty = ty[valid]
    bd = bd[valid]
    if bd.size < 3:
        return np.full_like(grid_x, np.nan, dtype=float)

    dst_mask = np.asarray(domain_mask, dtype=bool)
    out = np.full_like(grid_x, np.nan, dtype=float)
    if np.count_nonzero(dst_mask) == 0:
        return out

    src_xyz = np.column_stack([tx, ty, np.zeros_like(tx)])
    dst_xyz = np.column_stack([grid_x[dst_mask], grid_y[dst_mask], np.zeros(np.count_nonzero(dst_mask))])
    vals = _idw_interpolate(src_xyz, bd, dst_xyz, k=k, power=power, radius=radius, extrapolate=extrapolate)
    out[dst_mask] = vals
    return out


def _interpolate_speed_volume(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    z_levels: np.ndarray,
    domain_mask: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    src_z: np.ndarray,
    src_speed: np.ndarray,
    k: int,
    power: float,
    radius: float,
    extrapolate: bool,
    bathy_depth: np.ndarray | None = None,
) -> np.ndarray:
    src_xyz = np.column_stack([src_x.reshape(-1), src_y.reshape(-1), src_z.reshape(-1)])
    src_speed = src_speed.reshape(-1)
    ny, nx = grid_x.shape
    nz = z_levels.size
    vol = np.full((nz, ny, nx), np.nan, dtype=float)

    base_mask = np.asarray(domain_mask, dtype=bool)
    if np.count_nonzero(base_mask) == 0:
        return vol

    for iz, zlev in enumerate(z_levels):
        m = base_mask.copy()
        if bathy_depth is not None:
            m &= np.isfinite(bathy_depth) & ((-float(zlev)) <= bathy_depth)
        if np.count_nonzero(m) == 0:
            continue
        dst = np.column_stack([grid_x[m], grid_y[m], np.full(np.count_nonzero(m), float(zlev))])
        vals = _idw_interpolate(src_xyz, src_speed, dst, k=k, power=power, radius=radius, extrapolate=extrapolate)
        vol[iz, m] = vals
    return vol


def _pick_slice_indices(nz: int, max_slices: int) -> np.ndarray:
    if nz <= 0:
        return np.zeros(0, dtype=int)
    ns = max(1, min(int(max_slices), nz))
    idx = np.linspace(0, nz - 1, ns)
    return np.unique(np.round(idx).astype(int))


def _plotly_layer_controls_post_script(default_flow_scale: str = "Turbo", default_bathy_scale: str = "Earth") -> str:
    script = r"""
(function() {
  const gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) return;

  const flowIdx = [];
  const bathyIdx = [];
  for (let i = 0; i < gd.data.length; i++) {
    const t = gd.data[i];
    if (!t) continue;
    const layer = (t.meta && t.meta.layer) ? String(t.meta.layer).toLowerCase() : "";
    if (layer === "flow") flowIdx.push(i);
    if (layer === "bathy") bathyIdx.push(i);
  }
  if (!flowIdx.length && !bathyIdx.length) return;

  const scales = ["Turbo", "Viridis", "Plasma", "Cividis", "Jet", "Earth", "YlGnBu", "Portland", "RdBu", "Bluered"];
  const meta = (gd.layout && gd.layout.meta) ? gd.layout.meta : {};
  let flowMin0 = Number(meta.flow_min);
  let flowMax0 = Number(meta.flow_max);
  let bathyMin0 = Number(meta.bathy_min);
  let bathyMax0 = Number(meta.bathy_max);
  let flowScale0 = String(meta.flow_scale || __DEFAULT_FLOW__);
  let bathyScale0 = String(meta.bathy_scale || __DEFAULT_BATHY__);

  if ((!Number.isFinite(flowMin0) || !Number.isFinite(flowMax0)) && flowIdx.length) {
    const t = gd.data[flowIdx[0]];
    flowMin0 = Number(t.cmin);
    flowMax0 = Number(t.cmax);
    if (t.colorscale && typeof t.colorscale === 'string') flowScale0 = String(t.colorscale);
  }
  if ((!Number.isFinite(bathyMin0) || !Number.isFinite(bathyMax0)) && bathyIdx.length) {
    const t = gd.data[bathyIdx[0]];
    bathyMin0 = Number(t.cmin);
    bathyMax0 = Number(t.cmax);
    if (t.colorscale && typeof t.colorscale === 'string') bathyScale0 = String(t.colorscale);
  }

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

  const rowStyle = "display:grid;grid-template-columns:130px 150px;gap:8px;align-items:center;margin:4px 0;";
  panel.innerHTML = ''
    + '<div style="font-weight:600;margin-bottom:6px;">3D lagen en kleuren</div>'
    + '<div style="' + rowStyle + '"><label><input id="m9-flow-on" type="checkbox"> Toon stroming</label><span></span></div>'
    + '<div style="' + rowStyle + '"><label for="m9-flow-scale">Flow kleurschaal</label><select id="m9-flow-scale"></select></div>'
    + '<div style="' + rowStyle + '"><label for="m9-flow-min">Flow min (m/s)</label><input id="m9-flow-min" type="number" step="0.01"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-flow-max">Flow max (m/s)</label><input id="m9-flow-max" type="number" step="0.01"></div>'
    + '<div style="height:1px;background:#d9d9d9;margin:6px 0;"></div>'
    + '<div style="' + rowStyle + '"><label><input id="m9-bathy-on" type="checkbox"> Toon bathymetrie</label><span></span></div>'
    + '<div style="' + rowStyle + '"><label for="m9-bathy-scale">Bathy kleurschaal</label><select id="m9-bathy-scale"></select></div>'
    + '<div style="' + rowStyle + '"><label for="m9-bathy-min">Bathy min (m)</label><input id="m9-bathy-min" type="number" step="0.1"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-bathy-max">Bathy max (m)</label><input id="m9-bathy-max" type="number" step="0.1"></div>'
    + '<div style="margin-top:6px;"><button id="m9-apply" type="button">Toepassen</button> <button id="m9-reset" type="button">Reset</button></div>';

  root.appendChild(panel);

  const flowOn = panel.querySelector('#m9-flow-on');
  const flowScale = panel.querySelector('#m9-flow-scale');
  const flowMin = panel.querySelector('#m9-flow-min');
  const flowMax = panel.querySelector('#m9-flow-max');
  const bathyOn = panel.querySelector('#m9-bathy-on');
  const bathyScale = panel.querySelector('#m9-bathy-scale');
  const bathyMin = panel.querySelector('#m9-bathy-min');
  const bathyMax = panel.querySelector('#m9-bathy-max');
  const applyBtn = panel.querySelector('#m9-apply');
  const resetBtn = panel.querySelector('#m9-reset');
  if (!flowOn || !flowScale || !flowMin || !flowMax || !bathyOn || !bathyScale || !bathyMin || !bathyMax || !applyBtn || !resetBtn) return;

  scales.forEach((s) => {
    const o1 = document.createElement('option');
    o1.value = s;
    o1.textContent = s;
    flowScale.appendChild(o1);
    const o2 = document.createElement('option');
    o2.value = s;
    o2.textContent = s;
    bathyScale.appendChild(o2);
  });

  function isVisible(trace) {
    return !(trace && (trace.visible === false || String(trace.visible) === 'legendonly'));
  }
  const flowVisible0 = flowIdx.length ? isVisible(gd.data[flowIdx[0]]) : false;
  const bathyVisible0 = bathyIdx.length ? isVisible(gd.data[bathyIdx[0]]) : false;

  function setDefaults() {
    flowOn.checked = flowVisible0;
    bathyOn.checked = bathyVisible0;
    flowScale.value = scales.includes(flowScale0) ? flowScale0 : scales[0];
    bathyScale.value = scales.includes(bathyScale0) ? bathyScale0 : scales[0];
    flowMin.value = Number.isFinite(flowMin0) ? flowMin0.toFixed(3) : "";
    flowMax.value = Number.isFinite(flowMax0) ? flowMax0.toFixed(3) : "";
    bathyMin.value = Number.isFinite(bathyMin0) ? bathyMin0.toFixed(3) : "";
    bathyMax.value = Number.isFinite(bathyMax0) ? bathyMax0.toFixed(3) : "";

    const hasFlow = flowIdx.length > 0;
    const hasBathy = bathyIdx.length > 0;
    flowOn.disabled = !hasFlow;
    flowScale.disabled = !hasFlow;
    flowMin.disabled = !hasFlow;
    flowMax.disabled = !hasFlow;
    bathyOn.disabled = !hasBathy;
    bathyScale.disabled = !hasBathy;
    bathyMin.disabled = !hasBathy;
    bathyMax.disabled = !hasBathy;
  }

  function apply() {
    if (flowIdx.length) {
      let fmin = Number(flowMin.value);
      let fmax = Number(flowMax.value);
      if (!Number.isFinite(fmin)) fmin = flowMin0;
      if (!Number.isFinite(fmax)) fmax = flowMax0;
      if (Number.isFinite(fmin) && Number.isFinite(fmax) && fmax <= fmin) {
        fmax = fmin + 0.001;
        flowMax.value = fmax.toFixed(3);
      }
      Plotly.restyle(gd, { visible: !!flowOn.checked }, flowIdx);
      if (Number.isFinite(fmin) && Number.isFinite(fmax)) {
        Plotly.restyle(gd, { colorscale: flowScale.value, cmin: fmin, cmax: fmax }, flowIdx);
      } else {
        Plotly.restyle(gd, { colorscale: flowScale.value }, flowIdx);
      }
    }
    if (bathyIdx.length) {
      let bmin = Number(bathyMin.value);
      let bmax = Number(bathyMax.value);
      if (!Number.isFinite(bmin)) bmin = bathyMin0;
      if (!Number.isFinite(bmax)) bmax = bathyMax0;
      if (Number.isFinite(bmin) && Number.isFinite(bmax) && bmax <= bmin) {
        bmax = bmin + 0.001;
        bathyMax.value = bmax.toFixed(3);
      }
      Plotly.restyle(gd, { visible: !!bathyOn.checked }, bathyIdx);
      if (Number.isFinite(bmin) && Number.isFinite(bmax)) {
        Plotly.restyle(gd, { colorscale: bathyScale.value, cmin: bmin, cmax: bmax }, bathyIdx);
      } else {
        Plotly.restyle(gd, { colorscale: bathyScale.value }, bathyIdx);
      }
    }
  }

  setDefaults();
  applyBtn.addEventListener('click', apply);
  resetBtn.addEventListener('click', () => { setDefaults(); apply(); });
})();
"""
    return (
        script.replace("__DEFAULT_FLOW__", json.dumps(default_flow_scale))
        .replace("__DEFAULT_BATHY__", json.dumps(default_bathy_scale))
    )


def build_3d_slice_figure(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    z_levels: np.ndarray,
    speed_vol: np.ndarray,
    track_x: np.ndarray,
    track_y: np.ndarray,
    bathy_depth: np.ndarray | None,
    color_scale: str,
    opacity: float,
    max_slices: int,
    depth_exaggeration: float = 1.0,
    speed_cmin: float | None = None,
    speed_cmax: float | None = None,
    bathy_color_scale: str = "Earth",
    bathy_cmin: float | None = None,
    bathy_cmax: float | None = None,
) -> go.Figure:
    vals = speed_vol[np.isfinite(speed_vol)]
    if vals.size == 0:
        raise ValueError("Geen geinterpoleerde snelheidspunten om te tekenen.")
    smin = float(np.nanpercentile(vals, 2))
    smax = float(np.nanpercentile(vals, 98))
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        smin = float(np.nanmin(vals))
        smax = float(np.nanmax(vals))
    if speed_cmin is not None and np.isfinite(speed_cmin):
        smin = float(speed_cmin)
    if speed_cmax is not None and np.isfinite(speed_cmax):
        smax = float(speed_cmax)
    if smax <= smin:
        smax = smin + 1e-6

    fig = go.Figure()
    shown_scale = False
    z_exag = max(0.1, float(depth_exaggeration))
    bmin_meta = np.nan
    bmax_meta = np.nan
    for iz in _pick_slice_indices(z_levels.size, max_slices):
        sc = speed_vol[iz]
        valid_sc = np.isfinite(sc)
        if np.count_nonzero(valid_sc) < 8:
            continue
        z_real = float(z_levels[iz])
        z_plot = z_real * z_exag
        z_slice = np.full_like(grid_x, z_plot, dtype=float)
        z_slice[~valid_sc] = np.nan
        sc_plot = np.where(valid_sc, sc, np.nan)
        fig.add_trace(
            go.Surface(
                x=grid_x,
                y=grid_y,
                z=z_slice,
                surfacecolor=sc_plot,
                cmin=smin,
                cmax=smax,
                colorscale=color_scale,
                opacity=float(opacity),
                connectgaps=False,
                showscale=not shown_scale,
                colorbar=dict(title="Snelheid (m/s)", x=1.02, len=0.72, thickness=16) if not shown_scale else None,
                name=f"z={z_real:.2f} m",
                meta=dict(layer="flow"),
                customdata=np.full_like(grid_x, z_real, dtype=float),
                hovertemplate=(
                    "E=%{x:.2f}<br>N=%{y:.2f}<br>z=%{customdata:.2f} m<br>"
                    "Snelheid=%{surfacecolor:.3f} m/s<extra></extra>"
                ),
            )
        )
        shown_scale = True

    tx = np.asarray(track_x, dtype=float).reshape(-1)
    ty = np.asarray(track_y, dtype=float).reshape(-1)
    t_ok = np.isfinite(tx) & np.isfinite(ty)
    if np.count_nonzero(t_ok) >= 2:
        fig.add_trace(
            go.Scatter3d(
                x=tx[t_ok],
                y=ty[t_ok],
                z=np.zeros(np.count_nonzero(t_ok), dtype=float),
                mode="lines",
                line=dict(color="#111111", width=5),
                name="Track",
                meta=dict(layer="track"),
                hoverinfo="skip",
            )
        )

    if bathy_depth is not None and np.count_nonzero(np.isfinite(bathy_depth)) > 8:
        valid_bathy = np.isfinite(bathy_depth)
        z_bathy = np.where(valid_bathy, -bathy_depth * z_exag, np.nan)
        bathy_plot = np.where(valid_bathy, bathy_depth, np.nan)
        bvals = bathy_plot[np.isfinite(bathy_plot)]
        if bvals.size > 0:
            bmin = float(np.nanpercentile(bvals, 2))
            bmax = float(np.nanpercentile(bvals, 98))
            if not np.isfinite(bmin) or not np.isfinite(bmax) or bmax <= bmin:
                bmin = float(np.nanmin(bvals))
                bmax = float(np.nanmax(bvals))
        else:
            bmin, bmax = 0.0, 1.0
        if bathy_cmin is not None and np.isfinite(bathy_cmin):
            bmin = float(bathy_cmin)
        if bathy_cmax is not None and np.isfinite(bathy_cmax):
            bmax = float(bathy_cmax)
        if bmax <= bmin:
            bmax = bmin + 1e-6
        bmin_meta = bmin
        bmax_meta = bmax
        fig.add_trace(
            go.Surface(
                x=grid_x,
                y=grid_y,
                z=z_bathy,
                surfacecolor=bathy_plot,
                cmin=bmin,
                cmax=bmax,
                colorscale=bathy_color_scale,
                showscale=True,
                colorbar=dict(title="Bodemdiepte (m)", x=1.13, len=0.72, thickness=16),
                opacity=0.45,
                connectgaps=False,
                name="Bodem (bathymetrie)",
                meta=dict(layer="bathy"),
                hovertemplate="E=%{x:.2f}<br>N=%{y:.2f}<br>Bodemdiepte=%{customdata:.2f} m<extra></extra>",
                customdata=bathy_depth,
            )
        )

    fig.update_layout(
        title="3D geinterpoleerde stroomsnelheid",
        template="plotly_white",
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title=f"Diepte (m, negatief, x{z_exag:.2f})",
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.80)),
        ),
        margin=dict(l=10, r=180, b=10, t=45),
        legend=dict(itemsizing="constant"),
        meta=dict(
            flow_scale=str(color_scale),
            bathy_scale=str(bathy_color_scale),
            flow_min=float(smin),
            flow_max=float(smax),
            bathy_min=float(bmin_meta),
            bathy_max=float(bmax_meta),
        ),
    )
    return fig


def write_esri_ascii_dtm(
    out_path: str | Path,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    depth_grid: np.ndarray,
    nodata: float = -9999.0,
) -> Path:
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(x_centers, dtype=float).reshape(-1)
    y = np.asarray(y_centers, dtype=float).reshape(-1)
    z = np.asarray(depth_grid, dtype=float)
    if z.shape != (y.size, x.size):
        raise ValueError(f"DTM shape {z.shape} past niet bij x/y ({y.size}, {x.size}).")
    if x.size < 2 or y.size < 2:
        raise ValueError("Te weinig gridpunten voor DTM-export.")

    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        raise ValueError("Ongeldige gridresolutie voor DTM.")
    if abs(dx - dy) > 1e-6:
        raise ValueError("DTM-export vereist vierkante cellen (dx == dy).")

    xll = float(x[0] - 0.5 * dx)
    yll = float(y[0] - 0.5 * dy)
    z_out = np.flipud(z)
    z_out = np.where(np.isfinite(z_out), z_out, float(nodata))

    with outp.open("w", encoding="utf-8") as f:
        f.write(f"ncols {x.size}\n")
        f.write(f"nrows {y.size}\n")
        f.write(f"xllcorner {xll:.6f}\n")
        f.write(f"yllcorner {yll:.6f}\n")
        f.write(f"cellsize {dx:.6f}\n")
        f.write(f"NODATA_value {float(nodata):.6f}\n")
        for row in z_out:
            f.write(" ".join(f"{float(v):.6f}" for v in row) + "\n")
    return outp


def _median_track_spacing(datasets: list[M9PointData]) -> float:
    d_all: list[np.ndarray] = []
    for d in datasets:
        te = np.asarray(d.track_e, dtype=float).reshape(-1)
        tn = np.asarray(d.track_n, dtype=float).reshape(-1)
        ok = np.isfinite(te) & np.isfinite(tn)
        te = te[ok]
        tn = tn[ok]
        if te.size < 2:
            continue
        dd = np.hypot(np.diff(te), np.diff(tn))
        dd = dd[np.isfinite(dd) & (dd > 0.0)]
        if dd.size:
            d_all.append(dd)
    if not d_all:
        return 2.0
    return float(np.nanmedian(np.concatenate(d_all)))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M9 3D interpolatie + bathymetrie DTM")
        self.geometry("1240x860")

        cwd = Path.cwd()
        self.var_out_html = tk.StringVar(value=str(cwd / "m9_3d_interpolatie.html"))
        self.var_out_dtm = tk.StringVar(value=str(cwd / "m9_bathymetry_dtm.asc"))

        self.var_track_is_ne = tk.BooleanVar(value=False)
        self.var_ens_step = tk.IntVar(value=1)
        self.var_cell_step = tk.IntVar(value=1)
        self.var_grid_step = tk.DoubleVar(value=2.0)
        self.var_z_step = tk.DoubleVar(value=0.25)
        self.var_domain_buffer = tk.DoubleVar(value=20.0)
        self.var_limit_to_track_bounds = tk.BooleanVar(value=True)
        self.var_idw_k = tk.IntVar(value=12)
        self.var_idw_power = tk.DoubleVar(value=2.0)
        self.var_idw_radius = tk.DoubleVar(value=25.0)
        self.var_extrapolate = tk.BooleanVar(value=True)
        self.var_interp_bathy = tk.BooleanVar(value=True)
        self.var_export_dtm = tk.BooleanVar(value=True)
        self.var_slice_opacity = tk.DoubleVar(value=0.88)
        self.var_depth_exaggeration = tk.DoubleVar(value=1.0)
        self.var_max_slices = tk.IntVar(value=12)
        self.var_open_after = tk.BooleanVar(value=True)

        self._mat_files: list[Path] = []
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(4, weight=1)

        box_files = ttk.LabelFrame(root, text="MAT-bestanden", padding=10)
        box_files.grid(row=0, column=0, sticky="nsew")

        row_btn = ttk.Frame(box_files)
        row_btn.pack(fill="x")
        ttk.Button(row_btn, text="Toevoegen...", command=self._add_files).pack(side="left")
        ttk.Button(row_btn, text="Verwijder selectie", command=self._remove_selected).pack(side="left", padx=(8, 0))
        ttk.Button(row_btn, text="Leegmaken", command=self._clear_files).pack(side="left", padx=(8, 0))

        lst_wrap = ttk.Frame(box_files)
        lst_wrap.pack(fill="both", expand=True, pady=(8, 0))
        self.listbox = tk.Listbox(lst_wrap, height=10, selectmode=tk.EXTENDED)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lst_wrap, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        box_opt = ttk.LabelFrame(root, text="Interpolatie-opties", padding=10)
        box_opt.grid(row=1, column=0, sticky="we", pady=(10, 0))

        ttk.Checkbutton(box_opt, text="Summary.Track is N,E (swap)", variable=self.var_track_is_ne).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Checkbutton(
            box_opt, text="Extrapoleer binnen vaargebied-domein", variable=self.var_extrapolate
        ).grid(row=0, column=2, columnspan=2, sticky="w", padx=(16, 0))
        ttk.Checkbutton(
            box_opt, text="Interpoleer bodem (bathymetrie)", variable=self.var_interp_bathy
        ).grid(row=0, column=4, columnspan=2, sticky="w", padx=(16, 0))
        ttk.Checkbutton(
            box_opt, text="Beperk 3D tot trackgrenzen", variable=self.var_limit_to_track_bounds
        ).grid(row=0, column=6, columnspan=2, sticky="w", padx=(16, 0))
        ttk.Checkbutton(box_opt, text="Schrijf DTM (.asc)", variable=self.var_export_dtm).grid(
            row=0, column=8, sticky="w", padx=(16, 0)
        )

        ttk.Label(box_opt, text="Ensemble step:").grid(row=1, column=0, sticky="e", pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_ens_step, width=8).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(box_opt, text="Cell step:").grid(row=1, column=2, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_cell_step, width=8).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(box_opt, text="Grid XY stap (m):").grid(row=1, column=4, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_grid_step, width=8).grid(row=1, column=5, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(box_opt, text="Grid Z stap (m):").grid(row=1, column=6, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_z_step, width=8).grid(row=1, column=7, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(box_opt, text="Domein buffer (m):").grid(row=2, column=0, sticky="e", pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_domain_buffer, width=8).grid(
            row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0)
        )
        ttk.Label(box_opt, text="IDW k buren:").grid(row=2, column=2, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_idw_k, width=8).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(box_opt, text="IDW power:").grid(row=2, column=4, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_idw_power, width=8).grid(row=2, column=5, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(box_opt, text="IDW radius (m):").grid(row=2, column=6, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_idw_radius, width=8).grid(row=2, column=7, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(box_opt, text="Slice opacity:").grid(row=3, column=0, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_slice_opacity, width=8).grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(box_opt, text="Max slices:").grid(row=3, column=2, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_max_slices, width=8).grid(row=3, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(box_opt, text="Diepte overdrijving (x):").grid(row=3, column=4, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Entry(box_opt, textvariable=self.var_depth_exaggeration, width=8).grid(
            row=3, column=5, sticky="w", padx=(6, 0), pady=(8, 0)
        )
        ttk.Checkbutton(box_opt, text="Open HTML na export", variable=self.var_open_after).grid(
            row=3, column=6, columnspan=2, sticky="w", padx=(12, 0), pady=(8, 0)
        )

        box_out = ttk.LabelFrame(root, text="Output", padding=10)
        box_out.grid(row=2, column=0, sticky="we", pady=(10, 0))
        box_out.grid_columnconfigure(0, weight=1)

        ttk.Label(box_out, text="3D HTML:").grid(row=0, column=0, sticky="w")
        ttk.Entry(box_out, textvariable=self.var_out_html).grid(row=1, column=0, sticky="we")
        ttk.Button(box_out, text="Opslaan als...", command=self._pick_out_html).grid(row=1, column=1, padx=(8, 0))

        ttk.Label(box_out, text="DTM ASC:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(box_out, textvariable=self.var_out_dtm).grid(row=3, column=0, sticky="we")
        ttk.Button(box_out, text="Opslaan als...", command=self._pick_out_dtm).grid(row=3, column=1, padx=(8, 0))

        box_bot = ttk.Frame(root)
        box_bot.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        box_bot.grid_columnconfigure(0, weight=1)
        box_bot.grid_rowconfigure(2, weight=1)

        row_run = ttk.Frame(box_bot)
        row_run.grid(row=0, column=0, sticky="we")
        ttk.Button(row_run, text="Run interpolatie + export", command=self._run).pack(side="left")
        ttk.Button(row_run, text="Sluiten", command=self.destroy).pack(side="right")

        ttk.Label(box_bot, text="Log").grid(row=1, column=0, sticky="w")
        self.txt = tk.Text(box_bot, height=13, wrap="word")
        self.txt.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

    def _log(self, msg: str) -> None:
        self.txt.insert("end", msg + "\n")
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

    def _pick_out_html(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Kies output 3D HTML",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("All", "*.*")],
        )
        if p:
            self.var_out_html.set(p)
            dtm_guess = str(Path(p).with_name(Path(p).stem + "_dtm.asc"))
            if not self.var_out_dtm.get().strip():
                self.var_out_dtm.set(dtm_guess)

    def _pick_out_dtm(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Kies output DTM (.asc)",
            defaultextension=".asc",
            filetypes=[("ASC", "*.asc"), ("All", "*.*")],
        )
        if p:
            self.var_out_dtm.set(p)

    def _run(self) -> None:
        try:
            if not _PLOTLY_OK:
                messagebox.showerror("Dependency", "plotly niet beschikbaar. Installeer: pip install plotly")
                return
            if not self._mat_files:
                messagebox.showerror("Input", "Voeg eerst een of meer MAT-bestanden toe.")
                return

            out_html = self.var_out_html.get().strip()
            if not out_html:
                messagebox.showerror("Output", "Geef een output HTML-bestand op.")
                return

            out_dtm = self.var_out_dtm.get().strip()
            track_is_ne = bool(self.var_track_is_ne.get())
            ens_step = max(1, int(self.var_ens_step.get()))
            cell_step = max(1, int(self.var_cell_step.get()))
            grid_step = float(self.var_grid_step.get())
            z_step = max(0.05, float(self.var_z_step.get()))
            domain_buffer = max(0.5, float(self.var_domain_buffer.get()))
            limit_to_track_bounds = bool(self.var_limit_to_track_bounds.get())
            idw_k = max(1, int(self.var_idw_k.get()))
            idw_power = max(0.1, float(self.var_idw_power.get()))
            idw_radius = float(self.var_idw_radius.get())
            idw_radius = np.inf if idw_radius <= 0 else idw_radius
            extrapolate = bool(self.var_extrapolate.get())
            interp_bathy = bool(self.var_interp_bathy.get())
            export_dtm = bool(self.var_export_dtm.get())
            color_scale = "Turbo"
            bathy_color_scale = "Earth"
            speed_cmin = None
            speed_cmax = None
            bathy_cmin = None
            bathy_cmax = None
            slice_opacity = min(1.0, max(0.05, float(self.var_slice_opacity.get())))
            depth_exaggeration = max(0.1, float(self.var_depth_exaggeration.get()))
            max_slices = max(1, int(self.var_max_slices.get()))
            open_after = bool(self.var_open_after.get())

            datasets: list[M9PointData] = []
            failed: list[str] = []
            self._log("Inlezen MAT-bestanden...")
            for p in self._mat_files:
                try:
                    d = read_mat_points(
                        p,
                        track_is_ne=track_is_ne,
                        ensemble_step=ens_step,
                        cell_step=cell_step,
                    )
                    datasets.append(d)
                    self._log(
                        f"  OK: {p.name} | punten={d.speed.size} | track ensembles={d.track_e.size}"
                    )
                    if d.note:
                        self._log(f"      note: {d.note}")
                except Exception as e:
                    failed.append(f"{p.name}: {e}")
                    self._log(f"  ERROR: {p.name} -> {e}")

            if not datasets:
                raise ValueError("Geen geldig MAT-bestand kunnen verwerken.")

            src_x = np.concatenate([d.x for d in datasets])
            src_y = np.concatenate([d.y for d in datasets])
            src_z = np.concatenate([d.z for d in datasets])
            src_speed = np.concatenate([d.speed for d in datasets])
            tr_x = np.concatenate([d.track_e for d in datasets])
            tr_y = np.concatenate([d.track_n for d in datasets])
            tr_bed = np.concatenate([d.bed_depth for d in datasets])

            finite_src = np.isfinite(src_x) & np.isfinite(src_y) & np.isfinite(src_z) & np.isfinite(src_speed)
            src_x = src_x[finite_src]
            src_y = src_y[finite_src]
            src_z = src_z[finite_src]
            src_speed = src_speed[finite_src]
            if src_speed.size < 20:
                raise ValueError("Te weinig geldige snelheidspunten voor interpolatie.")

            finite_track = np.isfinite(tr_x) & np.isfinite(tr_y)
            tr_x = tr_x[finite_track]
            tr_y = tr_y[finite_track]
            tr_bed = tr_bed[finite_track]
            if tr_x.size < 2:
                raise ValueError("Te weinig geldige trackpunten voor domein.")

            if not np.isfinite(grid_step) or grid_step <= 0:
                grid_step = _median_track_spacing(datasets)
                grid_step = max(0.5, float(grid_step))
                self._log(f"Grid XY stap automatisch gezet op {grid_step:.3f} m")

            x_min = float(np.nanmin(tr_x) - domain_buffer)
            x_max = float(np.nanmax(tr_x) + domain_buffer)
            y_min = float(np.nanmin(tr_y) - domain_buffer)
            y_max = float(np.nanmax(tr_y) + domain_buffer)
            x_vec = np.arange(x_min, x_max + 0.5 * grid_step, grid_step, dtype=float)
            y_vec = np.arange(y_min, y_max + 0.5 * grid_step, grid_step, dtype=float)
            if x_vec.size < 2 or y_vec.size < 2:
                raise ValueError("Grid te klein; controleer domein/grootte.")
            gx, gy = np.meshgrid(x_vec, y_vec)

            self._log("Vaargebied-domein opbouwen...")
            domain_mask = _build_domain_mask(gx, gy, tr_x, tr_y, domain_buffer)
            if np.count_nonzero(domain_mask) == 0:
                raise ValueError("Domeinmasker leeg. Verhoog Domein buffer.")

            hull_mask = _build_track_hull_mask(gx, gy, tr_x, tr_y)
            if hull_mask is not None:
                flow_mask = domain_mask & hull_mask
                if np.count_nonzero(flow_mask) == 0:
                    raise ValueError("Stromingsdomein leeg na trackgrens-filter. Controleer buffer/trackdata.")
                self._log(
                    f"Stromingsdomein (strict trackgrenzen): {np.count_nonzero(flow_mask)} / {flow_mask.size}"
                )
            else:
                flow_mask = domain_mask
                self._log("WAARSCHUWING: trackgrenzen (hull) niet bruikbaar, val terug op corridor-domein.")

            if limit_to_track_bounds and (hull_mask is not None):
                domain_mask = flow_mask
            self._log(f"Domeincellen algemeen: {np.count_nonzero(domain_mask)} / {domain_mask.size}")
            self._log(f"Diepte overdrijving: x{depth_exaggeration:.2f}")

            max_depth = float(np.nanmax(-src_z))
            if not np.isfinite(max_depth) or max_depth <= 0.0:
                raise ValueError("Kan geen geldige diepte bepalen.")
            depth_levels = np.arange(0.0, max_depth + 0.5 * z_step, z_step, dtype=float)
            if depth_levels.size < 2:
                depth_levels = np.array([0.0, max_depth], dtype=float)
            z_levels = -depth_levels
            self._log(f"Z levels: {z_levels.size} (max diepte {max_depth:.2f} m)")

            bathy_grid: np.ndarray | None = None
            if interp_bathy:
                self._log("Bodem (bathymetrie) interpoleren...")
                bathy_grid = _interpolate_bathymetry_grid(
                    grid_x=gx,
                    grid_y=gy,
                    domain_mask=flow_mask,
                    track_x=tr_x,
                    track_y=tr_y,
                    bed_depth=tr_bed,
                    k=idw_k,
                    power=idw_power,
                    radius=idw_radius,
                    extrapolate=extrapolate,
                )
                n_bathy = np.count_nonzero(np.isfinite(bathy_grid))
                self._log(f"Bathy grid punten: {n_bathy}")
                if export_dtm:
                    if not out_dtm:
                        raise ValueError("DTM export staat aan, maar pad is leeg.")
                    self._log("DTM (.asc) schrijven...")
                    p_dtm = write_esri_ascii_dtm(out_dtm, x_vec, y_vec, bathy_grid)
                    self._log(f"DTM opgeslagen: {p_dtm}")

            self._log("3D snelheid interpoleren...")
            speed_vol = _interpolate_speed_volume(
                grid_x=gx,
                grid_y=gy,
                z_levels=z_levels,
                domain_mask=flow_mask,
                src_x=src_x,
                src_y=src_y,
                src_z=src_z,
                src_speed=src_speed,
                k=idw_k,
                power=idw_power,
                radius=idw_radius,
                extrapolate=extrapolate,
                bathy_depth=bathy_grid if interp_bathy else None,
            )
            n_interp = np.count_nonzero(np.isfinite(speed_vol))
            if n_interp == 0:
                raise ValueError("Interpolatie gaf geen geldige snelheidswaarden.")
            self._log(f"Geinterpoleerde 3D punten: {n_interp}")

            self._log("3D figuur bouwen...")
            fig = build_3d_slice_figure(
                grid_x=gx,
                grid_y=gy,
                z_levels=z_levels,
                speed_vol=speed_vol,
                track_x=tr_x,
                track_y=tr_y,
                bathy_depth=bathy_grid if interp_bathy else None,
                color_scale=color_scale,
                opacity=slice_opacity,
                max_slices=max_slices,
                depth_exaggeration=depth_exaggeration,
                speed_cmin=speed_cmin,
                speed_cmax=speed_cmax,
                bathy_color_scale=bathy_color_scale,
                bathy_cmin=bathy_cmin,
                bathy_cmax=bathy_cmax,
            )

            outp = Path(out_html)
            outp.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(
                str(outp),
                include_plotlyjs="cdn",
                full_html=True,
                config={"responsive": True},
                post_script=_plotly_layer_controls_post_script(default_flow_scale=color_scale, default_bathy_scale=bathy_color_scale),
            )
            self._log(f"3D HTML opgeslagen: {outp}")

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

            messagebox.showinfo("Klaar", "Interpolatie en export voltooid.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"FOUT: {e}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
