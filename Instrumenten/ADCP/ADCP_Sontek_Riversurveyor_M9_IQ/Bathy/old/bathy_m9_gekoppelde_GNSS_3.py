#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATHY GUI â€“ 1D-lasso â†’ Interpolatie/Extrapolatie â†’ 3D-raster bewerken (zonder her-interpolatie)
- Bodemdiepte altijd via BT_Depth (per sessie).
- Positie: DGPS (UTM of Lat/Lonâ†’UTM); zo niet: Summary.Track + UTM-anker.
- Stap 1: 1D-profiel met Lasso outlier-selectie (Delete/Undo/Reset).
- Stap 2: Interpolatie naar 3D-grid (cubic/linear) + optioneel extrapolatie (nearest) buiten convex hull.
- Stap 3: 3D Rasterbewerker: selecteer in het ge(r)asterde oppervlak en zet geselecteerde cellen op NaN (Undo/Redo).
- Verdere opties: spikefilter, min/max-diepte, randtrim, hoogteoverdrijving, planview tracks, DAE-export.

Auteur: Stinus-helper
"""

import os
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from scipy.ndimage import median_filter, generic_filter, binary_dilation
from scipy.stats import zscore
from scipy.spatial import Delaunay, QhullError
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import proj3d
import trimesh

# ---- optionele CRS/projectie
try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False


# ============== .MAT helpers ==============

def loadmat(filepath):
    mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    def _todict(mo):
        d = {}
        if not hasattr(mo, "_fieldnames"):
            return mo
        for f in mo._fieldnames:
            v = getattr(mo, f)
            if isinstance(v, scipy.io.matlab.mat_struct):
                d[f] = _todict(v)
            elif isinstance(v, np.ndarray):
                d[f] = _tolist(v)
            else:
                d[f] = v
        return d
    def _tolist(a):
        if not isinstance(a, np.ndarray):
            return a
        out = []
        for v in a:
            if isinstance(v, scipy.io.matlab.mat_struct):
                out.append(_todict(v))
            elif isinstance(v, np.ndarray):
                out.append(_tolist(v))
            else:
                out.append(v)
        return out
    def _check(d):
        for k in list(d.keys()):
            if k.startswith("__"): continue
            if isinstance(d[k], scipy.io.matlab.mat_struct):
                d[k] = _todict(d[k])
            elif isinstance(d[k], np.ndarray):
                d[k] = _tolist(d[k])
        return d
    return _check(mat)

def _as_array(x):
    if x is None: return None
    return np.array(x).squeeze()

def _first_present(dct, keys):
    if dct is None: return None, None
    low = {k.lower(): k for k in dct.keys()}
    for k in keys:
        if k.lower() in low:
            kreal = low[k.lower()]
            return kreal, dct[kreal]
    return None, None

def _matlab_datenum_to_datetime64(dnum):
    base = np.datetime64('1970-01-01T00:00:00')
    sec = (np.array(dnum, dtype=float) - 719529.0) * 86400.0
    return base + (sec.astype('timedelta64[s]')).astype('timedelta64[ns]')

def _to_datetime64(t):
    a = _as_array(t)
    if a is None: return None
    a = a.astype('float64', copy=False) if np.issubdtype(np.array(a).dtype, np.number) else a
    try:
        if np.issubdtype(np.array(a).dtype, np.number):
            mx = np.nanmax(a); mn = np.nanmin(a)
            if 1e5 < mn and mx < 1e7:
                return _matlab_datenum_to_datetime64(a)
            if mx > 1e10:
                return np.datetime64('1970-01-01') + a.astype('timedelta64[ms]')
            return np.datetime64('1970-01-01') + a.astype('timedelta64[s]')
        else:
            return a.astype('datetime64[ns]')
    except Exception:
        return None


# ============== Diepte & positie ==============

_DEPTH_CHOICE = None  # 'vb' of 'bt' â€” Ã©Ã©nmalig per sessie

def extract_depth(data):
    """Gebruik altijd BottomTrack.BT_Depth; geef Z (m) en t (optioneel)."""
    if 'BottomTrack' not in data:
        raise KeyError("BottomTrack ontbreekt in .mat")
    bt = data['BottomTrack']
    cand_bt = ['BT_Depth', 'BTdepth', 'BT_Depths']

    if isinstance(bt, dict):
        _, bz = _first_present(bt, cand_bt)
    else:
        bz = None
        for k in cand_bt:
            if hasattr(bt, k):
                bz = getattr(bt, k)
                break

    if bz is None:
        raise KeyError("Geen BT_Depth gevonden.")

    Z = _as_array(bz).astype(float)
    # tijd
    t_depth = None
    if 'Summary' in data and isinstance(data['Summary'], dict):
        t_depth = _to_datetime64(data['Summary'].get('Time', None))
    if t_depth is None and isinstance(bt, dict) and 'Time' in bt:
        t_depth = _to_datetime64(bt.get('Time'))
    return Z, t_depth

# Lat/Lon â†’ UTM (optioneel)
try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

def _latlon_to_utm(lat, lon, epsg=None, auto_zone=True):
    if not _HAS_PYPROJ:
        raise ImportError("pyproj vereist (pip install pyproj)")
    lat = np.asarray(lat, float).ravel()
    lon = np.asarray(lon, float).ravel()
    if epsg is None and auto_zone:
        zones = np.floor((lon + 180)/6).astype(int) + 1
        zone = int(np.bincount(zones[np.isfinite(zones)]).argmax()) if np.any(np.isfinite(zones)) else 31
        epsg = 32600 + zone
    elif epsg is None:
        epsg = 32631
    x, y = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(int(epsg)), always_xy=True).transform(lon, lat)
    return np.asarray(x), np.asarray(y), int(epsg)

def extract_xy(data, epsg_pref=32631, allow_auto_zone=True):
    """Probeer DGPS (UTM of Lat/Lon)."""
    # 1) UTM
    for key in ['Navigation','GPS','GNSS','NMEA','Summary','gps','navigation']:
        c = data.get(key) if isinstance(data, dict) else None
        if not isinstance(c, dict): continue
        _, X = _first_present(c, ['Easting','UTM_E','X','E','East'])
        _, Y = _first_present(c, ['Northing','UTM_N','Y','N','North'])
        if X is not None and Y is not None:
            X = _as_array(X).astype(float); Y = _as_array(Y).astype(float)
            if np.nanstd(X) > 1e-9 or np.nanstd(Y) > 1e-9:
                return X, Y, "DGPS UTM"
    # 2) Lat/Lon â†’ UTM
    for key in ['Navigation','GPS','GNSS','NMEA','Summary','gps','navigation']:
        c = data.get(key) if isinstance(data, dict) else None
        if not isinstance(c, dict): continue
        _, LA = _first_present(c, ['Latitude','Lat','LAT'])
        _, LO = _first_present(c, ['Longitude','Lon','LON','Long'])
        if LA is not None and LO is not None:
            lat = _as_array(LA).astype(float); lon = _as_array(LO).astype(float)
            if np.nanstd(lat) > 1e-12 or np.nanstd(lon) > 1e-12:
                X, Y, used = _latlon_to_utm(lat, lon, epsg=epsg_pref, auto_zone=allow_auto_zone)
                return X, Y, f"DGPS Lat/Lonâ†’UTM (EPSG:{used})"
    raise KeyError("Geen DGPS (UTM/LatLon) gevonden.")


# ============== Interpolatie & plots ==============

def _cumulative_distance(X, Y):
    d = np.hypot(np.diff(X), np.diff(Y))
    return np.concatenate(([0.0], np.cumsum(d)))

def _dedup_valid_xy(X, Y, Z):
    m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[m], Y[m], Z[m]
    if X.size == 0: return X, Y, Z
    pts = np.column_stack((X, Y))
    key = np.round(pts, 3)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Z_mean = np.zeros(idx.size, float); np.add.at(Z_mean, inv, Z)
    cnt = np.bincount(inv); Z_mean = Z_mean/np.maximum(cnt,1)
    XYu = pts[idx]
    return XYu[:,0], XYu[:,1], Z_mean

def _nearly_collinear(X, Y, eps_rel=1e-6):
    if X.size < 3: return True
    XY = np.column_stack((X - np.mean(X), Y - np.mean(Y)))
    cov = np.cov(XY.T); w, _ = np.linalg.eigh(cov); w = np.sort(np.maximum(w,0))
    if w[-1] == 0: return True
    return (w[0]/w[-1]) < eps_rel

class BathyModel:
    def interpolate(self, X, Y, Z, grid_res=150, extrapolate=False):
        """
        Maak 2D grid (Xi,Yi,Zi). Buiten hull op NaN, tenzij extrapolate=True (dan nearest).
        """
        X, Y, Z = _dedup_valid_xy(X, Y, Z)
        points = np.column_stack((X, Y))
        if X.size < 10:
            raise ValueError("Te weinig geldige punten na deduplicatie.")
        if _nearly_collinear(X, Y) or np.ptp(X)==0 or np.ptp(Y)==0:
            raise ValueError("XY (bijna) 1D â€“ raster niet zinvol.")

        xi = np.linspace(np.nanmin(X), np.nanmax(X), grid_res)
        yi = np.linspace(np.nanmin(Y), np.nanmax(Y), grid_res)
        Xi, Yi = np.meshgrid(xi, yi)

        Zi = None
        for method in ('cubic', 'linear'):
            try:
                Zi = griddata(points, Z, (Xi, Yi), method=method)
                break
            except QhullError:
                continue
        if Zi is None:
            # laatste redmiddel â€“ jitter + linear
            jitter = 1e-3 * max(np.ptp(X), np.ptp(Y))
            Pj = points + np.random.uniform(-jitter, jitter, size=points.shape)
            Zi = griddata(Pj, Z, (Xi, Yi), method='linear')

        # mask buiten convex hull => NaN
        try:
            hull = Delaunay(points)
            mask = hull.find_simplex(np.column_stack((Xi.ravel(), Yi.ravel()))) >= 0
            Zi.ravel()[~mask] = np.nan
        except Exception:
            pass

        if extrapolate:
            # vul NaN via nearest vanaf de ruwe punten (eenmalig, geen â€œher-interpolatieâ€ bij bewerken)
            Zi_near = griddata(points, Z, (Xi, Yi), method='nearest')
            Zi = np.where(np.isnan(Zi), Zi_near, Zi)

        Zi = self._postprocess(Xi, Yi, Zi)
        return Xi, Yi, Zi

    def _postprocess(self, Xi, Yi, Zi, max_slope=0.5, n_iter=2):
        def nanmean_filter(vals):
            v = vals[~np.isnan(vals)]
            return np.mean(v) if len(v) else np.nan
        for _ in range(2):
            if np.any(np.isnan(Zi)):
                Zi = generic_filter(Zi, nanmean_filter, size=3, mode='nearest')
        # lichte smoothing + hellingbegrenzer
        from numpy import hypot
        for _ in range(n_iter):
            Zi = median_filter(Zi, size=3)
            rows, cols = Zi.shape
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    zc = Zi[i, j]
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        zn = Zi[i+di, j+dj]
                        dx = hypot(Xi[i,j]-Xi[i+di,j+dj], Yi[i,j]-Yi[i+di,j+dj])
                        if dx > 0 and abs(zc-zn) > max_slope*dx:
                            Zi[i, j] = zn + np.sign(zc-zn)*max_slope*dx
        return Zi

    def trim_edges(self, Zi, n_cells=0):
        if n_cells <= 0: return Zi
        Zt = Zi.copy(); r, c = Zt.shape
        n = int(min(n_cells, r//2, c//2))
        if n>0:
            Zt[:n,:]=np.nan; Zt[-n:,:]=np.nan; Zt[:,:n]=np.nan; Zt[:,-n:]=np.nan
        return Zt

    def plot_surface(self, Xi, Yi, Zi, zexag=1.0, title="Bathymetrisch raster"):
        fig = plt.figure(figsize=(10,7)); ax = fig.add_subplot(111, projection='3d')
        depth = -Zi
        norm = plt.Normalize(np.nanmin(depth), np.nanmax(depth))
        cmap = 'turbo' if 'turbo' in plt.colormaps() else 'rainbow'
        colors = plt.cm.get_cmap(cmap)(norm(depth))
        ax.plot_surface(Xi, Yi, depth, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=True)
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.5); cbar.set_label("Diepte (m)")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_zlabel("Diepte (m)")
        ax.set_title(title)
        try: ax.set_box_aspect([1,1,zexag])
        except Exception: pass
        plt.tight_layout(); plt.show()

    def plot_profile(self, S, Z, title="Bathymetrisch profiel (1D)"):
        Zm = median_filter(Z, size=3)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(S, -Z, '.', alpha=0.6, label='punten')
        ax.plot(S, -Zm, '-', lw=1.4, label='mediaan(3)')
        ax.set_xlabel("Afstand (m)"); ax.set_ylabel("Diepte (m)")
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.show()

    def apply_spike_filter(self, Z, thresh=3.0, median_size=3):
        Z_med = median_filter(Z, size=median_size, mode='nearest')
        Zf = Z.copy()
        mask = np.abs(zscore(Zf, nan_policy='omit')) > thresh
        Zf[mask] = Z_med[mask]
        return Zf, int(mask.sum())


# ============== 1D Lasso editor ==============

class LassoEditor:
    def __init__(self, S, Z):
        self.S = np.array(S, float); self.Z = np.array(Z, float)
        self.keep = np.isfinite(self.S) & np.isfinite(self.Z)
        self.undo_stack = []; self._last_sel = np.array([], int)
        self.fig, self.ax = plt.subplots(figsize=(11,5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.lasso = LassoSelector(self.ax, onselect=self.on_lasso)
        self.redraw(); self._closed=False

    def redraw(self):
        self.ax.clear()
        s, z = self.S[self.keep], self.Z[self.keep]
        self.ax.scatter(s, -z, s=9, alpha=0.75)
        self.ax.set_xlabel("Afstand (m)"); self.ax.set_ylabel("Diepte (m)")
        self.ax.set_title(f"1D Lasso: D=delete, U=undo, R=reset, Q=sluit (N={s.size})")
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw_idle()

    def on_lasso(self, verts):
        path = Path(verts)
        pts = np.column_stack((self.S[self.keep], -self.Z[self.keep]))
        sel = path.contains_points(pts)
        self._last_sel = np.where(self.keep)[0][sel] if sel.any() else np.array([],int)
        if sel.any():
            self.ax.scatter(pts[sel,0], pts[sel,1], s=25, facecolors='none', edgecolors='r')
            self.fig.canvas.draw_idle()

    def on_key(self, ev):
        k = (ev.key or '').lower()
        if k=='d' and self._last_sel.size:
            self.undo_stack.append(self.keep.copy())
            self.keep[self._last_sel]=False; self._last_sel=np.array([],int); self.redraw()
        elif k=='u' and self.undo_stack:
            self.keep = self.undo_stack.pop(); self.redraw()
        elif k=='r':
            self.keep[:] = True; self.undo_stack.clear(); self.redraw()
        elif k=='q':
            plt.close(self.fig); self._closed=True

    def show(self):
        plt.show(block=True); return self.keep


# ============== 3D Rasterbewerker (op grid) ==============

class Lasso2DSelector:
    """
    Zuivere 2D scatter + lasso (geen lijnen). Retourneert boolean mask over inputpunten.
    Keys:
      - D / Enter: bevestig selectie (return)
      - U: deselect alles
      - Q / Escape: annuleer (geen selectie)
    """
    def __init__(self, pts2d, title="2D selectie (Lasso)"):
        import matplotlib.pyplot as plt
        self.pts2d = np.asarray(pts2d, float)
        self.keep = np.ones(self.pts2d.shape[0], dtype=bool)  # initieel allemaal keep
        self.sel_mask = np.zeros_like(self.keep)
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.scat = self.ax.scatter(self.pts2d[:,0], self.pts2d[:,1], s=6, alpha=0.9)  # puntenwolk, geen lijnen
        self.ax.set_title(title + " â€” Lasso tekenen; D/Enter=bevestig, U=reset, Q=annuleer")
        self.ax.set_xlabel("screen x (px)"); self.ax.set_ylabel("screen y (px)")
        self.ax.invert_yaxis()  # scherm-coÃ¶rdinaten: y naar beneden
        self.ax.grid(True, alpha=0.2)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        # duidelijke lasso-lijn
        try:
            # Meeste versies (3.3+): 'lineprops'
            self.lasso = LassoSelector(
                self.ax, onselect=self._on_lasso, useblit=False,
                lineprops=dict(color='yellow', linewidth=1.8, alpha=0.95)
            )
        except TypeError:
            # Oudere varianten: zonder lineprops; stijl rechtstreeks op de lijn zetten
            self.lasso = LassoSelector(self.ax, onselect=self._on_lasso, useblit=False)
            try:
                self.lasso.line.set_color('yellow')
                self.lasso.line.set_linewidth(1.8)
                self.lasso.line.set_alpha(0.95)
            except Exception:
                pass

    def on_lasso(self, verts):
        path = Path(verts)
        sel = path.contains_points(self.pts2d)
        self.sel_mask = sel
        self._highlight(sel)

    def _highlight(self, sel):
        # kleurpuntjes: rood = geselecteerd, blauw = niet geselecteerd
        c = np.full((self.pts2d.shape[0], 4), [0.2, 0.4, 1.0, 0.9])  # blauw
        if sel is not None and np.any(sel):
            c[sel] = [1.0, 0.2, 0.2, 0.95]  # rood
        self.scat.set_color(c)
        self.fig.canvas.draw_idle()

    def on_key(self, ev):
        k = (ev.key or '').lower()
        if k in ('d','enter','return'):
            plt.close(self.fig)
        elif k == 'u':
            self.sel_mask[:] = False
            self._highlight(self.sel_mask)
        elif k in ('q','escape'):
            self.sel_mask[:] = False
            plt.close(self.fig)

    def show(self):
        import matplotlib.pyplot as plt
        plt.show(block=True)
        return self.sel_mask


class RasterEditor3D:
    """
    3D-raster bewerken zonder herinterpolatie.
    Werkwijze:
      - Roteer/zoom 3D naar wens.
      - Druk L â†’ er opent een 2D scatter van de geprojecteerde grid-nodes (geen lijnen).
      - Lasso in 2D, D/Enter bevestigt â†’ de gekozen grid-nodes worden op NaN gezet.
      - U = undo, Q = sluit.
    """
    def __init__(self, Xi, Yi, Zi, zexag=1.0, dilate_pixels=0, visible_only=True):
        import matplotlib.pyplot as plt
        self.Xi, self.Yi = Xi, Yi
        self.Zi = Zi.copy()
        self.zexag = float(zexag)
        self.dilate_px = int(max(0, dilate_pixels))
        self.visible_only = bool(visible_only)

        self.undo_stack = []
        self.fig = plt.figure(figsize=(10,7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self._draw()

    # ---------- 3D rendering ----------
    def _draw(self, subtitle=None):
        self.ax.clear()
        depth = -self.Zi
        norm = plt.Normalize(np.nanmin(depth), np.nanmax(depth))
        cmap = 'turbo' if 'turbo' in plt.colormaps() else 'rainbow'
        colors = plt.cm.get_cmap(cmap)(norm(depth))
        facecolors = colors.copy()
        facecolors[np.isnan(depth)] = [0,0,0,0]
        self.ax.plot_surface(self.Xi, self.Yi, depth, facecolors=facecolors,
                             rstride=1, cstride=1, linewidth=0, antialiased=True)
        cbar = self.fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=self.ax, shrink=0.5)
        cbar.set_label("Diepte (m)")
        self.ax.set_xlabel("Easting (m)"); self.ax.set_ylabel("Northing (m)"); self.ax.set_zlabel("Diepte (m)")
        title = "3D Rasterbewerker â€” L: 2D-lasso  D: delete  U: undo  Q: quit"
        if subtitle: title += f"  [{subtitle}]"
        self.ax.set_title(title)
        try: self.ax.set_box_aspect([1,1,self.zexag])
        except Exception: pass
        self.fig.canvas.draw_idle()

    # ---------- projectie helpers ----------
    def _project_nodes_to_pixels(self):
        """Projecteer grid-nodes naar schermpixels (2D)."""
        Zplot = -self.Zi
        xs, ys, zs = proj3d.proj_transform(self.Xi.ravel(), self.Yi.ravel(), Zplot.ravel(), self.ax.get_proj())
        pts2d = np.vstack([xs, ys]).T
        pix = self.ax.transData.transform(pts2d)
        return pix  # shape (N,2), in pixels

    def _front_facing_mask_nodes(self):
        """Benader zichtbaarheidsmasker (front-facing)."""
        elev = np.deg2rad(self.ax.elev); azim = np.deg2rad(self.ax.azim)
        v = np.array([np.cos(elev)*np.sin(azim),
                      -np.cos(elev)*np.cos(azim),
                      np.sin(elev)], float)
        Zp = -self.Zi
        dXi_i = np.gradient(self.Xi, axis=0); dYi_i = np.gradient(self.Yi, axis=0); dZi_i = np.gradient(Zp, axis=0)
        dXi_j = np.gradient(self.Xi, axis=1); dYi_j = np.gradient(self.Yi, axis=1); dZi_j = np.gradient(Zp, axis=1)
        nx = dYi_i * dZi_j - dZi_i * dYi_j
        ny = dZi_i * dXi_j - dXi_i * dZi_j
        nz = dXi_i * dYi_j - dYi_i * dXi_j
        dot = nx*v[0] + ny*v[1] + nz*v[2]
        front = (dot < 0)
        node = np.zeros(self.Zi.shape, bool)
        node[:-1,:-1] |= front[:-1,:-1]
        node[1: ,:-1] |= front[:-1,:-1]
        node[:-1,1: ] |= front[:-1,:-1]
        node[1: ,1: ] |= front[:-1,:-1]
        return node.ravel()

    # ---------- acties ----------
    def on_key(self, ev):
        k = (ev.key or '').lower()
        if k == 'l':
            self._start_2d_selector()
        elif k == 'd':
            # niets te doen: delete gebeurt direct na 2D-selectie
            pass
        elif k == 'u':
            self._undo()
        elif k == 'q':
            import matplotlib.pyplot as plt
            plt.close(self.fig)

    def _start_2d_selector(self):
        # 0) toolbar-modus uit
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and hasattr(tb, "mode"):
            tb.mode = ''
        # 1) snapshot: projecteer nodes naar 2D (pixels)
        pix = self._project_nodes_to_pixels()
        # 2) optioneel enkel zichtbare nodes
        if self.visible_only:
            vis = self._front_facing_mask_nodes()
        else:
            vis = np.ones(self.Xi.size, bool)
        idx_all = np.arange(self.Xi.size)
        idx_vis = idx_all[vis]
        pts_vis = pix[vis]

        # 3) open 2D-scatter lasso (geen lijnen)
        selector = Lasso2DSelector(pts_vis, title="Geprojecteerde puntenwolk (screen-space)")
        sel_mask_vis = selector.show()  # booleans over pts_vis

        if sel_mask_vis is None or not np.any(sel_mask_vis):
            self._draw()
            return

        # 4) map terug naar gridvorm, dilate optioneel, zet op NaN
        sel_flat = np.zeros(self.Xi.size, bool)
        sel_flat[idx_vis[sel_mask_vis]] = True
        sel_grid = sel_flat.reshape(self.Xi.shape)

        if self.dilate_px > 0:
            sel_grid = binary_dilation(sel_grid, iterations=self.dilate_px)

        # undo + toepassen
        self.undo_stack.append(self.Zi.copy())
        self.Zi[sel_grid] = np.nan
        self._draw(subtitle=f"verwijderd: {sel_mask_vis.sum()} nodes")

    def _undo(self):
        if not self.undo_stack:
            return
        self.Zi = self.undo_stack.pop()
        self._draw()

    def show(self):
        import matplotlib.pyplot as plt
        plt.show(block=True)
        return self.Zi


# ============== Hoofd-GUI ==============

class BathyGUI:
    def __init__(self, root):
        root.title("Bathy GUI â€“ 1D â†’ 3D + Rasterbewerker (zonder her-interpolatie)")
        self.model = BathyModel()

        # data containers
        self.files=[]; self.X_list=[]; self.Y_list=[]; self.Z_list=[]; self.depth_types=[]
        self.X_all=self.Y_all=self.Z_all=None; self.S_all=None
        self.keep_mask=None          # 1D-keuzes
        self.Xi=self.Yi=self.Zi=None # 3D grid (bewerkbaar!)
        self.zexag_val=1.0

        frm = tk.Frame(root); frm.pack(padx=10, pady=6)
        tk.Label(frm, text="EPSG (UTM):").grid(row=0,column=0,sticky='e')
        self.epsg=tk.Entry(frm,width=10); self.epsg.insert(0,"32631"); self.epsg.grid(row=0,column=1,sticky='w')
        self.auto_zone=tk.BooleanVar(value=True)
        tk.Checkbutton(frm,text="Auto zone bij Lat/Lon",var=self.auto_zone).grid(row=0,column=2,sticky='w')

        tk.Label(frm, text="Tijdstolerantie (s):").grid(row=1,column=0,sticky='e')
        self.tol=tk.Scale(frm, from_=0, to=30, resolution=0.5, orient='horizontal', length=220); self.tol.set(2.0); self.tol.grid(row=1,column=1)

        tk.Label(frm, text="Spike (z-score):").grid(row=2,column=0,sticky='e')
        self.spike=tk.Scale(frm, from_=0, to=10, resolution=0.1, orient='horizontal', length=220); self.spike.set(3.0); self.spike.grid(row=2,column=1)
        tk.Label(frm, text="Median (punten):").grid(row=3,column=0,sticky='e')
        self.median=tk.Scale(frm, from_=1, to=9, resolution=2, orient='horizontal', length=220); self.median.set(3); self.median.grid(row=3,column=1)

        tk.Label(frm, text="Min diepte (m):").grid(row=4,column=0,sticky='e')
        self.minz=tk.Entry(frm,width=8); self.minz.insert(0,""); self.minz.grid(row=4,column=1,sticky='w')
        tk.Label(frm, text="Max diepte (m):").grid(row=5,column=0,sticky='e')
        self.maxz=tk.Entry(frm,width=8); self.maxz.insert(0,""); self.maxz.grid(row=5,column=1,sticky='w')

        tk.Label(frm, text="Grid resolutie:").grid(row=6,column=0,sticky='e')
        self.gridres=tk.Entry(frm,width=8); self.gridres.insert(0,"150"); self.gridres.grid(row=6,column=1,sticky='w')

        self.extrap=tk.BooleanVar(value=True)
        tk.Checkbutton(frm,text="Extrapoleer buiten huls (nearest)",var=self.extrap).grid(row=6,column=2,sticky='w')

        tk.Label(frm, text="Randtrim (cellen):").grid(row=7,column=0,sticky='e')
        self.trim=tk.Scale(frm, from_=0, to=10, resolution=1, orient='horizontal', length=220); self.trim.set(0); self.trim.grid(row=7,column=1)

        tk.Label(frm, text="Hoogteoverdrijving:").grid(row=8,column=0,sticky='e')
        self.zexag=tk.Scale(frm, from_=0.1, to=10, resolution=0.1, orient='horizontal', length=220); self.zexag.set(1.0); self.zexag.grid(row=8,column=1)

        btns=tk.Frame(root); btns.pack(pady=8)
        tk.Button(btns,text="1) Selecteer .mat & 1D Lasso",command=self.load_and_lasso).pack(side='left',padx=5)
        tk.Button(btns,text="2) Interpoleer / Plot 3D",command=self.build_and_plot_3d).pack(side='left',padx=5)
        tk.Button(btns,text="3D Rasterbewerker",command=self.edit_raster_3d).pack(side='left',padx=5)
        tk.Button(btns,text="Planview Tracks",command=self.show_tracks).pack(side='left',padx=5)
        tk.Button(btns,text="Export DAE",command=self.export_dae).pack(side='left',padx=5)

    # --------- loading + 1D lasso ---------

    def load_and_lasso(self):
        files = filedialog.askopenfilenames(filetypes=[("MAT files","*.mat")])
        if not files: return
        epsg = self._parse_epsg()

        # reset
        self.files.clear(); self.X_list.clear(); self.Y_list.clear(); self.Z_list.clear(); self.depth_types.clear()
        self.X_all=self.Y_all=self.Z_all=None; self.S_all=None; self.keep_mask=None
        self.Xi=self.Yi=self.Zi=None

        global _DEPTH_CHOICE
        _DEPTH_CHOICE=None

        for f in files:
            try:
                data = loadmat(f); base=os.path.basename(f)
                Z, tZ = extract_depth(data); Z=Z.astype(float)
                X, Y, src = extract_xy(data, epsg_pref=epsg, allow_auto_zone=self.auto_zone.get())
                # lengte afstemmen
                L = min(len(X),len(Y),len(Z)); X,Y,Z = X[:L],Y[:L],Z[:L]
                self.files.append(base); self.X_list.append(X); self.Y_list.append(Y); self.Z_list.append(Z)
                self.depth_types.append(_DEPTH_CHOICE.upper())
                messagebox.showinfo("Positiebron", f"{base}\nBron: {src}")
            except Exception as ex:
                messagebox.showwarning("Bestand overgeslagen", f"{os.path.basename(f)}:\n{ex}")

        if not self.Z_list: return
        self.X_all = np.concatenate(self.X_list); self.Y_all = np.concatenate(self.Y_list); self.Z_all = np.concatenate(self.Z_list)
        self.S_all = _cumulative_distance(self.X_all, self.Y_all)

        ed = LassoEditor(self.S_all, self.Z_all)
        keep = ed.show()
        self.keep_mask = keep.astype(bool)

    def _parse_epsg(self):
        try: return int(self.epsg.get().strip())
        except Exception: return 32631

    def _bounds_mask(self, Z):
        md=self.minz.get().strip(); xd=self.maxz.get().strip()
        min_d=float(md) if md!="" else None; max_d=float(xd) if xd!="" else None
        m = np.ones(Z.shape, bool)
        if min_d is not None: m &= (Z >= min_d)
        if max_d is not None: m &= (Z <= max_d)
        return m

    # --------- interpolatie + plot ---------

    def build_and_plot_3d(self):
        if self.keep_mask is None:
            messagebox.showwarning("Let op","Laad data en voer 1D lasso eerst uit."); return
        try:
            grid_res = int(self.gridres.get().strip())
        except Exception:
            grid_res = 150

        # selectie + spikes
        keep = self.keep_mask & self._bounds_mask(self.Z_all)
        Zf = self.Z_all.copy()
        sel = np.where(keep)[0]
        if sel.size > 0:
            Zsub_f,_ = self.model.apply_spike_filter(Zf[keep], thresh=float(self.spike.get()), median_size=int(self.median.get()))
            Zf[keep] = Zsub_f

        # punten voor raster
        Xk, Yk, Zk = self.X_all[keep], self.Y_all[keep], Zf[keep]
        if Zk.size < 10:
            messagebox.showwarning("Te weinig punten","Pas selectie/filters aan."); return

        # maak raster (eenmalig; hier mag extrapolatie)
        Xi, Yi, Zi = self.model.interpolate(Xk, Yk, Zk, grid_res=grid_res, extrapolate=bool(self.extrap.get()))
        Zi = self.model.trim_edges(Zi, n_cells=int(self.trim.get()))

        # bewaar als huidig bewerkbaar raster
        self.Xi, self.Yi, self.Zi = Xi, Yi, Zi
        self.zexag_val = float(self.zexag.get())

        tag=set(self.depth_types); suffix=f" ({tag.pop()})" if len(tag)==1 else " (gemengd)"
        self.model.plot_surface(Xi, Yi, Zi, zexag=self.zexag_val, title=f"Bathymetrisch raster{suffix}")

    # --------- 3D Rasterbewerker ---------

    def edit_raster_3d(self):
        if self.Zi is None:
            messagebox.showwarning("Geen raster", "Maak eerst het 3D-raster (knop 2).")
            return
        # aantal cellen rand meenemen (optioneel)
        try:
            dil = simpledialog.askinteger("Selectie-dilatie", "Aantal dilatie-iteraties (0â€“3):",
                                        initialvalue=0, minvalue=0, maxvalue=5)
            if dil is None: dil = 0
        except Exception:
            dil = 0

        editor = RasterEditor3D(self.Xi, self.Yi, self.Zi,
                        zexag=float(self.zexag.get()),
                        dilate_pixels=int(dil),   # <-- niet 'dilate_cells'
                        visible_only=True)
        # snelle replot:
        self.model.plot_surface(self.Xi, self.Yi, self.Zi,
                                zexag=float(self.zexag.get()),
                                title="Bathymetrisch raster (bewerkt)")

    # --------- Tracks ---------

    def show_tracks(self):
        if not self.X_list or self.keep_mask is None:
            messagebox.showwarning("Let op","Geen data of selectie."); return
        keep = self.keep_mask & self._bounds_mask(self.Z_all)
        fig,ax = plt.subplots(figsize=(8,7))
        start=0
        for base,X,Y,Z in zip(self.files,self.X_list,self.Y_list,self.Z_list):
            sl=slice(start,start+len(Z)); start+=len(Z)
            kk=keep[sl]
            if kk.any():
                ax.plot(X[kk],Y[kk],'-',lw=1.3,label=base)
        ax.set_aspect('equal',adjustable='datalim')
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.set_title("Planview Tracks (na selectie)"); ax.grid(True,alpha=0.3)
        ax.legend(fontsize='small'); plt.tight_layout(); plt.show()

    # --------- DAE export ---------

    def export_dae(self):
        if self.Zi is None:
            messagebox.showwarning("Let op","Geen raster beschikbaar."); return
        path = filedialog.asksaveasfilename(defaultextension=".dae", filetypes=[("COLLADA","*.dae")])
        if not path: return
        try:
            verts, faces = self._grid_to_mesh(self.Xi, self.Yi, self.Zi, zexag=self.zexag_val)
            trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)
            messagebox.showinfo("Klaar", f"DAE opgeslagen:\n{path}")
        except Exception as ex:
            messagebox.showerror("Fout bij export", str(ex))

    def _grid_to_mesh(self, Xi, Yi, Zi, zexag=1.0):
        mask=np.isfinite(Zi); r,c=Zi.shape
        verts=np.vstack((Xi[mask],Yi[mask],-Zi[mask]*zexag)).T
        idx=np.full(Zi.shape,-1,int); idx[mask]=np.arange(mask.sum())
        faces=[]
        for i in range(r-1):
            for j in range(c-1):
                a,b,c1,d=idx[i,j],idx[i,j+1],idx[i+1,j],idx[i+1,j+1]
                if a>=0 and b>=0 and c1>=0: faces.append([a,b,c1])
                if d>=0 and b>=0 and c1>=0: faces.append([c1,b,d])
        return verts, np.array(faces)


# ============== main ==============

def main():
    root = tk.Tk()
    BathyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()



