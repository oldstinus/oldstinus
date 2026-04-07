#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bathymetrie Viewer (DGPS-robust)
- Haalt X,Y uit opgenomen DGPS (UTM of Lat/Lon -> UTM via pyproj).
- Tijdssynchronisatie (nearest met tolerantie).
- Robuuste interpolatie:
  * detectie collineariteit/nul-spreiding
  * fallback: linear -> jitter -> 1D-profiel
- Fallback bron: Summary.Track + offsets.
- Spikefilter (z-score + median).
- 3D-plot of 1D-profielplot (auto-switch).
- DAE-export met z-exag + rotatie.

Auteur: Stinus-helper
"""

import numpy as np
import scipy.io
from scipy.interpolate import griddata
from scipy.ndimage import median_filter, generic_filter
from scipy.stats import zscore
from scipy.spatial import Delaunay, QhullError
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import trimesh

# ---- optionele CRS/projectie
try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False


# ================= Helpers voor .mat =================

def loadmat(filepath):
    """Laad .mat als geneste dict inclusief structuren/arrays."""
    def _todict(matobj):
        d = {}
        if not hasattr(matobj, "_fieldnames"):
            return matobj
        for field in matobj._fieldnames:
            elem = getattr(matobj, field)
            if isinstance(elem, scipy.io.matlab.mat_struct):
                d[field] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[field] = _tolist(elem)
            else:
                d[field] = elem
        return d

    def _tolist(ndarray):
        if not isinstance(ndarray, np.ndarray):
            return ndarray
        lst = []
        for elem in ndarray:
            if isinstance(elem, scipy.io.matlab.mat_struct):
                lst.append(_todict(elem))
            elif isinstance(elem, np.ndarray):
                lst.append(_tolist(elem))
            else:
                lst.append(elem)
        return lst

    def _check_keys(d):
        for key in list(d.keys()):
            if key.startswith("__"):
                continue
            if isinstance(d[key], scipy.io.matlab.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _tolist(d[key])
        return d

    mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    return _check_keys(mat)


def _as_array(x):
    if x is None:
        return None
    a = np.array(x).squeeze()
    return a


def _first_present(dct, keys):
    """Haal eerste aanwezige veldnaam op (case-insensitief)."""
    if dct is None:
        return None, None
    lower = {k.lower(): k for k in dct.keys()}
    for k in keys:
        if k.lower() in lower:
            kn = lower[k.lower()]
            return kn, dct[kn]
    return None, None


def _matlab_datenum_to_datetime64(dnum):
    base = np.datetime64('1970-01-01T00:00:00')
    sec = (np.array(dnum, dtype=float) - 719529.0) * 86400.0
    return base + (sec.astype('timedelta64[s]')).astype('timedelta64[ns]')


def _to_datetime64(t):
    a = _as_array(t)
    if a is None:
        return None
    a = a.astype('float64', copy=False) if np.issubdtype(a.dtype, np.number) else a
    try:
        if np.issubdtype(a.dtype, np.number):
            mx = np.nanmax(a); mn = np.nanmin(a)
            if 1e5 < mn and mx < 1e7:         # MATLAB datenum
                return _matlab_datenum_to_datetime64(a)
            if mx > 1e10:                      # ms
                return np.datetime64('1970-01-01') + a.astype('timedelta64[ms]')
            return np.datetime64('1970-01-01') + a.astype('timedelta64[s]')  # s
        else:
            return a.astype('datetime64[ns]')
    except Exception:
        return None


# ================= Extractie diepte & DGPS =================

def extract_depth(data):
    """Kies VB_Depth of BT_Depth; pak tijd uit Summary/BottomTrack indien aanwezig."""
    if 'BottomTrack' not in data:
        raise KeyError("Struct 'BottomTrack' ontbreekt in .mat")
    bt = data['BottomTrack']

    cand_vb = ['VB_Depth', 'VBdepth', 'VB_Depths']
    cand_bt = ['BT_Depth', 'BTdepth', 'BT_Depths']

    if isinstance(bt, dict):
        _, vb_val = _first_present(bt, cand_vb)
        _, bt_val = _first_present(bt, cand_bt)
    else:
        vb_val = None; bt_val = None
        for k in cand_vb:
            if hasattr(bt, k): vb_val = getattr(bt, k); break
        for k in cand_bt:
            if hasattr(bt, k): bt_val = getattr(bt, k); break

    if vb_val is not None and bt_val is not None:
        choice = simpledialog.askstring(
            "Kies diepte",
            "Typ 'vb' voor Vertical Beam (VB_Depth)\n"
            "of 'bt' voor BottomTrack (BT_Depth):"
        )
        use_vb = bool(choice and choice.strip().lower().startswith('vb'))
    elif vb_val is not None:
        use_vb = True
    elif bt_val is not None:
        use_vb = False
    else:
        raise KeyError("Geen VB_Depth of BT_Depth gevonden in 'BottomTrack'.")

    Z = _as_array(vb_val if use_vb else bt_val).astype('float64')
    dtype = 'VB_Depth' if use_vb else 'BT_Depth'

    t_depth = None
    if 'Summary' in data and isinstance(data['Summary'], dict):
        t_depth = _to_datetime64(data['Summary'].get('Time', None))
    if t_depth is None and isinstance(bt, dict) and 'Time' in bt:
        t_depth = _to_datetime64(bt.get('Time'))

    return Z, dtype, t_depth


def _latlon_to_utm(lat, lon, epsg=None, auto_zone=True):
    if not _HAS_PYPROJ:
        raise ImportError("pyproj vereist (pip install pyproj)")
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    if epsg is None and auto_zone:
        zones = np.floor((lon + 180)/6).astype(int) + 1
        zone = int(np.bincount(zones[np.isfinite(zones)]).argmax()) if np.any(np.isfinite(zones)) else 31
        epsg = 32600 + zone
    elif epsg is None:
        epsg = 32631
    crs_from = CRS.from_epsg(4326); crs_to = CRS.from_epsg(int(epsg))
    x, y = Transformer.from_crs(crs_from, crs_to, always_xy=True).transform(lon, lat)
    return np.asarray(x), np.asarray(y), int(epsg)


def extract_dgps_xy(data, epsg_pref=32631, allow_auto_zone=True):
    """Zoek UTM (E,N) of Lat/Lon + tijd in Navigation/GPS/GNSS/NMEA/Summary."""
    containers = []
    for key in ['Navigation', 'GPS', 'GNSS', 'NMEA', 'Gps', 'gps', 'navigation', 'Summary']:
        if key in data and isinstance(data[key], dict):
            containers.append(data[key])

    x_keys = ['Easting', 'UTM_E', 'X', 'E', 'East']
    y_keys = ['Northing', 'UTM_N', 'Y', 'N', 'North']
    t_keys = ['Time', 'Timestamp', 'GPSTime', 'DateTime']

    # UTM direct
    for c in containers:
        kx, X = _first_present(c, x_keys)
        ky, Y = _first_present(c, y_keys)
        if X is not None and Y is not None:
            tname, T = _first_present(c, t_keys)
            t_gps = _to_datetime64(T) if T is not None else None
            return _as_array(X).astype('float64'), _as_array(Y).astype('float64'), t_gps, f"DGPS UTM '{kx}/{ky}'"

    # Lat/Lon -> UTM
    lat_keys = ['Latitude', 'Lat', 'LAT']
    lon_keys = ['Longitude', 'Lon', 'LON', 'Long']
    for c in containers:
        kla, LA = _first_present(c, lat_keys)
        klo, LO = _first_present(c, lon_keys)
        if LA is not None and LO is not None:
            tname, T = _first_present(c, t_keys)
            t_gps = _to_datetime64(T) if T is not None else None
            lat = _as_array(LA).astype('float64'); lon = _as_array(LO).astype('float64')
            X, Y, used_epsg = _latlon_to_utm(lat, lon, epsg=pref_epsg_global or epsg_pref, auto_zone=allow_auto_zone)
            return X, Y, t_gps, f"DGPS Lat/Lon→UTM (EPSG:{used_epsg}) '{kla}/{klo}'"

    # Summary.Track
    if 'Summary' in data and isinstance(data['Summary'], dict) and 'Track' in data['Summary']:
        tr = _as_array(data['Summary']['Track'])
        if tr is not None and tr.ndim == 2 and tr.shape[1] >= 2:
            t_gps = _to_datetime64(data['Summary'].get('Time', None))
            return tr[:, 0].astype('float64'), tr[:, 1].astype('float64'), t_gps, "Summary.Track (UTM verondersteld)"

    raise KeyError("Geen DGPS-positievelden gevonden.")


def align_xy_to_depth_time(X, Y, t_gps, t_depth, tol_seconds=2.0):
    """Nearest-neighbor tijdsmatch; valt terug op lengte-match zonder tijd."""
    X = _as_array(X); Y = _as_array(Y)
    if t_gps is not None and t_depth is not None:
        tg = t_gps.astype('datetime64[ns]'); td = t_depth.astype('datetime64[ns]')
        if tg.size == 0 or td.size == 0:
            return None, None
        tg_sec = tg.astype('datetime64[s]').astype('int64')
        td_sec = td.astype('datetime64[s]').astype('int64')
        idx = np.searchsorted(tg_sec, td_sec)
        idx = np.clip(idx, 0, tg_sec.size - 1)
        left = np.maximum(idx - 1, 0)
        choose_left = (np.abs(td_sec - tg_sec[left]) <= np.abs(td_sec - tg_sec[idx]))
        best = np.where(choose_left, left, idx)
        dt_abs = np.abs(td_sec - tg_sec[best]).astype(float)
        ok = dt_abs <= tol_seconds
        Xd = np.full(td_sec.shape, np.nan); Yd = np.full(td_sec.shape, np.nan)
        Xd[ok] = X[best[ok]]; Yd[ok] = Y[best[ok]]
        if np.isfinite(Xd).mean() < 0.5:
            return None, None
        return Xd, Yd

    # Zonder tijden maar gelijke lengte → directe map
    if (t_gps is None or t_gps.size == 0) and (t_depth is None or t_depth.size == 0):
        if X is not None and Y is not None and Z_global_len is not None and X.size == Z_global_len == Y.size:
            return X, Y
    return None, None


# ================ Interpolatie & plotting ================

def _dedup_valid_xy(X, Y, Z):
    """Verwijder NaN en duplicaten in XY (houd gemiddelde Z bij duplicaatclusters)."""
    m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[m], Y[m], Z[m]
    if X.size == 0:
        return X, Y, Z
    pts = np.column_stack((X, Y))
    # rond af op millimeter om bijna-duplicaten te clusteren:
    key = np.round(pts, 3)
    # gebruik unieke rows en gemiddelde Z per key
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Z_mean = np.zeros(idx.size, dtype=float)
    np.add.at(Z_mean, inv, Z)
    counts = np.bincount(inv)
    Z_mean = Z_mean / np.maximum(counts, 1)
    XYu = pts[idx]
    return XYu[:, 0], XYu[:, 1], Z_mean


def _nearly_collinear(X, Y, eps_rel=1e-6):
    """Check of XY bijna eendimensionaal is via covariantiematrix-rang/ratio."""
    if X.size < 3:
        return True
    XY = np.column_stack((X - np.mean(X), Y - np.mean(Y)))
    cov = np.cov(XY.T)
    w, _ = np.linalg.eigh(cov)
    w = np.sort(np.maximum(w, 0))
    if w[-1] == 0:
        return True
    ratio = w[0] / w[-1]
    return ratio < eps_rel  # heel kleine tweede eigenwaarde → quasi-lijn


def _cumulative_distance(X, Y):
    d = np.hypot(np.diff(X), np.diff(Y))
    s = np.concatenate(([0.0], np.cumsum(d)))
    return s


class BathyModel:
    def interpolate(self, X, Y, Z, grid_res=150):
        """
        Probeer eerst 2D grid te maken; zo niet, raise ValueError zodat de caller kan overschakelen naar 1D-profiel.
        Retourneert (Xi, Yi, Zi, is_surface: bool)
        """
        # 0) dedup & valid
        X, Y, Z = _dedup_valid_xy(X, Y, Z)
        if X.size < 10:
            raise ValueError("Te weinig geldige punten na deduplicatie.")

        # 1) check spreiding
        xspan = np.ptp(X); yspan = np.ptp(Y)
        if xspan == 0 and yspan == 0:
            raise ValueError("Geen ruimtelijke spreiding in X en Y.")
        collinear = _nearly_collinear(X, Y)

        if not collinear and (xspan > 0) and (yspan > 0):
            # 2D mogelijk → probeeer cubic/linear, met jitter fallback
            points = np.column_stack((X, Y))
            xi = np.linspace(np.nanmin(X), np.nanmax(X), grid_res)
            yi = np.linspace(np.nanmin(Y), np.nanmax(Y), grid_res)
            Xi, Yi = np.meshgrid(xi, yi)

            # poging 1: cubic
            try:
                Zi = griddata(points, Z, (Xi, Yi), method='cubic')
                # mask buiten convex hull
                hull = Delaunay(points)
                mask = hull.find_simplex(np.column_stack((Xi.ravel(), Yi.ravel()))) >= 0
                Zi.ravel()[~mask] = np.nan
                return Xi, Yi, self._postprocess(Xi, Yi, Zi), True
            except QhullError:
                pass

            # poging 2: linear
            try:
                Zi = griddata(points, Z, (Xi, Yi), method='linear')
                hull = Delaunay(points)
                mask = hull.find_simplex(np.column_stack((Xi.ravel(), Yi.ravel()))) >= 0
                Zi.ravel()[~mask] = np.nan
                return Xi, Yi, self._postprocess(Xi, Yi, Zi), True
            except QhullError:
                pass

            # poging 3: mini-jitter
            jitter = 1e-3 * max(xspan, yspan)
            Pj = points + np.random.uniform(-jitter, jitter, size=points.shape)
            try:
                Zi = griddata(Pj, Z, (Xi, Yi), method='linear')
                hull = Delaunay(Pj)
                mask = hull.find_simplex(np.columnStack((Xi.ravel(), Yi.ravel()))) >= 0
                Zi.ravel()[~mask] = np.nan
                return Xi, Yi, self._postprocess(Xi, Yi, Zi), True
            except Exception:
                # val terug naar 1D
                pass

        # 1D-profiel fallback
        raise ValueError("XY is (bijna) ééndimensionaal – overschakelen naar 1D-profiel.")

    def _postprocess(self, Xi, Yi, Zi, max_slope=0.5, n_iter=3):
        # vul NaN binnen kern en beperk randslope + median
        def nanmean_filter(vals):
            v = vals[~np.isnan(vals)]
            return np.mean(v) if len(v) else np.nan

        for _ in range(2):
            if np.any(np.isnan(Zi)):
                Zi = generic_filter(Zi, nanmean_filter, size=3, mode='nearest')
        Zi = self.grid_postprocess(Xi, Yi, Zi, max_slope=max_slope, n_iter=n_iter)
        return Zi

    def grid_postprocess(self, Xi, Yi, Zi, max_slope=0.5, n_iter=3):
        from numpy import hypot
        Zi = np.array(Zi, dtype=float, copy=True)
        for _ in range(n_iter):
            Zi = median_filter(Zi, size=3)
            rows, cols = Zi.shape
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    zc = Zi[i, j]
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        zn = Zi[i+di, j+dj]
                        dx = hypot(Xi[i, j]-Xi[i+di, j+dj], Yi[i, j]-Yi[i+di, j+dj])
                        if dx > 0 and abs(zc-zn) > max_slope*dx:
                            Zi[i, j] = zn + np.sign(zc-zn)*max_slope*dx
        return Zi

    def plot_surface(self, Xi, Yi, Zi, zexag=1.0, title="Bathymetrisch oppervlak"):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        norm = plt.Normalize(np.nanmin(-Zi), np.nanmax(-Zi))
        cmap = 'turbo' if 'turbo' in plt.colormaps() else 'rainbow'
        colors = plt.cm.get_cmap(cmap)(norm(-Zi))
        ax.plot_surface(Xi, Yi, -Zi, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=True)
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.5)
        cbar.set_label("Diepte onder instrument (m)")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_zlabel("Diepte (m)")
        ax.set_title(title)
        try:
            ax.set_box_aspect([1, 1, zexag])
        except Exception:
            pass
        plt.tight_layout(); plt.show()

    def plot_profile(self, X, Y, Z, zexag=1.0, title="Bathymetrisch profiel (1D)"):
        # s als cumulatieve afstand
        order = np.argsort(_cumulative_distance(X, Y))
        X, Y, Z = X[order], Y[order], Z[order]
        s = _cumulative_distance(X, Y)

        # 1D smoothing
        Zm = median_filter(Z, size=3)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(s, -Z, '.', alpha=0.5, label='punten')
        ax.plot(s, -Zm, '-', lw=1.5, label='mediaan(3)')
        ax.set_xlabel("Afstand langs traject (m)")
        ax.set_ylabel("Diepte (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout(); plt.show()

    def apply_spike_filter(self, Z, thresh=3.0, median_size=3):
        Z_med = median_filter(Z, size=median_size, mode='nearest')
        Zf = Z.copy()
        mask = np.abs(zscore(Zf, nan_policy='omit')) > thresh
        Zf[mask] = Z_med[mask]
        return Zf, int(mask.sum())

    def build_mesh(self, Xi, Yi, Zi, zexag=1.0):
        # maak een simpele triangulatie binnen geldige cellen
        mask = np.isfinite(Zi)
        rows, cols = Zi.shape
        verts = np.vstack((Xi[mask], Yi[mask], -Zi[mask]*zexag)).T
        idx = np.full(Zi.shape, -1, int); idx[mask] = np.arange(mask.sum())
        faces = []
        for i in range(rows-1):
            for j in range(cols-1):
                a, b, c, d = idx[i, j], idx[i, j+1], idx[i+1, j], idx[i+1, j+1]
                if a >= 0 and b >= 0 and c >= 0: faces.append([a, b, c])
                if d >= 0 and b >= 0 and c >= 0: faces.append([c, b, d])
        return verts, np.array(faces)

    def export_to_dae(self, Xi, Yi, Zi, path, zexag=1.0):
        verts, faces = self.build_mesh(Xi, Yi, Zi, zexag=zexag)
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


# ================= GUI =================

Z_global_len = None
pref_epsg_global = None  # GUI leest dit in

class BathyProfileGUI:
    def __init__(self, root):
        root.title("Bathymetrie Viewer (DGPS-robust)")

        self.model = BathyModel()
        self.X_list = []; self.Y_list = []; self.Z_list = []
        self.depth_types = []
        self.surface_ready = False
        self.Xi = self.Yi = self.Zi = None  # grid
        self.flat_X = self.flat_Y = self.flat_Z = None  # 1D fallback data

        frm = tk.Frame(root); frm.pack(padx=10, pady=5)

        tk.Label(frm, text="Positiebron:").grid(row=0, column=0, sticky='e')
        self.pos_src = tk.StringVar(value='DGPS')
        tk.OptionMenu(frm, self.pos_src, 'DGPS', 'Offsets').grid(row=0, column=1, sticky='w')

        tk.Label(frm, text="EPSG (UTM):").grid(row=1, column=0, sticky='e')
        self.epsg_entry = tk.Entry(frm, width=10); self.epsg_entry.insert(0, "32631")
        self.epsg_entry.grid(row=1, column=1, sticky='w')
        self.auto_zone = tk.BooleanVar(value=True)
        tk.Checkbutton(frm, text="Auto UTM zone bij Lat/Lon", var=self.auto_zone).grid(row=1, column=2, sticky='w')

        tk.Label(frm, text="Tijdstolerantie (s):").grid(row=2, column=0, sticky='e')
        self.tol_scale = tk.Scale(frm, from_=0, to=30, resolution=0.5, orient='horizontal', length=200)
        self.tol_scale.set(2.0); self.tol_scale.grid(row=2, column=1, padx=5, pady=2)

        tk.Label(frm, text="Spike drempel (z-score):").grid(row=3, column=0, sticky='e')
        self.spike = tk.Scale(frm, from_=0, to=10, resolution=0.1, orient='horizontal', length=200)
        self.spike.set(3.0); self.spike.grid(row=3, column=1, padx=5, pady=2)

        tk.Label(frm, text="Median filter (punten):").grid(row=4, column=0, sticky='e')
        self.median = tk.Scale(frm, from_=1, to=9, resolution=2, orient='horizontal', length=200)
        self.median.set(3); self.median.grid(row=4, column=1, padx=5, pady=2)

        tk.Label(frm, text="Hoogteoverdrijving (z-as):").grid(row=5, column=0, sticky='e')
        self.zexag = tk.Scale(frm, from_=0.1, to=10, resolution=0.1, orient='horizontal', length=200)
        self.zexag.set(1.0); self.zexag.grid(row=5, column=1, padx=5, pady=2)

        btns = tk.Frame(root); btns.pack(pady=10)
        tk.Button(btns, text="Selecteer .mat & Plot", command=self.load_and_plot).pack(side='left', padx=5)
        tk.Button(btns, text="Filter Spikes", command=self.filter_spikes).pack(side='left', padx=5)
        tk.Button(btns, text="Export DAE", command=self.export_dae).pack(side='left', padx=5)

    def load_and_plot(self):
        global Z_global_len, pref_epsg_global
        files = filedialog.askopenfilenames(filetypes=[("MAT files", "*.mat")])
        if not files:
            return

        try:
            pref_epsg_global = int(self.epsg_entry.get().strip())
        except Exception:
            pref_epsg_global = 32631

        use_dgps = (self.pos_src.get().lower() == 'dgps')
        tol = float(self.tol_scale.get())
        e0 = n0 = h0 = 0.0

        self.X_list.clear(); self.Y_list.clear(); self.Z_list.clear()
        self.depth_types.clear()
        self.surface_ready = False
        self.Xi = self.Yi = self.Zi = None
        self.flat_X = self.flat_Y = self.flat_Z = None

        for f in files:
            try:
                data = loadmat(f)
                Z, dtype_str, t_depth = extract_depth(data)
                Z = Z.astype('float64'); Z_global_len = Z.size

                Xc = Yc = None
                info_src = ""

                if use_dgps:
                    try:
                        Xgps, Ygps, t_gps, info_src = extract_dgps_xy(
                            data, epsg_pref=pref_epsg_global, allow_auto_zone=self.auto_zone.get()
                        )
                        Xd, Yd = align_xy_to_depth_time(Xgps, Ygps, t_gps, t_depth, tol_seconds=tol)
                        if Xd is None or Yd is None:
                            raise ValueError("DGPS tijds-alignatie faalde (onvoldoende overlap of geen tijd).")
                        # verwijder records zonder positie
                        m = np.isfinite(Xd) & np.isfinite(Yd) & np.isfinite(Z)
                        if m.sum() < max(20, int(0.2*Z.size)):
                            raise ValueError("Te weinig geldige DGPS-matches.")
                        Xc, Yc, Z = Xd[m], Yd[m], Z[m]
                    except Exception as ex_dgps:
                        messagebox.showwarning("DGPS niet bruikbaar",
                                               f"{f}:\n{ex_dgps}\nVal terug op manuele offsets.")
                if Xc is None or Yc is None:
                    if 'Summary' in data and isinstance(data['Summary'], dict) and 'Track' in data['Summary']:
                        tr = _as_array(data['Summary']['Track'])
                        if tr is not None and tr.ndim == 2 and tr.shape[1] >= 2:
                            Xt, Yt = tr[:, 0].astype('float64'), tr[:, 1].astype('float64')
                            if Xt.size != Z.size:
                                L = min(Xt.size, Z.size)
                                Xt, Yt, Z = Xt[:L], Yt[:L], Z[:L]
                            e = simpledialog.askfloat("Offset", f"Easting offset voor {f}:", initialvalue=e0)
                            n = simpledialog.askfloat("Offset", f"Northing offset voor {f}:", initialvalue=n0)
                            h = simpledialog.askfloat("Offset", f"Hoogte offset (m) voor {f}:", initialvalue=h0)
                            if None in (e, n, h):
                                continue
                            e0, n0, h0 = e, n, h
                            Xc, Yc = Xt + e, Yt + n
                            Z = Z + h
                            info_src = "Summary.Track + offsets"
                        else:
                            raise KeyError("Geen bruikbaar 'Summary.Track' voor fallback.")
                    else:
                        raise KeyError("Geen DGPS en geen 'Summary.Track' beschikbaar.")

                # Spikefilter
                Zf, _ = self.model.apply_spike_filter(Z, thresh=self.spike.get(), median_size=int(self.median.get()))
                self.X_list.append(Xc); self.Y_list.append(Yc); self.Z_list.append(Zf)
                self.depth_types.append(dtype_str)

                messagebox.showinfo("Positiebron", f"{f}\nBron: {info_src}\n(tol={tol}s)")

            except Exception as ex:
                messagebox.showerror("Fout bij laden", f"{f}:\n{ex}")

        self.update_and_plot()

    def update_and_plot(self):
        if not self.X_list:
            messagebox.showwarning("Let op", "Geen profielen geladen")
            return

        X = np.concatenate(self.X_list)
        Y = np.concatenate(self.Y_list)
        Z = np.concatenate(self.Z_list)

        try:
            Xi, Yi, Zi, _ = self.model.interpolate(X, Y, Z)  # 2D
            self.Xi, self.Yi, self.Zi = Xi, Yi, Zi
            self.surface_ready = True
            dt_unique = set(self.depth_types)
            suffix = f" ({dt_unique.pop()})" if len(dt_unique) == 1 else " (gemengd)"
            self.model.plot_surface(Xi, Yi, Zi, zexag=self.zexag.get(), title=f"Bathymetrisch oppervlak{suffix}")
        except ValueError:
            # 1D-profiel
            self.surface_ready = False
            self.flat_X, self.flat_Y, self.flat_Z = _dedup_valid_xy(X, Y, Z)
            self.model.plot_profile(self.flat_X, self.flat_Y, self.flat_Z,
                                    zexag=self.zexag.get(), title="Bathymetrisch profiel (1D)")

    def filter_spikes(self):
        if not self.Z_list:
            messagebox.showwarning("Let op", "Laad eerst profielen")
            return
        self.Z_list = [self.model.apply_spike_filter(Z, thresh=self.spike.get(),
                                                     median_size=int(self.median.get()))[0]
                       for Z in self.Z_list]
        self.update_and_plot()

    def export_dae(self):
        if not self.surface_ready or self.Xi is None:
            messagebox.showwarning("Let op", "Geen 2D-oppervlak beschikbaar (profielmodus).")
            return
        path = filedialog.asksaveasfilename(defaultextension=".dae", filetypes=[("COLLADA", "*.dae")])
        if not path:
            return
        try:
            self.model.export_to_dae(self.Xi, self.Yi, self.Zi, path, zexag=self.zexag.get())
            messagebox.showinfo("Klaar", f"DAE opgeslagen:\n{path}")
        except Exception as ex:
            messagebox.showerror("Fout bij export", str(ex))


def main():
    root = tk.Tk()
    BathyProfileGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
