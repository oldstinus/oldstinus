#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bathy Profiel (1D) met Lasso-filter
- Eénmalige keuze: VB_Depth of BT_Depth
- Positie: DGPS (UTM of Lat/Lon->UTM) of georef Track met UTM-anker
- Interactief outliers weggooien met Lasso (Delete/Undo)
- Opslaan: CSV (s,E,N,Z) en mask (npz)
"""

import os
import numpy as np
import scipy.io
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

# --- optioneel: pyproj voor Lat/Lon -> UTM
try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

# ======== .mat helpers ========

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
            rk = low[k.lower()]
            return rk, dct[rk]
    return None, None

def _matlab_datenum_to_datetime64(dnum):
    base = np.datetime64('1970-01-01')
    sec = (np.array(dnum, dtype=float) - 719529.0) * 86400.0
    return base + sec.astype('timedelta64[s]')

def _to_datetime64(t):
    a = _as_array(t)
    if a is None: return None
    if np.issubdtype(np.array(a).dtype, np.number):
        mx = float(np.nanmax(a)); mn = float(np.nanmin(a))
        if 1e5 < mn < 1e7:  # MATLAB datenum
            return _matlab_datenum_to_datetime64(a)
        if mx > 1e10:  # ms
            return np.datetime64('1970-01-01') + a.astype('timedelta64[ms]')
        return np.datetime64('1970-01-01') + a.astype('timedelta64[s]')
    return a.astype('datetime64[ns]')

# ======== diepte & positie ========

_DEPTH_CHOICE = None  # 'vb' of 'bt'

def extract_depth(data):
    """Éénmalig VB_Depth of BT_Depth kiezen; geef Z (m) en t (optioneel)."""
    global _DEPTH_CHOICE
    if 'BottomTrack' not in data:
        raise KeyError("BottomTrack ontbreekt in .mat")

    bt = data['BottomTrack']
    cand_vb = ['VB_Depth', 'VBdepth', 'VB_Depths']
    cand_bt = ['BT_Depth', 'BTdepth', 'BT_Depths']

    if isinstance(bt, dict):
        _, vb = _first_present(bt, cand_vb)
        _, bz = _first_present(bt, cand_bt)
    else:
        vb = None; bz = None
        for k in cand_vb:
            if hasattr(bt, k): vb = getattr(bt, k); break
        for k in cand_bt:
            if hasattr(bt, k): bz = getattr(bt, k); break

    if vb is None and bz is None:
        raise KeyError("Geen VB_Depth of BT_Depth gevonden.")

    if _DEPTH_CHOICE is None:
        if vb is not None and bz is not None:
            choice = simpledialog.askstring("Dieptekeuze (éénmalig)",
                                            "Typ 'vb' voor VB_Depth of 'bt' voor BT_Depth:")
            use_vb = bool(choice and choice.strip().lower().startswith('vb'))
            _DEPTH_CHOICE = 'vb' if use_vb else 'bt'
        else:
            _DEPTH_CHOICE = 'vb' if vb is not None else 'bt'

    Z = _as_array(vb if _DEPTH_CHOICE == 'vb' else bz).astype(float)
    # tijd (niet strikt nodig voor 1D, maar handig voor toekomst)
    t_depth = None
    if 'Summary' in data and isinstance(data['Summary'], dict):
        t_depth = _to_datetime64(data['Summary'].get('Time', None))
    if t_depth is None and isinstance(bt, dict) and 'Time' in bt:
        t_depth = _to_datetime64(bt.get('Time'))
    return Z, t_depth

def _latlon_to_utm(lat, lon, epsg=None, auto_zone=True):
    if not _HAS_PYPROJ:
        raise ImportError("pyproj is nodig voor lat/lon -> UTM (pip install pyproj)")
    lat = np.asarray(lat, float).ravel()
    lon = np.asarray(lon, float).ravel()
    if epsg is None and auto_zone:
        zones = np.floor((lon + 180)/6).astype(int) + 1
        zone = int(np.bincount(zones[np.isfinite(zones)]).argmax()) if np.any(np.isfinite(zones)) else 31
        epsg = 32600 + zone
    elif epsg is None:
        epsg = 32631
    xform = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(int(epsg)), always_xy=True)
    x, y = xform.transform(lon, lat)
    return np.asarray(x), np.asarray(y), int(epsg)

def extract_xy(data, epsg_pref=32631, allow_auto_zone=True):
    """Probeer DGPS (UTM of Lat/Lon). Zoniet: georef Track met UTM-anker."""
    # 1) directe UTM
    for key in ['Navigation','GPS','GNSS','NMEA','Summary','gps','navigation']:
        c = data.get(key) if isinstance(data, dict) else None
        if not isinstance(c, dict): continue
        kx, X = _first_present(c, ['Easting','UTM_E','X','E','East'])
        ky, Y = _first_present(c, ['Northing','UTM_N','Y','N','North'])
        if X is not None and Y is not None:
            X = _as_array(X).astype(float); Y = _as_array(Y).astype(float)
            if np.nanstd(X) > 1e-9 or np.nanstd(Y) > 1e-9:
                return X, Y, "DGPS UTM"
    # 2) lat/lon -> UTM
    for key in ['Navigation','GPS','GNSS','NMEA','Summary','gps','navigation']:
        c = data.get(key) if isinstance(data, dict) else None
        if not isinstance(c, dict): continue
        _, LA = _first_present(c, ['Latitude','Lat','LAT'])
        _, LO = _first_present(c, ['Longitude','Lon','LON','Long'])
        if LA is not None and LO is not None:
            lat = _as_array(LA).astype(float); lon = _as_array(LO).astype(float)
            if np.nanstd(lat) > 1e-12 or np.nanstd(lon) > 1e-12:
                X, Y, used = _latlon_to_utm(lat, lon, epsg=epsg_pref, auto_zone=allow_auto_zone)
                return X, Y, f"DGPS Lat/Lon→UTM (EPSG:{used})"
    # 3) Summary.Track + anker
    if 'Summary' in data and isinstance(data['Summary'], dict) and 'Track' in data['Summary']:
        tr = _as_array(data['Summary']['Track'])
        if isinstance(tr, np.ndarray) and tr.ndim == 2 and tr.shape[1] >= 2:
            e0 = simpledialog.askfloat("UTM anker (Easting)", "Geef UTM Easting (m) voor startpunt:", initialvalue=650000.0)
            n0 = simpledialog.askfloat("UTM anker (Northing)", "Geef UTM Northing (m) voor startpunt:", initialvalue=5650000.0)
            if e0 is None or n0 is None:
                raise ValueError("Georeferentie geannuleerd.")
            X = tr[:,0].astype(float) + float(e0)
            Y = tr[:,1].astype(float) + float(n0)
            return X, Y, "Summary.Track + UTM anker"
    raise KeyError("Geen DGPS of Track gevonden.")

# ======== 1D-profiel (s) ========

def cumulative_distance(X, Y):
    d = np.hypot(np.diff(X), np.diff(Y))
    return np.concatenate(([0.0], np.cumsum(d)))

# ======== Interactieve viewer ========

class ProfileLassoApp:
    def __init__(self, master):
        self.master = master
        master.title("Bathy 1D Profiel (Lasso outlier filter)")

        # UI
        frm = tk.Frame(master); frm.pack(padx=8, pady=6)
        tk.Label(frm, text="EPSG (UTM):").grid(row=0, column=0, sticky='e')
        self.epsg = tk.Entry(frm, width=8); self.epsg.insert(0, "32631")
        self.epsg.grid(row=0, column=1, sticky='w')
        self.auto_zone = tk.BooleanVar(value=True)
        tk.Checkbutton(frm, text="Auto zone bij Lat/Lon", var=self.auto_zone).grid(row=0, column=2, sticky='w')

        tk.Button(frm, text="Kies .mat & Plot", command=self.load_and_plot).grid(row=1, column=0, pady=4, sticky='w')
        tk.Button(frm, text="Opslaan CSV (s,E,N,Z)", command=self.save_csv).grid(row=1, column=1, pady=4, sticky='w')
        tk.Button(frm, text="Opslaan MASK (npz)", command=self.save_mask).grid(row=1, column=2, pady=4, sticky='w')

        # data containers
        self.S = None; self.Z = None
        self.X = None; self.Y = None
        self.keep = None            # boolean mask
        self.undo_stack = []        # list of masks to undo

        # figure
        self.fig, self.ax = plt.subplots(figsize=(10,5))
        self.scat = None
        self.lasso = None
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        tk.Label(master, text="Gebruik Lasso (muisklik, 2D polygon) -> D=delete, U=undo, R=reset, S=save CSV, M=save mask, Q=quit").pack(pady=(0,6))

    # ---------- loading ----------

    def load_and_plot(self):
        files = filedialog.askopenfilenames(filetypes=[("MAT files","*.mat")])
        if not files: return
        try:
            epsg_pref = int(self.epsg.get().strip())
        except Exception:
            epsg_pref = 32631

        global _DEPTH_CHOICE
        _DEPTH_CHOICE = None  # opnieuw per sessie

        X_all, Y_all, Z_all = [], [], []

        for f in files:
            try:
                data = loadmat(f)
                Z, tZ = extract_depth(data)
                X, Y, src = extract_xy(data, epsg_pref=epsg_pref, allow_auto_zone=self.auto_zone.get())

                # lengte-match als nodig
                L = min(Z.size, X.size, Y.size)
                X, Y, Z = X[:L], Y[:L], Z[:L]

                X_all.append(X); Y_all.append(Y); Z_all.append(Z)
            except Exception as ex:
                messagebox.showwarning("File overgeslagen", f"{os.path.basename(f)}:\n{ex}")

        if not Z_all:
            messagebox.showerror("Geen data", "Geen bruikbare bestanden.")
            return

        X = np.concatenate(X_all); Y = np.concatenate(Y_all); Z = np.concatenate(Z_all)
        S = cumulative_distance(X, Y)

        self.X, self.Y, self.Z = X, Y, Z
        self.S = S
        self.keep = np.isfinite(S) & np.isfinite(Z)

        self.undo_stack.clear()
        self.redraw()

    # ---------- plotting & lasso ----------

    def redraw(self):
        self.ax.clear()
        s = self.S[self.keep]; z = self.Z[self.keep]
        self.scat = self.ax.scatter(s, -z, s=10, alpha=0.7)  # -Z: diepte naar beneden
        self.ax.set_xlabel("Afstand langs traject (m)")
        self.ax.set_ylabel("Diepte (m)")
        self.ax.set_title(f"Profiel (N={z.size} punten) — Lasso selecteren, D=delete, U=undo")
        self.ax.grid(True, alpha=0.3)

        # (her)start lasso
        if self.lasso is not None:
            self.lasso.disconnect_events()
        self.lasso = LassoSelector(self.ax, onselect=self.on_lasso)

        self.fig.canvas.draw_idle()
        plt.show(block=False)

    def on_lasso(self, verts):
        if self.S is None: return
        path = Path(verts)
        pts = np.column_stack((self.S[self.keep], -self.Z[self.keep]))
        selected = path.contains_points(pts)
        # tijdelijke markering
        if selected.any():
            self.ax.scatter(pts[selected,0], pts[selected,1], s=20, facecolors='none', edgecolors='r')
            self.fig.canvas.draw_idle()
            # bewaar selectie indices in keep-space
            self._last_selection = np.where(self.keep)[0][selected]
        else:
            self._last_selection = np.array([], dtype=int)

    # ---------- key bindings ----------

    def on_key(self, event):
        if event.key is None: return
        k = event.key.lower()
        if k == 'd':
            self.delete_selection()
        elif k == 'u':
            self.undo()
        elif k == 'r':
            self.reset_keep()
        elif k == 's':
            self.save_csv()
        elif k == 'm':
            self.save_mask()
        elif k == 'q':
            plt.close(self.fig)

    def delete_selection(self):
        if getattr(self, '_last_selection', None) is None or self._last_selection.size == 0:
            return
        # push undo
        self.undo_stack.append(self.keep.copy())
        # drop punten
        self.keep[self._last_selection] = False
        self._last_selection = np.array([], dtype=int)
        self.redraw()

    def undo(self):
        if not self.undo_stack:
            return
        self.keep = self.undo_stack.pop()
        self.redraw()

    def reset_keep(self):
        if self.S is None: return
        self.keep[:] = True
        self.undo_stack.clear()
        self.redraw()

    # ---------- save ----------

    def save_csv(self):
        if self.S is None or self.keep is None:
            messagebox.showwarning("Geen data", "Laad eerst data.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")],
                                            initialfile="bathy_profile_cleaned.csv")
        if not path: return
        s = self.S[self.keep]; x = self.X[self.keep]; y = self.Y[self.keep]; z = self.Z[self.keep]
        arr = np.column_stack((s, x, y, z))
        np.savetxt(path, arr, delimiter=",", header="s_m,E_m,N_m,Z_m", comments="", fmt="%.6f")
        messagebox.showinfo("Opgeslagen", f"CSV weggeschreven:\n{path}")

    def save_mask(self):
        if self.keep is None:
            messagebox.showwarning("Geen data", "Laad eerst data.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz",
                                            filetypes=[("NumPy NPZ","*.npz")],
                                            initialfile="bathy_profile_mask.npz")
        if not path: return
        np.savez(path, keep=self.keep)
        messagebox.showinfo("Opgeslagen", f"Mask weggeschreven:\n{path}")

# ======== main ========

def main():
    root = tk.Tk()
    app = ProfileLassoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
