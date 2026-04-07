import os
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


# ============================================================
# 1) .mat laden: MATLAB structs -> Python dict (recursief)
# ============================================================
def _todict(matobj):
    if not hasattr(matobj, "_fieldnames"):
        return matobj
    d = {}
    for field in matobj._fieldnames:
        elem = getattr(matobj, field)
        d[field] = _convert_mat_elem(elem)
    return d


def _tolist(ndarray):
    out = []
    for elem in ndarray:
        out.append(_convert_mat_elem(elem))
    return out


def _convert_mat_elem(elem):
    if isinstance(elem, scipy.io.matlab.mat_struct):
        return _todict(elem)
    if isinstance(elem, np.ndarray):
        if elem.dtype == object:
            return _tolist(elem)
        return elem
    return elem


def loadmat_as_dict(filepath):
    """Laad .mat en converteer MATLAB-structen naar geneste dicts."""
    raw = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    out = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        out[k] = _convert_mat_elem(v)
    return out


# ============================================================
# 2) Bathy punten uit M9-mat halen
# ============================================================
def extract_track_xy(data_dict):
    try:
        track = np.asarray(data_dict["Summary"]["Track"], dtype=float)
    except Exception as e:
        raise KeyError("Kan Summary['Track'] niet vinden/lezen.") from e
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError(f"Onverwachte Track-shape: {track.shape}")
    x = track[:, 0].astype(float)
    y = track[:, 1].astype(float)
    return x, y


def extract_depth(data_dict, source="BT"):
    """
    source:
      - 'BT' : BottomTrack['BT_Depth']
      - 'VB' : BottomTrack['VB_Depth'] (vertical beam)
    """
    bt = data_dict.get("BottomTrack", {})
    key = "VB_Depth" if source.upper() == "VB" else "BT_Depth"
    if key not in bt:
        raise KeyError(f"BottomTrack['{key}'] ontbreekt in .mat")
    z = np.asarray(bt[key], dtype=float).squeeze()
    return z


def build_bathy_pointcloud(data_dict, depth_source, max_depth=None):
    """
    Geeft terug:
      - x, y, z (diepte, positief naar beneden)
      - idx (indices t.o.v. originele samples) die bij deze punten horen
      - NS (aantal originele samples)
    """
    x_all, y_all = extract_track_xy(data_dict)
    z_all = extract_depth(data_dict, depth_source)

    NS = len(x_all)
    if z_all.shape[0] != NS:
        raise ValueError(f"Track heeft NS={NS}, diepte-array heeft shape={z_all.shape}")

    idx = np.arange(NS)
    ok = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(z_all)
    ok &= (z_all > 0)  # 0/negatief meestal invalid / geen bodem

    if max_depth is not None and np.isfinite(max_depth) and max_depth > 0:
        ok &= (z_all <= max_depth)

    x = x_all[ok]
    y = y_all[ok]
    z = z_all[ok]
    idx = idx[ok]

    return x, y, z, idx, NS


# ============================================================
# 3) Robuuste spike-filter (met focus op randen/uitbijters)
# ============================================================
def mad(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    med = np.median(a)
    return np.median(np.abs(a - med))


def spike_filter_pointcloud(x, y, z, k=12, z_mad_thresh=6.0, edge_frac=0.03, edge_strict=4.0):
    """
    Automatisch spikes verwijderen via lokale (kNN) mediaan + MAD,
    met strengere drempel op de 'randen' van de track (eerste/laatste edge_frac).

    Retourneert keep_mask (len = N).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    n = z.size
    if n < max(50, k + 5):
        return np.ones(n, dtype=bool)

    pts = np.c_[x, y]
    tree = cKDTree(pts)
    _, idxs = tree.query(pts, k=min(k, n))  # (n,k)

    z_nei = z[idxs]
    z_med = np.median(z_nei, axis=1)
    res = z - z_med

    s = mad(res)
    if not np.isfinite(s) or s == 0:
        return np.ones(n, dtype=bool)

    rz = np.abs(res) / (1.4826 * s)  # robuuste z-score
    keep = rz <= z_mad_thresh

    # strengere filter op begin/einde van de track (typisch randspikes)
    ds = np.hypot(np.diff(x), np.diff(y))
    sdist = np.r_[0.0, np.cumsum(ds)]
    order = np.argsort(sdist)
    n_edge = int(np.ceil(edge_frac * n))
    if n_edge >= 5:
        edge_idx = np.r_[order[:n_edge], order[-n_edge:]]
        keep[edge_idx] = keep[edge_idx] & (rz[edge_idx] <= edge_strict)

    return keep


# ============================================================
# 4) Interpolatie naar raster (3D bathymetrie)
# ============================================================
def default_grid_res(x, y):
    pts = np.c_[x, y]
    if pts.shape[0] < 10:
        return 1.0
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        return 1.0
    return max(0.2, float(np.median(nn)) * 0.5)


def grid_bathy(x, y, z, grid_res=None, method="linear"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    if grid_res is None or not np.isfinite(grid_res) or grid_res <= 0:
        grid_res = default_grid_res(x, y)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    pad_x = 0.01 * max(1e-9, xmax - xmin)
    pad_y = 0.01 * max(1e-9, ymax - ymin)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    gx = np.arange(xmin, xmax + grid_res, grid_res)
    gy = np.arange(ymin, ymax + grid_res, grid_res)
    Xg, Yg = np.meshgrid(gx, gy)

    Zi = griddata((x, y), z, (Xg, Yg), method=method)

    # vul gaten met nearest (alleen waar NaN)
    if np.isnan(Zi).any():
        Zin = griddata((x, y), z, (Xg, Yg), method="nearest")
        Zi = np.where(np.isnan(Zi), Zin, Zi)

    return Xg, Yg, Zi, grid_res


# ============================================================
# 5) Filter volledige .mat structuur op sample-mask (NS)
# ============================================================
def filter_structure_by_samples(obj, sample_mask, NS):
    """
    Recursief slicen van arrays/lijsten die een sample-dimensie NS bevatten.
    Heuristiek:
      - 1D: shape (NS,)  -> slice
      - 2D: if axis0==NS slice axis0; elif axis1==NS slice axis1
      - >=3D: prefer last axis==NS slice last axis; else axis0==NS slice axis0
    """
    if isinstance(obj, dict):
        return {k: filter_structure_by_samples(v, sample_mask, NS) for k, v in obj.items()}

    if isinstance(obj, list):
        if len(obj) == NS:
            return [obj[i] for i, ok in enumerate(sample_mask) if ok]
        return [filter_structure_by_samples(v, sample_mask, NS) for v in obj]

    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.dtype == object:
            return filter_structure_by_samples(arr.tolist(), sample_mask, NS)

        if arr.ndim == 1 and arr.shape[0] == NS:
            return arr[sample_mask]

        if arr.ndim == 2:
            if arr.shape[0] == NS:
                return arr[sample_mask, :]
            if arr.shape[1] == NS:
                return arr[:, sample_mask]
            return arr

        if arr.ndim >= 3:
            if arr.shape[-1] == NS:
                sl = [slice(None)] * arr.ndim
                sl[-1] = sample_mask
                return arr[tuple(sl)]
            if arr.shape[0] == NS:
                sl = [sample_mask] + [slice(None)] * (arr.ndim - 1)
                return arr[tuple(sl)]
            return arr

    return obj


# ============================================================
# 6) Interactieve viewer (3D + selectie in planview)
# ============================================================
class BathyViewer:
    def __init__(self, parent_tk, data_dict, mat_path):
        self.parent_tk = parent_tk
        self.data_dict = data_dict
        self.mat_path = mat_path

        self.depth_source = "BT"
        self.max_depth = None
        self.grid_res = None
        self.do_spike_filter = True

        self.z_exag = 1.0  # alleen visualisatie, data blijft identiek

        self.NS = None
        self.idx_current = None  # indices t.o.v. origineel (0..NS-1) die behouden zijn
        self.x = self.y = self.z = None

        self.fig = None
        self.ax3d = None
        self.ax2d = None
        self.surf = None
        self.scat3d = None
        self.scat2d = None
        self.poly = None
        self.poly_selector = None
        self.poly_verts = None

        self._lims3d = None

    def set_params(self, depth_source, max_depth, grid_res, do_spike_filter, z_exag):
        self.depth_source = depth_source
        self.max_depth = max_depth
        self.grid_res = grid_res
        self.do_spike_filter = do_spike_filter
        self.z_exag = float(z_exag) if z_exag is not None else 1.0

    def _recompute_points(self):
        x, y, z, idx, NS = build_bathy_pointcloud(
            self.data_dict, self.depth_source, max_depth=self.max_depth
        )

        if self.do_spike_filter:
            keep = spike_filter_pointcloud(x, y, z)
            x, y, z, idx = x[keep], y[keep], z[keep], idx[keep]

        self.x, self.y, self.z = x, y, z
        self.idx_current = idx
        self.NS = NS

    def _draw(self):
        self.fig = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title("3D Bathymetrie + selectie (s) en proces (p)")

        gs = self.fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
        self.ax3d = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax2d = self.fig.add_subplot(gs[0, 1])

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update_plot(first=True)

        self.ax2d.set_title("Planview selectie (XY)\nDruk 's' om polygon te tekenen")
        self.ax2d.set_xlabel("X")
        self.ax2d.set_ylabel("Y")
        self.ax2d.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()

    def _update_plot(self, first=False):
        Xg, Yg, Zg, _ = grid_bathy(self.x, self.y, self.z, grid_res=self.grid_res, method="linear")

        # 3D: verwijder oude artists
        if self.surf is not None:
            self.surf.remove()
            self.surf = None
        if self.scat3d is not None:
            self.scat3d.remove()
            self.scat3d = None

        # visualisatie: diepte negatief, met z-overdrijving
        Zg_vis = -Zg * self.z_exag
        z_vis = -self.z * self.z_exag

        self.surf = self.ax3d.plot_surface(
            Xg, Yg, Zg_vis, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.8
        )
        self.scat3d = self.ax3d.scatter(self.x, self.y, z_vis, s=3)

        self.ax3d.set_title(
            "3D bathymetrie (interpolatie) + puntenwolk\n's' selecteren, 'p' verwerken (punten verwijderen)"
        )
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Diepte (m) [visueel]")

        # 2D planview
        if self.scat2d is not None:
            self.scat2d.remove()
            self.scat2d = None
        self.scat2d = self.ax2d.scatter(self.x, self.y, s=6, c=self.z)

        if first:
            # limieten vastzetten zodat updates in hetzelfde kader blijven
            self._lims3d = (self.ax3d.get_xlim(), self.ax3d.get_ylim(), self.ax3d.get_zlim())
        else:
            if self._lims3d is not None:
                self.ax3d.set_xlim(self._lims3d[0])
                self.ax3d.set_ylim(self._lims3d[1])
                self.ax3d.set_zlim(self._lims3d[2])

        # polygon overlay
        if self.poly is not None:
            self.poly.remove()
            self.poly = None
        if self.poly_verts is not None and len(self.poly_verts) >= 3:
            vx = [p[0] for p in self.poly_verts] + [self.poly_verts[0][0]]
            vy = [p[1] for p in self.poly_verts] + [self.poly_verts[0][1]]
            self.poly, = self.ax2d.plot(vx, vy, linewidth=2)

        self.fig.canvas.draw_idle()

    def _on_select_poly(self, verts):
        self.poly_verts = verts
        self._update_plot(first=False)

    def _enable_polygon_selector(self):
        if self.poly_selector is not None:
            self.poly_selector.disconnect_events()
            self.poly_selector = None

        self.poly_selector = PolygonSelector(
            self.ax2d,
            onselect=self._on_select_poly,
            useblit=True,
            props=dict(linewidth=2),
        )
        self.ax2d.set_title("Teken polygon (klikpunten, sluit met dubbelklik).\nDaarna druk 'p' om te verwerken.")
        self.fig.canvas.draw_idle()

    def _process_polygon(self):
        if self.poly_verts is None or len(self.poly_verts) < 3:
            messagebox.showinfo("Geen selectie", "Geen polygon geselecteerd. Druk eerst 's' en teken een polygon.")
            return

        path = Path(self.poly_verts)
        inside = path.contains_points(np.c_[self.x, self.y])
        keep = ~inside  # punten binnen polygon verwijderen

        if keep.sum() < 10:
            messagebox.showwarning("Te weinig punten", "Selectie verwijdert bijna alles. Teken een kleinere selectie.")
            return

        self.x, self.y, self.z = self.x[keep], self.y[keep], self.z[keep]
        self.idx_current = self.idx_current[keep]

        # optioneel: na selectie mild opnieuw spikes weg
        if self.do_spike_filter and self.x.size > 30:
            keep2 = spike_filter_pointcloud(self.x, self.y, self.z, z_mad_thresh=7.0, edge_strict=5.0)
            self.x, self.y, self.z = self.x[keep2], self.y[keep2], self.z[keep2]
            self.idx_current = self.idx_current[keep2]

        self._update_plot(first=False)

    def _on_key(self, event):
        if event.key == "s":
            self._enable_polygon_selector()
        elif event.key == "p":
            self._process_polygon()

    def show(self):
        self._recompute_points()
        if self.x.size < 10:
            messagebox.showerror("Onvoldoende data", "Na filtering blijven er te weinig punten over om te interpoleren.")
            return
        self._draw()

    def export_filtered_mat(self, out_path):
        if self.NS is None or self.idx_current is None:
            raise RuntimeError("Geen data geladen/gefiterd om te exporteren.")

        sample_mask = np.zeros(self.NS, dtype=bool)
        sample_mask[self.idx_current] = True

        filtered = {k: filter_structure_by_samples(v, sample_mask, self.NS) for k, v in self.data_dict.items()}
        scipy.io.savemat(out_path, filtered)
        return out_path

    def export_bathy_xyz_csv(self, out_path):
        """
        Exporteer XYZ (geïnterpoleerde bathymetrie-grid) als CSV.
        Kolommen: x, y, z  (z = diepte, positief naar beneden)
        """
        if self.x is None or self.y is None or self.z is None or self.x.size < 3:
            self._recompute_points()

        Xg, Yg, Zg, _ = grid_bathy(self.x, self.y, self.z, grid_res=self.grid_res, method="linear")

        xf = Xg.ravel()
        yf = Yg.ravel()
        zf = Zg.ravel()

        ok = np.isfinite(xf) & np.isfinite(yf) & np.isfinite(zf)
        xf, yf, zf = xf[ok], yf[ok], zf[ok]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["x", "y", "z"])
            w.writerows(zip(xf.tolist(), yf.tolist(), zf.tolist()))

        return out_path


# ============================================================
# 7) Tkinter GUI
# ============================================================
class BathyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("M9 .mat -> 3D bathymetrie + puntenwolk + selectie")

        self.mat_path = None
        self.data_dict = None
        self.viewer = None

        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Button(frm, text="Selecteer .mat bestand", command=self.select_mat).grid(row=0, column=0, sticky="w")
        self.lbl_file = ttk.Label(frm, text="(geen bestand geselecteerd)", width=70)
        self.lbl_file.grid(row=0, column=1, columnspan=3, sticky="w", padx=8)

        ttk.Label(frm, text="Dieptebron:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.cmb_depth = ttk.Combobox(
            frm,
            values=["Bottom track (BT_Depth)", "Vertical beam (VB_Depth)"],
            state="readonly",
            width=28,
        )
        self.cmb_depth.current(0)
        self.cmb_depth.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frm, text="Max. diepte onder boot (m):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.ent_maxd = ttk.Entry(frm, width=12)
        self.ent_maxd.insert(0, "15")
        self.ent_maxd.grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(frm, text="Grid resolutie (m, leeg=auto):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.ent_grid = ttk.Entry(frm, width=12)
        self.ent_grid.insert(0, "")
        self.ent_grid.grid(row=3, column=1, sticky="w", pady=(6, 0))

        ttk.Label(frm, text="Hoogte-overdrijving Z (alleen visualisatie):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.ent_zex = ttk.Entry(frm, width=12)
        self.ent_zex.insert(0, "1.0")
        self.ent_zex.grid(row=4, column=1, sticky="w", pady=(6, 0))

        self.var_spike = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Auto spike-filter (incl. randen)", variable=self.var_spike).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        ttk.Button(frm, text="Plot 3D + puntenwolk (matplotlib)", command=self.plot).grid(
            row=6, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Button(frm, text="Exporteer gefilterde .mat", command=self.export_mat).grid(
            row=6, column=1, sticky="w", pady=(10, 0)
        )
        ttk.Button(frm, text="Exporteer bathy XYZ CSV", command=self.export_xyz_csv).grid(
            row=6, column=2, sticky="w", pady=(10, 0)
        )

        help_txt = (
            "Interactief in figuur:\n"
            "  - Druk 's' en teken een polygon in het planview-paneel (rechts)\n"
            "  - Druk 'p' om punten binnen de polygon te verwijderen en de 3D-figuur te updaten\n"
        )
        ttk.Label(frm, text=help_txt, justify="left").grid(row=7, column=0, columnspan=4, sticky="w", pady=(10, 0))

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frm.columnconfigure(3, weight=1)

    def select_mat(self):
        path = filedialog.askopenfilename(
            title="Selecteer een M9 .mat bestand",
            filetypes=[("MAT files", "*.mat")],
        )
        if not path:
            return
        self.mat_path = path
        self.lbl_file.config(text=os.path.basename(path))
        try:
            self.data_dict = loadmat_as_dict(path)
            self.viewer = BathyViewer(self.root, self.data_dict, path)
        except Exception as e:
            messagebox.showerror("Fout bij laden", str(e))
            self.data_dict = None
            self.viewer = None

    def _get_params(self):
        depth_source = "VB" if self.cmb_depth.get().startswith("Vertical") else "BT"

        maxd_txt = self.ent_maxd.get().strip()
        max_depth = None
        if maxd_txt:
            try:
                max_depth = float(maxd_txt)
            except ValueError:
                raise ValueError("Max. diepte is geen geldig getal.")

        grid_txt = self.ent_grid.get().strip()
        grid_res = None
        if grid_txt:
            try:
                grid_res = float(grid_txt)
            except ValueError:
                raise ValueError("Grid resolutie is geen geldig getal.")

        zex_txt = self.ent_zex.get().strip()
        z_exag = 1.0
        if zex_txt:
            try:
                z_exag = float(zex_txt)
            except ValueError:
                raise ValueError("Hoogte-overdrijving (Z) is geen geldig getal.")
        if not np.isfinite(z_exag) or z_exag <= 0:
            raise ValueError("Hoogte-overdrijving (Z) moet > 0 zijn.")

        do_spike = bool(self.var_spike.get())
        return depth_source, max_depth, grid_res, do_spike, z_exag

    def plot(self):
        if self.viewer is None:
            messagebox.showinfo("Geen bestand", "Selecteer eerst een .mat bestand.")
            return
        try:
            depth_source, max_depth, grid_res, do_spike, z_exag = self._get_params()
            self.viewer.set_params(depth_source, max_depth, grid_res, do_spike, z_exag)
            self.viewer.show()
        except Exception as e:
            messagebox.showerror("Plot-fout", str(e))

    def export_mat(self):
        if self.viewer is None:
            messagebox.showinfo("Geen bestand", "Selecteer eerst een .mat bestand.")
            return
        try:
            depth_source, max_depth, grid_res, do_spike, z_exag = self._get_params()
            self.viewer.set_params(depth_source, max_depth, grid_res, do_spike, z_exag)

            out_path = filedialog.asksaveasfilename(
                title="Sla gefilterde .mat op",
                defaultextension=".mat",
                filetypes=[("MAT files", "*.mat")],
            )
            if not out_path:
                return
            saved = self.viewer.export_filtered_mat(out_path)
            messagebox.showinfo("Export klaar", f"Opgeslagen:\n{saved}")
        except Exception as e:
            messagebox.showerror("Export-fout", str(e))

    def export_xyz_csv(self):
        if self.viewer is None:
            messagebox.showinfo("Geen bestand", "Selecteer eerst een .mat bestand.")
            return
        try:
            depth_source, max_depth, grid_res, do_spike, z_exag = self._get_params()
            self.viewer.set_params(depth_source, max_depth, grid_res, do_spike, z_exag)

            out_path = filedialog.asksaveasfilename(
                title="Sla bathymetrie XYZ CSV op",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
            )
            if not out_path:
                return
            saved = self.viewer.export_bathy_xyz_csv(out_path)
            messagebox.showinfo("Export klaar", f"Opgeslagen:\n{saved}")
        except Exception as e:
            messagebox.showerror("Export-fout", str(e))


def main():
    root = tk.Tk()
    _ = BathyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
