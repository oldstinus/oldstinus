#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-mat 3D Bathymetrie Viewer met geavanceerde spikefilter
- Meerdere .mat-bestanden met individuele UTM- en hoogte-offsets
- Multibeam-kleuren, median/z-score filtering, randspikefix
"""

import numpy as np
import scipy.io
from scipy.interpolate import griddata
from scipy.stats import zscore
from scipy.ndimage import median_filter
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Transformer
import datetime as dt

# ------- .mat helpers -------
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

def loadmat(filepath):
    mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    return _check_keys(mat)

def extract_bottom_profile(data):
    if 'Summary' not in data or 'BottomTrack' not in data:
        raise KeyError("Structuren 'Summary' of 'BottomTrack' ontbreken.")
    track = np.array(data['Summary']['Track'])
    depth = np.array(data['BottomTrack']['BT_Depth']).squeeze()
    return track[:, 0], track[:, 1], depth

def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    θ = np.deg2rad(angle_deg)
    c, s = np.cos(θ), np.sin(θ)
    ax = axis.lower()
    if ax == 'x':
        return np.array([[1, 0,  0],[0, c, -s],[0, s,  c]])
    elif ax == 'y':
        return np.array([[ c, 0, s],[ 0, 1, 0],[-s, 0, c]])
    elif ax == 'z':
        return np.array([[c, -s, 0],[s,  c, 0],[0,  0, 1]])
    else:
        raise ValueError("Axis moet 'x', 'y' of 'z' zijn.")

def enforce_edge_slope_limit(Xi, Yi, Zi, max_slope=0.5):
    Xi, Yi, Zi = Xi.copy(), Yi.copy(), Zi.copy()
    n_rows, n_cols = Zi.shape
    for j in range(1, n_cols):
        dz = Zi[0, j] - Zi[0, j-1]
        dx = np.hypot(Xi[0, j] - Xi[0, j-1], Yi[0, j] - Yi[0, j-1])
        max_dz = max_slope * dx
        if abs(dz) > max_dz:
            Zi[0, j] = Zi[0, j-1] + np.sign(dz) * max_dz
    for j in range(1, n_cols):
        dz = Zi[-1, j] - Zi[-1, j-1]
        dx = np.hypot(Xi[-1, j] - Xi[-1, j-1], Yi[-1, j] - Yi[-1, j-1])
        max_dz = max_slope * dx
        if abs(dz) > max_dz:
            Zi[-1, j] = Zi[-1, j-1] + np.sign(dz) * max_dz
    for i in range(1, n_rows):
        dz = Zi[i, 0] - Zi[i-1, 0]
        dx = np.hypot(Xi[i, 0] - Xi[i-1, 0], Yi[i, 0] - Yi[i-1, 0])
        max_dz = max_slope * dx
        if abs(dz) > max_dz:
            Zi[i, 0] = Zi[i-1, 0] + np.sign(dz) * max_dz
    for i in range(1, n_rows):
        dz = Zi[i, -1] - Zi[i-1, -1]
        dx = np.hypot(Xi[i, -1] - Xi[i-1, -1], Yi[i, -1] - Yi[i-1, -1])
        max_dz = max_slope * dx
        if abs(dz) > max_dz:
            Zi[i, -1] = Zi[i-1, -1] + np.sign(dz) * max_dz
    return Xi, Yi, Zi

class BodemModel:
    def __init__(self):
        self.transformer = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    def interpolate(self, X, Y, Z, grid_res=150):
        xi = np.linspace(np.nanmin(X), np.nanmax(X), grid_res)
        yi = np.linspace(np.nanmin(Y), np.nanmax(Y), grid_res)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic')
        Xi, Yi, Zi = enforce_edge_slope_limit(Xi, Yi, Zi, max_slope=0.5)
        return Xi, Yi, Zi
    def plot_surface(self, Xi, Yi, Zi, title="3D Bathymetrie"):
        Zi = median_filter(Zi, size=3)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            Xi, Yi, -Zi,
            rstride=1, cstride=1, linewidth=0, antialiased=True,
            cmap='turbo' if 'turbo' in plt.colormaps() else 'rainbow'
        )
        cb = fig.colorbar(surf, ax=ax, shrink=0.5)
        cb.set_label("Diepte (m)")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_zlabel("Diepte (m)")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    def apply_spike_filter(self, Z, thresh=3.0, median_size=3):
        Z_med = median_filter(Z, size=median_size, mode='nearest')
        Zf = Z.copy()
        z = zscore(Zf, nan_policy='omit')
        mask = np.abs(z) > thresh
        if np.any(mask):
            Zf[mask] = Z_med[mask]
        return Zf, int(mask.sum())
    def build_mesh(self, Xi, Yi, Zi, axis, angle_deg):
        verts = np.vstack((Xi.ravel(), Yi.ravel(), -Zi.ravel())).T
        R1 = rotation_matrix(axis, angle_deg)
        verts1 = verts.dot(R1.T)
        R2 = rotation_matrix('x', -90.0)
        verts_final = verts1.dot(R2.T)
        n_rows, n_cols = Xi.shape
        faces = []
        for i in range(n_rows - 1):
            for j in range(n_cols - 1):
                v0 = i * n_cols + j
                v1 = v0 + 1
                v2 = v0 + n_cols
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v2, v1, v3])
        faces = np.array(faces, dtype=np.int64)
        # Verwijder triangles die NaN/inf vertices bevatten en remap indices compact.
        valid_v = np.isfinite(verts_final).all(axis=1)
        keep_f = valid_v[faces].all(axis=1)
        faces = faces[keep_f]
        if faces.size == 0:
            raise ValueError("Geen geldige triangles na filtering (controleer interpolatie/NaN).")
        used = np.unique(faces.ravel())
        new_index = np.full(verts_final.shape[0], -1, dtype=np.int64)
        new_index[used] = np.arange(used.size, dtype=np.int64)
        verts_final = verts_final[used]
        faces = new_index[faces]
        return verts_final, faces

    def _face_normals(self, vertices, faces):
        tri = vertices[faces]  # (F,3,3)
        a = tri[:, 1, :] - tri[:, 0, :]
        b = tri[:, 2, :] - tri[:, 0, :]
        n = np.cross(a, b)
        ln = np.linalg.norm(n, axis=1)
        good = ln > 1e-12
        n[good] = (n[good].T / ln[good]).T
        n[~good] = np.array([0.0, 0.0, 1.0], dtype=float)
        return n

    def _write_collada_ge(self, vertices, faces, output_path):
        normals = self._face_normals(vertices, faces)

        pos_flat = " ".join(f"{v:.6f}" for v in vertices.reshape(-1))
        nrm_flat = " ".join(f"{v:.6f}" for v in normals.reshape(-1))

        # P-array met offsets: (vertex_index normal_index)
        p_data = []
        for fi, f in enumerate(faces):
            p_data.extend([int(f[0]), fi, int(f[1]), fi, int(f[2]), fi])
        p_flat = " ".join(str(v) for v in p_data)

        now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        dae = f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor><authoring_tool>ADCP Bathy Export</authoring_tool></contributor>
    <created>{now}</created>
    <modified>{now}</modified>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_effects>
    <effect id="mat-effect">
      <profile_COMMON>
        <technique sid="common">
          <lambert>
            <diffuse><color>0.3 0.6 0.9 1</color></diffuse>
          </lambert>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="mat-material" name="mat-material">
      <instance_effect url="#mat-effect"/>
    </material>
  </library_materials>
  <library_geometries>
    <geometry id="mesh-geometry" name="mesh-geometry">
      <mesh>
        <source id="mesh-positions">
          <float_array id="mesh-positions-array" count="{vertices.shape[0]*3}">{pos_flat}</float_array>
          <technique_common>
            <accessor source="#mesh-positions-array" count="{vertices.shape[0]}" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="mesh-normals">
          <float_array id="mesh-normals-array" count="{normals.shape[0]*3}">{nrm_flat}</float_array>
          <technique_common>
            <accessor source="#mesh-normals-array" count="{normals.shape[0]}" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="mesh-vertices">
          <input semantic="POSITION" source="#mesh-positions"/>
        </vertices>
        <triangles count="{faces.shape[0]}" material="mat-material">
          <input semantic="VERTEX" source="#mesh-vertices" offset="0"/>
          <input semantic="NORMAL" source="#mesh-normals" offset="1"/>
          <p>{p_flat}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="mesh-node" name="mesh-node">
        <instance_geometry url="#mesh-geometry">
          <bind_material>
            <technique_common>
              <instance_material symbol="mat-material" target="#mat-material"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(dae)

    def export_to_dae(self, Xi, Yi, Zi, output_path, axis: str, angle_deg: float):
        verts, faces = self.build_mesh(Xi, Yi, Zi, axis, angle_deg)
        self._write_collada_ge(verts, faces, output_path)

class BottomProfileGUI:
    def __init__(self, root):
        root.title("Multi-mat 3D Bathymetrie Viewer (Offset per file)")
        self.model = BodemModel()
        self.X_list, self.Y_list, self.Z_list = [], [], []
        self.Xi = self.Yi = self.Zi = None
        frm = tk.Frame(root)
        tk.Label(frm, text="Spike drempel (z-score):").grid(row=0, column=0, sticky='e')
        self.spike_slider = tk.Scale(frm, from_=0.0, to=10.0, orient='horizontal', resolution=0.1, length=200)
        self.spike_slider.set(3.0)
        self.spike_slider.grid(row=0, column=1, padx=5, pady=2)
        tk.Label(frm, text="Median filter (punten):").grid(row=1, column=0, sticky='e')
        self.median_slider = tk.Scale(frm, from_=1, to=9, orient='horizontal', resolution=2, length=200)
        self.median_slider.set(3)
        self.median_slider.grid(row=1, column=1, padx=5, pady=2)
        tk.Label(frm, text="Rotatie-as:").grid(row=2, column=0, sticky='e')
        self.axis_var = tk.StringVar(value='z')
        tk.OptionMenu(frm, self.axis_var, 'x', 'y', 'z').grid(row=2, column=1, padx=5, pady=2, sticky='w')
        tk.Label(frm, text="Rotatiehoek (°):").grid(row=3, column=0, sticky='e')
        self.angle_slider = tk.Scale(frm, from_=0, to=360, orient='horizontal', resolution=1, length=200)
        self.angle_slider.set(0)
        self.angle_slider.grid(row=3, column=1, padx=5, pady=2)
        frm.pack(padx=10, pady=5)
        btns = tk.Frame(root)
        tk.Button(btns, text="Selecteer .mat & Plot", command=self.load_and_plot_multi).pack(side='left', padx=5)
        tk.Button(btns, text="Filter Spikes",    command=self.filter_spikes).pack(side='left', padx=5)
        tk.Button(btns, text="Preview Mesh",     command=self.preview_mesh).pack(side='left', padx=5)
        tk.Button(btns, text="Export DAE",       command=self.export_dae).pack(side='left', padx=5)
        btns.pack(pady=10)
    def load_and_plot_multi(self):
        files = filedialog.askopenfilenames(title="Kies een of meer .mat bestanden", filetypes=[("MAT files","*.mat")])
        if not files: return
        # Gebruik laatst gebruikte/gesuggereerde offset als default
        e0, n0, h0 = 0, 0, 0
        if self.X_list:
            e0, n0, h0 = self.X_list[-1][0], self.Y_list[-1][0], self.Z_list[-1][0]
        for file in files:
            try:
                data = loadmat(file)
                Xr, Yr, Zr = extract_bottom_profile(data)
                e = simpledialog.askfloat("Easting offset", f"Geef UTM31 Easting offset voor {file}:", initialvalue=e0)
                n = simpledialog.askfloat("Northing offset", f"Geef UTM31 Northing offset voor {file}:", initialvalue=n0)
                h = simpledialog.askfloat("Hoogte offset", f"Geef hoogte-offset (m) voor {file}:", initialvalue=h0)
                if e is None or n is None or h is None:
                    continue
                e0, n0, h0 = e, n, h
                Xc = Xr + e
                Yc = Yr + n
                Zc = Zr + h
                # Spike filtering direct op de data
                thresh = self.spike_slider.get()
                median_sz = int(self.median_slider.get())
                Zc, _ = self.model.apply_spike_filter(Zc, thresh=thresh, median_size=median_sz)
                self.X_list.append(Xc)
                self.Y_list.append(Yc)
                self.Z_list.append(Zc)
            except Exception as e:
                messagebox.showerror("Fout bij laden", f"{file}:\n{e}")
        self.update_and_plot_combined()
    def update_and_plot_combined(self):
        # Combineer alle geladen profielen tot één grid
        if not self.X_list:
            messagebox.showwarning("Let op", "Geen profielen geladen."); return
        X_all = np.concatenate(self.X_list)
        Y_all = np.concatenate(self.Y_list)
        Z_all = np.concatenate(self.Z_list)
        self.Xi, self.Yi, self.Zi = self.model.interpolate(X_all, Y_all, Z_all)
        self.model.plot_surface(self.Xi, self.Yi, self.Zi, title="Bathymetrisch overzicht (multibeam-style, alle profielen)")
    def filter_spikes(self):
        if not self.X_list:
            messagebox.showwarning("Let op", "Laad eerst profielen in."); return
        new_Z_list = []
        thresh = self.spike_slider.get()
        median_sz = int(self.median_slider.get())
        for Z in self.Z_list:
            Zf, _ = self.model.apply_spike_filter(Z, thresh=thresh, median_size=median_sz)
            new_Z_list.append(Zf)
        self.Z_list = new_Z_list
        self.update_and_plot_combined()
    def preview_mesh(self):
        if self.Xi is None:
            messagebox.showwarning("Let op", "Laad eerst profielen in."); return
        axis  = self.axis_var.get()
        angle = float(self.angle_slider.get())
        verts, faces = self.model.build_mesh(self.Xi, self.Yi, self.Zi, axis, angle)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(verts[:,0], verts[:,1], verts[:,2], s=1)
        ax.set_xlabel("X'")
        ax.set_ylabel("Y' (up)")
        ax.set_zlabel("Z'")
        ax.set_title(f"Preview (axis={axis.upper()}, hoek={angle}° + GE-flatten)")
        plt.tight_layout()
        plt.show()
    def export_dae(self):
        if self.Xi is None:
            messagebox.showwarning("Let op", "Laad eerst profielen in."); return
        path = filedialog.asksaveasfilename(defaultextension=".dae", filetypes=[("COLLADA files","*.dae")])
        if not path: return
        try:
            axis  = self.axis_var.get()
            angle = float(self.angle_slider.get())
            self.model.export_to_dae(self.Xi, self.Yi, self.Zi, path, axis, angle)
            messagebox.showinfo("Klaar", f"DAE succesvol opgeslagen:\n{path}\nModel ligt nu vlak in het XY-vlak bij import in Google Earth.")
        except Exception as e:
            messagebox.showerror("Fout bij export", str(e))

def main():
    root = tk.Tk()
    BottomProfileGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
