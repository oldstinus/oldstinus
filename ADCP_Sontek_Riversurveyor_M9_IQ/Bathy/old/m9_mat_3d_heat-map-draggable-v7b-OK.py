#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADCP Multi-Profile 3D • Drag X&Y, Offsets, MAT/DAE export (S-kleur) + per-profiel meshes + Topview
+ Auto-Attach op bestandsnaam (X&Y) + Scrollbare GUI + 3D Navigatie & Schaalpaneel
---------------------------------------------------------------------------------------------------
v12:
- Apart 3D-venster heeft nu een rechter zijpaneel met:
  * Elevatie (°) en Azimut (°) sliders + presets (Topview, Iso)
  * Verticale Overdrijving (VE) voor Z (diepte) — real-time
  * Aslimieten X/Y/Z (min/max) + Toepassen, Fit Data, Nudge knoppen (pan)
- Navigatie & schaal werken gesynchroniseerd met de ingebedde 3D-plot.

Overige features (v11): scrollbare GUI, per-profiel DAE (met S-vertex-kleur), MAT-export,
auto-attach op bestandsnaam (X&Y), slepen in 2D, offsets-tabel, groot 3D-venster, topview default.
"""

import os, re
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.patches
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement

# ---------------- MATLAB struct helpers ----------------
def _todict(matobj):
    if not hasattr(matobj, "_fieldnames"):
        return matobj
    d = {}
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
    elem_list = []
    for elem in ndarray:
        if isinstance(elem, scipy.io.matlab.mat_struct):
            elem_list.append(_todict(elem))
        elif isinstance(elem, np.ndarray):
            elem_list.append(_tolist(elem))
        else:
            elem_list.append(elem)
    return elem_list

def _check_keys(d):
    for key in list(d.keys()):
        if key.startswith("__"):
            continue
        val = d[key]
        if isinstance(val, scipy.io.matlab.mat_struct):
            d[key] = _todict(val)
        elif isinstance(val, np.ndarray):
            d[key] = _tolist(val)
    return d

def loadmat(filepath):
    mat_data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    return _check_keys(mat_data)

# ---------------- CSV export (optioneel) ----------------
def save_structure_to_csv(struct_data, struct_name, out_dir, base_filename):
    try:
        df = pd.DataFrame(struct_data)
        out_path = os.path.join(out_dir, f"{base_filename}_{struct_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] CSV voor '{struct_name}': {out_path}")
    except Exception as e:
        if isinstance(struct_data, dict):
            print(f"[!] Kan '{struct_name}' niet direct exporteren als CSV: {e}")
            for key, value in struct_data.items():
                sub_name = f"{struct_name}_{key}"
                try:
                    df_sub = pd.DataFrame(value)
                    out_path = os.path.join(out_dir, f"{base_filename}_{sub_name}.csv")
                    df_sub.to_csv(out_path, index=False)
                    print(f"[OK] CSV veld '{key}' in '{struct_name}': {out_path}")
                except Exception as ex_sub:
                    if isinstance(value, (str, int, float)):
                        out_path = os.path.join(out_dir, f"{base_filename}_{sub_name}.csv")
                        df_scalar = pd.DataFrame({"Value": [value]})
                        df_scalar.to_csv(out_path, index=False)
                        print(f"[OK] CSV scalar veld '{key}' in '{struct_name}': {out_path}")
                    else:
                        print(f"[!] Kan veld '{key}' van '{struct_name}' niet exporteren: {ex_sub}")
        elif isinstance(struct_data, (str, int, float)):
            out_path = os.path.join(out_dir, f"{base_filename}_{struct_name}.csv")
            df_scalar = pd.DataFrame({"Value": [struct_data]})
            df_scalar.to_csv(out_path, index=False)
            print(f"[OK] CSV scalar veld '{struct_name}': {out_path}")
        else:
            print(f"[!] Kan '{struct_name}' niet exporteren: {e}")

def export_top_level_to_csv(data_dict, mat_filepath):
    mat_dir = os.path.dirname(mat_filepath)
    base_fname = os.path.splitext(os.path.basename(mat_filepath))[0]
    for key in data_dict:
        if key.startswith("__"):
            continue
        save_structure_to_csv(data_dict[key], key, mat_dir, base_fname)

# ---------------- Timestamp parsing ----------------
_ts14 = re.compile(r"(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")  # YYYYMMDDhhmmss

def parse_timestamp_from_name(path):
    name = os.path.basename(path)
    m = _ts14.search(name.replace("-", "").replace("_", ""))
    if not m:
        m2 = re.search(r"\d{14}", name)
        if not m2:
            return None
        ts = m2.group(0)
    else:
        ts = "".join(m.groups())
    try:
        return int(ts)
    except Exception:
        return None

# ---------------- 3D data extract ----------------
def extract_3d_arrays(data_dict, interp_factor=2):
    reqs = ["Summary", "System", "BottomTrack", "WaterTrack"]
    for r in reqs:
        if r not in data_dict:
            raise ValueError(f"Struct '{r}' ontbreekt in data.")
    summary = data_dict["Summary"]
    system  = data_dict["System"]
    btrack  = data_dict["BottomTrack"]
    wtrack  = data_dict["WaterTrack"]

    track    = np.array(summary["Track"])               # (NS,2)
    cstart   = np.array(system["Cell_Start"]).squeeze() # (NS,)
    csize    = np.array(system["Cell_Size"]).squeeze()  # (NS,)
    bt_depth = np.array(btrack["BT_Depth"]).squeeze()   # (NS,)
    velocity = np.array(wtrack["Velocity"])             # (NC,4,NS)

    if "Time" not in system:
        start_time = np.nan
    else:
        time_arr = np.array(system["Time"]).squeeze()
        start_time = float(np.min(time_arr))

    NC, four, NS = velocity.shape
    if four < 2:
        raise ValueError("Velocity array heeft minder dan 2 velocity-componenten.")
    u = velocity[:,0,:]; v = velocity[:,1,:]
    speed = np.sqrt(u**2 + v**2)  # (NC, NS)

    nVert = (NC-1)*interp_factor + 1
    X2d = np.full((NS, nVert), np.nan)
    Y2d = np.full((NS, nVert), np.nan)
    Z2d = np.full((NS, nVert), np.nan)
    S2d = np.full((NS, nVert), np.nan)

    for i in range(NS):
        z_raw = cstart[i] + (np.arange(NC)+0.5)*csize[i]
        s_raw = speed[:, i]
        valid = z_raw < bt_depth[i]
        z_raw = z_raw[valid]; s_raw = s_raw[valid]
        if len(z_raw) < 2: continue
        if interp_factor > 1:
            z_new = np.linspace(z_raw[0], z_raw[-1], (len(z_raw)-1)*interp_factor + 1)
            s_new = np.interp(z_new, z_raw, s_raw)
            z_use, s_use = z_new, s_new
        else:
            z_use, s_use = z_raw, s_raw

        npts = len(z_use)
        X2d[i,:npts] = track[i,0]
        Y2d[i,:npts] = track[i,1]
        Z2d[i,:npts] = z_use
        S2d[i,:npts] = s_use

    start_xy = np.array(track[0], dtype=float)
    end_xy   = np.array(track[-1], dtype=float)

    return X2d.T, Y2d.T, Z2d.T, S2d.T, start_time, (start_xy, end_xy), track

# ---------------- COLLADA helpers ----------------
def _collada_root():
    root = Element("COLLADA", {
        "xmlns": "http://www.collada.org/2005/11/COLLADASchema",
        "version": "1.4.1"
    })
    asset = SubElement(root, "asset")
    SubElement(asset, "up_axis").text = "Z_UP"
    return root

def _float_array(text_id, arr):
    flat = " ".join(f"{v:.6f}" for v in arr.flatten())
    node = Element("float_array", {"id": text_id, "count": str(arr.size)})
    node.text = flat
    return node

def _add_default_fx_and_mat(root):
    lib_fx = SubElement(root, "library_effects")
    effect = SubElement(lib_fx, "effect", {"id": "fx_default"})
    prof  = SubElement(effect, "profile_COMMON")
    tech  = SubElement(prof, "technique", {"sid": "common"})
    lambert = SubElement(tech, "lambert")
    color = SubElement(SubElement(lambert, "diffuse"), "color")
    color.text = "0.7 0.7 0.75 1"

    lib_mat = SubElement(root, "library_materials")
    material = SubElement(lib_mat, "material", {"id": "mat_default", "name": "mat_default"})
    SubElement(material, "instance_effect", {"url": "#fx_default"})

def _add_scene_root(root):
    lib_vs = SubElement(root, "library_visual_scenes")
    vs = SubElement(lib_vs, "visual_scene", {"id": "Scene", "name": "Scene"})
    scene = SubElement(root, "scene")
    SubElement(scene, "instance_visual_scene", {"url": "#Scene"})
    return vs

def _append_geometry(mesh_parent, geom_id, vertices, triangles, colors_rgba=None):
    geom = SubElement(mesh_parent, "geometry", {"id": geom_id, "name": geom_id})
    mesh = SubElement(geom, "mesh")

    src_pos = SubElement(mesh, "source", {"id": f"{geom_id}-positions"})
    arr = np.asarray(vertices, dtype=float).reshape(-1,3)
    src_pos.append(_float_array(f"{geom_id}-positions-array", arr))
    tc = SubElement(src_pos, "technique_common")
    accessor = SubElement(tc, "accessor", {
        "source": f"#{geom_id}-positions-array",
        "count": str(arr.shape[0]), "stride": "3"
    })
    SubElement(accessor, "param", {"name":"X","type":"float"})
    SubElement(accessor, "param", {"name":"Y","type":"float"})
    SubElement(accessor, "param", {"name":"Z","type":"float"})

    has_colors = colors_rgba is not None and len(colors_rgba) == arr.shape[0]
    if has_colors:
        src_col = SubElement(mesh, "source", {"id": f"{geom_id}-colors"})
        carr = np.asarray(colors_rgba, dtype=float).reshape(-1,4)
        src_col.append(_float_array(f"{geom_id}-colors-array", carr))
        tc2 = SubElement(src_col, "technique_common")
        accessor2 = SubElement(tc2, "accessor", {
            "source": f"#{geom_id}-colors-array",
            "count": str(carr.shape[0]), "stride": "4"
        })
        SubElement(accessor2, "param", {"name":"R","type":"float"})
        SubElement(accessor2, "param", {"name":"G","type":"float"})
        SubElement(accessor2, "param", {"name":"B","type":"float"})
        SubElement(accessor2, "param", {"name":"A","type":"float"})

    verts = SubElement(mesh, "vertices", {"id": f"{geom_id}-vertices"})
    SubElement(verts, "input", {"semantic": "POSITION", "source": f"#{geom_id}-positions"})

    tris = SubElement(mesh, "triangles", {"count": str(len(triangles))})
    SubElement(tris, "input", {"semantic": "VERTEX", "source": f"#{geom_id}-vertices", "offset": "0"})
    if has_colors:
        SubElement(tris, "input", {"semantic": "COLOR", "source": f"#{geom_id}-colors", "offset": "1"})

    if has_colors:
        idx_text = " ".join(" ".join(f"{i} {i}" for i in tri) for tri in triangles)
    else:
        idx_text = " ".join(" ".join(str(i) for i in tri) for tri in triangles)
    SubElement(tris, "p").text = idx_text

def write_collada_multi_profile(out_path, per_profile_data, with_colors=True):
    root = _collada_root()
    _add_default_fx_and_mat(root)

    lib_geo = SubElement(root, "library_geometries")
    for d in per_profile_data:
        geom_id = f"geom_{d['name']}"
        _append_geometry(lib_geo, geom_id, d["vertices"], d["triangles"], d["colors"] if with_colors else None)

    vs = _add_scene_root(root)
    for d in per_profile_data:
        geom_id = f"geom_{d['name']}"
        node = SubElement(vs, "node", {"id": f"node_{d['name']}", "name": d["name"]})
        inst = SubElement(node, "instance_geometry", {"url": f"#{geom_id}"})
        bind = SubElement(inst, "bind_material")
        tc = SubElement(bind, "technique_common")
        SubElement(tc, "instance_material", {"symbol": "mat_default", "target": "#mat_default"})

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ", level=0)
    except Exception:
        pass
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

# ---------------- Scrollable container ----------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.vbar.pack(side="right", fill="y")
        self.hbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.content = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.content)

    def _on_content_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, widget):
        widget.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind_all("<Button-4>", self._on_mousewheel, add="+")
        widget.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            delta = -1 * int(event.delta/120) if event.delta != 0 else 0
            if delta != 0:
                self.canvas.yview_scroll(delta, "units")

# ---------------- GUI hoofd-app ----------------
class InteractiveMultiProfileGUI:
    def __init__(self, master):
        self.root = master
        self.toplevel = self.root.winfo_toplevel()
        self.toplevel.title("ADCP Multi-Profile 3D – v12 (scroll, per-profiel DAE, auto-attach, nav & schaal)")

        self.mat_files = []
        self.profiles = []
        # 3D controls state
        self.ve_factor = 1.0     # Vertical Exaggeration
        self.elev = 90.0         # deg
        self.azim = -90.0        # deg
        self._custom_limits = None  # (xmin,xmax,ymin,ymax,zmin,zmax) or None for auto-fit

        # externe 3D-venster refs
        self.win3d = None
        self.fig3d_ext = None
        self.ax3d_ext = None
        self.canvas3d_ext = None
        self.toolbar3d_ext = None

        # --- Top bar ---
        top = ttk.Frame(self.root); top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Button(top, text="Selecteer .mat bestanden", command=self.select_mat_files).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Exporteer CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Laad & Plot", command=self.load_and_plot).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Open 3D in apart venster", command=self.open_3d_window).pack(side=tk.LEFT, padx=12)
        ttk.Button(top, text="Export .mat (samengevoegd)", command=self.export_combined_mat).pack(side=tk.LEFT, padx=12)
        ttk.Button(top, text="Export DAE (samengevoegd)", command=self.export_combined_dae).pack(side=tk.LEFT, padx=4)

        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, pady=(0,6))

        # --- Figure (embedded) ---
        self.fig = Figure(figsize=(8.8, 6.8), dpi=100)
        self.ax3d = self.fig.add_subplot(2,1,1, projection='3d')
        self.ax2d = self.fig.add_subplot(2,1,2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        self.toolbar.update(); self.toolbar.pack(side=tk.TOP, fill=tk.X)

        # --- Bottom controls ---
        bottom = ttk.Frame(self.root); bottom.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(bottom, text="Tijd→Y schaal (m/s):").grid(row=0, column=0, padx=(0,4))
        self.time_scale_var = tk.DoubleVar(value=0.1)
        ttk.Entry(bottom, textvariable=self.time_scale_var, width=8).grid(row=0, column=1, padx=(0,10))

        ttk.Button(bottom, text="Auto-Attach (Y)", command=self.auto_attach_Y_only).grid(row=0, column=2, padx=4)

        ttk.Label(bottom, text="Gap (m):").grid(row=0, column=3, padx=(16,4))
        self.attach_gap_var = tk.DoubleVar(value=0.0)
        ttk.Entry(bottom, textvariable=self.attach_gap_var, width=8).grid(row=0, column=4, padx=(0,10))
        ttk.Button(bottom, text="Auto-Attach (op bestandsnaam, X&Y)", command=self.auto_attach_by_filename)\
            .grid(row=0, column=5, padx=4)

        ttk.Button(bottom, text="Reset Y (tijd)", command=self.reset_y_offsets).grid(row=0, column=6, padx=10)

        ttk.Label(bottom, text="Interp factor:").grid(row=0, column=7, padx=(12,2))
        self.interp_var = tk.IntVar(value=2)
        ttk.Spinbox(bottom, from_=1, to=10, textvariable=self.interp_var, width=5).grid(row=0, column=8, padx=(0,12))

        ttk.Label(bottom, text="Nudge-stap:").grid(row=0, column=9, padx=(0,2))
        self.nudge_var = tk.DoubleVar(value=0.5)
        ttk.Entry(bottom, textvariable=self.nudge_var, width=6).grid(row=0, column=10, padx=(0,12))

        ttk.Label(bottom, text="DAE stride:").grid(row=0, column=11, padx=(0,2))
        self.dae_stride_var = tk.IntVar(value=1)
        ttk.Spinbox(bottom, from_=1, to=10, textvariable=self.dae_stride_var, width=5).grid(row=0, column=12, padx=(0,12))

        # MAT export options
        ttk.Label(bottom, text="MAT compressie:").grid(row=0, column=13, padx=(6,2))
        self.mat_compress_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bottom, variable=self.mat_compress_var).grid(row=0, column=14, padx=(0,12))

        ttk.Label(bottom, text="Forceer float64:").grid(row=0, column=15, padx=(0,2))
        self.mat_float64_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bottom, variable=self.mat_float64_var).grid(row=0, column=16, padx=(0,12))

        # DAE opties
        color_frame = ttk.LabelFrame(self.root, text="DAE opties")
        color_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        self.dae_color_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(color_frame, text="DAE kleur (S) aan", variable=self.dae_color_var)\
            .grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.dae_manual_vrange = tk.BooleanVar(value=False)
        ttk.Checkbutton(color_frame, text="Handmatige vmin/vmax", variable=self.dae_manual_vrange)\
            .grid(row=0, column=1, padx=6, pady=2)
        ttk.Label(color_frame, text="vmin:").grid(row=0, column=2, padx=(8,2))
        self.vmin_entry = tk.DoubleVar(value=0.0)
        ttk.Entry(color_frame, textvariable=self.vmin_entry, width=8).grid(row=0, column=3, padx=(0,8))
        ttk.Label(color_frame, text="vmax:").grid(row=0, column=4, padx=(8,2))
        self.vmax_entry = tk.DoubleVar(value=1.0)
        ttk.Entry(color_frame, textvariable=self.vmax_entry, width=8).grid(row=0, column=5, padx=(0,8))
        ttk.Label(color_frame, text="(viridis RGBA)").grid(row=0, column=6, padx=(6,2))
        self.dae_separate_meshes = tk.BooleanVar(value=True)
        ttk.Checkbutton(color_frame, text="DAE per-profiel meshes", variable=self.dae_separate_meshes)\
            .grid(row=0, column=7, padx=(16,4), pady=2, sticky="w")

        # --- Offsets table ---
        self.offset_panel = ttk.LabelFrame(self.root, text="Offsets per profiel (X en Y)")
        self.offset_panel.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        header = ttk.Frame(self.offset_panel); header.pack(fill=tk.X, pady=(2,2))
        ttk.Label(header, text="#", width=4).grid(row=0, column=0, padx=2)
        ttk.Label(header, text="Bestand", width=36).grid(row=0, column=1, padx=2)
        ttk.Label(header, text="Timestamp", width=16).grid(row=0, column=2, padx=2)
        ttk.Label(header, text="X-offset", width=12).grid(row=0, column=3, padx=2)
        ttk.Label(header, text="Y-offset", width=12).grid(row=0, column=4, padx=2)
        self.offset_rows_frame = ttk.Frame(self.offset_panel); self.offset_rows_frame.pack(fill=tk.X)
        sync_bar = ttk.Frame(self.offset_panel); sync_bar.pack(fill=tk.X, pady=(4,2))
        ttk.Button(sync_bar, text="Offsets naar lijst (sync ←)", command=self.sync_offsets_to_list)\
            .pack(side=tk.LEFT, padx=4)
        ttk.Button(sync_bar, text="Toepassen (lijst →)", command=self.apply_list_to_offsets)\
            .pack(side=tk.LEFT, padx=4)

        # events
        self.dragging = False
        self.selected_idx = None
        self._drag_start_xy = None
        self._orig_offsets = None

        self.cid_press = self.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key = self.canvas.mpl_connect("key_press_event", self.on_key)

        self.norm = None; self.cmap = cm.viridis; self.sm = None
        self.x_vars = []; self.y_vars = []

    # ---------- Files ----------
    def select_mat_files(self):
        files = filedialog.askopenfilenames(title="Selecteer meerdere .mat bestanden",
                                            filetypes=[("MAT files","*.mat")])
        if files:
            self.mat_files = list(files)
            print("Geselecteerde bestanden:")
            for f in self.mat_files: print("  ", f)

    def export_csv(self):
        if not self.mat_files:
            messagebox.showwarning("Let op", "Geen bestanden geselecteerd."); return
        for fpath in self.mat_files:
            print(f"--- CSV-export: {fpath} ---")
            try:
                data_dict = loadmat(fpath)
                export_top_level_to_csv(data_dict, fpath)
            except Exception as e:
                print(f"[!] Fout bij exporteren: {e}")

    # ---------- Load & plot ----------
    def load_and_plot(self):
        if not self.mat_files:
            messagebox.showwarning("Let op", "Geen bestanden geselecteerd."); return

        self.profiles.clear(); self.x_vars.clear(); self.y_vars.clear()
        for child in self.offset_rows_frame.winfo_children(): child.destroy()

        interp = int(self.interp_var.get())
        for fpath in self.mat_files:
            print(f"--- Laad 3D arrays: {fpath} ---")
            try:
                data_dict = loadmat(fpath)
                Xp, Yp, Zp, Sp, t0, (start_xy, end_xy), track = extract_3d_arrays(data_dict, interp_factor=interp)
                ts_name = parse_timestamp_from_name(fpath)
                self.profiles.append({
                    "Xp": Xp, "Yp0": Yp, "Zp": Zp, "Sp": Sp,
                    "t0": (float(t0) if np.isfinite(t0) else np.nan),
                    "xoffset": 0.0, "yoffset": 0.0,
                    "surf": None, "rect": None, "label": None, "path": fpath,
                    "interp": interp,
                    "start_xy": np.array(start_xy, float),
                    "end_xy": np.array(end_xy, float),
                    "track": np.array(track, float),
                    "ts_name": ts_name
                })
            except Exception as e:
                print(f"[!] Fout bij 3D-extract: {e}")

        if not self.profiles:
            messagebox.showerror("Fout", "Geen valide profielen geladen."); return

        all_speeds = np.concatenate([p["Sp"][~np.isnan(p["Sp"])] for p in self.profiles if np.any(~np.isnan(p["Sp"]))])
        vmin, vmax = float(np.nanmin(all_speeds)), float(np.nanmax(all_speeds))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        self.norm = Normalize(vmin=vmin, vmax=vmax)

        self.compute_y_offsets_from_time()

        for i, p in enumerate(self.profiles):
            row = ttk.Frame(self.offset_rows_frame); row.grid(row=i, column=0, sticky="ew", pady=1)
            ttk.Label(row, text=str(i+1), width=4).grid(row=0, column=0, padx=2)
            ttk.Label(row, text=os.path.basename(p["path"]), width=36).grid(row=0, column=1, padx=2)
            ts_show = p["ts_name"] if p["ts_name"] is not None else "n/a"
            ttk.Label(row, text=str(ts_show), width=16).grid(row=0, column=2, padx=2)
            xv = tk.DoubleVar(value=p["xoffset"]); yv = tk.DoubleVar(value=p["yoffset"])
            self.x_vars.append(xv); self.y_vars.append(yv)
            x_entry = ttk.Entry(row, textvariable=xv, width=12)
            y_entry = ttk.Entry(row, textvariable=yv, width=12)
            x_entry.grid(row=0, column=3, padx=2); y_entry.grid(row=0, column=4, padx=2)
            x_entry.bind("<Return>", lambda e, idx=i: self._apply_single_offset(idx))
            x_entry.bind("<FocusOut>", lambda e, idx=i: self._apply_single_offset(idx))
            y_entry.bind("<Return>", lambda e, idx=i: self._apply_single_offset(idx))
            y_entry.bind("<FocusOut>", lambda e, idx=i: self._apply_single_offset(idx))

        # reset custom limits on new load
        self._custom_limits = None
        self.redraw_all()

    def compute_y_offsets_from_time(self):
        if not self.profiles: return
        finite_t0 = [p for p in self.profiles if np.isfinite(p["t0"])]
        if not finite_t0: return
        ref_t0 = finite_t0[0]["t0"]
        scale = float(self.time_scale_var.get())
        for p in self.profiles:
            if np.isfinite(p["t0"]):
                dt = p["t0"] - ref_t0
                p["yoffset"] = dt * scale

    # ---------- 3D external window ----------
    def open_3d_window(self):
        if self.win3d is not None and tk.Toplevel.winfo_exists(self.win3d):
            self.win3d.lift(); return

        self.win3d = tk.Toplevel(self.toplevel)
        self.win3d.title("ADCP 3D – Groot venster (live) + Navigatie & Schaal")
        self.win3d.geometry("1500x900")

        # Layout: left plot, right control panel
        main = ttk.Frame(self.win3d); main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.LabelFrame(main, text="3D Navigatie & Schaal"); right.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        self.fig3d_ext = Figure(figsize=(12, 8), dpi=100)
        self.ax3d_ext = self.fig3d_ext.add_subplot(1,1,1, projection='3d')
        self.canvas3d_ext = FigureCanvasTkAgg(self.fig3d_ext, master=left)
        self.canvas3d_ext.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar3d_ext = NavigationToolbar2Tk(self.canvas3d_ext, left, pack_toolbar=False)
        self.toolbar3d_ext.update(); self.toolbar3d_ext.pack(fill=tk.X)

        # --- Controls on the right ---
        # Elev/Azim
        ttk.Label(right, text="Elev (°)").grid(row=0, column=0, sticky="w", padx=6, pady=(6,2))
        self.elev_var = tk.DoubleVar(value=self.elev)
        ttk.Scale(right, from_=-10, to=90, variable=self.elev_var, command=lambda v: self._update_camera())\
            .grid(row=0, column=1, sticky="ew", padx=6, pady=(6,2))
        ttk.Label(right, text="Azim (°)").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.azim_var = tk.DoubleVar(value=self.azim)
        ttk.Scale(right, from_=-180, to=180, variable=self.azim_var, command=lambda v: self._update_camera())\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=2)

        # Presets
        preset = ttk.Frame(right); preset.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(preset, text="Topview", command=self._preset_top).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset, text="Isometrisch", command=self._preset_iso).pack(side=tk.LEFT, padx=3)

        # Vertical Exaggeration
        ttk.Label(right, text="VE (Z-overdrijving)").grid(row=3, column=0, sticky="w", padx=6, pady=(8,2))
        self.ve_var = tk.DoubleVar(value=self.ve_factor)
        ve_spin = ttk.Spinbox(right, from_=0.1, to=10.0, increment=0.1, textvariable=self.ve_var, width=8)
        ve_spin.grid(row=3, column=1, sticky="w", padx=6, pady=(8,2))
        ve_spin.bind("<Return>", lambda e: self._apply_ve())
        ve_spin.bind("<FocusOut>", lambda e: self._apply_ve())
        ttk.Button(right, text="VE toepassen", command=self._apply_ve).grid(row=4, column=1, sticky="w", padx=6, pady=(2,6))

        # Axis limits
        limits = ttk.LabelFrame(right, text="As-limieten")
        limits.grid(row=5, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        for c in range(2):
            limits.grid_columnconfigure(c, weight=1)

        self.xmin_var = tk.DoubleVar(); self.xmax_var = tk.DoubleVar()
        self.ymin_var = tk.DoubleVar(); self.ymax_var = tk.DoubleVar()
        self.zmin_var = tk.DoubleVar(); self.zmax_var = tk.DoubleVar()
        ttk.Label(limits, text="Xmin").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(limits, textvariable=self.xmin_var, width=10).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(limits, text="Xmax").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(limits, textvariable=self.xmax_var, width=10).grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(limits, text="Ymin").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(limits, textvariable=self.ymin_var, width=10).grid(row=2, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(limits, text="Ymax").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(limits, textvariable=self.ymax_var, width=10).grid(row=3, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(limits, text="Zmin").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(limits, textvariable=self.zmin_var, width=10).grid(row=4, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(limits, text="Zmax").grid(row=5, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(limits, textvariable=self.zmax_var, width=10).grid(row=5, column=1, sticky="ew", padx=4, pady=2)

        btns = ttk.Frame(limits); btns.grid(row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        ttk.Button(btns, text="Toepassen", command=self._apply_limits).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Fit data", command=self._fit_limits).pack(side=tk.LEFT, padx=2)

        # Nudge/pan
        nudge = ttk.LabelFrame(right, text="Pan (nudge)")
        nudge.grid(row=6, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        self.pan_step_var = tk.DoubleVar(value=0.1)  # fractie van huidig bereik
        ttk.Label(nudge, text="Stap (% van bereik):").grid(row=0, column=0, padx=4, pady=2)
        ttk.Entry(nudge, textvariable=self.pan_step_var, width=6).grid(row=0, column=1, padx=4, pady=2, sticky="w")
        # pijltjes
        arrows = ttk.Frame(nudge); arrows.grid(row=1, column=0, columnspan=2, pady=4)
        ttk.Button(arrows, text="←", width=3, command=lambda: self._pan('x', -1)).grid(row=1, column=0, padx=2)
        ttk.Button(arrows, text="→", width=3, command=lambda: self._pan('x', 1)).grid(row=1, column=2, padx=2)
        ttk.Button(arrows, text="↑", width=3, command=lambda: self._pan('y', 1)).grid(row=0, column=1, padx=2)
        ttk.Button(arrows, text="↓", width=3, command=lambda: self._pan('y', -1)).grid(row=2, column=1, padx=2)
        ttk.Button(arrows, text="Z+", width=3, command=lambda: self._pan('z', 1)).grid(row=0, column=3, padx=2)
        ttk.Button(arrows, text="Z−", width=3, command=lambda: self._pan('z', -1)).grid(row=2, column=3, padx=2)

        # configure scaling of right side
        for c in range(2):
            right.grid_columnconfigure(c, weight=1)

        def _on_close():
            self.win3d.destroy()
            self.win3d = None; self.fig3d_ext=None; self.ax3d_ext=None; self.canvas3d_ext=None; self.toolbar3d_ext=None
        self.win3d.protocol("WM_DELETE_WINDOW", _on_close)

        self._fit_limits()  # init limits from data
        self.redraw_all()

    # ---------- Drawing ----------
    def _current_data_bounds(self):
        """Bereken data-bounds (met offsets en VE) als (xmin,xmax,ymin,ymax,zmin,zmax)."""
        if not self.profiles:
            return (0,1,0,1,0,1)
        xs, ys, zs = [], [], []
        for p in self.profiles:
            Xp = p["Xp"] + p["xoffset"]
            Yp = p["Yp0"] + p["yoffset"]
            Zp = -p["Zp"] * self.ve_factor
            xs.extend([np.nanmin(Xp), np.nanmax(Xp)])
            ys.extend([np.nanmin(Yp), np.nanmax(Yp)])
            zs.extend([np.nanmin(Zp), np.nanmax(Zp)])
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        zmin, zmax = float(np.nanmin(zs)), float(np.nanmax(zs))
        return xmin, xmax, ymin, ymax, zmin, zmax

    def _apply_view_and_limits(self, ax):
        # camera
        try:
            ax.view_init(elev=self.elev, azim=self.azim)
        except Exception:
            pass
        # limieten
        if self._custom_limits is None:
            xmin, xmax, ymin, ymax, zmin, zmax = self._current_data_bounds()
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = self._custom_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

    def _draw_3d_axes(self, ax):
        ax.clear()
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Diepte (m)")
        for p in self.profiles:
            Xp, Yp0, Zp, Sp = p["Xp"], p["Yp0"], p["Zp"], p["Sp"]
            Xp_off = Xp + p["xoffset"]; Yp_off = Yp0 + p["yoffset"]
            facecolors = cm.viridis(self.norm(Sp))
            ax.plot_surface(Xp_off, Yp_off, -Zp * self.ve_factor, facecolors=facecolors,
                            rstride=1, cstride=1, linewidth=0, shade=False, antialiased=False)
        self._apply_view_and_limits(ax)

    def redraw_all(self):
        # embedded
        self.ax3d.clear()
        self._draw_3d_axes(self.ax3d)
        if self.sm is None:
            self.sm = cm.ScalarMappable(norm=self.norm, cmap=cm.viridis)
            self.sm.set_array([])
            self.fig.colorbar(self.sm, ax=self.ax3d, shrink=0.6, aspect=14, label="Horiz. snelheid (m/s)")
        else:
            self.sm.set_norm(self.norm)
        # 2D layout
        self.ax2d.clear(); self.ax2d.set_title("Layout (bovenaanzicht): sleep profielen in X & Y")
        self.ax2d.set_xlabel("X (m)"); self.ax2d.set_ylabel("Y (m)")
        allX, allY = [], []
        for p in self.profiles:
            Xp = p["Xp"] + p["xoffset"]; Yp = p["Yp0"] + p["yoffset"]
            allX += [np.nanmin(Xp), np.nanmax(Xp)]; allY += [np.nanmin(Yp), np.nanmax(Yp)]
        if allX and allY and np.isfinite(allX).all() and np.isfinite(allY).all():
            xmin, xmax = float(np.nanmin(allX)), float(np.nanmax(allX))
            ymin, ymax = float(np.nanmin(allY)), float(np.nanmax(allY))
            xspan = xmax - xmin if xmax > xmin else 1.0
            yspan = ymax - ymin if ymax > ymin else 1.0
            self.ax2d.set_xlim(xmin - 0.05*xspan, xmax + 0.05*xspan)
            self.ax2d.set_ylim(ymin - 0.05*yspan, ymax + 0.05*yspan)
        for idx, p in enumerate(self.profiles):
            Xp = p["Xp"] + p["xoffset"]; Yp = p["Yp0"] + p["yoffset"]
            x0, x1 = float(np.nanmin(Xp)), float(np.nanmax(Xp))
            y0, y1 = float(np.nanmin(Yp)), float(np.nanmax(Yp))
            rect = matplotlib.patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                                fill=False, lw=2, ec="tab:blue", picker=True)
            self.ax2d.add_patch(rect)
            label = self.ax2d.text((x0+x1)/2.0, (y0+y1)/2.0, f"{idx+1}",
                                   ha="center", va="center", fontsize=10, color="tab:blue")
            p["rect"] = rect; p["label"] = label
        self.canvas.draw_idle()

        # external
        if self.ax3d_ext is not None and self.canvas3d_ext is not None:
            self.ax3d_ext.clear()
            self._draw_3d_axes(self.ax3d_ext)
            self.canvas3d_ext.draw_idle()

    # ---------- Dragging ----------
    def pick_profile_at(self, event):
        if event.inaxes != self.ax2d: return None
        for i, p in enumerate(self.profiles):
            rect = p["rect"]
            contains, _ = rect.contains(event)
            if contains: return i
        return None

    def on_press(self, event):
        if event.dblclick and event.inaxes == self.ax2d:
            self.selected_idx = None; self.highlight_selection(None); self.canvas.draw_idle(); return
        idx = self.pick_profile_at(event)
        if idx is not None:
            self.selected_idx = idx; self.highlight_selection(idx)
            self.dragging = True; self._drag_start_xy = (event.xdata, event.ydata)
            self._orig_offsets = (self.profiles[idx]["xoffset"], self.profiles[idx]["yoffset"])

    def on_motion(self, event):
        if not self.dragging or self.selected_idx is None: return
        if event.inaxes != self.ax2d or event.xdata is None or event.ydata is None or self._drag_start_xy is None:
            return
        dx = event.xdata - self._drag_start_xy[0]; dy = event.ydata - self._drag_start_xy[1]
        x0, y0 = self._orig_offsets
        self.profiles[self.selected_idx]["xoffset"] = x0 + dx
        self.profiles[self.selected_idx]["yoffset"] = y0 + dy
        self.redraw_all(); self.sync_single_to_list(self.selected_idx)

    def on_release(self, event):
        if self.dragging:
            self.dragging = False; self._drag_start_xy = None; self._orig_offsets = None
            self.redraw_all()

    def on_key(self, event):
        if self.selected_idx is None: return
        step = float(self.nudge_var.get())
        p = self.profiles[self.selected_idx]
        if event.key == "up":    p["yoffset"] += step
        elif event.key == "down":  p["yoffset"] -= step
        elif event.key == "right": p["xoffset"] += step
        elif event.key == "left":  p["xoffset"] -= step
        else: return
        self.redraw_all(); self.sync_single_to_list(self.selected_idx)

    def highlight_selection(self, idx):
        for i, p in enumerate(self.profiles):
            rect = p["rect"]; lbl = p["label"]
            if i == idx:
                rect.set_ec("tab:red"); rect.set_linewidth(2.5); lbl.set_color("tab:red")
            else:
                rect.set_ec("tab:blue"); rect.set_linewidth(2.0); lbl.set_color("tab:blue")

    # ---------- Offsets GUI sync ----------
    def _apply_single_offset(self, idx):
        try:
            xval = float(self.x_vars[idx].get()); yval = float(self.y_vars[idx].get())
        except Exception:
            messagebox.showerror("Fout", "Ongeldige offsetwaarde."); return
        self.profiles[idx]["xoffset"] = xval; self.profiles[idx]["yoffset"] = yval
        self.redraw_all()

    def sync_single_to_list(self, idx):
        self.x_vars[idx].set(self.profiles[idx]["xoffset"])
        self.y_vars[idx].set(self.profiles[idx]["yoffset"])

    def sync_offsets_to_list(self):
        for i in range(len(self.profiles)): self.sync_single_to_list(i)

    def apply_list_to_offsets(self):
        for i in range(len(self.profiles)): self._apply_single_offset(i)

    # ---------- Attach helpers ----------
    def auto_attach_Y_only(self):
        if not self.profiles: return
        order = np.argsort([p["t0"] if np.isfinite(p["t0"]) else np.inf for p in self.profiles])
        Ymins0 = [float(np.nanmin(p["Yp0"])) for p in self.profiles]
        Ymaxs0 = [float(np.nanmax(p["Yp0"])) for p in self.profiles]
        base = order[0]; cur_tail = Ymaxs0[base] + self.profiles[base]["yoffset"]
        for idx in order[1:]:
            want_y = cur_tail - Ymins0[idx]
            self.profiles[idx]["yoffset"] = want_y
            cur_tail = Ymaxs0[idx] + want_y
        self.redraw_all(); self.sync_offsets_to_list()

    def auto_attach_by_filename(self):
        if not self.profiles: return
        gap = float(self.attach_gap_var.get())
        keys = []
        for i, p in enumerate(self.profiles):
            if p["ts_name"] is not None:
                keys.append((p["ts_name"], i))
            elif np.isfinite(p["t0"]):
                keys.append((int(p["t0"]*1e6), i))
            else:
                keys.append((int(9e18), i))
        order = [idx for _, idx in sorted(keys, key=lambda x: x[0])]
        for p in self.profiles: p["xoffset"] = 0.0; p["yoffset"] = 0.0

        def best_anchor(prev_end_xy, start_xy, end_xy):
            d_start = np.linalg.norm(prev_end_xy - start_xy)
            d_end   = np.linalg.norm(prev_end_xy - end_xy)
            return ("start", start_xy) if d_start <= d_end else ("end", end_xy)

        for k in range(1, len(order)):
            prev = self.profiles[order[k-1]]
            curr = self.profiles[order[k]]
            prev_end = prev["end_xy"] + np.array([prev["xoffset"], prev["yoffset"]])
            which, anchor_xy = best_anchor(prev_end, curr["start_xy"], curr["end_xy"])
            target = prev_end + np.array([0.0, gap])  # eenvoudige gap in Y
            off = target - anchor_xy
            curr["xoffset"] = float(off[0]); curr["yoffset"] = float(off[1])

        self._custom_limits = None  # her-fit na auto-attach
        self.redraw_all(); self.sync_offsets_to_list()

    def reset_y_offsets(self):
        self.compute_y_offsets_from_time()
        self._custom_limits = None
        self.redraw_all(); self.sync_offsets_to_list()

    # ---------- MAT export (combined) ----------
    def export_combined_mat(self):
        if not self.profiles:
            messagebox.showwarning("Let op", "Laad eerst profielen."); return
        out_path = filedialog.asksaveasfilename(defaultextension=".mat",
                                                filetypes=[("MATLAB file","*.mat")],
                                                initialfile="adcp_profiles_combined.mat",
                                                title="Sla gecombineerd .mat-bestand op")
        if not out_path: return
        do_comp = bool(self.mat_compress_var.get())
        force_f64 = bool(self.mat_float64_var.get())
        try:
            prof_list = []; max_rows = 0; total_cols = 0; col_ranges = []
            for p in self.profiles:
                Xp = p["Xp"] + p["xoffset"]; nrows, ncols = Xp.shape
                max_rows = max(max_rows, nrows); total_cols += ncols
            dtype = np.float64 if force_f64 else np.float32
            Xcat = np.full((max_rows, total_cols), np.nan, dtype=dtype)
            Ycat = np.full((max_rows, total_cols), np.nan, dtype=dtype)
            Zcat = np.full((max_rows, total_cols), np.nan, dtype=dtype)
            Scat = np.full((max_rows, total_cols), np.nan, dtype=dtype)
            cur = 0
            for p in self.profiles:
                Xp = (p["Xp"] + p["xoffset"]).astype(dtype, copy=False)
                Yp = (p["Yp0"] + p["yoffset"]).astype(dtype, copy=False)
                Zp = (p["Zp"]).astype(dtype, copy=False)
                Sp = (p["Sp"]).astype(dtype, copy=False)
                nrows, ncols = Xp.shape
                Xcat[:nrows, cur:cur+ncols] = Xp
                Ycat[:nrows, cur:cur+ncols] = Yp
                Zcat[:nrows, cur:cur+ncols] = Zp
                Scat[:nrows, cur:cur+ncols] = Sp
                col_ranges.append([cur+1, cur+ncols]); cur += ncols
                prof_list.append({
                    "X": Xp, "Y": Yp, "Z": Zp, "S": Sp,
                    "xoffset": float(p["xoffset"]), "yoffset": float(p["yoffset"]),
                    "t0": float(p["t0"]) if np.isfinite(p["t0"]) else np.nan,
                    "source": os.path.basename(p["path"]),
                    "interp_factor": int(p.get("interp", 2)),
                    "start_xy": p["start_xy"], "end_xy": p["end_xy"],
                    "ts_name": p["ts_name"]
                })
            mdict = {
                "profiles": np.array(prof_list, dtype=object),
                "Xcat": Xcat, "Ycat": Ycat, "Zcat": Zcat, "Scat": Scat,
                "profile_col_ranges": np.array(col_ranges, dtype=np.int32),
                "note": "Concatenatiekolommen zijn NaN-gepad; offsets zijn toegepast in Xcat/Ycat.",
                "generated_by": "m9_mat_3d_heat-map-draggable-v12.py",
                "options": {"compressed": do_comp, "float": "float64" if force_f64 else "float32"}
            }
            savemat(out_path, mdict, do_compression=do_comp)
            messagebox.showinfo("OK", f"Samengevoegd .mat-bestand opgeslagen:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Exportfout", f"Kon .mat niet opslaan:\n{e}")

    # ---------- DAE export (combined) ----------
    def export_combined_dae(self):
        if not self.profiles:
            messagebox.showwarning("Let op", "Laad eerst profielen."); return
        out_path = filedialog.asksaveasfilename(defaultextension=".dae",
                                                filetypes=[("COLLADA DAE","*.dae")],
                                                initialfile="adcp_profiles_combined.dae",
                                                title="Sla gecombineerd .dae-bestand op")
        if not out_path: return
        stride = int(self.dae_stride_var.get()); stride = 1 if stride < 1 else stride
        make_colors = bool(self.dae_color_var.get())
        if bool(self.dae_manual_vrange.get()):
            vmin = float(self.vmin_entry.get()); vmax = float(self.vmax_entry.get())
        else:
            vmin = float(self.norm.vmin); vmax = float(self.norm.vmax)
        if vmax <= vmin: vmax = vmin + 1e-6
        norm = Normalize(vmin=vmin, vmax=vmax)
        separate = bool(self.dae_separate_meshes.get())
        try:
            if separate:
                per_profile_data = []
                for p in self.profiles:
                    base = os.path.splitext(os.path.basename(p["path"]))[0]
                    name = re.sub(r"[^A-Za-z0-9_\-]", "_", base)
                    X = (p["Xp"] + p["xoffset"]).astype(float)
                    Y = (p["Yp0"] + p["yoffset"]).astype(float)
                    Z = (-p["Zp"]).astype(float)  # let op: VE is visueel, niet export
                    S = (p["Sp"]).astype(float)
                    nrows, ncols = X.shape
                    idx = -np.ones((nrows, ncols), dtype=int)
                    vertices, triangles, colors = [], [], []
                    v_off = 0
                    for i in range(0, nrows, stride):
                        for j in range(0, ncols, stride):
                            if np.isfinite(X[i,j]) and np.isfinite(Y[i,j]) and np.isfinite(Z[i,j]):
                                vertices.append([X[i,j], Y[i,j], Z[i,j]])
                                if make_colors and np.isfinite(S[i,j]):
                                    r,g,b,a = cm.viridis(norm(S[i,j])); colors.append([r,g,b,a])
                                elif make_colors:
                                    colors.append([0.0,0.0,0.0,0.0])
                                idx[i,j] = v_off; v_off += 1
                    for i in range(0, nrows-stride, stride):
                        for j in range(0, ncols-stride, stride):
                            i2 = i+stride; j2 = j+stride
                            a,b,c,d = idx[i,j], idx[i2,j], idx[i,j2], idx[i2,j2]
                            if a>=0 and b>=0 and c>=0: triangles.append([a,b,c])
                            if b>=0 and d>=0 and c>=0: triangles.append([b,d,c])
                    if len(vertices)==0 or len(triangles)==0:
                        print(f"[!] Profiel '{name}': geen geldige geometrie, overgeslagen."); continue
                    per_profile_data.append({
                        "name": name,
                        "vertices": np.asarray(vertices, float),
                        "triangles": np.asarray(triangles, int),
                        "colors": (np.asarray(colors, float) if make_colors else None)
                    })
                if not per_profile_data:
                    messagebox.showerror("Exportfout", "Geen enkele profiel-geometrie kon worden opgebouwd."); return
                write_collada_multi_profile(out_path, per_profile_data, with_colors=make_colors)
                messagebox.showinfo("OK", f"DAE opgeslagen (per-profiel meshes):\n{out_path}\nProfielen: {len(per_profile_data)}")
            else:
                vertices, triangles, colors = [], [], []
                v_offset = 0
                for p in self.profiles:
                    X = (p["Xp"] + p["xoffset"]).astype(float)
                    Y = (p["Yp0"] + p["yoffset"]).astype(float)
                    Z = (-p["Zp"]).astype(float)
                    S = (p["Sp"]).astype(float)
                    nrows, ncols = X.shape
                    idx = -np.ones((nrows, ncols), dtype=int)
                    for i in range(0, nrows, stride):
                        for j in range(0, ncols, stride):
                            if np.isfinite(X[i,j]) and np.isfinite(Y[i,j]) and np.isfinite(Z[i,j]):
                                vertices.append([X[i,j], Y[i,j], Z[i,j]])
                                if make_colors and np.isfinite(S[i,j]):
                                    rgba = cm.viridis(norm(S[i,j])); colors.append([rgba[0],rgba[1],rgba[2],rgba[3]])
                                elif make_colors:
                                    colors.append([0.0,0.0,0.0,0.0])
                                idx[i,j] = v_offset; v_offset += 1
                    for i in range(0, nrows-stride, stride):
                        for j in range(0, ncols-stride, stride):
                            i2 = i+stride; j2 = j+stride
                            a,b,c,d = idx[i,j], idx[i2,j], idx[i,j2], idx[i2,j2]
                            if a>=0 and b>=0 and c>=0: triangles.append([a,b,c])
                            if b>=0 and d>=0 and c>=0: triangles.append([b,d,c])
                if len(vertices)==0 or len(triangles)==0:
                    messagebox.showerror("Exportfout", "Geen geldige vertices/triangles om te exporteren."); return
                per_profile_data = [{
                    "name": "all",
                    "vertices": np.asarray(vertices, float),
                    "triangles": np.asarray(triangles, int),
                    "colors": (np.asarray(colors, float) if make_colors else None)
                }]
                write_collada_multi_profile(out_path, per_profile_data, with_colors=make_colors)
                messagebox.showinfo("OK", f"DAE opgeslagen (enkele mesh):\n{out_path}\nVertices: {len(vertices)} • Triangles: {len(triangles)}")
        except Exception as e:
            messagebox.showerror("Exportfout", f"Kon DAE niet opslaan:\n{e}")

    # ---------- 3D controls (right panel) ----------
    def _update_camera(self):
        self.elev = float(self.elev_var.get()); self.azim = float(self.azim_var.get())
        self.redraw_all()

    def _preset_top(self):
        self.elev = 90.0; self.azim = -90.0
        if hasattr(self, 'elev_var'): self.elev_var.set(self.elev)
        if hasattr(self, 'azim_var'): self.azim_var.set(self.azim)
        self.redraw_all()

    def _preset_iso(self):
        self.elev = 30.0; self.azim = -60.0
        if hasattr(self, 'elev_var'): self.elev_var.set(self.elev)
        if hasattr(self, 'azim_var'): self.azim_var.set(self.azim)
        self.redraw_all()

    def _apply_ve(self):
        try:
            ve = float(self.ve_var.get())
            if ve <= 0: raise ValueError
            self.ve_factor = ve
        except Exception:
            messagebox.showerror("Fout", "VE moet > 0 zijn."); return
        # bij VE-wijziging: als custom limits actief, schaal Z-limieten mee naar nieuwe VE
        if self._custom_limits is not None:
            xmin,xmax,ymin,ymax,zmin,zmax = self._custom_limits
            # herbereken z-limieten o.b.v. verhouding oude/nieuwe VE is lastig zonder oude VE,
            # daarom fitten we opnieuw op data en heroverschrijven Z met dezelfde min/max-fractie
            self._custom_limits = None
            self._fit_limits()
        self.redraw_all()

    def _fit_limits(self):
        xmin,xmax,ymin,ymax,zmin,zmax = self._current_data_bounds()
        self.xmin_var.set(xmin); self.xmax_var.set(xmax)
        self.ymin_var.set(ymin); self.ymax_var.set(ymax)
        self.zmin_var.set(zmin); self.zmax_var.set(zmax)
        self._custom_limits = (xmin,xmax,ymin,ymax,zmin,zmax)
        self.redraw_all()

    def _apply_limits(self):
        try:
            xmin = float(self.xmin_var.get()); xmax = float(self.xmax_var.get())
            ymin = float(self.ymin_var.get()); ymax = float(self.ymax_var.get())
            zmin = float(self.zmin_var.get()); zmax = float(self.zmax_var.get())
        except Exception:
            messagebox.showerror("Fout", "Ongeldige as-limietwaarde."); return
        if not (xmax > xmin and ymax > ymin and zmax > zmin):
            messagebox.showerror("Fout", "Max > Min moet gelden voor alle assen."); return
        self._custom_limits = (xmin,xmax,ymin,ymax,zmin,zmax)
        self.redraw_all()

    def _pan(self, axis, direction):
        # direction: -1 / +1; verschuif een fractie van huidig bereik
        if self._custom_limits is None:
            self._fit_limits()  # init limits
        xmin,xmax,ymin,ymax,zmin,zmax = self._custom_limits
        fx = float(self.pan_step_var.get()); fx = max(0.001, fx)
        if axis == 'x':
            rng = xmax - xmin; shift = direction * fx * rng
            xmin += shift; xmax += shift
        elif axis == 'y':
            rng = ymax - ymin; shift = direction * fx * rng
            ymin += shift; ymax += shift
        elif axis == 'z':
            rng = zmax - zmin; shift = direction * fx * rng
            zmin += shift; zmax += shift
        self._custom_limits = (xmin,xmax,ymin,ymax,zmin,zmax)
        # update UI fields
        self.xmin_var.set(xmin); self.xmax_var.set(xmax)
        self.ymin_var.set(ymin); self.ymax_var.set(ymax)
        self.zmin_var.set(zmin); self.zmax_var.set(zmax)
        self.redraw_all()

# ---------------- main ----------------
def main():
    root = tk.Tk()
    try:
        style = ttk.Style(); style.theme_use("clam")
    except Exception:
        pass
    root.geometry("1200x800")
    sf = ScrollableFrame(root); sf.pack(fill="both", expand=True)
    app = InteractiveMultiProfileGUI(sf.content)
    root.mainloop()

if __name__ == "__main__":
    main()
