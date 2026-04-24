#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAT track splitter (bovenaanzicht) + export deeltracks als .mat
- Laadt .mat met behoud van MATLAB structuren (mat_struct)
- Plot Summary.Track (XY)
- Klik begin + eind om segmenten te maken
- Exporteert elk segment als .mat met dezelfde structuur:
  * alle per-ensemble velden (lengte NS of as==NS) worden gesliced naar segment
  * algemene/config velden blijven behouden
"""

from __future__ import annotations

import os
import copy
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Optioneel: snelle nearest-neighbour
try:
    from scipy.spatial import cKDTree
    _HAS_KDTREE = True
except Exception:
    cKDTree = None
    _HAS_KDTREE = False


# -------------------------
# MAT helpers (struct behoud)
# -------------------------

def load_mat_keep_struct(mat_path: str | Path) -> dict:
    """
    Laadt .mat met behoud van MATLAB structs als mat_struct objecten.
    (zelfde stijl als je georef-script) :contentReference[oaicite:1]{index=1}
    """
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    return mat

def savemat_keep_struct(out_path: str | Path, mat_dict: dict) -> None:
    """
    Schrijft .mat weg, met behoud van mat_struct objecten.
    Verwijdert __header__/__version__/__globals__ om SciPy warnings te vermijden.
    :contentReference[oaicite:2]{index=2}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mdict = {k: v for k, v in mat_dict.items() if not str(k).startswith("__")}
    sio.savemat(
        str(out_path),
        mdict,
        do_compression=True,
        long_field_names=True,
        oned_as="row",
    )

def get_summary_track(mat: dict) -> np.ndarray:
    if "Summary" not in mat:
        raise ValueError("MAT mist top-level struct 'Summary'.")
    summ = mat["Summary"]
    if not hasattr(summ, "Track"):
        raise ValueError("MAT: Summary.Track niet gevonden.")
    tr = np.asarray(getattr(summ, "Track"), dtype=float)
    if tr.ndim != 2 or tr.shape[1] < 2:
        raise ValueError(f"Summary.Track heeft onverwachte vorm: {tr.shape} (verwacht Nx2).")
    return tr[:, :2].astype(float)

# -------------------------
# Slicer: slice alles wat per-ensemble is
# -------------------------

def _slice_array_by_NS(arr: np.ndarray, idx: np.ndarray, NS: int) -> np.ndarray:
    """Slice numpy arrays langs een as die == NS."""
    if not isinstance(arr, np.ndarray) or arr.ndim == 0:
        return arr

    axes = [a for a, s in enumerate(arr.shape) if s == NS]
    if not axes:
        return arr

    # Speciaal geval: (NS, NS) => slice beide assen
    if arr.ndim == 2 and arr.shape[0] == NS and arr.shape[1] == NS:
        return arr[np.ix_(idx, idx)]

    # Kies 'meest waarschijnlijke' ensemble-as:
    # prefer last axis (bv. Velocity: NC x 4 x NS), anders eerste
    axis = axes[-1]

    try:
        return np.take(arr, idx, axis=axis)
    except Exception:
        return arr

def slice_inplace_keep_structure(obj, idx: np.ndarray, NS: int):
    """
    Recursief slicen:
    - mat_struct: loop velden af en zet geslicede waarden terug
    - np.ndarray: slice langs as == NS
    - list/tuple: recursief
    - scalars: unchanged
    """
    # numpy array
    if isinstance(obj, np.ndarray):
        return _slice_array_by_NS(obj, idx, NS)

    # MATLAB struct (scipy.io.matlab.mat_struct) herkennen via _fieldnames
    if hasattr(obj, "_fieldnames"):
        for f in getattr(obj, "_fieldnames", []):
            try:
                v = getattr(obj, f)
            except Exception:
                continue
            newv = slice_inplace_keep_structure(v, idx, NS)
            try:
                setattr(obj, f, newv)
            except Exception:
                pass

        # (optioneel) update teller-velden als ze bestaan
        for name in (
            "NumEnsembles", "Num_Ensembles", "NEnsembles", "nEnsembles",
            "Total_Ensembles", "TotalEnsembles"
        ):
            if hasattr(obj, name):
                try:
                    setattr(obj, name, int(len(idx)))
                except Exception:
                    pass

        return obj

    # containers
    if isinstance(obj, list):
        return [slice_inplace_keep_structure(v, idx, NS) for v in obj]
    if isinstance(obj, tuple):
        return tuple(slice_inplace_keep_structure(v, idx, NS) for v in obj)

    return obj

def build_segment_mat(original_mat: dict, seg_start: int, seg_end: int) -> dict:
    """
    Maakt een deep copy van de originele mat dict (structuur intact),
    en sliced alle per-ensemble velden naar [seg_start:seg_end] (incl.).
    """
    tr = get_summary_track(original_mat)
    NS = tr.shape[0]

    a = int(min(seg_start, seg_end))
    b = int(max(seg_start, seg_end))
    a = max(0, a)
    b = min(NS - 1, b)
    idx = np.arange(a, b + 1, dtype=int)

    mat_copy = copy.deepcopy(original_mat)
    # Slice alle top-level keys (behalve __*)
    for k in list(mat_copy.keys()):
        if str(k).startswith("__"):
            continue
        mat_copy[k] = slice_inplace_keep_structure(mat_copy[k], idx, NS)

    return mat_copy


# -------------------------
# GUI + plot + interactie
# -------------------------

class TrackSplitterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MAT Track Splitter – klik begin/eind → export deeltracks")
        self.geometry("1200x780")

        self.mat_path: str | None = None
        self.mat_data: dict | None = None
        self.track_xy: np.ndarray | None = None

        self.pending_start: int | None = None
        self.segments: list[tuple[int, int]] = []

        self._kdtree = None  # optioneel

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="1) Laad .mat", command=self.load_mat).pack(side="left")
        ttk.Button(top, text="2) Plot track", command=self.plot_track).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Wis segmenten", command=self.clear_segments).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Verwijder geselecteerd segment", command=self.remove_selected_segment).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="3) Export segments", command=self.export_segments).pack(side="left", padx=(8, 0))

        self.lbl = ttk.Label(top, text="Geen bestand geladen.")
        self.lbl.pack(side="right")

        mid = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        mid.pack(fill="both", expand=True)

        # links: plot
        left = ttk.Frame(mid, padding=8)
        mid.add(left, weight=4)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, left)
        self.toolbar.update()

        self.canvas.mpl_connect("button_press_event", self.on_click)

        # rechts: segmentenlijst + info
        right = ttk.Frame(mid, padding=8)
        mid.add(right, weight=1)

        ttk.Label(right, text="Segmenten (klik start+eind):").pack(anchor="w")
        self.listbox = tk.Listbox(right, height=18)
        self.listbox.pack(fill="x", pady=(6, 10))

        self.txt = tk.Text(right, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True)
        self._log("Gebruik:\n- Klik 1x op track = start\n- Klik 2e keer = eind → segment toegevoegd\n- Herhaal voor meerdere segmenten\n- Export segments → schrijft .mat per segment\n")

    def _log(self, s: str):
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def load_mat(self):
        p = filedialog.askopenfilename(
            title="Selecteer .mat",
            filetypes=[("MAT-files", "*.mat"), ("All files", "*.*")]
        )
        if not p:
            return
        try:
            mat = load_mat_keep_struct(p)
            tr = get_summary_track(mat)

            self.mat_path = p
            self.mat_data = mat
            self.track_xy = tr
            self.pending_start = None
            self.segments = []
            self._kdtree = None

            self.lbl.configure(text=os.path.basename(p))
            self._log(f"[OK] Geladen: {p}")
            self._log(f"     Track punten: {tr.shape[0]}")
            messagebox.showinfo("OK", "MAT geladen. Klik nu 'Plot track'.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"[ERROR] {e}")

    def plot_track(self):
        if self.track_xy is None:
            messagebox.showwarning("Geen data", "Laad eerst een .mat.")
            return

        xy = self.track_xy
        self.ax.clear()
        self.ax.plot(xy[:, 0], xy[:, 1], linewidth=1.5)
        self.ax.scatter(xy[:, 0], xy[:, 1], s=6)

        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Bovenaanzicht track — klik start en eind")

        # KDTree voor snelle nearest neighbour
        if _HAS_KDTREE and xy.shape[0] >= 2000:
            try:
                self._kdtree = cKDTree(xy[:, :2])
                self._log("[INFO] KDTree actief voor snelle puntselectie.")
            except Exception:
                self._kdtree = None

        # herteken bestaande segmenten
        for (a, b) in self.segments:
            self._draw_segment(a, b, redraw=False)

        self.canvas.draw()

    def _nearest_index(self, x: float, y: float) -> int:
        xy = self.track_xy
        if xy is None:
            return 0
        pt = np.array([x, y], dtype=float)

        if self._kdtree is not None:
            _, idx = self._kdtree.query(pt, k=1)
            return int(idx)

        d2 = (xy[:, 0] - pt[0]) ** 2 + (xy[:, 1] - pt[1]) ** 2
        return int(np.argmin(d2))

    def on_click(self, event):
        if self.track_xy is None or self.mat_data is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        # enkel linker muisknop
        if getattr(event, "button", None) != 1:
            return

        idx = self._nearest_index(event.xdata, event.ydata)

        if self.pending_start is None:
            self.pending_start = idx
            self._draw_marker(idx, label="START")
            self._log(f"Start gekozen: index={idx} (1-based={idx+1})")
        else:
            a = self.pending_start
            b = idx
            self.pending_start = None
            self.segments.append((a, b))
            self._add_segment_to_list(a, b)
            self._draw_segment(a, b, redraw=True)
            self._log(f"Segment toegevoegd: {min(a,b)} → {max(a,b)} (1-based: {min(a,b)+1} → {max(a,b)+1})")

    def _draw_marker(self, idx: int, label: str = ""):
        xy = self.track_xy
        if xy is None:
            return
        self.ax.scatter([xy[idx, 0]], [xy[idx, 1]], s=80)
        if label:
            self.ax.text(xy[idx, 0], xy[idx, 1], f" {label}", fontsize=9)
        self.canvas.draw()

    def _draw_segment(self, a: int, b: int, redraw: bool = True):
        xy = self.track_xy
        if xy is None:
            return
        i0, i1 = sorted([int(a), int(b)])
        seg = xy[i0:i1+1, :]
        self.ax.plot(seg[:, 0], seg[:, 1], linewidth=3.0)
        if redraw:
            self.canvas.draw()

    def _add_segment_to_list(self, a: int, b: int):
        i0, i1 = sorted([int(a), int(b)])
        n = i1 - i0 + 1
        self.listbox.insert("end", f"Seg {len(self.segments):02d}: {i0+1}→{i1+1} ({n} pts)")

    def clear_segments(self):
        self.segments = []
        self.pending_start = None
        self.listbox.delete(0, "end")
        self._log("Segmenten gewist.")
        if self.track_xy is not None:
            self.plot_track()

    def remove_selected_segment(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        k = int(sel[0])
        if k < 0 or k >= len(self.segments):
            return
        seg = self.segments.pop(k)
        self.listbox.delete(k)
        # herlabel
        items = list(self.listbox.get(0, "end"))
        self.listbox.delete(0, "end")
        for i, _ in enumerate(self.segments, start=1):
            a, b = self.segments[i-1]
            i0, i1 = sorted([a, b])
            n = i1 - i0 + 1
            self.listbox.insert("end", f"Seg {i:02d}: {i0+1}→{i1+1} ({n} pts)")
        self._log(f"Segment verwijderd: {seg}")
        if self.track_xy is not None:
            self.plot_track()

    def export_segments(self):
        if self.mat_data is None or self.mat_path is None or self.track_xy is None:
            messagebox.showwarning("Geen data", "Laad eerst een .mat en maak segmenten.")
            return
        if not self.segments:
            messagebox.showwarning("Geen segmenten", "Geen segmenten gedefinieerd (klik start+eind).")
            return

        out_dir = filedialog.askdirectory(title="Kies output map voor deeltracks")
        if not out_dir:
            return

        base = Path(self.mat_path).stem
        NS = self.track_xy.shape[0]
        self._log(f"Export naar: {out_dir}")

        try:
            for i, (a, b) in enumerate(self.segments, start=1):
                i0, i1 = sorted([int(a), int(b)])
                i0 = max(0, i0)
                i1 = min(NS - 1, i1)

                seg_mat = build_segment_mat(self.mat_data, i0, i1)

                out_name = f"{base}_seg{i:02d}_{i0+1}-{i1+1}.mat"
                out_path = Path(out_dir) / out_name
                savemat_keep_struct(out_path, seg_mat)

                self._log(f"[OK] {out_name}")

            messagebox.showinfo("Klaar", f"{len(self.segments)} deeltracks weggeschreven.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"[ERROR] {e}")


def main():
    app = TrackSplitterApp()
    app.mainloop()

if __name__ == "__main__":
    main()
