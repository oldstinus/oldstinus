#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aquadopp (AQP) / Aquadopp-profiel verwerking:
- .hdr bevat metadata (starttijd, interval, cell distances, transformation matrix)
- .v1 .v2 .v3 bevatten beam-snelheden per cel (whitespace separated, geen tijdkolom)
- .csv bevat o.a. Speed#i / Dir#i per cel (controle/verwerkte data)

GUI:
- kies .hdr (basisnaam), vink cellen aan, herbereken en teken
- figuren:
  * beam1/beam2/beam3: time series per geselecteerde cel + gemiddelde
  * resultante speed+dir uit .csv over geselecteerde cellen
  * X/Y/Z projecties via transformation matrix: per geselecteerde cel + gemiddelde
- export: tab-gescheiden, decimaal ',' en duizendtallen '.'
"""

import os
import re
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Parsing helpers
# ----------------------------
def _parse_hdr_text(hdr_path: str) -> str:
    with open(hdr_path, "r", errors="ignore") as f:
        return f.read()


def _find_one(text: str, pattern: str, cast=None, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    val = m.group(1).strip()
    return cast(val) if cast else val


def parse_hdr(hdr_path: str) -> dict:
    """
    Leest essentiële metadata uit .hdr:
    - n_meas, n_cells, profile_interval_sec, t0 (datetime)
    - cell_distances_m (list)
    - transformation_matrix (3x3)
    """
    text = _parse_hdr_text(hdr_path)

    n_meas = _find_one(text, r"Number of measurements\s+(\d+)", int)
    n_cells = _find_one(text, r"Number of cells\s+(\d+)", int)
    dt_sec = _find_one(text, r"Profile interval\s+([0-9.]+)\s*sec", float)

    t0_str = _find_one(
        text,
        r"Time of first measurement\s+(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})",
        str
    )
    if not t0_str:
        raise ValueError("Kon 'Time of first measurement' niet vinden in .hdr.")
    t0 = pd.to_datetime(t0_str, format="%d/%m/%Y %H:%M:%S", dayfirst=True)

    # cell center distances block
    cell_distances = []
    block = re.search(
        r"Current profile cell center distance from head \(m\)\s*\n[-]+\s*\n(.*?)(?:\n\s*\n|Data file format)",
        text,
        flags=re.S
    )
    if block:
        for line in block.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            # expected: "1    0.20"
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                try:
                    cell_distances.append(float(parts[1]))
                except Exception:
                    pass

    # Transformation matrix (3x3)
    mat_m = re.search(
        r"Transformation matrix\s+([0-9\.\-\s]+)\n\s*([0-9\.\-\s]+)\n\s*([0-9\.\-\s]+)",
        text
    )
    if not mat_m:
        raise ValueError("Kon 'Transformation matrix' niet vinden in .hdr.")
    rows = []
    for i in range(1, 4):
        row = [float(x) for x in mat_m.group(i).split()]
        if len(row) != 3:
            raise ValueError("Transformation matrix rij is niet 3 waarden breed.")
        rows.append(row)
    T = np.array(rows, dtype=float)

    # sanity checks
    if n_meas is None or n_cells is None or dt_sec is None:
        raise ValueError("Kon n_meas / n_cells / Profile interval niet volledig vinden in .hdr.")

    if cell_distances and len(cell_distances) != n_cells:
        # Niet fatal: we laten toe, maar melden via dict
        pass

    return {
        "n_meas": n_meas,
        "n_cells": n_cells,
        "dt_sec": dt_sec,
        "t0": t0,
        "cell_distances_m": cell_distances,
        "T": T,
    }


def build_time_index(t0: pd.Timestamp, n_meas: int, dt_sec: float) -> pd.DatetimeIndex:
    # Gebruik pandas date_range: exact n_meas, freq=dt_sec
    freq = pd.to_timedelta(dt_sec, unit="s")
    return pd.date_range(start=t0, periods=n_meas, freq=freq)


def load_v_file(path: str, n_meas: int, n_cells: int) -> np.ndarray:
    """
    .v* bevat whitespace-separated floats met n_meas rijen en n_cells kolommen.
    """
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Controleer vorm
    if arr.shape[1] != n_cells:
        raise ValueError(f"{os.path.basename(path)}: verwacht {n_cells} kolommen, kreeg {arr.shape[1]}.")
    if arr.shape[0] != n_meas:
        # Niet altijd fatal, maar meestal wel: matchen op kleinste lengte
        min_n = min(arr.shape[0], n_meas)
        arr = arr[:min_n, :]
    return arr


def load_csv_processed(csv_path: str) -> pd.DataFrame:
    """
    Leest de controle/verwerkte CSV:
    - delimiter ';'
    - DateTime in formaat dd/mm/yyyy HH:MM:SS
    """
    df = pd.read_csv(csv_path, sep=";", engine="python")
    # drop eventuele lege laatste kolom
    if df.columns.size > 0 and (df.columns[-1].strip() == "" or "Unnamed" in df.columns[-1]):
        df = df.iloc[:, :-1]

    # strip kolomnamen
    df.columns = [c.strip() for c in df.columns]

    if "DateTime" not in df.columns:
        raise ValueError("CSV mist kolom 'DateTime'.")

    df["DateTime"] = pd.to_datetime(df["DateTime"].astype(str).str.strip(),
                                    format="%d/%m/%Y %H:%M:%S",
                                    dayfirst=True,
                                    errors="coerce")

    # maak numeriek: veel velden hebben leading spaces
    for c in df.columns:
        if c == "DateTime":
            continue
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False).str.strip(),
                              errors="coerce")
    return df


def speed_dir_to_uv(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Zet (speed, dir) om naar (u, v).
    Aanname: dir in graden, 0° = Noord, 90° = Oost (oceanografisch, 'to' richting).
    Dan: u (oost) = speed * sin(theta), v (noord) = speed * cos(theta)
    """
    th = np.deg2rad(direction_deg)
    u = speed * np.sin(th)
    v = speed * np.cos(th)
    return u, v


def uv_to_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u*u + v*v)
    # dir = atan2(u, v) zodat 0° noord, 90° oost
    direction = (np.rad2deg(np.arctan2(u, v)) + 360.0) % 360.0
    return speed, direction


def compute_xyz_from_beams(T: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Past transformation matrix toe:
    [x;y;z] = T * [b1;b2;b3]
    met b1,b2,b3 arrays (n_meas, n_cells)
    """
    # stack beams -> (n_meas, n_cells, 3)
    B = np.stack([b1, b2, b3], axis=-1)
    # matrix multiply per (t,cell): (3,3) x (3,) -> (3,)
    XYZ = np.einsum("ij,tcj->tci", T, B)
    x = XYZ[..., 0]
    y = XYZ[..., 1]
    z = XYZ[..., 2]
    return x, y, z


# ----------------------------
# Plot helpers
# ----------------------------
def _new_or_clear_figure(fig_key: str, registry: dict, title: str):
    if fig_key in registry and plt.fignum_exists(registry[fig_key].number):
        fig = registry[fig_key]
        fig.clf()
    else:
        fig = plt.figure()
        registry[fig_key] = fig
    fig.suptitle(title)
    return fig


def plot_beam(time_index, beam_arr, selected_cells, cell_distances, beam_name, fig_registry):
    fig = _new_or_clear_figure(f"beam_{beam_name}", fig_registry, f"{beam_name} – snelheden per cel + gemiddelde")
    ax = fig.add_subplot(111)

    if not selected_cells:
        ax.text(0.5, 0.5, "Geen cellen geselecteerd.", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.show()
        return

    # lijnen per cel
    for ci in selected_cells:
        label = f"Cel {ci+1}"
        if cell_distances and ci < len(cell_distances):
            label += f" ({cell_distances[ci]:.2f} m)"
        ax.plot(time_index, beam_arr[:, ci], label=label, linewidth=0.9)

    # gemiddelde over cellen
    avg = np.nanmean(beam_arr[:, selected_cells], axis=1)
    ax.plot(time_index, avg, label="Gemiddelde (geselecteerde cellen)", linewidth=2.0)

    ax.set_xlabel("Tijd")
    ax.set_ylabel("Snelheid langs beam (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.show()


def plot_resultant_from_csv(df_csv, selected_cells, fig_registry):
    """
    Berekent resultante speed+dir over geselecteerde cellen vanuit Speed#i / Dir#i in CSV.
    """
    fig = _new_or_clear_figure("resultant_csv", fig_registry, "Resultante snelheid + richting (uit CSV, over geselecteerde cellen)")
    ax1 = fig.add_subplot(111)

    if df_csv is None or df_csv.empty:
        ax1.text(0.5, 0.5, "CSV niet ingelezen.", ha="center", va="center", transform=ax1.transAxes)
        fig.tight_layout()
        fig.show()
        return

    if not selected_cells:
        ax1.text(0.5, 0.5, "Geen cellen geselecteerd.", ha="center", va="center", transform=ax1.transAxes)
        fig.tight_layout()
        fig.show()
        return

    # Verzamel Speed#k en Dir#k kolommen
    # CSV cellen zijn 1-based: Speed#1 ... Speed#N
    u_list = []
    v_list = []
    for ci in selected_cells:
        k = ci + 1
        speed_col = [c for c in df_csv.columns if c.strip().startswith(f"Speed#{k}")]
        dir_col = [c for c in df_csv.columns if c.strip().startswith(f"Dir#{k}")]
        if not speed_col or not dir_col:
            continue
        sp = df_csv[speed_col[0]].to_numpy(dtype=float)
        dr = df_csv[dir_col[0]].to_numpy(dtype=float)
        u, v = speed_dir_to_uv(sp, dr)
        u_list.append(u)
        v_list.append(v)

    if not u_list:
        ax1.text(0.5, 0.5, "Geen Speed#/Dir# kolommen gevonden voor selectie.", ha="center", va="center", transform=ax1.transAxes)
        fig.tight_layout()
        fig.show()
        return

    U = np.nanmean(np.vstack(u_list), axis=0)
    V = np.nanmean(np.vstack(v_list), axis=0)
    spd, direc = uv_to_speed_dir(U, V)

    t = df_csv["DateTime"]
    ax1.plot(t, spd, linewidth=1.5)
    ax1.set_xlabel("Tijd")
    ax1.set_ylabel("Resultante snelheid (m/s)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(t, direc, linewidth=1.0)
    ax2.set_ylabel("Richting (°)")

    fig.tight_layout()
    fig.show()


def plot_component(time_index, comp_arr, selected_cells, cell_distances, title, ylabel, fig_key, fig_registry):
    fig = _new_or_clear_figure(fig_key, fig_registry, title)
    ax = fig.add_subplot(111)

    if not selected_cells:
        ax.text(0.5, 0.5, "Geen cellen geselecteerd.", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.show()
        return

    for ci in selected_cells:
        label = f"Cel {ci+1}"
        if cell_distances and ci < len(cell_distances):
            label += f" ({cell_distances[ci]:.2f} m)"
        ax.plot(time_index, comp_arr[:, ci], label=label, linewidth=0.9)

    avg = np.nanmean(comp_arr[:, selected_cells], axis=1)
    ax.plot(time_index, avg, label="Gemiddelde (geselecteerde cellen)", linewidth=2.0)

    ax.set_xlabel("Tijd")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.show()


# ----------------------------
# Export helper (EU formatting)
# ----------------------------
def export_tsv_eu(df: pd.DataFrame, out_path: str):
    """
    Schrijft tab-separated, met decimaal ',' en duizendtallen '.'
    """
    df.to_csv(out_path, sep="\t", index=False, decimal=",", thousands=".")


# ----------------------------
# GUI App
# ----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aquadopp – beams (v1/v2/v3) + CSV controle – GUI")
        self.geometry("980x620")

        self.meta = None
        self.base_hdr = None
        self.base_prefix = None

        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.df_csv = None
        self.time_index = None

        self.fig_registry = {}

        # UI state
        self.cell_vars = []
        self.export_var = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        btn_pick = ttk.Button(top, text="Kies .hdr bestand…", command=self.pick_hdr)
        btn_pick.pack(side=tk.LEFT)

        self.lbl_file = ttk.Label(top, text="(geen bestand gekozen)")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        mid = ttk.Frame(self, padding=10)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # left: cell selection (scrollable)
        left = ttk.LabelFrame(mid, text="Cellen (aanvinken)", padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas = tk.Canvas(left, width=320, height=420)
        self.scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # right: actions/info
        right = ttk.Frame(mid, padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.txt_info = tk.Text(right, height=12, wrap="word")
        self.txt_info.pack(fill=tk.X)
        self.txt_info.insert("end", "Kies een .hdr bestand om te starten.\n")
        self.txt_info.configure(state="disabled")

        action = ttk.Frame(right)
        action.pack(fill=tk.X, pady=10)

        chk_export = ttk.Checkbutton(action, text="Export bij herbereken", variable=self.export_var)
        chk_export.pack(side=tk.LEFT)

        btn_run = ttk.Button(action, text="Herbereken en teken", command=self.recompute_and_plot)
        btn_run.pack(side=tk.LEFT, padx=10)

        btn_save_npz = ttk.Button(action, text="Schrijf NPZ weg (beams+xyz)", command=self.save_npz)
        btn_save_npz.pack(side=tk.LEFT)

        tips = ttk.Label(right, text=(
            "Opmerking:\n"
            "• Tijdas wordt opgebouwd uit starttijd + Profile interval.\n"
            "• XYZ wordt berekend via Transformation matrix uit .hdr: [x;y;z]=T*[b1;b2;b3].\n"
            "• Resultante speed/dir wordt berekend uit CSV Speed# / Dir# over geselecteerde cellen."
        ))
        tips.pack(fill=tk.X, pady=10)

    def log(self, msg: str):
        self.txt_info.configure(state="normal")
        self.txt_info.insert("end", msg.rstrip() + "\n")
        self.txt_info.see("end")
        self.txt_info.configure(state="disabled")

    def pick_hdr(self):
        path = filedialog.askopenfilename(
            title="Kies .hdr bestand",
            filetypes=[("HDR files", "*.hdr"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            meta = parse_hdr(path)
        except Exception as e:
            messagebox.showerror("Fout", f"Kan .hdr niet parsen:\n{e}")
            return

        base_prefix = os.path.splitext(path)[0]
        v1_path = base_prefix + ".v1"
        v2_path = base_prefix + ".v2"
        v3_path = base_prefix + ".v3"
        csv_path = base_prefix + ".csv"

        missing = [p for p in [v1_path, v2_path, v3_path, csv_path] if not os.path.exists(p)]
        if missing:
            messagebox.showwarning(
                "Bestanden ontbreken",
                "Volgende bestanden ontbreken:\n\n" + "\n".join(missing) +
                "\n\nIk ga verder met wat beschikbaar is."
            )

        # load v-files
        try:
            self.v1 = load_v_file(v1_path, meta["n_meas"], meta["n_cells"]) if os.path.exists(v1_path) else None
            self.v2 = load_v_file(v2_path, meta["n_meas"], meta["n_cells"]) if os.path.exists(v2_path) else None
            self.v3 = load_v_file(v3_path, meta["n_meas"], meta["n_cells"]) if os.path.exists(v3_path) else None
        except Exception as e:
            messagebox.showerror("Fout", f"Kan v-bestanden niet inlezen:\n{e}")
            return

        # time index length = min beschikbare arrays (als er afwijking was)
        n_meas_eff = meta["n_meas"]
        for arr in [self.v1, self.v2, self.v3]:
            if arr is not None:
                n_meas_eff = min(n_meas_eff, arr.shape[0])
        self.time_index = build_time_index(meta["t0"], n_meas_eff, meta["dt_sec"])

        # trim v arrays to n_meas_eff
        if self.v1 is not None:
            self.v1 = self.v1[:n_meas_eff, :]
        if self.v2 is not None:
            self.v2 = self.v2[:n_meas_eff, :]
        if self.v3 is not None:
            self.v3 = self.v3[:n_meas_eff, :]

        # load csv
        self.df_csv = None
        if os.path.exists(csv_path):
            try:
                self.df_csv = load_csv_processed(csv_path)
            except Exception as e:
                messagebox.showwarning("CSV probleem", f"CSV kon niet goed ingelezen worden:\n{e}")

        self.meta = meta
        self.base_hdr = path
        self.base_prefix = base_prefix

        self.lbl_file.configure(text=os.path.basename(path))
        self.log(f"Gekozen: {path}")
        self.log(f"n_meas={meta['n_meas']} | n_cells={meta['n_cells']} | dt={meta['dt_sec']} s | start={meta['t0']}")
        if meta["cell_distances_m"]:
            self.log("Cell center distances (m): " + ", ".join(f"{x:.2f}" for x in meta["cell_distances_m"]))
        self.log("Transformation matrix T:\n" + "\n".join("  " + "  ".join(f"{v:8.4f}" for v in r) for r in meta["T"]))

        # rebuild cell checkboxes
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self.cell_vars = []
        n_cells = meta["n_cells"]
        dists = meta["cell_distances_m"]

        for i in range(n_cells):
            v = tk.BooleanVar(value=True if i < n_cells else False)
            self.cell_vars.append(v)
            label = f"Cel {i+1}"
            if dists and i < len(dists):
                label += f"  ({dists[i]:.2f} m)"
            cb = ttk.Checkbutton(self.scroll_frame, text=label, variable=v)
            cb.pack(anchor="w")

        self.log("Cellen aangevinkt (standaard: allemaal).")

    def selected_cells(self):
        return [i for i, v in enumerate(self.cell_vars) if v.get()]

    def recompute_and_plot(self):
        if self.meta is None or self.time_index is None:
            messagebox.showinfo("Info", "Kies eerst een .hdr bestand.")
            return
        if self.v1 is None or self.v2 is None or self.v3 is None:
            messagebox.showerror("Fout", "Minstens één van v1/v2/v3 ontbreekt. XYZ en beamplots vereisen alle 3.")
            return

        sel = self.selected_cells()
        dists = self.meta["cell_distances_m"]

        # Beam figures
        plot_beam(self.time_index, self.v1, sel, dists, "Beam 1 (v1)", self.fig_registry)
        plot_beam(self.time_index, self.v2, sel, dists, "Beam 2 (v2)", self.fig_registry)
        plot_beam(self.time_index, self.v3, sel, dists, "Beam 3 (v3)", self.fig_registry)

        # Resultant from CSV (speed+dir over selected cells)
        if self.df_csv is not None:
            plot_resultant_from_csv(self.df_csv, sel, self.fig_registry)
        else:
            self.log("Geen CSV geladen -> resultante speed/dir figuur wordt overgeslagen.")

        # XYZ via transformation matrix
        T = self.meta["T"]
        x, y, z = compute_xyz_from_beams(T, self.v1, self.v2, self.v3)

        plot_component(self.time_index, x, sel, dists,
                       "X-projectie (via T·[b1 b2 b3])", "X (m/s)", "comp_x", self.fig_registry)
        plot_component(self.time_index, y, sel, dists,
                       "Y-projectie (via T·[b1 b2 b3])", "Y (m/s)", "comp_y", self.fig_registry)
        plot_component(self.time_index, z, sel, dists,
                       "Z-projectie (via T·[b1 b2 b3])", "Z (m/s)", "comp_z", self.fig_registry)

        # Optional export
        if self.export_var.get():
            try:
                self._export_current(sel, x, y, z)
            except Exception as e:
                messagebox.showwarning("Export fout", f"Export faalde:\n{e}")

    def _export_current(self, sel, x, y, z):
        """
        Exporteert gemiddelden per beam + xyz + (indien aanwezig) CSV resultante + pressure/attitude.
        """
        out_default = self.base_prefix + "_export.tsv"
        out_path = filedialog.asksaveasfilename(
            title="Kies exportbestand",
            defaultextension=".tsv",
            initialfile=os.path.basename(out_default),
            filetypes=[("TSV (tab separated)", "*.tsv"), ("All files", "*.*")]
        )
        if not out_path:
            return

        # gemiddelden
        b1_avg = np.nanmean(self.v1[:, sel], axis=1) if sel else np.full(len(self.time_index), np.nan)
        b2_avg = np.nanmean(self.v2[:, sel], axis=1) if sel else np.full(len(self.time_index), np.nan)
        b3_avg = np.nanmean(self.v3[:, sel], axis=1) if sel else np.full(len(self.time_index), np.nan)

        x_avg = np.nanmean(x[:, sel], axis=1) if sel else np.full(len(self.time_index), np.nan)
        y_avg = np.nanmean(y[:, sel], axis=1) if sel else np.full(len(self.time_index), np.nan)
        z_avg = np.nanmean(z[:, sel], axis=1) if sel else np.full(len(self.time_index), np.nan)

        df_out = pd.DataFrame({
            "DateTime": self.time_index,
            "beam1_avg": b1_avg,
            "beam2_avg": b2_avg,
            "beam3_avg": b3_avg,
            "x_avg": x_avg,
            "y_avg": y_avg,
            "z_avg": z_avg,
        })

        # Voeg CSV info toe als beschikbaar (align op DateTime via merge_asof)
        if self.df_csv is not None and "DateTime" in self.df_csv.columns:
            df_csv = self.df_csv.copy().sort_values("DateTime")
            df_out = df_out.sort_values("DateTime")

            # resultante uit csv (op basis van selectie)
            u_list, v_list = [], []
            for ci in sel:
                k = ci + 1
                speed_col = [c for c in df_csv.columns if c.strip().startswith(f"Speed#{k}")]
                dir_col = [c for c in df_csv.columns if c.strip().startswith(f"Dir#{k}")]
                if not speed_col or not dir_col:
                    continue
                sp = df_csv[speed_col[0]].to_numpy(dtype=float)
                dr = df_csv[dir_col[0]].to_numpy(dtype=float)
                u, v = speed_dir_to_uv(sp, dr)
                u_list.append(u)
                v_list.append(v)

            if u_list:
                U = np.nanmean(np.vstack(u_list), axis=0)
                V = np.nanmean(np.vstack(v_list), axis=0)
                spd, direc = uv_to_speed_dir(U, V)
                df_csv["resultant_speed_sel"] = spd
                df_csv["resultant_dir_sel"] = direc

            # selecteer extra kolommen (als ze bestaan)
            keep_cols = [c for c in ["Battery", "Heading", "Pitch", "Roll", "Pressure", "Temperature",
                                     "resultant_speed_sel", "resultant_dir_sel"] if c in df_csv.columns]
            df_csv_small = df_csv[["DateTime"] + keep_cols].copy()

            # merge_asof: dichtstbijzijnde tijdstempel (CSV heeft meestal dezelfde dt)
            df_out = pd.merge_asof(df_out, df_csv_small, on="DateTime", direction="nearest")

        export_tsv_eu(df_out, out_path)
        self.log(f"Export geschreven: {out_path}")

    def save_npz(self):
        if self.meta is None or self.time_index is None:
            messagebox.showinfo("Info", "Kies eerst een .hdr bestand.")
            return
        if self.v1 is None or self.v2 is None or self.v3 is None:
            messagebox.showerror("Fout", "v1/v2/v3 vereist.")
            return

        out_default = self.base_prefix + "_beams_xyz.npz"
        out_path = filedialog.asksaveasfilename(
            title="Kies NPZ opslagbestand",
            defaultextension=".npz",
            initialfile=os.path.basename(out_default),
            filetypes=[("NPZ", "*.npz"), ("All files", "*.*")]
        )
        if not out_path:
            return

        T = self.meta["T"]
        x, y, z = compute_xyz_from_beams(T, self.v1, self.v2, self.v3)

        np.savez_compressed(
            out_path,
            time=self.time_index.astype("datetime64[ns]").values,
            cell_distances_m=np.array(self.meta["cell_distances_m"], dtype=float) if self.meta["cell_distances_m"] else np.array([]),
            T=T,
            v1=self.v1,
            v2=self.v2,
            v3=self.v3,
            x=x, y=y, z=z
        )
        self.log(f"NPZ geschreven: {out_path}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
