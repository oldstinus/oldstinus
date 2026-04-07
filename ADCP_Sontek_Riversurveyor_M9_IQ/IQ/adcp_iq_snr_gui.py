#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SonTek IQ ADCP – VELBEAM + SNR (2 CSV files) analyser + GUI

INPUT
- multi_VELBEAM.csv : beam-snelheden per cel (Cell# Velocity (beam).b)
- multi_SNR.csv     : SNR per cel en beam (Cell# SNR.b)

AANNAMES
- Beide files hebben dezelfde samples en delen minstens "Sample Number" en/of "Sample Time".
- Cellen zijn typisch 0.04 m hoog (maar script gebruikt de "Cell# Location (...)(m)" uit de file).

OUTPUT
1) Tijdlijn-figuur (heatmap) per beam: SNR (dB) als functie van tijd en afstand.
2) Max bruikbare afstand per beam:
   - Automatisch: detecteert een "plotse verhoging" in de mediane SNR-profiel vs afstand (knik in dSNR/dz).
   - Manueel: overschrijven in GUI.
3) GUI-gestuurde herwerking:
   - beams aan/uit vinken
   - hoogte/afstand-range kiezen
   - per beam max afstand instellen
   - "Herwerk" => gemiddelde horizontale snelheid (SonTek IQ Janus benadering)
       u = (b1 - b2)/(2*sin(theta))
       v = (b4 - b3)/(2*sin(theta))
       speed = sqrt(u^2 + v^2)

STARTEN (Windows)
    py adcp_iq_snr_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math

CELL_LOC_RE = re.compile(r"^Cell(\d+)\s+Location\s+\((Center|Skew)\)\s+\(m\)$", re.IGNORECASE)
VEL_BEAM_RE = re.compile(r"^Cell(\d+)\s+Velocity\s+\(beam\)\.(\d+)\s+\(m/s\)$", re.IGNORECASE)
SNR_BEAM_RE = re.compile(r"^Cell(\d+)\s+SNR\.(\d+)\s+\(dB\)$", re.IGNORECASE)

def find_time_column(df: pd.DataFrame) -> str:
    for cand in ["Sample Time", "Time", "DateTime", "Datetime", "Timestamp", "2 Sample Time"]:
        if cand in df.columns:
            return cand
    for c in df.columns[:10]:
        try:
            pd.to_datetime(df[c].iloc[:50], errors="raise", dayfirst=True)
            return c
        except Exception:
            pass
    raise ValueError("Geen tijdkolom gevonden.")

def find_pressure_column(df: pd.DataFrame) -> str:
    for cand in ["Pressure (dbar)", "Pressure", "P", "Druk", "Pressure (uncorrected) (dbar)"]:
        if cand in df.columns:
            return cand
    for c in df.columns[:10]:
        try:
            pd.to_numeric(df[c].iloc[:50], errors="raise")
            return c
        except Exception:
            pass
    raise ValueError("Geen pressure kolom gevonden.")

def find_velocity_column(df: pd.DataFrame) -> str:
    for cand in ["Velocity (X-OBS).X-Center (Beam 1 Only) (m/s)", "Velocity", "Vel", "X Velocity"]:
        if cand in df.columns:
            return cand
    for c in df.columns[:10]:
        try:
            pd.to_numeric(df[c].iloc[:50], errors="raise")
            return c
        except Exception:
            pass
    raise ValueError("Geen velocity kolom gevonden.")

def parse_cells(df: pd.DataFrame):
    cells=set()
    loc_center, loc_skew = {}, {}
    vel_cols = {}
    snr_cols = {}

    for c in df.columns:
        m = CELL_LOC_RE.match(c)
        if m:
            cell = int(m.group(1))
            kind = m.group(2).lower()
            cells.add(cell)
            (loc_center if kind=="center" else loc_skew)[cell]=c
            continue
        m = VEL_BEAM_RE.match(c)
        if m:
            cell = int(m.group(1)); beam = int(m.group(2))
            cells.add(cell)
            vel_cols[(cell, beam)] = c
            continue
        m = SNR_BEAM_RE.match(c)
        if m:
            cell = int(m.group(1)); beam = int(m.group(2))
            cells.add(cell)
            snr_cols[(cell, beam)] = c
            continue

    return sorted(cells), loc_center, loc_skew, vel_cols, snr_cols

def merge_vel_snr(df_vel: pd.DataFrame, df_snr: pd.DataFrame) -> pd.DataFrame:
    # prefer Sample Number; fallback Sample Time
    if "Sample Number" in df_vel.columns and "Sample Number" in df_snr.columns:
        df = pd.merge(df_vel, df_snr, on="Sample Number", suffixes=("", "_snr"), how="inner")
        return df
    if "Sample Time" in df_vel.columns and "Sample Time" in df_snr.columns:
        df = pd.merge(df_vel, df_snr, on="Sample Time", suffixes=("", "_snr"), how="inner")
        return df
    raise ValueError("Kan VEL en SNR niet mergen: geen gemeenschappelijke sleutel (Sample Number/Sample Time).")

def distances_for_beam(df: pd.DataFrame, cells, beam):
    d=[]
    for cell in cells:
        if beam in (1,2):
            col = f"Cell{cell} Location (Center) (m)"
        else:
            col = f"Cell{cell} Location (Skew) (m)"
        if col in df.columns:
            val = pd.to_numeric(df[col].iloc[0], errors="coerce")
        else:
            val = np.nan
        d.append(val)
    return np.array(d, dtype=float)

def snr_matrix(df: pd.DataFrame, cells, snr_cols, beam):
    cols=[]
    for cell in cells:
        key=(cell, beam)
        if key in snr_cols:
            cols.append(snr_cols[key])
    if not cols:
        return None, None, None
    # figure out which cells we actually have
    cells_have=[int(SNR_BEAM_RE.match(c).group(1)) for c in cols]
    d = distances_for_beam(df, cells_have, beam)
    M = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy()  # time x bins
    return np.array(cells_have, dtype=int), d, M

def vel_matrix(df: pd.DataFrame, cells, vel_cols, beam):
    cols=[]
    for cell in cells:
        key=(cell, beam)
        if key in vel_cols:
            cols.append(vel_cols[key])
    if not cols:
        return None, None, None
    cells_have=[int(VEL_BEAM_RE.match(c).group(1)) for c in cols]
    d = distances_for_beam(df, cells_have, beam)
    M = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy()
    return np.array(cells_have, dtype=int), d, M

def detect_cutoff_from_snr(distances, snr_profile, z=3.0):
    """
    Detecteer "plotse verhoging" in SNR-profiel vs afstand via dSNR/dz threshold (robust MAD).
    cutoff = afstand waar dSNR/dz voor het eerst boven drempel komt.
    """
    if distances is None or snr_profile is None:
        return np.nan
    d = np.array(distances, dtype=float)
    p = np.array(snr_profile, dtype=float)
    mask = np.isfinite(d) & np.isfinite(p)
    d = d[mask]; p = p[mask]
    if len(d) < 6:
        return np.nan

    order = np.argsort(d)
    d = d[order]; p = p[order]

    # median smoothing
    k=5
    ps=[]
    for i in range(len(p)):
        a=max(0,i-k//2); b=min(len(p), i+k//2+1)
        slice_p = p[a:b]
        if np.any(np.isfinite(slice_p)):
            ps.append(np.nanmedian(slice_p))
        else:
            ps.append(np.nan)
    ps=np.array(ps)

    dp = np.diff(ps)  # per bin (4 cm)
    if np.any(np.isfinite(dp)):
        med = np.nanmedian(dp)
        mad = np.nanmedian(np.abs(dp - med)) + 1e-12
    else:
        med = np.nan
        mad = 1e-12
    thresh = med + z*1.4826*mad

    start = min(3, len(dp)-1)
    for i in range(start, len(dp)):
        if dp[i] > thresh:
            return float(d[i+1])
    return float(np.nanmax(d))

def plot_snr_heatmaps(self, time, per_beam, cutoff_by_beam):
    """
    per_beam[beam] = (dists, M) with time x bins
    """
    for beam in sorted(per_beam.keys()):
        d, M = per_beam[beam]
        if M is None or d is None or M.size==0:
            continue
        order = np.argsort(d)
        d_sorted = d[order]
        M_sorted = M[:, order]

        fig = plt.figure(figsize=(11,4))
        ax = fig.add_subplot(1,1,1)
        
        # Set extent to show up to cutoff
        ymin = float(np.nanmin(d_sorted)) if np.any(np.isfinite(d_sorted)) else 0
        ymax = cutoff_by_beam.get(beam, np.nan)
        if not np.isfinite(ymax):
            ymax = float(np.nanmax(d_sorted)) if np.any(np.isfinite(d_sorted)) else 1
        
        im = ax.imshow(
            M_sorted.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[0, len(time)-1, ymin, ymax],
            cmap='viridis'
        )
        ax.set_title(f"SNR (dB) – Beam {beam} (Click to set cutoff)")
        ax.set_xlabel("Tijd-index (zie ticklabels)")
        ax.set_ylabel("Afstand (m)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("SNR (dB)")

        cutoff = cutoff_by_beam.get(beam, np.nan)
        line, = ax.plot([0, len(time)-1], [cutoff, cutoff], 'r--', linewidth=2, label='Cutoff')

        nticks=6
        idx = np.linspace(0, len(time)-1, nticks).astype(int)
        ax.set_xticks(idx)
        ax.set_xticklabels([str(time[i])[:19] for i in idx], rotation=20, ha="right")

        def onclick(event):
            if event.inaxes == ax and event.button == 1:  # left click
                cutoff_new = event.ydata
                self.max_dist[beam].set(cutoff_new)
                line.set_ydata([cutoff_new, cutoff_new])
                fig.canvas.draw()
                self._log(f"Cutoff for beam {beam} set to {cutoff_new:.3f} m")

        fig.canvas.mpl_connect('button_press_event', onclick)

        fig.tight_layout()

def horizontal_speed_from_beams(b1, b2, b3, b4, theta_deg=25.0):
    theta = math.radians(theta_deg)
    sin_t = math.sin(theta) + 1e-12
    u = (b1 - b2) / (2.0*sin_t)
    v = (b4 - b3) / (2.0*sin_t)
    return np.sqrt(u*u + v*v)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SonTek IQ – SNR per beam + cutoff + gemiddelde snelheid (VELBEAM + SNR)")
        self.geometry("1040x700")

        self.vel_path = tk.StringVar(value="")
        self.snr_path = tk.StringVar(value="")
        self.pressure_path = tk.StringVar(value="")
        self.time_col = tk.StringVar(value="Sample Time")
        self.theta_deg = tk.DoubleVar(value=25.0)

        self.beam_on = {b: tk.BooleanVar(value=True) for b in (1,2,3,4)}
        self.max_dist = {b: tk.DoubleVar(value=np.nan) for b in (1,2,3,4)}
        self.hmin = tk.DoubleVar(value=0.0)
        self.hmax = tk.DoubleVar(value=999.0)
        self.speed_mode = tk.StringVar(value="Resultant")
        self.auto_cutoff = tk.BooleanVar(value=True)

        self.df = None
        self.cells = []
        self.loc_center = {}
        self.loc_skew = {}
        self.vel_cols = {}
        self.snr_cols = {}

        self._build()

    def _build(self):
        pad=6
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, padx=10, pady=10)

        # File selectors
        r1=ttk.LabelFrame(frm, text="Input files"); r1.pack(fill="x", pady=pad)

        rr=ttk.Frame(r1); rr.pack(fill="x", pady=3)
        ttk.Label(rr, text="VELBEAM CSV:").pack(side="left")
        ttk.Entry(rr, textvariable=self.vel_path, width=92).pack(side="left", padx=pad)
        ttk.Button(rr, text="Browse...", command=self.browse_vel).pack(side="left")

        rr=ttk.Frame(r1); rr.pack(fill="x", pady=3)
        ttk.Label(rr, text="SNR CSV:").pack(side="left")
        ttk.Entry(rr, textvariable=self.snr_path, width=96).pack(side="left", padx=pad)
        ttk.Button(rr, text="Browse...", command=self.browse_snr).pack(side="left")

        rr=ttk.Frame(r1); rr.pack(fill="x", pady=3)
        ttk.Label(rr, text="PRESSURE CSV:").pack(side="left")
        ttk.Entry(rr, textvariable=self.pressure_path, width=90).pack(side="left", padx=pad)
        ttk.Button(rr, text="Browse...", command=self.browse_pressure).pack(side="left")

        # Load row
        r2=ttk.Frame(frm); r2.pack(fill="x", pady=pad)
        ttk.Button(r2, text="Laad & auto-cutoff", command=self.on_load).pack(side="left")
        ttk.Checkbutton(r2, text="Auto cutoff", variable=self.auto_cutoff).pack(side="left", padx=(10,0))
        ttk.Label(r2, text="Tijdkolom:").pack(side="left", padx=(18,6))
        self.time_combo = ttk.Combobox(r2, textvariable=self.time_col, width=40, values=["Sample Time"])
        self.time_combo.pack(side="left")

        # Beam + cutoff
        r3=ttk.LabelFrame(frm, text="Beams + max bruikbare afstand (m)"); r3.pack(fill="x", pady=pad)
        for b in (1,2,3,4):
            row=ttk.Frame(r3); row.pack(fill="x", pady=2)
            ttk.Checkbutton(row, text=f"Beam {b}", variable=self.beam_on[b]).pack(side="left", padx=pad)
            ttk.Label(row, text="Max afstand (m):").pack(side="left")
            ttk.Entry(row, textvariable=self.max_dist[b], width=10).pack(side="left", padx=pad)

        # Height + theta
        r4=ttk.LabelFrame(frm, text="Selectie voor gemiddelde snelheid"); r4.pack(fill="x", pady=pad)
        row=ttk.Frame(r4); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Min afstand (m):").pack(side="left", padx=pad)
        ttk.Entry(row, textvariable=self.hmin, width=10).pack(side="left")
        ttk.Label(row, text="Max afstand (m):").pack(side="left", padx=pad)
        ttk.Entry(row, textvariable=self.hmax, width=10).pack(side="left")
        ttk.Label(row, text="Beam angle θ (deg):").pack(side="left", padx=pad)
        ttk.Entry(row, textvariable=self.theta_deg, width=10).pack(side="left")
        ttk.Label(row, text="Speed mode:").pack(side="left", padx=pad)
        self.speed_mode_cb = ttk.Combobox(row, textvariable=self.speed_mode, values=["Resultant", "Beam 1", "Beam 2", "Beam 3", "Beam 4", "Beam 1+2"], width=12)
        self.speed_mode_cb.pack(side="left")

        # Actions
        r5=ttk.Frame(frm); r5.pack(fill="x", pady=pad)
        ttk.Button(r5, text="Plot SNR heatmaps", command=self.on_plot_snr).pack(side="left")
        ttk.Button(r5, text="Set cutoff manually", command=self.on_set_cutoff_manual).pack(side="left", padx=pad)
        ttk.Button(r5, text="Herwerk: gemiddelde horizontale snelheid", command=self.on_recalc).pack(side="left")

        # Output
        out=ttk.LabelFrame(frm, text="Log / resultaat"); out.pack(fill="both", expand=True, pady=pad)
        self.txt=tk.Text(out, height=16, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=8, pady=8)
        self._log("Selecteer VELBEAM + SNR CSV (+ optioneel PRESSURE CSV) en klik 'Laad & auto-cutoff'.")

    def _log(self, msg):
        self.txt.insert("end", msg+"\n")
        self.txt.see("end")

    def browse_vel(self):
        fp=filedialog.askopenfilename(title="Selecteer VELBEAM CSV", filetypes=[("CSV","*.csv"),("All","*.*")])
        if fp: self.vel_path.set(fp)

    def browse_snr(self):
        fp=filedialog.askopenfilename(title="Selecteer SNR CSV", filetypes=[("CSV","*.csv"),("All","*.*")])
        if fp: self.snr_path.set(fp)

    def browse_pressure(self):
        fp=filedialog.askopenfilename(title="Selecteer PRESSURE CSV", filetypes=[("CSV","*.csv"),("All","*.*")])
        if fp: self.pressure_path.set(fp)

    def on_load(self):
        vp=self.vel_path.get().strip()
        sp=self.snr_path.get().strip()
        pp=self.pressure_path.get().strip()
        if not vp or not sp:
            messagebox.showerror("Fout","Selecteer zowel VELBEAM als SNR CSV.")
            return
        try:
            df_vel=pd.read_csv(vp)
            df_snr=pd.read_csv(sp)
            df=merge_vel_snr(df_vel, df_snr)
            self.pressure_df = pd.read_csv(pp) if pp else None
        except Exception as e:
            messagebox.showerror("Fout", f"Kon files niet laden/mergen:\n{e}")
            return

        self.df=df
        # time col
        try:
            tcol=find_time_column(df)
            self.time_col.set(tcol)
        except Exception:
            pass
        self.time_combo["values"]=list(df.columns)

        self.cells, self.loc_center, self.loc_skew, self.vel_cols, self.snr_cols = parse_cells(df)

        self._log(f"Geladen & gemerged.")
        self._log(f"Rijen: {len(df)}; kolommen: {len(df.columns)}")
        self._log(f"Cellen: {min(self.cells)}..{max(self.cells)} (n={len(self.cells)})")
        self._log(f"Velocity cols: {len(self.vel_cols)}; SNR cols: {len(self.snr_cols)}")

        # Auto cutoff per beam based on SNR
        tcol=self.time_col.get().strip()
        time=pd.to_datetime(df[tcol], errors="coerce", dayfirst=True)
        time=time.ffill().bfill()

        if self.auto_cutoff.get():
            self._log("Auto cutoff per beam op basis van 'plotse verhoging' in SNR-profiel...")
            max_d_all=[]
            for b in (1,2,3,4):
                cells_b, d, M = snr_matrix(df, self.cells, self.snr_cols, b)
                if M is None:
                    self._log(f"  Beam {b}: geen SNR data gevonden.")
                    continue
                prof=np.array([np.nanmedian(M[:, i]) if np.any(np.isfinite(M[:, i])) else np.nan for i in range(M.shape[1])])  # median over time per bin
                cutoff=detect_cutoff_from_snr(d, prof, z=3.0)
                if np.isfinite(cutoff):
                    self.max_dist[b].set(round(cutoff, 3))
                    self._log(f"  Beam {b}: auto cutoff ~ {cutoff:.3f} m")
                if d is not None and np.isfinite(d).any():
                    max_d_all.append(float(np.nanmax(d)))

            if max_d_all:
                self.hmax.set(round(float(np.nanmax(max_d_all)), 3))

    def on_plot_snr(self):
        if self.df is None:
            messagebox.showerror("Fout","Laad eerst de files.")
            return
        df=self.df
        tcol=self.time_col.get().strip()
        time=pd.to_datetime(df[tcol], errors="coerce", dayfirst=True)
        time=time.ffill().bfill()

        per_beam={}
        cutoff_by_beam={}
        for b in (1,2,3,4):
            if not self.beam_on[b].get():
                continue
            _, d, M = snr_matrix(df, self.cells, self.snr_cols, b)
            if M is None:
                continue
            per_beam[b]=(d, M)
            cutoff_by_beam[b]=float(self.max_dist[b].get()) if np.isfinite(self.max_dist[b].get()) else np.nan

        if not per_beam:
            messagebox.showerror("Fout","Geen SNR matrices gevonden om te plotten.")
            return
        plot_snr_heatmaps(self, time.to_numpy(), per_beam, cutoff_by_beam)
        plt.show()

        # Plot pressure if available
        if self.pressure_df is not None:
            try:
                p_col = find_pressure_column(self.pressure_df)
                pressure = pd.to_numeric(self.pressure_df[p_col], errors="coerce")
                fig = plt.figure(figsize=(11,4))
                ax = fig.add_subplot(1,1,1)
                ax.plot(time, pressure)
                ax.set_title("Ruwe pressure sensor")
                ax.set_xlabel("Tijd")
                ax.set_ylabel("Pressure (dbar)")
                fig.tight_layout()
            except Exception as e:
                self._log(f"Kon pressure niet plotten: {e}")

        plt.show()

    def on_recalc(self):
        if self.df is None:
            messagebox.showerror("Fout","Laad eerst de files.")
            return
        df=self.df
        tcol=self.time_col.get().strip()
        time=pd.to_datetime(df[tcol], errors="coerce", dayfirst=True)

        hmin=float(self.hmin.get()); hmax=float(self.hmax.get())
        theta=float(self.theta_deg.get())

        # Select cells in height-range using available location columns from parse_cells
        selected_cells=[]
        for cell in self.cells:
            loc_center=self.loc_center.get(cell)
            loc_skew=self.loc_skew.get(cell)
            d=np.nan
            if loc_center and loc_center in df.columns:
                d=pd.to_numeric(df[loc_center].iloc[0], errors="coerce")
            elif loc_skew and loc_skew in df.columns:
                d=pd.to_numeric(df[loc_skew].iloc[0], errors="coerce")
            if np.isfinite(d) and (d >= hmin) and (d <= hmax):
                selected_cells.append(cell)

        if not selected_cells:
            messagebox.showerror("Fout","Geen cellen binnen de gekozen afstand/hoogte-range.")
            return

        def beam_bin_matrix(beam):
            # build matrix for selected cells, with per-beam cutoff NaN-ing
            cols=[]
            dists=[]
            for cell in selected_cells:
                key=(cell, beam)
                if key not in self.vel_cols:
                    continue
                cols.append(self.vel_cols[key])
                # distance for this beam
                loc=None
                if beam in (1,2):
                    loc=self.loc_center.get(cell) or self.loc_skew.get(cell)
                else:
                    loc=self.loc_skew.get(cell) or self.loc_center.get(cell)
                if loc and loc in df.columns:
                    d=pd.to_numeric(df[loc].iloc[0], errors="coerce")
                else:
                    d=np.nan
                dists.append(float(d) if np.isfinite(d) else np.nan)

            if not cols:
                return None
            M=np.array(df[cols].apply(pd.to_numeric, errors="coerce"), dtype=float, copy=True)  # time x bins
            dists=np.array(dists, dtype=float)
            cutoff=float(self.max_dist[beam].get())
            if np.isfinite(cutoff):
                mask = dists <= cutoff
                M[:, ~mask] = np.nan
            return M

        B1 = beam_bin_matrix(1) if self.beam_on[1].get() else None
        B2 = beam_bin_matrix(2) if self.beam_on[2].get() else None
        B3 = beam_bin_matrix(3) if self.beam_on[3].get() else None
        B4 = beam_bin_matrix(4) if self.beam_on[4].get() else None

        def mean_over_bins(M):
            if M is None:
                return None
            return np.nanmean(M, axis=1)

        b1 = mean_over_bins(B1)
        b2 = mean_over_bins(B2)
        b3 = mean_over_bins(B3)
        b4 = mean_over_bins(B4)

        sin_t = math.sin(math.radians(theta)) + 1e-12
        u=None; v=None
        if (b1 is not None) and (b2 is not None):
            u = (b1 - b2)/2.0
        if (b4 is not None) and (b3 is not None):
            v = (b4 - b3)/2.0

        mode = self.speed_mode.get()
        if mode == "Resultant":
            if u is None and v is None:
                messagebox.showerror("Fout","Onvoldoende beams: kies minstens (1&2) of (3&4).")
                return

            if u is None:
                speed = np.abs(v)
            elif v is None:
                speed = np.abs(u)
            else:
                speed = np.sqrt(u*u + v*v)
            speed_label = "horizontale snelheid"
        elif mode == "Beam 1+2":
            if b1 is not None and b2 is not None:
                speed = (b1 - b2) / (2 * sin_t)
            else:
                messagebox.showerror("Fout", "Beam 1 en 2 niet beschikbaar.")
                return
            speed_label = "snelheid beam 1+2"
        else:
            beam = int(mode.split()[-1])
            if beam == 1 and b1 is not None:
                speed = b1
            elif beam == 2 and b2 is not None:
                speed = b2
            elif beam == 3 and b3 is not None:
                speed = b3
            elif beam == 4 and b4 is not None:
                speed = b4
            else:
                messagebox.showerror("Fout", f"Beam {beam} niet beschikbaar.")
                return
            speed_label = f"snelheid beam {beam}"

        mean_speed=float(np.nanmean(speed))

        self._log("---- Herwerk ----")
        self._log(f"Afstand-range: {hmin:.3f} .. {hmax:.3f} m (cellen n={len(selected_cells)})")
        self._log(f"θ = {theta:.2f}°")
        self._log(f"Gemiddelde {speed_label}: {mean_speed:.4f} m/s")

        fig=plt.figure(figsize=(11,4))
        ax=fig.add_subplot(1,1,1)
        ax.plot(time, speed, label=speed_label)
        ax.set_title(f"{speed_label.capitalize()} – geselecteerde beams + hoogte-range + cutoff")
        ax.set_xlabel("Tijd")
        ax.set_ylabel("Snelheid (m/s)")

        # Add pressure if available
        if self.pressure_df is not None:
            try:
                p_col = find_pressure_column(self.pressure_df)
                pressure = pd.to_numeric(self.pressure_df[p_col], errors="coerce")
                ax2 = ax.twinx()
                ax2.plot(time, pressure, 'r-', linewidth=1, label='Pressure')
                ax2.set_ylabel('Pressure (dbar)', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
            except Exception as e:
                self._log(f"Kon pressure niet toevoegen: {e}")

        # Add observed velocity if available
        if self.pressure_df is not None:
            try:
                vel_col = find_velocity_column(self.pressure_df)
                observed_vel = pd.to_numeric(self.pressure_df[vel_col], errors="coerce")
                ax.plot(time, observed_vel, 'g-', linewidth=1, label='Observed X Velocity')
                ax.legend()
            except Exception as e:
                self._log(f"Kon observed velocity niet toevoegen: {e}")

        fig.tight_layout()
        plt.show()

    def on_set_cutoff_manual(self):
        if self.df is None:
            messagebox.showerror("Fout", "Laad eerst de files.")
            return
        
        # Create a dialog to input cutoffs
        dialog = tk.Toplevel(self)
        dialog.title("Set cutoffs manually")
        dialog.geometry("300x200")
        
        entries = {}
        for b in (1,2,3,4):
            row = ttk.Frame(dialog)
            row.pack(fill="x", pady=5)
            ttk.Label(row, text=f"Beam {b} cutoff (m):").pack(side="left")
            entries[b] = ttk.Entry(row, width=10)
            entries[b].pack(side="left")
            entries[b].insert(0, str(self.max_dist[b].get()))
        
        def ok():
            for b in (1,2,3,4):
                try:
                    val = float(entries[b].get())
                    self.max_dist[b].set(val)
                except ValueError:
                    pass
            dialog.destroy()
            self._log("Manual cutoffs set.")
        
        ttk.Button(dialog, text="OK", command=ok).pack(pady=10)

def main():
    App().mainloop()

if __name__ == "__main__":
    main()
