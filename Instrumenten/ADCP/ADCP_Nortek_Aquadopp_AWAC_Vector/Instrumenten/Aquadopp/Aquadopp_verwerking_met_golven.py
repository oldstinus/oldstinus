#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import *
from tkinter import filedialog, ttk, messagebox


# =====================================================================
# Hulpfunctie: laatste getal uit een tekstregel halen
# =====================================================================
def last_float_in_line(line):
    """
    Zoekt het laatste numerieke stukje in een tekstregel (incl. decimalen)
    en geeft dat als float terug. Geeft None als er niets gevonden wordt.
    """
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line.replace(",", "."))
    if not nums:
        return None
    return float(nums[-1])


# =====================================================================
# 1) HDR INLEZEN
# =====================================================================
def read_hdr(path):
    """
    Leest alle relevante Aquadopp parameters uit een *.hdr file.
    """

    hdr = {
        "n_cells": None,
        "cell_size": None,
        "blanking": None,
        "beam_count": None,
        "pressure": None,
        "wave_enabled": False,
        "wave_fs": None,
        "wave_nsamp": None,
        "wave_cell": None
    }

    with open(path, "r", errors="ignore") as f:
        for line in f:

            if "Number of cells" in line:
                val = last_float_in_line(line)
                if val is not None:
                    hdr["n_cells"] = int(val)

            if "Cell size" in line and "cm" in line:
                # bv. "Cell size   10 cm"
                val = last_float_in_line(line)
                if val is not None:
                    hdr["cell_size"] = val / 100.0  # cm → m

            if "Blanking distance" in line:
                # bv. "Blanking distance 0.10 m"
                val = last_float_in_line(line)
                if val is not None:
                    hdr["blanking"] = val  # al in meter

            if "Number of beams" in line:
                val = last_float_in_line(line)
                if val is not None:
                    hdr["beam_count"] = int(val)

            if "Pressure sensor" in line:
                hdr["pressure"] = ("YES" in line)

            if "Wave measurements" in line:
                if "ENABLED" in line:
                    hdr["wave_enabled"] = True

            if "Wave - Sampling rate" in line:
                val = last_float_in_line(line)
                if val is not None:
                    hdr["wave_fs"] = val

            if "Wave - Number of samples" in line:
                val = last_float_in_line(line)
                if val is not None:
                    hdr["wave_nsamp"] = int(val)

            if "Wave - Cell size" in line:
                val = last_float_in_line(line)
                if val is not None:
                    hdr["wave_cell"] = val  # meter

    # Controle / logging
    print("[HDR INFO]")
    for k, v in hdr.items():
        print(f"  {k}: {v}")

    return hdr


# =====================================================================
# 2) Verticale matrix opbouwen (met cellen uit HDR)
# =====================================================================
def build_vertical_matrices(df, hdr, time_col, press_col):
    """
    Bouwt speed- en directionmatrices op basis van cell count + cell size + blanking.
    """

    df = df.copy()

    # Tijd parser
    t = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    if t.isna().all():
        raise ValueError("Tijdkolom kon niet geparsed worden.")
    df["__time__"] = t
    df = df.sort_values("__time__")
    t = df["__time__"].values

    # Druk → waterhoogte (vereenvoudigd)
    H = pd.to_numeric(df[press_col], errors="coerce").values

    # Hoogtes van cell centers boven instrument
    n = hdr["n_cells"]
    if n is None:
        # fallback: tel kolommen Speed#
        speed_cols = [c for c in df.columns if c.lower().startswith("speed#")]
        n = len(speed_cols)
        print(f"[WAARSCHUWING] n_cells onbekend in HDR, gebruik {n} uit CSV.")

    dz = hdr["cell_size"] if hdr["cell_size"] is not None else 0.1
    blanking = hdr["blanking"] if hdr["blanking"] is not None else 0.0

    heights = blanking + np.arange(n) * dz

    # Kolomnamen in CSV
    speed_cols = [f"Speed#{i+1}" for i in range(n)]
    dir_cols   = [f"Dir#{i+1}"   for i in range(n)]

    # Matrices
    nt = len(df)
    speed_mat = np.full((nt, n), np.nan)
    dir_mat   = np.full((nt, n), np.nan)

    for i, col in enumerate(speed_cols):
        if col in df.columns:
            speed_mat[:, i] = pd.to_numeric(df[col], errors="coerce").values

    for i, col in enumerate(dir_cols):
        if col in df.columns:
            dir_mat[:, i] = pd.to_numeric(df[col], errors="coerce").values

    # Afknippen boven wateroppervlak
    for i in range(nt):
        if np.isnan(H[i]):
            continue
        mask = heights > H[i]
        speed_mat[i, mask] = np.nan
        dir_mat[i, mask] = np.nan

    return t, heights, speed_mat, dir_mat


# =====================================================================
# 3) Tijdreeksen analyseren
# =====================================================================
def analyse_timeseries(df, time_col, speed_col, dir_col, heading_col,
                       pitch_col, roll_col, heave_col):

    df = df.copy()

    # Tijd
    t = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    if t.isna().all():
        raise ValueError("Tijdkolom kon niet geparsed worden.")
    df["__time__"] = t
    df = df.sort_values("__time__")

    def get(col):
        if not col:
            return np.full(len(df), np.nan)
        return pd.to_numeric(df[col], errors="coerce").values

    out = pd.DataFrame()
    out["time"] = df["__time__"]
    out["speed"] = get(speed_col)
    out["dir"] = get(dir_col)
    out["heading"] = get(heading_col)
    out["pitch"] = get(pitch_col)
    out["roll"] = get(roll_col)
    out["heave"] = get(heave_col)

    return out


# =====================================================================
# 4) GUI
# =====================================================================
class AquadoppGUI:
    def __init__(self, master):
        self.master = master
        master.title("Aquadopp analyse – HDR automatisch uit CSV-dir")

        self.dirpath = StringVar()
        self.csvpath = StringVar()
        self.hdrpath = StringVar()

        # Directory (wordt automatisch gezet bij CSV-selectie, maar laat staan als info)
        Label(master, text="Directory:").grid(row=0, column=0, sticky="e")
        Entry(master, textvariable=self.dirpath, width=55, state="readonly").grid(row=0, column=1, sticky="w")

        # CSV-selectie
        Label(master, text="CSV-bestand:").grid(row=1, column=0, sticky="e")
        Entry(master, textvariable=self.csvpath, width=55).grid(row=1, column=1, sticky="w")
        Button(master, text="Browse CSV", command=self.browse_csv).grid(row=1, column=2)

        # HDR (wordt automatisch ingevuld, maar kan je manueel aanpassen indien nodig)
        Label(master, text="HDR-bestand:").grid(row=2, column=0, sticky="e")
        Entry(master, textvariable=self.hdrpath, width=55).grid(row=2, column=1, sticky="w")
        Button(master, text="Browse HDR", command=self.browse_hdr).grid(row=2, column=2)

        Button(master, text="Kolommen laden", command=self.load_cols).grid(row=3, column=0, columnspan=3, pady=5)

        # Drop-down kolommen
        self.dd_time = StringVar()
        self.dd_speed = StringVar()
        self.dd_dir = StringVar()
        self.dd_heading = StringVar()
        self.dd_pitch = StringVar()
        self.dd_roll = StringVar()
        self.dd_heave = StringVar()
        self.dd_press = StringVar()

        row = 4
        labels = [
            ("Tijdkolom", self.dd_time),
            ("Snelheid (één Speed# voor tijdreeks)", self.dd_speed),
            ("Richting (één Dir# voor tijdreeks)", self.dd_dir),
            ("Heading", self.dd_heading),
            ("Pitch", self.dd_pitch),
            ("Roll", self.dd_roll),
            ("Heave", self.dd_heave),
            ("Druk (Pressure)", self.dd_press)
        ]

        self.combos = []
        for txt, var in labels:
            Label(master, text=txt + ":").grid(row=row, column=0, sticky="e")
            cb = ttk.Combobox(master, textvariable=var, width=30)
            cb.grid(row=row, column=1, sticky="w")
            self.combos.append(cb)
            row += 1

        Button(master, text="Analyseer en plot", command=self.run).grid(row=row, column=0, columnspan=3, pady=15)

        self.df = None
        self.hdr = None

    # ------------------------------- auto CSV → dir + hdr --------------------
    def browse_csv(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not f:
            return
        self.csvpath.set(f)

        # directory automatisch
        d = os.path.dirname(f)
        self.dirpath.set(d)

        # probeer bijpassende HDR te vinden: zelfde basename
        base = os.path.splitext(os.path.basename(f))[0]
        candidates = [
            os.path.join(d, base + ".hdr"),
            os.path.join(d, base + ".HDR"),
            os.path.join(d, base + ".hrd"),
            os.path.join(d, base + ".HRD"),
        ]
        hdr_file = None
        for c in candidates:
            if os.path.exists(c):
                hdr_file = c
                break

        # fallback: als er maar één *.hdr in de dir staat, neem die
        if hdr_file is None:
            hdrs = glob.glob(os.path.join(d, "*.hdr")) + glob.glob(os.path.join(d, "*.HDR"))
            if len(hdrs) == 1:
                hdr_file = hdrs[0]

        if hdr_file:
            self.hdrpath.set(hdr_file)
            try:
                self.hdr = read_hdr(hdr_file)
            except Exception as e:
                messagebox.showerror("HDR-fout", f"Kon HDR niet lezen:\n{e}")
        else:
            messagebox.showwarning("HDR niet gevonden",
                                   "Geen bijpassend .hdr-bestand gevonden in dezelfde directory.")

    def browse_hdr(self):
        f = filedialog.askopenfilename(filetypes=[("HDR", "*.hdr;*.HRD"), ("All files", "*.*")])
        if f:
            self.hdrpath.set(f)

    def load_cols(self):
        try:
            self.df = pd.read_csv(self.csvpath.get(), sep=";")
        except Exception as e:
            messagebox.showerror("Fout CSV", str(e))
            return

        # HDR opnieuw inlezen (als pad bestaat)
        hdr_path = self.hdrpath.get()
        if not hdr_path or not os.path.exists(hdr_path):
            messagebox.showerror("HDR ontbreekt", "Geen geldig HDR-bestand gevonden.")
            return

        try:
            self.hdr = read_hdr(hdr_path)
        except Exception as e:
            messagebox.showerror("HDR-fout", str(e))
            return

        cols = list(self.df.columns)
        for cb in self.combos:
            cb["values"] = cols

        # eenvoudige autodetectie
        for c in cols:
            lc = c.lower()
            if ("time" in lc or "date" in lc) and not self.dd_time.get():
                self.dd_time.set(c)
            if "speed#" in lc and not self.dd_speed.get():
                self.dd_speed.set(c)
            if "dir#" in lc and not self.dd_dir.get():
                self.dd_dir.set(c)
            if "press" in lc and not self.dd_press.get():
                self.dd_press.set(c)
            if "heading" in lc and not self.dd_heading.get():
                self.dd_heading.set(c)
            if "pitch" in lc and not self.dd_pitch.get():
                self.dd_pitch.set(c)
            if "roll" in lc and not self.dd_roll.get():
                self.dd_roll.set(c)
            if ("analog" in lc or "heave" in lc) and not self.dd_heave.get():
                self.dd_heave.set(c)

        messagebox.showinfo("OK", "Kolommen en HDR geladen.")

    def run(self):
        if self.df is None or self.hdr is None:
            messagebox.showerror("Fout", "Geen data of HDR geladen. Klik eerst op 'Kolommen laden'.")
            return

        try:
            ts = analyse_timeseries(
                self.df,
                self.dd_time.get(),
                self.dd_speed.get(),
                self.dd_dir.get(),
                self.dd_heading.get(),
                self.dd_pitch.get(),
                self.dd_roll.get(),
                self.dd_heave.get()
            )
        except Exception as e:
            messagebox.showerror("Analysefout", str(e))
            return

        # ----------------- Tijdreeksen -----------------
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        ax1, ax2, ax3, ax4 = axes

        ax1.plot(ts["time"], ts["speed"])
        ax1.set_ylabel("Speed (m/s)")
        ax1.set_title("Stroomsnelheid")

        ax2.plot(ts["time"], ts["dir"])
        ax2.set_ylabel("Richting (°)")

        ax3.plot(ts["time"], ts["heading"])
        ax3.set_ylabel("Heading (°)")

        ax4.plot(ts["time"], ts["pitch"], label="Pitch")
        ax4.plot(ts["time"], ts["roll"], label="Roll")
        if not np.all(np.isnan(ts["heave"])):
            ax4.plot(ts["time"], ts["heave"], label="Heave")
        ax4.legend()

        fig.autofmt_xdate()
        plt.tight_layout()

        # ----------------- Verticale kleurplots -----------------
        try:
            t2, heights, Sm, Dm = build_vertical_matrices(
                self.df, self.hdr, self.dd_time.get(), self.dd_press.get()
            )
            tnum = mdates.date2num(t2)
            X, Y = np.meshgrid(tnum, heights)

            fig2, (axS, axD) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

            im1 = axS.pcolormesh(X, Y, Sm.T, shading="auto")
            fig2.colorbar(im1, ax=axS, label="Speed (m/s)")
            axS.set_ylabel("Hoogte boven instrument (m)")
            axS.set_title("Snelheid over de waterkolom")

            im2 = axD.pcolormesh(X, Y, Dm.T, shading="auto")
            fig2.colorbar(im2, ax=axD, label="Dir (°)")
            axD.set_ylabel("Hoogte boven instrument (m)")
            axD.set_xlabel("Tijd")
            axD.set_title("Richting over de waterkolom")

            fig2.autofmt_xdate()
            plt.tight_layout()

        except Exception as e:
            print("Verticale matrix niet mogelijk:", e)

        plt.show()


# =====================================================================
# 5) MAIN
# =====================================================================
if __name__ == "__main__":
    root = Tk()
    app = AquadoppGUI(root)
    root.mainloop()
