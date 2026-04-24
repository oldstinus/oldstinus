#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import *
from tkinter import filedialog, ttk, messagebox


################################################################
# Analyse: ensembles (dt ≈ 300 s) – stroming + attitude
################################################################
def analyse_ensembles(df,
                      time_col,
                      speed_col=None,
                      dir_col=None,
                      heading_col=None,
                      pitch_col=None,
                      roll_col=None,
                      heave_col=None):
    """
    Maakt één DataFrame met:
      time, speed, dir, heading, pitch, roll, heave
    Kolommen die niet opgegeven zijn, worden NaN.
    """

    df = df.copy()

    # Tijd parsen (dag-maand-jaar)
    t = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    if t.isna().all():
        raise ValueError("Tijdkolom kon niet geparsed worden. Formaat ok?")

    df["__time__"] = t
    df = df.sort_values("__time__")

    out = pd.DataFrame()
    out["time"] = df["__time__"]

    def get_col(name):
        if name is None or name == "":
            return np.full(len(df), np.nan)
        return pd.to_numeric(df[name], errors="coerce").values

    out["speed"]   = get_col(speed_col)
    out["dir_deg"] = get_col(dir_col)
    out["heading"] = get_col(heading_col)
    out["pitch"]   = get_col(pitch_col)
    out["roll"]    = get_col(roll_col)
    out["heave"]   = get_col(heave_col)

    # dt-info (ter controle)
    dt = np.median(np.diff(out["time"]).astype("timedelta64[ns]").astype(float)) / 1e9
    print(f"dt ≈ {dt:.3f} s")

    return out


################################################################
# Functie: maak matrice voor speed/dir over hoogte
################################################################
def build_vertical_matrices(df):
    """
    Zoekt kolommen 'Speed#i(x.xx m)' en 'Dir#i(x.xx m)'.
    Geeft:
      time (array),
      depths (array, m),
      speed_mat [nt x nz],
      dir_mat   [nt x nz]
    """
    cols = list(df.columns)

    # Regex om diepte uit kolomnaam te halen, bv. "Speed#1(0.10m)"
    speed_pattern = re.compile(r"^Speed#\d+\(([\d\.]+)m\)", re.IGNORECASE)
    dir_pattern   = re.compile(r"^Dir#\d+\(([\d\.]+)m\)", re.IGNORECASE)

    speed_cols = []
    dir_cols   = []
    depths_s   = []
    depths_d   = []

    for c in cols:
        m = speed_pattern.match(c)
        if m:
            depth = float(m.group(1))
            speed_cols.append(c)
            depths_s.append(depth)
        m2 = dir_pattern.match(c)
        if m2:
            depth = float(m2.group(1))
            dir_cols.append(c)
            depths_d.append(depth)

    if not speed_cols and not dir_cols:
        return None, None, None, None

    # Sorteren op diepte
    speed_info = sorted(zip(depths_s, speed_cols), key=lambda x: x[0])
    dir_info   = sorted(zip(depths_d, dir_cols),   key=lambda x: x[0])

    depths_s_sorted = [d for d, _ in speed_info]
    speed_cols_sorted = [c for _, c in speed_info]

    depths_d_sorted = [d for d, _ in dir_info]
    dir_cols_sorted = [c for _, c in dir_info]

    # Matrices bouwen (nt x nz)
    ntime = len(df)
    speed_mat = None
    dir_mat   = None

    if speed_cols_sorted:
        speed_mat = df[speed_cols_sorted].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if dir_cols_sorted:
        dir_mat = df[dir_cols_sorted].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Tijd
    # Probeer Date/Time-achtige kolom te vinden (we gebruiken dezelfde heuristiek als GUI)
    time_col = None
    for c in cols:
        if "time" in c.lower() or "date" in c.lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError("Geen tijdkolom gevonden voor vertical plot.")

    t = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")

    return t, np.array(depths_s_sorted), speed_mat, dir_mat


################################################################
# GUI
################################################################
class AquadoppGUI:
    def __init__(self, master):
        self.master = master
        master.title("Aquadopp ensembles – stroming + heave/roll/pitch + verticale kleurplots")

        self.filepath = StringVar()

        # Bestand
        Label(master, text="CSV-bestand (*.csv, ';'-gescheiden):").grid(row=0, column=0, sticky="e")
        Entry(master, textvariable=self.filepath, width=55).grid(row=0, column=1, sticky="w")
        Button(master, text="Browse", command=self.browse).grid(row=0, column=2)

        Button(master, text="Kolommen laden", command=self.load_columns).grid(row=1, column=0, columnspan=3, pady=5)

        # Dropdowns voor kolommen
        self.dd_time    = StringVar()
        self.dd_speed   = StringVar()
        self.dd_dir     = StringVar()
        self.dd_heading = StringVar()
        self.dd_pitch   = StringVar()
        self.dd_roll    = StringVar()
        self.dd_heave   = StringVar()

        row = 2
        Label(master, text="Tijdkolom:").grid(row=row, column=0, sticky="e")
        self.cb_time = ttk.Combobox(master, textvariable=self.dd_time, width=30)
        self.cb_time.grid(row=row, column=1, sticky="w"); row += 1

        Label(master, text="Snelheid (één Speed#X voor tijdreeks):").grid(row=row, column=0, sticky="e")
        self.cb_speed = ttk.Combobox(master, textvariable=self.dd_speed, width=30)
        self.cb_speed.grid(row=row, column=1, sticky="w"); row += 1

        Label(master, text="Richting (één Dir#X voor tijdreeks):").grid(row=row, column=0, sticky="e")
        self.cb_dir = ttk.Combobox(master, textvariable=self.dd_dir, width=30)
        self.cb_dir.grid(row=row, column=1, sticky="w"); row += 1

        Label(master, text="Heading:").grid(row=row, column=0, sticky="e")
        self.cb_heading = ttk.Combobox(master, textvariable=self.dd_heading, width=30)
        self.cb_heading.grid(row=row, column=1, sticky="w"); row += 1

        Label(master, text="Pitch:").grid(row=row, column=0, sticky="e")
        self.cb_pitch = ttk.Combobox(master, textvariable=self.dd_pitch, width=30)
        self.cb_pitch.grid(row=row, column=1, sticky="w"); row += 1

        Label(master, text="Roll:").grid(row=row, column=0, sticky="e")
        self.cb_roll = ttk.Combobox(master, textvariable=self.dd_roll, width=30)
        self.cb_roll.grid(row=row, column=1, sticky="w"); row += 1

        Label(master, text="Heave (bv. Analog1):").grid(row=row, column=0, sticky="e")
        self.cb_heave = ttk.Combobox(master, textvariable=self.dd_heave, width=30)
        self.cb_heave.grid(row=row, column=1, sticky="w"); row += 1

        Button(master, text="Analyseer en plot", command=self.run).grid(row=row, column=0, columnspan=3, pady=10)

        self.df = None

    def browse(self):
        path = filedialog.askopenfilename(
            title="Kies Aquadopp CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.filepath.set(path)

    def load_columns(self):
        try:
            # jouw CSV is ';'-gescheiden
            self.df = pd.read_csv(self.filepath.get(), sep=";")
            cols = list(self.df.columns)

            for cb in [self.cb_time, self.cb_speed, self.cb_dir,
                       self.cb_heading, self.cb_pitch, self.cb_roll, self.cb_heave]:
                cb["values"] = cols

            # eenvoudige auto-detectie
            for c in cols:
                lc = c.lower()
                if "time" in lc or "date" in lc:
                    if not self.dd_time.get():
                        self.dd_time.set(c)
                if "speed#" in lc and not self.dd_speed.get():
                    self.dd_speed.set(c)
                if "dir#" in lc and not self.dd_dir.get():
                    self.dd_dir.set(c)
                if "heading" in lc and not self.dd_heading.get():
                    self.dd_heading.set(c)
                if "pitch" in lc and not self.dd_pitch.get():
                    self.dd_pitch.set(c)
                if "roll" in lc and not self.dd_roll.get():
                    self.dd_roll.set(c)
                if ("analog" in lc or "heave" in lc) and not self.dd_heave.get():
                    self.dd_heave.set(c)

            messagebox.showinfo("OK", "Kolommennamen geladen.")
        except Exception as e:
            messagebox.showerror("Fout bij inlezen CSV", str(e))

    def run(self):
        if self.df is None:
            messagebox.showerror("Fout", "Geen data geladen. Klik eerst op 'Kolommen laden'.")
            return
        try:
            out = analyse_ensembles(
                df=self.df,
                time_col=self.dd_time.get(),
                speed_col=self.dd_speed.get(),
                dir_col=self.dd_dir.get(),
                heading_col=self.dd_heading.get(),
                pitch_col=self.dd_pitch.get(),
                roll_col=self.dd_roll.get(),
                heave_col=self.dd_heave.get()
            )
        except Exception as e:
            messagebox.showerror("Fout tijdens analyse", str(e))
            return

        # ------------- 1) Tijdreeksen -------------
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        ax1, ax2, ax3, ax4 = axes

        # Snelheid
        ax1.plot(out["time"], out["speed"])
        ax1.set_ylabel("Speed (m/s)")
        ax1.set_title("Snelheid per ensemble")
        ax1.grid(True)

        # Richting
        ax2.plot(out["time"], out["dir_deg"])
        ax2.set_ylabel("Dir (°)")
        ax2.set_title("Stromingsrichting (één gekozen cel)")
        ax2.grid(True)

        # Heading
        ax3.plot(out["time"], out["heading"])
        ax3.set_ylabel("Heading (°)")
        ax3.set_title("Instrument heading")
        ax3.grid(True)

        # Pitch / Roll / Heave
        ax4.plot(out["time"], out["pitch"], label="Pitch")
        ax4.plot(out["time"], out["roll"],  label="Roll")
        if not np.all(np.isnan(out["heave"])):
            ax4.plot(out["time"], out["heave"], label="Heave")
        ax4.set_ylabel("° / m")
        ax4.set_title("Pitch / Roll / Heave")
        ax4.grid(True)
        ax4.legend(loc="upper right")

        axes[-1].set_xlabel("Tijd")
        fig.autofmt_xdate()
        plt.tight_layout()

        # ------------- 2) Verticale kleurplots -------------
        try:
            t2, depths, speed_mat, dir_mat = build_vertical_matrices(self.df)
        except Exception as e:
            print("Verticale plot niet mogelijk:", e)
            plt.show()
            return

        if speed_mat is None and dir_mat is None:
            print("Geen Speed#/Dir# kolommen gevonden voor verticale plot.")
            plt.show()
            return

        # T naar matplotlib datums
        tnum = mdates.date2num(t2)

        fig2, (axS, axD) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        if speed_mat is not None:
            # nt x nz → nz x nt voor pcolormesh
            X, Y = np.meshgrid(tnum, depths)
            imS = axS.pcolormesh(X, Y, speed_mat.T, shading="auto")
            axS.set_ylabel("Diepte (m)")
            axS.set_title("Speed over diepte (alle Speed#i)")
            axS.invert_yaxis()
            fig2.colorbar(imS, ax=axS, label="Speed (m/s)")

        if dir_mat is not None:
            X, Y = np.meshgrid(tnum, depths)
            imD = axD.pcolormesh(X, Y, dir_mat.T, shading="auto")
            axD.set_ylabel("Diepte (m)")
            axD.set_title("Richting over diepte (alle Dir#i)")
            axD.invert_yaxis()
            fig2.colorbar(imD, ax=axD, label="Dir (°)")

        axD.set_xlabel("Tijd")
        axD.xaxis_date()
        fig2.autofmt_xdate()
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    root = Tk()
    app = AquadoppGUI(root)
    root.mainloop()
