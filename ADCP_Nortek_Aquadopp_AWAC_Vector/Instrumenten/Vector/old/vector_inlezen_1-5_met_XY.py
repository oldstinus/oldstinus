#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Only: .sen/.dat Reader & Plotter met Rolling Average
– Inlezen Nortek-Vector .sen + .dat
– Berekenen Resultant_Speed
– GUI voor parameter-keuze, time-slice, rolling average
– Plot Tijdreeks & Colormap
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.dates import DateFormatter, AutoDateLocator

# --- Logging setup (optioneel) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_vector_data(dat_file: str, sen_file: str) -> pd.DataFrame:
    """Lees .dat en .sen, return DataFrame met datetime-index en Resultant_Speed."""
    dat_cols = [
        'Burst_counter','Ensemble_counter',
        'Velocity_Beam1','Velocity_Beam2','Velocity_Beam3',
        'Amplitude_Beam1','Amplitude_Beam2','Amplitude_Beam3',
        'SNR_Beam1','SNR_Beam2','SNR_Beam3',
        'Correlation_Beam1','Correlation_Beam2','Correlation_Beam3',
        'Pressure','Analog_input1','Analog_input2','Checksum'
    ]
    sen_cols = [
        'Month','Day','Year','Hour','Minute','Second',
        'Error_code','Status_code','Battery_voltage',
        'Soundspeed','Heading','Pitch','Roll','Temperature',
        'Analog_input','Checksum'
    ]

    df_dat = pd.read_csv(dat_file, sep=r'\s+', header=None,
                         names=dat_cols, comment='#', engine='python')
    df_sen = pd.read_csv(sen_file, sep=r'\s+', header=None,
                         names=sen_cols, comment='#', engine='python')

    # Zorg dat ze even lang zijn
    n = min(len(df_dat), len(df_sen))
    df_dat = df_dat.iloc[:n].reset_index(drop=True)
    df_sen = df_sen.iloc[:n].reset_index(drop=True)

    # Bouw datetime-kolom uit sen-data
    df_sen['Datetime'] = pd.to_datetime(df_sen[['Year','Month','Day',
                                                 'Hour','Minute','Second']])
    df_dat['Datetime'] = df_sen['Datetime']

    # Keep alleen volle bursts
    df = df_dat[df_dat['Checksum']==0].copy()

    # Bereken resultant speed
    df['Resultant_Speed'] = np.sqrt(
        df['Velocity_Beam1']**2 +
        df['Velocity_Beam2']**2 +
        df['Velocity_Beam3']**2
    )

    # Zet index op datetime
    return df.set_index('Datetime')


class VectorOnlyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vector Only Reader & Plotter")
        self.geometry("620x500")
        self.df: pd.DataFrame | None = None

        # --- 1) Laadknop ---
        ttk.Button(self, text="Laad .dat + .sen", command=self.cmd_load).pack(pady=8)

        # --- 2) Parameter-keuze ---
        frm1 = ttk.Frame(self); frm1.pack(fill='x', padx=10, pady=4)
        ttk.Label(frm1, text="Parameter:").pack(side='left')
        self.param_cb = ttk.Combobox(frm1, state='readonly')
        self.param_cb.pack(side='left', fill='x', expand=True, padx=5)

        # --- 3) Rolling Average invoer ---
        frm_ma = ttk.Frame(self); frm_ma.pack(fill='x', padx=10, pady=4)
        ttk.Label(frm_ma, text="Rolling avg (n-punten):").pack(side='left')
        self.ma_entry = ttk.Entry(frm_ma, width=5)
        self.ma_entry.insert(0, "1")
        self.ma_entry.pack(side='left', padx=5)

        # --- 4) Time-slice invoer ---
        frm2 = ttk.Frame(self); frm2.pack(fill='x', padx=10, pady=4)
        ttk.Label(frm2, text="Start (YYYY-MM-DD HH:MM:SS):")\
            .grid(row=0, column=0, sticky='w')
        self.start_entry = ttk.Entry(frm2)
        self.start_entry.grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Label(frm2, text="Eind  (YYYY-MM-DD HH:MM:SS):")\
            .grid(row=1, column=0, sticky='w')
        self.end_entry = ttk.Entry(frm2)
        self.end_entry.grid(row=1, column=1, sticky='ew', padx=5)
        frm2.columnconfigure(1, weight=1)

        # --- 5) Sliders (%) ---
        ttk.Label(self, text="Selecteer bereik (%)").pack()
        slf = ttk.Frame(self); slf.pack(fill='x', padx=10)
        self.sldr0 = tk.Scale(slf, from_=0, to=100, orient='horizontal',
                              label='Start %', command=self.on_slider_start)
        self.sldr0.pack(side='left', fill='x', expand=True, padx=5)
        self.sldr1 = tk.Scale(slf, from_=0, to=100, orient='horizontal',
                              label='Eind %', command=self.on_slider_end)
        self.sldr1.set(100); self.sldr1.pack(side='left', fill='x', expand=True, padx=5)

        # --- 6) Plot-knoppen ---
        btnf = ttk.Frame(self); btnf.pack(pady=12)
        ttk.Button(btnf, text="Plot Tijdreeks", command=self.plot_timeseries)\
            .pack(side='left', padx=6)
        ttk.Button(btnf, text="Plot Colormap", command=self.plot_colormap)\
            .pack(side='left', padx=6)

    def cmd_load(self):
        datf = filedialog.askopenfilename(
            title="Selecteer .dat bestand", filetypes=[("DAT","*.dat")])
        if not datf: return
        senf = filedialog.askopenfilename(
            title="Selecteer .sen bestand",
            initialdir=os.path.dirname(datf),
            filetypes=[("SEN","*.sen")])
        if not senf: return

        try:
            self.df = load_vector_data(datf, senf)
        except Exception as e:
            messagebox.showerror("Fout laden data", str(e))
            return

        # Vul parameter-lijst met alle numerieke kolommen
        numerics = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.param_cb['values'] = numerics
        default_idx = numerics.index('Resultant_Speed') if 'Resultant_Speed' in numerics else 0
        self.param_cb.current(default_idx)

        # Vul time-range in
        t0 = self.df.index.min().strftime('%Y-%m-%d %H:%M:%S')
        t1 = self.df.index.max().strftime('%Y-%m-%d %H:%M:%S')
        self.start_entry.delete(0,'end'); self.start_entry.insert(0, t0)
        self.end_entry.delete(0,'end');   self.end_entry.insert(0, t1)
        self.sldr0.set(0); self.sldr1.set(100)

        logger.info(f"Data geladen: {len(self.df)} rijen van {t0} tot {t1}")

    def on_slider_start(self, pct):
        if self.df is None: return
        idx = int(int(pct)/100 * (len(self.df.index)-1))
        ts = self.df.index[idx].strftime('%Y-%m-%d %H:%M:%S')
        self.start_entry.delete(0,'end'); self.start_entry.insert(0, ts)

    def on_slider_end(self, pct):
        if self.df is None: return
        idx = int(int(pct)/100 * (len(self.df.index)-1))
        ts = self.df.index[idx].strftime('%Y-%m-%d %H:%M:%S')
        self.end_entry.delete(0,'end'); self.end_entry.insert(0, ts)

    def get_slice(self) -> pd.DataFrame:
        if self.df is None:
            messagebox.showwarning("Geen data", "Laad eerst je Vector-data.")
            return pd.DataFrame()
        try:
            start = pd.to_datetime(self.start_entry.get())
            end   = pd.to_datetime(self.end_entry.get())
            df2 = self.df.loc[start:end]
            logger.debug(f"Slice: {len(df2)} rijen tussen {start} en {end}")
            return df2
        except Exception as e:
            messagebox.showerror("Slice fout", str(e))
            return pd.DataFrame()

    def _apply_rolling(self, ser: pd.Series) -> pd.Series:
        try:
            n = int(self.ma_entry.get())
            if n > 1:
                return ser.rolling(window=n, min_periods=1, center=True).mean()
        except ValueError:
            messagebox.showwarning("Invalid", "Gebruik een geheel ≥1 voor rolling avg.")
        return ser

    def plot_timeseries(self):
        df2 = self.get_slice()
        if df2.empty: return
        p = self.param_cb.get()
        y = self._apply_rolling(df2[p])
        x = df2.index.to_pydatetime()

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f"Tijdreeks: {p}")
        ax.set_xlabel("Tijd"); ax.set_ylabel(p)
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d-%m"))
        fig.autofmt_xdate()
        plt.show()

    def plot_colormap(self):
        df2 = self.get_slice()
        if df2.empty: return
        p = self.param_cb.get()
        if 'Pressure' not in df2.columns:
            messagebox.showerror("No Pressure", "Voor colormap is Pressure nodig.")
            return

        c = self._apply_rolling(df2[p])
        x = df2.index.to_pydatetime()

        fig, ax = plt.subplots()
        sc = ax.scatter(x, df2['Pressure'], c=c, marker='s', s=8, cmap='viridis')
        ax.set_title(f"Colormap: {p} vs Pressure")
        ax.set_xlabel("Tijd"); ax.set_ylabel("Pressure")
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d-%m"))
        fig.autofmt_xdate()
        fig.colorbar(sc, ax=ax, label=p)
        plt.show()

if __name__ == "__main__":
    app = VectorOnlyApp()
    app.mainloop()
