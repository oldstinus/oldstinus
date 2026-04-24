import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# Nortek Vector Data Reader
# -------------------------
# GUI to load .sen and .dat files, select parameters and visualize

class VectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Nortek Vector Data Reader')
        self.geometry('400x350')
        
        # Data placeholders
        self.df = None

        # GUI elements
        ttk.Button(self, text='Load Working Directory', command=self.load_directory).pack(pady=5)
        ttk.Button(self, text='Load SEN & DAT Files', command=self.load_files).pack(pady=5)

        ttk.Label(self, text='Parameter:').pack(pady=2)
        self.param_cb = ttk.Combobox(self, values=['U','V','W','Vel'], state='readonly')
        self.param_cb.current(3)
        self.param_cb.pack(pady=2)

        ttk.Label(self, text='Start Time (YYYY-MM-DD HH:MM:SS):').pack(pady=2)
        self.start_entry = ttk.Entry(self)
        self.start_entry.pack(pady=2)
        ttk.Label(self, text='End Time (YYYY-MM-DD HH:MM:SS):').pack(pady=2)
        self.end_entry = ttk.Entry(self)
        self.end_entry.pack(pady=2)

        ttk.Button(self, text='Plot Time Series', command=self.plot_timeseries).pack(pady=5)
        ttk.Button(self, text='Plot Colormap', command=self.plot_colormap).pack(pady=5)

    def load_directory(self):
        d = filedialog.askdirectory()
        if d:
            os.chdir(d)
            messagebox.showinfo('Directory', f'Working directory set to:\n{d}')

    def load_files(self):
        # .sen file
        sen_file = filedialog.askopenfilename(filetypes=[('SEN files','*.sen'),('All files','*.*')])
        if not sen_file:
            return
        tim_dat = pd.read_csv(sen_file, header=None, sep=r'\s+', na_values=['NA',''], engine='python')
        tim_dat.columns = [f'V{i+1}' for i in range(tim_dat.shape[1])]
        # Extract date and time columns
        day = tim_dat[['V1','V2','V3']].astype(int).astype(str).agg('/'.join, axis=1)
        time = tim_dat[['V4','V5','V6']].astype(int).apply(
            lambda row: f"{row['V4']:02d}:{row['V5']:02d}:{row['V6']:02d}", axis=1)
        ts_all = pd.to_datetime(day + ' ' + time, format='%m/%d/%Y %H:%M:%S', utc=True)

        # .dat file
        dat_file = filedialog.askopenfilename(filetypes=[('DAT files','*.dat'),('All files','*.*')])
        if not dat_file:
            return
        dat_val = pd.read_csv(dat_file, header=None, sep=r'\s+', na_values=['NA',''], engine='python')
        dat_val.columns = [f'V{i+1}' for i in range(dat_val.shape[1])]

        # Align lengths if mismatch
        n_time = len(ts_all)
        n_data = len(dat_val)
        if n_time != n_data:
            min_len = min(n_time, n_data)
            ts = ts_all.iloc[:min_len].reset_index(drop=True)
            dat_val = dat_val.iloc[:min_len].reset_index(drop=True)
        else:
            ts = ts_all

        # Build DataFrame
        df = pd.DataFrame({
            'U': dat_val['V3'],
            'V': dat_val['V4'],
            'W': dat_val['V5'],
            'P': dat_val['V15']
        })
        df['Vel'] = np.sqrt(df['U']**2 + df['V']**2)
        df.index = ts
        self.df = df
        messagebox.showinfo('Files Loaded', f'Data loaded successfully. \nRecords: {len(df)}')

    def get_time_slice(self):
        if self.df is None:
            messagebox.showwarning('No Data', 'Load data first.')
            return None
        try:
            start = pd.to_datetime(self.start_entry.get(), utc=True)
            end = pd.to_datetime(self.end_entry.get(), utc=True)
        except Exception as e:
            messagebox.showerror('Time Parsing Error', str(e))
            return None
        return self.df.loc[start:end]

    def plot_timeseries(self):
        df_slice = self.get_time_slice()
        if df_slice is None or df_slice.empty:
            return
        param = self.param_cb.get()
        plt.figure()
        plt.plot(df_slice.index, df_slice[param])
        plt.xlabel('Time')
        plt.ylabel(param)
        plt.title(f'Time Series of {param}')
        plt.tight_layout()
        plt.show()

    def plot_colormap(self):
        df_slice = self.get_time_slice()
        if df_slice is None or df_slice.empty:
            return
        param = self.param_cb.get()
        x = df_slice.index
        y = df_slice['P']
        z = df_slice[param]
        plt.figure()
        sc = plt.scatter(x, y, c=z, marker='s', s=5)
        plt.xlabel('Time')
        plt.ylabel('Pressure')
        plt.title(f'Colormap of {param} vs Time & Pressure')
        plt.colorbar(sc, label=param)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = VectorApp()
    app.mainloop()
