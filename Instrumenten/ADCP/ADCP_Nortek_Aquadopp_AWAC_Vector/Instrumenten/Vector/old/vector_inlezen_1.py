import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from datetime import datetime

class VectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Nortek Vector Data Reader')
        self.geometry('600x450')

        self.df = None
        self.timestamps = []

        # GUI
        ttk.Button(self, text='Load Working Directory', command=self.load_directory).pack(pady=5)
        ttk.Button(self, text='Load SEN & DAT Files', command=self.load_files).pack(pady=5)

        ttk.Label(self, text='Parameter:').pack()
        self.param_cb = ttk.Combobox(self, values=['U','V','W','Vel'], state='readonly')
        self.param_cb.current(3)
        self.param_cb.pack()

        ttk.Label(self, text='Start Time (YYYY-MM-DD HH:MM:SS):').pack()
        self.start_entry = ttk.Entry(self, width=30)
        self.start_entry.pack()

        ttk.Label(self, text='End Time (YYYY-MM-DD HH:MM:SS):').pack()
        self.end_entry = ttk.Entry(self, width=30)
        self.end_entry.pack()

        ttk.Label(self, text='Select Range (Slider based on % of data):').pack(pady=3)
        self.slider_frame = ttk.Frame(self)
        self.slider_frame.pack()

        self.start_slider = tk.Scale(self.slider_frame, from_=0, to=100, orient='horizontal', label='Start %', command=self.update_start_entry)
        self.start_slider.pack(side='left', padx=5)

        self.end_slider = tk.Scale(self.slider_frame, from_=0, to=100, orient='horizontal', label='End %', command=self.update_end_entry)
        self.end_slider.set(100)
        self.end_slider.pack(side='left', padx=5)

        ttk.Button(self, text='Plot Time Series', command=self.plot_timeseries).pack(pady=5)
        ttk.Button(self, text='Plot Colormap', command=self.plot_colormap).pack(pady=5)

    def load_directory(self):
        d = filedialog.askdirectory()
        if d:
            os.chdir(d)
            messagebox.showinfo('Directory', f'Working directory set to:\n{d}')

    def load_files(self):
        try:
            sen_file = filedialog.askopenfilename(title='Select .SEN file', filetypes=[('SEN files','*.sen')])
            if not sen_file: return
            dat_file = filedialog.askopenfilename(title='Select .DAT file', filetypes=[('DAT files','*.dat')])
            if not dat_file: return

            tim_dat = pd.read_csv(sen_file, header=None, delim_whitespace=True, na_values=['NA',''], engine='python')
            tim_dat.columns = [f'V{i+1}' for i in range(tim_dat.shape[1])]
            day = tim_dat[['V1','V2','V3']].astype(int).astype(str).agg('/'.join, axis=1)
            time = tim_dat[['V4','V5','V6']].astype(int).apply(lambda row: f"{row['V4']:02d}:{row['V5']:02d}:{row['V6']:02d}" , axis=1)
            ts = pd.to_datetime(day + ' ' + time, format='%m/%d/%Y %H:%M:%S', utc=True)

            dat_val = pd.read_csv(dat_file, header=None, delim_whitespace=True, na_values=['NA',''], engine='python')
            dat_val.columns = [f'V{i+1}' for i in range(dat_val.shape[1])]

            df = pd.DataFrame({
                'U': dat_val['V3'],
                'V': dat_val['V4'],
                'W': dat_val['V5'],
                'P': dat_val['V15']
            })
            df['Vel'] = np.sqrt(df['U']**2 + df['V']**2 + df['W']**2)
            df.index = ts
            self.df = df.sort_index()
            self.timestamps = self.df.index

            # Init entry fields
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, self.timestamps[0].strftime('%Y-%m-%d %H:%M:%S'))
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, self.timestamps[-1].strftime('%Y-%m-%d %H:%M:%S'))

            messagebox.showinfo('Files Loaded', 'Data loaded successfully.')
        except Exception as e:
            messagebox.showerror('Fout bij laden', str(e))

    def update_start_entry(self, val):
        if self.df is not None:
            idx = int(int(val) / 100 * (len(self.timestamps) - 1))
            start_time = self.timestamps[idx].strftime('%Y-%m-%d %H:%M:%S')
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, start_time)

    def update_end_entry(self, val):
        if self.df is not None:
            idx = int(int(val) / 100 * (len(self.timestamps) - 1))
            end_time = self.timestamps[idx].strftime('%Y-%m-%d %H:%M:%S')
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, end_time)

    def get_time_slice(self):
        if self.df is None:
            messagebox.showwarning('No Data', 'Load data first.')
            return None
        try:
            start = pd.to_datetime(self.start_entry.get(), utc=True)
            end = pd.to_datetime(self.end_entry.get(), utc=True)
            return self.df.loc[start:end]
        except Exception as e:
            messagebox.showerror('Time Parsing Error', str(e))
            return None

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
        sc = plt.scatter(x, y, c=z, marker='s', s=5, cmap='viridis')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M\n%d-%m'))
        plt.xticks(rotation=45)
        plt.xlabel('Time')
        plt.ylabel('Pressure')
        plt.title(f'Colormap of {param} vs Time & Pressure')
        plt.colorbar(sc, label=param)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = VectorApp()
    app.mainloop()
