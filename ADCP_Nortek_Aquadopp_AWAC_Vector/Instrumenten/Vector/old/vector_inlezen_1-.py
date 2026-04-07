import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from datetime import datetime
import importlib.util

class VectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Nortek Vector Data Reader')
        self.geometry('500x480')

        # Paths for custom modules
        self.files_path = None
        self.arrays_path = None
        self.DataFile = None
        self.GenericDataArray = None

        # Data placeholder
        self.df = None

        # --- Module loaders ---
        mod_frame = ttk.LabelFrame(self, text='Modules', padding=10)
        mod_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(mod_frame, text='Load files.py', command=self.load_files_module).pack(side='left', padx=5)
        ttk.Button(mod_frame, text='Load arrays.py', command=self.load_arrays_module).pack(side='left', padx=5)

        # --- Vector file loader ---
        ttk.Button(self, text='Load Vector File', command=self.load_vector_file).pack(pady=10)

        # --- Outlier filter ---
        ttk.Label(self, text='Filter outliers:').pack(pady=2)
        self.filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text='Adaptive Outlier Removal', variable=self.filter_var).pack()

        # --- Parameter selection ---
        ttk.Label(self, text='Parameter:').pack(pady=2)
        self.param_cb = ttk.Combobox(self, values=['U','V','W','Vel'], state='readonly')
        self.param_cb.current(3)
        self.param_cb.pack()

        # --- Time window entries ---
        ttk.Label(self, text='Start Time (YYYY-MM-DD HH:MM:SS):').pack(pady=2)
        self.start_entry = ttk.Entry(self, width=25)
        self.start_entry.pack()
        ttk.Label(self, text='End Time (YYYY-MM-DD HH:MM:SS):').pack(pady=2)
        self.end_entry = ttk.Entry(self, width=25)
        self.end_entry.pack()

        # --- Plot buttons ---
        ttk.Button(self, text='Plot Time Series', command=self.plot_timeseries).pack(pady=5)
        ttk.Button(self, text='Plot Colormap', command=self.plot_colormap).pack(pady=5)

    def load_files_module(self):
        path = filedialog.askopenfilename(title='Select files.py', filetypes=[('Python file','files.py')])
        if path:
            self.files_path = path
            messagebox.showinfo('Module geladen', f'files.py geladen: {path}')

    def load_arrays_module(self):
        path = filedialog.askopenfilename(title='Select arrays.py', filetypes=[('Python file','arrays.py')])
        if path:
            self.arrays_path = path
            messagebox.showinfo('Module geladen', f'arrays.py geladen: {path}')

    def dynamic_import(self, name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def ensure_modules(self):
        if not self.files_path or not self.arrays_path:
            messagebox.showwarning('Modules missen', 'Laad eerst files.py en arrays.py modules.')
            return False
        try:
            files_mod = self.dynamic_import('files_mod', self.files_path)
            arrays_mod = self.dynamic_import('arrays_mod', self.arrays_path)
            self.DataFile = files_mod.DataFile
            self.GenericDataArray = arrays_mod.GenericDataArray
            return True
        except ImportError as e:
            msg = str(e)
            if 'nortek' in msg:
                msg = ("Module 'nortek' ontbreekt. \n" 
                       "Installeer via: pip install nortek-python")
            messagebox.showerror('ImportError', msg)
            return False
        except Exception as e:
            messagebox.showerror('ImportError', str(e))
            return False

    def load_vector_file(self):
        if not self.ensure_modules():
            return
        fpath = filedialog.askopenfilename(title='Select Vector file', filetypes=[('Vector','*.vno;*.dat;*.sen')])
        if not fpath:
            return
        try:
            datafile = self.DataFile(fpath)
        except ImportError as e:
            # Handle missing nortek dependency
            msg = str(e)
            if 'nortek' in msg:
                messagebox.showerror('Dependency fout', 
                                     "Module 'nortek' niet gevonden. Installeer via 'pip install nortek-python'.")
            else:
                messagebox.showerror('Read Error', msg)
            return
        except Exception as e:
            messagebox.showerror('Read Error', str(e))
            return

        vel_array = datafile['velocity']
        if not isinstance(vel_array, self.GenericDataArray):
            messagebox.showerror('Format Error', 'Geen velocity data in bestand.')
            return

        if self.filter_var.get():
            vel_array.adaptiveOutlierRemoval()

        data = vel_array['data']
        u = data[:, :, 0].mean(axis=0)
        v = data[:, :, 1].mean(axis=0)
        w = data[:, :, 2].mean(axis=0)
        vel = np.sqrt(u**2 + v**2)

        times = pd.to_datetime(datafile['time'], unit='D', origin='julian', utc=True)
        min_len = min(len(times), len(u))
        self.df = pd.DataFrame({
            'U': u[:min_len],
            'V': v[:min_len],
            'W': w[:min_len],
            'Vel': vel[:min_len]
        }, index=times[:min_len]).sort_index()
        messagebox.showinfo('Loaded', f'Data geladen: {len(self.df)} records.')

    def get_time_slice(self):
        if self.df is None:
            messagebox.showwarning('No Data', 'Laad eerst data.')
            return None
        try:
            start = pd.to_datetime(self.start_entry.get(), utc=True)
            end = pd.to_datetime(self.end_entry.get(), utc=True)
        except Exception as e:
            messagebox.showerror('Tijd fout', str(e))
            return None
        return self.df.loc[start:end]

    def plot_timeseries(self):
        df_slice = self.get_time_slice()
        if df_slice is None or df_slice.empty:
            return
        param = self.param_cb.get()
        plt.figure()
        plt.plot(df_slice.index, df_slice[param])
        plt.xlabel('Tijd')
        plt.ylabel(param)
        plt.title(f'Tijdserie van {param}')
        plt.tight_layout()
        plt.show()

    def plot_colormap(self):
        df_slice = self.get_time_slice()
        if df_slice is None or df_slice.empty:
            return
        param = self.param_cb.get()
        x = df_slice.index
        y = np.linspace(0, 1, len(df_slice))
        z = df_slice[param]
        plt.figure()
        sc = plt.scatter(x, y, c=z, marker='s', s=8)
        plt.xlabel('Tijd')
        plt.ylabel('Genormaliseerde index')
        plt.title(f'Colormap van {param}')
        plt.colorbar(sc, label=param)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = VectorApp()
    app.mainloop()
