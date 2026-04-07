import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

class DataPlotterApp:
    def __init__(self, master):
        self.master = master
        master.title("Data Plotter")
        master.geometry("1000x750")

        # DataFrames
        self.df_debiet = None
        self.df_params = None
        self.df_merged = None
        self.df_filtered = None
        self.selection = None  # (start_datetime, end_datetime)

        # Controls frame
        ctrl = tk.Frame(master)
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Buttons
        tk.Button(ctrl, text="Laad Debiet CSV", command=self.load_debiet).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Laad Parameters CSV", command=self.load_params).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Plot Tijdreeks", command=self.plot_time_series).pack(side=tk.LEFT, padx=5)

        # Dropdown voor parameterselectie
        self.selected_param = tk.StringVar(value="Selecteer parameter")
        self.param_menu = tk.OptionMenu(ctrl, self.selected_param, ())
        self.param_menu.pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Plot Scatter", command=self.plot_scatter).pack(side=tk.LEFT, padx=5)

        # Delta filter controls
        tk.Label(ctrl, text="Delta (debiet max):").pack(side=tk.LEFT, padx=(20,5))
        self.delta_entry = tk.Entry(ctrl, width=8)
        self.delta_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Filter Peaks", command=self.apply_filter).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Reset Data", command=self.reset_filter).pack(side=tk.LEFT, padx=5)

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(10,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Add navigation toolbar for save/zoom
        toolbar = NavigationToolbar2Tk(self.canvas, master)
        toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

        # Span selector for data selection
        self.selector = SpanSelector(
            self.ax, self.onselect, direction='horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='blue')
        )

    def load_debiet(self):
        path = filedialog.askopenfilename(
            title="Selecteer Debiet CSV",
            filetypes=[("CSV bestanden", "*.csv"), ("Alle bestanden", "*.*")]
        )
        if not path:
            return
        try:
            df = self._read_debiet(path)
            self.df_debiet = df
            print("=== Eerste 5 rijen Debietdata ===")
            print(df.head())
            messagebox.showinfo("Succes", "Debietdata ingelezen.")
            self.merge_data()
        except Exception as e:
            messagebox.showerror("Fout", f"Kon debietdata niet inlezen:\n{e}")

    def load_params(self):
        path = filedialog.askopenfilename(
            title="Selecteer Parameters CSV",
            filetypes=[("CSV bestanden", "*.csv"), ("Alle bestanden", "*.*")]
        )
        if not path:
            return
        try:
            df = self._read_params(path)
            self.df_params = df
            print("=== Eerste 5 rijen Parametersdata ===")
            print(df.head())
            messagebox.showinfo("Succes", "Parametersdata ingelezen.")
            self.merge_data()
        except Exception as e:
            messagebox.showerror("Fout", f"Kon parametersdata niet inlezen:\n{e}")

    def merge_data(self):
        if self.df_debiet is None or self.df_params is None:
            return
        dfm = self._interpolate_merge(self.df_debiet, self.df_params)
        self.df_merged = dfm
        self.df_filtered = None
        print("=== Eerste 5 rijen Geïntegreerde Data ===")
        print(dfm.head())
        # update dropdown
        menu = self.param_menu['menu']
        menu.delete(0, 'end')
        for c in [c for c in dfm.columns if c != 'debiet']:
            menu.add_command(label=c, command=lambda v=c: self.selected_param.set(v))
        if dfm.columns.size > 1:
            self.selected_param.set(dfm.columns[1])

    def plot_time_series(self):
        df = self.df_filtered if self.df_filtered is not None else self.df_merged
        if df is None:
            messagebox.showerror("Fout", "Laad eerst beide datasets.")
            return
        param = self.selected_param.get()
        self.ax.clear()
        # primary axis debiet
        ax1 = self.ax
        ax1.plot(df.index, df['debiet'], label='Debiet')
        ax1.set_xlabel('Tijd')
        ax1.set_ylabel('Debiet (m³/s)')
        # secondary axis parameter
        ax2 = ax1.twinx()
        if param and param in df.columns:
            ax2.plot(df.index, df[param], label=param, color='orange')
            ax2.set_ylabel(param)
        # formatting
        locator = mdates.AutoDateLocator()
        fmt = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(fmt)
        self.fig.autofmt_xdate()
        # legends
        lines, labels = ax1.get_legend_handles_labels()
        if param and param in df.columns:
            l2, lab2 = ax2.get_legend_handles_labels()
            lines += l2; labels += lab2
        ax1.legend(lines, labels)
        self.canvas.draw()

    def plot_scatter(self):
        df = self.df_filtered if self.df_filtered is not None else self.df_merged
        if df is None:
            messagebox.showerror("Fout", "Laad eerst beide datasets.")
            return
        param = self.selected_param.get()
        if not param or param not in df.columns:
            messagebox.showerror("Fout", "Selecteer een geldige parameter.")
            return
        self.ax.clear()
        pos = df[df['debiet'] >= 0]
        neg = df[df['debiet'] < 0]
        self.ax.scatter(pos['debiet'].abs(), pos[param], label='Positief debiet', alpha=0.7)
        self.ax.scatter(neg['debiet'].abs(), neg[param], label='Negatief debiet', alpha=0.7)
        self.ax.set_xlabel('Absoluut Debiet (m³/s)')
        self.ax.set_ylabel(param)
        self.ax.set_title(f'{param} vs. Absoluut Debiet')
        self.ax.legend()
        self.canvas.draw()

    def apply_filter(self):
        if self.df_merged is None or self.selection is None:
            messagebox.showerror("Fout", "Laad en selecteer eerst data.")
            return
        try:
            delta = float(self.delta_entry.get())
        except ValueError:
            messagebox.showerror("Fout", "Ongeldige delta.")
            return
        start, end = self.selection
        df = self.df_merged.copy()
        mask = (df.index >= start) & (df.index <= end)
        df_region = df.loc[mask]
        df_region = df_region[df_region['debiet'].abs() <= delta]
        df_out = df.loc[~mask]
        self.df_filtered = pd.concat([df_out, df_region]).sort_index()
        self.plot_time_series()

    def reset_filter(self):
        # Reset data and clear selection/filter
        self.df_filtered = None
        self.selection = None
        self.delta_entry.delete(0, tk.END)
        # Clear the plot area
        self.ax.clear()
        self.canvas.draw()

    def onselect(self, xmin, xmax):
        dt_min = mdates.num2date(xmin).replace(tzinfo=None)
        dt_max = mdates.num2date(xmax).replace(tzinfo=None)
        self.selection = (dt_min, dt_max)
        print(f"Geselecteerde periode: {dt_min} - {dt_max}")

    @staticmethod
    def _read_debiet(path):
        try:
            df = pd.read_csv(path, sep='\t', skiprows=15, usecols=['Date','Time','Aggregated mean [m³/s]'])
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep='\t', skiprows=15, usecols=['Date','Time','Aggregated mean [m³/s]'], encoding='latin-1')
        df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Time'])
        df = df.drop(columns=['Date']).rename(columns={'Aggregated mean [m³/s]':'debiet'})
        df.set_index('Time', inplace=True)
        return df.infer_objects()

    @staticmethod
    def _read_params(path):
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        try:
            df = pd.read_csv(path, usecols=cols)
        except UnicodeDecodeError:
            df = pd.read_csv(path, usecols=cols, encoding='latin-1')
        time_col = cols[0]
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col]).rename(columns={time_col:'Time'})
        df.set_index('Time', inplace=True)
        return df.sort_index().infer_objects()

    @staticmethod
    def _interpolate_merge(df1, df2):
        d1 = df1.copy(); d2 = df2.copy()
        f = min(d1.index.to_series().diff().dropna().min(), d2.index.to_series().diff().dropna().min())
        start = max(d1.index.min(), d2.index.min()); end = min(d1.index.max(), d2.index.max())
        idx = pd.date_range(start, end, freq=f)
        d1i = d1.reindex(idx).interpolate(method='time')
        d2i = d2.reindex(idx).interpolate(method='time')
        merged = pd.concat([d1i, d2i], axis=1).dropna(subset=['debiet'])
        return merged.sort_index()

if __name__ == '__main__':
    root = tk.Tk()
    app = DataPlotterApp(root)
    root.mainloop()
