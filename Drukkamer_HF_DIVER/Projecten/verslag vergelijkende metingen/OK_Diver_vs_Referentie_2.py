
import os
import re
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RectangleSelector
from matplotlib.dates import num2date
from datetime import datetime, timedelta
import chardet
from scipy.stats import linregress
import numpy as np

plt.ion()

# ---------- Helpers ----------

def select_directory():
    root = tk.Tk(); root.withdraw()
    return filedialog.askdirectory(title="Selecteer Directory voor export")

def get_offset_input(label):
    root = tk.Tk(); root.withdraw()
    try:
        val = simpledialog.askfloat(f"Voer {label} offset in", f"Voer de {label} offset in:")
        return 0.0 if val is None else float(val)
    except Exception:
        messagebox.showerror("Ongeldige Invoer", "Voer een geldig nummer in.")
        return 0.0

def detect_file_encoding(path):
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        return chardet.detect(raw)['encoding']
    except Exception:
        return 'utf-8'

# ---------- Datum/tijd parsing uit bestandsnaam en eerste regel ----------

DATETIME_PATTERNS = [
    r'(?P<Y>\d{4})[-_/\.]?(?P<m>\d{2})[-_/\.]?(?P<d>\d{2})[ T_-]?(?P<H>\d{2})[:\-]?(?P<M>\d{2})[:\-]?(?P<S>\d{2})',
    r'(?P<d>\d{2})[-_/\.](?P<m>\d{2})[-_/\.](?P<Y>\d{4})[ T_-](?P<H>\d{2})[:\-](?P<M>\d{2})[:\-](?P<S>\d{2})',
    r'(?P<d>\d{2})(?P<m>\d{2})(?P<Y>\d{4})[ T_-]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})',
]

def try_parse_dt_dict(d):
    try:
        return datetime(int(d['Y']), int(d['m']), int(d['d']), int(d['H']), int(d['M']), int(d['S']))
    except Exception:
        return None

def extract_start_datetime_from_text(text):
    if not isinstance(text, str) or not text:
        return None
    for pat in DATETIME_PATTERNS:
        m = re.search(pat, text)
        if m:
            dt = try_parse_dt_dict(m.groupdict())
            if dt:
                return dt
    # fallback brede detectie
    candidates = re.findall(r'[\d/\-:\._ ]{8,}', text)
    for cand in candidates:
        for dayfirst in (True, False):
            try:
                dt = pd.to_datetime(cand.strip().replace('_', ' ').replace('.', ':'), dayfirst=dayfirst, errors='raise')
                if isinstance(dt, pd.Timestamp):
                    return dt.to_pydatetime()
            except Exception:
                continue
    return None

def extract_start_datetime_from_filename_and_firstline(file_path):
    fname = os.path.basename(file_path)
    dt_fname = extract_start_datetime_from_text(fname)

    enc = detect_file_encoding(file_path)
    dt_firstline = None
    try:
        with open(file_path, 'r', encoding=enc, errors='ignore') as f:
            first_line = f.readline().strip()
        dt_firstline = extract_start_datetime_from_text(first_line)
    except Exception as e:
        print(f"Kon eerste regel niet lezen voor {file_path}: {e}")

    return dt_fname if dt_fname else dt_firstline

# ---------- Diver .mon ----------

def read_mon_file(path):
    enc = detect_file_encoding(path)
    lines = []
    try:
        with open(path, 'r', encoding=enc, errors='ignore') as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if 'END OF DATA' in s:
                    break
                if i >= 54:
                    lines.append(s)
    except Exception as e:
        print(f"Fout .mon lezen: {e}")
    return lines

def parse_mon_data(lines, pressure_offset=0.0, time_offset=0.0):
    data = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 3 and "/" in parts[0] and ":" in parts[1]:
            date_str = parts[0] + " " + parts[1]
            try:
                try:
                    dt = datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    dt = datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S")
                dt = dt + timedelta(seconds=time_offset)
                p = float(parts[2].replace(',', '.')) + pressure_offset
                data.append({'Datetime': dt, 'Pressure': p})
            except Exception as e:
                print(f"Parse fout: {ln} -> {e}")
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('Datetime').reset_index(drop=True)
    return df

# ---------- Referentie (CSV: skip 2 lines; col0=seconden; col1=waarde) ----------

def read_reference_csv_seconds_since_start(path):
    """
    Verwacht:
    - CSV met comma delimiter
    - Data start vanaf 3de lijn (skiprows=2)
    - Kolom 0: seconden sinds start
    - Kolom 1: druk (Pa) of gemeten waarde (decimaal komma/punt)
    - Starttijd in bestandsnaam of eerste regel (voorkeur bestandsnaam)
    """
    dt_start = extract_start_datetime_from_filename_and_firstline(path)
    if dt_start is None:
        messagebox.showerror("Starttijd niet gevonden",
                             "Kon geen starttijd vinden in bestandsnaam of eerste regel van het referentiebestand.")
        return pd.DataFrame(columns=['Datetime','Pressure'])

    enc = detect_file_encoding(path)
    try:
        df_raw = pd.read_csv(path, sep=',', header=None, skiprows=2, encoding=enc, engine='python')
    except Exception as e:
        messagebox.showerror("Leesfout", f"Kon referentie CSV niet inlezen: {e}")
        return pd.DataFrame(columns=['Datetime','Pressure'])

    if df_raw.shape[1] < 2:
        messagebox.showerror("Onvoldoende kolommen",
                             "Referentie CSV moet minstens 2 kolommen hebben: seconden, waarde.")
        return pd.DataFrame(columns=['Datetime','Pressure'])

    sec = pd.to_numeric(df_raw.iloc[:, 0], errors='coerce')
    val = pd.to_numeric(df_raw.iloc[:, 1].astype(str).str.replace(',', '.', regex=False), errors='coerce')

    mask = (~sec.isna()) & (~val.isna())
    sec = sec[mask].to_numpy()
    val = val[mask].to_numpy()

    times = [dt_start + timedelta(seconds=float(s)) for s in sec]
    df = pd.DataFrame({'Datetime': times, 'Pressure': val})
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df

# ---------- Samenvatting ----------

def convert_pressure_units(pressure_pa):
    pressure_mmH2O = pressure_pa / 9.80665
    pressure_mmHg = pressure_pa / 133.322
    pressure_bar = pressure_pa / 100000
    return pressure_pa, pressure_mmH2O, pressure_mmHg, pressure_bar

def display_pressure_summary(df_diver, df_reference):
    try:
        # Veiligheidscontroles op lege dataframes
        if df_diver.empty:
            messagebox.showwarning("Geen Diver-data", "Diver-dataset is leeg; samenvatting kan niet worden opgebouwd.")
            return

        div_start = float(df_diver['Pressure'].iloc[0])
        div_end   = float(df_diver['Pressure'].iloc[-1])
        div_max   = float(df_diver['Pressure'].max())

        labels   = ['Diver Start', 'Diver Eind', 'Diver Max']
        pa_vals  = [div_start, div_end, div_max]
        mmh2o    = [convert_pressure_units(v)[1] for v in pa_vals]
        mmhg     = [convert_pressure_units(v)[2] for v in pa_vals]
        bar_vals = [convert_pressure_units(v)[3] for v in pa_vals]

        if not df_reference.empty:
            ref_start = float(df_reference['Pressure'].iloc[0])
            ref_end   = float(df_reference['Pressure'].iloc[-1])
            ref_max   = float(df_reference['Pressure'].max())

            labels  += ['Referentie Start', 'Referentie Eind', 'Referentie Max']
            pa_vals += [ref_start, ref_end, ref_max]
            mmh2o   += [convert_pressure_units(ref_start)[1],
                        convert_pressure_units(ref_end)[1],
                        convert_pressure_units(ref_max)[1]]
            mmhg    += [convert_pressure_units(ref_start)[2],
                        convert_pressure_units(ref_end)[2],
                        convert_pressure_units(ref_max)[2]]
            bar_vals+= [convert_pressure_units(ref_start)[3],
                        convert_pressure_units(ref_end)[3],
                        convert_pressure_units(ref_max)[3]]

        # Controle: alle lijsten even lang
        n = len(labels)
        assert all(len(lst) == n for lst in [pa_vals, mmh2o, mmhg, bar_vals]), "Kolomlengtes inconsistent"

        df_summary = pd.DataFrame({
            'Druksoort': labels,
            'Druk (Pa)': pa_vals,
            'Druk (mmH₂O)': mmh2o,
            'Druk (mmHg)': mmhg,
            'Druk (bar)': bar_vals
        })

        summary_window = tk.Toplevel()
        summary_window.title("Druk Samenvatting (Diver & Referentie)")

        cols = ("Druksoort", "Druk (Pa)", "Druk (mmH₂O)", "Druk (mmHg)", "Druk (bar)")
        tree = ttk.Treeview(summary_window, columns=cols, show='headings')
        for c in cols:
            tree.heading(c, text=c)
        for _, row in df_summary.iterrows():
            tree.insert("", "end", values=(
                row['Druksoort'],
                f"{row['Druk (Pa)']:.2f}",
                f"{row['Druk (mmH₂O)']:.2f}",
                f"{row['Druk (mmHg)']:.2f}",
                f"{row['Druk (bar)']:.5f}"
            ))
        tree.pack(expand=True, fill='both')
        ttk.Button(summary_window, text="Sluiten", command=summary_window.destroy).pack(pady=10)
    except AssertionError as e:
        messagebox.showerror("Samenvatting Fout", f"Interne lengtecontrole faalde: {e}")
    except Exception as e:
        print(f"Fout bij het tonen van druk samenvatting: {e}")

# ---------- Tijdselectie GUI (met drukcap instelling) ----------

def create_time_selection_gui(df_diver, df_reference, selected_directory,
                              update_plot_callback, export_and_close_callback,
                              initial_pressure_cap=2100):
    root = tk.Tk()
    root.title("Tijdselectie & Interactieve filtering (Diver vs Referentie)")

    min_time = min(df_diver['Datetime'].min(), df_reference['Datetime'].min())
    max_time = max(df_diver['Datetime'].max(), df_reference['Datetime'].max())

    ttk.Label(root, text="Begin tijd (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    start_time_entry = ttk.Entry(root, width=25); start_time_entry.insert(0, min_time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Eind tijd (YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    end_time_entry = ttk.Entry(root, width=25); end_time_entry.insert(0, max_time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time_entry.grid(row=1, column=1, padx=5, pady=5)

    # Drukcap GUI
    cap_var = tk.DoubleVar(value=float(initial_pressure_cap) if initial_pressure_cap is not None else 2100.0)
    use_cap_var = tk.BooleanVar(value=(initial_pressure_cap is not None))

    ttk.Label(root, text="Drukcap (Pa):").grid(row=2, column=0, padx=5, pady=5, sticky='e')
    cap_entry = ttk.Entry(root, width=10, textvariable=cap_var); cap_entry.grid(row=2, column=1, sticky='w', padx=5, pady=5)
    cap_check = ttk.Checkbutton(root, text="Cap toepassen", variable=use_cap_var)
    cap_check.grid(row=2, column=1, sticky='e', padx=5, pady=5)

    info = tk.Label(root, text="Tip: sleep een kader op de tijdreeks om punten te selecteren.\nGebruik de knoppen onderaan de figuur om te filteren (Diver/Ref/Beide) of te resetten.", fg="blue")
    info.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def get_cap_value():
        if not use_cap_var.get():
            return None
        try:
            return float(cap_var.get())
        except Exception:
            messagebox.showerror("Ongeldige cap", "De drukcap moet een getal zijn (Pa).")
            return None

    def update_plot():
        try:
            start_time = datetime.strptime(start_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            if start_time >= end_time:
                messagebox.showerror("Ongeldige Tijd", "Begin tijd moet voor eind tijd zijn."); return
            cap_value = get_cap_value()
            update_plot_callback(start_time, end_time, cap_value)
        except ValueError:
            messagebox.showerror("Ongeldige Invoer", "Gebruik formaat: YYYY-MM-DD HH:MM:SS")

    def export_and_close():
        try:
            start_time = datetime.strptime(start_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            if start_time >= end_time:
                messagebox.showerror("Ongeldige Tijd", "Begin tijd moet voor eind tijd zijn."); return
            cap_value = get_cap_value()
            export_and_close_callback(start_time, end_time, cap_value)
            root.destroy()
        except ValueError:
            messagebox.showerror("Ongeldige Invoer", "Gebruik formaat: YYYY-MM-DD HH:MM:SS")

    ttk.Button(root, text="Update Grafiek", command=update_plot).grid(row=4, column=0, padx=5, pady=10, sticky='e')
    ttk.Button(root, text="Export Data en Sluiten", command=export_and_close).grid(row=4, column=1, padx=5, pady=10, sticky='w')
    root.mainloop()

# ---------- Interactieve selectie & plotten ----------

class InteractiveTimeSeries:
    def __init__(self, df_diver, df_reference, start_time=None, end_time=None, pressure_cap=2100):
        self.df_diver_original = df_diver.copy()
        self.df_ref_original   = df_reference.copy()
        self.df_diver = df_diver.copy()
        self.df_ref   = df_reference.copy()
        self.start_time = start_time
        self.end_time   = end_time
        self.pressure_cap = pressure_cap

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)  # ruimte voor knoppen

        # RectangleSelector
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],  # linker muis
            minspanx=0, minspany=0,
            spancoords='data'
        )
        self.selection = None  # (xmin_dt, xmax_dt, ymin, ymax) met xmin/xmax als pandas Timestamps

        # Knoppen
        ax_div = plt.axes([0.10, 0.05, 0.15, 0.06])
        ax_ref = plt.axes([0.28, 0.05, 0.15, 0.06])
        ax_both= plt.axes([0.46, 0.05, 0.18, 0.06])
        ax_rst = plt.axes([0.67, 0.05, 0.15, 0.06])

        self.btn_filter_div = Button(ax_div,  'Filter Diver')
        self.btn_filter_ref = Button(ax_ref,  'Filter Ref')
        self.btn_filter_both= Button(ax_both, 'Filter Beide')
        self.btn_reset      = Button(ax_rst,  'Reset')

        self.btn_filter_div.on_clicked(lambda evt: self.apply_filter(target='diver'))
        self.btn_filter_ref.on_clicked(lambda evt: self.apply_filter(target='ref'))
        self.btn_filter_both.on_clicked(lambda evt: self.apply_filter(target='both'))
        self.btn_reset.on_clicked(self.reset_data)

        self.update_plot()

    def _to_timestamp(self, x):
        """Converteer x (float datenumber of datetime) naar pandas.Timestamp."""
        if x is None:
            return None
        if isinstance(x, (float, np.floating)):
            return pd.to_datetime(num2date(x)).tz_localize(None)
        if isinstance(x, datetime):
            return pd.to_datetime(x)
        try:
            return pd.to_datetime(x)
        except Exception:
            return None

    def on_select(self, eclick, erelease):
        xmin = self._to_timestamp(eclick.xdata)
        xmax = self._to_timestamp(erelease.xdata)
        if xmin is None or xmax is None:
            self.selection = None
            return
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        ymin, ymax = sorted([float(eclick.ydata), float(erelease.ydata)])
        self.selection = (xmin, xmax, ymin, ymax)

    def current_window(self, df):
        if self.start_time and self.end_time:
            m = (df['Datetime'] >= self.start_time) & (df['Datetime'] <= self.end_time)
            return df[m]
        return df

    def _apply_cap(self, df):
        if self.pressure_cap is not None:
            return df[df['Pressure'] <= self.pressure_cap]
        return df

    def set_pressure_cap(self, cap):
        self.pressure_cap = cap
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        df_div = self._apply_cap(self.current_window(self.df_diver))
        df_ref = self._apply_cap(self.current_window(self.df_ref))

        self.ax.plot(df_div['Datetime'], df_div['Pressure'], label='Diver druk', marker='x', markersize=2, linewidth=0.5)
        self.ax.plot(df_ref['Datetime'], df_ref['Pressure'], label='Referentie sensor', marker='s', markersize=4, linestyle='--')
        cap_txt = f" (cap={self.pressure_cap} Pa)" if self.pressure_cap is not None else " (geen cap)"
        self.ax.set_title('Diver druk vs Referentie sensor over Tijd' + cap_txt)
        self.ax.set_xlabel('Tijd'); self.ax.set_ylabel('Druk (Pa)'); self.ax.legend(); self.ax.grid(True)
        plt.gcf().autofmt_xdate()
        self.fig.canvas.draw_idle()

    def apply_filter(self, target='both'):
        if not self.selection:
            messagebox.showwarning("Geen selectie", "Sleep eerst een kader over de punten die je wil verwijderen.")
            return
        xmin, xmax, ymin, ymax = self.selection

        def filter_df(df):
            m_time = (df['Datetime'] >= xmin) & (df['Datetime'] <= xmax)
            m_val  = (df['Pressure'] >= ymin) & (df['Pressure'] <= ymax)
            return df[~(m_time & m_val)].reset_index(drop=True)

        if target in ('diver', 'both'):
            self.df_diver = filter_df(self.df_diver)
        if target in ('ref', 'both'):
            self.df_ref = filter_df(self.df_ref)

        self.update_plot()

    def reset_data(self, event=None):
        self.df_diver = self.df_diver_original.copy()
        self.df_ref   = self.df_ref_original.copy()
        self.selection = None
        self.update_plot()

# XY-regressie (gereflecteerd na filtering)
def plot_xy_regression_with_slider(df_diver, df_reference, start_time=None, end_time=None, pressure_cap=2100):
    try:
        fig, ax = plt.subplots(figsize=(8, 6)); plt.subplots_adjust(bottom=0.25)

        def filtered_window(df):
            if start_time and end_time:
                m = (df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)
                df = df[m]
            if pressure_cap is not None:
                df = df[df['Pressure'] <= pressure_cap]
            return df

        df_div_f = filtered_window(df_diver)
        df_ref_f = filtered_window(df_reference)

        merged_df = pd.merge_asof(df_ref_f.sort_values('Datetime'),
                                  df_div_f.sort_values('Datetime'),
                                  on='Datetime', suffixes=('_ref', '_div'))
        merged_df = merged_df.dropna(subset=['Pressure_ref','Pressure_div'])
        if pressure_cap is not None:
            merged_df = merged_df[(merged_df['Pressure_ref'] <= pressure_cap) & (merged_df['Pressure_div'] <= pressure_cap)]

        x, y = merged_df['Pressure_ref'], merged_df['Pressure_div']
        ax.scatter(x, y, label='Data')

        if len(x) > 1:
            slope, intercept, r, _, _ = linregress(x, y)
            ax.plot(x, slope*x + intercept, label=f'Lineaire Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r**2:.4f}')

        ax.set_xlabel('Referentie druk (Pa)'); ax.set_ylabel('Diver druk (Pa)')
        ax.set_title('X-Y Plot: Referentie (x) vs Diver (y)'); ax.legend(); ax.grid(True)

        ax_shift = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider_shift = Slider(ax_shift, 'Tijd Shift Diver (sec)', -60, 60, valinit=0, valstep=0.5)

        def update_regression(val):
            try:
                shift = slider_shift.val
                shifted = df_diver.copy(); shifted['Datetime'] = shifted['Datetime'] + timedelta(seconds=shift)
                df_div_s = filtered_window(shifted)
                df_ref_s = filtered_window(df_reference)

                merged = pd.merge_asof(df_ref_s.sort_values('Datetime'),
                                       df_div_s.sort_values('Datetime'),
                                       on='Datetime', suffixes=('_ref','_div')).dropna(subset=['Pressure_ref','Pressure_div'])
                if pressure_cap is not None:
                    merged = merged[(merged['Pressure_ref'] <= pressure_cap) & (merged['Pressure_div'] <= pressure_cap)]
                x_new, y_new = merged['Pressure_ref'], merged['Pressure_div']

                ax.clear()
                if len(x_new) > 1:
                    slope, intercept, r, _, _ = linregress(x_new, y_new)
                    ax.scatter(x_new, y_new, label='Data')
                    ax.plot(x_new, slope*x_new + intercept, label=f'Lineaire Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r**2:.4f}')
                else:
                    ax.scatter(x_new, y_new, label='Data')

                ax.set_xlabel('Referentie druk (Pa)'); ax.set_ylabel('Diver druk (Pa)')
                ax.set_title('X-Y Plot: Referentie (x) vs Diver (y)'); ax.legend(); ax.grid(True)
                fig.canvas.draw_idle()
            except Exception as e:
                print(f"Fout in update_regression: {e}")

        slider_shift.on_changed(update_regression)
        plt.show(block=False); plt.pause(0.001)
    except Exception as e:
        print(f"Fout in plot_xy_regression_with_slider: {e}")

# ---------- Export ----------

def export_data(df_diver, df_reference, outdir, start_time, end_time, pressure_cap=2100):
    try:
        def window(df):
            m = (df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)
            df = df[m]
            if pressure_cap is not None:
                df = df[df['Pressure'] <= pressure_cap]
            return df

        df_div_f = window(df_diver)
        df_ref_f = window(df_reference)

        merged_df = pd.merge_asof(df_ref_f.sort_values('Datetime'),
                                  df_div_f.sort_values('Datetime'),
                                  on='Datetime', suffixes=('_ref', '_div'))

        f1 = os.path.join(outdir, 'diver_druk_data_filtered.csv')
        f2 = os.path.join(outdir, 'reference_druk_data_filtered.csv')
        f3 = os.path.join(outdir, 'combined_ref_diver_filtered.csv')
        df_div_f.to_csv(f1, index=False); df_ref_f.to_csv(f2, index=False); merged_df.to_csv(f3, index=False)

        messagebox.showinfo("Export Succesvol", f"Data geëxporteerd naar:\n{f1}\n{f2}\n{f3}")
    except Exception as e:
        messagebox.showerror("Export Fout", f"Er is een fout opgetreden bij het exporteren van de data:\n{e}")

# ---------- Main ----------

def main():
    outdir = select_directory()
    if not outdir:
        messagebox.showwarning("Geen Directory Geselecteerd", "Er is geen directory geselecteerd."); return

    mon_file = filedialog.askopenfilename(title="Selecteer een Diver .mon bestand", filetypes=[("MON bestanden", "*.mon")])
    if not mon_file:
        messagebox.showwarning("Geen Bestand Geselecteerd", "Er is geen .mon bestand geselecteerd."); return
    print(f".mon: {mon_file}")

    mon_pressure_offset = get_offset_input('Diver druk (Pa, + = optellen)')
    mon_time_offset = get_offset_input('Diver tijd (in seconden, + = later)')

    lines = read_mon_file(mon_file)
    if not lines:
        messagebox.showerror("Geen Data", "Geen data in .mon bestand."); return
    df_diver = parse_mon_data(lines, mon_pressure_offset, mon_time_offset)
    if df_diver.empty:
        messagebox.showerror("Geen Data", "Geen geldige .mon data."); return

    ref_file = filedialog.askopenfilename(
        title="Selecteer referentie CSV (comma delimited; data vanaf lijn 3; kolom1=seconden, kolom2=waarde)",
        filetypes=[("CSV", "*.csv"), ("TXT", "*.txt"), ("Alle bestanden", "*.*")]
    )
    if not ref_file:
        messagebox.showwarning("Geen Bestand Geselecteerd", "Er is geen referentie-bestand geselecteerd."); return
    print(f"Referentie: {ref_file}")

    df_reference = read_reference_csv_seconds_since_start(ref_file)
    if df_reference.empty:
        messagebox.showerror("Geen Data", "Geen geldige referentie-data."); return

    display_pressure_summary(df_diver, df_reference)

    # 1) Interactieve tijdreeks met kader-selectie en filterknoppen
    its = InteractiveTimeSeries(df_diver, df_reference, pressure_cap=2100)

    # 2) XY-regressie die de (eventueel gefilterde) data gebruikt
    def update_plot_with_time_range(start_time, end_time, cap_value):
        its.start_time = start_time
        its.end_time   = end_time
        its.set_pressure_cap(cap_value)
        its.update_plot()
        plot_xy_regression_with_slider(its.df_diver, its.df_ref, start_time, end_time, pressure_cap=cap_value)

    def export_and_close(start_time, end_time, cap_value):
        plot_xy_regression_with_slider(its.df_diver, its.df_ref, start_time, end_time, pressure_cap=cap_value)
        export_data(its.df_diver, its.df_ref, outdir, start_time, end_time, pressure_cap=cap_value)

    create_time_selection_gui(its.df_diver, its.df_ref, outdir, update_plot_with_time_range, export_and_close, initial_pressure_cap=2100)

if __name__ == "__main__":
    main()
