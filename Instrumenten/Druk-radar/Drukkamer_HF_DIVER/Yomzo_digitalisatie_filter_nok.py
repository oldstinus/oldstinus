import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime, timedelta
import chardet
from scipy.stats import linregress
import numpy as np

# Schakel interactieve modus in
plt.ion()

# Functie om bestandsencoding te detecteren
def detect_file_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read()
            return chardet.detect(rawdata)['encoding']
    except:
        return 'utf-8'

# Functie om HF CSV-bestand te verwerken
def process_hf_file(file_path, pressure_offset=0):
    print(f"Inlezen HF: {file_path}")
    try:
        # Lees numerieke data vanaf regel 11 zonder header
        data = pd.read_csv(file_path,
                           encoding=detect_file_encoding(file_path),
                           skiprows=10,
                           header=None)
        # Filter op kolom 1 gelijk aan 'C1'
        filtered = data[data[1] == 'C1']
        pressures = filtered[2].astype(float) * 1000 + pressure_offset
        # Tijdstempels gebaseerd op creatietijd bestand
        creation = os.path.getctime(file_path)
        start_time = datetime.fromtimestamp(creation)
        times = [start_time + timedelta(seconds=i/8) for i in range(len(pressures))]
        return pd.DataFrame({'Datetime': times, 'Pressure': pressures})
    except Exception as e:
        print(f"Fout bij HF verwerken {file_path}: {e}")
        return pd.DataFrame(columns=['Datetime', 'Pressure'])

# Functie om referentie CSV-bestand te verwerken volgens Yomzo-formaat
def process_ref_file(file_path, pressure_offset=0):
    print(f"Inlezen Referentie: {file_path}")
    try:
        with open(file_path, 'r', encoding=detect_file_encoding(file_path)) as f:
            # Eerste regel bevat bestandsnaam met timestamp
            header_line = f.readline().strip()
            if header_line.startswith("DDR_"):
                header_line = header_line[len("DDR_"):]
            header_line = header_line.split('.')[0]
            base_datetime = datetime.strptime(header_line, "%Y%m%d_%H%M%S")

            # Tweede regel met kolomnamen
            col_line = f.readline().strip()
            cols = col_line.split(',')

            # Lees de data
            df = pd.read_csv(
                f,
                delimiter=',',
                decimal='.',
                header=None,
                names=cols,
                skip_blank_lines=True
            )

        # Controle kolommen
        if 'time[sec]' not in df.columns or 'data[xxxx]' not in df.columns:
            raise ValueError("Verwachte kolommen 'time[sec]' en 'data[xxxx]' niet gevonden.")

        # Bepaal absolute tijd
        df['Datetime'] = df['time[sec]'].apply(lambda sec: base_datetime + timedelta(seconds=sec))
        # Drukwaarden toepassen met offset
        df['Pressure'] = df['data[xxxx]'].astype(float) + pressure_offset
        return df[['Datetime', 'Pressure']]
    except Exception as e:
        print(f"Fout bij referentie verwerken {file_path}: {e}")
        return pd.DataFrame(columns=['Datetime', 'Pressure'])

# Functie om eenheden om te rekenen
def convert_pressure_units(p):
    return {
        'Pa': p,
        'mmH2O': p / 9.80665,
        'mmHg': p / 133.322,
        'bar': p / 100000
    }

# Toon samenvatting van drukwaarden in tabel
def display_pressure_summary(df_hf, df_ref):
    try:
        entries = []
        for label, df in [('HF Sensor', df_hf), ('Referentie Sensor', df_ref)]:
            if not df.empty:
                start = df['Pressure'].iloc[0]
                end = df['Pressure'].iloc[-1]
                mx = df['Pressure'].max()
            else:
                start = end = mx = np.nan
            for kind, val in [('Start', start), ('Eind', end), ('Max', mx)]:
                u = convert_pressure_units(val)
                entries.append([f"{label} {kind}", f"{u['Pa']:.2f}", f"{u['mmH2O']:.2f}", f"{u['mmHg']:.2f}", f"{u['bar']:.5f}"])

        win = tk.Tk(); win.title("Druk Samenvatting")
        cols = ["Soort", "Pa", "mmH₂O", "mmHg", "bar"]
        tree = ttk.Treeview(win, columns=cols, show='headings')
        for c in cols:
            tree.heading(c, text=c)
        for row in entries:
            tree.insert('', 'end', values=row)
        tree.pack(expand=True, fill='both')
        ttk.Button(win, text="Sluiten", command=win.destroy).pack(pady=5)
        win.mainloop()
    except Exception as e:
        print(f"Fout in druk samenvatting: {e}")

# GUI voor tijdselectie
def create_time_selection_gui(df_hf, df_ref, update_callback, export_callback):
    root = tk.Tk(); root.title("Tijdselectie voor Grafiek")
    min_t = min(df_hf['Datetime'].min(), df_ref['Datetime'].min())
    max_t = max(df_hf['Datetime'].max(), df_ref['Datetime'].max())

    ttk.Label(root, text="Begin (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0)
    start_e = ttk.Entry(root, width=25);
    start_e.insert(0, min_t.strftime("%Y-%m-%d %H:%M:%S")); start_e.grid(row=0, column=1)
    ttk.Label(root, text="Eind (YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0)
    end_e = ttk.Entry(root, width=25);
    end_e.insert(0, max_t.strftime("%Y-%m-%d %H:%M:%S")); end_e.grid(row=1, column=1)

    def on_update():
        try:
            s = datetime.strptime(start_e.get(), "%Y-%m-%d %H:%M:%S")
            e = datetime.strptime(end_e.get(), "%Y-%m-%d %H:%M:%S")
            if s>=e: raise ValueError
            update_callback(s, e)
        except:
            messagebox.showerror("Ongeldige Tijd", "Controleer het formaat en volgorde.")

    def on_export():
        try:
            s = datetime.strptime(start_e.get(), "%Y-%m-%d %H:%M:%S")
            e = datetime.strptime(end_e.get(), "%Y-%m-%d %H:%M:%S")
            if s>=e: raise ValueError
            export_callback(s, e)
            root.destroy()
        except:
            messagebox.showerror("Ongeldige Tijd", "Controleer het formaat en volgorde.")

    ttk.Button(root, text="Update Grafiek", command=on_update).grid(row=2, column=0, pady=5)
    ttk.Button(root, text="Export en Sluit", command=on_export).grid(row=2, column=1, pady=5)
    root.mainloop()

# Plot gecombineerde grafiek
def plot_combined_graph(df_hf, df_ref, start=None, end=None):
    fig, ax = plt.subplots(figsize=(10,5))
    df_h = df_hf[(df_hf['Datetime']>=start)&(df_hf['Datetime']<=end)] if start and end else df_hf
    df_r = df_ref[(df_ref['Datetime']>=start)&(df_ref['Datetime']<=end)] if start and end else df_ref
    df_h = df_h[df_h['Pressure']<=2100]; df_r = df_r[df_r['Pressure']<=2100]
    ax.plot(df_h['Datetime'], df_h['Pressure'], label='HF Sensor', marker='o', markersize=2, linewidth=0.5)
    ax.plot(df_r['Datetime'], df_r['Pressure'], label='Referentie Sensor', marker='s', markersize=2, linewidth=0.5)
    ax.set_xlabel('Tijd'); ax.set_ylabel('Druk (Pa)'); ax.legend(); ax.grid(True)
    plt.gcf().autofmt_xdate(); plt.show(block=False)

# XY-regressie met slider
def plot_xy_regression_with_slider(df_hf, df_ref, start=None, end=None):
    fig, ax = plt.subplots(figsize=(6,6)); plt.subplots_adjust(bottom=0.25)
    df_h = df_hf[(df_hf['Datetime']>=start)&(df_hf['Datetime']<=end)] if start and end else df_hf
    df_r = df_ref[(df_ref['Datetime']>=start)&(df_ref['Datetime']<=end)] if start and end else df_ref
    merged = pd.merge_asof(df_h.sort_values('Datetime'), df_r.sort_values('Datetime'), on='Datetime', suffixes=('_hf','_ref'))
    merged = merged[(merged['Pressure_hf']<=2100)&(merged['Pressure_ref']<=2100)]
    x, y = merged['Pressure_hf'], merged['Pressure_ref']
    ax.scatter(x, y, label='Data')
    if len(x)>0:
        slope, intercept, r, _, _ = linregress(x, y); ax.plot(x, slope*x+intercept, label=f'y={slope:.2f}x+{intercept:.2f}\n$R^2$={r**2:.4f}')
    ax.set_xlabel('HF (Pa)'); ax.set_ylabel('Ref (Pa)'); ax.legend(); ax.grid(True)
    ax_shift = plt.axes([0.25,0.1,0.65,0.03])
    slider = Slider(ax_shift, 'Shift(s)', -60,60, valinit=0, valstep=0.5)
    def update(val):
        shift = slider.val
        df_r_shift = df_ref.copy(); df_r_shift['Datetime'] = df_r_shift['Datetime'] + timedelta(seconds=shift)
        m = pd.merge_asof(df_h.sort_values('Datetime'), df_r_shift.sort_values('Datetime'), on='Datetime', suffixes=('_hf','_ref'))
        m = m[(m['Pressure_hf']<=2100)&(m['Pressure_ref']<=2100)]
        ax.clear()
        x2, y2 = m['Pressure_hf'], m['Pressure_ref']
        ax.scatter(x2, y2)
        if len(x2)>1:
            s, i, _, _, _ = linregress(x2, y2); ax.plot(x2, s*x2+i)
        ax.set_xlabel('HF (Pa)'); ax.set_ylabel('Ref (Pa)'); ax.grid(True)
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show(block=False)

# Data export
def export_data(df_hf, df_ref, dir_path, start, end):
    df_h = df_hf[(df_hf['Datetime']>=start)&(df_hf['Datetime']<=end)]; df_r = df_ref[(df_ref['Datetime']>=start)&(df_ref['Datetime']<=end)]
    df_h.to_csv(os.path.join(dir_path, 'hf_filtered.csv'), index=False)
    df_r.to_csv(os.path.join(dir_path, 'ref_filtered.csv'), index=False)
    merged = pd.merge_asof(df_h.sort_values('Datetime'), df_r.sort_values('Datetime'), on='Datetime')
    merged.to_csv(os.path.join(dir_path, 'combined_filtered.csv'), index=False)
    messagebox.showinfo("Export Gereed", f"Bestanden opgeslagen in {dir_path}")

# Hoofdprogramma
def main():
    root = tk.Tk(); root.withdraw()

    # HF sensor bestanden
    hf_files = filedialog.askopenfilenames(title="Selecteer HF sensor CSV-bestanden", filetypes=[("CSV", "*.csv")])
    if not hf_files:
        messagebox.showwarning("Geen HF bestanden", "Geen HF CSV-bestanden geselecteerd.")
        return
    hf_offset = simpledialog.askfloat("HF Offset", "Voer HF sensor druk offset in:", initialvalue=0.0) or 0.0
    dfs_hf = [process_hf_file(fp, hf_offset) for fp in hf_files]
    df_hf = pd.concat(dfs_hf, ignore_index=True)

    # Referentie sensor bestanden
    ref_files = filedialog.askopenfilenames(title="Selecteer referentie CSV-bestanden", filetypes=[("CSV", "*.csv")])
    if not ref_files:
        messagebox.showwarning("Geen referentie bestanden", "Geen referentie CSV-bestanden geselecteerd.")
        return
    ref_offset = simpledialog.askfloat("Referentie Offset", "Voer referentie sensor druk offset in:", initialvalue=0.0) or 0.0
    dfs_ref = [process_ref_file(fp, ref_offset) for fp in ref_files]
    df_ref = pd.concat(dfs_ref, ignore_index=True)

    display_pressure_summary(df_hf, df_ref)

    def update_plots(s, e):
        plot_combined_graph(df_hf, df_ref, s, e)
        plot_xy_regression_with_slider(df_hf, df_ref, s, e)

    def export_and_close(s, e):
        plot_combined_graph(df_hf, df_ref, s, e)
        plot_xy_regression_with_slider(df_hf, df_ref, s, e)
        export_data(df_hf, df_ref, os.path.dirname(hf_files[0]), s, e)

    create_time_selection_gui(df_hf, df_ref, update_plots, export_and_close)

if __name__ == "__main__":
    main()
