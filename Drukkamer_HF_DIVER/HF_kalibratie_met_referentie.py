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

# Zet interactieve modus aan voor matplotlib\plt.ion()

def detect_file_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        result = chardet.detect(raw)
        return result.get('encoding') or 'utf-8'
    except:
        return 'utf-8'

# Verwerk HF-sensor data
def process_hf_file(path, offset=0.0):
    print(f"[DEBUG] Inlezen HF: {path}")
    try:
        df = pd.read_csv(
            path,
            encoding=detect_file_encoding(path),
            skiprows=10,
            header=None,
            names=[None, 'marker', 'value']
        )
        print(f"[DEBUG] Raw HF rows: {len(df)}")
        sel = df[df['marker'] == 'C1']
        print(f"[DEBUG] Gefilterde HF rows: {len(sel)}")
        pressures = sel['value'].astype(float) * 1000 + offset
        base = os.path.getctime(path)
        start = datetime.fromtimestamp(base)
        timestamps = [start + timedelta(seconds=i/8) for i in range(len(pressures))]
        return pd.DataFrame({'Datetime': timestamps, 'Pressure': pressures})
    except Exception as e:
        print(f"[ERROR] HF verwerken mislukt: {e}")
        return pd.DataFrame(columns=['Datetime', 'Pressure'])

# Verwerk referentie-sensor data volgens Yomzo-formaat
def process_ref_file(path, offset=0.0):
    print(f"[DEBUG] Inlezen Referentie: {path}")
    try:
        enc = detect_file_encoding(path)
        with open(path, 'r', encoding=enc) as f:
            header = f.readline().strip()
            if header.startswith('DDR_'):
                header = header[4:]
            key = header.split('.')[0]
            base_dt = datetime.strptime(key, '%Y%m%d_%H%M%S')
            cols = f.readline().strip().split(',')
            df = pd.read_csv(
                f,
                names=cols,
                header=None,
                delimiter=',',
                skip_blank_lines=True
            )
        print(f"[DEBUG] Raw Ref rows: {len(df)}")
        if 'time[sec]' not in df.columns or 'data[xxxx]' not in df.columns:
            raise KeyError('Kolom time[sec] of data[xxxx] ontbreekt')
        df['Datetime'] = df['time[sec]'].astype(float).apply(lambda s: base_dt + timedelta(seconds=s))
        df['Pressure'] = pd.to_numeric(
            df['data[xxxx]'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True),
            errors='coerce'
        ) + offset
        df = df.dropna(subset=['Pressure'])
        print(f"[DEBUG] Gefilterde Ref rows: {len(df)}")
        return df[['Datetime', 'Pressure']]
    except Exception as e:
        print(f"[ERROR] Ref verwerken mislukt: {e}")
        return pd.DataFrame(columns=['Datetime', 'Pressure'])

# Converteer drukeenheden
def convert_pressure_units(p):
    return {
        'Pa': p,
        'mmH2O': p / 9.80665,
        'mmHg': p / 133.322,
        'bar': p / 100000
    }

# Toon samenvatting van drukwaarden
def display_pressure_summary(df_hf, df_ref):
    entries = []
    for label, df in [('HF Sensor', df_hf), ('Referentie Sensor', df_ref)]:
        if not df.empty:
            start_v = df['Pressure'].iloc[0]
            end_v = df['Pressure'].iloc[-1]
            max_v = df['Pressure'].max()
        else:
            start_v = end_v = max_v = np.nan
        for kind, val in [('Start', start_v), ('Eind', end_v), ('Max', max_v)]:
            u = convert_pressure_units(val)
            entries.append([f"{label} {kind}", f"{u['Pa']:.2f}", f"{u['mmH2O']:.2f}", f"{u['mmHg']:.2f}", f"{u['bar']:.5f}"])
    win = tk.Tk()
    win.title("Druk Samenvatting")
    cols = ["Soort","Pa","mmH₂O","mmHg","bar"]
    tree = ttk.Treeview(win, columns=cols, show='headings')
    for c in cols:
        tree.heading(c, text=c)
    for row in entries:
        tree.insert('', 'end', values=row)
    tree.pack(expand=True, fill='both')
    ttk.Button(win, text="Sluiten", command=win.destroy).pack(pady=5)
    win.mainloop()

# GUI voor tijdselectie
    def create_time_selection_gui(df_hf, df_ref, update_cb, export_cb):
        root = tk.Tk()
        root.title("Tijdselectie voor Grafiek")
        min_t = min(df_hf['Datetime'].min(), df_ref['Datetime'].min())
        max_t = max(df_hf['Datetime'].max(), df_ref['Datetime'].max())
        ttk.Label(root, text="Begin (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0)
        start_e = ttk.Entry(root, width=25)
        start_e.insert(0, min_t.strftime("%Y-%m-%d %H:%M:%S"))
        start_e.grid(row=0, column=1)
        ttk.Label(root, text="Eind (YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0)
        end_e = ttk.Entry(root, width=25)
        end_e.insert(0, max_t.strftime("%Y-%m-%d %H:%M:%S"))
        end_e.grid(row=1, column=1)
    def on_update():
        try:
            s = datetime.strptime(start_e.get(), "%Y-%m-%d %H:%M:%S")
            e = datetime.strptime(end_e.get(), "%Y-%m-%d %H:%M:%S")
            if s >= e:
                raise ValueError
            update_cb(s, e)
        except:
            messagebox.showerror("Ongeldige Tijd","Controleer formaat en volgorde.")
    def on_export():
        try:
            s = datetime.strptime(start_e.get(), "%Y-%m-%d %H:%M:%S")
            e = datetime.strptime(end_e.get(), "%Y-%m-%d %H:%M:%S")
            if s >= e:
                raise ValueError
            export_cb(s, e)
            root.destroy()
        except:
            messagebox.showerror("Ongeldige Tijd","Controleer formaat en volgorde.")
    ttk.Button(root, text="Update Grafiek", command=on_update).grid(row=2, column=0, pady=5)
    ttk.Button(root, text="Export en Sluit", command=on_export).grid(row=2, column=1, pady=5)
    root.mainloop()

# Plot gecombineerde grafiek
def plot_combined_graph(df_hf, df_ref, start=None, end=None):
    fig, ax = plt.subplots(figsize=(10,5))
    h = df_hf.copy()
    r = df_ref.copy()
    if start and end:
        h = h[(h['Datetime']>=start)&(h['Datetime']<=end)]
        r = r[(r['Datetime']>=start)&(r['Datetime']<=end)]
    h = h[h['Pressure']<=2100]
    r = r[r['Pressure']<=2100]
    ax.plot(h['Datetime'],h['Pressure'],label='HF Sensor',marker='o',markersize=2,linewidth=0.5)
    ax.plot(r['Datetime'],r['Pressure'],label='Referentie Sensor',marker='s',markersize=2,linewidth=0.5)
    ax.set_xlabel('Tijd')
    ax.set_ylabel('Druk (Pa)')
    ax.legend(); ax.grid(True)
    plt.gcf().autofmt_xdate(); plt.show(block=False)

# XY-regressie met schuifregelaar
def plot_xy_regression_with_slider(df_hf, df_ref, start=None, end=None):
    fig, ax = plt.subplots(figsize=(6,6)); plt.subplots_adjust(bottom=0.25)
    h = df_hf.copy(); r = df_ref.copy()
    if start and end:
        h = h[(h['Datetime']>=start)&(h['Datetime']<=end)]
        r = r[(r['Datetime']>=start)&(r['Datetime']<=end)]
    merged = pd.merge_asof(h.sort_values('Datetime'),r.sort_values('Datetime'),on='Datetime',suffixes=('_hf','_ref'))
    merged = merged[(merged['Pressure_hf']<=2100)&(merged['Pressure_ref']<=2100)]
    x,y = merged['Pressure_hf'],merged['Pressure_ref']
    ax.scatter(x,y,label='Data')
    if len(x)>1:
        s,i,r2,_,_ = linregress(x,y); ax.plot(x,s*x+i,label=f'y={s:.2f}x+{i:.2f}\n$R^2$={r2**2:.4f}')
    ax.set_xlabel('HF (Pa)'); ax.set_ylabel('Ref (Pa)'); ax.legend(); ax.grid(True)
    ax_shift = plt.axes([0.25,0.1,0.65,0.03])
    slider = Slider(ax_shift,'Shift(s)',-60,60,valinit=0,valstep=0.5)
    def update(val):
        shift = slider.val; r2=df_ref.copy(); r2['Datetime']+=timedelta(seconds=shift)
        m2=pd.merge_asof(df_hf.sort_values('Datetime'),r2.sort_values('Datetime'),on='Datetime',suffixes=('_hf','_ref'))
        m2=m2[(m2['Pressure_hf']<=2100)&(m2['Pressure_ref']<=2100)]
        ax.clear(); xs,ys=m2['Pressure_hf'],m2['Pressure_ref']; ax.scatter(xs,ys)
        if len(xs)>1: s2,i2,_,_,_=linregress(xs,ys); ax.plot(xs,s2*xs+i2)
        ax.set_xlabel('HF (Pa)'); ax.set_ylabel('Ref (Pa)'); ax.grid(True); fig.canvas.draw_idle()
    slider.on_changed(update);

# Exporteer data
def export_data(df_hf, df_ref, path, start, end):
    h = df_hf[(df_hf['Datetime']>=start)&(df_hf['Datetime']<=end)]
    r = df_ref[(df_ref['Datetime']>=start)&(df_ref['Datetime']<=end)]
    h.to_csv(os.path.join(path,'hf_filtered.csv'),index=False)
    r.to_csv(os.path.join(path,'ref_filtered.csv'),index=False)
    merged = pd.merge_asof(h.sort_values('Datetime'),r.sort_values('Datetime'),on='Datetime')
    merged.to_csv(os.path.join(path,'combined_filtered.csv'),index=False)
    messagebox.showinfo("Export Gereed",f"Bestanden opgeslagen in {path}")

# Main
def main():
    root=tk.Tk(); root.withdraw()
    hf_files = filedialog.askopenfilenames(title="Selecteer HF CSV-bestanden", filetypes=[("CSV","*.csv")])
    if not hf_files: messagebox.showwarning("Geen HF CSV","Geen HF CSV-bestanden geselecteerd."); return
    hf_off = simpledialog.askfloat("HF Offset","Voer HF offset in:",initialvalue=0.0) or 0.0
    dfs_hf = [process_hf_file(fp,hf_off) for fp in hf_files]; df_hf=pd.concat(dfs_hf,ignore_index=True)
    ref_files = filedialog.askopenfilenames(title="Selecteer Referentie CSV-bestanden", filetypes=[("CSV","*.csv")])
    if not ref_files: messagebox.showwarning("Geen Referentie CSV","Geen referentie CSV-bestanden geselecteerd."); return
    ref_off = simpledialog.askfloat("Referentie Offset","Voer referentie offset in:",initialvalue=0.0) or 0.0
    dfs_ref = [process_ref_file(fp,ref_off) for fp in ref_files]; df_ref=pd.concat(dfs_ref,ignore_index=True)
    display_pressure_summary(df_hf,df_ref)
    def upd(s,e): plot_combined_graph(df_hf,df_ref,s,e); plot_xy_regression_with_slider(df_hf,df_ref,s,e)
    def exp(s,e): upd(s,e); export_data(df_hf,df_ref,os.path.dirname(hf_files[0]),s,e)
    create_time_selection_gui(df_hf,df_ref,upd,exp)

if __name__=="__main__": main()
