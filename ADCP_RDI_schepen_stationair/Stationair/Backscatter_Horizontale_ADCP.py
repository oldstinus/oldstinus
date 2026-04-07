# -*- coding: utf-8 -*-
"""
FIX_v2 – Multi-file ADCP backscatter (beam/pair × datetime) + TVG + 3D interactief
----------------------------------------------------------------------------------
Robuust bestandbeheer:
- CLI-argumenten worden gecontroleerd op bestaan; ontbrekende paden worden gemeld.
- Indien niets bruikbaars via CLI, opent GUI-bestandskeuze (tkinter).
- Duidelijke samenvatting: aantal ingelezen bestanden, records (tijdstappen) en aantal beams.

Uitvoer:
- Interactieve 3D HTML: backscatter_bins_TVG_3D_interactive.html, backscatter_pairs_TVG_3D_interactive.html
- CSV: concat_counts.csv, concat_bins_TVG_dB.csv, concat_pairs_TVG_dB.csv
"""

import os, sys, io, numpy as np, pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

# -------------------- USER SETTINGS --------------------
BIN_SIZE_M        = 0.25
BLANKING_M        = 0.50
KC_DB_PER_COUNT   = 0.45
USE_ABSORPTION    = False
FREQ_KHZ          = 1200
ALPHA_DB_PER_M    = 0.00217 * (FREQ_KHZ/1000.0)**2 * 100.0
Y_TICK_COUNT      = 12
COLOR_SCALE_3D    = 'Viridis'
Z_SCALES          = [0.5, 1.0, 1.5, 2.0, 3.0]
# ------------------------------------------------------

def choose_files_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        paths = filedialog.askopenfilenames(title="Selecteer ADCP files",
                                            filetypes=[("Data files","*.dac *.csv *.txt"),("All files","*.*")])
        return list(paths)
    except Exception as e:
        print("[INFO] GUI niet beschikbaar:", e)
        return []

def read_dac_like(path):
    rows = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=[p.strip() for p in line.split(",")]
            if len(parts)<3: continue
            ts = parts[0].strip('"').strip("'")
            try: t = pd.to_datetime(ts).to_pydatetime()
            except: continue
            nums=[]
            for p in parts[2:]:
                try: nums.append(float(p))
                except: nums.append(np.nan)
            rows.append((t, nums))
    if not rows: return [], np.zeros((0,0), dtype=float)
    max_bins = max(len(r[1]) for r in rows)
    times=[]; data=np.full((len(rows), max_bins), np.nan, dtype=float)
    for i,(t,nums) in enumerate(rows):
        times.append(t); data[i,:len(nums)] = nums
    return times, data

def concat_chronologically(all_times, all_data):
    if not all_times: return [], np.zeros((0,0), dtype=float)
    max_bins = max(d.shape[1] for d in all_data) if all_data else 0
    rows = sum(d.shape[0] for d in all_data)
    cat = np.full((rows, max_bins), np.nan, dtype=float); tt = []
    idx = 0
    for times, M in zip(all_times, all_data):
        T, N = M.shape
        cat[idx:idx+T, :N] = M
        tt.extend(times)
        idx += T
    order = np.argsort(np.array(tt, dtype='datetime64[ns]'))
    cat = cat[order, :]
    tt  = [tt[i] for i in order]
    valid = ~((np.all(np.isnan(cat), axis=0)) | (np.nanmax(cat, axis=0)==0))
    cat = cat[:, valid]
    return tt, cat

def counts_to_db(M):
    E = KC_DB_PER_COUNT * M.astype(float); E[M<=0] = np.nan; return E

def compute_R(n_bins):
    return BLANKING_M + (np.arange(n_bins)+0.5)*BIN_SIZE_M

def tvg(E_dB, R):
    if E_dB.size==0: return E_dB
    R_ref = float(np.nanmedian(R)); TVG = 20.0*np.log10(R/R_ref)
    if USE_ABSORPTION: TVG += 2.0*ALPHA_DB_PER_M*(R-R_ref)
    return E_dB + TVG[np.newaxis,:]

def pair_means(M):
    n_bins = M.shape[1]; n_pair=(n_bins//2)*2
    if n_pair==0: return np.zeros((M.shape[0],0))
    Me = M[:,:n_pair]; return 0.5*(Me[:,0::2] + Me[:,1::2])

def pair_ranges(R):
    n_pair=(len(R)//2)*2; R=R[:n_pair]; return 0.5*(R[0::2] + R[1::2])

def make_time_ticks(times, n_ticks=Y_TICK_COUNT):
    y = np.arange(len(times), dtype=float)
    if len(times)==0: return y, [], []
    step = max(1, len(times)//max(1, n_ticks))
    tickvals = list(range(0, len(times), step))
    ticktext = [pd.to_datetime(times[i]).strftime("%Y-%m-%d %H:%M:%S") for i in tickvals]
    return y, tickvals, ticktext

def aspect_buttons():
    return [dict(type="buttons", direction="right", x=0.0, y=1.10, xanchor="left", yanchor="top",
                 buttons=[
                     dict(label="Cube (1:1:1)", method="relayout",
                          args=[{"scene.aspectmode":"manual", "scene.aspectratio":dict(x=1,y=1,z=1)}]),
                     dict(label="Breed X (2:1:1)", method="relayout",
                          args=[{"scene.aspectmode":"manual", "scene.aspectratio":dict(x=2,y=1,z=1)}]),
                     dict(label="Hoog Y (1:2:1)", method="relayout",
                          args=[{"scene.aspectmode":"manual", "scene.aspectratio":dict(x=1,y=2,z=1)}]),
                     dict(label="Z uitrekken (1:1:2)", method="relayout",
                          args=[{"scene.aspectmode":"manual", "scene.aspectratio":dict(x=1,y=1,z=2)}]),
                     dict(label="Plat Z (1:1:0.5)", method="relayout",
                          args=[{"scene.aspectmode":"manual", "scene.aspectratio":dict(x=1,y=1,z=0.5)}]),
                     dict(label="Data aspect", method="relayout",
                          args=[{"scene.aspectmode":"data"}])
                 ])]

def z_slider(Z_list, scales):
    steps=[]
    for i,s in enumerate(scales):
        steps.append(dict(method="update", label=f"{s}×",
                          args=[{"z":[Z_list[i]]}, {"title": None}]))
    return [dict(active=scales.index(1.0) if 1.0 in scales else 0,
                 currentvalue={"prefix":"Z-schaal: "}, pad={"t": 40}, steps=steps)]

def surface_beam_datetime(Z, times, html_out, title, beam_label="Beamnummer", z_scales=(0.5,1.0,1.5,2.0,3.0)):
    if Z.size==0: print("[WARN] Lege matrix:", html_out); return
    x = np.arange(1, Z.shape[1]+1)
    y, tickvals, ticktext = make_time_ticks(times, n_ticks=Y_TICK_COUNT)
    XX, YY = np.meshgrid(x, y)
    zmin = np.nanmin(Z); zmax = np.nanmax(Z)
    Z_scaled = [Z * s for s in z_scales]
    fig = go.Figure(data=[go.Surface(x=XX, y=YY, z=Z_scaled[z_scales.index(1.0)] if 1.0 in z_scales else Z,
                                     colorscale=COLOR_SCALE_3D, cmin=zmin, cmax=zmax,
                                     colorbar=dict(title="Backscatter (dB, TVG)"),
                                     customdata=np.array([[pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S") for _ in x] for t in times]),
                                     hovertemplate=("Beam %{x}<br>Tijd %{customdata}<br>Backscatter %{z:.2f} dB<extra></extra>"))])
    fig.update_layout(title=title,
                      scene=dict(xaxis_title=beam_label, yaxis_title="Tijd", zaxis_title="Backscatter (dB, TVG)",
                                 yaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext),
                                 aspectmode="manual", aspectratio=dict(x=1,y=1,z=1)),
                      updatemenus=aspect_buttons(),
                      sliders=z_slider(Z_scaled, list(z_scales)),
                      margin=dict(l=0,r=0,t=60,b=0))
    plot_offline(fig, filename=html_out, auto_open=False,
                 config=dict(displaylogo=False,
                             toImageButtonOptions=dict(format='png',
                                                       filename=html_out.split('/')[-1].replace('.html',''),
                                                       height=700, width=1100, scale=2)))
    print("[OK] 3D HTML:", html_out)

def main():
    # 1) Verzamel paden
    cli_paths = sys.argv[1:]
    existing = [p for p in cli_paths if os.path.isfile(p)]
    missing  = [p for p in cli_paths if not os.path.isfile(p)]
    if missing:
        print("[WARN] Bestanden niet gevonden:", *missing, sep="\n  - ")
    if not existing:
        gui = choose_files_gui()
        existing = gui if gui else []
    if not existing:
        print("[FOUT] Geen bruikbare bestanden geselecteerd/gezien."); return

    # 2) Inlezen en samenvoegen
    all_t, all_M = [], []
    for p in existing:
        t, M = read_dac_like(p)
        if len(t)==0:
            print(f"[WARN] Geen data in: {p}")
            continue
        all_t.append(t); all_M.append(M)
    if not all_t:
        print("[FOUT] Geen bruikbare data in de geselecteerde bestanden."); return

    times, data = concat_chronologically(all_t, all_M)
    if data.size==0:
        print("[FOUT] Lege matrix na merge (mogelijk alleen 0/NaN-kolommen)."); return

    # 3) Berekeningen
    R  = compute_R(data.shape[1])
    E  = counts_to_db(data); ET  = tvg(E, R)
    P  = pair_means(data);  Rpr = pair_ranges(R)
    EP = counts_to_db(P);   EPT = tvg(EP, Rpr)

    # 4) Samenvatting
    print("=== SAMENVATTING ===")
    print(f"  Bestanden ingelezen: {len(existing)}")
    for p in existing: print("   -", p)
    print(f"  Records (tijdstappen): {len(times)}")
    print(f"  Aantal beams (na pruning): {data.shape[1]}")
    print("=====================")

    # 5) CSV-uitvoer
    def save_csv(times, M, prefix, name):
        cols = [f"{prefix}{i+1}" for i in range(M.shape[1])]
        df = pd.DataFrame(M, columns=cols); df.insert(0, "time", pd.to_datetime(times))
        df.to_csv(name, index=False); print("[OK] CSV:", name)
    save_csv(times, data,  "bin_",  "concat_counts.csv")
    save_csv(times, ET,    "bin_",  "concat_bins_TVG_dB.csv")
    save_csv(times, EPT,   "pair_", "concat_pairs_TVG_dB.csv")

    # 6) 3D HTML
    surface_beam_datetime(ET,  times, "backscatter_bins_TVG_3D_interactive.html",  "3D Backscatter per bin (TVG)", "Beamnummer")
    surface_beam_datetime(EPT, times, "backscatter_pairs_TVG_3D_interactive.html", "3D Backscatter kanaalparen (TVG)", "Pairnummer")

if __name__ == "__main__":
    main()
