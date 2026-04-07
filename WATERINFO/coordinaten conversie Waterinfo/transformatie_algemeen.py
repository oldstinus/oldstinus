#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE CRS Converter GUI
Tweerichtingsconversies en batchverwerking tussen:
- BD72 / Lambert 72 (EPSG:31370)
- ETRS89: geo (EPSG:4258), UTM31N (EPSG:25831), Lambert 2008 (EPSG:3812)
- WGS84:  geo (EPSG:4326), UTM31N (EPSG:32631), Web Mercator (EPSG:3857)
- ED50:   geo (EPSG:4230), UTM31N (EPSG:23031)

Extra:
- Interactieve kaart-export (Leaflet via folium), met pop-ups met alle velden.

Vereist:
    pip install pyproj pandas openpyxl folium
"""

import sys
import math
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from pyproj import CRS, Transformer
import folium


# ──────────────────────────────────────────────────────────────────────────────
# CRS-definities
# ──────────────────────────────────────────────────────────────────────────────
CRS_MAP = {
    "BD72 / Lambert72 (EPSG:31370)": CRS.from_epsg(31370),
    "ETRS89 (lat,lon) EPSG:4258": CRS.from_epsg(4258),
    "ETRS89 / UTM31N EPSG:25831": CRS.from_epsg(25831),
    "ETRS89 / Lambert 2008 EPSG:3812": CRS.from_epsg(3812),

    "WGS84 (lat,lon) EPSG:4326": CRS.from_epsg(4326),
    "WGS84 / UTM31N EPSG:32631": CRS.from_epsg(32631),
    "Web Mercator (Google) EPSG:3857": CRS.from_epsg(3857),

    "ED50 (lat,lon) EPSG:4230": CRS.from_epsg(4230),
    "ED50 / UTM31N EPSG:23031": CRS.from_epsg(23031),
}

# Voor UI: invoertypes en welke velden daarbij horen
INPUT_SCHEMAS = {
    "BD72 / Lambert72 (EPSG:31370)": ("X (m)", "Y (m)"),
    "ETRS89 (lat,lon) EPSG:4258": ("Lat (°)", "Lon (°)"),
    "ETRS89 / UTM31N EPSG:25831": ("E (m)", "N (m)"),
    "ETRS89 / Lambert 2008 EPSG:3812": ("X (m)", "Y (m)"),

    "WGS84 (lat,lon) EPSG:4326": ("Lat (°)", "Lon (°)"),
    "WGS84 / UTM31N EPSG:32631": ("E (m)", "N (m)"),
    "Web Mercator (Google) EPSG:3857": ("X (m)", "Y (m)"),

    "ED50 (lat,lon) EPSG:4230": ("Lat (°)", "Lon (°)"),
    "ED50 / UTM31N EPSG:23031": ("E (m)", "N (m)"),
}

# Doelvelden voor output per CRS (kolomnamen)
OUTPUT_FIELDS = {
    "BD72 / Lambert72 (EPSG:31370)": ("X_L72", "Y_L72"),
    "ETRS89 (lat,lon) EPSG:4258": ("ETRS89_lat", "ETRS89_lon"),
    "ETRS89 / UTM31N EPSG:25831": ("ETRS89_UTM31_E", "ETRS89_UTM31_N"),
    "ETRS89 / Lambert 2008 EPSG:3812": ("L08_X", "L08_Y"),

    "WGS84 (lat,lon) EPSG:4326": ("WGS84_lat", "WGS84_lon"),
    "WGS84 / UTM31N EPSG:32631": ("WGS84_UTM31_E", "WGS84_UTM31_N"),
    "Web Mercator (Google) EPSG:3857": ("WM_X", "WM_Y"),

    "ED50 (lat,lon) EPSG:4230": ("ED50_lat", "ED50_lon"),
    "ED50 / UTM31N EPSG:23031": ("ED50_UTM31_E", "ED50_UTM31_N"),
}


def make_transformer(crs_from: CRS, crs_to: CRS) -> Transformer:
    # always_xy=True ⇒ volgorde is (lon/easting, lat/north) voor zowel geo als projected
    return Transformer.from_crs(crs_from, crs_to, always_xy=True)


def convert_pair(crs_from_key: str, crs_to_key: str, a: float, b: float):
    """Converteer één coördinaatpaar van crs_from → crs_to.
       a,b zijn (X,Y) voor projected, (lon,lat) voor geo vanwege always_xy.
       Let op: UI levert Lat/Lon in natuurlijke volgorde; we draaien zonodig om.
    """
    crs_from = CRS_MAP[crs_from_key]
    crs_to = CRS_MAP[crs_to_key]
    tr = make_transformer(crs_from, crs_to)
    x_out, y_out = tr.transform(a, b)
    return x_out, y_out


def dms_str(dd: float, is_lat=True) -> str:
    neg = dd < 0
    dd = abs(dd)
    d = int(dd)
    m_float = (dd - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    hemi = ("S" if neg else "N") if is_lat else ("W" if neg else "E")
    return f"{d}° {m:02d}′ {s:05.2f}″ {hemi}"


def geo_for_map(row: dict):
    """Bepaal lat/lon voor kaart: prefereer WGS84 lat/lon, anders ETRS89 lat/lon → naar WGS84."""
    if pd.notna(row.get("WGS84_lat")) and pd.notna(row.get("WGS84_lon")):
        return float(row["WGS84_lat"]), float(row["WGS84_lon"])
    elif pd.notna(row.get("ETRS89_lat")) and pd.notna(row.get("ETRS89_lon")):
        # Transformeer ETRS89 → WGS84 voor kaart (verschil is cm-orde; voor kaart ~identiek)
        lon, lat = make_transformer(CRS_MAP["ETRS89 (lat,lon) EPSG:4258"],
                                    CRS_MAP["WGS84 (lat,lon) EPSG:4326"]).transform(
                                        float(row["ETRS89_lon"]), float(row["ETRS89_lat"])
                                    )
        return lat, lon
    else:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("België CRS Converter • BD72 ⇄ ETRS89 ⇄ WGS84 ⇄ ED50 • + Kaart")
        self.geometry("980x720")
        self.minsize(940, 700)

        nb = ttk.Notebook(self)
        self.tab_point = ttk.Frame(nb)
        self.tab_batch = ttk.Frame(nb)
        self.tab_map = ttk.Frame(nb)
        nb.add(self.tab_point, text="Enkel punt")
        nb.add(self.tab_batch, text="CSV/XLSX batch")
        nb.add(self.tab_map, text="Kaart-export")
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_point_tab()
        self._build_batch_tab()
        self._build_map_tab()

        foot = ttk.Frame(self); foot.pack(fill="x", padx=8, pady=(0,8))
        ttk.Label(foot, text="EPSG-conforme transformaties via pyproj/PROJ • Nauwkeurigheid: meterorde in BE (sub-meter met gridshifts).").pack(side="left")

        # data cache voor kaartexport (laatste batchresultaat of enkel punt)
        self.last_df = None
        self.last_single = None  # dict met outputs van enkel punt

    # ── Enkel punt ────────────────────────────────────────────────────────────
    def _build_point_tab(self):
        frm = ttk.Frame(self.tab_point); frm.pack(fill="both", expand=True, padx=10, pady=10)

        inf = ttk.LabelFrame(frm, text="Invoer")
        inf.pack(fill="x", padx=4, pady=4)
        ttk.Label(inf, text="Invoer-CRS:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.in_crs_combo = ttk.Combobox(inf, state="readonly", width=36, values=list(CRS_MAP.keys()))
        self.in_crs_combo.set("BD72 / Lambert72 (EPSG:31370)")
        self.in_crs_combo.grid(row=0, column=1, padx=6, pady=6, sticky="w")
        self.in_crs_combo.bind("<<ComboboxSelected>>", lambda e: self._rebuild_input_fields())

        self.in_fields_frame = ttk.Frame(inf); self.in_fields_frame.grid(row=1, column=0, columnspan=2, sticky="we", padx=6, pady=6)
        self._rebuild_input_fields()

        opt = ttk.LabelFrame(frm, text="Doel-CRS (vink aan wat je wil)")
        opt.pack(fill="x", padx=4, pady=6)
        self.out_vars = {}
        for i, key in enumerate(CRS_MAP.keys()):
            var = tk.BooleanVar(value=(key in ["WGS84 (lat,lon) EPSG:4326", "WGS84 / UTM31N EPSG:32631", "BD72 / Lambert72 (EPSG:31370)"]))
            self.out_vars[key] = var
            ttk.Checkbutton(opt, text=key, variable=var).grid(row=i//2, column=i%2, sticky="w", padx=6, pady=3)

        btnf = ttk.Frame(frm); btnf.pack(fill="x", padx=4, pady=4)
        ttk.Button(btnf, text="Converteer", command=self._convert_point).pack(side="left", padx=4)
        ttk.Button(btnf, text="Wis", command=self._clear_point).pack(side="left", padx=4)

        outf = ttk.LabelFrame(frm, text="Resultaat")
        outf.pack(fill="both", expand=True, padx=4, pady=4)
        self.point_text = tk.Text(outf, height=14, wrap="none", font=("Consolas", 10))
        self.point_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _rebuild_input_fields(self):
        for w in self.in_fields_frame.winfo_children():
            w.destroy()
        key = self.in_crs_combo.get()
        a_label, b_label = INPUT_SCHEMAS[key]
        ttk.Label(self.in_fields_frame, text=f"{a_label}:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Label(self.in_fields_frame, text=f"{b_label}:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        self.in_a = ttk.Entry(self.in_fields_frame, width=24); self.in_a.grid(row=0, column=1, padx=6, pady=6, sticky="w")
        self.in_b = ttk.Entry(self.in_fields_frame, width=24); self.in_b.grid(row=1, column=1, padx=6, pady=6, sticky="w")

    def _clear_point(self):
        self.point_text.delete("1.0", "end")
        self.in_a.delete(0, "end"); self.in_b.delete(0, "end")
        self.last_single = None

    def _convert_point(self):
        self.point_text.delete("1.0", "end")
        try:
            key_in = self.in_crs_combo.get()
            a = float(self.in_a.get().strip())
            b = float(self.in_b.get().strip())

            # Voor geo-invoer in UI is volgorde Lat,Lon; Transformer verwacht (lon,lat)
            if "lat,lon" in key_in:
                a_for_tr, b_for_tr = float(self.in_b.get().strip()), float(self.in_a.get().strip())  # lon, lat
            else:
                a_for_tr, b_for_tr = a, b

            results = {}
            for key_out, flag in self.out_vars.items():
                if not flag.get():
                    continue
                x, y = convert_pair(key_in, key_out, a_for_tr, b_for_tr)
                # Presenteer netjes (lat,lon) voor geo, (E,N) of (X,Y) voor projected
                fa, fb = OUTPUT_FIELDS[key_out]
                if "lat,lon" in key_out:
                    # x=lon, y=lat
                    results[fa] = y
                    results[fb] = x
                else:
                    results[fa] = x
                    results[fb] = y

            # Bouw leesbare output
            lines = [f"Invoer ({key_in}): {INPUT_SCHEMAS[key_in][0]}={a}, {INPUT_SCHEMAS[key_in][1]}={b}"]
            for k in CRS_MAP.keys():
                if k not in results or OUTPUT_FIELDS[k][0] not in results:
                    continue
                fa, fb = OUTPUT_FIELDS[k]
                va = results[fa]; vb = results[fb]
                if "lat,lon" in k:
                    lines.append(f"{k}: lat={va:.9f}, lon={vb:.9f}")
                    lines.append(f"  DMS: {dms_str(va,True)}, {dms_str(vb,False)}")
                else:
                    lines.append(f"{k}: {fa}={va:.3f}, {fb}={vb:.3f}")

            self.point_text.insert("1.0", "\n".join(lines))
            # Cache voor kaart
            self.last_single = results

        except Exception as ex:
            messagebox.showerror("Fout bij conversie", f"{ex}\n\n{traceback.format_exc()}")

    # ── Batch ─────────────────────────────────────────────────────────────────
    def _build_batch_tab(self):
        frm = ttk.Frame(self.tab_batch); frm.pack(fill="both", expand=True, padx=10, pady=10)

        bf = ttk.LabelFrame(frm, text="Bronbestand (CSV/XLSX)")
        bf.pack(fill="x", padx=4, pady=4)
        self.src_path_var = tk.StringVar()
        ttk.Entry(bf, textvariable=self.src_path_var, width=72).grid(row=0, column=0, padx=6, pady=6, sticky="we")
        ttk.Button(bf, text="Bladeren…", command=self._browse_src).grid(row=0, column=1, padx=6, pady=6)
        bf.grid_columnconfigure(0, weight=1)

        inf = ttk.LabelFrame(frm, text="Invoer-CRS & kolommen")
        inf.pack(fill="x", padx=4, pady=6)
        ttk.Label(inf, text="Invoer-CRS:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.batch_in_combo = ttk.Combobox(inf, state="readonly", width=36, values=list(CRS_MAP.keys()))
        self.batch_in_combo.set("BD72 / Lambert72 (EPSG:31370)")
        self.batch_in_combo.grid(row=0, column=1, padx=6, pady=6, sticky="w")
        self.batch_in_combo.bind("<<ComboboxSelected>>", lambda e: self._build_batch_cols())

        self.cols_frame = ttk.Frame(inf); self.cols_frame.grid(row=1, column=0, columnspan=4, sticky="we", padx=6, pady=6)
        self._build_batch_cols()

        opt = ttk.LabelFrame(frm, text="Doel-CRS (meerdere mogelijk)")
        opt.pack(fill="x", padx=4, pady=6)
        self.out_vars_b = {}
        for i, key in enumerate(CRS_MAP.keys()):
            var = tk.BooleanVar(value=(key in ["WGS84 (lat,lon) EPSG:4326", "WGS84 / UTM31N EPSG:32631", "BD72 / Lambert72 (EPSG:31370)"]))
            self.out_vars_b[key] = var
            ttk.Checkbutton(opt, text=key, variable=var).grid(row=i//2, column=i%2, sticky="w", padx=6, pady=3)

        of = ttk.LabelFrame(frm, text="Doelbestand")
        of.pack(fill="x", padx=4, pady=4)
        self.dst_path_var = tk.StringVar()
        ttk.Entry(of, textvariable=self.dst_path_var, width=72).grid(row=0, column=0, padx=6, pady=6, sticky="we")
        ttk.Button(of, text="Opslaan als…", command=self._browse_dst).grid(row=0, column=1, padx=6, pady=6)
        of.grid_columnconfigure(0, weight=1)

        btnf = ttk.Frame(frm); btnf.pack(fill="x", padx=4, pady=6)
        ttk.Button(btnf, text="Converteer bestand", command=self._convert_file).pack(side="left", padx=4)

        outf = ttk.LabelFrame(frm, text="Log & preview (top 10)")
        outf.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text = tk.Text(outf, height=14, wrap="none", font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_batch_cols(self):
        for w in self.cols_frame.winfo_children():
            w.destroy()
        key = self.batch_in_combo.get()
        a_label, b_label = INPUT_SCHEMAS[key]
        ttk.Label(self.cols_frame, text=f"{a_label} kolom:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Label(self.cols_frame, text=f"{b_label} kolom:").grid(row=0, column=2, padx=6, pady=6, sticky="e")
        self.col_a = ttk.Combobox(self.cols_frame, state="readonly", width=28, values=[])
        self.col_b = ttk.Combobox(self.cols_frame, state="readonly", width=28, values=[])
        self.col_a.grid(row=0, column=1, padx=6, pady=6, sticky="w")
        self.col_b.grid(row=0, column=3, padx=6, pady=6, sticky="w")

    def _browse_src(self):
        path = filedialog.askopenfilename(
            title="Kies CSV of Excel",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx;*.xls"), ("Alle bestanden", "*.*")]
        )
        if not path: return
        self.src_path_var.set(path)
        try:
            df = pd.read_excel(path) if Path(path).suffix.lower() in (".xlsx",".xls") else pd.read_csv(path)
            cols = list(df.columns)
            self.col_a["values"] = cols; self.col_b["values"] = cols
            # simpele auto-guess
            key = self.batch_in_combo.get().lower()
            if "lat,lon" in key:
                self.col_a.set(_guess(cols, ["lat","latitude"]) or "")
                self.col_b.set(_guess(cols, ["lon","lng","longitude"]) or "")
            elif "utm" in key:
                self.col_a.set(_guess(cols, ["e","easting","utm_e"]) or "")
                self.col_b.set(_guess(cols, ["n","northing","utm_n"]) or "")
            else:
                # projected X,Y
                self.col_a.set(_guess(cols, ["x","x_l72","lambert","l72","l08","wm_x"]) or "")
                self.col_b.set(_guess(cols, ["y","y_l72","lambert","l72","l08","wm_y"]) or "")
            # preview
            self.log_text.delete("1.0","end")
            self.log_text.insert("1.0", f"Ingelezen kolommen: {cols}\n\nPreview:\n")
            self.log_text.insert("end", df.head(10).to_string(index=False))
        except Exception as ex:
            messagebox.showerror("Inleesfout", f"{ex}\n\n{traceback.format_exc()}")

    def _browse_dst(self):
        path = filedialog.asksaveasfilename(
            title="Opslaan als",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("Excel","*.xlsx")]
        )
        if path: self.dst_path_var.set(path)

    def _convert_file(self):
        src = self.src_path_var.get().strip()
        if not src:
            messagebox.showerror("Fout","Kies eerst een bronbestand."); return
        dst = self.dst_path_var.get().strip()
        if not dst:
            dst = str(Path(src).with_name(Path(src).stem + "_converted.csv"))
            self.dst_path_var.set(dst)

        key_in = self.batch_in_combo.get()
        col_a = self.col_a.get().strip()
        col_b = self.col_b.get().strip()
        if not col_a or not col_b:
            messagebox.showerror("Fout", "Selecteer de kolommen voor de invoercoördinaten."); return

        try:
            df = pd.read_excel(src) if Path(src).suffix.lower() in (".xlsx",".xls") else pd.read_csv(src)
            out_df = df.copy()

            # Doel-CRS die aangevinkt zijn
            targets = [k for k,v in self.out_vars_b.items() if v.get()]
            # Maak kolommen aan
            for k in targets:
                fa, fb = OUTPUT_FIELDS[k]
                out_df[fa] = pd.NA; out_df[fb] = pd.NA

            crs_in = CRS_MAP[key_in]
            tr_cache = {}

            def tr_to(target_key):
                if target_key not in tr_cache:
                    tr_cache[target_key] = make_transformer(crs_in, CRS_MAP[target_key])
                return tr_cache[target_key]

            for idx, row in df.iterrows():
                try:
                    a_val = float(row[col_a]); b_val = float(row[col_b])
                except Exception:
                    continue
                # voor geo-invoer: UI kolom a=Lat, b=Lon, maar transformer verwacht (lon,lat)
                if "lat,lon" in key_in:
                    a_use, b_use = b_val, a_val
                else:
                    a_use, b_use = a_val, b_val

                for k in targets:
                    fa, fb = OUTPUT_FIELDS[k]
                    tr = tr_to(k)
                    x, y = tr.transform(a_use, b_use)
                    if "lat,lon" in k:
                        out_df.at[idx, fa] = y
                        out_df.at[idx, fb] = x
                    else:
                        out_df.at[idx, fa] = x
                        out_df.at[idx, fb] = y

            # schrijven
            if Path(dst).suffix.lower()==".xlsx":
                out_df.to_excel(dst, index=False)
            else:
                out_df.to_csv(dst, index=False)

            self.last_df = out_df  # cache voor kaart
            self.log_text.delete("1.0","end")
            self.log_text.insert("1.0", f"Gereed: {dst}\n")
            self.log_text.insert("end", out_df.head(10).to_string(index=False))
            messagebox.showinfo("Succes", f"Conversie voltooid.\nBestand geschreven: {dst}")

        except Exception as ex:
            messagebox.showerror("Conversiefout", f"{ex}\n\n{traceback.format_exc()}")

    # ── Kaart-export ──────────────────────────────────────────────────────────
    def _build_map_tab(self):
        frm = ttk.Frame(self.tab_map); frm.pack(fill="both", expand=True, padx=10, pady=10)

        info = ttk.LabelFrame(frm, text="Bron voor kaart")
        info.pack(fill="x", padx=4, pady=4)
        ttk.Label(info, text="Gebruik laatste batchresultaat of laatste enkel-punt conversie. "
                             "Voor batch: vink WGS84 (lat,lon) mee aan voor beste kaartlabels.").pack(anchor="w", padx=6, pady=6)

        of = ttk.LabelFrame(frm, text="Kaart-export")
        of.pack(fill="x", padx=4, pady=6)
        ttk.Label(of, text="Bestandsnaam (.html):").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.map_path = tk.StringVar()
        ttk.Entry(of, textvariable=self.map_path, width=60).grid(row=0, column=1, padx=6, pady=6, sticky="we")
        ttk.Button(of, text="Opslaan als…", command=self._browse_map).grid(row=0, column=2, padx=6, pady=6)

        ttk.Button(of, text="Maak kaart", command=self._make_map).grid(row=1, column=1, padx=6, pady=6)

        self.map_log = tk.Text(frm, height=16, wrap="none", font=("Consolas", 10))
        self.map_log.pack(fill="both", expand=True, padx=6, pady=6)

    def _browse_map(self):
        path = filedialog.asksaveasfilename(
            title="Opslaan als",
            defaultextension=".html",
            filetypes=[("HTML","*.html")]
        )
        if path: self.map_path.set(path)

    def _make_map(self):
        self.map_log.delete("1.0","end")
        try:
            if self.last_df is not None and not self.last_df.empty:
                df = self.last_df
                pts = []
                for _, row in df.iterrows():
                    lat, lon = geo_for_map(row)
                    if lat is None: continue
                    # popup met kernvelden
                    popup = []
                    for k in ["WGS84_lat","WGS84_lon","X_L72","Y_L72","L08_X","L08_Y",
                              "ETRS89_lat","ETRS89_lon","ED50_lat","ED50_lon",
                              "WGS84_UTM31_E","WGS84_UTM31_N",
                              "ETRS89_UTM31_E","ETRS89_UTM31_N",
                              "ED50_UTM31_E","ED50_UTM31_N",
                              "WM_X","WM_Y"]:
                        if k in df.columns and pd.notna(row.get(k)):
                            popup.append(f"{k}: {row.get(k)}")
                    pts.append((lat, lon, "<br>".join(popup)))
                if not pts:
                    raise ValueError("Geen punten met (WGS84/ETRS) geo-coördinaten gevonden in het laatst geconverteerde bestand.")

                m = folium.Map(location=[sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts)], zoom_start=11)
                for lat, lon, pop in pts:
                    folium.Marker([lat, lon], popup=folium.Popup(pop, max_width=420)).add_to(m)

                out = self.map_path.get().strip() or "kaart_export.html"
                m.save(out)
                self.map_log.insert("1.0", f"Kaart geschreven: {out}\nPunten: {len(pts)}\n")

            elif self.last_single is not None:
                row = self.last_single
                # Probeer WGS84, anders ETRS
                if "WGS84_lat" in row and "WGS84_lon" in row:
                    lat, lon = float(row["WGS84_lat"]), float(row["WGS84_lon"])
                elif "ETRS89_lat" in row and "ETRS89_lon" in row:
                    # ETRS89 → WGS84
                    lon_w, lat_w = make_transformer(CRS_MAP["ETRS89 (lat,lon) EPSG:4258"],
                                                    CRS_MAP["WGS84 (lat,lon) EPSG:4326"]).transform(
                                                        float(row["ETRS89_lon"]), float(row["ETRS89_lat"])
                                                    )
                    lat, lon = lat_w, lon_w
                else:
                    raise ValueError("Geen geo-coördinaat (WGS84/ETRS89) aanwezig in laatste enkel-punt conversie.")

                m = folium.Map(location=[lat, lon], zoom_start=14)
                # popup
                popup = "<br>".join([f"{k}: {v}" for k,v in row.items()])
                folium.Marker([lat, lon], popup=folium.Popup(popup, max_width=420)).add_to(m)
                out = self.map_path.get().strip() or "kaart_export.html"
                m.save(out)
                self.map_log.insert("1.0", f"Kaart geschreven: {out}\n1 punt.\n")
            else:
                raise ValueError("Geen data beschikbaar. Converteer eerst een punt of een bestand.")

        except Exception as ex:
            messagebox.showerror("Kaartfout", f"{ex}\n\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────────────────────
# Hulpfuncties
# ──────────────────────────────────────────────────────────────────────────────
def _guess(columns, patterns):
    lc = {c.lower(): c for c in columns}
    for p in patterns:
        if p in lc: return lc[p]
    for c in columns:
        l = c.lower()
        if any(p in l for p in patterns):
            return c
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception as e:
        print("Onherstelbare fout:", e, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
