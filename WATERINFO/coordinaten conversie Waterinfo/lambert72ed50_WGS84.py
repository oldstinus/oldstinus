#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lambert72 (BD72, EPSG:31370) → WGS84 (EPSG:4326) en optioneel UTM31N (EPSG:32631)
GUI-tool met:
  - Enkel punt: X,Y ingave
  - CSV/XLSX batch: bestand kiezen, X/Y-kolommen selecteren, output opslaan

Nauwkeurigheid:
  - Gebruikt pyproj/PROJ met EPSG-gedefinieerde transformatie voor 31370→4326.
  - Typisch ~meterorde in België (afhankelijk van beschikbare gridshifts in jouw PROJ-installatie).

Vereist:
  pip install pyproj pandas openpyxl
"""

import sys
import math
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import pandas as pd
from pyproj import CRS, Transformer


# ── CRS-definities ─────────────────────────────────────────────────────────────
CRS_L72 = CRS.from_epsg(31370)       # Belgian Lambert 72 (BD72)
CRS_WGS84 = CRS.from_epsg(4326)      # WGS84 geographic
CRS_UTM31_WGS84 = CRS.from_epsg(32631)  # WGS84 / UTM zone 31N

# EPSG-conforme transformers; always_xy=True => (Easting/X, Northing/Y) volgorde
T_L72_to_WGS84 = Transformer.from_crs(CRS_L72, CRS_WGS84, always_xy=True)
T_L72_to_UTM31 = Transformer.from_crs(CRS_L72, CRS_UTM31_WGS84, always_xy=True)


# ── Kernfuncties ───────────────────────────────────────────────────────────────
def convert_point_l72(x, y, want_wgs=True, want_utm=False):
    """
    Converteer één Lambert72-punt (X,Y in meters) naar:
      - WGS84 (lat, lon)  [indien want_wgs=True]
      - UTM31N (E, N)     [indien want_utm=True]
    Retourneert dict met gevraagde velden.
    """
    out = {"X_L72": x, "Y_L72": y}
    if want_wgs:
        # pyproj geeft bij (always_xy=True) terug: (lon, lat)
        lon, lat = T_L72_to_WGS84.transform(x, y)
        out.update({
            "WGS84_lat": lat,
            "WGS84_lon": lon,
        })
    if want_utm:
        E, N = T_L72_to_UTM31.transform(x, y)
        out.update({
            "UTM31_WGS84_E": E,
            "UTM31_WGS84_N": N,
        })
    return out


def df_convert_l72(df, xcol, ycol, want_wgs=True, want_utm=False):
    """
    Converteer alle rijen in DataFrame df met Lambert72-kolommen xcol,ycol.
    Geeft een kopie terug met extra kolommen voor de gevraagde systemen.
    """
    out_df = df.copy()
    # Vooraf kolommen aanmaken zodat dtype consistent blijft
    if want_wgs:
        out_df["WGS84_lat"] = pd.NA
        out_df["WGS84_lon"] = pd.NA
    if want_utm:
        out_df["UTM31_WGS84_E"] = pd.NA
        out_df["UTM31_WGS84_N"] = pd.NA

    for idx, row in df.iterrows():
        try:
            x = float(row[xcol])
            y = float(row[ycol])
        except Exception:
            continue  # sla lege/ongeldige over

        conv = convert_point_l72(x, y, want_wgs=want_wgs, want_utm=want_utm)
        for k, v in conv.items():
            if k in out_df.columns:
                out_df.at[idx, k] = v
            else:
                out_df[k] = pd.NA
                out_df.at[idx, k] = v

    return out_df


# ── GUI ────────────────────────────────────────────────────────────────────────
class L72ConverterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lambert72 → WGS84 (en UTM31N) • EPSG-conform")
        self.geometry("860x600")
        self.minsize(820, 560)

        # Tabs
        nb = ttk.Notebook(self)
        self.tab_point = ttk.Frame(nb)
        self.tab_csv = ttk.Frame(nb)
        nb.add(self.tab_point, text="Enkel punt")
        nb.add(self.tab_csv, text="CSV/XLSX batch")
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_point_tab()
        self._build_csv_tab()

        # Footer
        footer = ttk.Frame(self)
        footer.pack(fill="x", padx=8, pady=(0, 8))
        lab = ttk.Label(
            footer,
            text="EPSG:31370 → 4326 (en optioneel 32631) via pyproj/PROJ • Nauwkeurigheid ~meterorde in België."
        )
        lab.pack(side="left")

    # ── Tab: Enkel punt ────────────────────────────────────────────────────────
    def _build_point_tab(self):
        frm = ttk.Frame(self.tab_point)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # Invoer
        inf = ttk.LabelFrame(frm, text="Invoer (Lambert 72 / BD72, EPSG:31370)")
        inf.pack(fill="x", padx=4, pady=4)

        self.x_entry = ttk.Entry(inf, width=20)
        self.y_entry = ttk.Entry(inf, width=20)
        ttk.Label(inf, text="X (m):").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.x_entry.grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(inf, text="Y (m):").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        self.y_entry.grid(row=1, column=1, padx=6, pady=6, sticky="w")

        # Opties
        opt = ttk.LabelFrame(frm, text="Uitvoeropties")
        opt.pack(fill="x", padx=4, pady=6)

        self.want_wgs_var = tk.BooleanVar(value=True)
        self.want_utm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="WGS84 (lat, lon)", variable=self.want_wgs_var).grid(
            row=0, column=0, padx=6, pady=6, sticky="w"
        )
        ttk.Checkbutton(opt, text="WGS84 / UTM31N (E, N)", variable=self.want_utm_var).grid(
            row=0, column=1, padx=6, pady=6, sticky="w"
        )

        # Actieknoppen
        btnf = ttk.Frame(frm)
        btnf.pack(fill="x", padx=4, pady=4)
        ttk.Button(btnf, text="Converteer", command=self.convert_point).pack(side="left", padx=4)
        ttk.Button(btnf, text="Wis velden", command=self.clear_point).pack(side="left", padx=4)

        # Output
        outf = ttk.LabelFrame(frm, text="Resultaat")
        outf.pack(fill="both", expand=True, padx=4, pady=4)

        self.out_text = tk.Text(outf, height=10, wrap="none")
        self.out_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.out_text.configure(font=("Consolas", 10))

    def clear_point(self):
        self.x_entry.delete(0, "end")
        self.y_entry.delete(0, "end")
        self.out_text.delete("1.0", "end")

    def convert_point(self):
        self.out_text.delete("1.0", "end")
        try:
            x = float(self.x_entry.get().strip())
            y = float(self.y_entry.get().strip())
        except Exception:
            messagebox.showerror("Fout", "Geef geldige numerieke X en Y in (meters, Lambert72).")
            return

        try:
            res = convert_point_l72(
                x, y,
                want_wgs=self.want_wgs_var.get(),
                want_utm=self.want_utm_var.get()
            )
            lines = []
            lines.append(f"Invoer (Lambert72): X={res['X_L72']:.3f} m, Y={res['Y_L72']:.3f} m")
            if self.want_wgs_var.get():
                lat = res["WGS84_lat"]
                lon = res["WGS84_lon"]
                lines.append(f"WGS84 (lat, lon): {lat:.9f}, {lon:.9f}")
                lines.append(f"WGS84 (DMS):  {deg2dms(lat, is_lat=True)}, {deg2dms(lon, is_lat=False)}")
            if self.want_utm_var.get():
                e = res["UTM31_WGS84_E"]
                n = res["UTM31_WGS84_N"]
                lines.append(f"UTM31N (WGS84): E={e:.3f} m, N={n:.3f} m")

            self.out_text.insert("1.0", "\n".join(lines))
        except Exception as ex:
            messagebox.showerror("Fout bij conversie", f"{ex}\n\n{traceback.format_exc()}")

    # ── Tab: CSV/XLSX batch ────────────────────────────────────────────────────
    def _build_csv_tab(self):
        frm = ttk.Frame(self.tab_csv)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # Bronbestand
        bf = ttk.LabelFrame(frm, text="Bronbestand (CSV of XLSX) met Lambert72 X,Y")
        bf.pack(fill="x", padx=4, pady=4)

        self.src_path_var = tk.StringVar()
        ttk.Entry(bf, textvariable=self.src_path_var, width=70).grid(row=0, column=0, padx=6, pady=6, sticky="we")
        ttk.Button(bf, text="Bladeren…", command=self.browse_src).grid(row=0, column=1, padx=6, pady=6)
        bf.grid_columnconfigure(0, weight=1)

        self.cols_frame = ttk.Frame(bf)
        self.cols_frame.grid(row=1, column=0, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Label(self.cols_frame, text="X-kolom:").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Label(self.cols_frame, text="Y-kolom:").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        self.xcol_combo = ttk.Combobox(self.cols_frame, state="readonly", width=30, values=[])
        self.ycol_combo = ttk.Combobox(self.cols_frame, state="readonly", width=30, values=[])
        self.xcol_combo.grid(row=0, column=1, padx=4, pady=4, sticky="w")
        self.ycol_combo.grid(row=0, column=3, padx=4, pady=4, sticky="w")

        # Uitvoeropties
        opt = ttk.LabelFrame(frm, text="Uitvoeropties")
        opt.pack(fill="x", padx=4, pady=6)
        self.want_wgs_var2 = tk.BooleanVar(value=True)
        self.want_utm_var2 = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="WGS84 (lat, lon)", variable=self.want_wgs_var2).grid(
            row=0, column=0, padx=6, pady=6, sticky="w"
        )
        ttk.Checkbutton(opt, text="WGS84 / UTM31N (E, N)", variable=self.want_utm_var2).grid(
            row=0, column=1, padx=6, pady=6, sticky="w"
        )

        # Doelbestand
        of = ttk.LabelFrame(frm, text="Doelbestand")
        of.pack(fill="x", padx=4, pady=4)
        self.dst_path_var = tk.StringVar()
        ttk.Entry(of, textvariable=self.dst_path_var, width=70).grid(row=0, column=0, padx=6, pady=6, sticky="we")
        ttk.Button(of, text="Opslaan als…", command=self.browse_dst).grid(row=0, column=1, padx=6, pady=6)
        of.grid_columnconfigure(0, weight=1)

        # Actie
        btnf = ttk.Frame(frm)
        btnf.pack(fill="x", padx=4, pady=6)
        ttk.Button(btnf, text="Converteer bestand", command=self.convert_file).pack(side="left", padx=4)

        # Log/preview
        outf = ttk.LabelFrame(frm, text="Log & preview (eerste 10 rijen)")
        outf.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text = tk.Text(outf, height=12, wrap="none")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.log_text.configure(font=("Consolas", 10))

    def browse_src(self):
        path = filedialog.askopenfilename(
            title="Kies CSV of Excel",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx;*.xls"), ("Alle bestanden", "*.*")]
        )
        if not path:
            return
        self.src_path_var.set(path)
        try:
            df, cols = self._read_preview_columns(Path(path))
            # Stel comboboxen
            self.xcol_combo["values"] = cols
            self.ycol_combo["values"] = cols
            # Auto-detecteer X/Y
            guess_x = _guess_col(cols, ["x", "x_l72", "lambertx", "lambert_x", "e", "easting"])
            guess_y = _guess_col(cols, ["y", "y_l72", "lamberty", "lambert_y", "n", "northing"])
            if guess_x: self.xcol_combo.set(guess_x)
            if guess_y: self.ycol_combo.set(guess_y)
            # Preview
            self.log_text.delete("1.0", "end")
            self.log_text.insert("1.0", f"Ingelezen kolommen: {cols}\n\nPreview (top 10):\n")
            self.log_text.insert("end", df.head(10).to_string(index=False))
        except Exception as ex:
            messagebox.showerror("Inleesfout", f"{ex}\n\n{traceback.format_exc()}")

    def _read_preview_columns(self, path: Path):
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        cols = list(df.columns)
        return df, cols

    def browse_dst(self):
        path = filedialog.asksaveasfilename(
            title="Opslaan als",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")]
        )
        if not path:
            return
        self.dst_path_var.set(path)

    def convert_file(self):
        src = self.src_path_var.get().strip()
        dst = self.dst_path_var.get().strip()
        xcol = self.xcol_combo.get().strip()
        ycol = self.ycol_combo.get().strip()
        want_wgs = self.want_wgs_var2.get()
        want_utm = self.want_utm_var2.get()

        if not src:
            messagebox.showerror("Fout", "Kies eerst een bronbestand.")
            return
        if not xcol or not ycol:
            messagebox.showerror("Fout", "Selecteer de X- en Y-kolommen.")
            return
        if not dst:
            # standaard naast bronbestand
            dst = str(Path(src).with_name(Path(src).stem + "_converted.csv"))
            self.dst_path_var.set(dst)

        try:
            # Inlezen
            if Path(src).suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(src)
            else:
                df = pd.read_csv(src)

            # Conversie
            out_df = df_convert_l72(df, xcol, ycol, want_wgs=want_wgs, want_utm=want_utm)

            # Wegschrijven
            if Path(dst).suffix.lower() == ".xlsx":
                out_df.to_excel(dst, index=False)
            else:
                out_df.to_csv(dst, index=False)

            # Log
            self.log_text.delete("1.0", "end")
            self.log_text.insert("1.0", f"Gereed: {dst}\n")
            self.log_text.insert("end", out_df.head(10).to_string(index=False))

            messagebox.showinfo("Succes", f"Conversie voltooid.\nBestand geschreven: {dst}")
        except Exception as ex:
            messagebox.showerror("Conversiefout", f"{ex}\n\n{traceback.format_exc()}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _guess_col(columns, candidates_lower):
    """Kies automatisch een kolomnaam uit 'columns' op basis van kandidaatstrings."""
    lc_map = {c.lower(): c for c in columns}
    for cand in candidates_lower:
        # exact
        if cand in lc_map:
            return lc_map[cand]
    # bevat
    for c in columns:
        lc = c.lower()
        if any(cand in lc for cand in candidates_lower):
            return c
    return None


def deg2dms(dd, is_lat=True):
    """Decimal degrees → DMS-string."""
    neg = dd < 0
    dd = abs(dd)
    d = int(dd)
    m_float = (dd - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    hemi = ""
    if is_lat:
        hemi = "S" if neg else "N"
    else:
        hemi = "W" if neg else "E"
    return f"{d}° {m:02d}′ {s:05.2f}″ {hemi}"


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        app = L72ConverterGUI()
        app.mainloop()
    except Exception as e:
        print("Onherstelbare fout in de GUI:", e, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
