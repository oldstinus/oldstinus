#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M9 (.mat) + Stationaire RTK-punten (.txt/.csv met ';') koppelen op tijd.

Scenario:
- De M9 heeft geen geïntegreerde GPS: enkel bottom-track XY (relatief) in Summary.Track.
- Je hebt slechts enkele stationaire RTK-punten (E,N,H) opgenomen (bv. Sokkia) met tijd+datum.
- We gebruiken RTK-punten als "anchors" om offsets te bepalen op basis van tijdmatching.
- Voor elk M9-ensemble: gebruik de laatst beschikbare offset (forward-fill).
- Buiten RTK-tijdbereik:
    - na laatste RTK: laatste offset blijft gelden
    - vóór eerste RTK: eerste offset wordt gebruikt

Hoogte:
- In dit script GEEN -2.0 m correctie.
- Wel een vaste hoogte-correctie van +0.05 m (default) op RTK_H:
    boat_H = RTK_H + 0.05

Outputs:
- <out>.csv         : per M9-ensemble absolute Lambert72 (E,N) + (optioneel) diepte/bedhoogte + lat/lon
- <out>_anchors.csv : de RTK anchors met berekende offsets
- <out>_map.html    : (optioneel) folium kaart met M9 track + RTK anchor markers

RTK puntenbestand (voorbeeld uit Alles.txt):
- records zijn ';' gescheiden:
  idx ; omschrijving ; East ; North ; Height ; sigmaE ; sigmaN ; HH:MM:SS ; YYYY/MM/DD
  fileciteturn0file0

Tijdzone:
- RTK-tijden in zo'n export zijn vaak "lokale tijd" zonder timezone.
- Optie: interpreteer RTK tijd als Europe/Brussels en converteer naar UTC (default aan).
- M9 tijd: System.Time is een epoch (s sinds 2000-01-01). Optie om 1 uur af te trekken.

"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import importlib.util


_HTML_STATE_HELPER_ROOT = Path(__file__).resolve().parents[1]
if not (_HTML_STATE_HELPER_ROOT / "html_state_saver.py").exists():
    _HTML_STATE_HELPER_ROOT = Path(__file__).resolve().parents[2]
_HTML_STATE_HELPER_SPEC = importlib.util.spec_from_file_location(
    "html_state_saver", _HTML_STATE_HELPER_ROOT / "html_state_saver.py"
)
if _HTML_STATE_HELPER_SPEC is None or _HTML_STATE_HELPER_SPEC.loader is None:
    raise ImportError("html_state_saver.py kon niet worden geladen.")
_html_state_saver = importlib.util.module_from_spec(_HTML_STATE_HELPER_SPEC)
_HTML_STATE_HELPER_SPEC.loader.exec_module(_html_state_saver)
add_interactive_html_saver = _html_state_saver.add_interactive_html_saver
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy.io as sio

from pyproj import Transformer
import folium


# ---------------------------
# M9 reader
# ---------------------------

def _m9_time_to_utc(seconds_since_2000: np.ndarray) -> pd.DatetimeIndex:
    base = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    td = pd.to_timedelta(seconds_since_2000.astype(float), unit="s")
    return (pd.Timestamp(base) + td).tz_convert("UTC")


def read_m9_mat(mat_path: str | Path, m9_hour_shift: int = -1) -> pd.DataFrame:
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "System" not in mat or "Summary" not in mat:
        raise ValueError("MAT mist 'System' en/of 'Summary'.")

    sys_ = mat["System"]
    summ = mat["Summary"]

    time_sec = np.asarray(sys_.Time).astype(float).reshape(-1)
    time_utc = _m9_time_to_utc(time_sec)
    # Tijdshift (uren): -1 = 1 uur aftrekken, 0 = geen shift, +1 = 1 uur bijtellen
    if int(m9_hour_shift) != 0:
        time_utc = time_utc + pd.Timedelta(hours=int(m9_hour_shift))

    track = np.asarray(summ.Track).astype(float)
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError(f"Summary.Track vorm onverwacht: {track.shape} (verwacht Nx2).")

    df = pd.DataFrame({
        "time_utc": time_utc,
        "m9_E_rel_m": track[:, 0],
        "m9_N_rel_m": track[:, 1],
    })

    bt = mat.get("BottomTrack")

    # bodemdiepte optioneel, enkel via BottomTrack
    if bt is not None and hasattr(bt, "BT_Depth"):
        depth = np.asarray(bt.BT_Depth).astype(float).reshape(-1)
        if len(depth) == len(df):
            df["m9_depth_m"] = depth

    df["t"] = df["time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("t").reset_index(drop=True)


# ---------------------------
# Stationary RTK points reader
# ---------------------------

def read_stationary_rtk_points(
    path: str | Path,
    assume_local_brussels: bool = True,
) -> pd.DataFrame:
    """
    Verwacht ';' gescheiden records:
      idx ; name ; E ; N ; H ; sigmaE ; sigmaN ; time ; date
    time = HH:MM:SS
    date = YYYY/MM/DD of YYYY-MM-DD (we proberen beide)
    """
    p = Path(path)

    # read as raw with no header
    df = pd.read_csv(p, sep=";", header=None, dtype=str, engine="python")
    if df.shape[1] < 9:
        raise ValueError(f"RTK puntenbestand heeft {df.shape[1]} kolommen; verwacht ≥9.")

    df.columns = ["idx", "name", "E", "N", "H", "sigmaE", "sigmaN", "time", "date"] + \
                 [f"extra_{i}" for i in range(df.shape[1] - 9)]

    # numeric columns
    for c in ["E", "N", "H", "sigmaE", "sigmaN"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")


    # FIX: stationaire RTK export = North;East;Height (N,E,H). Swap zodat E=East en N=North.
    df[['E', 'N']] = df[['N', 'E']]
    # parse datetime
    # combine date + time
    def _parse_dt(row) -> dt.datetime | None:
        date_s = str(row["date"]).strip()
        time_s = str(row["time"]).strip()
        if date_s.lower() == "nan" or time_s.lower() == "nan":
            return None
        # try formats
        for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return dt.datetime.strptime(f"{date_s} {time_s}", fmt)
            except ValueError:
                pass
        return None

    naive = df.apply(_parse_dt, axis=1)
    if naive.isna().all():
        raise ValueError("Kon geen datum/tijd parsen uit RTK puntenbestand (kolommen time/date).")

    tz_local = ZoneInfo("Europe/Brussels")
    if assume_local_brussels:
        # localize to Brussels, convert to UTC
        utc = [d.replace(tzinfo=tz_local).astimezone(dt.timezone.utc) if d is not None else None for d in naive]
    else:
        # treat as UTC naive
        utc = [d.replace(tzinfo=dt.timezone.utc) if d is not None else None for d in naive]

    df["time_utc"] = pd.to_datetime(utc, utc=True)
    df["t"] = df["time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)

    # drop invalid rows
    df = df.dropna(subset=["t", "E", "N", "H"]).copy()
    df["idx"] = df["idx"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    # unique id
    df["anchor_id"] = np.arange(len(df)) + 1

    return df.sort_values("t").reset_index(drop=True)


# ---------------------------
# Coupling
# ---------------------------

def couple_m9_to_stationary_rtk(
    m9: pd.DataFrame,
    rtk_pts: pd.DataFrame,
    tol_s: float = 2.0,
    height_add_m: float = 0.05,  # only +5 cm
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Per RTK anchor point: nearest M9 ensemble -> compute offsets
    2) Per M9 ensemble: last known offset (backward merge_asof) + fill start with first
    """
    # anchors matched to nearest M9
    anchors = pd.merge_asof(
        rtk_pts[["t", "anchor_id", "idx", "name", "E", "N", "H", "sigmaE", "sigmaN", "time_utc"]],
        m9[["t", "m9_E_rel_m", "m9_N_rel_m"]],
        on="t",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tol_s)),
    )

    anchors = anchors.dropna(subset=["m9_E_rel_m", "m9_N_rel_m"]).copy()
    if anchors.empty:
        raise ValueError("Geen tijd-overlap tussen RTK punten en M9 binnen tolerantie. Controleer uurcorrectie/RTK timezone/tol.")

    anchors.rename(columns={"E": "rtk_E_m", "N": "rtk_N_m", "H": "rtk_H_m"}, inplace=True)
    anchors["offset_E_m"] = anchors["rtk_E_m"] - anchors["m9_E_rel_m"]
    anchors["offset_N_m"] = anchors["rtk_N_m"] - anchors["m9_N_rel_m"]
    anchors["boat_H_m"] = anchors["rtk_H_m"] + float(height_add_m)

    anchors = anchors.sort_values("t").reset_index(drop=True)

    # apply offsets to M9 using last known anchor
    merged = pd.merge_asof(
        m9,
        anchors[["t", "anchor_id", "idx", "name", "rtk_E_m", "rtk_N_m", "rtk_H_m", "boat_H_m", "offset_E_m", "offset_N_m"]],
        on="t",
        direction="backward",
    )

    # before first anchor: use first anchor
    first = anchors.iloc[0]
    for col in ["anchor_id", "idx", "name", "rtk_E_m", "rtk_N_m", "rtk_H_m", "boat_H_m", "offset_E_m", "offset_N_m"]:
        merged[col] = merged[col].fillna(first[col])

    merged["m9_E_abs_m"] = merged["m9_E_rel_m"] + merged["offset_E_m"]
    merged["m9_N_abs_m"] = merged["m9_N_rel_m"] + merged["offset_N_m"]

    # bed height if depth present
    if "m9_depth_m" in merged.columns:
        merged["bed_H_m"] = merged["boat_H_m"] - merged["m9_depth_m"]

    merged["dt_to_anchor_s"] = (merged["t"] - anchors.iloc[np.searchsorted(anchors["t"].to_numpy(), merged["t"].to_numpy(), side="right") - 1]["t"].to_numpy())
    # Above is messy; simpler:
    # We'll compute dt via merge's t - anchor time by storing anchor time
    # Let's store anchor time:
    anchors_for_dt = anchors[["t", "anchor_id"]].rename(columns={"t":"anchor_time"})
    # Re-merge to get anchor_time for each m9 (same backward logic)
    merged2 = pd.merge_asof(
        m9[["t"]],
        anchors_for_dt.sort_values("anchor_time"),
        left_on="t",
        right_on="anchor_time",
        direction="backward",
    )
    merged["anchor_time"] = merged2["anchor_time"].fillna(anchors_for_dt.iloc[0]["anchor_time"])
    merged["dt_to_anchor_s"] = (merged["t"] - merged["anchor_time"]).dt.total_seconds()

    merged["time_utc"] = merged["t"].dt.tz_localize("UTC")

    # lat/lon for mapping
    tr = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(merged["m9_E_abs_m"].to_numpy(), merged["m9_N_abs_m"].to_numpy())
    merged["lon"] = lon
    merged["lat"] = lat

    out_cols = [
        "time_utc",
        "m9_E_rel_m", "m9_N_rel_m",
        "m9_E_abs_m", "m9_N_abs_m",
        "anchor_id", "idx", "name",
        "rtk_E_m", "rtk_N_m",
        "rtk_H_m", "boat_H_m",
        "offset_E_m", "offset_N_m",
        "anchor_time", "dt_to_anchor_s",
        "lat", "lon",
    ]
    if "m9_depth_m" in merged.columns:
        out_cols.insert(out_cols.index("offset_E_m"), "m9_depth_m")
        out_cols.insert(out_cols.index("offset_E_m"), "bed_H_m")

    merged_out = merged[out_cols].copy()
    merged_out.sort_values("time_utc", inplace=True)
    merged_out.reset_index(drop=True, inplace=True)

    anchors_out = anchors[[
        "time_utc", "anchor_id", "idx", "name",
        "rtk_E_m", "rtk_N_m", "rtk_H_m", "boat_H_m",
        "m9_E_rel_m", "m9_N_rel_m",
        "offset_E_m", "offset_N_m",
        "sigmaE", "sigmaN",
    ]].copy()
    anchors_out.sort_values("time_utc", inplace=True)
    anchors_out.reset_index(drop=True, inplace=True)

    return merged_out, anchors_out


def make_map_html(
    merged: pd.DataFrame,
    anchors: pd.DataFrame,
    out_html: str | Path,
    every_n: int = 5,
) -> None:
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    center = [float(merged["lat"].median()), float(merged["lon"].median())]
    m = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap")

    folium.PolyLine(merged[["lat", "lon"]].astype(float).values.tolist(), weight=4, opacity=0.9).add_to(m)

    # Mark anchors
    tr = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
    a_lon, a_lat = tr.transform(anchors["rtk_E_m"].to_numpy(), anchors["rtk_N_m"].to_numpy())
    anchors = anchors.copy()
    anchors["alon"] = a_lon
    anchors["alat"] = a_lat

    for _, r in anchors.iterrows():
        popup = f"Anchor {r.anchor_id} - {r['name']}<br>{r['time_utc']}<br>E={r.rtk_E_m:.3f} N={r.rtk_N_m:.3f}"
        folium.Marker([float(r.alat), float(r.alon)], popup=popup).add_to(m)

    # Downsample corrected track points
    ds = merged.iloc[::max(1, int(every_n))].copy()
    for _, r in ds.iterrows():
        popup = f"{r['time_utc']}<br>E={r['m9_E_abs_m']:.3f}<br>N={r['m9_N_abs_m']:.3f}<br>dt_anchor={r['dt_to_anchor_s']:.1f}s"
        folium.CircleMarker([float(r.lat), float(r.lon)], radius=3, fill=True, opacity=0.9, fill_opacity=0.9,
                            popup=folium.Popup(popup, max_width=320)).add_to(m)

    m.save(str(out_html))
    add_interactive_html_saver(out_html)


def save_csv(df: pd.DataFrame, path: str | Path, sep: str = ";") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep)


# ---------------------------
# GUI
# ---------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M9 ↔ Stationaire RTK punten (Lambert72) – tijdmatch + forward-fill offsets")
        self.geometry("1020x640")

        self.var_m9 = tk.StringVar()
        self.var_rtk = tk.StringVar()
        self.var_out = tk.StringVar()

        self.var_m9_shift = tk.StringVar(value='-1')  # -1, 0, +1
        self.var_rtk_local = tk.BooleanVar(value=True)

        self.var_tol = tk.DoubleVar(value=2.0)
        self.var_height_add = tk.DoubleVar(value=0.05)

        self.var_make_map = tk.BooleanVar(value=True)
        self.var_every_n = tk.IntVar(value=10)

        self._build()

    def _build(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="M9 .mat").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_m9, width=96).grid(row=1, column=0, sticky="we", padx=(0,8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_m9).grid(row=1, column=1, sticky="e")

        ttk.Label(frm, text="RTK stationaire puntenbestand (.txt/.csv, ';' gescheiden)").grid(row=2, column=0, sticky="w", pady=(10,0))
        ttk.Entry(frm, textvariable=self.var_rtk, width=96).grid(row=3, column=0, sticky="we", padx=(0,8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_rtk).grid(row=3, column=1, sticky="e")

        ttk.Label(frm, text="Output basisnaam (.csv)").grid(row=4, column=0, sticky="w", pady=(10,0))
        ttk.Entry(frm, textvariable=self.var_out, width=96).grid(row=5, column=0, sticky="we", padx=(0,8))
        ttk.Button(frm, text="Opslaan als…", command=self._pick_out).grid(row=5, column=1, sticky="e")

        opt = ttk.LabelFrame(frm, text="Tijd / matching / hoogte", padding=10)
        opt.grid(row=6, column=0, columnspan=2, sticky="we", pady=(12,0))

        ttk.Label(opt, text="M9 tijdshift:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opt, textvariable=self.var_m9_shift, values=['-1','0','+1'], width=6, state='readonly').grid(row=0, column=1, sticky='w', padx=(6,0))
        ttk.Label(opt, text="(uren; -1=aftrekken, 0=geen, +1=bijtellen)").grid(row=0, column=2, sticky='w', padx=(10,0))
        ttk.Checkbutton(opt, text="RTK tijden: Europe/Brussels → UTC", variable=self.var_rtk_local).grid(row=0, column=3, sticky='w', padx=(20,0))

        ttk.Label(opt, text="Match tolerantie (s):").grid(row=1, column=0, sticky="w", pady=(8,0))
        ttk.Entry(opt, textvariable=self.var_tol, width=10).grid(row=1, column=1, sticky="w", pady=(8,0))

        ttk.Label(opt, text="Hoogtecorrectie +m (boat_H = RTK_H + add):").grid(row=1, column=2, sticky="e", padx=(20,0), pady=(8,0))
        ttk.Entry(opt, textvariable=self.var_height_add, width=10).grid(row=1, column=3, sticky="w", pady=(8,0))

        mfrm = ttk.LabelFrame(frm, text="Kaart (optioneel)", padding=10)
        mfrm.grid(row=7, column=0, columnspan=2, sticky="we", pady=(10,0))

        ttk.Checkbutton(mfrm, text="Maak folium kaart HTML", variable=self.var_make_map).grid(row=0, column=0, sticky="w")
        ttk.Label(mfrm, text="Toon elke n trackpunten:").grid(row=0, column=1, sticky="e", padx=(20,0))
        ttk.Entry(mfrm, textvariable=self.var_every_n, width=10).grid(row=0, column=2, sticky="w", padx=(6,0))

        btns = ttk.Frame(frm)
        btns.grid(row=8, column=0, columnspan=2, sticky="we", pady=(12,0))
        ttk.Button(btns, text="Verwerken & opslaan", command=self._run).pack(side="left")
        ttk.Button(btns, text="Sluiten", command=self.destroy).pack(side="right")

        self.txt = tk.Text(frm, height=12, wrap="word")
        self.txt.grid(row=9, column=0, columnspan=2, sticky="nsew", pady=(10,0))
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(9, weight=1)

    def _log(self, s: str):
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _pick_m9(self):
        p = filedialog.askopenfilename(title="Selecteer M9 .mat", filetypes=[("MAT", "*.mat"), ("All", "*.*")])
        if p:
            self.var_m9.set(p)
            if not self.var_out.get():
                self.var_out.set(str(Path(p).with_suffix("").as_posix()) + "_stationary_rtk.csv")

    def _pick_rtk(self):
        p = filedialog.askopenfilename(title="Selecteer RTK puntenbestand", filetypes=[("Text/CSV", "*.txt *.csv"), ("All", "*.*")])
        if p:
            self.var_rtk.set(p)

    def _pick_out(self):
        p = filedialog.asksaveasfilename(title="Kies output CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.var_out.set(p)

    def _run(self):
        try:
            m9p = self.var_m9.get().strip()
            rtkp = self.var_rtk.get().strip()
            outp = self.var_out.get().strip()
            if not m9p or not rtkp or not outp:
                messagebox.showerror("Input", "Kies M9, RTK puntenbestand en output.")
                return

            base = Path(outp).with_suffix("")
            m9_shift = int(self.var_m9_shift.get())
            rtk_local = bool(self.var_rtk_local.get())
            tol = float(self.var_tol.get())
            h_add = float(self.var_height_add.get())
            make_map = bool(self.var_make_map.get())
            every_n = int(self.var_every_n.get())

            self._log("Lezen M9…")
            m9 = read_m9_mat(m9p, m9_hour_shift=m9_shift)
            self._log(f"  M9 ensembles: {len(m9)} | {m9.time_utc.min()} → {m9.time_utc.max()}")

            self._log("Lezen RTK stationaire punten…")
            rtk = read_stationary_rtk_points(rtkp, assume_local_brussels=rtk_local)
            self._log(f"  RTK punten: {len(rtk)} | {rtk.time_utc.min()} → {rtk.time_utc.max()}")

            self._log(f"Koppelen (tol={tol:.2f}s), hoogte add={h_add:.3f} m…")
            merged, anchors = couple_m9_to_stationary_rtk(m9, rtk, tol_s=tol, height_add_m=h_add)

            out_csv = str(base) + ".csv"
            anc_csv = str(base) + "_anchors.csv"
            save_csv(merged, out_csv, sep=";")
            save_csv(anchors, anc_csv, sep=";")
            self._log(f"Saved: {out_csv}")
            self._log(f"Saved: {anc_csv}")

            if make_map:
                map_html = str(base) + "_map.html"
                make_map_html(merged, anchors, map_html, every_n=every_n)
                self._log(f"Saved: {map_html}")

            messagebox.showinfo("OK", "Klaar.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")


# ---------------------------
# CLI
# ---------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M9 ↔ stationaire RTK punten: tijdmatch + forward-fill offsets")
    ap.add_argument("--m9", type=str, help="M9 .mat")
    ap.add_argument("--rtk", type=str, help="RTK puntenbestand (txt/csv ';')")
    ap.add_argument("--out", type=str, help="Output basisnaam .csv")
    ap.add_argument("--m9-shift", type=int, default=-1, choices=[-1,0,1],
                    help="M9 tijdshift in uren: -1=aftrekken, 0=geen, +1=bijtellen")
    ap.add_argument("--rtk-local", action="store_true", help="RTK: interpreteer als Europe/Brussels (→UTC)")
    ap.add_argument("--rtk-utc", action="store_true", help="RTK: behandel als UTC")
    ap.add_argument("--tol", type=float, default=2.0, help="Tolerantie (s)")
    ap.add_argument("--h-add", type=float, default=0.05, help="boat_H = RTK_H + h_add")
    ap.add_argument("--map", action="store_true", help="Maak map HTML")
    ap.add_argument("--every-n", type=int, default=10)
    ap.add_argument("--nogui", action="store_true")
    args = ap.parse_args(argv)

    if not args.nogui and (args.m9 is None or args.rtk is None or args.out is None):
        App().mainloop()
        return 0

    if args.m9 is None or args.rtk is None or args.out is None:
        ap.error("--m9 --rtk --out zijn verplicht in CLI.")
    rtk_local = True
    if args.rtk_utc:
        rtk_local = False
    if args.rtk_local:
        rtk_local = True

    m9 = read_m9_mat(args.m9, m9_hour_shift=args.m9_shift)
    rtk = read_stationary_rtk_points(args.rtk, assume_local_brussels=rtk_local)
    merged, anchors = couple_m9_to_stationary_rtk(m9, rtk, tol_s=args.tol, height_add_m=args.h_add)

    base = Path(args.out).with_suffix("")
    save_csv(merged, str(base) + ".csv", sep=";")
    save_csv(anchors, str(base) + "_anchors.csv", sep=";")
    if args.map:
        make_map_html(merged, anchors, str(base) + "_map.html", every_n=args.every_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
