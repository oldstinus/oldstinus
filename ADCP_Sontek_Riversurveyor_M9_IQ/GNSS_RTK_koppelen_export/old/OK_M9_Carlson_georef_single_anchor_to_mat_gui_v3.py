#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M9 (.mat) georefereren met Carlson RTK via 1 vast ankerpunt (tijdmatching) en
opslaan naar nieuwe .mat waarbij de volledige structuur intact blijft
(we overschrijven enkel Summary.Track).

Kern:
- M9 tijd: System.Time = seconden sinds 2000-01-01 (UTC).
- M9 track: Summary.Track = Nx2 (relatieve XY in meter) -> wordt absoluut Lambert72 (E,N) na offset.
- Carlson RTK: ';' gescheiden, met tijd in voorlaatste kolom (HH:MM:SS) en datum in laatste kolom (YYYY/MM/DD).
  Typische opbouw:
    idx ; omschrijving ; East ; North ; Height ; sigmaE ; sigmaN ; ... ; HH:MM:SS ; YYYY/MM/DD
  (Sommige exports hebben North;East;Height i.p.v. East;North;Height -> optionele swap)

GUI:
- Selecteer M9 .mat
- Selecteer Carlson RTK (.txt/.csv)
- Lees RTK -> kies 1 ankerpunt
- Stel toleranties/timezone/tijdshift in
- Run -> schrijft nieuwe .mat (Summary.Track overschreven met absolute E,N)

Opmerking:
Met 1 ankerpunt wordt enkel een translatie toegepast (geen rotatie/schaal).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import scipy.io as sio
from zoneinfo import ZoneInfo


# ---------------------------
# Helpers
# ---------------------------

def _is_number_like(x: str) -> bool:
    if x is None:
        return False
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return False
    s2 = s.replace(",", ".")
    try:
        float(s2)
        return True
    except Exception:
        return False


def _to_float(x: str):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    return float(s.replace(",", "."))


def _parse_datetime(date_s: str, time_s: str):
    date_s = str(date_s).strip()
    time_s = str(time_s).strip()
    if date_s.lower() == "nan" or time_s.lower() == "nan":
        return None

    candidates = [
        ("%Y/%m/%d %H:%M:%S", f"{date_s} {time_s}"),
        ("%Y-%m-%d %H:%M:%S", f"{date_s} {time_s}"),
        ("%d/%m/%Y %H:%M:%S", f"{date_s} {time_s}"),
        ("%d-%m-%Y %H:%M:%S", f"{date_s} {time_s}"),
    ]
    for fmt, s in candidates:
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass

    ds = re.sub(r"[-\.]", "/", date_s)
    candidates2 = [
        ("%Y/%m/%d %H:%M:%S", f"{ds} {time_s}"),
        ("%d/%m/%Y %H:%M:%S", f"{ds} {time_s}"),
    ]
    for fmt, s in candidates2:
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _m9_time_to_utc(seconds_since_2000: np.ndarray) -> pd.DatetimeIndex:
    base = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    td = pd.to_timedelta(np.asarray(seconds_since_2000, dtype=float).reshape(-1), unit="s")
    return (pd.Timestamp(base) + td).tz_convert("UTC")


# ---------------------------
# M9 reader + writer
# ---------------------------

@dataclass
class M9Data:
    mat_dict: dict
    sys_obj: object
    summ_obj: object
    time_utc: pd.DatetimeIndex
    t_naive: pd.Series
    track_rel: np.ndarray  # Nx2 float
    track_is_NE: bool      # if true, columns were (N,E) and we swapped to (E,N) internally


def read_m9_mat(mat_path: str | Path, m9_hour_shift: int = -1, m9_track_is_NE: bool = False) -> M9Data:
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    if "System" not in mat or "Summary" not in mat:
        raise ValueError("MAT mist 'System' en/of 'Summary'.")

    sys_ = mat["System"]
    summ = mat["Summary"]

    if not hasattr(sys_, "Time"):
        raise ValueError("MAT: System.Time niet gevonden.")
    time_sec = np.asarray(sys_.Time).astype(float).reshape(-1)
    time_utc = _m9_time_to_utc(time_sec)
    if int(m9_hour_shift) != 0:
        time_utc = time_utc + pd.Timedelta(hours=int(m9_hour_shift))

    if not hasattr(summ, "Track"):
        raise ValueError("MAT: Summary.Track niet gevonden.")
    track = np.asarray(summ.Track).astype(float)
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError(f"Summary.Track vorm onverwacht: {track.shape} (verwacht Nx2).")

    track = track[:, :2].astype(float)

    track_is_NE = bool(m9_track_is_NE)
    if track_is_NE:
        track = track[:, [1, 0]]  # (N,E) -> (E,N) internally

    t_naive = pd.Series(pd.to_datetime(time_utc).tz_convert("UTC").tz_localize(None))

    return M9Data(
        mat_dict=mat,
        sys_obj=sys_,
        summ_obj=summ,
        time_utc=time_utc,
        t_naive=t_naive,
        track_rel=track,
        track_is_NE=track_is_NE,
    )


def write_m9_mat_overwrite_track(m9: M9Data, new_track_EN: np.ndarray, out_path: str | Path) -> None:
    new_track_EN = np.asarray(new_track_EN, dtype=float)
    if new_track_EN.ndim != 2 or new_track_EN.shape[1] != 2:
        raise ValueError("new_track_EN moet Nx2 zijn (E,N).")

    track_to_store = new_track_EN.copy()
    if m9.track_is_NE:
        track_to_store = track_to_store[:, [1, 0]]  # store back as (N,E)

    setattr(m9.summ_obj, "Track", track_to_store)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sio.savemat(
        str(out_path),
        m9.mat_dict,
        do_compression=True,
        long_field_names=True,
        oned_as="row",
    )


# ---------------------------
# Carlson RTK reader (robust)
# ---------------------------

def read_carlson_rtk_points(
    path: str | Path,
    assume_local_brussels: bool = True,
    swap_NE: bool = True,
) -> pd.DataFrame:
    p = Path(path)
    df_raw = pd.read_csv(p, sep=";", header=None, dtype=str, engine="python")

    if df_raw.shape[1] < 6:
        raise ValueError(f"RTK bestand heeft te weinig kolommen ({df_raw.shape[1]}). Verwacht minstens 6.")

    ncol = df_raw.shape[1]

    # Detect header row
    row0 = df_raw.iloc[0].tolist()
    header_keywords = ("east", "north", "height", "sigma", "time", "date", "datum", "tijd", "omschrijving", "idx")
    looks_like_header = False

    if ncol >= 5:
        if (not _is_number_like(row0[2])) and (not _is_number_like(row0[3])):
            looks_like_header = True

    tail = " ".join([str(x).lower() for x in row0[-3:]])
    if any(k in tail for k in ("time", "date", "datum", "tijd")):
        looks_like_header = True

    whole = " ".join([str(x).lower() for x in row0])
    if any(k in whole for k in header_keywords):
        looks_like_header = True

    if looks_like_header:
        df_raw = df_raw.iloc[1:].reset_index(drop=True)

    ncol = df_raw.shape[1]
    if ncol < 5:
        raise ValueError("RTK bestand heeft onvoldoende kolommen voor E/N/H (verwacht minstens 5).")

    time_col = ncol - 2
    date_col = ncol - 1

    out = pd.DataFrame()
    out["idx"] = df_raw.iloc[:, 0].astype(str).str.strip()
    out["name"] = df_raw.iloc[:, 1].astype(str).str.strip() if ncol >= 2 else ""

    out["E"] = df_raw.iloc[:, 2].map(_to_float)
    out["N"] = df_raw.iloc[:, 3].map(_to_float)
    out["H"] = df_raw.iloc[:, 4].map(_to_float)

    if ncol >= 7:
        out["sigmaE"] = df_raw.iloc[:, 5].map(_to_float)
        out["sigmaN"] = df_raw.iloc[:, 6].map(_to_float)
    else:
        out["sigmaE"] = np.nan
        out["sigmaN"] = np.nan

    out["time_s"] = df_raw.iloc[:, time_col].astype(str).str.strip()
    out["date_s"] = df_raw.iloc[:, date_col].astype(str).str.strip()

    if bool(swap_NE):
        out[["E", "N"]] = out[["N", "E"]]

    naive = out.apply(lambda r: _parse_datetime(r["date_s"], r["time_s"]), axis=1)
    if pd.isna(naive).all():
        raise ValueError("Kon geen datum/tijd parsen uit RTK (laatste 2 kolommen).")

    tz_local = ZoneInfo("Europe/Brussels")
    if assume_local_brussels:
        utc = [d.replace(tzinfo=tz_local).astimezone(dt.timezone.utc) if d is not None else None for d in naive]
    else:
        utc = [d.replace(tzinfo=dt.timezone.utc) if d is not None else None for d in naive]

    out["time_utc"] = pd.to_datetime(utc, utc=True)
    out["t"] = out["time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)

    out = out.dropna(subset=["t", "E", "N"]).copy()
    out = out.sort_values("t").reset_index(drop=True)
    out["anchor_id"] = np.arange(len(out)) + 1

    return out


# ---------------------------
# Georeferencing with single anchor
# ---------------------------

def georef_track_by_single_anchor(
    m9: M9Data,
    rtk: pd.DataFrame,
    anchor_anchor_id: int,
    tol_s: float = 2.0,
    rtk_time_shift_s: float = 0.0,
):
    if rtk.empty:
        raise ValueError("RTK dataframe is leeg.")
    if "anchor_id" not in rtk.columns:
        raise ValueError("RTK dataframe mist 'anchor_id'.")

    anchor = rtk.loc[rtk["anchor_id"] == int(anchor_anchor_id)]
    if anchor.empty:
        raise ValueError(f"Ankerpunt {anchor_anchor_id} niet gevonden.")
    anchor = anchor.iloc[0]

    anchor_t = pd.to_datetime(anchor["t"])
    if float(rtk_time_shift_s) != 0.0:
        anchor_t = anchor_t + pd.Timedelta(seconds=float(rtk_time_shift_s))

    m9_times = m9.t_naive.values.astype("datetime64[ns]")
    target = np.datetime64(anchor_t.to_datetime64())
    idx = int(np.argmin(np.abs(m9_times - target)))
    dt_s = float(np.abs((pd.Timestamp(m9_times[idx]) - pd.Timestamp(target)).total_seconds()))

    if dt_s > float(tol_s):
        raise ValueError(
            f"Geen M9 punt binnen tolerantie. Dichtste dt={dt_s:.3f}s (tol={tol_s:.3f}s). "
            "Controleer M9 uurshift / RTK timezone / RTK timeshift."
        )

    m9_e_rel = float(m9.track_rel[idx, 0])
    m9_n_rel = float(m9.track_rel[idx, 1])

    rtk_e = float(anchor["E"])
    rtk_n = float(anchor["N"])

    dE = rtk_e - m9_e_rel
    dN = rtk_n - m9_n_rel

    new_track = m9.track_rel.copy()
    new_track[:, 0] = new_track[:, 0] + dE
    new_track[:, 1] = new_track[:, 1] + dN

    info = {
        "anchor_id": int(anchor["anchor_id"]),
        "anchor_idx": str(anchor.get("idx", "")),
        "anchor_name": str(anchor.get("name", "")),
        "anchor_time_utc": str(anchor.get("time_utc", "")),
        "anchor_t_used": str(anchor_t),
        "matched_m9_index": idx,
        "matched_m9_time_utc": str(pd.to_datetime(m9.time_utc[idx])),
        "dt_seconds": dt_s,
        "rtk_E": rtk_e,
        "rtk_N": rtk_n,
        "m9_E_rel_at_match": m9_e_rel,
        "m9_N_rel_at_match": m9_n_rel,
        "offset_dE": dE,
        "offset_dN": dN,
    }
    return new_track, info


# ---------------------------
# GUI
# ---------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M9 georefereren met Carlson RTK (1 ankerpunt) → .mat (Summary.Track overschrijven)")
        self.geometry("1180x740")

        self.var_m9 = tk.StringVar()
        self.var_rtk = tk.StringVar()
        self.var_out = tk.StringVar()

        self.var_m9_shift = tk.StringVar(value="-1")
        self.var_tol = tk.DoubleVar(value=2.0)
        self.var_rtk_local = tk.BooleanVar(value=True)
        self.var_rtk_shift_s = tk.DoubleVar(value=0.0)
        self.var_rtk_swap_NE = tk.BooleanVar(value=True)
        self.var_m9_track_is_NE = tk.BooleanVar(value=False)

        self._rtk_df = None
        self._m9 = None

        self._build()

    def _build(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        row = 0
        ttk.Label(frm, text="M9 .mat").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Entry(frm, textvariable=self.var_m9, width=110).grid(row=row, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_m9).grid(row=row, column=1, sticky="e")
        row += 1

        ttk.Label(frm, text="Carlson RTK bestand (.txt/.csv, ';')").grid(row=row, column=0, sticky="w", pady=(10, 0))
        row += 1
        ttk.Entry(frm, textvariable=self.var_rtk, width=110).grid(row=row, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_rtk).grid(row=row, column=1, sticky="e")
        row += 1

        ttk.Label(frm, text="Output .mat (georef)").grid(row=row, column=0, sticky="w", pady=(10, 0))
        row += 1
        ttk.Entry(frm, textvariable=self.var_out, width=110).grid(row=row, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Opslaan als…", command=self._pick_out).grid(row=row, column=1, sticky="e")
        row += 1

        opt = ttk.LabelFrame(frm, text="Opties tijd / kolommen", padding=10)
        opt.grid(row=row, column=0, columnspan=2, sticky="we", pady=(12, 0))
        row += 1

        ttk.Label(opt, text="M9 tijdshift (uren):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opt, textvariable=self.var_m9_shift, values=["-2", "-1", "0", "+1", "+2"], width=6, state="readonly").grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(opt, text="Match tolerantie (s):").grid(row=0, column=2, sticky="e", padx=(18, 0))
        ttk.Entry(opt, textvariable=self.var_tol, width=10).grid(row=0, column=3, sticky="w", padx=(6, 0))

        ttk.Checkbutton(opt, text="RTK tijden: Europe/Brussels → UTC", variable=self.var_rtk_local).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(opt, text="Extra RTK time shift (s):").grid(row=1, column=2, sticky="e", padx=(18, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_rtk_shift_s, width=10).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Checkbutton(opt, text="RTK export is N;E;H (swap N↔E)", variable=self.var_rtk_swap_NE).grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(opt, text="M9 Summary.Track is N,E (swap bij in-/uitschrijven)", variable=self.var_m9_track_is_NE).grid(row=2, column=2, sticky="w", pady=(8, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=2, sticky="we", pady=(12, 0))
        row += 1
        ttk.Button(btns, text="1) Lees M9", command=self._load_m9).pack(side="left")
        ttk.Button(btns, text="2) Lees RTK", command=self._load_rtk).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="3) Georefereren & .mat opslaan", command=self._run).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Sluiten", command=self.destroy).pack(side="right")

        pan = ttk.Panedwindow(frm, orient=tk.VERTICAL)
        pan.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        frm.grid_rowconfigure(row, weight=1)
        frm.grid_columnconfigure(0, weight=1)

        top = ttk.Frame(pan)
        bot = ttk.Frame(pan)
        pan.add(top, weight=3)
        pan.add(bot, weight=2)

        ttk.Label(top, text="Kies 1 ankerpunt (selecteer 1 rij):").pack(anchor="w")

        self.tree = ttk.Treeview(top, columns=("anchor_id", "idx", "name", "time_utc", "E", "N", "H"), show="headings", height=14)
        for c, w in [("anchor_id", 80), ("idx", 80), ("name", 260), ("time_utc", 210), ("E", 130), ("N", 130), ("H", 110)]:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)

        vsb = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

        ttk.Label(bot, text="Log:").pack(anchor="w")
        self.txt = tk.Text(bot, height=10, wrap="word")
        self.txt.pack(fill="both", expand=True)

    def _log(self, s: str):
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _pick_m9(self):
        p = filedialog.askopenfilename(title="Selecteer M9 .mat", filetypes=[("MAT", "*.mat"), ("All", "*.*")])
        if p:
            self.var_m9.set(p)
            if not self.var_out.get():
                self.var_out.set(str(Path(p).with_suffix("")) + "_georef.mat")

    def _pick_rtk(self):
        p = filedialog.askopenfilename(title="Selecteer Carlson RTK", filetypes=[("Text/CSV", "*.txt *.csv"), ("All", "*.*")])
        if p:
            self.var_rtk.set(p)

    def _pick_out(self):
        p = filedialog.asksaveasfilename(title="Kies output .mat", defaultextension=".mat", filetypes=[("MAT", "*.mat")])
        if p:
            self.var_out.set(p)

    def _load_m9(self):
        try:
            m9p = self.var_m9.get().strip()
            if not m9p:
                messagebox.showerror("Input", "Kies een M9 .mat.")
                return
            shift = int(self.var_m9_shift.get())
            track_is_NE = bool(self.var_m9_track_is_NE.get())
            self._log("Lezen M9…")
            self._m9 = read_m9_mat(m9p, m9_hour_shift=shift, m9_track_is_NE=track_is_NE)
            self._log(f"  OK: ensembles={len(self._m9.track_rel)} | {self._m9.time_utc.min()} → {self._m9.time_utc.max()}")
            messagebox.showinfo("OK", "M9 ingelezen.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")

    def _load_rtk(self):
        try:
            rtkp = self.var_rtk.get().strip()
            if not rtkp:
                messagebox.showerror("Input", "Kies een RTK bestand.")
                return
            local = bool(self.var_rtk_local.get())
            swap = bool(self.var_rtk_swap_NE.get())

            self._log("Lezen RTK…")
            rtk = read_carlson_rtk_points(rtkp, assume_local_brussels=local, swap_NE=swap)
            self._rtk_df = rtk
            self._log(f"  OK: punten={len(rtk)} | {rtk.time_utc.min()} → {rtk.time_utc.max()}")
            for it in self.tree.get_children():
                self.tree.delete(it)

            for _, r in rtk.iterrows():
                self.tree.insert(
                    "", "end",
                    values=(
                        int(r["anchor_id"]),
                        r.get("idx", ""),
                        r.get("name", ""),
                        str(r.get("time_utc", "")),
                        f"{float(r['E']):.3f}" if pd.notna(r["E"]) else "",
                        f"{float(r['N']):.3f}" if pd.notna(r["N"]) else "",
                        f"{float(r['H']):.3f}" if pd.notna(r["H"]) else "",
                    )
                )
            messagebox.showinfo("OK", "RTK ingelezen. Kies nu 1 ankerpunt in de tabel.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")

    def _get_selected_anchor_id(self):
        sel = self.tree.selection()
        if not sel:
            return None
        vals = self.tree.item(sel[0], "values")
        if not vals:
            return None
        return int(vals[0])

    def _run(self):
        try:
            if self._m9 is None:
                self._load_m9()
            if self._rtk_df is None:
                self._load_rtk()
            if self._m9 is None or self._rtk_df is None:
                return

            anchor_id = self._get_selected_anchor_id()
            if anchor_id is None:
                messagebox.showerror("Ankerpunt", "Selecteer 1 ankerpunt (1 rij) in de RTK-tabel.")
                return

            outp = self.var_out.get().strip()
            if not outp:
                messagebox.showerror("Output", "Kies output .mat.")
                return

            tol = float(self.var_tol.get())
            rtk_shift_s = float(self.var_rtk_shift_s.get())

            self._log(f"Georefereren met ankerpunt {anchor_id} (tol={tol}s, RTK shift={rtk_shift_s}s)…")
            new_track, info = georef_track_by_single_anchor(
                m9=self._m9,
                rtk=self._rtk_df,
                anchor_anchor_id=anchor_id,
                tol_s=tol,
                rtk_time_shift_s=rtk_shift_s,
            )

            self._log("Offset berekend:")
            self._log(f"  dE = {info['offset_dE']:.3f} m | dN = {info['offset_dN']:.3f} m")
            self._log(f"  anchor time (used) = {info['anchor_t_used']}")
            self._log(f"  matched M9 idx = {info['matched_m9_index']} | dt = {info['dt_seconds']:.3f} s")

            self._log("Schrijven .mat (Summary.Track overschrijven)…")
            write_m9_mat_overwrite_track(self._m9, new_track, outp)
            self._log(f"  Saved: {outp}")

            messagebox.showinfo("OK", "Klaar. Nieuwe .mat geschreven.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
