#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tkinter import Tk, StringVar, BooleanVar, IntVar, Label, Entry, Button
from tkinter import filedialog, ttk, messagebox


# ==============================================================
# Helpers: robust CSV read + column detection
# ==============================================================

def _sniff_delimiter_and_decimal(filepath, n_lines=30):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [f.readline() for _ in range(n_lines)]
    sample = "".join(lines)

    candidates = [";", ",", "\t"]
    counts = {c: sample.count(c) for c in candidates}
    sep = max(counts, key=counts.get)

    comma_decimal_hits = len(re.findall(r"\d+,\d+", sample))
    dot_decimal_hits = len(re.findall(r"\d+\.\d+", sample))
    decimal = "," if comma_decimal_hits > dot_decimal_hits else "."

    return sep, decimal


def read_aquadopp_csv(filepath):
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError("Bestand niet gevonden.")

    sep, decimal = _sniff_delimiter_and_decimal(filepath)
    try:
        df = pd.read_csv(filepath, sep=sep, decimal=decimal, engine="python")
    except Exception as e:
        raise ValueError(f"CSV kon niet gelezen worden met sep='{sep}' en decimal='{decimal}'.\n{e}")

    if df.empty:
        raise ValueError("CSV is leeg of werd verkeerd ingelezen (controleer delimiter/decimal).")

    df.columns = [str(c).strip() for c in df.columns]
    return df, sep, decimal


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def autodetect_columns(cols):
    normed = {c: _norm(c) for c in cols}

    def pick(patterns, exclude_patterns=()):
        for c in cols:
            nc = normed[c]
            if any(p in nc for p in patterns) and not any(ep in nc for ep in exclude_patterns):
                return c
        return ""

    return {
        "time": pick(["time", "date", "datum", "tijd", "timestamp", "datetime"]),
        "speed": pick(["speed", "vel", "velocity", "spd"]),
        "dir": pick(["dir", "direction", "bearing"], exclude_patterns=("speed",)),
        "heading": pick(["heading", "head"]),
        "pitch": pick(["pitch"]),
        "roll": pick(["roll"]),
        # "heave" wordt hier gebruikt als "extra kanaal" (bv. Pressure)
        "heave": pick(["pressure", "prs", "heave", "analog", "an1", "an2", "analog1", "analog2"]),
    }


def find_cell_columns(cols, kind="speed"):
    """
    Robust detection of meetcelkolommen, ook wanneer er (diepte/units) achter staan:
      - Speed#1(0.2m), Speed # 1 [m/s], Dir#10 (2.0 m) ...
    Returns dict: {cell_index(int): column_name(str)}
    """
    out = {}
    k = kind.lower()

    if k == "speed":
        pat = re.compile(r"(?:^|[^a-z0-9])speed\s*#\s*(\d+)", re.IGNORECASE)
    elif k in ("dir", "direction"):
        pat = re.compile(r"(?:^|[^a-z0-9])dir\s*#\s*(\d+)", re.IGNORECASE)
    else:
        raise ValueError("kind moet 'speed' of 'dir' zijn")

    for c in cols:
        m = pat.search(str(c))
        if m:
            out[int(m.group(1))] = c

    return out


# ==============================================================
# Vector-mean over selected cells
# ==============================================================

def vector_mean_speed_dir(speed_mat, dir_deg_mat):
    """
    speed_mat: 2D array (n_times, n_cells)
    dir_deg_mat: 2D array (n_times, n_cells), degrees (0-360), 0=N, 90=E
    Returns:
      mean_speed (n_times,), mean_dir_deg (n_times,)
    Ignores NaNs pairwise.
    """
    speed = np.array(speed_mat, dtype=float)
    direc = np.array(dir_deg_mat, dtype=float)

    valid = np.isfinite(speed) & np.isfinite(direc)
    speed = np.where(valid, speed, np.nan)
    direc = np.where(valid, direc, np.nan)

    theta = np.deg2rad(direc)

    # u east, v north (0°=N, 90°=E)
    u = speed * np.sin(theta)
    v = speed * np.cos(theta)

    u_mean = np.nanmean(u, axis=1)
    v_mean = np.nanmean(v, axis=1)

    mean_speed = np.sqrt(u_mean**2 + v_mean**2)
    mean_dir = (np.rad2deg(np.arctan2(u_mean, v_mean)) + 360.0) % 360.0
    return mean_speed, mean_dir


# ==============================================================
# Core analysis
# ==============================================================

def analyse_ensembles(
    df,
    time_col,
    speed_col=None,
    dir_col=None,
    heading_col=None,
    pitch_col=None,
    roll_col=None,
    heave_col=None,
    # range:
    use_cell_range=False,
    cell_start=None,
    cell_end=None,
    speed_cell_map=None,
    dir_cell_map=None,
):
    """
    Output DataFrame with:
      time, speed, dir_deg, heading, pitch, roll, heave

    If use_cell_range=True:
      speed & dir are computed as vector-mean over Speed#start..end and Dir#start..end.
    """
    df = df.copy()

    if not time_col or time_col not in df.columns:
        raise ValueError("Geen geldige tijdkolom gekozen.")

    # Robust datetime parse
    t1 = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    if t1.isna().mean() > 0.9:
        t2 = pd.to_datetime(df[time_col], dayfirst=False, errors="coerce")
        t = t2 if t2.isna().mean() < t1.isna().mean() else t1
    else:
        t = t1

    valid = t.notna().sum()
    if valid == 0:
        raise ValueError(
            "Tijdkolom kon niet geparsed worden.\n"
            "Controleer formaat (bv. 'dd-mm-jjjj hh:mm:ss') en delimiter/encoding."
        )

    df["__time__"] = t
    df = df.sort_values("__time__")
    df = df.loc[df["__time__"].notna()].reset_index(drop=True)

    out = pd.DataFrame({"time": df["__time__"]})

    def get_numeric(colname):
        if not colname or colname not in df.columns:
            return np.full(len(df), np.nan)
        return pd.to_numeric(df[colname], errors="coerce").to_numpy()

    # ---- speed/dir: single column OR cell range ----
    if use_cell_range:
        if speed_cell_map is None or dir_cell_map is None:
            raise ValueError("Interne fout: speed/dir cell maps ontbreken.")

        if cell_start is None or cell_end is None:
            raise ValueError("Kies start- en eindcel.")

        cell_start = int(cell_start)
        cell_end = int(cell_end)
        if cell_end < cell_start:
            raise ValueError("Eindcel moet >= startcel zijn.")

        cell_ids = list(range(cell_start, cell_end + 1))

        missing_speed = [i for i in cell_ids if i not in speed_cell_map]
        missing_dir = [i for i in cell_ids if i not in dir_cell_map]
        if missing_speed or missing_dir:
            msg = []
            if missing_speed:
                msg.append(f"Ontbrekende Speed# kolommen voor cellen: {missing_speed}")
            if missing_dir:
                msg.append(f"Ontbrekende Dir# kolommen voor cellen: {missing_dir}")
            raise ValueError("\n".join(msg))

        speed_cols = [speed_cell_map[i] for i in cell_ids]
        dir_cols = [dir_cell_map[i] for i in cell_ids]

        speed_mat = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy() for c in speed_cols])
        dir_mat = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy() for c in dir_cols])

        mean_speed, mean_dir = vector_mean_speed_dir(speed_mat, dir_mat)
        out["speed"] = mean_speed
        out["dir_deg"] = mean_dir
    else:
        out["speed"] = get_numeric(speed_col)
        out["dir_deg"] = get_numeric(dir_col)

    out["heading"] = get_numeric(heading_col)
    out["pitch"] = get_numeric(pitch_col)
    out["roll"] = get_numeric(roll_col)

    # "heave" = extra kanaal (bv. Pressure)
    out["heave"] = get_numeric(heave_col)

    # dt checks
    if len(out) >= 3:
        dt_s = np.diff(out["time"].to_numpy(dtype="datetime64[ns]")).astype("timedelta64[ns]").astype(np.int64) / 1e9
        dt_s = dt_s[np.isfinite(dt_s)]
        if len(dt_s):
            dt_med = float(np.median(dt_s))
            dt_min = float(np.min(dt_s))
            dt_max = float(np.max(dt_s))
            gaps = int(np.sum(dt_s > 2.0 * dt_med)) if dt_med > 0 else 0
        else:
            dt_med = dt_min = dt_max = np.nan
            gaps = 0
    else:
        dt_med = dt_min = dt_max = np.nan
        gaps = 0

    stats = {
        "n_rows": int(len(out)),
        "n_time_valid": int(valid),
        "dt_median_s": dt_med,
        "dt_min_s": dt_min,
        "dt_max_s": dt_max,
        "n_gaps_gt_2x_median": gaps
    }

    return out, stats


# ==============================================================
# GUI
# ==============================================================

class AquadoppGUI:
    def __init__(self, master):
        self.master = master
        master.title("Aquadopp ensembles – stroming + attitude")

        self.filepath = StringVar(value="")
        self.export_csv = BooleanVar(value=False)

        self.use_cell_range = BooleanVar(value=False)
        self.cell_start = IntVar(value=1)
        self.cell_end = IntVar(value=1)

        Label(master, text="CSV-bestand:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        Entry(master, textvariable=self.filepath, width=60).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        Button(master, text="Browse", command=self.browse).grid(row=0, column=2, padx=6, pady=4)

        Button(master, text="Kolommen laden", command=self.load_columns).grid(row=1, column=0, columnspan=3, pady=6)

        self.info_lbl = Label(master, text="", fg="gray", justify="left")
        self.info_lbl.grid(row=2, column=0, columnspan=3, sticky="w", padx=6)

        # Dropdowns + dynamic labels
        self.dd_time = StringVar(value="")
        self.dd_speed = StringVar(value="")
        self.dd_dir = StringVar(value="")
        self.dd_heading = StringVar(value="")
        self.dd_pitch = StringVar(value="")
        self.dd_roll = StringVar(value="")
        self.dd_heave = StringVar(value="")  # extra kanaal (bv Pressure)

        row = 3
        self.lbl_time, self.cb_time = self._add_combo(row, "Tijd:", self.dd_time); row += 1
        self.lbl_speed, self.cb_speed = self._add_combo(row, "Snelheid:", self.dd_speed); row += 1
        self.lbl_dir, self.cb_dir = self._add_combo(row, "Richting:", self.dd_dir); row += 1
        self.lbl_heading, self.cb_heading = self._add_combo(row, "Heading:", self.dd_heading); row += 1
        self.lbl_pitch, self.cb_pitch = self._add_combo(row, "Pitch:", self.dd_pitch); row += 1
        self.lbl_roll, self.cb_roll = self._add_combo(row, "Roll:", self.dd_roll); row += 1
        self.lbl_heave, self.cb_heave = self._add_combo(row, "Extra kanaal:", self.dd_heave); row += 1

        # Bind selection changes -> update label text
        self.cb_time.bind("<<ComboboxSelected>>", lambda e: self._update_label(self.lbl_time, "Tijd", self.dd_time.get()))
        self.cb_speed.bind("<<ComboboxSelected>>", lambda e: self._update_label(self.lbl_speed, "Snelheid", self.dd_speed.get()))
        self.cb_dir.bind("<<ComboboxSelected>>", lambda e: self._update_label(self.lbl_dir, "Richting", self.dd_dir.get()))
        self.cb_heading.bind("<<ComboboxSelected>>", lambda e: self._update_label(self.lbl_heading, "Heading", self.dd_heading.get()))
        self.cb_pitch.bind("<<ComboboxSelected>>", lambda e: self._update_label(self.lbl_pitch, "Pitch", self.dd_pitch.get()))
        self.cb_roll.bind("<<ComboboxSelected>>", lambda e: self._update_label(self.lbl_roll, "Roll", self.dd_roll.get()))
        self.cb_heave.bind("<<ComboboxSelected>>", lambda e: self._update_label(self.lbl_heave, "Extra kanaal", self.dd_heave.get()))

        # Range controls
        self.chk_range = ttk.Checkbutton(
            master,
            text="Gemiddelde over meetcellen (range)",
            variable=self.use_cell_range,
            command=self._update_range_state
        )
        self.chk_range.grid(row=row, column=1, sticky="w", padx=6, pady=6); row += 1

        Label(master, text="Start cel:").grid(row=row, column=0, sticky="e", padx=6, pady=3)
        self.sp_start = ttk.Spinbox(master, from_=1, to=9999, textvariable=self.cell_start, width=10)
        self.sp_start.grid(row=row, column=1, sticky="w", padx=6, pady=3)
        row += 1

        Label(master, text="Eind cel:").grid(row=row, column=0, sticky="e", padx=6, pady=3)
        self.sp_end = ttk.Spinbox(master, from_=1, to=9999, textvariable=self.cell_end, width=10)
        self.sp_end.grid(row=row, column=1, sticky="w", padx=6, pady=3)
        row += 1

        self.chk_export = ttk.Checkbutton(master, text="Exporteer analyse-CSV naast input", variable=self.export_csv)
        self.chk_export.grid(row=row, column=1, sticky="w", padx=6, pady=6); row += 1

        Button(master, text="Analyseer en plot", command=self.run).grid(row=row, column=0, columnspan=3, pady=10)

        self.df = None
        self.read_sep = None
        self.read_decimal = None

        # detected cell columns
        self.speed_cell_map = {}
        self.dir_cell_map = {}
        self.detected_cells = []

        self._update_range_state()

    def _add_combo(self, row, label_text, var):
        lbl = Label(self.master, text=label_text)
        lbl.grid(row=row, column=0, sticky="e", padx=6, pady=3)
        cb = ttk.Combobox(self.master, textvariable=var, width=40, state="readonly")
        cb.grid(row=row, column=1, sticky="w", padx=6, pady=3)
        return lbl, cb

    def _update_label(self, lbl, base_name, selected_col):
        if selected_col and str(selected_col).strip():
            lbl.config(text=f"{base_name} ({selected_col}):")
        else:
            lbl.config(text=f"{base_name}:")

    def _update_range_state(self):
        state = "normal" if self.use_cell_range.get() else "disabled"
        self.sp_start.configure(state=state)
        self.sp_end.configure(state=state)

    def browse(self):
        path = filedialog.askopenfilename(
            title="Kies Aquadopp CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.filepath.set(path)
            self.info_lbl.config(text="")

    def load_columns(self):
        try:
            fp = self.filepath.get()
            df, sep, dec = read_aquadopp_csv(fp)
            self.df = df
            self.read_sep = sep
            self.read_decimal = dec

            cols = list(df.columns)

            for cb in [self.cb_time, self.cb_speed, self.cb_dir,
                       self.cb_heading, self.cb_pitch, self.cb_roll, self.cb_heave]:
                cb["values"] = cols

            guesses = autodetect_columns(cols)
            self.dd_time.set(guesses["time"])
            self.dd_speed.set(guesses["speed"])
            self.dd_dir.set(guesses["dir"])
            self.dd_heading.set(guesses["heading"])
            self.dd_pitch.set(guesses["pitch"])
            self.dd_roll.set(guesses["roll"])
            self.dd_heave.set(guesses["heave"])

            # update labels after autodetect
            self._update_label(self.lbl_time, "Tijd", self.dd_time.get())
            self._update_label(self.lbl_speed, "Snelheid", self.dd_speed.get())
            self._update_label(self.lbl_dir, "Richting", self.dd_dir.get())
            self._update_label(self.lbl_heading, "Heading", self.dd_heading.get())
            self._update_label(self.lbl_pitch, "Pitch", self.dd_pitch.get())
            self._update_label(self.lbl_roll, "Roll", self.dd_roll.get())
            self._update_label(self.lbl_heave, "Extra kanaal", self.dd_heave.get())

            # detect Speed#n and Dir#n columns (robust)
            self.speed_cell_map = find_cell_columns(cols, kind="speed")
            self.dir_cell_map = find_cell_columns(cols, kind="dir")
            self.detected_cells = sorted(set(self.speed_cell_map.keys()) & set(self.dir_cell_map.keys()))

            if self.detected_cells:
                mn, mx = self.detected_cells[0], self.detected_cells[-1]
                self.sp_start.configure(from_=mn, to=mx)
                self.sp_end.configure(from_=mn, to=mx)
                self.cell_start.set(mn)
                self.cell_end.set(mx)
                range_txt = f"Meetcellen gevonden: {mn}–{mx} (Speed# & Dir#)."
            else:
                self.use_cell_range.set(False)
                self._update_range_state()
                range_txt = "Geen Speed#n/Dir#n meetcelkolommen gevonden (range-gemiddelde niet beschikbaar)."

            self.info_lbl.config(
                text=f"Ingelezen met sep='{sep}' en decimal='{dec}'. Kolommen: {len(cols)}.\n{range_txt}"
            )
            messagebox.showinfo("OK", "Kolomnamen geladen.")
        except Exception as e:
            messagebox.showerror("Fout bij inlezen CSV", str(e))

    def run(self):
        if self.df is None:
            messagebox.showerror("Fout", "Geen data geladen. Klik eerst op 'Kolommen laden'.")
            return

        use_range = bool(self.use_cell_range.get())
        if use_range and not self.detected_cells:
            messagebox.showerror(
                "Fout",
                "Range-gemiddelde is aangevinkt, maar er zijn geen Speed#n/Dir#n kolommen gedetecteerd."
            )
            return

        try:
            out, stats = analyse_ensembles(
                df=self.df,
                time_col=self.dd_time.get(),
                speed_col=self.dd_speed.get(),
                dir_col=self.dd_dir.get(),
                heading_col=self.dd_heading.get(),
                pitch_col=self.dd_pitch.get(),
                roll_col=self.dd_roll.get(),
                heave_col=self.dd_heave.get(),
                use_cell_range=use_range,
                cell_start=int(self.cell_start.get()) if use_range else None,
                cell_end=int(self.cell_end.get()) if use_range else None,
                speed_cell_map=self.speed_cell_map,
                dir_cell_map=self.dir_cell_map,
            )
        except Exception as e:
            messagebox.showerror("Fout tijdens analyse", str(e))
            return

        # Optional export
        if self.export_csv.get():
            try:
                in_fp = self.filepath.get()
                base, _ = os.path.splitext(in_fp)
                suffix = f"_cells_{self.cell_start.get()}_{self.cell_end.get()}" if use_range else "_singlecol"
                out_fp = base + f"_ensembles_analyse{suffix}.csv"
                out.to_csv(out_fp, index=False)
            except Exception as e:
                messagebox.showwarning("Export waarschuwing", f"Kon analyse-CSV niet wegschrijven:\n{e}")

        # Info text
        mode_txt = (
            f"Mode: gemiddelde cellen {self.cell_start.get()}–{self.cell_end.get()}"
            if use_range else
            "Mode: enkele kolom"
        )
        msg = (
            f"{mode_txt}\n"
            f"Rijen: {stats['n_rows']}\n"
            f"Geldige tijden: {stats['n_time_valid']}\n"
            f"dt median: {stats['dt_median_s']:.3f} s\n"
            f"dt min/max: {stats['dt_min_s']:.3f} / {stats['dt_max_s']:.3f} s\n"
            f"Gaps (>2× median dt): {stats['n_gaps_gt_2x_median']}"
        )
        self.info_lbl.config(text=msg)

        # ==========================================================
        # Plot: Pressure/extra kanaal in apart deelfiguur
        #   - speed
        #   - dir
        #   - heading
        #   - pitch+roll
        #   - pressure/extra kanaal (apart)
        # ==========================================================

        has_extra = not np.all(np.isnan(out["heave"]))
        nrows = 5 if has_extra else 4

        fig, axes = plt.subplots(nrows, 1, figsize=(11, 10 if has_extra else 9), sharex=True)

        # axes handling
        if nrows == 4:
            ax_speed, ax_dir, ax_head, ax_pr = axes
            ax_extra = None
        else:
            ax_speed, ax_dir, ax_head, ax_pr, ax_extra = axes

        title_suffix = f" (cellen {self.cell_start.get()}–{self.cell_end.get()})" if use_range else ""

        # Speed
        ax_speed.plot(out["time"], out["speed"])
        ax_speed.set_ylabel("Speed (m/s)")
        ax_speed.set_title("Snelheid per ensemble" + title_suffix)
        ax_speed.grid(True)

        # Direction
        ax_dir.plot(out["time"], out["dir_deg"])
        ax_dir.set_ylabel("Dir (°)")
        ax_dir.set_title("Stromingsrichting / Dir" + title_suffix)
        ax_dir.grid(True)

        # Heading
        ax_head.plot(out["time"], out["heading"])
        ax_head.set_ylabel("Heading (°)")
        ax_head.set_title("Instrument heading")
        ax_head.grid(True)

        # Pitch/Roll
        ax_pr.plot(out["time"], out["pitch"], label="Pitch", linestyle="-")
        ax_pr.plot(out["time"], out["roll"],  label="Roll",  linestyle="--")
        ax_pr.set_ylabel("°")
        ax_pr.set_title("Pitch / Roll")
        ax_pr.grid(True)
        ax_pr.legend(loc="upper right")

        # Extra kanaal apart (bv Pressure)
        if ax_extra is not None:
            extra_name = self.dd_heave.get().strip() if self.dd_heave.get() else "Extra"
            ax_extra.plot(out["time"], out["heave"])
            ax_extra.set_title(f"{extra_name} (apart)")
            ax_extra.set_ylabel("waarde")
            ax_extra.grid(True)

        axes[-1].set_xlabel("Tijd")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = Tk()
    app = AquadoppGUI(root)
    root.mainloop()
