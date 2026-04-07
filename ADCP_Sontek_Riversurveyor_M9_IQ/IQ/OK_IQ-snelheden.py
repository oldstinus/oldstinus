#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SonTek IQ ADCP – VELBEAM analyser + GUI (SNR functies verwijderd)
AANPASSINGEN:
- Celselectie scrollbaar
- Resultante snelheid nu expliciet MET teken (+/-)
- u- en v-component worden gelogd, geplot en geëxporteerd (met teken)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math

CELL_LOC_RE = re.compile(r"^Cell(\d+)\s+Location\s+\((Center|Skew)\)\s+\(m\)$", re.IGNORECASE)
VEL_BEAM_RE = re.compile(r"^Cell(\d+)\s+Velocity\s+\(beam\)\.(\d+)\s+\(m/s\)$", re.IGNORECASE)


def find_time_column(df: pd.DataFrame) -> str:
    for cand in ["Sample Time", "Time", "DateTime", "Datetime", "Timestamp", "Sample time", "2 Sample Time"]:
        if cand in df.columns:
            return cand
    for c in df.columns[:10]:
        try:
            pd.to_datetime(df[c].iloc[:50], errors="raise", dayfirst=True)
            return c
        except Exception:
            continue
    raise ValueError("Geen tijdkolom gevonden.")


def find_pressure_column(df: pd.DataFrame) -> str:
    for cand in ["Pressure (dbar)", "Pressure", "P", "Druk", "Pressure (uncorrected) (dbar)"]:
        if cand in df.columns:
            return cand
    for c in df.columns[:10]:
        try:
            pd.to_numeric(df[c].iloc[:50], errors="raise")
            return c
        except Exception:
            continue
    raise ValueError("Geen drukkolom gevonden.")


def find_velocity_column(df: pd.DataFrame) -> str:
    for cand in [
        "Velocity (X-OBS).X-Center (Beam 1 Only) (m/s)",
        "Velocity (mean) (m/s)",
        "Velocity",
        "Vel",
        "X Velocity",
    ]:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "Velocity" in c:
            try:
                pd.to_numeric(df[c].iloc[:50], errors="raise")
                return c
            except Exception:
                continue
    raise ValueError("Geen velocity kolom gevonden.")


def parse_cells(df: pd.DataFrame):
    cells = set()
    loc_center = {}
    loc_skew = {}
    vel_cols = {}

    for col in df.columns:
        m_loc = CELL_LOC_RE.match(col)
        if m_loc:
            cell = int(m_loc.group(1))
            kind = m_loc.group(2).lower()
            cells.add(cell)
            if kind == "center":
                loc_center[cell] = col
            else:
                loc_skew[cell] = col

        m_vel = VEL_BEAM_RE.match(col)
        if m_vel:
            cell = int(m_vel.group(1))
            beam = int(m_vel.group(2))
            cells.add(cell)
            vel_cols[(cell, beam)] = col

    return sorted(cells), loc_center, loc_skew, vel_cols


def format_value(val) -> str:
    if pd.isna(val):
        return ""
    try:
        num = float(val)
    except Exception:
        return str(val)

    num = round(num, 4)
    neg = (num < 0)
    num = abs(num)

    integer_part = int(math.floor(num))
    frac_part = num - integer_part

    int_str = f"{integer_part:,}".replace(",", ".")
    frac_str = f"{frac_part:.4f}"[2:]
    sign_str = "-" if neg else ""

    return f"{sign_str}{int_str},{frac_str}"


def signed_resultant(u: pd.Series | None, v: pd.Series | None):
    """
    Geef resultante MET teken.
    - beide: |U,V| -> magnitude * sign(u)
    - enkel u: u
    - enkel v: v
    """
    if u is None and v is None:
        return None, "horizontale snelheid"
    if u is None:
        return v, "horizontale snelheid (alleen v, met teken)"
    if v is None:
        return u, "horizontale snelheid (alleen u, met teken)"

    mag = np.sqrt(u**2 + v**2)
    sign_u = np.sign(u.to_numpy())
    sign_u = np.where(sign_u == 0, 1, sign_u)  # 0 -> 1 (geen flip naar 0)
    speed_signed = pd.Series(mag.to_numpy() * sign_u, index=u.index)
    return speed_signed, "horizontale snelheid (met teken op basis van u)"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SonTek IQ – Snelheid per beam")
        self.geometry("1100x700")

        self.vel_path = tk.StringVar(value="")
        self.pressure_path = tk.StringVar(value="")

        self.beam_on = {b: tk.BooleanVar(value=True) for b in (1, 2, 3, 4)}
        self.cell_on_vars = {}
        self.theta_deg = tk.DoubleVar(value=25.0)

        self.df = None
        self.pressure_df = None
        self.time_col = None
        self.vel_cols = {}
        self.loc_center = {}
        self.loc_skew = {}
        self.cells = []
        self.cell_heights = {}

        self._build_interface()

    def _build_interface(self):
        pad = 6
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        file_frame = ttk.LabelFrame(frm, text="Input bestanden")
        file_frame.pack(fill="x", pady=pad)

        f_row1 = ttk.Frame(file_frame)
        f_row1.pack(fill="x", pady=3)
        ttk.Label(f_row1, text="VELBEAM CSV:").pack(side="left")
        ttk.Entry(f_row1, textvariable=self.vel_path, width=85).pack(side="left", padx=pad)
        ttk.Button(f_row1, text="Bladeren...", command=self.browse_vel).pack(side="left")

        f_row2 = ttk.Frame(file_frame)
        f_row2.pack(fill="x", pady=3)
        ttk.Label(f_row2, text="Druk CSV:").pack(side="left")
        ttk.Entry(f_row2, textvariable=self.pressure_path, width=89).pack(side="left", padx=pad)
        ttk.Button(f_row2, text="Bladeren...", command=self.browse_pressure).pack(side="left")

        load_frame = ttk.Frame(frm)
        load_frame.pack(fill="x", pady=pad)
        ttk.Button(load_frame, text="Laad data", command=self.on_load).pack(side="left")

        beam_frame = ttk.LabelFrame(frm, text="Beams selecteren")
        beam_frame.pack(fill="x", pady=pad)
        for b in (1, 2, 3, 4):
            row = ttk.Frame(beam_frame)
            row.pack(fill="x", pady=2)
            ttk.Checkbutton(row, text=f"Beam {b}", variable=self.beam_on[b]).pack(side="left", padx=pad)

        # Scrollbare celselectie
        self.cell_frame = ttk.LabelFrame(frm, text="Cellen (hoogtes) selecteren")
        self.cell_frame.pack(fill="x", pady=pad)

        self.cell_canvas = tk.Canvas(self.cell_frame, highlightthickness=0, height=140)
        self.cell_scrollbar = ttk.Scrollbar(self.cell_frame, orient="vertical", command=self.cell_canvas.yview)
        self.cell_canvas.configure(yscrollcommand=self.cell_scrollbar.set)

        self.cell_canvas.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        self.cell_scrollbar.pack(side="right", fill="y", padx=(0, 6), pady=6)

        self.cell_inner = ttk.Frame(self.cell_canvas)
        self._cell_inner_window = self.cell_canvas.create_window((0, 0), window=self.cell_inner, anchor="nw")

        self.cell_inner.bind("<Configure>", self._on_cell_inner_configure)
        self.cell_canvas.bind("<Configure>", self._on_cell_canvas_configure)
        self._bind_mousewheel(self.cell_canvas)

        param_frame = ttk.Frame(frm)
        param_frame.pack(fill="x", pady=pad)
        ttk.Label(param_frame, text="Bundelhoek θ (deg):").pack(side="left", padx=pad)
        ttk.Entry(param_frame, textvariable=self.theta_deg, width=5).pack(side="left")

        action_frame = ttk.Frame(frm)
        action_frame.pack(fill="x", pady=pad)
        ttk.Button(action_frame, text="Toon grafieken per beam", command=self.on_plot_beam_graphs).pack(side="left")
        ttk.Button(action_frame, text="Bereken gemiddelde", command=self.on_calculate_average).pack(side="left", padx=pad)
        ttk.Button(action_frame, text="Exporteer naar CSV", command=self.on_export_csv).pack(side="left")

        output_frame = ttk.LabelFrame(frm, text="Log / Resultaat")
        output_frame.pack(fill="both", expand=True, pady=pad)
        self.txt = tk.Text(output_frame, height=15, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=8, pady=8)

        self._log("Selecteer een VELBEAM CSV (en eventueel een Druk CSV) en klik 'Laad data'.")

    def _on_cell_inner_configure(self, _event=None):
        self.cell_canvas.configure(scrollregion=self.cell_canvas.bbox("all"))

    def _on_cell_canvas_configure(self, event=None):
        if event is not None:
            self.cell_canvas.itemconfigure(self._cell_inner_window, width=event.width)

    def _bind_mousewheel(self, widget):
        widget.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        widget.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

    def _on_mousewheel(self, event):
        x, y = self.winfo_pointerx(), self.winfo_pointery()
        w = self.cell_canvas.winfo_containing(x, y)
        if w is None:
            return
        if str(w).startswith(str(self.cell_canvas)) or str(w).startswith(str(self.cell_inner)):
            delta = int(-1 * (event.delta / 120)) if event.delta != 0 else 0
            if delta != 0:
                self.cell_canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        x, y = self.winfo_pointerx(), self.winfo_pointery()
        w = self.cell_canvas.winfo_containing(x, y)
        if w is None:
            return
        if str(w).startswith(str(self.cell_canvas)) or str(w).startswith(str(self.cell_inner)):
            if event.num == 4:
                self.cell_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.cell_canvas.yview_scroll(1, "units")

    def _log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")

    def browse_vel(self):
        fp = filedialog.askopenfilename(
            title="Selecteer VELBEAM CSV",
            filetypes=[("CSV bestanden", "*.csv"), ("Alle bestanden", "*.*")],
        )
        if fp:
            self.vel_path.set(fp)

    def browse_pressure(self):
        fp = filedialog.askopenfilename(
            title="Selecteer DRUK CSV",
            filetypes=[("CSV bestanden", "*.csv"), ("Alle bestanden", "*.*")],
        )
        if fp:
            self.pressure_path.set(fp)

    def _read_csv_auto(self, filepath: str) -> pd.DataFrame:
        with open(filepath, "r", newline="") as f:
            first_line = f.readline()

        if first_line.count(";") > first_line.count(",") and first_line.count(";") > first_line.count("\t"):
            delim = ";"
        elif first_line.count("\t") > first_line.count(","):
            delim = "\t"
        else:
            delim = ","

        decimal_char = "."
        if delim != ",":
            with open(filepath, "r", newline="") as f:
                _ = f.readline()
                second_line = f.readline()
            if re.search(r"\d+,\d+", second_line):
                decimal_char = ","

        return pd.read_csv(filepath, sep=delim, decimal=decimal_char)

    def on_load(self):
        vp = self.vel_path.get().strip()
        pp = self.pressure_path.get().strip()

        if not vp:
            messagebox.showerror("Fout", "Selecteer een VELBEAM CSV-bestand.")
            return

        try:
            self.df = self._read_csv_auto(vp)
            self.pressure_df = self._read_csv_auto(pp) if pp else None
        except Exception as e:
            messagebox.showerror("Fout", f"Kon bestanden niet laden:\n{e}")
            return

        if self.pressure_df is not None:
            if "Sample number" in self.df.columns and "Sample Number" in self.pressure_df.columns:
                self.df.rename(columns={"Sample number": "Sample Number"}, inplace=True)
            if "Sample number" in self.pressure_df.columns and "Sample Number" in self.df.columns:
                self.pressure_df.rename(columns={"Sample number": "Sample Number"}, inplace=True)

        try:
            self.time_col = find_time_column(self.df)
        except Exception:
            self.time_col = None

        self.cells, self.loc_center, self.loc_skew, self.vel_cols = parse_cells(self.df)

        self.cell_heights = {}
        for cell in list(self.cells):
            loc_c = self.loc_center.get(cell)
            loc_s = self.loc_skew.get(cell)
            max_c = float("nan")
            max_s = float("nan")

            if loc_c and loc_c in self.df.columns:
                c_vals = pd.to_numeric(self.df[loc_c], errors="coerce").dropna()
                if not c_vals.empty:
                    max_c = float(c_vals.max())
            if loc_s and loc_s in self.df.columns:
                s_vals = pd.to_numeric(self.df[loc_s], errors="coerce").dropna()
                if not s_vals.empty:
                    max_s = float(s_vals.max())

            if math.isnan(max_c) and math.isnan(max_s):
                continue
            elif math.isnan(max_c):
                val = max_s
            elif math.isnan(max_s):
                val = max_c
            else:
                val = max(max_c, max_s)

            self.cell_heights[cell] = val

        self.cells = sorted(self.cell_heights.keys())

        for widget in self.cell_inner.winfo_children():
            widget.destroy()
        self.cell_on_vars.clear()

        max_cols = 6
        for i, cell in enumerate(self.cells):
            var = tk.BooleanVar(value=True)
            self.cell_on_vars[cell] = var
            h = self.cell_heights.get(cell, float("nan"))
            label = f"Cel {cell}" if math.isnan(h) else f"Cel {cell} ({h:.3f} m)"
            cb = ttk.Checkbutton(self.cell_inner, text=label, variable=var)
            r = i // max_cols
            c = i % max_cols
            cb.grid(row=r, column=c, sticky="w", padx=6, pady=2)

        for c in range(max_cols):
            self.cell_inner.grid_columnconfigure(c, weight=1)

        self.cell_canvas.update_idletasks()
        self.cell_canvas.configure(scrollregion=self.cell_canvas.bbox("all"))

        self._log("Data geladen.")
        self._log(f"Rijen: {len(self.df)}; Kolommen: {len(self.df.columns)}")
        if self.cells:
            self._log(f"Cellen: {min(self.cells)} .. {max(self.cells)} (n={len(self.cells)})")
        else:
            self._log("Geen cellen gevonden in data.")
        self._log(f"Velocity kolommen: {len(self.vel_cols)}")

        if self.pressure_df is not None:
            try:
                pcol = find_pressure_column(self.pressure_df)
                self._log(f"Druk-CSV geladen: {len(self.pressure_df)} rijen; kolom '{pcol}' gebruikt als druk.")
            except Exception as e:
                self._log(f"Druk-CSV geladen, maar drukkolom niet gevonden: {e}")

    def on_plot_beam_graphs(self):
        if self.df is None:
            messagebox.showerror("Fout", "Laad eerst de data.")
            return

        time = None
        if self.time_col:
            try:
                time = pd.to_datetime(self.df[self.time_col], errors="coerce", dayfirst=True)
            except Exception:
                time = self.df[self.time_col]
        if time is None:
            time = np.arange(len(self.df))

        for b in (1, 2, 3, 4):
            if not self.beam_on[b].get():
                continue

            cols = [
                self.vel_cols[(cell, b)]
                for cell, var in self.cell_on_vars.items()
                if var.get() and (cell, b) in self.vel_cols
            ]
            if not cols:
                continue

            data = self.df[cols].apply(pd.to_numeric, errors="coerce")

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)

            for col in cols:
                m = VEL_BEAM_RE.match(col)
                cell_num = m.group(1) if m else col
                ax.plot(time, data[col], label=f"Cel {cell_num}")

            ax.axhline(0.0, linewidth=0.8)
            ax.set_title(f"Snelheid Beam {b} per geselecteerde cel (met teken +/−)")
            ax.set_xlabel("Tijd")
            ax.set_ylabel("Snelheid (m/s)")
            ax.legend(title="Cel")
            fig.tight_layout()

        if plt.get_fignums():
            plt.show()

    def on_calculate_average(self):
        if self.df is None:
            messagebox.showerror("Fout", "Laad eerst de data.")
            return

        selected_cells = [cell for cell, var in self.cell_on_vars.items() if var.get()]
        selected_beams = [b for b, var in self.beam_on.items() if var.get()]

        if not selected_cells:
            messagebox.showerror("Fout", "Geen cellen geselecteerd.")
            return
        if not selected_beams:
            messagebox.showerror("Fout", "Geen beams geselecteerd.")
            return

        # Gemiddelde per beam (over geselecteerde cellen) -> tijdreeks (met teken, zoals in data)
        beam_series = {}
        for b in selected_beams:
            cols = [self.vel_cols[(cell, b)] for cell in selected_cells if (cell, b) in self.vel_cols]
            if cols:
                data = self.df[cols].apply(pd.to_numeric, errors="coerce")
                beam_series[b] = data.mean(axis=1, skipna=True)
            else:
                beam_series[b] = None

        theta = math.radians(self.theta_deg.get())
        sin_t = math.sin(theta) if math.sin(theta) != 0 else 1e-12

        u = v = None
        if beam_series.get(1) is not None and beam_series.get(2) is not None:
            u = (beam_series[1] - beam_series[2]) / (2.0 * sin_t)
        if beam_series.get(4) is not None and beam_series.get(3) is not None:
            v = (beam_series[4] - beam_series[3]) / (2.0 * sin_t)

        if u is None and v is None:
            messagebox.showerror("Fout", "Onvoldoende beams (kies minstens 1&2 of 3&4).")
            return

        speed_signed, speed_label = signed_resultant(u, v)

        # Logging
        heights = [self.cell_heights.get(cell, np.nan) for cell in selected_cells]
        heights = [h for h in heights if not (pd.isna(h) or math.isnan(h))]
        hmin = min(heights) if heights else float("nan")
        hmax = max(heights) if heights else float("nan")

        self._log("---- Gemiddelde berekend ----")
        if not math.isnan(hmin):
            self._log(f"Hoogte-range: {hmin:.3f} .. {hmax:.3f} m (cellen n={len(selected_cells)})")
        else:
            self._log(f"Aantal geselecteerde cellen: {len(selected_cells)}")
        self._log(f"θ = {self.theta_deg.get():.2f}°")

        if u is not None:
            self._log(f"Gemiddelde u (met teken): {float(np.nanmean(u)):.4f} m/s")
        if v is not None:
            self._log(f"Gemiddelde v (met teken): {float(np.nanmean(v)):.4f} m/s")
        self._log(f"Gemiddelde {speed_label}: {float(np.nanmean(speed_signed)):.4f} m/s")

        # Plot: u, v en signed speed
        time = None
        if self.time_col:
            try:
                time = pd.to_datetime(self.df[self.time_col], errors="coerce", dayfirst=True)
            except Exception:
                time = self.df[self.time_col]
        if time is None:
            time = np.arange(len(self.df))

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(time, speed_signed, label="Resultante (met teken)")
        if u is not None:
            ax.plot(time, u, linestyle="--", label="u (met teken)")
        if v is not None:
            ax.plot(time, v, linestyle=":", label="v (met teken)")

        ax.axhline(0.0, linewidth=0.8)
        ax.set_title("Horizontale snelheid: u, v en resultante (allemaal met +/−)")
        ax.set_xlabel("Tijd")
        ax.set_ylabel("Snelheid (m/s)")

        # Druk op tweede as indien aanwezig
        if self.pressure_df is not None:
            try:
                p_col = find_pressure_column(self.pressure_df)
                pressure_vals = pd.to_numeric(self.pressure_df[p_col], errors="coerce")
                ax2 = ax.twinx()
                ax2.plot(time, pressure_vals, "r-", linewidth=1, label="Druk")
                ax2.set_ylabel("Druk (dbar)", color="r")
                ax2.tick_params(axis="y", labelcolor="r")
            except Exception as e:
                self._log(f"Kon druk niet plotten: {e}")

            # Optioneel: observed velocity
            try:
                vel_col = find_velocity_column(self.pressure_df)
                observed_vel = pd.to_numeric(self.pressure_df[vel_col], errors="coerce")
                ax.plot(time, observed_vel, "g--", linewidth=1, label="Gemeten snelheid")
            except Exception:
                pass

        ax.legend()
        fig.tight_layout()
        plt.show()

    def on_export_csv(self):
        if self.df is None:
            messagebox.showerror("Fout", "Laad eerst de data.")
            return

        export_path = filedialog.asksaveasfilename(
            title="Opslaan als",
            defaultextension=".csv",
            filetypes=[("CSV bestanden", "*.csv"), ("Alle bestanden", "*.*")],
        )
        if not export_path:
            return

        try:
            export_text = self._generate_export_text()
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(export_text)
            messagebox.showinfo("Export voltooid", f"Data succesvol opgeslagen naar {export_path}")
            self._log(f"CSV export opgeslagen: {export_path}")
        except Exception as e:
            messagebox.showerror("Fout", f"Export mislukt:\n{e}")

    def _generate_export_text(self) -> str:
        selected_cells = [cell for cell, var in self.cell_on_vars.items() if var.get()]
        selected_beams = [b for b, var in self.beam_on.items() if var.get()]

        if not selected_cells:
            raise RuntimeError("Geen cellen geselecteerd")
        if not selected_beams:
            raise RuntimeError("Geen beams geselecteerd")

        out_df = pd.DataFrame()

        sample_col = None
        for cand in ["Sample Number", "Sample number"]:
            if cand in self.df.columns:
                sample_col = cand
                break

        if self.time_col:
            out_df["Tijd"] = self.df[self.time_col].astype(str)

        if sample_col:
            out_df.insert(0, sample_col, self.df[sample_col])

        # Beam gemiddelden (over geselecteerde cellen) -> met teken
        beam_series = {}
        for b in selected_beams:
            cols = [self.vel_cols[(cell, b)] for cell in selected_cells if (cell, b) in self.vel_cols]
            if cols:
                data = self.df[cols].apply(pd.to_numeric, errors="coerce")
                series = data.mean(axis=1, skipna=True)
            else:
                series = pd.Series([np.nan] * len(self.df))
            out_df[f"Beam {b} (m/s)"] = series
            beam_series[b] = series

        # u, v en resultante (met teken)
        theta = math.radians(self.theta_deg.get())
        sin_t = math.sin(theta) if math.sin(theta) != 0 else 1e-12

        u = v = None
        if 1 in beam_series and 2 in beam_series and not beam_series[1].isna().all() and not beam_series[2].isna().all():
            u = (beam_series[1] - beam_series[2]) / (2.0 * sin_t)
            out_df["u (m/s)"] = u
        if 4 in beam_series and 3 in beam_series and not beam_series[4].isna().all() and not beam_series[3].isna().all():
            v = (beam_series[4] - beam_series[3]) / (2.0 * sin_t)
            out_df["v (m/s)"] = v

        spd, _ = signed_resultant(u, v)
        if spd is not None:
            out_df["Resultante (m/s)"] = spd

        # Druk toevoegen
        if self.pressure_df is not None:
            try:
                p_col = find_pressure_column(self.pressure_df)
            except Exception:
                p_col = None

            if p_col:
                if sample_col and sample_col in self.pressure_df.columns:
                    out_df = pd.merge(out_df, self.pressure_df[[sample_col, p_col]], on=sample_col, how="left")
                elif self.time_col and self.time_col in self.pressure_df.columns:
                    out_df = pd.merge(
                        out_df,
                        self.pressure_df[[self.time_col, p_col]],
                        left_on="Tijd",
                        right_on=self.time_col,
                        how="left",
                    )
                    if self.time_col in out_df.columns and "Tijd" in out_df.columns and self.time_col != "Tijd":
                        out_df.drop(columns=[self.time_col], inplace=True)
                else:
                    out_df[p_col] = self.pressure_df[p_col].values

                out_df.rename(columns={p_col: "Druk (dbar)"}, inplace=True)

        if sample_col and sample_col in out_df.columns:
            out_df.drop(columns=[sample_col], inplace=True)

        col_names = list(out_df.columns)
        lines = ["\t".join(col_names)]

        for _, row in out_df.iterrows():
            values = [format_value(row[col]) for col in col_names]
            lines.append("\t".join(values))

        return "\n".join(lines)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
