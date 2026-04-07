#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
from typing import Optional

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


TIME_CANDIDATES = ["tijd", "time", "datetime", "date", "datum", "sample time", "timestamp"]


def _norm(s: str) -> str:
    return str(s).strip().lower()


def detect_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = "".join([f.readline() for _ in range(40)])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ";"


def detect_header_row(path: str, delimiter: str) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for i, line in enumerate(lines[:80]):
        cells = [_norm(x) for x in line.strip().split(delimiter)]
        if any(c in TIME_CANDIDATES for c in cells):
            return i
    return 0


def find_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cn = _norm(c)
        if cn in TIME_CANDIDATES:
            return c
    for c in df.columns:
        cn = _norm(c)
        if any(t in cn for t in TIME_CANDIDATES):
            return c
    return None


def read_csv_robust(path: str) -> pd.DataFrame:
    sep = detect_delimiter(path)
    header_row = detect_header_row(path, sep)
    df = pd.read_csv(path, sep=sep, header=header_row, encoding="utf-8", engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Drop empty/unnamed columns that come from malformed exports.
    keep = []
    for c in df.columns:
        lc = _norm(c)
        if lc == "" or lc.startswith("unnamed"):
            continue
        keep.append(c)
    df = df[keep].copy()

    tcol = find_time_col(df)
    if tcol:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce", dayfirst=True)
        df = df.dropna(subset=[tcol]).sort_values(tcol)

    for c in df.columns:
        if c == tcol:
            continue
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce",
        )
    return df


class TimePlotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Tijdgrafiek")
        self.geometry("1200x800")

        self.df: Optional[pd.DataFrame] = None
        self.path: Optional[str] = None
        self.time_col: Optional[str] = None

        self.y_var = tk.StringVar(value="")
        self.msg_var = tk.StringVar(value="Kies een CSV om te starten.")

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="CSV openen...", command=self.open_csv).pack(side=tk.LEFT)
        self.file_lbl = ttk.Label(top, text="(geen bestand)")
        self.file_lbl.pack(side=tk.LEFT, padx=8)

        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Y-kolom:").pack(side=tk.LEFT)
        self.y_combo = ttk.Combobox(ctrl, textvariable=self.y_var, state="readonly", width=40)
        self.y_combo.pack(side=tk.LEFT, padx=6)

        ttk.Button(ctrl, text="Teken grafiek", command=self.plot).pack(side=tk.LEFT, padx=6)
        ttk.Label(ctrl, textvariable=self.msg_var).pack(side=tk.LEFT, padx=10)

        fig_frame = ttk.Frame(self)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.fig = plt.Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, fig_frame).update()

    def open_csv(self):
        p = filedialog.askopenfilename(
            title="Kies CSV",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return
        try:
            df = read_csv_robust(p)
        except Exception as exc:
            messagebox.showerror("Fout", f"Kon CSV niet inlezen:\n{exc}")
            return

        tcol = find_time_col(df)
        if not tcol:
            messagebox.showerror("Fout", "Geen tijdkolom gevonden (bv. tijd/time/datetime).")
            return

        numeric_cols = [c for c in df.columns if c != tcol and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            messagebox.showerror("Fout", "Geen numerieke kolommen gevonden om te plotten.")
            return

        self.df = df
        self.path = p
        self.time_col = tcol
        self.file_lbl.configure(text=os.path.basename(p))
        self.y_combo.configure(values=numeric_cols)
        self.y_var.set("druk" if "druk" in numeric_cols else numeric_cols[0])
        self.msg_var.set(f"{len(df)} rijen geladen | tijdkolom: {tcol}")
        self.plot()

    def plot(self):
        if self.df is None or self.time_col is None:
            return
        ycol = self.y_var.get().strip()
        if ycol == "" or ycol not in self.df.columns:
            return

        dfp = self.df[[self.time_col, ycol]].dropna()
        if dfp.empty:
            self.msg_var.set("Geen geldige punten om te tekenen.")
            return

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.plot(dfp[self.time_col], dfp[ycol], linewidth=1.3)
        ax.set_title(f"{ycol} over tijd")
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(ycol)
        ax.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()
        self.canvas.draw()
        self.msg_var.set(f"Geplot: {ycol} ({len(dfp)} punten)")


if __name__ == "__main__":
    app = TimePlotApp()
    app.mainloop()
