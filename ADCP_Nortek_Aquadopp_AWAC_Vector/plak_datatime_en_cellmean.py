#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SEN-tijd (mm dd yyyy HH MM SS) rij-voor-rij kopiëren naar *.v* files
+ gemiddelde over cell range (positie-gebaseerd)

Output per v-file:
  DateTime, Mean_Cell_<min>_<max>

BELANGRIJK:
- zink301.sen heeft géén header -> we lezen ALWAYS header=None en whitespace-separated
- DateTime = kolommen 0..5 exact, geen auto-detect
- v-files: whitespace-separated, header=None
- Als #rijen sen != #rijen v: knipt op min(n_sen, n_v)
"""

import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd


# ----------------------------
# Read SEN: strict
# ----------------------------
def read_sen_datetime_series(sen_path: str) -> pd.Series:
    """
    SEN format per rij (zoals jouw file):
    0: Month, 1: Day, 2: Year, 3: Hour, 4: Minute, 5: Second, ...
    """
    df = pd.read_csv(sen_path, sep=r"\s+", engine="python", header=None)

    if df.shape[1] < 6:
        raise ValueError(f"SEN file heeft te weinig kolommen (<6): {os.path.basename(sen_path)}")

    mm = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    dd = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    yy = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    hh = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    mi = pd.to_numeric(df.iloc[:, 4], errors="coerce")
    ss = pd.to_numeric(df.iloc[:, 5], errors="coerce")

    dt = pd.to_datetime(
        dict(year=yy, month=mm, day=dd, hour=hh, minute=mi, second=ss),
        errors="coerce"
    )

    if dt.notna().sum() == 0:
        raise ValueError(
            "Kon geen geldige DateTime maken uit SEN kolommen 0..5. "
            "Controleer of format echt: mm dd yyyy HH MM SS is."
        )

    return dt


# ----------------------------
# Read V-file: strict whitespace, header=None
# ----------------------------
def read_vfile(v_path: str) -> pd.DataFrame:
    df = pd.read_csv(v_path, sep=r"\s+", engine="python", header=None)
    return df


# ----------------------------
# Cell mean by position (columns are cells)
# ----------------------------
def compute_cell_mean(df_v: pd.DataFrame, cell_min: int, cell_max: int) -> pd.Series:
    """
    V-file heeft enkel cellen per rij:
      cel 1 = kolom 0
      cel 2 = kolom 1
      ...
    Input cell_min/cell_max zijn 1-based.
    """
    if cell_min < 1:
        raise ValueError("cell_min moet >= 1 zijn")
    if cell_max < cell_min:
        raise ValueError("cell_max moet >= cell_min zijn")

    ncols = df_v.shape[1]
    if ncols == 0:
        raise ValueError("V-file bevat geen kolommen.")

    if cell_max > ncols:
        raise ValueError(f"cell_max={cell_max}, maar V-file heeft slechts {ncols} kolommen (cellen).")

    # slice: 0-based
    subset = df_v.iloc[:, cell_min - 1:cell_max].apply(pd.to_numeric, errors="coerce")
    return subset.mean(axis=1, skipna=True)


# ----------------------------
# GUI
# ----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SEN-tijd -> V-files + cell-mean")
        self.minsize(980, 450)

        self.sen_path = tk.StringVar(value="")
        self.out_dir = tk.StringVar(value="")
        self.cell_min = tk.StringVar(value="2")
        self.cell_max = tk.StringVar(value="3")
        self.datetime_col = tk.StringVar(value="DateTime")

        self.v_paths = []
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # SEN
        frm1 = tk.LabelFrame(self, text="1) SEN file (mm dd yyyy HH MM SS per rij)")
        frm1.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        frm1.columnconfigure(0, weight=1)

        ent_sen = tk.Entry(frm1, textvariable=self.sen_path)
        ent_sen.grid(row=0, column=0, sticky="ew", padx=(10, 8), pady=10)
        tk.Button(frm1, text="Kies SEN...", command=self.pick_sen, width=16)\
            .grid(row=0, column=1, sticky="e", padx=(0, 10), pady=10)
        ent_sen.bind("<Double-Button-1>", lambda _e: self.pick_sen())

        # V-files
        frm2 = tk.LabelFrame(self, text="2) V-files (*.v1/*.v2/*.v3/... whitespace)")
        frm2.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
        frm2.columnconfigure(0, weight=1)
        frm2.rowconfigure(1, weight=1)

        bar = tk.Frame(frm2)
        bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 6))
        tk.Button(bar, text="Kies 1+ V-files...", command=self.pick_vfiles).grid(row=0, column=0, sticky="w")
        tk.Button(bar, text="Clear lijst", command=self.clear_vfiles).grid(row=0, column=1, sticky="w", padx=8)

        self.listbox = tk.Listbox(frm2, height=10)
        self.listbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Opties
        frm3 = tk.LabelFrame(self, text="3) Opties")
        frm3.grid(row=2, column=0, sticky="ew", padx=10, pady=8)
        frm3.columnconfigure(6, weight=1)

        tk.Label(frm3, text="Cell# min").grid(row=0, column=0, sticky="w", padx=(10, 4), pady=8)
        tk.Entry(frm3, textvariable=self.cell_min, width=8).grid(row=0, column=1, sticky="w", padx=4, pady=8)

        tk.Label(frm3, text="Cell# max").grid(row=0, column=2, sticky="w", padx=(16, 4), pady=8)
        tk.Entry(frm3, textvariable=self.cell_max, width=8).grid(row=0, column=3, sticky="w", padx=4, pady=8)

        tk.Label(frm3, text="Datetime kolomnaam").grid(row=0, column=4, sticky="w", padx=(16, 4), pady=8)
        tk.Entry(frm3, textvariable=self.datetime_col, width=16).grid(row=0, column=5, sticky="w", padx=4, pady=8)

        tk.Label(frm3, text="Output folder (leeg = naast v-file)").grid(row=1, column=0, sticky="w", padx=(10, 4), pady=(0, 10))
        tk.Entry(frm3, textvariable=self.out_dir).grid(row=1, column=1, columnspan=5, sticky="ew", padx=4, pady=(0, 10))
        tk.Button(frm3, text="Kies folder...", command=self.pick_out_dir, width=16)\
            .grid(row=1, column=6, sticky="e", padx=(8, 10), pady=(0, 10))

        bottom = tk.Frame(self)
        bottom.grid(row=3, column=0, sticky="ew", padx=10, pady=(4, 10))
        bottom.columnconfigure(0, weight=1)
        tk.Button(bottom, text="RUN", command=self.run, height=2).grid(row=0, column=0, sticky="e")

    def pick_sen(self):
        p = filedialog.askopenfilename(
            title="Kies SEN file",
            filetypes=[("SEN files", "*.sen"), ("Text", "*.txt"), ("All files", "*.*")]
        )
        if p:
            self.sen_path.set(p)

    def pick_vfiles(self):
        ps = filedialog.askopenfilenames(
            title="Kies V-files",
            filetypes=[("V files", "*.v1;*.v2;*.v3;*.v4;*.v5;*.*"), ("All files", "*.*")]
        )
        if ps:
            self.v_paths = list(ps)
            self._refresh_listbox()

    def clear_vfiles(self):
        self.v_paths = []
        self._refresh_listbox()

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for p in self.v_paths:
            self.listbox.insert(tk.END, p)

    def pick_out_dir(self):
        d = filedialog.askdirectory(title="Kies output folder")
        if d:
            self.out_dir.set(d)

    def run(self):
        try:
            sen = self.sen_path.get().strip()
            if not sen or not os.path.exists(sen):
                messagebox.showerror("Fout", "Kies eerst een geldige SEN file.")
                return

            if not self.v_paths:
                messagebox.showerror("Fout", "Kies minstens 1 V-file.")
                return

            cell_min = int(self.cell_min.get())
            cell_max = int(self.cell_max.get())
            if cell_min <= 0 or cell_max <= 0 or cell_max < cell_min:
                messagebox.showerror("Fout", "Cell-range is ongeldig. Voorbeeld: min=2, max=3.")
                return

            dt_colname = self.datetime_col.get().strip() or "DateTime"
            out_dir = self.out_dir.get().strip()

            # 1) lees SEN tijdreeks (STRICT)
            sen_dt = read_sen_datetime_series(sen)

            outputs = []
            for vpath in self.v_paths:
                df_v = read_vfile(vpath)

                mean_series = compute_cell_mean(df_v, cell_min=cell_min, cell_max=cell_max)

                n = min(len(sen_dt), len(mean_series))
                if n <= 0:
                    raise ValueError("0 rijen na inlezen/aligneren.")

                out_df = pd.DataFrame({
                    dt_colname: sen_dt.iloc[:n].reset_index(drop=True),
                    f"Mean_Cell_{cell_min}_{cell_max}": mean_series.iloc[:n].reset_index(drop=True),
                })

                base = os.path.splitext(os.path.basename(vpath))[0]
                out_name = f"{base}_meanCells_{cell_min}_{cell_max}.csv"
                target_dir = out_dir if out_dir else os.path.dirname(vpath)
                out_path = os.path.join(target_dir, out_name)

                out_df.to_csv(out_path, index=False, encoding="utf-8")
                outputs.append(out_path)

            messagebox.showinfo(
                "Klaar",
                f"Succes.\nSEN rijen: {len(sen_dt)}\nV-files: {len(self.v_paths)}\n\nEerste output:\n{outputs[0]}"
            )

        except Exception as e:
            messagebox.showerror("Fout", f"{e}\n\n{traceback.format_exc()}")


if __name__ == "__main__":
    App().mainloop()
