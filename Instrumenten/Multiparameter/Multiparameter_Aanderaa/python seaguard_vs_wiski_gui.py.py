import os
import re
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# =========================
# Helpers: Seaguard lezen
# =========================

def _to_float_clean(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.strip().str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)  # decimale komma -> punt
    return pd.to_numeric(s, errors="coerce")

def read_seaguard_block_csv(path: str) -> pd.DataFrame:
    rows = []
    station = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                station = None
                continue
            parts = line.split(",")
            if len(parts) == 1 and parts[0] and "Date" not in parts[0]:
                station = parts[0].strip()
                continue
            if len(parts) >= 2 and parts[0].strip() == "Date" and parts[1].strip() == "Time":
                continue
            if station is not None and len(parts) >= 6:
                date_str = parts[0].strip()
                time_str = parts[1].strip()
                cond_str = parts[2].strip()
                temp_str = parts[5].strip()
                dt = pd.to_datetime(f"{date_str} {time_str}", dayfirst=True, errors="coerce")
                rows.append({
                    "station": station,
                    "datetime": dt,
                    "cond_uScm": None if cond_str == "" else cond_str,
                    "temp_C":   None if temp_str == "" else temp_str
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["cond_uScm"] = _to_float_clean(df["cond_uScm"])
    df["temp_C"]    = _to_float_clean(df["temp_C"])
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return df

def _parse_two_line_content(text: str, vpath: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{os.path.basename(vpath)}: minder dan 2 regels.")
    p1 = [p.strip() for p in lines[0].split("|")]
    p2 = [p.strip() for p in lines[1].split("|")]
    if len(p1) < 3 or len(p2) < 3:
        raise ValueError(f"{os.path.basename(vpath)}: onvoldoende kolommen.")
    station = p1[0]
    ts_raw = p1[1].split(".")[0]
    try:
        dt = pd.to_datetime(ts_raw, format="%Y-%m-%d %H:%M:%S", errors="raise")
    except Exception:
        base = os.path.basename(vpath)
        stamp = os.path.splitext(base)[0].split("_")[-1]
        dt = pd.to_datetime(stamp, format="%Y%m%d%H%M%S", errors="coerce")
    temp = _to_float_clean(pd.Series([p1[2]])).iloc[0]
    cond = _to_float_clean(pd.Series([p2[2]])).iloc[0]
    return station, dt, cond, temp

def _iter_txt_from_paths(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith(".txt"):
                        full = os.path.join(root, fn)
                        with open(full, "rb") as f:
                            yield full, f.read()
        elif os.path.isfile(p):
            if p.lower().endswith(".zip"):
                with zipfile.ZipFile(p, "r") as zf:
                    for zi in zf.infolist():
                        if not zi.is_dir() and zi.filename.lower().endswith(".txt"):
                            with zf.open(zi) as f:
                                yield zi.filename, f.read()
            elif p.lower().endswith(".txt"):
                with open(p, "rb") as f:
                    yield p, f.read()

def read_seaguard_two_line_from_paths(paths) -> pd.DataFrame:
    rows = []
    for vpath, content in _iter_txt_from_paths(paths):
        try:
            txt = content.decode("utf-8", errors="replace")
            station, dt, cond, temp = _parse_two_line_content(txt, vpath)
            rows.append({"station": station, "datetime": dt, "cond_uScm": cond, "temp_C": temp})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return df

# =========================
# WISKI lezen
# =========================

def read_wiski_csv_known_tab_format(path: str) -> pd.DataFrame:
    """
    SPECIFIEK: vanaf lijn 17 (skiprows=16), tab-delimited,
    kolom 1 = dd/mm/yyyy, kolom 2 = hh:mm:ss, waardekolom = derde of laatste.
    """
    # Probeer meest waarschijnlijke encodings
    for enc in ["utf-8", "cp1252", "latin-1", "utf-16"]:
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                header=None,         # geen header in de datasectie
                skiprows=16,         # vanaf lijn 17 start data
                engine="python",
                encoding=enc
            )
            if df.shape[1] < 2:
                continue
            # datetime construeren uit kolom 0 en 1
            dt = pd.to_datetime(df[0].astype(str).str.strip() + " " + df[1].astype(str).str.strip(),
                                dayfirst=True, errors="coerce")
            # kies waarde: bij voorkeur kolom 2, anders laatste kolom
            val_col_idx = 2 if df.shape[1] >= 3 else df.shape[1]-1
            val = _to_float_clean(df[val_col_idx])
            out = pd.DataFrame({"datetime": dt, "value": val}).dropna(subset=["datetime"]).sort_values("datetime")
            if not out.empty:
                return out
        except Exception:
            continue
    # als alles faalt, laat de aanroeper terugvallen op de generieke reader
    return pd.DataFrame(columns=["datetime", "value"])

def read_wiski_csv_generic(path: str) -> pd.DataFrame:
    """
    Generieke fallback: probeer diverse encodings & separators, zoek datetime- en value-kolom.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bestand niet gevonden: {path}")

    encodings = ["utf-8", "cp1252", "latin-1", "utf-16"]
    seps = [None, ";", ",", "\t"]

    df = None
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                tmp = pd.read_csv(path, sep=sep, engine="python", dtype=str, encoding=enc)
                if tmp.shape[0] == 0 or tmp.shape[1] == 0:
                    continue
                df = tmp
                break
            except Exception as e:
                last_err = e
                continue
        if df is not None:
            break
    if df is None:
        raise ValueError(f"WISKI-bestand kon niet gelezen worden: {path}\nLaatste fout: {last_err}")

    df.columns = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns]

    def find_datetime_col(_df):
        cands = [c for c in _df.columns if any(k in c for k in ["date", "datum", "time", "tijd"])]
        if cands:
            both = [c for c in cands if ("date" in c and "time" in c) or ("datum" in c and "tijd" in c)]
            return both[0] if both else cands[0]
        return None

    dt_col = find_datetime_col(df)
    if dt_col is None:
        if df.shape[1] >= 2:
            dt_series = (df.iloc[:, 0].astype(str).str.strip() + " " +
                         df.iloc[:, 1].astype(str).str.strip())
        else:
            dt_series = df.iloc[:, 0].astype(str)
        dt = pd.to_datetime(dt_series, dayfirst=True, errors="coerce")
    else:
        dt = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")

    def to_float_series(s):
        s = s.astype(str).str.replace(" ", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    candidate_cols = [c for c in df.columns if c != dt_col]
    value = None
    for c in reversed(candidate_cols):
        vals = to_float_series(df[c])
        if vals.notna().sum() >= max(3, int(0.3 * len(vals))):
            value = vals
            break
    if value is None:
        value = to_float_series(df.iloc[:, -1])

    out = pd.DataFrame({"datetime": dt, "value": value})
    out = out.dropna(subset=["datetime"]).sort_values("datetime")
    if out.empty:
        raise ValueError(f"Geen bruikbare datetime/waarden in: {os.path.basename(path)}")
    return out

def read_wiski_csv(path: str) -> pd.DataFrame:
    """
    Eerst het **specifieke** WISKI-formaat (tab, skiprows=16, kolom1+2 = datum+tijd).
    Als dat leeg/faalt, val terug op de generieke reader.
    """
    df = read_wiski_csv_known_tab_format(path)
    if df.empty:
        df = read_wiski_csv_generic(path)
    return df

# =========================
# Analyse / alignment
# =========================

def compute_stats(df: pd.DataFrame, a: str, b: str):
    sub = df[[a, b]].dropna()
    n = len(sub)
    if n == 0:
        return {"N": 0, "Bias": np.nan, "RMSE": np.nan, "MAE": np.nan, "R": np.nan}
    diff = sub[a] - sub[b]
    return {
        "N": int(n),
        "Bias": float(diff.mean()),
        "RMSE": float(np.sqrt((diff**2).mean())),
        "MAE": float(np.abs(diff).mean()),
        "R": float(sub[a].corr(sub[b]))
    }

def merge_asof_left(left: pd.DataFrame, right: pd.DataFrame, right_value_name: str,
                    tolerance: pd.Timedelta) -> pd.DataFrame:
    l = left.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    r = right.drop_duplicates(subset=["datetime"]).sort_values("datetime").rename(columns={"value": right_value_name})
    m = pd.merge_asof(l, r, on="datetime", direction="nearest", tolerance=tolerance)
    return m

# =========================
# GUI
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Seaguard vs WISKI – Tijdlijnvergelijking")
        self.geometry("920x560")

        # Seaguard bron
        self.sg_mode = tk.StringVar(value="block")
        self.sg_block_path = tk.StringVar()
        self.sg_raw_paths = []
        self.sg_shift_minutes = tk.IntVar(value=0)

        # WISKI
        self.path_wiski_cond = tk.StringVar()
        self.path_wiski_temp = tk.StringVar()

        # Matching
        self.tolerance_seconds = tk.IntVar(value=180)

        # Uitvoer
        self.out_dir = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        r = 0
        ttk.Label(frm, text="Seaguard bron:").grid(row=r, column=0, sticky="w")
        ttk.Radiobutton(frm, text="Blok-CSV", variable=self.sg_mode, value="block").grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(frm, text="Ruwe TXT/ZIP", variable=self.sg_mode, value="raw").grid(row=r, column=2, sticky="w"); r+=1

        ttk.Label(frm, text="Seaguard blok-CSV:").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.sg_block_path, width=70).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="Bladeren…", command=self.pick_seaguard_block).grid(row=r, column=2, sticky="e"); r+=1

        ttk.Label(frm, text="Seaguard ruwe TXT/ZIP (map of bestanden):").grid(row=r, column=0, sticky="w")
        self.lst_raw = tk.Listbox(frm, height=4)
        self.lst_raw.grid(row=r, column=1, sticky="nsew", padx=(6,6))
        ttk.Button(frm, text="Map kiezen…", command=self.pick_seaguard_dir).grid(row=r, column=2, sticky="e"); r+=1
        ttk.Button(frm, text="Bestanden kiezen…", command=self.pick_seaguard_files).grid(row=r, column=2, sticky="e"); r+=1

        ttk.Label(frm, text="Tijdshift Seaguard (minuten, + = vooruit):").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.sg_shift_minutes, width=10).grid(row=r, column=1, sticky="w"); r+=1

        ttk.Label(frm, text="WISKI Conductiviteit CSV/TXT:").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.path_wiski_cond, width=70).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="Bladeren…", command=self.pick_wiski_cond).grid(row=r, column=2, sticky="e"); r+=1

        ttk.Label(frm, text="WISKI Temperatuur CSV/TXT:").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.path_wiski_temp, width=70).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="Bladeren…", command=self.pick_wiski_temp).grid(row=r, column=2, sticky="e"); r+=1

        ttk.Label(frm, text="Match-tolerantie (seconden):").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.tolerance_seconds, width=10).grid(row=r, column=1, sticky="w"); r+=1

        ttk.Label(frm, text="Uitvoer-map:").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_dir, width=70).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="Kies map…", command=self.pick_outdir).grid(row=r, column=2, sticky="e"); r+=1

        ttk.Button(frm, text="Vergelijk & Exporteer", command=self.run).grid(row=r, column=2, sticky="e", pady=10); r+=1

        ttk.Label(frm, text="Log:").grid(row=r, column=0, sticky="w")
        self.txt = tk.Text(frm, height=10)
        self.txt.grid(row=r, column=1, columnspan=2, sticky="nsew")
        frm.rowconfigure(r, weight=1)
        frm.columnconfigure(1, weight=1)

    # --- UI handlers ---
    def log(self, s: str):
        self.txt.insert(tk.END, s + "\n"); self.txt.see(tk.END); self.update_idletasks()

    def pick_seaguard_block(self):
        p = filedialog.askopenfilename(title="Kies Seaguard blok-CSV", filetypes=[("CSV/TXT", "*.csv *.txt"), ("Alle", "*.*")])
        if p: self.sg_block_path.set(p)

    def pick_seaguard_dir(self):
        d = filedialog.askdirectory(title="Kies directory met Seaguard TXT of ZIP")
        if d:
            self.sg_raw_paths = [d]; self.refresh_raw_list()

    def pick_seaguard_files(self):
        files = filedialog.askopenfilenames(title="Kies Seaguard TXT/ZIP bestanden", filetypes=[("TXT/ZIP", "*.txt *.zip"), ("Alle", "*.*")])
        if files:
            self.sg_raw_paths = list(files); self.refresh_raw_list()

    def refresh_raw_list(self):
        self.lst_raw.delete(0, tk.END)
        for p in self.sg_raw_paths:
            self.lst_raw.insert(tk.END, p)

    def pick_wiski_cond(self):
        p = filedialog.askopenfilename(title="Kies WISKI Conductiviteit CSV/TXT", filetypes=[("CSV/TXT", "*.csv *.txt"), ("Alle", "*.*")])
        if p: self.path_wiski_cond.set(p)

    def pick_wiski_temp(self):
        p = filedialog.askopenfilename(title="Kies WISKI Temperatuur CSV/TXT", filetypes=[("CSV/TXT", "*.csv *.txt"), ("Alle", "*.*")])
        if p: self.path_wiski_temp.set(p)

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Kies uitvoermap")
        if d: self.out_dir.set(d)

    # --- Kern ---
    def run(self):
        try:
            if not self.out_dir.get():
                self.out_dir.set(os.getcwd())

            # Seaguard
            self.log("Lezen van Seaguard…")
            if self.sg_mode.get() == "block":
                if not self.sg_block_path.get():
                    raise ValueError("Kies een Seaguard blok-CSV of schakel naar 'Ruwe TXT/ZIP'.")
                df_sg = read_seaguard_block_csv(self.sg_block_path.get())
            else:
                if not self.sg_raw_paths:
                    raise ValueError("Selecteer map/bestanden voor Seaguard ruwe TXT/ZIP.")
                df_sg = read_seaguard_two_line_from_paths(self.sg_raw_paths)
            if df_sg.empty:
                raise ValueError("Seaguard: geen data gelezen.")
            shift = int(self.sg_shift_minutes.get())
            if shift != 0:
                df_sg["datetime"] = df_sg["datetime"] + pd.to_datetime(shift, unit="m")
                self.log(f"Tijdshift toegepast op Seaguard: {shift} min")

            # WISKI
            self.log("Lezen van WISKI conductiviteit…")
            df_wk_cond = read_wiski_csv(self.path_wiski_cond.get())
            if df_wk_cond.empty:
                raise ValueError("WISKI Conductiviteit: geen bruikbare rijen (controleer export).")

            self.log("Lezen van WISKI temperatuur…")
            df_wk_temp = read_wiski_csv(self.path_wiski_temp.get())
            if df_wk_temp.empty:
                raise ValueError("WISKI Temperatuur: geen bruikbare rijen (controleer export).")

            tol = pd.Timedelta(seconds=int(self.tolerance_seconds.get()))

            # Align
            self.log("Matchen (merge_asof) van conductiviteit…")
            df_cond = merge_asof_left(df_sg[["datetime","cond_uScm","station"]], df_wk_cond, "wiski_cond_uScm", tolerance=tol)
            self.log("Matchen (merge_asof) van temperatuur…")
            df_temp = merge_asof_left(df_sg[["datetime","temp_C","station"]], df_wk_temp, "wiski_temp_C", tolerance=tol)

            # Stats
            stats_cond = compute_stats(df_cond, "cond_uScm", "wiski_cond_uScm")
            stats_temp = compute_stats(df_temp, "temp_C", "wiski_temp_C")

            # Export
            out_cond_csv = os.path.join(self.out_dir.get(), "seaguard_vs_wiski_conductivity.csv")
            out_temp_csv = os.path.join(self.out_dir.get(), "seaguard_vs_wiski_temperature.csv")
            df_cond.to_csv(out_cond_csv, index=False)
            df_temp.to_csv(out_temp_csv, index=False)

            plt.figure(figsize=(10,4))
            plt.plot(df_cond["datetime"], df_cond["cond_uScm"], label="Seaguard cond [µS/cm]")
            plt.plot(df_cond["datetime"], df_cond["wiski_cond_uScm"], label="WISKI cond [µS/cm]")
            plt.xlabel("Time"); plt.ylabel("Conductivity [µS/cm]"); plt.title("Conductivity timeline: Seaguard vs WISKI"); plt.legend()
            cond_png = os.path.join(self.out_dir.get(), "seaguard_vs_wiski_conductivity.png")
            plt.tight_layout(); plt.savefig(cond_png); plt.close()

            plt.figure(figsize=(10,4))
            plt.plot(df_temp["datetime"], df_temp["temp_C"], label="Seaguard temp [°C]")
            plt.plot(df_temp["datetime"], df_temp["wiski_temp_C"], label="WISKI temp [°C]")
            plt.xlabel("Time"); plt.ylabel("Temperature [°C]"); plt.title("Temperature timeline: Seaguard vs WISKI"); plt.legend()
            temp_png = os.path.join(self.out_dir.get(), "seaguard_vs_wiski_temperature.png")
            plt.tight_layout(); plt.savefig(temp_png); plt.close()

            stats_df = pd.DataFrame([
                {"Parameter":"Conductivity [µS/cm]", **stats_cond},
                {"Parameter":"Temperature [°C]",     **stats_temp},
            ])
            stats_csv = os.path.join(self.out_dir.get(), "seaguard_vs_wiski_stats.csv")
            stats_df.to_csv(stats_csv, index=False)

            self.log("Klaar.")
            self.log(f"CSV:  {out_cond_csv}")
            self.log(f"CSV:  {out_temp_csv}")
            self.log(f"PNG:  {cond_png}")
            self.log(f"PNG:  {temp_png}")
            self.log(f"Stats:{stats_csv}")
            messagebox.showinfo("Succes", "Vergelijking afgerond. Bestanden zijn weggeschreven.")

        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self.log(f"Fout: {e}")

# -----------------------
if __name__ == "__main__":
    App().mainloop()
