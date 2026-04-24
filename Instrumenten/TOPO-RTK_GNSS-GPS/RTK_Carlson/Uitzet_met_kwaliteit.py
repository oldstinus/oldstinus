#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from pyproj import Transformer
from tkinter import filedialog, messagebox, ttk

try:
    import folium
except Exception:
    folium = None

NONE = "(none)"

DEFAULT_SIGMA_XY_A = 0.005
DEFAULT_SIGMA_Z_A = 0.010
DEFAULT_SIGMA_XY_B = 0.010
DEFAULT_SIGMA_Z_B = 0.020
DEFAULT_OUTLIER_SIGMA = 2.0


@dataclass
class Guess:
    name: Optional[str] = None
    group: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    z: Optional[str] = None
    sigma_xy: Optional[str] = None
    sigma_z: Optional[str] = None
    time: Optional[str] = None
    date: Optional[str] = None
    profile: str = "Onbekend"
    note: str = ""


@dataclass
class ProcessOptions:
    output_dir: Path
    crs: str
    name_col: Optional[str]
    group_col: Optional[str]
    x_col: str
    y_col: str
    z_col: str
    sigma_xy_col: str
    sigma_z_col: str
    time_col: Optional[str]
    date_col: Optional[str]
    sigma_xy_a: float
    sigma_z_a: float
    sigma_xy_b: float
    sigma_z_b: float
    outlier_sigma: float
    worst_allowed_quality: str
    max_sigma_xy: Optional[float]
    max_sigma_z: Optional[float]
    apply_outlier_filter: bool


def s(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() == "nan" else text


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def is_num(value: object) -> bool:
    try:
        float(s(value).replace(",", "."))
        return bool(s(value))
    except ValueError:
        return False


def sniff_delim(path: str | Path, default: str = ";") -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()][:50]
    candidates = ["\t", ";", "|", ","]
    best_delim: Optional[str] = None
    best_score: tuple[float, float, float] | None = None

    # Kies eerst de delimiter die over meerdere regels het meest consistent is.
    for delim in candidates:
        counts = [line.count(delim) for line in lines]
        positive = [count for count in counts if count > 0]
        if len(positive) < 2:
            continue
        mean_count = float(np.mean(positive))
        std_count = float(np.std(positive))
        consistency = len(positive) / max(1, len(lines))
        score = (consistency, mean_count, -std_count)
        if best_score is None or score > best_score:
            best_delim, best_score = delim, score

    if best_delim is not None:
        return best_delim

    sample = text[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        return default


def detect_header(path: str | Path, delim: str) -> bool:
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        for row in csv.reader(fh, delimiter=delim):
            if row and any(str(x).strip() for x in row):
                rows.append([str(x).strip() for x in row])
            if len(rows) >= 3:
                break
    if len(rows) < 2:
        return True
    hdr = {
        "pt id", "point id", "id", "name", "desc", "description", "setup",
        "x", "y", "z", "e", "n", "sigmaxy", "sigmaz", "time", "date", "datum",
    }
    if {x.lower() for x in rows[0]} & hdr:
        return True
    a = sum(is_num(x) for x in rows[0]) / max(1, len(rows[0]))
    b = sum(is_num(x) for x in rows[1]) / max(1, len(rows[1]))
    return a < 0.35 and b >= 0.35


def read_table(path: str | Path, delim: str, header_mode: str) -> tuple[pd.DataFrame, bool, str]:
    if delim == "auto":
        delim = sniff_delim(path)
    has_header = detect_header(path, delim) if header_mode == "auto" else header_mode == "yes"
    df = pd.read_csv(
        path,
        sep=delim,
        header=0 if has_header else None,
        dtype=str,
        keep_default_na=False,
        engine="python",
    )
    if has_header:
        df.columns = [s(col) or f"col{i+1}" for i, col in enumerate(df.columns)]
    else:
        df.columns = [f"col{i+1}" for i in range(df.shape[1])]
    for col in df.columns:
        df[col] = df[col].map(s)
    return df, has_header, delim


def guess_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def num_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if float(numeric(df[col]).notna().mean()) >= 0.75]


def guess_cols(df: pd.DataFrame, has_header: bool) -> Guess:
    g = Guess(
        name=guess_col(df, ["pt id", "point id", "id", "name"]),
        group=guess_col(df, ["desc", "description", "omschrijving", "groep", "setup"]),
        x=guess_col(df, ["x", "e", "east", "easting"]),
        y=guess_col(df, ["y", "n", "north", "northing"]),
        z=guess_col(df, ["z", "h", "height", "hoogte"]),
        sigma_xy=guess_col(df, ["sigmaxy", "sigma xy", "sigmaxy", "hz precision", "precisionxy", "rmsxy"]),
        sigma_z=guess_col(df, ["sigmaz", "sigma z", "sigmah", "vertical precision", "precisionz", "rmsz"]),
        time=guess_col(df, ["time", "tijd", "timestamp"]),
        date=guess_col(df, ["date", "datum"]),
        profile="Header-profiel",
    )
    if g.x or g.y or g.name:
        return g

    cols = list(df.columns)
    nums = num_cols(df)
    if not has_header and len(cols) >= 9:
        return Guess(
            name=cols[0],
            group=cols[1],
            x=cols[2],
            y=cols[3],
            z=cols[4],
            sigma_xy=cols[5],
            sigma_z=cols[6],
            time=cols[7],
            date=cols[8],
            profile="Carlson TXT zonder header",
            note="Kolomvolgorde 1-9 als naam, groep, X, Y, Z, SigmaXY, SigmaZ, tijd, datum.",
        )
    if len(nums) >= 5:
        return Guess(
            name=cols[0] if cols else None,
            group=cols[1] if len(cols) > 1 else None,
            x=nums[0],
            y=nums[1],
            z=nums[2],
            sigma_xy=nums[3],
            sigma_z=nums[4],
            profile="Generieke detectie",
            note="Eerste vijf numerieke kolommen gekozen als X, Y, Z, SigmaXY en SigmaZ.",
        )
    return g


def quality_flag(sigma_xy: float, sigma_z: float, opts: ProcessOptions) -> str:
    if pd.isna(sigma_xy) or pd.isna(sigma_z):
        return "?"
    if sigma_xy <= opts.sigma_xy_a and sigma_z <= opts.sigma_z_a:
        return "A"
    if sigma_xy <= opts.sigma_xy_b and sigma_z <= opts.sigma_z_b:
        return "B"
    return "C"


def parse_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    stamp = (date_series.map(s) + " " + time_series.map(s)).str.strip()
    stamp = stamp.mask(stamp.eq(""))
    year_first_mask = stamp.fillna("").str.match(r"^\d{4}[/\-]")
    if bool(year_first_mask.any()):
        dt = pd.to_datetime(stamp, errors="coerce", yearfirst=True)
    else:
        dt = pd.to_datetime(stamp, errors="coerce", dayfirst=True)
    if dt.notna().any():
        return dt
    return pd.to_datetime(stamp, errors="coerce", yearfirst=True)


def derive_base_name(name_series: pd.Series, group_series: pd.Series) -> pd.Series:
    group = group_series.map(s)
    name = name_series.map(s)
    base = group.where(group.ne(""), name)
    base = base.str.replace(r"\d+$", "", regex=True).str.strip()
    return base.where(base.ne(""), "Onbekend punt")


def add_reason(series: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    out = series.astype(str).copy()
    out.loc[mask] = np.where(out.loc[mask].eq(""), reason, out.loc[mask] + "; " + reason)
    return out


def apply_filters(df: pd.DataFrame, opts: ProcessOptions) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = {"A": 0, "B": 1, "C": 2}
    raw = df.copy()
    raw["FilterReason"] = ""

    if opts.max_sigma_xy is not None:
        mask = raw["SigmaXY"] > opts.max_sigma_xy
        raw["FilterReason"] = add_reason(raw["FilterReason"], mask, f"SigmaXY>{opts.max_sigma_xy:.4f}")

    if opts.max_sigma_z is not None:
        mask = raw["SigmaZ"] > opts.max_sigma_z
        raw["FilterReason"] = add_reason(raw["FilterReason"], mask, f"SigmaZ>{opts.max_sigma_z:.4f}")

    allowed_rank = ranks.get(opts.worst_allowed_quality, 2)
    quality_rank = raw["Quality"].map(ranks).fillna(99)
    mask = quality_rank > allowed_rank
    raw["FilterReason"] = add_reason(raw["FilterReason"], mask, f"Kwaliteit>{opts.worst_allowed_quality}")
    raw["PassInitialFilter"] = raw["FilterReason"].eq("")

    filtered = raw.loc[raw["PassInitialFilter"]].copy()
    filtered["PassOutlierFilter"] = True
    filtered["OutlierReason"] = ""

    if not opts.apply_outlier_filter or filtered.empty:
        return raw, filtered

    keep_index: list[int] = []
    for _, group in filtered.groupby("BaseName"):
        z_std = group["Z"].std()
        if pd.isna(z_std) or z_std == 0:
            keep_index.extend(group.index.tolist())
            continue
        z_mean = group["Z"].mean()
        ok_mask = (group["Z"] - z_mean).abs() <= opts.outlier_sigma * z_std
        keep_index.extend(group.index[ok_mask].tolist())
        bad_index = group.index[~ok_mask]
        filtered.loc[bad_index, "PassOutlierFilter"] = False
        filtered.loc[bad_index, "OutlierReason"] = f"|Z-mean|>{opts.outlier_sigma:.2f}*std"

    filtered = filtered.loc[filtered.index.isin(keep_index)].copy()
    return raw, filtered


def summarise_quality(series: pd.Series) -> str:
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else "?"


def make_quality_plot(df: pd.DataFrame, out_dir: Path, opts: ProcessOptions) -> None:
    colors = df["Quality"].map({"A": "#1a9850", "B": "#fee08b", "C": "#d73027", "?": "#7f7f7f"}).fillna("#7f7f7f")
    plt.figure(figsize=(7, 5))
    plt.scatter(df["SigmaXY"], df["SigmaZ"], c=colors, s=28, edgecolors="none")
    plt.axvline(opts.sigma_xy_a, color="#1a9850", linestyle="--", linewidth=1)
    plt.axhline(opts.sigma_z_a, color="#1a9850", linestyle="--", linewidth=1)
    plt.axvline(opts.sigma_xy_b, color="#d95f02", linestyle="--", linewidth=1)
    plt.axhline(opts.sigma_z_b, color="#d95f02", linestyle="--", linewidth=1)
    if opts.max_sigma_xy is not None:
        plt.axvline(opts.max_sigma_xy, color="#2c7fb8", linestyle=":", linewidth=1)
    if opts.max_sigma_z is not None:
        plt.axhline(opts.max_sigma_z, color="#2c7fb8", linestyle=":", linewidth=1)
    plt.xlabel("Sigma XY (m)")
    plt.ylabel("Sigma Z (m)")
    plt.title("RTK kwaliteit")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "quality.png", dpi=160)
    plt.close()


def make_z_stability_plot(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    for base_name, group in df.groupby("BaseName"):
        g = group.sort_values(["Datetime", "PointName"]).copy()
        if g["Datetime"].notna().any():
            x = g["Datetime"]
            xlabel = "Tijd"
        else:
            x = np.arange(1, len(g) + 1)
            xlabel = "Meting"
        plt.plot(x, g["Z"], marker="o", linewidth=1.2, label=base_name)
    plt.xlabel(xlabel)
    plt.ylabel("Z (m)")
    plt.title("Z stabiliteit")
    plt.grid(alpha=0.25)
    plt.xticks(rotation=45)
    if df["BaseName"].nunique() <= 12:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "z_stability.png", dpi=160)
    plt.close()


def make_sigma_plot(agg: pd.DataFrame, out_dir: Path) -> None:
    if agg.empty:
        return
    plt.figure(figsize=(10, 5))
    agg_sorted = agg.sort_values("SigmaZ")
    plt.plot(agg_sorted["BaseName"], agg_sorted["SigmaZ"], marker="o")
    plt.xticks(rotation=90)
    plt.ylabel("Sigma Z (m)")
    plt.title("Sigma Z per punt")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "sigmaZ.png", dpi=160)
    plt.close()


def map_color(quality: str) -> str:
    return {"A": "#1a9850", "B": "#fdae61", "C": "#d73027", "?": "#7f7f7f"}.get(quality, "#7f7f7f")


DEFAULT_MAP_TILES = "CartoDB positron"


def make_points_map(agg: pd.DataFrame, out_html: Path) -> bool:
    if folium is None or agg.empty:
        return False

    center = [float(agg["Latitude"].median()), float(agg["Longitude"].median())]
    # Avoid direct OpenStreetMap tile requests from local file:// HTML, which OSM blocks without a valid Referer.
    m = folium.Map(location=center, zoom_start=16, tiles=DEFAULT_MAP_TILES)

    for _, row in agg.sort_values("BaseName").iterrows():
        popup = "<br>".join(
            [
                f"<b>Punt:</b> {html.escape(str(row['BaseName']))}",
                f"<b>Kwaliteit:</b> {html.escape(str(row['Quality']))}",
                f"<b>Aantal metingen:</b> {int(row['N'])}",
                f"<b>X:</b> {float(row['X']):.4f}",
                f"<b>Y:</b> {float(row['Y']):.4f}",
                f"<b>Z:</b> {float(row['Z']):.4f}",
                f"<b>SigmaXY:</b> {float(row['SigmaXY']):.4f}",
                f"<b>SigmaZ:</b> {float(row['SigmaZ']):.4f}",
                f"<b>Lat:</b> {float(row['Latitude']):.8f}",
                f"<b>Lon:</b> {float(row['Longitude']):.8f}",
            ]
        )
        folium.CircleMarker(
            location=[float(row["Latitude"]), float(row["Longitude"])],
            radius=7,
            color=map_color(str(row["Quality"])),
            fill=True,
            fill_color=map_color(str(row["Quality"])),
            fill_opacity=0.9,
            tooltip=f"{row['BaseName']} | kwaliteit {row['Quality']}",
            popup=folium.Popup(popup, max_width=420),
        ).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return True


def process_dataframe(df_in: pd.DataFrame, opts: ProcessOptions) -> dict[str, object]:
    df = df_in.copy()
    df["PointName"] = df[opts.name_col].map(s) if opts.name_col else ""
    df["GroupName"] = df[opts.group_col].map(s) if opts.group_col else ""
    df["X"] = numeric(df[opts.x_col])
    df["Y"] = numeric(df[opts.y_col])
    df["Z"] = numeric(df[opts.z_col])
    df["SigmaXY"] = numeric(df[opts.sigma_xy_col])
    df["SigmaZ"] = numeric(df[opts.sigma_z_col])
    df["Time"] = df[opts.time_col].map(s) if opts.time_col else ""
    df["Date"] = df[opts.date_col].map(s) if opts.date_col else ""
    df["Datetime"] = parse_datetime(df["Date"], df["Time"])
    df["BaseName"] = derive_base_name(df["PointName"], df["GroupName"])

    needed = ["X", "Y", "Z", "SigmaXY", "SigmaZ"]
    df = df.dropna(subset=needed).copy()
    if df.empty:
        raise ValueError("Geen geldige X, Y, Z, SigmaXY en SigmaZ waarden gevonden.")

    df["Quality"] = [quality_flag(xy, z, opts) for xy, z in zip(df["SigmaXY"], df["SigmaZ"])]

    raw, filtered = apply_filters(df, opts)
    if filtered.empty:
        raise ValueError("Na filtering blijven geen punten over. Versoepel de filterinstellingen.")

    agg = (
        filtered.groupby("BaseName", as_index=False)
        .agg(
            X=("X", "mean"),
            Y=("Y", "mean"),
            Z=("Z", "mean"),
            SigmaXY=("SigmaXY", "mean"),
            SigmaZ=("SigmaZ", "mean"),
            N=("BaseName", "size"),
            Quality=("Quality", summarise_quality),
            FirstTime=("Datetime", "min"),
            LastTime=("Datetime", "max"),
        )
    )
    agg["SigmaZ_reduced"] = agg["SigmaZ"] / np.sqrt(agg["N"].clip(lower=1))

    if opts.crs == "EPSG:4326":
        agg["Longitude"] = agg["X"]
        agg["Latitude"] = agg["Y"]
    else:
        tr = Transformer.from_crs(opts.crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(agg["X"].to_numpy(dtype=float), agg["Y"].to_numpy(dtype=float))
        agg["Longitude"] = lon
        agg["Latitude"] = lat

    opts.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(opts.output_dir / "raw_with_quality.csv", sep=";", decimal=",", index=False)
    filtered.to_csv(opts.output_dir / "filtered.csv", sep=";", decimal=",", index=False)
    agg.to_csv(opts.output_dir / "aggregated_with_WGS84.csv", sep=";", decimal=",", index=False)

    make_quality_plot(raw, opts.output_dir, opts)
    make_z_stability_plot(filtered, opts.output_dir)
    make_sigma_plot(agg, opts.output_dir)
    map_created = make_points_map(agg, opts.output_dir / "puntenkaart.html")

    quality_counts = raw["Quality"].value_counts().sort_index()
    summary_lines = [
        f"Input rijen: {len(df_in)}",
        f"Geldige meetrijen: {len(raw)}",
        f"Na filtering: {len(filtered)}",
        f"Geaggregeerde punten: {len(agg)}",
        f"Kwaliteitsverdeling: {quality_counts.to_dict()}",
        f"Initiale filter OK: {int(raw['PassInitialFilter'].sum())}",
        f"Outlier filter toegepast: {'ja' if opts.apply_outlier_filter else 'nee'}",
        f"CRS naar WGS84: {opts.crs} -> EPSG:4326",
        f"Kaart aangemaakt: {'ja' if map_created else 'nee'}",
    ]
    (opts.output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "raw": raw,
        "filtered": filtered,
        "agg": agg,
        "summary_lines": summary_lines,
    }


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RTK kwaliteitsanalyse")
        self.geometry("1220x860")
        self.df: Optional[pd.DataFrame] = None
        self.v_in = tk.StringVar()
        self.v_out = tk.StringVar()
        self.v_delim = tk.StringVar(value="auto")
        self.v_header = tk.StringVar(value="auto")
        self.v_crs = tk.StringVar(value="EPSG:31370")

        self.v_name = tk.StringVar(value=NONE)
        self.v_group = tk.StringVar(value=NONE)
        self.v_x = tk.StringVar()
        self.v_y = tk.StringVar()
        self.v_z = tk.StringVar()
        self.v_sigma_xy = tk.StringVar()
        self.v_sigma_z = tk.StringVar()
        self.v_time = tk.StringVar(value=NONE)
        self.v_date = tk.StringVar(value=NONE)

        self.v_sigma_xy_a = tk.StringVar(value=f"{DEFAULT_SIGMA_XY_A:.4f}")
        self.v_sigma_z_a = tk.StringVar(value=f"{DEFAULT_SIGMA_Z_A:.4f}")
        self.v_sigma_xy_b = tk.StringVar(value=f"{DEFAULT_SIGMA_XY_B:.4f}")
        self.v_sigma_z_b = tk.StringVar(value=f"{DEFAULT_SIGMA_Z_B:.4f}")
        self.v_outlier_sigma = tk.StringVar(value=f"{DEFAULT_OUTLIER_SIGMA:.2f}")
        self.v_worst_quality = tk.StringVar(value="C")
        self.v_max_sigma_xy = tk.StringVar(value="")
        self.v_max_sigma_z = tk.StringVar(value="")
        self.v_apply_outlier = tk.BooleanVar(value=True)

        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Inputbestand").grid(row=0, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.v_in, width=110).grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(root, text="Bladeren...", command=self._pick_in).grid(row=1, column=1)

        ttk.Label(root, text="Outputmap").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(root, textvariable=self.v_out, width=110).grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(root, text="Kies...", command=self._pick_out).grid(row=3, column=1)

        read_box = ttk.LabelFrame(root, text="Inlezen", padding=10)
        read_box.grid(row=4, column=0, columnspan=2, sticky="we", pady=(12, 0))
        ttk.Label(read_box, text="Delimiter:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(read_box, textvariable=self.v_delim, values=["auto", ";", ",", "\\t", "|"], width=8, state="readonly").grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(read_box, text="Header:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        ttk.Combobox(read_box, textvariable=self.v_header, values=["auto", "yes", "no"], width=8, state="readonly").grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(read_box, text="Input CRS:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(read_box, textvariable=self.v_crs, values=["EPSG:31370", "EPSG:25831", "EPSG:32631", "EPSG:4326"], width=18, state="readonly").grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Button(read_box, text="Lees en detecteer kolommen", command=self._load).grid(row=2, column=0, sticky="w", pady=(10, 0))

        cols = ttk.LabelFrame(root, text="Kolommen", padding=10)
        cols.grid(row=5, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(cols, text="Puntnaam:").grid(row=0, column=0, sticky="w")
        ttk.Label(cols, text="Groep / setup:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Label(cols, text="X / Easting:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(cols, text="Y / Northing:").grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Label(cols, text="Z / Hoogte:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(cols, text="SigmaXY:").grid(row=2, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Label(cols, text="SigmaZ:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Label(cols, text="Tijd:").grid(row=3, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Label(cols, text="Datum:").grid(row=4, column=0, sticky="w", pady=(8, 0))

        self.cb_name = ttk.Combobox(cols, textvariable=self.v_name, values=[NONE], width=28, state="readonly")
        self.cb_group = ttk.Combobox(cols, textvariable=self.v_group, values=[NONE], width=28, state="readonly")
        self.cb_x = ttk.Combobox(cols, textvariable=self.v_x, values=[], width=28, state="readonly")
        self.cb_y = ttk.Combobox(cols, textvariable=self.v_y, values=[], width=28, state="readonly")
        self.cb_z = ttk.Combobox(cols, textvariable=self.v_z, values=[], width=28, state="readonly")
        self.cb_sigma_xy = ttk.Combobox(cols, textvariable=self.v_sigma_xy, values=[], width=28, state="readonly")
        self.cb_sigma_z = ttk.Combobox(cols, textvariable=self.v_sigma_z, values=[], width=28, state="readonly")
        self.cb_time = ttk.Combobox(cols, textvariable=self.v_time, values=[NONE], width=28, state="readonly")
        self.cb_date = ttk.Combobox(cols, textvariable=self.v_date, values=[NONE], width=28, state="readonly")

        self.cb_name.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.cb_group.grid(row=0, column=3, sticky="w", padx=(6, 0))
        self.cb_x.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_y.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_z.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_sigma_xy.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_sigma_z.grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_time.grid(row=3, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_date.grid(row=4, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        quality = ttk.LabelFrame(root, text="Kwaliteitsparameters en filtering", padding=10)
        quality.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(quality, text="A: max SigmaXY").grid(row=0, column=0, sticky="w")
        ttk.Entry(quality, textvariable=self.v_sigma_xy_a, width=10).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(quality, text="A: max SigmaZ").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Entry(quality, textvariable=self.v_sigma_z_a, width=10).grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(quality, text="B: max SigmaXY").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_sigma_xy_b, width=10).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="B: max SigmaZ").grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_sigma_z_b, width=10).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="Slechtste toegestane klasse").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(quality, textvariable=self.v_worst_quality, values=["A", "B", "C"], width=8, state="readonly").grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="Max SigmaXY filter").grid(row=2, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_max_sigma_xy, width=10).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="Max SigmaZ filter").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_max_sigma_z, width=10).grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(quality, text="Outlier sigma op Z").grid(row=3, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Entry(quality, textvariable=self.v_outlier_sigma, width=10).grid(row=3, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Checkbutton(quality, text="Outlierfilter op Z toepassen", variable=self.v_apply_outlier).grid(row=4, column=0, sticky="w", pady=(10, 0))

        actions = ttk.Frame(root)
        actions.grid(row=7, column=0, columnspan=2, sticky="we", pady=(12, 0))
        ttk.Button(actions, text="Verwerk bestand", command=self._process).pack(side="left")
        ttk.Button(actions, text="Sluiten", command=self.destroy).pack(side="right")

        self.txt = tk.Text(root, height=18, wrap="word")
        self.txt.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(8, weight=1)

    def _log(self, msg: str) -> None:
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _pick_in(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecteer Carlson CSV/TXT",
            filetypes=[("Text/CSV", "*.txt *.csv *.dat *.log"), ("All files", "*.*")],
        )
        if path:
            self.v_in.set(path)
            if not self.v_out.get():
                default_out = Path(path).with_suffix("").parent / f"{Path(path).stem}_RTK_output"
                self.v_out.set(str(default_out))

    def _pick_out(self) -> None:
        path = filedialog.askdirectory(title="Kies outputmap")
        if path:
            self.v_out.set(path)

    def _sel(self, value: str) -> Optional[str]:
        value = value.strip()
        return None if not value or value == NONE else value

    def _load(self) -> None:
        inp = self.v_in.get().strip()
        if not inp:
            messagebox.showerror("Fout", "Kies eerst een inputbestand.")
            return
        delim = "\t" if self.v_delim.get() == "\\t" else self.v_delim.get()
        try:
            self._log("Lezen bestand...")
            self.df, has_header, used = read_table(inp, "auto" if self.v_delim.get() == "auto" else delim, self.v_header.get().strip())
            self._log(f"  Rijen: {len(self.df)}, kolommen: {len(self.df.columns)}")
            self._log(f"  Delimiter: {repr(used)} | Header: {'ja' if has_header else 'nee'}")
            self._log(f"  Kolommen: {list(self.df.columns)}")
            g = guess_cols(self.df, has_header)
            values = list(self.df.columns)
            meta_values = [NONE] + values
            for cb, vals in (
                (self.cb_name, meta_values),
                (self.cb_group, meta_values),
                (self.cb_x, values),
                (self.cb_y, values),
                (self.cb_z, values),
                (self.cb_sigma_xy, values),
                (self.cb_sigma_z, values),
                (self.cb_time, meta_values),
                (self.cb_date, meta_values),
            ):
                cb["values"] = vals
            self.v_name.set(g.name or NONE)
            self.v_group.set(g.group or NONE)
            self.v_x.set(g.x or "")
            self.v_y.set(g.y or "")
            self.v_z.set(g.z or "")
            self.v_sigma_xy.set(g.sigma_xy or "")
            self.v_sigma_z.set(g.sigma_z or "")
            self.v_time.set(g.time or NONE)
            self.v_date.set(g.date or NONE)
            self._log(f"  Profiel: {g.profile}")
            if g.note:
                self._log(f"  {g.note}")
            if len(self.df):
                preview = " | ".join(f"{c}={s(self.df.iloc[0][c])}" for c in self.df.columns[:8])
                self._log(f"  Eerste rij: {preview}")
            messagebox.showinfo("OK", "Bestand ingelezen. Controleer de kolommen en klik daarna op 'Verwerk bestand'.")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc))
            self._log(f"ERROR: {exc}")

    def _opt_float(self, text: str) -> Optional[float]:
        text = text.strip().replace(",", ".")
        return None if text == "" else float(text)

    def _build_options(self) -> ProcessOptions:
        out = self.v_out.get().strip()
        if not out:
            raise ValueError("Kies eerst een outputmap.")
        x_col = self._sel(self.v_x.get())
        y_col = self._sel(self.v_y.get())
        z_col = self._sel(self.v_z.get())
        sigma_xy_col = self._sel(self.v_sigma_xy.get())
        sigma_z_col = self._sel(self.v_sigma_z.get())
        if not x_col or not y_col or not z_col or not sigma_xy_col or not sigma_z_col:
            raise ValueError("Selecteer kolommen voor X, Y, Z, SigmaXY en SigmaZ.")

        sigma_xy_a = float(self.v_sigma_xy_a.get().strip().replace(",", "."))
        sigma_z_a = float(self.v_sigma_z_a.get().strip().replace(",", "."))
        sigma_xy_b = float(self.v_sigma_xy_b.get().strip().replace(",", "."))
        sigma_z_b = float(self.v_sigma_z_b.get().strip().replace(",", "."))
        outlier_sigma = float(self.v_outlier_sigma.get().strip().replace(",", "."))
        if sigma_xy_a > sigma_xy_b or sigma_z_a > sigma_z_b:
            raise ValueError("A-grenzen moeten strenger of gelijk zijn aan B-grenzen.")

        return ProcessOptions(
            output_dir=Path(out),
            crs=self.v_crs.get().strip(),
            name_col=self._sel(self.v_name.get()),
            group_col=self._sel(self.v_group.get()),
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            sigma_xy_col=sigma_xy_col,
            sigma_z_col=sigma_z_col,
            time_col=self._sel(self.v_time.get()),
            date_col=self._sel(self.v_date.get()),
            sigma_xy_a=sigma_xy_a,
            sigma_z_a=sigma_z_a,
            sigma_xy_b=sigma_xy_b,
            sigma_z_b=sigma_z_b,
            outlier_sigma=outlier_sigma,
            worst_allowed_quality=self.v_worst_quality.get().strip() or "C",
            max_sigma_xy=self._opt_float(self.v_max_sigma_xy.get()),
            max_sigma_z=self._opt_float(self.v_max_sigma_z.get()),
            apply_outlier_filter=bool(self.v_apply_outlier.get()),
        )

    def _process(self) -> None:
        try:
            if self.df is None:
                self._load()
                if self.df is None:
                    return
            opts = self._build_options()
            self._log("Verwerken kwaliteitsanalyse...")
            result = process_dataframe(self.df, opts)
            for line in result["summary_lines"]:
                self._log("  " + line)
            self._log(f"Outputmap: {opts.output_dir}")
            messagebox.showinfo("Klaar", f"Verwerking compleet.\nOutput: {opts.output_dir}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc))
            self._log(f"ERROR: {exc}")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
