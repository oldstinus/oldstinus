#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M9 COMBI — RIGID XY op basis van Sokkia referenties + Z uit Carlson (meelopende RTK)

Dit is de aangepaste versie op jouw correctie:
- XY positie: NIET uit Carlson. XY wordt rigid georefererd op basis van enkele gekozen Sokkia RTK referentiepunten.
  => Afstanden/vorm van de M9-track blijven onvervormd (rotatie + translatie, schaal=1).
- Hoogte (Z): UIT Carlson RTK (meelopende punten), gekoppeld op tijd (nearest binnen tolerantie, daarna ffill/bfill).
  => Carlson H wordt gebruikt om boat_H te bepalen (eventueel met subtract/add zoals in je eerdere scripts).

Kaart:
- Standaard: enkel M9 punten (geen Sokkia/Carlson punten). Polyline optioneel.

Outputs (basisnaam):
- <base>_georef.mat          : M9 mat met Summary.Track overschreven (structuur blijft)
- <base>.csv                 : georef XY + Carlson-hoogte + (optioneel) depth/bed + WGS84
- <base>_xy_transform.csv    : rotatie/translatie + residuals op de gekozen XY referenties
- <base>_map.html            : kaart (optioneel)

Afhankelijkheden:
- verplicht: numpy, pandas, scipy, tkinter
- optioneel voor kaart: folium, pyproj, folium.plugins.MousePosition
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
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
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
import xml.etree.ElementTree as ET
import math

import numpy as np
import pandas as pd
import scipy.io as sio
from zoneinfo import ZoneInfo

# Optional map deps
try:
    import folium  # type: ignore
    from folium.plugins import MousePosition  # type: ignore
    from pyproj import Transformer  # type: ignore
    _MAP_OK = True
except Exception:
    folium = None
    MousePosition = None
    Transformer = None
    _MAP_OK = False


# -------------------------
# Scrollable container (main GUI)
# -------------------------

class ScrollableFrame(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # mousewheel
        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.inner)

    def _on_inner_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_mousewheel_linux, add="+")
        widget.bind("<Button-5>", self._on_mousewheel_linux, add="+")

    def _on_mousewheel(self, event):
        delta = int(-1 * (event.delta / 120)) if event.delta != 0 else 0
        if delta != 0:
            self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")


# -------------------------
# Helpers
# -------------------------

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


# -------------------------
# M9 read/write (struct intact, overwrite Summary.Track only)
# -------------------------

@dataclass
class M9Data:
    mat_dict: dict
    sys_obj: object
    summ_obj: object
    time_utc: pd.DatetimeIndex
    t_naive: pd.Series
    track_rel_EN: np.ndarray  # Nx2 (E,N) internal
    track_is_NE: bool
    depth_raw_m: np.ndarray | None


def read_m9_mat(
    mat_path: str | Path,
    m9_hour_shift: int = -1,
    m9_track_is_NE: bool = False,
) -> M9Data:
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    if "System" not in mat or "Summary" not in mat:
        raise ValueError("MAT mist 'System' en/of 'Summary'.")
    sys_ = mat["System"]
    summ = mat["Summary"]
    bt = mat.get("BottomTrack")

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
        track = track[:, [1, 0]]  # (N,E) -> (E,N) internal

    depth_raw = None
    if bt is not None and hasattr(bt, "BT_Depth"):
        try:
            d = np.asarray(bt.BT_Depth).astype(float).reshape(-1)
            if len(d) == len(track):
                depth_raw = d
        except Exception:
            depth_raw = None

    t_naive = pd.Series(pd.to_datetime(time_utc).tz_convert("UTC").tz_localize(None))

    return M9Data(
        mat_dict=mat,
        sys_obj=sys_,
        summ_obj=summ,
        time_utc=time_utc,
        t_naive=t_naive,
        track_rel_EN=track,
        track_is_NE=track_is_NE,
        depth_raw_m=depth_raw,
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

    mdict = {k: v for k, v in m9.mat_dict.items() if not str(k).startswith("__")}
    sio.savemat(
        str(out_path),
        mdict,
        do_compression=True,
        long_field_names=True,
        oned_as="row",
    )


# -------------------------
# Sokkia RTK (.mxl / MAXML) — enkel voor XY referenties
# -------------------------

def extract_type_solutions(mxl_path: str | Path) -> list[str]:
    try:
        tree = ET.parse(str(mxl_path))
        root = tree.getroot()
        ns_uri = root.tag[root.tag.find("{") + 1 : root.tag.find("}")] if root.tag.startswith("{") else ""
        ns = "{" + ns_uri + "}" if ns_uri else ""

        obs = root.find(f"{ns}ObservationSession")
        if obs is None:
            return []

        vals = []
        for gp in obs.findall(f"{ns}GPSPosition"):
            v = gp.findtext(f"{ns}TypeSolution")
            if v:
                v = str(v).strip()
                if v:
                    vals.append(v)
        return sorted(set(vals))
    except Exception:
        return []


def read_sokkia_maxml(mxl_path: str | Path) -> pd.DataFrame:
    tree = ET.parse(str(mxl_path))
    root = tree.getroot()

    ns_uri = root.tag[root.tag.find("{") + 1 : root.tag.find("}")] if root.tag.startswith("{") else ""
    ns = "{" + ns_uri + "}" if ns_uri else ""

    obs = root.find(f"{ns}ObservationSession")
    if obs is None:
        raise ValueError("MAXML: ObservationSession ontbreekt.")

    rows = []
    for gp in obs.findall(f"{ns}GPSPosition"):
        t_str = gp.findtext(f"{ns}Time")
        if not t_str:
            continue
        t = dt.datetime.fromisoformat(t_str.replace("Z", "+00:00")).astimezone(dt.timezone.utc)

        neh = gp.find(f"{ns}PlaneStation/{ns}Position/{ns}NEH")
        if neh is None:
            continue

        N = float(neh.findtext(f"{ns}North"))
        E = float(neh.findtext(f"{ns}East"))
        H = float(neh.findtext(f"{ns}Height"))
        sol = (gp.findtext(f"{ns}TypeSolution") or "").strip()

        rows.append((pd.Timestamp(t), E, N, H, sol))

    df = pd.DataFrame(rows, columns=["time_utc", "E_m", "N_m", "H_m", "solution"])
    if df.empty:
        raise ValueError("Geen Sokkia RTK posities (NEH) gevonden.")
    df["t"] = df["time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.sort_values("t").reset_index(drop=True)
    df["sokkia_id"] = np.arange(len(df)) + 1
    return df


# -------------------------
# Carlson RTK points (';' txt/csv) — voor hoogte (Z)
# -------------------------

def read_carlson_rtk_points(
    path: str | Path,
    assume_local_brussels: bool = True,
    swap_NE: bool = True,
) -> pd.DataFrame:
    """
    Leest je Carlson export met ';' scheiding.
    Verwacht: minstens 6 kolommen; laatste 2 zijn time en date.
    Kolommen 2,3,4 (0-based: 2=E? 3=N? 4=H) zoals in je eerdere script.
    swap_NE=True is behouden, maar voor hoogte maakt dat niet uit.
    """
    p = Path(path)
    df_raw = pd.read_csv(p, sep=";", header=None, dtype=str, engine="python")

    if df_raw.shape[1] < 6:
        raise ValueError(f"Carlson RTK bestand heeft te weinig kolommen ({df_raw.shape[1]}). Verwacht minstens 6.")

    ncol = df_raw.shape[1]

    # detect header row
    row0 = df_raw.iloc[0].tolist()
    header_keywords = ("east", "north", "height", "sigma", "time", "date", "datum", "tijd", "omschrijving", "idx")
    looks_like_header = False

    if ncol >= 5:
        if (not _is_number_like(row0[2])) and (not _is_number_like(row0[3])):
            looks_like_header = True

    whole = " ".join([str(x).lower() for x in row0])
    if any(k in whole for k in header_keywords):
        looks_like_header = True

    if looks_like_header:
        df_raw = df_raw.iloc[1:].reset_index(drop=True)

    ncol = df_raw.shape[1]
    time_col = ncol - 2
    date_col = ncol - 1

    out = pd.DataFrame()
    out["idx"] = df_raw.iloc[:, 0].astype(str).str.strip()
    out["name"] = df_raw.iloc[:, 1].astype(str).str.strip() if ncol >= 2 else ""

    out["E"] = df_raw.iloc[:, 2].map(_to_float)
    out["N"] = df_raw.iloc[:, 3].map(_to_float)
    out["H"] = df_raw.iloc[:, 4].map(_to_float)

    out["time_s"] = df_raw.iloc[:, time_col].astype(str).str.strip()
    out["date_s"] = df_raw.iloc[:, date_col].astype(str).str.strip()

    if bool(swap_NE):
        out[["E", "N"]] = out[["N", "E"]]

    naive = out.apply(lambda r: _parse_datetime(r["date_s"], r["time_s"]), axis=1)
    if pd.isna(naive).all():
        raise ValueError("Kon geen datum/tijd parsen uit Carlson RTK (laatste 2 kolommen).")

    tz_local = ZoneInfo("Europe/Brussels")
    if assume_local_brussels:
        utc = [d.replace(tzinfo=tz_local).astimezone(dt.timezone.utc) if d is not None else None for d in naive]
    else:
        utc = [d.replace(tzinfo=dt.timezone.utc) if d is not None else None for d in naive]

    out["time_utc"] = pd.to_datetime(utc, utc=True)
    out["t"] = out["time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)

    out = out.dropna(subset=["t", "H"]).copy()
    out = out.sort_values("t").reset_index(drop=True)
    out["carlson_id"] = np.arange(len(out)) + 1
    return out


# -------------------------
# Filtering (Hampel)
# -------------------------

def hampel(series: pd.Series, window: int = 31, n_sigmas: float = 4.0) -> tuple[pd.Series, pd.Series]:
    x = series.astype(float)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1

    med = x.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    sigma = 1.4826 * mad
    thr = n_sigmas * sigma

    is_out = (abs_dev > thr) & (sigma > 0)
    cleaned = x.copy()
    cleaned[is_out] = med[is_out]
    return cleaned, is_out.fillna(False)


# -------------------------
# Rigid transform (2D) — scale fixed to 1
# -------------------------

def rigid_transform_2d(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find R (2x2) and t (2,) such that Y ≈ X R^T + t  with scale=1.
    Returns R, t, residuals.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape or X.shape[1] != 2:
        raise ValueError("X en Y moeten Nx2 zijn en dezelfde vorm hebben.")

    n = X.shape[0]
    if n == 1:
        R = np.eye(2)
        t = (Y[0] - X[0]).reshape(2)
        res = np.linalg.norm((X @ R.T + t) - Y, axis=1)
        return R, t, res

    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0)
    Xc = X - x_mean
    Yc = Y - y_mean

    H = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = y_mean - (R @ x_mean)
    res = np.linalg.norm((X @ R.T + t) - Y, axis=1)
    return R, t, res


def rotation_deg_from_R(R: np.ndarray) -> float:
    ang = math.atan2(R[1, 0], R[0, 0])
    return float(ang * 180.0 / math.pi)


# -------------------------
# Core coupling
# -------------------------

def match_sokkia_refs_to_m9(
    m9: M9Data,
    sokkia: pd.DataFrame,
    selected_sokkia_ids: list[int],
    tol_s: float = 2.0,
    sokkia_time_shift_s: float = 0.0,
) -> pd.DataFrame:
    rtk = sokkia.copy()
    if float(sokkia_time_shift_s) != 0.0:
        rtk["t"] = rtk["t"] + pd.Timedelta(seconds=float(sokkia_time_shift_s))
        rtk["time_utc"] = rtk["t"].dt.tz_localize("UTC")

    sel = rtk[rtk["sokkia_id"].isin([int(x) for x in selected_sokkia_ids])].copy()
    if sel.empty:
        raise ValueError("Geen Sokkia referentiepunten geselecteerd.")

    m9_df = pd.DataFrame({
        "t": m9.t_naive,
        "m9_E_rel_m": m9.track_rel_EN[:, 0],
        "m9_N_rel_m": m9.track_rel_EN[:, 1],
    }).sort_values("t").reset_index(drop=True)

    sel = sel.sort_values("t").reset_index(drop=True)

    matched = pd.merge_asof(
        sel,
        m9_df,
        on="t",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tol_s)),
    ).dropna(subset=["m9_E_rel_m", "m9_N_rel_m"])

    if matched.empty:
        raise ValueError("Geen matches tussen geselecteerde Sokkia punten en M9 binnen tolerantietijd.")

    m9_times = m9_df["t"].values.astype("datetime64[ns]")
    dt_list = []
    m9_idx = []
    for tt in matched["t"].values.astype("datetime64[ns]"):
        i = int(np.argmin(np.abs(m9_times - tt)))
        m9_idx.append(i)
        dt_list.append(float(np.abs((pd.Timestamp(m9_times[i]) - pd.Timestamp(tt)).total_seconds())))
    matched["m9_index"] = m9_idx
    matched["dt_s"] = dt_list

    return matched


def apply_rigid_xy_to_all_m9(m9: M9Data, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    X = m9.track_rel_EN.astype(float)
    Y = X @ R.T + t.reshape(1, 2)
    return Y


def compute_carlson_height_for_m9(
    m9: M9Data,
    carlson: pd.DataFrame,
    tol_s: float = 2.0,
    carlson_time_shift_s: float = 0.0,
    h_sub: float = 2.0,
    h_add: float = 0.47,
) -> pd.DataFrame:
    """
    Map Carlson H to each M9 time (nearest within tol, else ffill/backfill).
    Carlson XY wordt NIET gebruikt.
    """
    c = carlson.copy()
    if float(carlson_time_shift_s) != 0.0:
        c["t"] = c["t"] + pd.Timedelta(seconds=float(carlson_time_shift_s))
        c["time_utc"] = c["t"].dt.tz_localize("UTC")

    c = c.sort_values("t").reset_index(drop=True)
    m9_df = pd.DataFrame({"t": m9.t_naive}).sort_values("t").reset_index(drop=True)

    h_near = pd.merge_asof(
        m9_df,
        c[["t", "H", "carlson_id"]],
        on="t",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tol_s)),
    )

    h_near["H"] = h_near["H"].ffill().bfill()
    h_near["carlson_id"] = h_near["carlson_id"].ffill().bfill()

    h_near["boat_H_m"] = h_near["H"].astype(float) - float(h_sub) + float(h_add)
    h_near["time_utc"] = h_near["t"].dt.tz_localize("UTC")
    return h_near


def build_output_dataframe(
    m9: M9Data,
    track_abs_EN: np.ndarray,
    height_map: pd.DataFrame,
    do_filter: bool,
    window: int,
    sigmas: float,
    filter_on: str,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "time_utc": pd.to_datetime(m9.time_utc),
        "E_abs_m": track_abs_EN[:, 0],
        "N_abs_m": track_abs_EN[:, 1],
        "carlson_H_m": height_map["H"].astype(float).to_numpy(),
        "boat_H_m": height_map["boat_H_m"].astype(float).to_numpy(),
        "carlson_id_for_H": height_map["carlson_id"].astype(int).to_numpy(),
    })

    if m9.depth_raw_m is not None:
        out["depth_raw_m"] = m9.depth_raw_m.astype(float)
        out["bed_H_raw_m"] = out["boat_H_m"] - out["depth_raw_m"]

        if do_filter:
            d_clean, d_sp = hampel(out["depth_raw_m"], window=window, n_sigmas=sigmas)
            b_clean, b_sp = hampel(out["bed_H_raw_m"], window=window, n_sigmas=sigmas)

            if filter_on == "depth":
                out["depth_clean_m"] = d_clean
                out["bed_H_clean_m"] = out["boat_H_m"] - out["depth_clean_m"]
                out["any_spike"] = d_sp
            elif filter_on == "both":
                combo = d_sp | b_sp
                b2 = out["bed_H_raw_m"].copy()
                b2[combo] = b_clean[combo]
                out["bed_H_clean_m"] = b2
                out["depth_clean_m"] = out["boat_H_m"] - out["bed_H_clean_m"]
                out["any_spike"] = combo
            else:  # bed
                out["bed_H_clean_m"] = b_clean
                out["depth_clean_m"] = out["boat_H_m"] - out["bed_H_clean_m"]
                out["any_spike"] = b_sp
        else:
            out["depth_clean_m"] = out["depth_raw_m"]
            out["bed_H_clean_m"] = out["bed_H_raw_m"]
            out["any_spike"] = False

    # WGS84
    if _MAP_OK:
        tr = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(out["E_abs_m"].to_numpy(), out["N_abs_m"].to_numpy())
        out["lon"] = lon
        out["lat"] = lat

    return out


def save_csv(df: pd.DataFrame, out_path: str | Path, sep: str = ";") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, sep=sep)


def make_map_html_points_only(
    df: pd.DataFrame,
    out_html: str | Path,
    every_n_markers: int = 5,
    draw_polyline: bool = False,
    polyline_step: int = 1,
) -> None:
    if not _MAP_OK:
        raise RuntimeError("folium/pyproj niet beschikbaar. Installeer: pip install folium pyproj")

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("lat/lon ontbreken (pyproj niet beschikbaar of geen WGS84 berekend).")

    center = [float(df["lat"].median()), float(df["lon"].median())]
    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap")

    if draw_polyline:
        step_line = max(1, int(polyline_step))
        coords = df[["lat", "lon"]].astype(float).values.tolist()
        folium.PolyLine(coords[::step_line], weight=3, opacity=0.7).add_to(m)

    step = max(1, int(every_n_markers))
    ds = df.iloc[::step].copy()
    for _, r in ds.iterrows():
        popup = (
            f"{r['time_utc']}<br>"
            f"E={float(r['E_abs_m']):.3f} N={float(r['N_abs_m']):.3f}<br>"
            f"boat_H={float(r.get('boat_H_m', np.nan)):.3f}<br>"
        )
        if "depth_clean_m" in df.columns:
            popup += f"depth_clean={float(r.get('depth_clean_m', np.nan)):.2f}<br>"
        if "bed_H_clean_m" in df.columns:
            popup += f"bed_H_clean={float(r.get('bed_H_clean_m', np.nan)):.2f}<br>"

        is_sp = bool(r.get("any_spike", False))
        color = "red" if is_sp else "blue"
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=450),
        ).add_to(m)

    if MousePosition is not None:
        MousePosition(
            position="topright",
            separator=" | ",
            prefix="WGS84",
            lat_formatter="function(num) {return L.Util.formatNum(num, 7);}",
            lng_formatter="function(num) {return L.Util.formatNum(num, 7);}",
        ).add_to(m)

    m.save(str(out_html))
    add_interactive_html_saver(out_html)


# -------------------------
# GUI
# -------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M9: rigid XY (Sokkia refs) + Z (Carlson) — punten-only kaart")
        self.geometry("1280x860")
        self.minsize(950, 650)

        # paths
        self.var_m9 = tk.StringVar()
        self.var_sokkia = tk.StringVar()
        self.var_carlson = tk.StringVar()
        self.var_out_base = tk.StringVar()

        # M9 options
        self.var_m9_shift = tk.StringVar(value="-1")
        self.var_m9_track_is_NE = tk.BooleanVar(value=False)

        # time options
        self.var_tol_xy = tk.DoubleVar(value=2.0)
        self.var_tol_h = tk.DoubleVar(value=2.0)
        self.var_sokkia_shift_s = tk.DoubleVar(value=0.0)
        self.var_carlson_shift_s = tk.DoubleVar(value=0.0)

        # Carlson parse options
        self.var_carlson_local = tk.BooleanVar(value=True)
        self.var_carlson_swap_NE = tk.BooleanVar(value=True)

        # Sokkia TypeSolution filter (dynamic)
        self.sol_vars: dict[str, tk.BooleanVar] = {}

        # height model (apply to Carlson H)
        self.var_hsub = tk.DoubleVar(value=2.0)
        self.var_hadd = tk.DoubleVar(value=0.47)

        # filtering
        self.var_filter = tk.BooleanVar(value=True)
        self.var_win = tk.IntVar(value=31)
        self.var_sig = tk.DoubleVar(value=4.0)
        self.var_filter_on = tk.StringVar(value="bed")

        # map
        self.var_make_map = tk.BooleanVar(value=True)
        self.var_open_map = tk.BooleanVar(value=True)
        self.var_map_every_markers = tk.IntVar(value=5)
        self.var_draw_polyline = tk.BooleanVar(value=False)
        self.var_polyline_step = tk.IntVar(value=1)

        # loaded
        self._m9: M9Data | None = None
        self._sokkia_df: pd.DataFrame | None = None
        self._carlson_df: pd.DataFrame | None = None

        self._build()

    def _build(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        scroll = ScrollableFrame(outer)
        scroll.pack(fill="both", expand=True)
        frm = scroll.inner
        frm.grid_columnconfigure(0, weight=1)

        r = 0
        ttk.Label(frm, text="M9 .mat").grid(row=r, column=0, sticky="w")
        r += 1
        ttk.Entry(frm, textvariable=self.var_m9, width=120).grid(row=r, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_m9).grid(row=r, column=1, sticky="e")
        r += 1

        ttk.Label(frm, text="Sokkia RTK .mxl (MAXML) — enkel voor XY referenties").grid(row=r, column=0, sticky="w", pady=(10, 0))
        r += 1
        ttk.Entry(frm, textvariable=self.var_sokkia, width=120).grid(row=r, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_sokkia).grid(row=r, column=1, sticky="e")
        r += 1

        ttk.Label(frm, text="Carlson RTK (.txt/.csv, ';') — voor hoogte (Z)").grid(row=r, column=0, sticky="w", pady=(10, 0))
        r += 1
        ttk.Entry(frm, textvariable=self.var_carlson, width=120).grid(row=r, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_carlson).grid(row=r, column=1, sticky="e")
        r += 1

        ttk.Label(frm, text="Output basisnaam (map + bestandsnaam zonder extensie)").grid(row=r, column=0, sticky="w", pady=(10, 0))
        r += 1
        ttk.Entry(frm, textvariable=self.var_out_base, width=120).grid(row=r, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Kies…", command=self._pick_out_base).grid(row=r, column=1, sticky="e")
        r += 1

        # --- Time & settings
        opt = ttk.LabelFrame(frm, text="Tijd / settings", padding=10)
        opt.grid(row=r, column=0, columnspan=2, sticky="we", pady=(12, 0))
        r += 1

        ttk.Label(opt, text="M9 uurshift (uren):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opt, textvariable=self.var_m9_shift, values=["-2", "-1", "0", "+1", "+2"], width=6, state="readonly").grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Checkbutton(opt, text="M9 Summary.Track is N,E (swap bij in-/uitschrijven)", variable=self.var_m9_track_is_NE).grid(row=0, column=2, sticky="w", padx=(18, 0))

        ttk.Label(opt, text="XY match tolerantie (s) (refs→M9):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_tol_xy, width=10).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="H match tolerantie (s) (Carlson→M9):").grid(row=1, column=2, sticky="e", padx=(18, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_tol_h, width=10).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Extra Sokkia time shift (s):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_sokkia_shift_s, width=10).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Extra Carlson time shift (s):").grid(row=2, column=2, sticky="e", padx=(18, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_carlson_shift_s, width=10).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        # --- Sokkia TypeSolution filter
        solf = ttk.LabelFrame(frm, text="Sokkia TypeSolution filter (wordt geladen uit .mxl)", padding=10)
        solf.grid(row=r, column=0, columnspan=2, sticky="we", pady=(10, 0))
        r += 1

        self.sol_container = ttk.Frame(solf)
        self.sol_container.grid(row=0, column=0, columnspan=4, sticky="we")

        self.sol_canvas = tk.Canvas(self.sol_container, height=90, highlightthickness=0)
        self.sol_scroll = ttk.Scrollbar(self.sol_container, orient="vertical", command=self.sol_canvas.yview)
        self.sol_inner = ttk.Frame(self.sol_canvas)

        self.sol_inner.bind("<Configure>", lambda e: self.sol_canvas.configure(scrollregion=self.sol_canvas.bbox("all")))
        self.sol_canvas.create_window((0, 0), window=self.sol_inner, anchor="nw")
        self.sol_canvas.configure(yscrollcommand=self.sol_scroll.set)

        self.sol_canvas.pack(side="left", fill="both", expand=True)
        self.sol_scroll.pack(side="right", fill="y")

        self.sol_hint = ttk.Label(self.sol_inner, text="Kies eerst een Sokkia .mxl om TypeSolution opties te laden…")
        self.sol_hint.grid(row=0, column=0, sticky="w")

        solbtn = ttk.Frame(solf)
        solbtn.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(solbtn, text="Selecteer alles", command=self._solutions_select_all).pack(side="left")
        ttk.Button(solbtn, text="Selecteer geen", command=self._solutions_select_none).pack(side="left", padx=(8, 0))

        # --- Sokkia reference selection (for XY transform)
        refbox = ttk.LabelFrame(frm, text="XY referentiepunten (Sokkia) — selecteer meerdere (Ctrl/Shift click)", padding=10)
        refbox.grid(row=r, column=0, columnspan=2, sticky="we", pady=(10, 0))
        r += 1
        refbox.grid_columnconfigure(0, weight=1)

        self.tree_sokkia = ttk.Treeview(refbox, columns=("sokkia_id", "time_utc", "E", "N", "solution"), show="headings", height=10, selectmode="extended")
        for c, w in [("sokkia_id", 90), ("time_utc", 220), ("E", 130), ("N", 130), ("solution", 220)]:
            self.tree_sokkia.heading(c, text=c)
            self.tree_sokkia.column(c, width=w, anchor="w")
        self.tree_sokkia.grid(row=0, column=0, columnspan=3, sticky="we")
        vsb = ttk.Scrollbar(refbox, orient="vertical", command=self.tree_sokkia.yview)
        self.tree_sokkia.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=3, sticky="ns")

        refbtn = ttk.Frame(refbox)
        refbtn.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(refbtn, text="Selecteer alles", command=self._sokkia_select_all).pack(side="left")
        ttk.Button(refbtn, text="Selecteer geen", command=self._sokkia_select_none).pack(side="left", padx=(8, 0))

        # --- Carlson parse options + hoogte model
        caropt = ttk.LabelFrame(frm, text="Carlson parsing + hoogte model", padding=10)
        caropt.grid(row=r, column=0, columnspan=2, sticky="we", pady=(10, 0))
        r += 1
        ttk.Checkbutton(caropt, text="Carlson tijden: Europe/Brussels → UTC", variable=self.var_carlson_local).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(caropt, text="Carlson export is N;E;H (swap N↔E)", variable=self.var_carlson_swap_NE).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(caropt, text="boat_H = Carlson_H - subtract + add").grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(caropt, text="subtract (m):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(caropt, textvariable=self.var_hsub, width=10).grid(row=2, column=1, sticky="w", padx=(6, 18), pady=(6, 0))
        ttk.Label(caropt, text="add (m):").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(caropt, textvariable=self.var_hadd, width=10).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(6, 0))

        # --- Filtering
        ffrm = ttk.LabelFrame(frm, text="Filtering (spikes door ondieptes)", padding=10)
        ffrm.grid(row=r, column=0, columnspan=2, sticky="we", pady=(10, 0))
        r += 1
        ttk.Checkbutton(ffrm, text="Filtering aan", variable=self.var_filter).grid(row=0, column=0, sticky="w")
        ttk.Label(ffrm, text="Hampel window:").grid(row=0, column=1, sticky="e", padx=(20, 0))
        ttk.Entry(ffrm, textvariable=self.var_win, width=10).grid(row=0, column=2, sticky="w", padx=(6, 0))
        ttk.Label(ffrm, text="Sigmas:").grid(row=0, column=3, sticky="e", padx=(20, 0))
        ttk.Entry(ffrm, textvariable=self.var_sig, width=10).grid(row=0, column=4, sticky="w", padx=(6, 0))
        ttk.Label(ffrm, text="Filter op:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(ffrm, textvariable=self.var_filter_on, values=["bed", "depth", "both"], width=10, state="readonly").grid(row=1, column=1, sticky="w", pady=(8, 0))

        # --- Map
        mfrm = ttk.LabelFrame(frm, text="Kaart (Folium) — M9 punten only", padding=10)
        mfrm.grid(row=r, column=0, columnspan=2, sticky="we", pady=(10, 0))
        r += 1
        ttk.Checkbutton(mfrm, text="Maak kaart HTML na succes", variable=self.var_make_map).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(mfrm, text="Open kaart automatisch", variable=self.var_open_map).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(mfrm, text="Marker elke n punten:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(mfrm, textvariable=self.var_map_every_markers, width=8).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Checkbutton(mfrm, text="Teken ook polyline (optioneel)", variable=self.var_draw_polyline).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(mfrm, text="Polyline step:").grid(row=2, column=1, sticky="e", padx=(12, 0), pady=(6, 0))
        ttk.Entry(mfrm, textvariable=self.var_polyline_step, width=8).grid(row=2, column=2, sticky="w", padx=(6, 0), pady=(6, 0))

        # --- Buttons + log
        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=2, sticky="we", pady=(12, 0))
        r += 1
        ttk.Button(btns, text="1) Lees M9", command=self._load_m9).pack(side="left")
        ttk.Button(btns, text="2) Lees Sokkia", command=self._load_sokkia).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="3) Lees Carlson", command=self._load_carlson).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="4) Run + export", command=self._run).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Sluiten", command=self.destroy).pack(side="right")

        ttk.Label(frm, text="Log:").grid(row=r, column=0, sticky="w", pady=(10, 0))
        r += 1
        self.txt = tk.Text(frm, height=12, wrap="word")
        self.txt.grid(row=r, column=0, columnspan=2, sticky="we")

    def _log(self, s: str):
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    # ---- pickers
    def _pick_m9(self):
        p = filedialog.askopenfilename(title="Selecteer M9 .mat", filetypes=[("MAT", "*.mat"), ("All", "*.*")])
        if p:
            self.var_m9.set(p)
            if not self.var_out_base.get():
                base = Path(p).with_suffix("")
                self.var_out_base.set(str(base) + "_RIGIDXY_CARLSONZ")

    def _populate_solutions(self, mxl_path: str):
        for w in list(self.sol_inner.winfo_children()):
            w.destroy()
        self.sol_vars = {}

        sols = extract_type_solutions(mxl_path)
        if not sols:
            ttk.Label(self.sol_inner, text="Geen TypeSolution gevonden in .mxl.").grid(row=0, column=0, sticky="w")
            return

        for i, s in enumerate(sols):
            var = tk.BooleanVar(value=True)
            self.sol_vars[s] = var
            cb = ttk.Checkbutton(self.sol_inner, text=s, variable=var)
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 18))

    def _pick_sokkia(self):
        p = filedialog.askopenfilename(title="Selecteer Sokkia RTK .mxl", filetypes=[("MXL", "*.mxl"), ("XML", "*.xml"), ("All", "*.*")])
        if p:
            self.var_sokkia.set(p)
            self._populate_solutions(p)

    def _pick_carlson(self):
        p = filedialog.askopenfilename(title="Selecteer Carlson RTK", filetypes=[("Text/CSV", "*.txt *.csv"), ("All", "*.*")])
        if p:
            self.var_carlson.set(p)

    def _pick_out_base(self):
        p = filedialog.asksaveasfilename(
            title="Kies basisnaam (zonder extensie)",
            defaultextension="",
            filetypes=[("Geen", "*.*")],
        )
        if p:
            self.var_out_base.set(str(Path(p)))

    # ---- solution buttons
    def _solutions_select_all(self):
        for v in self.sol_vars.values():
            v.set(True)

    def _solutions_select_none(self):
        for v in self.sol_vars.values():
            v.set(False)

    # ---- sokkia selection buttons
    def _sokkia_select_all(self):
        for iid in self.tree_sokkia.get_children():
            self.tree_sokkia.selection_add(iid)

    def _sokkia_select_none(self):
        self.tree_sokkia.selection_remove(self.tree_sokkia.selection())

    # ---- loaders
    def _load_m9(self):
        try:
            p = self.var_m9.get().strip()
            if not p:
                messagebox.showerror("Input", "Kies een M9 .mat.")
                return
            shift = int(self.var_m9_shift.get())
            is_ne = bool(self.var_m9_track_is_NE.get())
            self._log("Lezen M9…")
            self._m9 = read_m9_mat(p, m9_hour_shift=shift, m9_track_is_NE=is_ne)
            self._log(f"  OK: ensembles={len(self._m9.track_rel_EN)} | {self._m9.time_utc.min()} → {self._m9.time_utc.max()}")
            if self._m9.depth_raw_m is None:
                self._log("  Opmerking: BottomTrack.BT_Depth niet gevonden -> enkel XY+boat_H export (geen bed).")
            messagebox.showinfo("OK", "M9 ingelezen.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")

    def _load_sokkia(self):
        try:
            p = self.var_sokkia.get().strip()
            if not p:
                messagebox.showerror("Input", "Kies een Sokkia .mxl.")
                return
            self._log("Lezen Sokkia RTK…")
            df = read_sokkia_maxml(p)

            if self.sol_vars:
                allowed = [k for k, v in self.sol_vars.items() if bool(v.get())]
                if not allowed:
                    raise ValueError("Geen TypeSolution aangevinkt. Vink minstens één kwaliteit aan.")
                before = len(df)
                df["solution"] = df["solution"].astype(str).str.strip()
                df = df[df["solution"].isin([a.strip() for a in allowed])].copy()
                self._log(f"  TypeSolution filter: {len(df)} (van {before}) | allowed={allowed}")
                if df.empty:
                    raise ValueError("Na TypeSolution filter zijn er geen Sokkia punten meer.")
            else:
                self._log("  TypeSolution filter niet toegepast (geen opties).")

            self._sokkia_df = df.sort_values("t").reset_index(drop=True)

            for it in self.tree_sokkia.get_children():
                self.tree_sokkia.delete(it)
            for _, r in self._sokkia_df.iterrows():
                self.tree_sokkia.insert(
                    "", "end",
                    values=(
                        int(r["sokkia_id"]),
                        str(r["time_utc"]),
                        f"{float(r['E_m']):.3f}",
                        f"{float(r['N_m']):.3f}",
                        str(r["solution"]),
                    )
                )

            self._log(f"  OK: punten={len(self._sokkia_df)} | {self._sokkia_df.time_utc.min()} → {self._sokkia_df.time_utc.max()}")
            messagebox.showinfo("OK", "Sokkia ingelezen. Selecteer nu enkele Sokkia punten als XY-referentie (Ctrl/Shift).")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")

    def _load_carlson(self):
        try:
            p = self.var_carlson.get().strip()
            if not p:
                messagebox.showerror("Input", "Kies een Carlson RTK bestand.")
                return
            local = bool(self.var_carlson_local.get())
            swap = bool(self.var_carlson_swap_NE.get())
            self._log("Lezen Carlson RTK…")
            df = read_carlson_rtk_points(p, assume_local_brussels=local, swap_NE=swap)
            self._carlson_df = df
            self._log(f"  OK: punten={len(df)} | {df.time_utc.min()} → {df.time_utc.max()}")
            messagebox.showinfo("OK", "Carlson ingelezen (hoogtebron).")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")

    def _get_selected_sokkia_ids(self) -> list[int]:
        sel = self.tree_sokkia.selection()
        ids: list[int] = []
        for iid in sel:
            vals = self.tree_sokkia.item(iid, "values")
            if vals:
                ids.append(int(vals[0]))
        return ids

    # ---- run
    def _run(self):
        try:
            if self._m9 is None:
                self._load_m9()
            if self._sokkia_df is None:
                self._load_sokkia()
            if self._carlson_df is None:
                self._load_carlson()
            if self._m9 is None or self._sokkia_df is None or self._carlson_df is None:
                return

            out_base = self.var_out_base.get().strip()
            if not out_base:
                messagebox.showerror("Output", "Kies een output basisnaam.")
                return

            tol_xy = float(self.var_tol_xy.get())
            tol_h = float(self.var_tol_h.get())
            sokkia_shift_s = float(self.var_sokkia_shift_s.get())
            carlson_shift_s = float(self.var_carlson_shift_s.get())
            hsub = float(self.var_hsub.get())
            hadd = float(self.var_hadd.get())

            dof = bool(self.var_filter.get())
            win = int(self.var_win.get())
            sig = float(self.var_sig.get())
            fon = str(self.var_filter_on.get()).strip().lower()

            sel_ids = self._get_selected_sokkia_ids()
            if len(sel_ids) < 1:
                messagebox.showerror("XY referenties", "Selecteer minstens 1 Sokkia punt als XY-referentie (in de tabel).")
                return

            base = Path(out_base)

            self._log("1) Match geselecteerde Sokkia XY-referenties → M9…")
            matched = match_sokkia_refs_to_m9(
                m9=self._m9,
                sokkia=self._sokkia_df,
                selected_sokkia_ids=sel_ids,
                tol_s=tol_xy,
                sokkia_time_shift_s=sokkia_shift_s,
            )
            self._log(f"  OK: matches={len(matched)} (tol={tol_xy}s)")

            X = matched[["m9_E_rel_m", "m9_N_rel_m"]].to_numpy(dtype=float)
            Y = matched[["E_m", "N_m"]].to_numpy(dtype=float)

            self._log("2) Bepaal rigid XY transform (rotatie+translatie, schaal=1)…")
            R, t, res = rigid_transform_2d(X, Y)
            ang = rotation_deg_from_R(R)
            self._log(f"  Rotatie ≈ {ang:.3f}° | Translatie t=({t[0]:.3f}, {t[1]:.3f}) m")
            self._log(f"  Residuals op refs: mean={np.mean(res):.3f} m | max={np.max(res):.3f} m")

            self._log("3) Pas transform toe op ALLE M9 punten (afstanden onvervormd)…")
            track_abs = apply_rigid_xy_to_all_m9(self._m9, R, t)

            self._log("4) Hoogte uit Carlson (tijdkoppeling, los van XY)…")
            height_map = compute_carlson_height_for_m9(
                m9=self._m9,
                carlson=self._carlson_df,
                tol_s=tol_h,
                carlson_time_shift_s=carlson_shift_s,
                h_sub=hsub,
                h_add=hadd,
            )
            self._log("  OK: boat_H gekoppeld (ffill/bfill)")

            self._log("5) Output dataframe bouwen…")
            out_df = build_output_dataframe(
                m9=self._m9,
                track_abs_EN=track_abs,
                height_map=height_map,
                do_filter=dof,
                window=win,
                sigmas=sig,
                filter_on=fon,
            )

            out_csv = str(base) + ".csv"
            out_mat = str(base) + "_georef.mat"
            tf_csv = str(base) + "_xy_transform.csv"

            self._log("6) Schrijven CSV…")
            save_csv(out_df, out_csv, sep=";")
            self._log(f"  Saved: {out_csv}")

            self._log("7) Schrijven .mat (Summary.Track overschrijven)…")
            write_m9_mat_overwrite_track(self._m9, track_abs, out_mat)
            self._log(f"  Saved: {out_mat}")

            self._log("8) Schrijven transform info…")
            tf = pd.DataFrame({
                "selected_sokkia_id": matched["sokkia_id"].astype(int),
                "sokkia_time_utc": matched["time_utc"].astype(str),
                "m9_index": matched["m9_index"].astype(int),
                "m9_E_rel_m": matched["m9_E_rel_m"].astype(float),
                "m9_N_rel_m": matched["m9_N_rel_m"].astype(float),
                "sokkia_E_m": matched["E_m"].astype(float),
                "sokkia_N_m": matched["N_m"].astype(float),
                "dt_s": matched["dt_s"].astype(float),
                "residual_m": res.astype(float),
            })
            meta = pd.DataFrame({
                "param": ["R00","R01","R10","R11","tE","tN","rotation_deg","res_mean_m","res_max_m"],
                "value": [R[0,0], R[0,1], R[1,0], R[1,1], t[0], t[1], ang, float(np.mean(res)), float(np.max(res))]
            })
            save_csv(meta, tf_csv, sep=";")
            with open(tf_csv, "a", encoding="utf-8") as f:
                f.write("\n")
            tf.to_csv(tf_csv, mode="a", index=False, sep=";")
            self._log(f"  Saved: {tf_csv}")

            if bool(self.var_make_map.get()):
                if not _MAP_OK:
                    self._log("WAARSCHUWING: folium/pyproj niet beschikbaar -> kaart niet gemaakt.")
                else:
                    map_html = str(base) + "_map.html"
                    self._log(f"9) Kaart maken (M9 punten only): {map_html}")
                    make_map_html_points_only(
                        df=out_df,
                        out_html=map_html,
                        every_n_markers=int(self.var_map_every_markers.get()),
                        draw_polyline=bool(self.var_draw_polyline.get()),
                        polyline_step=int(self.var_polyline_step.get()),
                    )
                    self._log("  Kaart opgeslagen.")
                    if bool(self.var_open_map.get()):
                        webbrowser.open(Path(map_html).resolve().as_uri())

            messagebox.showinfo("OK", "Klaar. Rigid XY (Sokkia refs) + Z (Carlson) is uitgevoerd.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
