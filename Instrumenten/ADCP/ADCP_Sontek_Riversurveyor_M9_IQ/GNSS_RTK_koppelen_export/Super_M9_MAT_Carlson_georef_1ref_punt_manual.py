#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M9 (.mat) georefereren en opslaan naar nieuwe .mat met volledige originele data.
Alleen Summary.Track wordt overschreven.

Ondersteunde referentiemodi:
- RTK ankerpunt via tijdmatching (zoals origineel).
- Manuele referentie:
  - klik een punt op de gemeten track (kaart/plot),
  - geef coordinaten in als Lambert72, UTM 31N of WGS84 (lat/lon),
  - script rekent dit automatisch om naar Lambert72,
  - bereken en schrijf de volledige georefererde track weg.

Extra:
- Optioneel Folium HTML kaart na georeferering.

Kern:
- M9 tijd: System.Time = seconden sinds 2000-01-01 (UTC).
- M9 track: Summary.Track = Nx2 (relatieve XY in meter) -> wordt absoluut Lambert72 (E,N) na offset.
- Carlson RTK: ';' gescheiden, met tijd in voorlaatste kolom (HH:MM:SS) en datum in laatste kolom (YYYY/MM/DD).
  Typische opbouw:
    idx ; omschrijving ; East ; North ; Height ; sigmaE ; sigmaN ; ... ; HH:MM:SS ; YYYY/MM/DD
  (Sommige exports hebben North;East;Height i.p.v. East;North;Height -> optionele swap)
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

import numpy as np
import pandas as pd
import scipy.io as sio
from zoneinfo import ZoneInfo

# Optional deps for map and coordinate transforms
try:
    import folium  # type: ignore
    from folium.plugins import MousePosition  # type: ignore
    _FOLIUM_OK = True
except Exception:
    folium = None
    MousePosition = None
    _FOLIUM_OK = False

try:
    from pyproj import Transformer  # type: ignore
    _PYPROJ_OK = True
except Exception:
    Transformer = None
    _PYPROJ_OK = False

_MAP_OK = _FOLIUM_OK and _PYPROJ_OK

# Optional deps for interactive point picking
try:
    from matplotlib.figure import Figure  # type: ignore
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # type: ignore
    _PLOT_OK = True
except Exception:
    Figure = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    _PLOT_OK = False


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


def _as_1d(arr: object | None, n: int) -> np.ndarray:
    if arr is None:
        return np.full(n, np.nan, dtype=float)

    out = np.asarray(arr, dtype=float).reshape(-1)
    if len(out) >= n:
        out = out[:n]
    else:
        out = np.pad(out, (0, n - len(out)), mode="constant", constant_values=np.nan)
    out[~np.isfinite(out)] = np.nan
    return out


def rotate_track_clockwise(
    track_EN: np.ndarray,
    angle_deg_cw: float,
    pivot_E: float,
    pivot_N: float,
) -> np.ndarray:
    track = np.asarray(track_EN, dtype=float)
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError("track_EN moet Nx2 zijn (E,N).")

    angle = float(angle_deg_cw)
    if abs(angle) < 1e-12:
        return track[:, :2].copy()

    theta = np.deg2rad(angle)
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    out = track[:, :2].copy()
    dE = out[:, 0] - float(pivot_E)
    dN = out[:, 1] - float(pivot_N)

    # Klokwijs rotatie in (E,N)-vlak.
    out[:, 0] = float(pivot_E) + c * dE + s * dN
    out[:, 1] = float(pivot_N) - s * dE + c * dN
    return out


def convert_manual_reference_to_lambert72(value1: float, value2: float, coord_mode: str) -> tuple[float, float]:
    mode = str(coord_mode).strip()

    if mode == "Lambert72 (E,N)":
        return float(value1), float(value2)

    if not _PYPROJ_OK:
        raise RuntimeError("pyproj ontbreekt. Installeer: pip install pyproj")

    if mode == "UTM 31N (WGS84 / EPSG:32631)":
        tr = Transformer.from_crs("EPSG:32631", "EPSG:31370", always_xy=True)
        e_l72, n_l72 = tr.transform(float(value1), float(value2))
        return float(e_l72), float(n_l72)

    if mode == "WGS84 (lat,lon)":
        lat = float(value1)
        lon = float(value2)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError("WGS84 buiten bereik. Verwacht lat in [-90,90] en lon in [-180,180].")
        tr = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
        e_l72, n_l72 = tr.transform(lon, lat)  # always_xy=True => (lon, lat)
        return float(e_l72), float(n_l72)

    raise ValueError(f"Onbekende coordinate mode: {mode}")


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
    track_rel: np.ndarray  # Nx2 float (E,N internally)
    track_is_NE: bool      # if true, columns were (N,E) and we swapped to (E,N) internally
    gps_lat: np.ndarray
    gps_lon: np.ndarray


def read_m9_mat(mat_path: str | Path, m9_hour_shift: int = -1, m9_track_is_NE: bool = False) -> M9Data:
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    if "System" not in mat or "Summary" not in mat:
        raise ValueError("MAT mist 'System' en/of 'Summary'.")

    sys_ = mat["System"]
    summ = mat["Summary"]
    gps = mat.get("GPS")

    if not hasattr(sys_, "Time"):
        raise ValueError("MAT: System.Time niet gevonden.")
    time_sec = np.asarray(sys_.Time).astype(float).reshape(-1)
    if len(time_sec) == 0:
        raise ValueError("MAT: System.Time is leeg.")
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

    n = len(time_sec)
    gps_lat = _as_1d(getattr(gps, "Latitude", None) if gps is not None else None, n)
    gps_lon = _as_1d(getattr(gps, "Longitude", None) if gps is not None else None, n)
    zero_mask = (gps_lat == 0.0) & (gps_lon == 0.0)
    gps_lat[zero_mask] = np.nan
    gps_lon[zero_mask] = np.nan

    t_naive = pd.Series(pd.to_datetime(time_utc).tz_convert("UTC").tz_localize(None))

    return M9Data(
        mat_dict=mat,
        sys_obj=sys_,
        summ_obj=summ,
        time_utc=time_utc,
        t_naive=t_naive,
        track_rel=track,
        track_is_NE=track_is_NE,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
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

    # SciPy warnings voorkomen: __header__/__version__/__globals__ niet wegschrijven
    mdict = {k: v for k, v in m9.mat_dict.items() if not str(k).startswith("__")}

    sio.savemat(
        str(out_path),
        mdict,
        do_compression=True,
        long_field_names=True,
        oned_as="row",
    )


# ---------------------------
# Carlson RTK reader (robust, zoals je werkend script)
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

    # detect header row (ook als sommige exports toch headers hebben)
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

    # Carlson: coördinaten in kolommen 3-5 (0-index 2..4)
    out["E"] = df_raw.iloc[:, 2].map(_to_float)
    out["N"] = df_raw.iloc[:, 3].map(_to_float)
    out["H"] = df_raw.iloc[:, 4].map(_to_float)

    if ncol >= 7:
        out["sigmaE"] = df_raw.iloc[:, 5].map(_to_float)
        out["sigmaN"] = df_raw.iloc[:, 6].map(_to_float)
    else:
        out["sigmaE"] = np.nan
        out["sigmaN"] = np.nan

    # Jouw format: tijd = voorlaatste kolom, datum = laatste kolom
    out["time_s"] = df_raw.iloc[:, time_col].astype(str).str.strip()
    out["date_s"] = df_raw.iloc[:, date_col].astype(str).str.strip()

    # Optionele swap als export eigenlijk N;E;H is
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
    rotation_deg_cw: float = 0.0,
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

    rotated_track = rotate_track_clockwise(
        m9.track_rel,
        angle_deg_cw=float(rotation_deg_cw),
        pivot_E=m9_e_rel,
        pivot_N=m9_n_rel,
    )
    m9_e_rot = float(rotated_track[idx, 0])
    m9_n_rot = float(rotated_track[idx, 1])

    dE = rtk_e - m9_e_rot
    dN = rtk_n - m9_n_rot

    new_track = rotated_track.copy()
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
        "m9_E_rot_at_match": m9_e_rot,
        "m9_N_rot_at_match": m9_n_rot,
        "rotation_deg_cw": float(rotation_deg_cw),
        "offset_dE": dE,
        "offset_dN": dN,
    }
    return new_track, info


def georef_track_by_manual_reference(
    m9: M9Data,
    track_index: int,
    ref_E: float,
    ref_N: float,
    rotation_deg_cw: float = 0.0,
):
    idx = int(track_index)
    if idx < 0 or idx >= len(m9.track_rel):
        raise ValueError(f"Track index buiten bereik: {idx}.")

    m9_e_rel = float(m9.track_rel[idx, 0])
    m9_n_rel = float(m9.track_rel[idx, 1])

    rotated_track = rotate_track_clockwise(
        m9.track_rel,
        angle_deg_cw=float(rotation_deg_cw),
        pivot_E=m9_e_rel,
        pivot_N=m9_n_rel,
    )
    m9_e_rot = float(rotated_track[idx, 0])
    m9_n_rot = float(rotated_track[idx, 1])

    dE = float(ref_E) - m9_e_rot
    dN = float(ref_N) - m9_n_rot

    new_track = rotated_track.copy()
    new_track[:, 0] = new_track[:, 0] + dE
    new_track[:, 1] = new_track[:, 1] + dN

    info = {
        "matched_m9_index": idx,
        "matched_m9_time_utc": str(pd.to_datetime(m9.time_utc[idx])),
        "m9_E_rel_at_match": m9_e_rel,
        "m9_N_rel_at_match": m9_n_rel,
        "m9_E_rot_at_match": m9_e_rot,
        "m9_N_rot_at_match": m9_n_rot,
        "rotation_deg_cw": float(rotation_deg_cw),
        "offset_dE": dE,
        "offset_dN": dN,
        "ref_E": float(ref_E),
        "ref_N": float(ref_N),
    }
    return new_track, info


# ---------------------------
# Map (Folium)
# ---------------------------

def make_map_html(
    m9_times_utc: pd.DatetimeIndex,
    track_EN: np.ndarray,
    ref_E: float,
    ref_N: float,
    ref_label: str,
    out_html: str | Path,
    every_n_markers: int = 20,
    every_n_line: int = 1,
) -> None:
    if not _MAP_OK:
        raise RuntimeError("folium/pyproj niet beschikbaar. Installeer: pip install folium pyproj")

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    tr = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)

    E = np.asarray(track_EN[:, 0], dtype=float)
    N = np.asarray(track_EN[:, 1], dtype=float)
    lon, lat = tr.transform(E, N)

    center = [float(np.nanmedian(lat)), float(np.nanmedian(lon))]
    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap")

    # track line
    step_line = max(1, int(every_n_line))
    line_coords = np.column_stack([lat[::step_line], lon[::step_line]]).tolist()
    folium.PolyLine(line_coords, weight=4, opacity=0.9).add_to(m)

    # reference marker
    aE = float(ref_E)
    aN = float(ref_N)
    a_lon, a_lat = tr.transform(aE, aN)
    a_popup = (
        f"{str(ref_label)}<br>"
        f"E={aE:.3f} N={aN:.3f}<br>"
        f"lat={a_lat:.7f} lon={a_lon:.7f}"
    )
    folium.Marker([float(a_lat), float(a_lon)], popup=folium.Popup(a_popup, max_width=450)).add_to(m)

    # point markers (downsample)
    step = max(1, int(every_n_markers))
    for i in range(0, len(E), step):
        t = pd.to_datetime(m9_times_utc[i]).tz_convert("UTC")
        popup = (
            f"{t.isoformat()}<br>"
            f"E={E[i]:.3f} N={N[i]:.3f}<br>"
            f"lat={lat[i]:.7f} lon={lon[i]:.7f}"
        )
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=3,
            fill=True,
            opacity=0.9,
            fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=450),
        ).add_to(m)

    # mouse position readout
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


# ---------------------------
# GUI
# ---------------------------

class TrackPointPicker(tk.Toplevel):
    def __init__(self, parent: tk.Tk, m9: M9Data):
        super().__init__(parent)
        self.title("Kies punt op de gemeten track")
        self.geometry("980x700")
        self.result_index: int | None = None
        self._m9 = m9
        self._picked = None

        has_gps = np.count_nonzero(np.isfinite(m9.gps_lat) & np.isfinite(m9.gps_lon)) >= 10
        if has_gps:
            self._x = m9.gps_lon.astype(float)
            self._y = m9.gps_lat.astype(float)
            xlabel = "Longitude (WGS84)"
            ylabel = "Latitude (WGS84)"
            title = "Klik trackpunt (GPS weergave)"
        else:
            self._x = m9.track_rel[:, 0].astype(float)
            self._y = m9.track_rel[:, 1].astype(float)
            xlabel = "Track X (m)"
            ylabel = "Track Y (m)"
            title = "Klik trackpunt (Summary.Track weergave)"

        self._valid_idx = np.where(np.isfinite(self._x) & np.isfinite(self._y))[0]
        if len(self._valid_idx) == 0:
            raise ValueError("Geen geldige punten om te selecteren.")

        self._sel_txt = tk.StringVar(value="Klik op de track om een punt te kiezen.")

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        ttk.Label(root, textvariable=self._sel_txt).pack(anchor="w")

        fig = Figure(figsize=(8.2, 6.0), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(self._x[self._valid_idx], self._y[self._valid_idx], "-", lw=1.2, alpha=0.9)
        ax.scatter(self._x[self._valid_idx], self._y[self._valid_idx], s=8, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.35)

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        toolbar.pack(fill="x")

        def _on_click(event):
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                return
            dx = self._x[self._valid_idx] - float(event.xdata)
            dy = self._y[self._valid_idx] - float(event.ydata)
            i_local = int(np.argmin(dx * dx + dy * dy))
            idx = int(self._valid_idx[i_local])
            self.result_index = idx

            if self._picked is not None:
                self._picked.remove()
            self._picked = ax.scatter([self._x[idx]], [self._y[idx]], s=95, marker="x", linewidths=2.0)
            canvas.draw_idle()

            t = pd.to_datetime(self._m9.time_utc[idx]).tz_convert("UTC").isoformat()
            e = float(self._m9.track_rel[idx, 0])
            n = float(self._m9.track_rel[idx, 1])
            self._sel_txt.set(f"Gekozen index={idx} | UTC={t} | Track(E,N)=({e:.3f}, {n:.3f})")

        canvas.mpl_connect("button_press_event", _on_click)

        btns = ttk.Frame(root)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Gebruik dit punt", command=self._ok).pack(side="left")
        ttk.Button(btns, text="Annuleer", command=self._cancel).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.transient(parent)
        self.grab_set()

    def _ok(self):
        if self.result_index is None:
            messagebox.showerror("Selectie", "Klik eerst een punt op de track.")
            return
        self.destroy()

    def _cancel(self):
        self.result_index = None
        self.destroy()


class App(tk.Tk):
    MODE_RTK = "RTK ankerpunt (tijdmatch)"
    MODE_MANUAL = "Manueel (klik track + geef E/N in)"
    MANUAL_COORD_L72 = "Lambert72 (E,N)"
    MANUAL_COORD_UTM31 = "UTM 31N (WGS84 / EPSG:32631)"
    MANUAL_COORD_WGS84 = "WGS84 (lat,lon)"

    def __init__(self):
        super().__init__()
        self.title("M9 georefereren met RTK of manueel referentiepunt -> .mat")
        self.geometry("1280x860")

        self.var_m9 = tk.StringVar()
        self.var_rtk = tk.StringVar()
        self.var_out = tk.StringVar()
        self.var_mode = tk.StringVar(value=self.MODE_RTK)

        self.var_m9_shift = tk.StringVar(value="-1")
        self.var_tol = tk.DoubleVar(value=2.0)
        self.var_rtk_local = tk.BooleanVar(value=True)
        self.var_rtk_shift_s = tk.DoubleVar(value=0.0)
        self.var_rtk_swap_NE = tk.BooleanVar(value=True)
        self.var_m9_track_is_NE = tk.BooleanVar(value=False)
        self.var_track_rotation_cw_deg = tk.DoubleVar(value=0.0)

        self.var_manual_track_idx = tk.StringVar(value="")
        self.var_manual_E = tk.StringVar(value="")
        self.var_manual_N = tk.StringVar(value="")
        self.var_manual_coord_mode = tk.StringVar(value=self.MANUAL_COORD_L72)
        self.var_manual_label_1 = tk.StringVar(value="E (m):")
        self.var_manual_label_2 = tk.StringVar(value="N (m):")

        self.var_make_map = tk.BooleanVar(value=True)
        self.var_open_map = tk.BooleanVar(value=True)
        self.var_map_every_n_markers = tk.IntVar(value=25)
        self.var_map_every_n_line = tk.IntVar(value=1)

        self._rtk_df: pd.DataFrame | None = None
        self._m9: M9Data | None = None
        self._manual_index: int | None = None

        self._rtk_controls: list[tk.Widget] = []
        self._manual_controls: list[tk.Widget] = []

        self._build()
        self._update_mode_ui()
        self._update_manual_coord_labels()

    def _build(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        row = 0
        ttk.Label(frm, text="M9 .mat").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Entry(frm, textvariable=self.var_m9, width=116).grid(row=row, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Bladeren...", command=self._pick_m9).grid(row=row, column=1, sticky="e")
        row += 1

        ttk.Label(frm, text="Carlson RTK bestand (.txt/.csv, ';')").grid(row=row, column=0, sticky="w", pady=(10, 0))
        row += 1
        e_rtk = ttk.Entry(frm, textvariable=self.var_rtk, width=116)
        e_rtk.grid(row=row, column=0, sticky="we", padx=(0, 8))
        b_rtk = ttk.Button(frm, text="Bladeren...", command=self._pick_rtk)
        b_rtk.grid(row=row, column=1, sticky="e")
        self._rtk_controls.extend([e_rtk, b_rtk])
        row += 1

        ttk.Label(frm, text="Output .mat (georef)").grid(row=row, column=0, sticky="w", pady=(10, 0))
        row += 1
        ttk.Entry(frm, textvariable=self.var_out, width=116).grid(row=row, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Opslaan als...", command=self._pick_out).grid(row=row, column=1, sticky="e")
        row += 1

        mode = ttk.LabelFrame(frm, text="Referentiemodus", padding=10)
        mode.grid(row=row, column=0, columnspan=2, sticky="we", pady=(12, 0))
        row += 1
        ttk.Radiobutton(mode, text=self.MODE_RTK, value=self.MODE_RTK, variable=self.var_mode, command=self._update_mode_ui).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode, text=self.MODE_MANUAL, value=self.MODE_MANUAL, variable=self.var_mode, command=self._update_mode_ui).grid(row=1, column=0, sticky="w", pady=(4, 0))

        opt = ttk.LabelFrame(frm, text="Opties tijd / kolommen", padding=10)
        opt.grid(row=row, column=0, columnspan=2, sticky="we", pady=(10, 0))
        row += 1

        ttk.Label(opt, text="M9 tijdshift (uren):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opt, textvariable=self.var_m9_shift, values=["-2", "-1", "0", "+1", "+2"], width=6, state="readonly").grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(opt, text="Match tolerantie (s):").grid(row=0, column=2, sticky="e", padx=(18, 0))
        e_tol = ttk.Entry(opt, textvariable=self.var_tol, width=10)
        e_tol.grid(row=0, column=3, sticky="w", padx=(6, 0))
        cb_local = ttk.Checkbutton(opt, text="RTK tijden: Europe/Brussels -> UTC", variable=self.var_rtk_local)
        cb_local.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(opt, text="Extra RTK time shift (s):").grid(row=1, column=2, sticky="e", padx=(18, 0), pady=(8, 0))
        e_rtk_shift = ttk.Entry(opt, textvariable=self.var_rtk_shift_s, width=10)
        e_rtk_shift.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        cb_swap = ttk.Checkbutton(opt, text="RTK export is N;E;H (swap N<->E)", variable=self.var_rtk_swap_NE)
        cb_swap.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(opt, text="M9 Summary.Track is N,E (swap bij in-/uitschrijven)", variable=self.var_m9_track_is_NE).grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Label(opt, text="Track rotatie (graden, klokwijs):").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_track_rotation_cw_deg, width=10).grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        self._rtk_controls.extend([e_tol, cb_local, e_rtk_shift, cb_swap])

        man = ttk.LabelFrame(frm, text="Manuele referentie", padding=10)
        man.grid(row=row, column=0, columnspan=2, sticky="we", pady=(10, 0))
        row += 1
        ttk.Label(man, text="1) Kies punt op gemeten track:").grid(row=0, column=0, sticky="w")
        b_pick_manual = ttk.Button(man, text="Klik punt op kaart/plot...", command=self._pick_manual_track_point)
        b_pick_manual.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(man, text="Gekozen index:").grid(row=0, column=2, sticky="e", padx=(18, 0))
        e_manual_idx = ttk.Entry(man, textvariable=self.var_manual_track_idx, width=12, state="readonly")
        e_manual_idx.grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(man, text="2) Coordinate type van invoer:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        cb_manual_mode = ttk.Combobox(
            man,
            textvariable=self.var_manual_coord_mode,
            values=[self.MANUAL_COORD_L72, self.MANUAL_COORD_UTM31, self.MANUAL_COORD_WGS84],
            width=30,
            state="readonly",
        )
        cb_manual_mode.grid(row=1, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(10, 0))
        cb_manual_mode.bind("<<ComboboxSelected>>", lambda _evt: self._update_manual_coord_labels())

        ttk.Label(man, text="3) Geef coordinaat van dit punt in:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Label(man, textvariable=self.var_manual_label_1).grid(row=2, column=2, sticky="e", pady=(10, 0))
        e_manual_E = ttk.Entry(man, textvariable=self.var_manual_E, width=14)
        e_manual_E.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(10, 0))
        ttk.Label(man, textvariable=self.var_manual_label_2).grid(row=2, column=4, sticky="e", padx=(12, 0), pady=(10, 0))
        e_manual_N = ttk.Entry(man, textvariable=self.var_manual_N, width=14)
        e_manual_N.grid(row=2, column=5, sticky="w", padx=(6, 0), pady=(10, 0))
        self._manual_controls.extend([b_pick_manual, e_manual_idx, cb_manual_mode, e_manual_E, e_manual_N])

        mapopt = ttk.LabelFrame(frm, text="Kaart na georeferering (Folium)", padding=10)
        mapopt.grid(row=row, column=0, columnspan=2, sticky="we", pady=(10, 0))
        row += 1
        ttk.Checkbutton(mapopt, text="Maak kaart (HTML) na succes", variable=self.var_make_map).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(mapopt, text="Open kaart automatisch in browser", variable=self.var_open_map).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(mapopt, text="Marker elke n punten:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(mapopt, textvariable=self.var_map_every_n_markers, width=8).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(mapopt, text="Lijn downsample (1=alles):").grid(row=1, column=2, sticky="e", padx=(18, 0), pady=(8, 0))
        ttk.Entry(mapopt, textvariable=self.var_map_every_n_line, width=8).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=2, sticky="we", pady=(12, 0))
        row += 1
        ttk.Button(btns, text="1) Lees M9", command=self._load_m9).pack(side="left")
        b_load_rtk = ttk.Button(btns, text="2) Lees RTK", command=self._load_rtk)
        b_load_rtk.pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="3) Georefereren & .mat opslaan", command=self._run).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Sluiten", command=self.destroy).pack(side="right")
        self._rtk_controls.append(b_load_rtk)

        pan = ttk.Panedwindow(frm, orient=tk.VERTICAL)
        pan.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        frm.grid_rowconfigure(row, weight=1)
        frm.grid_columnconfigure(0, weight=1)

        top = ttk.Frame(pan)
        bot = ttk.Frame(pan)
        pan.add(top, weight=3)
        pan.add(bot, weight=2)

        ttk.Label(top, text="RTK ankerpunten (kies 1 rij in RTK-modus):").pack(anchor="w")
        self.tree = ttk.Treeview(top, columns=("anchor_id", "idx", "name", "time_utc", "E", "N", "H"), show="headings", height=14)
        for c, w in [("anchor_id", 80), ("idx", 80), ("name", 280), ("time_utc", 220), ("E", 130), ("N", 130), ("H", 110)]:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)

        vsb = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

        ttk.Label(bot, text="Log:").pack(anchor="w")
        self.txt = tk.Text(bot, height=10, wrap="word")
        self.txt.pack(fill="both", expand=True)

    def _set_state(self, widgets: list[tk.Widget], enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in widgets:
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _update_mode_ui(self):
        is_rtk = self.var_mode.get() == self.MODE_RTK
        self._set_state(self._rtk_controls, is_rtk)
        self._set_state(self._manual_controls, not is_rtk)

    def _update_manual_coord_labels(self):
        mode = self.var_manual_coord_mode.get()
        if mode == self.MANUAL_COORD_WGS84:
            self.var_manual_label_1.set("Latitude (deg):")
            self.var_manual_label_2.set("Longitude (deg):")
        else:
            self.var_manual_label_1.set("E (m):")
            self.var_manual_label_2.set("N (m):")

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
            self._log("Lezen M9...")
            self._m9 = read_m9_mat(m9p, m9_hour_shift=shift, m9_track_is_NE=track_is_NE)
            self._manual_index = None
            self.var_manual_track_idx.set("")
            self._log(f"  OK: ensembles={len(self._m9.track_rel)} | {self._m9.time_utc.min()} -> {self._m9.time_utc.max()}")
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

            self._log("Lezen RTK...")
            rtk = read_carlson_rtk_points(rtkp, assume_local_brussels=local, swap_NE=swap)
            self._rtk_df = rtk
            self._log(f"  OK: punten={len(rtk)} | {rtk.time_utc.min()} -> {rtk.time_utc.max()}")
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
            messagebox.showinfo("OK", "RTK ingelezen.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")

    def _pick_manual_track_point(self):
        if self._m9 is None:
            self._load_m9()
        if self._m9 is None:
            return
        if not _PLOT_OK:
            messagebox.showerror("Dependency", "matplotlib niet beschikbaar. Installeer: pip install matplotlib")
            return

        try:
            picker = TrackPointPicker(self, self._m9)
            self.wait_window(picker)
            if picker.result_index is not None:
                self._manual_index = int(picker.result_index)
                self.var_manual_track_idx.set(str(self._manual_index))
                t = pd.to_datetime(self._m9.time_utc[self._manual_index]).tz_convert("UTC").isoformat()
                self._log(f"Manueel trackpunt: index={self._manual_index} | UTC={t}")
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
            if self._m9 is None:
                return

            outp = self.var_out.get().strip()
            if not outp:
                messagebox.showerror("Output", "Kies output .mat.")
                return

            ref_label = ""
            ref_E = np.nan
            ref_N = np.nan
            rot_cw = float(self.var_track_rotation_cw_deg.get())

            if self.var_mode.get() == self.MODE_RTK:
                if self._rtk_df is None:
                    self._load_rtk()
                if self._rtk_df is None:
                    return

                anchor_id = self._get_selected_anchor_id()
                if anchor_id is None:
                    messagebox.showerror("Ankerpunt", "Selecteer 1 ankerpunt (1 rij) in de RTK-tabel.")
                    return

                tol = float(self.var_tol.get())
                rtk_shift_s = float(self.var_rtk_shift_s.get())
                self._log(f"Georefereren RTK-modus met ankerpunt {anchor_id} (tol={tol}s, RTK shift={rtk_shift_s}s)...")
                new_track, info = georef_track_by_single_anchor(
                    m9=self._m9,
                    rtk=self._rtk_df,
                    anchor_anchor_id=anchor_id,
                    tol_s=tol,
                    rtk_time_shift_s=rtk_shift_s,
                    rotation_deg_cw=rot_cw,
                )
                self._log(f"  rotatie = {info['rotation_deg_cw']:.3f} graden CW")
                self._log("Offset berekend:")
                self._log(f"  dE = {info['offset_dE']:.3f} m | dN = {info['offset_dN']:.3f} m")
                self._log(f"  matched M9 idx = {info['matched_m9_index']} | dt = {info['dt_seconds']:.3f} s")

                ref_E = float(info["rtk_E"])
                ref_N = float(info["rtk_N"])
                ref_label = f"RTK anker id={info['anchor_id']} idx={info['anchor_idx']}"
            else:
                if self._manual_index is None:
                    idx_text = self.var_manual_track_idx.get().strip()
                    if idx_text:
                        self._manual_index = int(idx_text)
                if self._manual_index is None:
                    messagebox.showerror("Manueel", "Kies eerst een punt op de gemeten track.")
                    return

                v1 = _to_float(self.var_manual_E.get())
                v2 = _to_float(self.var_manual_N.get())
                if not np.isfinite(v1) or not np.isfinite(v2):
                    messagebox.showerror("Manueel", "Geef geldige coordinaatwaarden in.")
                    return

                mode = self.var_manual_coord_mode.get()
                ref_E, ref_N = convert_manual_reference_to_lambert72(v1, v2, mode)
                self._log(
                    f"Manuele input omgezet ({mode}) -> Lambert72: E={ref_E:.3f}, N={ref_N:.3f}"
                )
                self._log(f"Georefereren manueel met track index {self._manual_index} naar E={ref_E:.3f}, N={ref_N:.3f}...")
                new_track, info = georef_track_by_manual_reference(
                    m9=self._m9,
                    track_index=int(self._manual_index),
                    ref_E=float(ref_E),
                    ref_N=float(ref_N),
                    rotation_deg_cw=rot_cw,
                )
                self._log(f"  rotatie = {info['rotation_deg_cw']:.3f} graden CW")
                self._log("Offset berekend:")
                self._log(f"  dE = {info['offset_dE']:.3f} m | dN = {info['offset_dN']:.3f} m")
                self._log(f"  matched M9 idx = {info['matched_m9_index']}")
                ref_label = f"Manueel referentiepunt (M9 index {info['matched_m9_index']}, bron={mode})"

            self._log("Schrijven .mat (Summary.Track overschrijven)...")
            write_m9_mat_overwrite_track(self._m9, new_track, outp)
            self._log(f"  Saved: {outp}")

            if bool(self.var_make_map.get()):
                if not _MAP_OK:
                    self._log("WAARSCHUWING: folium/pyproj niet beschikbaar -> kaart niet gemaakt.")
                else:
                    out_html = str(Path(outp).with_suffix("")) + "_map.html"
                    self._log(f"Kaart maken: {out_html}")
                    make_map_html(
                        m9_times_utc=self._m9.time_utc,
                        track_EN=new_track,
                        ref_E=float(ref_E),
                        ref_N=float(ref_N),
                        ref_label=ref_label,
                        out_html=out_html,
                        every_n_markers=int(self.var_map_every_n_markers.get()),
                        every_n_line=int(self.var_map_every_n_line.get()),
                    )
                    self._log("  Kaart opgeslagen.")
                    if bool(self.var_open_map.get()):
                        webbrowser.open(Path(out_html).resolve().as_uri())

            messagebox.showinfo("OK", "Klaar. Nieuwe .mat geschreven.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
