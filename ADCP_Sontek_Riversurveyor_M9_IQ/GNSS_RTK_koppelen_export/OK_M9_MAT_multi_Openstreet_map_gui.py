#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lees meerdere M9 MAT-bestanden in en toon alle tracks op een OpenStreetMap-kaart.

Hover-info per marker:
- tijd (UTC)
- E/N
- lat/lon
- hoogte
- ensemble nummer
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

import numpy as np
import pandas as pd
import scipy.io as sio

try:
    import folium  # type: ignore
    from pyproj import Transformer  # type: ignore
    from folium.plugins import Fullscreen, MeasureControl, MousePosition  # type: ignore

    _MAP_OK = True
except Exception:
    folium = None
    Transformer = None
    Fullscreen = None
    MeasureControl = None
    MousePosition = None
    _MAP_OK = False


COORD_TRACK = "Summary.Track (E/N -> EPSG)"
COORD_GPS = "GPS (lat/lon)"
HEIGHT_AUTO = "Auto"
HEIGHT_GPS_ALT = "GPS.Altitude"
HEIGHT_SUMMARY_DEPTH = "Summary.Depth"
HEIGHT_BT_DEPTH = "BottomTrack.BT_Depth"


@dataclass
class MatTrackData:
    path: Path
    df: pd.DataFrame
    height_label: str
    note: str = ""


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


def _as_track_EN(track_obj: object | None, n: int, track_is_NE: bool) -> tuple[np.ndarray, np.ndarray]:
    if track_obj is None:
        return np.full(n, np.nan), np.full(n, np.nan)

    track = np.asarray(track_obj, dtype=float)
    if track.ndim != 2 or track.shape[1] < 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    track = track[:, :2]
    if track_is_NE:
        track = track[:, [1, 0]]

    e = _as_1d(track[:, 0], n)
    n_arr = _as_1d(track[:, 1], n)
    return e, n_arr


def _has_signal(arr: np.ndarray) -> bool:
    valid = np.isfinite(arr)
    if not np.any(valid):
        return False
    return float(np.nanmax(np.abs(arr[valid]))) > 1e-9


def _looks_relative_track(e: np.ndarray, n_arr: np.ndarray) -> bool:
    valid = np.isfinite(e) & np.isfinite(n_arr)
    if np.count_nonzero(valid) < 10:
        return False
    e_v = e[valid]
    n_v = n_arr[valid]
    med_abs = max(float(np.nanmedian(np.abs(e_v))), float(np.nanmedian(np.abs(n_v))))
    spread = max(float(np.nanmax(e_v) - np.nanmin(e_v)), float(np.nanmax(n_v) - np.nanmin(n_v)))
    return med_abs < 5000 and spread < 10000


def _pick_height(
    gps_alt: np.ndarray,
    summary_depth: np.ndarray,
    bt_depth: np.ndarray,
    source: str,
) -> tuple[np.ndarray, str]:
    if source == HEIGHT_GPS_ALT:
        return gps_alt, HEIGHT_GPS_ALT
    if source == HEIGHT_SUMMARY_DEPTH:
        return summary_depth, HEIGHT_SUMMARY_DEPTH
    if source == HEIGHT_BT_DEPTH:
        return bt_depth, HEIGHT_BT_DEPTH

    for label, arr in [
        (HEIGHT_GPS_ALT, gps_alt),
        (HEIGHT_SUMMARY_DEPTH, summary_depth),
        (HEIGHT_BT_DEPTH, bt_depth),
    ]:
        if _has_signal(arr):
            return arr, label
    return np.full_like(gps_alt, np.nan), "geen hoogte gevonden"


def read_mat_for_map(
    mat_path: str | Path,
    m9_hour_shift: int = -1,
    track_is_NE: bool = False,
    coord_mode: str = COORD_TRACK,
    track_epsg: int = 31370,
    height_source: str = HEIGHT_AUTO,
) -> MatTrackData:
    if not _MAP_OK:
        raise RuntimeError("folium/pyproj niet beschikbaar. Installeer: pip install folium pyproj")

    p = Path(mat_path)
    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)

    if "System" not in mat:
        raise ValueError(f"{p.name}: structuur 'System' niet gevonden.")
    if "Summary" not in mat:
        raise ValueError(f"{p.name}: structuur 'Summary' niet gevonden.")

    sys_obj = mat["System"]
    sum_obj = mat["Summary"]
    gps_obj = mat.get("GPS")
    bt_obj = mat.get("BottomTrack")

    if not hasattr(sys_obj, "Time"):
        raise ValueError(f"{p.name}: System.Time ontbreekt.")

    time_sec = np.asarray(sys_obj.Time, dtype=float).reshape(-1)
    n = len(time_sec)
    if n == 0:
        raise ValueError(f"{p.name}: lege tijdreeks.")

    time_utc = _m9_time_to_utc(time_sec)
    if int(m9_hour_shift) != 0:
        time_utc = time_utc + pd.Timedelta(hours=int(m9_hour_shift))

    e_track, n_track = _as_track_EN(getattr(sum_obj, "Track", None), n, track_is_NE)

    gps_lat = _as_1d(getattr(gps_obj, "Latitude", None) if gps_obj is not None else None, n)
    gps_lon = _as_1d(getattr(gps_obj, "Longitude", None) if gps_obj is not None else None, n)
    zero_mask = (gps_lat == 0.0) & (gps_lon == 0.0)
    gps_lat[zero_mask] = np.nan
    gps_lon[zero_mask] = np.nan

    gps_alt = _as_1d(getattr(gps_obj, "Altitude", None) if gps_obj is not None else None, n)
    summary_depth = _as_1d(getattr(sum_obj, "Depth", None), n)
    bt_depth = _as_1d(getattr(bt_obj, "BT_Depth", None) if bt_obj is not None else None, n)
    h_arr, h_label = _pick_height(gps_alt, summary_depth, bt_depth, height_source)

    tr_to_wgs = Transformer.from_crs(f"EPSG:{int(track_epsg)}", "EPSG:4326", always_xy=True)
    lon_from_track, lat_from_track = tr_to_wgs.transform(e_track, n_track)
    lat_from_track = np.asarray(lat_from_track, dtype=float)
    lon_from_track = np.asarray(lon_from_track, dtype=float)

    note = ""
    if coord_mode == COORD_GPS:
        lat = gps_lat.copy()
        lon = gps_lon.copy()
        if np.count_nonzero(np.isfinite(lat) & np.isfinite(lon)) < 5:
            lat = lat_from_track
            lon = lon_from_track
            note = "GPS lat/lon ontbreekt; fallback naar Summary.Track."
    else:
        lat = lat_from_track
        lon = lon_from_track
        if _looks_relative_track(e_track, n_track):
            note = "Track lijkt relatief (niet gegeorefereerd)."

    df = pd.DataFrame(
        {
            "ensemble": np.arange(1, n + 1, dtype=int),
            "time_utc": pd.to_datetime(time_utc, utc=True),
            "E": e_track,
            "N": n_track,
            "lat": lat,
            "lon": lon,
            "H": h_arr,
        }
    )
    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].copy()
    if df.empty:
        raise ValueError(f"{p.name}: geen geldige kaartcoordinaten gevonden.")

    return MatTrackData(path=p, df=df, height_label=h_label, note=note)


def build_map_html(
    tracks: list[MatTrackData],
    out_html: str | Path,
    marker_step: int = 15,
    line_step: int = 1,
) -> None:
    if not tracks:
        raise ValueError("Geen tracks om te plotten.")

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    all_lat = np.concatenate([t.df["lat"].to_numpy(dtype=float) for t in tracks])
    all_lon = np.concatenate([t.df["lon"].to_numpy(dtype=float) for t in tracks])
    center = [float(np.nanmedian(all_lat)), float(np.nanmedian(all_lon))]

    m = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap", control_scale=True)

    if Fullscreen is not None:
        Fullscreen(position="topleft", title="Fullscreen", title_cancel="Sluit fullscreen").add_to(m)
    if MeasureControl is not None:
        MeasureControl(
            position="topleft",
            primary_length_unit="meters",
            secondary_length_unit="kilometers",
        ).add_to(m)
    if MousePosition is not None:
        MousePosition(
            position="topright",
            separator=" | ",
            prefix="WGS84",
            lat_formatter="function(num) {return L.Util.formatNum(num, 7);}",
            lng_formatter="function(num) {return L.Util.formatNum(num, 7);}",
        ).add_to(m)

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#7f7f7f",
    ]

    step_mark = max(1, int(marker_step))
    step_line = max(1, int(line_step))

    for i, track in enumerate(tracks):
        color = colors[i % len(colors)]
        df = track.df.reset_index(drop=True)
        grp = folium.FeatureGroup(name=f"{track.path.name} ({len(df)} pt)", show=True)

        line_xy = df.loc[::step_line, ["lat", "lon"]].to_numpy().tolist()
        folium.PolyLine(
            line_xy,
            color=color,
            weight=4,
            opacity=0.9,
            tooltip=f"{track.path.name} ({len(df)} punten)",
        ).add_to(grp)

        for _, row in df.loc[::step_mark].iterrows():
            ts = pd.Timestamp(row["time_utc"]).tz_convert("UTC").isoformat()
            if np.isfinite(float(row["H"])):
                h_txt = f"{float(row['H']):.3f}"
            else:
                h_txt = "n/a"

            tip = (
                f"<b>{track.path.name}</b><br>"
                f"ensemble={int(row['ensemble'])}<br>"
                f"{ts}<br>"
                f"E={float(row['E']):.3f} N={float(row['N']):.3f}<br>"
                f"lat={float(row['lat']):.7f} lon={float(row['lon']):.7f}<br>"
                f"{track.height_label}={h_txt}"
            )
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                opacity=0.95,
                tooltip=folium.Tooltip(tip, sticky=True),
            ).add_to(grp)

        grp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M9 MAT multi-kaart (OpenStreetMap)")
        self.geometry("1180x780")

        self.var_out = tk.StringVar(value=str(Path.cwd() / "m9_multi_map.html"))
        self.var_m9_shift = tk.StringVar(value="-1")
        self.var_track_is_NE = tk.BooleanVar(value=False)
        self.var_coord_mode = tk.StringVar(value=COORD_TRACK)
        self.var_epsg = tk.StringVar(value="31370")
        self.var_height_source = tk.StringVar(value=HEIGHT_AUTO)
        self.var_marker_step = tk.IntVar(value=15)
        self.var_line_step = tk.IntVar(value=1)
        self.var_open_map = tk.BooleanVar(value=True)

        self._mat_files: list[Path] = []
        self._build()

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        files_box = ttk.LabelFrame(frm, text="MAT-bestanden", padding=10)
        files_box.grid(row=0, column=0, sticky="nsew")
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(3, weight=1)

        btn_row = ttk.Frame(files_box)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Toevoegen...", command=self._add_files).pack(side="left")
        ttk.Button(btn_row, text="Verwijder selectie", command=self._remove_selected).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Leegmaken", command=self._clear_files).pack(side="left", padx=(8, 0))

        list_wrap = ttk.Frame(files_box)
        list_wrap.pack(fill="both", expand=True, pady=(8, 0))
        self.listbox = tk.Listbox(list_wrap, height=14, selectmode=tk.EXTENDED)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        opt = ttk.LabelFrame(frm, text="Opties", padding=10)
        opt.grid(row=1, column=0, sticky="we", pady=(10, 0))

        ttk.Label(opt, text="M9 tijdshift (uren):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            opt,
            textvariable=self.var_m9_shift,
            values=["-2", "-1", "0", "+1", "+2"],
            width=6,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Checkbutton(opt, text="Summary.Track is N,E (swap)", variable=self.var_track_is_NE).grid(
            row=0, column=2, sticky="w", padx=(14, 0)
        )

        ttk.Label(opt, text="Coordinaatbron:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            opt,
            textvariable=self.var_coord_mode,
            values=[COORD_TRACK, COORD_GPS],
            state="readonly",
            width=32,
        ).grid(row=1, column=1, columnspan=2, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Track EPSG:").grid(row=1, column=3, sticky="e", padx=(16, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_epsg, width=10).grid(row=1, column=4, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Hoogtebron:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            opt,
            textvariable=self.var_height_source,
            values=[HEIGHT_AUTO, HEIGHT_GPS_ALT, HEIGHT_SUMMARY_DEPTH, HEIGHT_BT_DEPTH],
            state="readonly",
            width=24,
        ).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Marker elke n punten:").grid(row=2, column=2, sticky="e", padx=(16, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_marker_step, width=10).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Lijn downsample:").grid(row=2, column=4, sticky="e", padx=(16, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_line_step, width=10).grid(row=2, column=5, sticky="w", padx=(6, 0), pady=(8, 0))

        out_box = ttk.LabelFrame(frm, text="Output kaart", padding=10)
        out_box.grid(row=2, column=0, sticky="we", pady=(10, 0))
        ttk.Entry(out_box, textvariable=self.var_out, width=112).grid(row=0, column=0, sticky="we")
        ttk.Button(out_box, text="Opslaan als...", command=self._pick_out).grid(row=0, column=1, padx=(8, 0))
        ttk.Checkbutton(out_box, text="Open kaart automatisch", variable=self.var_open_map).grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        out_box.grid_columnconfigure(0, weight=1)

        bot = ttk.Frame(frm)
        bot.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        bot.grid_columnconfigure(0, weight=1)
        bot.grid_rowconfigure(1, weight=1)

        btns = ttk.Frame(bot)
        btns.grid(row=0, column=0, sticky="we")
        ttk.Button(btns, text="Kaart maken", command=self._run).pack(side="left")
        ttk.Button(btns, text="Sluiten", command=self.destroy).pack(side="right")

        ttk.Label(bot, text="Log").grid(row=1, column=0, sticky="w")
        self.txt = tk.Text(bot, height=14, wrap="word")
        self.txt.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

    def _log(self, s: str) -> None:
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, "end")
        for p in self._mat_files:
            self.listbox.insert("end", str(p))

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Kies MAT-bestanden", filetypes=[("MAT", "*.mat"), ("All", "*.*")])
        if not paths:
            return
        existing = {str(p).lower() for p in self._mat_files}
        for p in paths:
            if str(p).lower() not in existing:
                self._mat_files.append(Path(p))
        self._refresh_listbox()

    def _remove_selected(self) -> None:
        sel = list(self.listbox.curselection())
        if not sel:
            return
        sel_set = set(sel)
        self._mat_files = [p for i, p in enumerate(self._mat_files) if i not in sel_set]
        self._refresh_listbox()

    def _clear_files(self) -> None:
        self._mat_files = []
        self._refresh_listbox()

    def _pick_out(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Kies output HTML",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("All", "*.*")],
        )
        if p:
            self.var_out.set(p)

    def _run(self) -> None:
        try:
            if not _MAP_OK:
                messagebox.showerror("Dependency", "folium/pyproj niet beschikbaar. Installeer: pip install folium pyproj")
                return
            if not self._mat_files:
                messagebox.showerror("Input", "Voeg eerst een of meer MAT-bestanden toe.")
                return

            out_html = self.var_out.get().strip()
            if not out_html:
                messagebox.showerror("Output", "Geef een output HTML-bestand op.")
                return

            m9_shift = int(self.var_m9_shift.get())
            track_is_ne = bool(self.var_track_is_NE.get())
            coord_mode = self.var_coord_mode.get()
            epsg = int(self.var_epsg.get())
            height_source = self.var_height_source.get()
            marker_step = max(1, int(self.var_marker_step.get()))
            line_step = max(1, int(self.var_line_step.get()))

            tracks: list[MatTrackData] = []
            failed: list[str] = []

            self._log("Inlezen MAT-bestanden...")
            for p in self._mat_files:
                try:
                    t = read_mat_for_map(
                        p,
                        m9_hour_shift=m9_shift,
                        track_is_NE=track_is_ne,
                        coord_mode=coord_mode,
                        track_epsg=epsg,
                        height_source=height_source,
                    )
                    tracks.append(t)
                    self._log(f"  OK: {p.name} | punten={len(t.df)} | hoogtebron={t.height_label}")
                    if t.note:
                        self._log(f"      note: {t.note}")
                except Exception as e:
                    failed.append(f"{p.name}: {e}")
                    self._log(f"  ERROR: {p.name} -> {e}")

            if not tracks:
                raise ValueError("Geen geldig MAT-bestand kunnen verwerken.")

            self._log("Kaart bouwen...")
            build_map_html(tracks, out_html=out_html, marker_step=marker_step, line_step=line_step)
            self._log(f"Klaar: {out_html}")

            if bool(self.var_open_map.get()):
                webbrowser.open(Path(out_html).resolve().as_uri())

            if failed:
                messagebox.showwarning(
                    "Klaar met waarschuwingen",
                    f"Kaart gemaakt met {len(tracks)} bestand(en).\n"
                    f"{len(failed)} bestand(en) faalden.\nZie log voor details.",
                )
            else:
                messagebox.showinfo("Klaar", f"Kaart gemaakt met {len(tracks)} bestand(en).")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
