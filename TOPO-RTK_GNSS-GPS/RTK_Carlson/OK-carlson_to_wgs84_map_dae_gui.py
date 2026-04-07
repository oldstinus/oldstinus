#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import html
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
import tkinter as tk
from pyproj import Transformer
from tkinter import filedialog, messagebox, ttk

NONE = "(none)"


@dataclass
class Guess:
    meta1: Optional[str] = None
    meta2: Optional[str] = None
    e: Optional[str] = None
    n: Optional[str] = None
    h: Optional[str] = None
    t: Optional[str] = None
    d: Optional[str] = None
    profile: str = "Onbekend"
    note: str = ""


@dataclass
class DaeOpts:
    cube: float = 0.75
    ribbon: bool = True
    ribbon_w: float = 0.30
    use_h: bool = False
    every: int = 1


def s(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() == "nan" else text


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def is_num(v: object) -> bool:
    try:
        float(s(v).replace(",", "."))
        return bool(s(v))
    except ValueError:
        return False


def sniff_delim(path: str | Path, default: str = ";") -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()][:50]
    candidates = ["\t", ";", "|", ","]
    best_delim: Optional[str] = None
    best_score: tuple[float, float, float] | None = None

    # Carlson-exporten bevatten vaak decimale komma's; daarom eerst de meest consistente veldscheider kiezen.
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
    hdr = {"pt id", "id", "desc", "x", "y", "z", "e", "n", "time", "date", "datum"}
    if {x.lower() for x in rows[0]} & hdr:
        return True
    a = sum(is_num(x) for x in rows[0]) / max(1, len(rows[0]))
    b = sum(is_num(x) for x in rows[1]) / max(1, len(rows[1]))
    return a < 0.35 and b >= 0.35


def read_table(path: str | Path, delim: str, header_mode: str) -> tuple[pd.DataFrame, bool, str]:
    if delim == "auto":
        delim = sniff_delim(path)
    has_header = detect_header(path, delim) if header_mode == "auto" else header_mode == "yes"
    df = pd.read_csv(path, sep=delim, header=0 if has_header else None, dtype=str, keep_default_na=False, engine="python")
    df.columns = [s(c) or f"col{i+1}" for i, c in enumerate(df.columns)] if has_header else [f"col{i+1}" for i in range(df.shape[1])]
    for c in df.columns:
        df[c] = df[c].map(s)
    return df, has_header, delim


def guess_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def num_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if float(numeric(df[c]).notna().mean()) >= 0.75]


def guess_cols(df: pd.DataFrame, has_header: bool) -> Guess:
    g = Guess(
        meta1=guess_col(df, ["pt id", "point id", "id", "name"]),
        meta2=guess_col(df, ["desc", "description", "omschrijving", "groep"]),
        e=guess_col(df, ["e", "east", "easting", "x"]),
        n=guess_col(df, ["n", "north", "northing", "y"]),
        h=guess_col(df, ["h", "height", "z", "hoogte"]),
        t=guess_col(df, ["time", "tijd", "timestamp"]),
        d=guess_col(df, ["date", "datum"]),
        profile="Header-profiel",
    )
    if g.e or g.n or g.meta1:
        return g
    cols = list(df.columns)
    nums = num_cols(df)
    if not has_header and len(cols) >= 5 and all(c not in nums for c in cols[:2]) and all(c in nums for c in cols[2:5]):
        return Guess(cols[0], cols[1], cols[3], cols[2], cols[4], cols[7] if len(cols) >= 8 else None, cols[8] if len(cols) >= 9 else None, "Carlson TXT zonder header", "Metadata 1-2, N in kolom 3, E in kolom 4.")
    if len(nums) >= 2:
        g.e, g.n = nums[0], nums[1]
        g.profile = "Generieke detectie"
        g.note = "Geen duidelijk profiel gevonden; eerste twee numerieke kolommen gekozen."
    if cols:
        g.meta1 = g.meta1 or cols[0]
    if len(cols) > 1:
        g.meta2 = g.meta2 or cols[1]
    return g


def label_for(row: pd.Series, idx: int) -> str:
    a, b = s(row.get("META1", "")), s(row.get("META2", ""))
    if a and b and a != b:
        return f"{a} | {b}"
    return a or b or f"Punt {idx + 1}"


def details_for(row: pd.Series, idx: int) -> list[tuple[str, str]]:
    out = [("Index", str(idx + 1))]
    if s(row.get("META1", "")):
        out.append(("Metadata 1", s(row["META1"])))
    if s(row.get("META2", "")):
        out.append(("Metadata 2", s(row["META2"])))
    out.append(("Easting", f"{float(row['E']):.3f}"))
    out.append(("Northing", f"{float(row['N']):.3f}"))
    if not pd.isna(row.get("H", np.nan)):
        out.append(("Hoogte", f"{float(row['H']):.3f}"))
    out.append(("Lat", f"{float(row['lat']):.8f}"))
    out.append(("Lon", f"{float(row['lon']):.8f}"))
    if s(row.get("TIME", "")):
        out.append(("Tijd", s(row["TIME"])))
    if s(row.get("DATE", "")):
        out.append(("Datum", s(row["DATE"])))
    return out


def popup_html(row: pd.Series, idx: int) -> str:
    return "<br>".join(f"<b>{html.escape(k)}:</b> {html.escape(v)}" for k, v in details_for(row, idx))


def tooltip(row: pd.Series, idx: int) -> str:
    return f"{label_for(row, idx)} | E={float(row['E']):.3f} N={float(row['N']):.3f}"


DEFAULT_MAP_TILES = "CartoDB positron"


def make_map(df: pd.DataFrame, out_html: str | Path, every: int) -> None:
    # Avoid direct OpenStreetMap tile requests from local file:// HTML, which OSM blocks without a valid Referer.
    m = folium.Map(location=[float(df["lat"].median()), float(df["lon"].median())], zoom_start=18, tiles=DEFAULT_MAP_TILES)
    pts = df[["lat", "lon"]].astype(float).values.tolist()
    if len(pts) >= 2:
        folium.PolyLine(pts, weight=4, color="#d95f02", opacity=0.9).add_to(m)
    for idx, row in df.iloc[:: max(1, int(every))].iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=4,
            color="#b73c00",
            fill=True,
            fill_color="#ff7f0e",
            fill_opacity=0.95,
            tooltip=tooltip(row, int(idx)),
            popup=folium.Popup(popup_html(row, int(idx)), max_width=500),
        ).add_to(m)
    for idx, tag in ((0, "START"), (len(df) - 1, "EINDE")):
        row = df.iloc[idx]
        folium.Marker(
            [float(row["lat"]), float(row["lon"])],
            tooltip=f"{tag} | {tooltip(row, idx)}",
            popup=folium.Popup(f"<b>{tag}</b><br>{popup_html(row, idx)}", max_width=500),
        ).add_to(m)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def dae_material() -> str:
    return """
  <library_effects>
    <effect id="track-effect"><profile_COMMON><technique sid="common"><lambert>
      <ambient><color>0.20 0.12 0.04 1</color></ambient>
      <diffuse><color>0.96 0.49 0.08 1</color></diffuse>
    </lambert></technique></profile_COMMON></effect>
  </library_effects>
  <library_materials>
    <material id="track-material" name="track-material"><instance_effect url="#track-effect"/></material>
  </library_materials>
"""


def write_dae(e: np.ndarray, n: np.ndarray, h: Optional[np.ndarray], out_dae: str | Path, opts: DaeOpts) -> tuple[float, float, float]:
    step = max(1, int(opts.every))
    e, n = e[::step], n[::step]
    if h is not None:
        h = h[::step]
    e0, n0 = float(e[0]), float(n[0])
    h0 = float(h[0]) if h is not None else 0.0
    x, y = e - e0, n - n0
    z = (h - h0) if (opts.use_h and h is not None) else np.zeros_like(x)
    s2 = float(opts.cube) / 2.0
    cube = np.array([[-s2,-s2,-s2],[s2,-s2,-s2],[s2,s2,-s2],[-s2,s2,-s2],[-s2,-s2,s2],[s2,-s2,s2],[s2,s2,s2],[-s2,s2,s2]], dtype=float)
    faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], dtype=int)
    verts, tris, off = [], [], 0
    for xi, yi, zi in zip(x, y, z):
        verts.append(cube + np.array([xi, yi, zi], dtype=float))
        tris.append(faces + off)
        off += 8
    verts = np.vstack(verts) if verts else np.zeros((0, 3), dtype=float)
    tris = np.vstack(tris) if tris else np.zeros((0, 3), dtype=int)
    if opts.ribbon and len(x) >= 2:
        rverts, rtris, halfw = [], [], float(opts.ribbon_w) / 2.0
        for i in range(len(x) - 1):
            p0, p1 = np.array([x[i], y[i], z[i]]), np.array([x[i+1], y[i+1], z[i+1]])
            dx, dy = (p1 - p0)[0], (p1 - p0)[1]
            L = math.hypot(dx, dy)
            if L < 1e-9:
                continue
            offv = np.array([-dy / L * halfw, dx / L * halfw, 0.0], dtype=float)
            rverts.append(np.vstack([p0-offv, p0+offv, p1+offv, p1-offv]))
            rtris.append(np.array([[0,1,2],[0,2,3]], dtype=int) + verts.shape[0] + len(rverts[:-1]) * 4)
        if rverts:
            verts = np.vstack([verts, np.vstack(rverts)])
            tris = np.vstack([tris, np.vstack(rtris)])
    pos = " ".join(f"{v:.6f}" for v in verts.reshape(-1))
    tri = " ".join(" ".join(str(int(v)) for v in t) for t in tris)
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><contributor><authoring_tool>OK-carlson_to_wgs84_map_dae_gui.py</authoring_tool></contributor><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
{dae_material()}
  <library_geometries><geometry id="trackGeom" name="CarlsonTrack"><mesh>
    <source id="trackGeom-positions"><float_array id="trackGeom-positions-array" count="{verts.size}">{pos}</float_array><technique_common><accessor source="#trackGeom-positions-array" count="{verts.shape[0]}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
    <vertices id="trackGeom-vertices"><input semantic="POSITION" source="#trackGeom-positions"/></vertices>
    <triangles material="track-symbol" count="{tris.shape[0]}"><input semantic="VERTEX" source="#trackGeom-vertices" offset="0"/><p>{tri}</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="Scene" name="Scene"><node id="TrackNode" name="TrackNode"><instance_geometry url="#trackGeom"><bind_material><technique_common><instance_material symbol="track-symbol" target="#track-material"/></technique_common></bind_material></instance_geometry></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
"""
    out_dae = Path(out_dae)
    out_dae.parent.mkdir(parents=True, exist_ok=True)
    out_dae.write_text(xml, encoding="utf-8")
    return e0, n0, h0


def cdata(text: str) -> str:
    return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def kml_desc(row: pd.Series, idx: int) -> str:
    return cdata("<br>".join(f"<b>{html.escape(k)}:</b> {html.escape(v)}" for k, v in details_for(row, idx)))


def point_coord(row: pd.Series, use_h: bool) -> str:
    alt = float(row["H"]) if use_h and not pd.isna(row.get("H", np.nan)) else 0.0
    return f"{float(row['lon']):.10f},{float(row['lat']):.10f},{alt:.3f}"


def write_kml(df: pd.DataFrame, dae_name: str, out_kml: str | Path, every: int, use_h: bool, lift: float) -> None:
    line = " ".join(point_coord(row, use_h) for _, row in df.iterrows())
    points = []
    for idx, (_, row) in enumerate(df.iloc[:: max(1, int(every))].iterrows()):
        real = idx * max(1, int(every))
        points.append(f"""
    <Placemark><name>{html.escape(label_for(row, real))}</name><description>{kml_desc(row, real)}</description>
      <Point><altitudeMode>{"absolute" if use_h else "clampToGround"}</altitudeMode><coordinates>{point_coord(row, use_h)}</coordinates></Point>
    </Placemark>""")
    first = df.iloc[0]
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Style id="track-line"><LineStyle><color>ff1478ff</color><width>4</width></LineStyle></Style>
  <Folder><name>Tracklijn</name><Placemark><name>Track</name><styleUrl>#track-line</styleUrl>
    <LineString><tessellate>1</tessellate><altitudeMode>{"absolute" if use_h else "clampToGround"}</altitudeMode><coordinates>{line}</coordinates></LineString>
  </Placemark></Folder>
  <Folder><name>Punten</name>{''.join(points)}</Folder>
  <Folder><name>Model</name><Placemark><name>DAE-model</name>
    <Model><altitudeMode>relativeToGround</altitudeMode><Location><longitude>{float(first['lon']):.10f}</longitude><latitude>{float(first['lat']):.10f}</latitude><altitude>{float(lift):.3f}</altitude></Location>
    <Orientation><heading>0</heading><tilt>0</tilt><roll>0</roll></Orientation><Scale><x>1</x><y>1</y><z>1</z></Scale><Link><href>{html.escape(dae_name)}</href></Link></Model>
  </Placemark></Folder>
</Document></kml>"""
    out_kml = Path(out_kml)
    out_kml.parent.mkdir(parents=True, exist_ok=True)
    out_kml.write_text(kml, encoding="utf-8")


def export_kmz(kmz_path: str | Path, kml_path: str | Path, dae_path: str | Path) -> None:
    kmz_path = Path(kmz_path)
    kmz_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(kml_path, arcname="doc.kml")
        zf.write(dae_path, arcname=Path(dae_path).name)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Carlson -> WGS84 kaart + DAE/KMZ export")
        self.geometry("1180x780")
        self.df: Optional[pd.DataFrame] = None
        self.df_proj: Optional[pd.DataFrame] = None
        self.v_in = tk.StringVar()
        self.v_out = tk.StringVar()
        self.v_delim = tk.StringVar(value="auto")
        self.v_header = tk.StringVar(value="auto")
        self.v_crs = tk.StringVar(value="EPSG:31370")
        self.v_meta1 = tk.StringVar(value=NONE)
        self.v_meta2 = tk.StringVar(value=NONE)
        self.v_e = tk.StringVar()
        self.v_n = tk.StringVar()
        self.v_h = tk.StringVar(value=NONE)
        self.v_t = tk.StringVar(value=NONE)
        self.v_d = tk.StringVar(value=NONE)
        self.v_map_every = tk.IntVar(value=1)
        self.v_dae_every = tk.IntVar(value=1)
        self.v_cube = tk.DoubleVar(value=0.75)
        self.v_ribbon = tk.BooleanVar(value=True)
        self.v_ribbon_w = tk.DoubleVar(value=0.30)
        self.v_use_h = tk.BooleanVar(value=False)
        self.v_lift = tk.DoubleVar(value=1.5)
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="Carlson data (CSV/TXT)").grid(row=0, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.v_in, width=108).grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(root, text="Bladeren...", command=self._pick_in).grid(row=1, column=1)
        ttk.Label(root, text="Output basisnaam (zonder extensie)").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(root, textvariable=self.v_out, width=108).grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(root, text="Kies...", command=self._pick_out).grid(row=3, column=1)
        box = ttk.LabelFrame(root, text="Inlezen", padding=10)
        box.grid(row=4, column=0, columnspan=2, sticky="we", pady=(12, 0))
        ttk.Label(box, text="Delimiter:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.v_delim, values=["auto", ";", ",", "\\t", "|"], width=8, state="readonly").grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(box, text="Header:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        ttk.Combobox(box, textvariable=self.v_header, values=["auto", "yes", "no"], width=8, state="readonly").grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(box, text="Input CRS:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(box, textvariable=self.v_crs, values=["EPSG:31370", "EPSG:25831", "EPSG:32631", "EPSG:4326"], width=18, state="readonly").grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Button(box, text="Lees en detecteer kolommen", command=self._load).grid(row=2, column=0, sticky="w", pady=(10, 0))
        cols = ttk.LabelFrame(root, text="Kolommen", padding=10)
        cols.grid(row=5, column=0, columnspan=2, sticky="we", pady=(10, 0))
        self.cb_meta1 = ttk.Combobox(cols, textvariable=self.v_meta1, values=[NONE], width=28, state="readonly"); self.cb_meta1.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.cb_meta2 = ttk.Combobox(cols, textvariable=self.v_meta2, values=[NONE], width=28, state="readonly"); self.cb_meta2.grid(row=0, column=3, sticky="w", padx=(6, 0))
        self.cb_e = ttk.Combobox(cols, textvariable=self.v_e, values=[], width=28, state="readonly"); self.cb_e.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_n = ttk.Combobox(cols, textvariable=self.v_n, values=[], width=28, state="readonly"); self.cb_n.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_h = ttk.Combobox(cols, textvariable=self.v_h, values=[NONE], width=28, state="readonly"); self.cb_h.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_t = ttk.Combobox(cols, textvariable=self.v_t, values=[NONE], width=28, state="readonly"); self.cb_t.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self.cb_d = ttk.Combobox(cols, textvariable=self.v_d, values=[NONE], width=28, state="readonly"); self.cb_d.grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(cols, text="Metadata 1:").grid(row=0, column=0, sticky="w"); ttk.Label(cols, text="Metadata 2:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Label(cols, text="Easting (E):").grid(row=1, column=0, sticky="w", pady=(8, 0)); ttk.Label(cols, text="Northing (N):").grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Label(cols, text="Hoogte:").grid(row=2, column=0, sticky="w", pady=(8, 0)); ttk.Label(cols, text="Tijd:").grid(row=2, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Label(cols, text="Datum:").grid(row=3, column=0, sticky="w", pady=(8, 0)); ttk.Label(cols, text="Kaart/KML: toon elke n punten:").grid(row=3, column=2, sticky="w", padx=(20, 0), pady=(8, 0))
        ttk.Entry(cols, textvariable=self.v_map_every, width=8).grid(row=3, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        dae = ttk.LabelFrame(root, text="DAE/KMZ export", padding=10)
        dae.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(dae, text="DAE sample elke n punten:").grid(row=0, column=0, sticky="w"); ttk.Entry(dae, textvariable=self.v_dae_every, width=8).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(dae, text="Marker cube size (m):").grid(row=0, column=2, sticky="w", padx=(20, 0)); ttk.Entry(dae, textvariable=self.v_cube, width=8).grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Checkbutton(dae, text="Pad-ribbon toevoegen", variable=self.v_ribbon).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(dae, text="Ribbon breedte (m):").grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(8, 0)); ttk.Entry(dae, textvariable=self.v_ribbon_w, width=8).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Checkbutton(dae, text="Gebruik hoogteverschillen in model-Z", variable=self.v_use_h).grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(dae, text="Model boven maaiveld (m):").grid(row=2, column=2, sticky="w", padx=(20, 0), pady=(8, 0)); ttk.Entry(dae, textvariable=self.v_lift, width=8).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        act = ttk.Frame(root); act.grid(row=7, column=0, columnspan=2, sticky="we", pady=(12, 0))
        ttk.Button(act, text="Projecteer -> WGS84", command=self._project).pack(side="left")
        ttk.Button(act, text="Maak kaart (HTML)", command=self._map).pack(side="left", padx=(8, 0))
        ttk.Button(act, text="Export DAE", command=self._dae).pack(side="left", padx=(8, 0))
        ttk.Button(act, text="Export KMZ (KML + DAE)", command=self._kmz).pack(side="left", padx=(8, 0))
        ttk.Button(act, text="Sluiten", command=self.destroy).pack(side="right")
        self.txt = tk.Text(root, height=16, wrap="word"); self.txt.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        root.grid_columnconfigure(0, weight=1); root.grid_rowconfigure(8, weight=1)

    def _log(self, msg: str) -> None:
        self.txt.insert("end", msg + "\n"); self.txt.see("end"); self.update_idletasks()

    def _pick_in(self) -> None:
        p = filedialog.askopenfilename(title="Selecteer Carlson CSV/TXT", filetypes=[("Text/CSV", "*.txt *.csv *.dat *.log"), ("All files", "*.*")])
        if p:
            self.v_in.set(p)
            if not self.v_out.get():
                self.v_out.set(str(Path(p).with_suffix("")) + "_wgs84")

    def _pick_out(self) -> None:
        p = filedialog.asksaveasfilename(title="Kies output basisnaam", defaultextension="")
        if p:
            self.v_out.set(str(Path(p).with_suffix("")))

    def _sel(self, val: str) -> Optional[str]:
        val = val.strip()
        return None if not val or val == NONE else val

    def _load(self) -> None:
        inp = self.v_in.get().strip()
        if not inp:
            messagebox.showerror("Fout", "Kies eerst een inputbestand."); return
        delim = "\t" if self.v_delim.get() == "\\t" else self.v_delim.get()
        try:
            self._log("Lezen bestand...")
            self.df, has_header, used = read_table(inp, "auto" if self.v_delim.get() == "auto" else delim, self.v_header.get().strip())
            self.df_proj = None
            self._log(f"  Rijen: {len(self.df)}, kolommen: {len(self.df.columns)}")
            self._log(f"  Delimiter: {repr(used)} | Header: {'ja' if has_header else 'nee'}")
            self._log(f"  Kolommen: {list(self.df.columns)}")
            g = guess_cols(self.df, has_header)
            values = list(self.df.columns); meta_values = [NONE] + values
            for cb, vals in ((self.cb_meta1, meta_values), (self.cb_meta2, meta_values), (self.cb_e, values), (self.cb_n, values), (self.cb_h, meta_values), (self.cb_t, meta_values), (self.cb_d, meta_values)):
                cb["values"] = vals
            self.v_meta1.set(g.meta1 or NONE); self.v_meta2.set(g.meta2 or NONE); self.v_e.set(g.e or ""); self.v_n.set(g.n or ""); self.v_h.set(g.h or NONE); self.v_t.set(g.t or NONE); self.v_d.set(g.d or NONE)
            self._log(f"  Profiel: {g.profile}")
            if g.note: self._log(f"  {g.note}")
            if len(self.df): self._log("  Eerste rij: " + " | ".join(f"{c}={s(self.df.iloc[0][c])}" for c in self.df.columns[:9]))
            messagebox.showinfo("OK", "Bestand ingelezen. Controleer de kolommen en klik daarna op 'Projecteer -> WGS84'.")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc)); self._log(f"ERROR: {exc}")

    def _project(self) -> None:
        try:
            if self.df is None:
                self._load()
                if self.df is None: return
            df = self.df.copy()
            ce, cn = self._sel(self.v_e.get()), self._sel(self.v_n.get())
            if not ce or not cn: raise ValueError("Selecteer kolommen voor Easting en Northing.")
            df["E"], df["N"] = numeric(df[ce]), numeric(df[cn])
            df = df.dropna(subset=["E", "N"]).copy()
            if df.empty: raise ValueError("Geen geldige coordinaten gevonden.")
            ch, cm1, cm2, ct, cd = self._sel(self.v_h.get()), self._sel(self.v_meta1.get()), self._sel(self.v_meta2.get()), self._sel(self.v_t.get()), self._sel(self.v_d.get())
            df["H"] = numeric(df[ch]) if ch else np.nan
            df["META1"] = df[cm1].map(s) if cm1 else ""
            df["META2"] = df[cm2].map(s) if cm2 else ""
            df["TIME"] = df[ct].map(s) if ct else ""
            df["DATE"] = df[cd].map(s) if cd else ""
            if self.v_crs.get().strip() == "EPSG:4326":
                df["lon"], df["lat"] = df["E"].astype(float), df["N"].astype(float)
            else:
                tr = Transformer.from_crs(self.v_crs.get().strip(), "EPSG:4326", always_xy=True)
                lon, lat = tr.transform(df["E"].to_numpy(dtype=float), df["N"].to_numpy(dtype=float))
                df["lon"], df["lat"] = lon, lat
            self.df_proj = df.reset_index(drop=True)
            self._log(f"  Punten: {len(self.df_proj)} | lat {self.df_proj['lat'].min():.6f}..{self.df_proj['lat'].max():.6f} | lon {self.df_proj['lon'].min():.6f}..{self.df_proj['lon'].max():.6f}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc)); self._log(f"ERROR: {exc}")

    def _base(self) -> Path:
        out = self.v_out.get().strip()
        if not out: raise ValueError("Kies eerst een output basisnaam.")
        return Path(out)

    def _map(self) -> None:
        try:
            if self.df_proj is None: self._project()
            if self.df_proj is None: return
            out = self._base().with_suffix(".html")
            make_map(self.df_proj, out, max(1, int(self.v_map_every.get())))
            self._log(f"Kaart opgeslagen: {out}")
            messagebox.showinfo("OK", f"Kaart opgeslagen:\n{out}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc)); self._log(f"ERROR: {exc}")

    def _dae(self) -> None:
        try:
            if self.df_proj is None: self._project()
            if self.df_proj is None: return
            out = self._base().with_suffix(".dae")
            df = self.df_proj
            h = df["H"].to_numpy(dtype=float) if not df["H"].isna().all() else None
            e0, n0, h0 = write_dae(df["E"].to_numpy(dtype=float), df["N"].to_numpy(dtype=float), h, out, DaeOpts(float(self.v_cube.get()), bool(self.v_ribbon.get()), float(self.v_ribbon_w.get()), bool(self.v_use_h.get()), max(1, int(self.v_dae_every.get()))))
            self._log(f"DAE opgeslagen: {out}")
            self._log(f"  Origin input CRS: E0={e0:.3f}, N0={n0:.3f}, H0={h0:.3f}")
            messagebox.showinfo("OK", f"DAE opgeslagen:\n{out}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc)); self._log(f"ERROR: {exc}")

    def _kmz(self) -> None:
        try:
            if self.df_proj is None: self._project()
            if self.df_proj is None: return
            base = self._base(); out_dae, out_kml, out_kmz = base.with_suffix(".dae"), base.with_suffix(".kml"), base.with_suffix(".kmz")
            df = self.df_proj
            h = df["H"].to_numpy(dtype=float) if not df["H"].isna().all() else None
            write_dae(df["E"].to_numpy(dtype=float), df["N"].to_numpy(dtype=float), h, out_dae, DaeOpts(float(self.v_cube.get()), bool(self.v_ribbon.get()), float(self.v_ribbon_w.get()), bool(self.v_use_h.get()), max(1, int(self.v_dae_every.get()))))
            write_kml(df, Path(out_dae).name, out_kml, max(1, int(self.v_map_every.get())), bool(self.v_use_h.get()), float(self.v_lift.get()))
            export_kmz(out_kmz, out_kml, out_dae)
            self._log(f"KML opgeslagen: {out_kml}")
            self._log(f"KMZ opgeslagen: {out_kmz}")
            messagebox.showinfo("OK", f"KML en KMZ opgeslagen:\n{out_kml}\n{out_kmz}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc)); self._log(f"ERROR: {exc}")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
