#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI: selecteer meerdere M9 MAT-bestanden en maak een interactieve 3D-waterkolomvisualisatie.

Resultaat:
- 3D draaibare HTML (Plotly)
- verticale waterkolom als gekleurde curtain (stromingssnelheid)
- optionele meetcel-inkleuring (gemeten punten, zonder interpolatie)
- tracklijn aan het oppervlak
- optionele stromingscontourlijnen
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
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
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio

try:
    import plotly.graph_objects as go  # type: ignore

    _PLOTLY_OK = True
except Exception:
    go = None
    _PLOTLY_OK = False

try:
    import folium  # type: ignore
    from pyproj import Transformer  # type: ignore
    from folium.plugins import Fullscreen, MeasureControl, MousePosition  # type: ignore

    _MAP_OK = True
    _PROJ_OK = True
except Exception:
    folium = None
    Transformer = None
    Fullscreen = None
    MeasureControl = None
    MousePosition = None
    _MAP_OK = False
    _PROJ_OK = False
    try:
        from pyproj import Transformer  # type: ignore

        _PROJ_OK = True
    except Exception:
        Transformer = None
        _PROJ_OK = False

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import cm as mpl_cm  # type: ignore
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore

        _MPL_TK_OK = True
    except Exception:
        FigureCanvasTkAgg = None
        _MPL_TK_OK = False

    _MPL_OK = True
except Exception:
    plt = None
    mpl_cm = None
    FigureCanvasTkAgg = None
    _MPL_OK = False
    _MPL_TK_OK = False


@dataclass
class Mat3DData:
    path: Path
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    speed: np.ndarray
    depth_abs: np.ndarray
    depth_top_abs: np.ndarray
    depth_bottom_abs: np.ndarray
    track_e: np.ndarray
    track_n: np.ndarray
    bed_depth: np.ndarray
    track_speed_surface: np.ndarray
    time_utc: pd.DatetimeIndex
    ensemble: np.ndarray
    note: str = ""
    processing_mode: str = "standard"
    compass_ok: bool | None = None
    compass_reason: str = ""
    report_lines: tuple[str, ...] = ()


@dataclass
class MatPreviewSummary:
    path: Path
    track_e: np.ndarray
    track_n: np.ndarray
    distance_m: np.ndarray
    elapsed_min: np.ndarray
    bed_depth: np.ndarray
    surface_speed: np.ndarray
    mean_speed_series: np.ndarray
    start_time: pd.Timestamp | None
    stop_time: pd.Timestamp | None
    speed_min: float | None
    speed_max: float | None
    speed_mean: float | None
    discharge_mean: float | None
    ensembles: int
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


def _as_xy(arr: object | None, n: int, ncols_min: int = 2) -> np.ndarray:
    if arr is None:
        return np.full((n, ncols_min), np.nan, dtype=float)
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        return np.full((n, ncols_min), np.nan, dtype=float)
    if out.shape[0] == n:
        pass
    elif out.shape[1] == n:
        out = out.T
    else:
        return np.full((n, ncols_min), np.nan, dtype=float)
    if out.shape[1] < ncols_min:
        out = np.pad(out, ((0, 0), (0, ncols_min - out.shape[1])), constant_values=np.nan)
    return np.asarray(out[:, :ncols_min], dtype=float)


def _bearing_from_en(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    ang = np.degrees(np.arctan2(np.asarray(east, dtype=float), np.asarray(north, dtype=float)))
    return (ang + 360.0) % 360.0


def _circular_mean_deg(values_deg: np.ndarray, weights: np.ndarray | None = None) -> float:
    vals = np.asarray(values_deg, dtype=float).reshape(-1)
    valid = np.isfinite(vals)
    if weights is None:
        w = np.ones_like(vals, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        valid &= np.isfinite(w)
    if not np.any(valid):
        return float("nan")
    vals = vals[valid]
    w = w[valid]
    s = float(np.sum(np.sin(np.radians(vals)) * w))
    c = float(np.sum(np.cos(np.radians(vals)) * w))
    return float((np.degrees(np.arctan2(s, c)) + 360.0) % 360.0)


def _parse_kmz_line(path: str | Path) -> tuple[tuple[float, float], tuple[float, float]]:
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(Path(path)) as zf:
        with zf.open("doc.kml") as fh:
            root = ET.parse(fh).getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coord_text = root.findtext(".//kml:LineString/kml:coordinates", namespaces=ns)
    if not coord_text:
        raise ValueError("Geen LineString coordinates gevonden in KMZ.")
    parts = [p for p in coord_text.strip().split() if p]
    if len(parts) < 2:
        raise ValueError("Te weinig coördinaten in KMZ.")
    lon1, lat1, *_ = map(float, parts[0].split(","))
    lon2, lat2, *_ = map(float, parts[1].split(","))
    return (lon1, lat1), (lon2, lat2)


def _geodesic_bearing_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lon1, lat1 = p1
    lon2, lat2 = p2
    phi1, phi2 = np.radians([lat1, lat2])
    lam1, lam2 = np.radians([lon1, lon2])
    dlam = lam2 - lam1
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def _angle_diff_deg(a: float, b: float) -> float:
    return float(((a - b + 180.0) % 360.0) - 180.0)


def _track_axis_bearing(track_e: np.ndarray, track_n: np.ndarray) -> float:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    valid = np.isfinite(te) & np.isfinite(tn)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    xy = np.column_stack((te[valid], tn[valid]))
    xy = xy - np.mean(xy, axis=0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    vec = vt[0]
    return float(_bearing_from_en(np.array([vec[0]]), np.array([vec[1]]))[0])


def _track_net_bearing(track_e: np.ndarray, track_n: np.ndarray) -> float:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    valid = np.isfinite(te) & np.isfinite(tn)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    te = te[valid]
    tn = tn[valid]
    de = float(te[-1] - te[0])
    dn = float(tn[-1] - tn[0])
    if np.hypot(de, dn) < 0.05:
        return float("nan")
    return float(_bearing_from_en(np.array([de]), np.array([dn]))[0])


def _rotate_points(track_e: np.ndarray, track_n: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ang = np.radians(float(angle_deg))
    ca = float(np.cos(ang))
    sa = float(np.sin(ang))
    te = np.asarray(track_e, dtype=float)
    tn = np.asarray(track_n, dtype=float)
    return te * ca - tn * sa, te * sa + tn * ca


def _first_valid_xy(track_e: np.ndarray, track_n: np.ndarray) -> tuple[float, float] | None:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    idx = np.where(np.isfinite(te) & np.isfinite(tn))[0]
    if idx.size == 0:
        return None
    i0 = int(idx[0])
    return float(te[i0]), float(tn[i0])


def _read_kml_or_kmz_geometry(path: str | Path) -> tuple[str, list[tuple[float, float]]]:
    import xml.etree.ElementTree as ET

    p = Path(path)
    if p.suffix.lower() == ".kmz":
        with zipfile.ZipFile(p) as zf:
            with zf.open("doc.kml") as fh:
                root = ET.parse(fh).getroot()
    else:
        root = ET.parse(str(p)).getroot()

    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coord_text = root.findtext(".//kml:LineString/kml:coordinates", namespaces=ns)
    if coord_text:
        parts = [q for q in coord_text.strip().split() if q]
        coords: list[tuple[float, float]] = []
        for part in parts:
            lon, lat, *_ = map(float, part.split(","))
            coords.append((lon, lat))
        if len(coords) >= 2:
            return "line", coords

    coord_text = root.findtext(".//kml:Point/kml:coordinates", namespaces=ns)
    if coord_text:
        lon, lat, *_ = map(float, coord_text.strip().split(","))
        return "point", [(lon, lat)]

    raise ValueError("Geen bruikbare Point of LineString gevonden in KML/KMZ.")


def _parse_kmz_line(path: str | Path) -> tuple[tuple[float, float], tuple[float, float]]:
    geom_type, coords = _read_kml_or_kmz_geometry(path)
    if geom_type != "line" or len(coords) < 2:
        raise ValueError("Geen bruikbare LineString gevonden in KML/KMZ.")
    return coords[0], coords[-1]


def _kmz_or_kml_reference_to_projected(path: str | Path, track_epsg: int) -> tuple[float, float, str]:
    if not _PROJ_OK:
        raise RuntimeError("pyproj niet beschikbaar; KML/KMZ referentie vereist coordinatentransformatie.")
    geom_type, coords = _read_kml_or_kmz_geometry(path)
    if geom_type != "point":
        raise ValueError("Alleen een puntreferentie in KML/KMZ wordt nog ondersteund.")
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(track_epsg)}", always_xy=True)
    lon, lat = coords[0]
    e, n = tr.transform(lon, lat)
    return float(e), float(n), "KMZ/KML punt"


def _wgs84_point_to_projected(lon: float, lat: float, track_epsg: int) -> tuple[float, float]:
    if not _PROJ_OK:
        raise RuntimeError("pyproj niet beschikbaar; WGS84 -> projected conversie vereist.")
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(track_epsg)}", always_xy=True)
    e, n = tr.transform(float(lon), float(lat))
    return float(e), float(n)


def _profile_reference_from_banks(
    left_lon: float,
    left_lat: float,
    right_lon: float,
    right_lat: float,
    track_e: np.ndarray,
    track_n: np.ndarray,
    track_epsg: int,
) -> tuple[float, float, float, str]:
    left_e, left_n = _wgs84_point_to_projected(left_lon, left_lat, track_epsg)
    right_e, right_n = _wgs84_point_to_projected(right_lon, right_lat, track_epsg)
    left_to_right = _geodesic_bearing_deg((left_lon, left_lat), (right_lon, right_lat))

    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    first_xy = _first_valid_xy(te, tn)
    rel_e = te.copy()
    rel_n = tn.copy()
    if first_xy is not None:
        rel_e = te - first_xy[0]
        rel_n = tn - first_xy[1]
    measured = _track_net_bearing(rel_e, rel_n)
    if not np.isfinite(measured):
        measured = _track_axis_bearing(rel_e, rel_n)

    candidates = [
        (float(left_to_right), float(left_e), float(left_n), "linkeroever -> rechteroever"),
        (float((left_to_right + 180.0) % 360.0), float(right_e), float(right_n), "rechteroever -> linkeroever"),
    ]
    if np.isfinite(measured):
        chosen = min(candidates, key=lambda item: abs(_angle_diff_deg(item[0], measured)))
    else:
        chosen = candidates[0]
    chosen_bearing, chosen_e, chosen_n, chosen_label = chosen
    return chosen_e, chosen_n, chosen_bearing, chosen_label


def _write_reference_kml(path: str | Path, coords: list[tuple[float, float]], geom_type: str, name: str) -> Path:
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if geom_type == "line":
        coord_str = " ".join(f"{lon:.9f},{lat:.9f},0" for lon, lat in coords)
        geom_xml = f"<LineString><tessellate>1</tessellate><coordinates>{coord_str}</coordinates></LineString>"
    elif geom_type == "point":
        lon, lat = coords[0]
        geom_xml = f"<Point><coordinates>{lon:.9f},{lat:.9f},0</coordinates></Point>"
    else:
        raise ValueError(f"Onbekend referentie-geometrie type: {geom_type}")

    txt = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            "<Document>",
            f"<name>{html.escape(name)}</name>",
            "<Placemark>",
            f"<name>{html.escape(name)}</name>",
            geom_xml,
            "</Placemark>",
            "</Document>",
            "</kml>",
            "",
        ]
    )
    outp.write_text(txt, encoding="utf-8")
    return outp


def _default_reference_center(reference_path: str | Path | None = None) -> tuple[float, float]:
    if reference_path:
        try:
            _, coords = _read_kml_or_kmz_geometry(reference_path)
            if coords:
                lon = float(np.mean([pt[0] for pt in coords]))
                lat = float(np.mean([pt[1] for pt in coords]))
                return lat, lon
        except Exception:
            pass
    return 50.8503, 4.3517


def _launch_reference_picker(
    initial_center: tuple[float, float] = (50.8503, 4.3517),
    initial_zoom: int = 16,
    initial_mode: str = "point",
) -> dict[str, object] | None:
    state: dict[str, object] = {"selection": None}
    ready = threading.Event()
    picked = threading.Event()

    class _PickerHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_bytes(self, payload: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/", "/index.html"):
                self._send_bytes(b"not found", content_type="text/plain; charset=utf-8", status=404)
                return

            lat0, lon0 = float(initial_center[0]), float(initial_center[1])
            html_txt = f"""<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <title>Referentiekaart</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  >
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    body {{ font-family: Segoe UI, Arial, sans-serif; }}
    .panel {{
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 1000;
      background: rgba(255,255,255,0.95);
      padding: 10px 12px;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.2);
      max-width: 420px;
    }}
    .panel h3 {{ margin: 0 0 8px 0; font-size: 16px; }}
    .panel p {{ margin: 6px 0; font-size: 13px; line-height: 1.35; }}
    .row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
    button {{
      border: 1px solid #667;
      background: #f5f7fa;
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    button.active {{ background: #d8ecff; border-color: #1967a5; }}
    #status {{ font-weight: 600; }}
    code {{ background: #eef2f5; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h3>Referentie kiezen</h3>
    <p>Kies een <b>punt</b> voor stationair of een <b>lijn</b> met 2 klikken voor dwarsprofiel.</p>
    <p id="status">Nog niets geselecteerd.</p>
    <div class="row">
      <button id="mode-point" class="active" type="button">Punt</button>
      <button id="mode-line" type="button">Lijn (2 klikken)</button>
      <button id="clear" type="button">Wis selectie</button>
      <button id="save" type="button">Opslaan</button>
    </div>
    <p>Klik op de kaart om een punt of 2 lijnpunten te kiezen.</p>
    <p>OpenStreetMap basislaag. Coords in <code>WGS84</code>.</p>
  </div>
  <div id="map"></div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const map = L.map('map').setView([{lat0:.7f}, {lon0:.7f}], {int(initial_zoom)});
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 22,
      attribution: '&copy; OpenStreetMap-bijdragers'
    }}).addTo(map);

    let mode = {json.dumps(str(initial_mode).strip().lower() if str(initial_mode).strip().lower() in {'point', 'line'} else 'point')};
    let markerA = null;
    let markerB = null;
    let line = null;
    let selected = [];

    const statusEl = document.getElementById('status');
    const btnPoint = document.getElementById('mode-point');
    const btnLine = document.getElementById('mode-line');
    function setMode(newMode) {{
      mode = newMode;
      btnPoint.classList.toggle('active', mode === 'point');
      btnLine.classList.toggle('active', mode === 'line');
      clearSelection();
      updateStatus();
    }}

    function clearSelection() {{
      selected = [];
      if (markerA) {{ map.removeLayer(markerA); markerA = null; }}
      if (markerB) {{ map.removeLayer(markerB); markerB = null; }}
      if (line) {{ map.removeLayer(line); line = null; }}
    }}

    function updateStatus() {{
      if (selected.length === 0) {{
        statusEl.textContent = mode === 'point' ? 'Nog niets geselecteerd.' : 'Nog geen lijn geselecteerd.';
        return;
      }}
      if (mode === 'point') {{
        const p = selected[0];
        statusEl.textContent = `Punt: lat=${{p.lat.toFixed(7)}}, lon=${{p.lon.toFixed(7)}}`;
        return;
      }}
      if (selected.length === 1) {{
        const p = selected[0];
        statusEl.textContent = `Lijn start: lat=${{p.lat.toFixed(7)}}, lon=${{p.lon.toFixed(7)}}. Klik eindpunt.`;
        return;
      }}
      const p1 = selected[0];
      const p2 = selected[1];
      statusEl.textContent = `Lijn: [${{p1.lat.toFixed(7)}}, ${{p1.lon.toFixed(7)}}] -> [${{p2.lat.toFixed(7)}}, ${{p2.lon.toFixed(7)}}]`;
    }}

    function renderSelection() {{
      if (markerA) {{ map.removeLayer(markerA); markerA = null; }}
      if (markerB) {{ map.removeLayer(markerB); markerB = null; }}
      if (line) {{ map.removeLayer(line); line = null; }}
      if (selected.length >= 1) {{
        markerA = L.marker([selected[0].lat, selected[0].lon]).addTo(map);
      }}
      if (selected.length >= 2) {{
        markerB = L.marker([selected[1].lat, selected[1].lon]).addTo(map);
        line = L.polyline([
          [selected[0].lat, selected[0].lon],
          [selected[1].lat, selected[1].lon]
        ], {{color: '#005bbb', weight: 4}}).addTo(map);
      }}
      updateStatus();
    }}

    map.on('click', (ev) => {{
      const pt = {{lat: ev.latlng.lat, lon: ev.latlng.lng}};
      if (mode === 'point') {{
        selected = [pt];
      }} else {{
        if (selected.length >= 2) {{
          selected = [pt];
        }} else {{
          selected.push(pt);
        }}
      }}
      renderSelection();
    }});

    document.getElementById('clear').addEventListener('click', () => {{
      clearSelection();
      updateStatus();
    }});
    btnPoint.addEventListener('click', () => setMode('point'));
    btnLine.addEventListener('click', () => setMode('line'));
    setMode(mode);

    document.getElementById('save').addEventListener('click', async () => {{
      if ((mode === 'point' && selected.length !== 1) || (mode === 'line' && selected.length !== 2)) {{
        alert(mode === 'point' ? 'Klik eerst een punt.' : 'Klik eerst twee punten voor het profiel.');
        return;
      }}
      const payload = {{mode: mode, coords: selected}};
      const resp = await fetch('/select', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
      }});
      if (!resp.ok) {{
        alert('Opslaan mislukt.');
        return;
      }}
      statusEl.textContent = 'Selectie opgeslagen. Dit venster mag gesloten worden.';
    }});
  </script>
</body>
</html>
"""
            self._send_bytes(html_txt.encode("utf-8"))

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/select":
                self._send_bytes(b"not found", content_type="text/plain; charset=utf-8", status=404)
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                mode = str(payload.get("mode", "")).strip().lower()
                coords = payload.get("coords", [])
                if mode not in {"point", "line"}:
                    raise ValueError("Ongeldige modus.")
                if not isinstance(coords, list):
                    raise ValueError("Ongeldige coördinaten.")
                parsed: list[tuple[float, float]] = []
                for item in coords[:2]:
                    if not isinstance(item, dict):
                        raise ValueError("Ongeldige coordinaatstructuur.")
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    parsed.append((lon, lat))
                if mode == "point" and len(parsed) != 1:
                    raise ValueError("Punt vereist exact 1 klik.")
                if mode == "line" and len(parsed) != 2:
                    raise ValueError("Raai vereist exact 2 klikken.")
                state["selection"] = {"mode": mode, "coords": parsed}
                picked.set()
                self._send_bytes(b'{"ok":true}', content_type="application/json; charset=utf-8")
            except Exception as exc:
                msg = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                self._send_bytes(msg, content_type="application/json; charset=utf-8", status=400)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _PickerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    ready.set()
    try:
        if not ready.wait(timeout=2.0):
            raise RuntimeError("Kaartserver kon niet gestart worden.")
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        webbrowser.open(url)
        if not picked.wait(timeout=600.0):
            return None
        return state["selection"] if isinstance(state["selection"], dict) else None
    finally:
        server.shutdown()
        server.server_close()


def _apply_reference_georef(
    track_e: np.ndarray,
    track_n: np.ndarray,
    anchor_e: float,
    anchor_n: float,
    target_bearing_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    report: list[str] = []
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    out_e = te.copy()
    out_n = tn.copy()
    first_xy = _first_valid_xy(te, tn)
    if first_xy is None:
        return out_e, out_n, ["Georeferentie overgeslagen: geen geldige trackpunten."]

    base_e, base_n = first_xy
    rel_e = te - base_e
    rel_n = tn - base_n

    if target_bearing_deg is not None and np.isfinite(target_bearing_deg):
        measured = _track_axis_bearing(rel_e, rel_n)
        if np.isfinite(measured):
            candidates = [float(target_bearing_deg), float((target_bearing_deg + 180.0) % 360.0)]
            chosen = min(candidates, key=lambda cand: abs(_angle_diff_deg(cand, measured)))
            delta = _angle_diff_deg(chosen, measured)
            rel_e, rel_n = _rotate_points(rel_e, rel_n, -delta)
            report.append(f"Track geroteerd met {delta:+.3f}° naar doelazimut {chosen:.3f}°.")
        else:
            report.append("Doelazimut aanwezig, maar trackas kon niet betrouwbaar bepaald worden.")

    out_e = rel_e + float(anchor_e)
    out_n = rel_n + float(anchor_n)
    report.append(f"Track verankerd op E={float(anchor_e):.3f}, N={float(anchor_n):.3f}.")
    return out_e, out_n, report


def _project_on_bearing(east: np.ndarray, north: np.ndarray, bearing_deg: float) -> np.ndarray:
    ang = np.radians(float(bearing_deg))
    unit_e = np.sin(ang)
    unit_n = np.cos(ang)
    return np.asarray(east, dtype=float) * unit_e + np.asarray(north, dtype=float) * unit_n


def _compass_status(
    setup: object,
    system: object,
    compass: object | None,
    ns: int,
    check_compass: bool,
    force_raai_fallback: bool,
    mag_error_threshold: float,
) -> tuple[bool | None, str, dict[str, object]]:
    heading = _as_1d(getattr(system, "Heading", None), ns)
    true_heading = _as_1d(getattr(system, "True_North_ADP_Heading", None), ns)
    gps_heading = _as_1d(getattr(system, "GPS_Compass_Heading", None), ns)
    mag_err = _as_1d(getattr(compass, "Magnetic_error", None) if compass is not None else None, ns)

    gps_valid = np.isfinite(gps_heading) & (gps_heading > 0.0) & (gps_heading < 360.0) & (np.abs(gps_heading - 655.35) > 1e-6)
    mag_valid = np.isfinite(mag_err)
    mag_mean = float(np.nanmean(mag_err[mag_valid])) if np.any(mag_valid) else float("nan")
    mag_max = float(np.nanmax(mag_err[mag_valid])) if np.any(mag_valid) else float("nan")
    decl = float(getattr(setup, "magneticDeclination", np.nan))
    heading_source = getattr(setup, "headingSource", None)
    true_eq_heading = bool(
        np.any(np.isfinite(heading) & np.isfinite(true_heading))
        and np.allclose(heading[np.isfinite(heading) & np.isfinite(true_heading)], true_heading[np.isfinite(heading) & np.isfinite(true_heading)])
    )

    if force_raai_fallback:
        compass_ok = False
        reason = "Raai-fallback handmatig geforceerd."
    elif not check_compass:
        compass_ok = None
        reason = "Kompascontrole uitgeschakeld."
    elif np.count_nonzero(gps_valid) >= max(5, int(0.2 * ns)):
        compass_ok = True
        reason = "GPS heading beschikbaar."
    elif np.isfinite(mag_mean) and np.isfinite(mag_max):
        compass_ok = bool(mag_mean <= float(mag_error_threshold) and mag_max <= float(mag_error_threshold) * 1.5)
        state = "OK" if compass_ok else "niet OK"
        reason = f"Interne kompascheck {state}: magnetic error mean={mag_mean:.2f}% max={mag_max:.2f}%."
    else:
        compass_ok = None
        reason = "Onvoldoende kompascijfers voor automatische check."

    info: dict[str, object] = {
        "heading_source": heading_source,
        "declination": decl,
        "gps_valid_count": int(np.count_nonzero(gps_valid)),
        "mag_mean": mag_mean,
        "mag_max": mag_max,
        "true_eq_heading": true_eq_heading,
    }
    return compass_ok, reason, info


def _apply_reference_georef(
    track_e: np.ndarray,
    track_n: np.ndarray,
    anchor_e: float,
    anchor_n: float,
    target_bearing_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    report: list[str] = []
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    out_e = te.copy()
    out_n = tn.copy()
    first_xy = _first_valid_xy(te, tn)
    if first_xy is None:
        return out_e, out_n, ["Georeferentie overgeslagen: geen geldige trackpunten."]

    base_e, base_n = first_xy
    rel_e = te - base_e
    rel_n = tn - base_n

    if target_bearing_deg is not None and np.isfinite(target_bearing_deg):
        measured = _track_net_bearing(rel_e, rel_n)
        measured_label = "netto verplaatsingsrichting"
        if not np.isfinite(measured):
            measured = _track_axis_bearing(rel_e, rel_n)
            measured_label = "hoofd-as"
        if np.isfinite(measured):
            candidates = [float(target_bearing_deg), float((target_bearing_deg + 180.0) % 360.0)]
            chosen = min(candidates, key=lambda cand: abs(_angle_diff_deg(cand, measured)))
            delta = _angle_diff_deg(chosen, measured)
            rel_e, rel_n = _rotate_points(rel_e, rel_n, delta)
            report.append(f"Track geroteerd met {delta:+.3f}° op basis van {measured_label} naar doelazimut {chosen:.3f}°.")
        else:
            report.append("Doelazimut aanwezig, maar trackrichting kon niet betrouwbaar bepaald worden.")

    out_e = rel_e + float(anchor_e)
    out_n = rel_n + float(anchor_n)
    report.append(f"Track verankerd op E={float(anchor_e):.3f}, N={float(anchor_n):.3f}.")
    return out_e, out_n, report


def _cluster_indices(indices: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    idx = np.asarray(indices, dtype=int).reshape(-1)
    if idx.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for val in idx[1:]:
        val = int(val)
        if val - prev <= max_gap:
            prev = val
            continue
        out.append((start, prev))
        start = prev = val
    out.append((start, prev))
    return out


def _apply_reference_georef(
    track_e: np.ndarray,
    track_n: np.ndarray,
    anchor_e: float,
    anchor_n: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    report: list[str] = []
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    out_e = te.copy()
    out_n = tn.copy()
    first_xy = _first_valid_xy(te, tn)
    if first_xy is None:
        return out_e, out_n, ["Georeferentie overgeslagen: geen geldige trackpunten."]

    base_e, base_n = first_xy
    rel_e = te - base_e
    rel_n = tn - base_n
    chosen_bearing = float("nan")

    if target_bearing_deg is not None and np.isfinite(target_bearing_deg):
        measured = _track_net_bearing(rel_e, rel_n)
        measured_label = "netto verplaatsingsrichting"
        if not np.isfinite(measured):
            measured = _track_axis_bearing(rel_e, rel_n)
            measured_label = "hoofd-as"
        if np.isfinite(measured):
            candidates = [float(target_bearing_deg), float((target_bearing_deg + 180.0) % 360.0)]
            chosen_bearing = min(candidates, key=lambda cand: abs(_angle_diff_deg(cand, measured)))
            delta = _angle_diff_deg(chosen_bearing, measured)
            rel_e, rel_n = _rotate_points(rel_e, rel_n, -delta)
            report.append(f"Track geroteerd met {delta:+.3f}° op basis van {measured_label} naar doelazimut {chosen_bearing:.3f}°.")
        else:
            report.append("Doelazimut aanwezig, maar trackrichting kon niet betrouwbaar bepaald worden.")

    if snap_to_line and np.isfinite(chosen_bearing):
        along = _project_on_bearing(rel_e, rel_n, chosen_bearing)
        ang = np.radians(chosen_bearing)
        rel_e = along * np.sin(ang)
        rel_n = along * np.cos(ang)
        report.append("Relatieve track op externe raai geprojecteerd.")

    out_e = rel_e + float(anchor_e)
    out_n = rel_n + float(anchor_n)
    report.append(f"Track verankerd op E={float(anchor_e):.3f}, N={float(anchor_n):.3f}.")
    return out_e, out_n, report


def _apply_point_reference_georef(
    track_e: np.ndarray,
    track_n: np.ndarray,
    anchor_e: float,
    anchor_n: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    first_xy = _first_valid_xy(te, tn)
    if first_xy is None:
        return te.copy(), tn.copy(), ["Georeferentie overgeslagen: geen geldige trackpunten."]

    base_e, base_n = first_xy
    rel_e = te - base_e
    rel_n = tn - base_n
    out_e = rel_e + float(anchor_e)
    out_n = rel_n + float(anchor_n)
    return out_e, out_n, [f"Track verankerd op E={float(anchor_e):.3f}, N={float(anchor_n):.3f}."]


def _align_track_to_profile(
    track_e: np.ndarray,
    track_n: np.ndarray,
    anchor_e: float,
    anchor_n: float,
    target_bearing_deg: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    first_xy = _first_valid_xy(te, tn)
    if first_xy is None:
        return te.copy(), tn.copy(), ["Profieluitlijning overgeslagen: geen geldige trackpunten."]

    base_e, base_n = first_xy
    rel_e = te - base_e
    rel_n = tn - base_n
    measured = _track_net_bearing(rel_e, rel_n)
    measured_label = "netto verplaatsingsrichting"
    if not np.isfinite(measured):
        measured = _track_axis_bearing(rel_e, rel_n)
        measured_label = "hoofd-as"
    if not np.isfinite(measured):
        out_e = rel_e + float(anchor_e)
        out_n = rel_n + float(anchor_n)
        return out_e, out_n, [f"Track verankerd op E={float(anchor_e):.3f}, N={float(anchor_n):.3f}."]

    candidates = [float(target_bearing_deg), float((target_bearing_deg + 180.0) % 360.0)]
    chosen = min(candidates, key=lambda cand: abs(_angle_diff_deg(cand, measured)))
    delta = _angle_diff_deg(chosen, measured)
    rel_e, rel_n = _rotate_points(rel_e, rel_n, -delta)
    out_e = rel_e + float(anchor_e)
    out_n = rel_n + float(anchor_n)
    report = [
        f"Track geroteerd met {delta:+.3f}° op basis van {measured_label} naar profielazimut {chosen:.3f}°.",
        f"Track verankerd op E={float(anchor_e):.3f}, N={float(anchor_n):.3f}.",
    ]
    return out_e, out_n, report


def _segments_between_clusters(length: int, clusters: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    if not clusters:
        return [(0, length - 1)]
    out: list[tuple[int, int]] = []
    start = 0
    for a, b in clusters:
        if start <= a - 1:
            out.append((start, a - 1))
        start = b + 1
    if start <= length - 1:
        out.append((start, length - 1))
    return out


def _analyze_compass_and_raai(
    p: Path,
    setup: object,
    system: object,
    compass: object | None,
    summary: object,
    track_e: np.ndarray,
    track_n: np.ndarray,
    ns: int,
    raai_kmz_path: str | Path | None = None,
    check_compass: bool = True,
    fallback_on_bad_compass: bool = True,
    force_raai_fallback: bool = False,
    relative_track_to_raai: bool = False,
    mag_error_threshold: float = 3.5,
) -> tuple[bool | None, str, str, tuple[str, ...]]:
    report: list[str] = []
    boat_xy = _as_xy(getattr(summary, "Boat_Vel", None), ns, ncols_min=2)
    water_xy = _as_xy(getattr(summary, "Mean_Vel", None), ns, ncols_min=2)
    compass_ok, reason, info = _compass_status(
        setup=setup,
        system=system,
        compass=compass,
        ns=ns,
        check_compass=check_compass,
        force_raai_fallback=force_raai_fallback,
        mag_error_threshold=mag_error_threshold,
    )
    heading_source = info["heading_source"]
    decl = float(info["declination"])
    gps_valid_count = int(info["gps_valid_count"])
    mag_mean = float(info["mag_mean"])
    mag_max = float(info["mag_max"])
    true_eq_heading = bool(info["true_eq_heading"])

    report.append(
        f"Kompas: headingSource={heading_source}, declination={decl:.3f}, "
        f"GPS_heading={'ja' if gps_valid_count else 'nee'}, "
        f"mag_error_mean={mag_mean:.2f}% max={mag_max:.2f}%, "
        f"true_equals_heading={'ja' if true_eq_heading else 'nee'}."
    )
    report.append(reason)
    if true_eq_heading and np.isfinite(decl) and abs(decl) < 1e-9:
        report.append("Opmerking: True_North_ADP_Heading is gelijk aan Heading en declinatie staat op 0; true north is dus niet echt gecorrigeerd.")

    use_fallback = bool(raai_kmz_path) and (
        relative_track_to_raai or force_raai_fallback or (fallback_on_bad_compass and compass_ok is False)
    )
    processing_mode = "standard"

    if use_fallback:
        p1, p2 = _parse_kmz_line(raai_kmz_path)
        raai_bearing = _geodesic_bearing_deg(p1, p2)
        normal_bearing = (raai_bearing + 90.0) % 360.0

        track_xy = np.column_stack((track_e[:ns], track_n[:ns])).astype(float)
        track_rel = track_xy - track_xy[0]
        track_along = _project_on_bearing(track_rel[:, 0], track_rel[:, 1], raai_bearing)
        track_normal = _project_on_bearing(track_rel[:, 0], track_rel[:, 1], normal_bearing)
        water_normal = _project_on_bearing(water_xy[:, 0], water_xy[:, 1], normal_bearing)
        water_along = _project_on_bearing(water_xy[:, 0], water_xy[:, 1], raai_bearing)
        boat_speed = np.linalg.norm(boat_xy[:, :2], axis=1)

        if ns >= 2:
            inc = np.diff(track_xy, axis=0)
            inc_normal = _project_on_bearing(inc[:, 0], inc[:, 1], normal_bearing)
            inc_abs = np.linalg.norm(inc, axis=1)
            event_idx = np.where((np.abs(inc_normal) > 0.20) | (inc_abs > 0.25))[0] + 1
            events = _cluster_indices(event_idx, max_gap=2)
        else:
            events = []

        segments = _segments_between_clusters(ns, events)
        if relative_track_to_raai:
            report.append("Lokale relatieve raai geprojecteerd op externe raai.")
        report.append(
            f"Raai-fallback actief: azimut={raai_bearing:.3f}°, normaal={normal_bearing:.3f}°, "
            f"clusters={len(events)}, segmenten={len(segments)}."
        )
        if events:
            txt = ", ".join(f"{a:03d}-{b:03d}" for a, b in events[:8])
            if len(events) > 8:
                txt += ", ..."
            report.append(f"Verplaatsingsclusters: {txt}")

        for idx, (a, b) in enumerate(segments, start=1):
            seg_vn = water_normal[a:b + 1]
            seg_va = water_along[a:b + 1]
            seg_bt = boat_speed[a:b + 1]
            seg_mean = float(np.nanmean(seg_vn))
            seg_std = float(np.nanstd(seg_vn))
            seg_bt_mean = float(np.nanmean(seg_bt))
            seg_dn = float(track_normal[b] - track_normal[a])
            seg_da = float(track_along[b] - track_along[a])
            report.append(
                f"Segment S{idx}: {a:03d}-{b:03d}, n={b-a+1}, "
                f"v_norm_mean={seg_mean:+.4f}, v_norm_std={seg_std:.4f}, "
                f"v_along_mean={float(np.nanmean(seg_va)):+.4f}, "
                f"bt_mean={seg_bt_mean:.4f}, d_norm={seg_dn:+.3f} m, d_along={seg_da:+.3f} m"
            )

        candidate_lines: list[str] = []
        for idx, (a, b) in enumerate(segments, start=1):
            seg_vn = water_normal[a:b + 1]
            seg_bt = boat_speed[a:b + 1]
            seg_mean = float(np.nanmean(seg_vn))
            seg_std = float(np.nanstd(seg_vn))
            seg_bt_mean = float(np.nanmean(seg_bt))
            if (b - a + 1) >= 20 and (abs(seg_mean) >= 0.05 or seg_std <= 0.08):
                reasons: list[str] = []
                if abs(seg_mean) >= 0.10:
                    reasons.append("duidelijke dwarscomponent")
                elif abs(seg_mean) >= 0.05:
                    reasons.append("matige dwarscomponent")
                if seg_std <= 0.08:
                    reasons.append("stabiel signaal")
                if seg_bt_mean <= 0.08:
                    reasons.append("beperkte gemiddelde bodemverplaatsing")
                candidate_lines.append(f"S{idx} ({a:03d}-{b:03d}): {', '.join(reasons) if reasons else 'aparte behandeling'}")
        if candidate_lines:
            report.append("Aanbevolen segmenten: " + " | ".join(candidate_lines))
        if relative_track_to_raai and not force_raai_fallback and compass_ok is not False:
            processing_mode = "raai_projected"
        else:
            processing_mode = "raai_fallback"

    return compass_ok, reason, processing_mode, tuple(report)


def _reorder_velocity(vel_raw: np.ndarray, ns_target: int) -> np.ndarray:
    vel = np.asarray(vel_raw, dtype=float)
    if vel.ndim != 3:
        raise ValueError(f"WaterTrack.Velocity moet 3D zijn, kreeg shape {vel.shape}")

    axes = [0, 1, 2]

    # component-as: meestal lengte 4
    if 4 in vel.shape:
        comp_axis = int(np.where(np.array(vel.shape) == 4)[0][0])
    else:
        comp_axis = min(axes, key=lambda a: abs(vel.shape[a] - 4))

    rem = [a for a in axes if a != comp_axis]
    ens_axis = min(rem, key=lambda a: abs(vel.shape[a] - ns_target))
    cell_axis = [a for a in rem if a != ens_axis][0]

    vel = np.moveaxis(vel, [cell_axis, comp_axis, ens_axis], [0, 1, 2])
    return vel


def _surface_speed_from_columns(speed: np.ndarray) -> np.ndarray:
    ncols = speed.shape[1]
    out = np.full(ncols, np.nan, dtype=float)
    for j in range(ncols):
        col = speed[:, j]
        idx = np.where(np.isfinite(col))[0]
        if idx.size > 0:
            out[j] = float(col[idx[0]])
    return out


def _fill_1d_linear(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).reshape(-1).copy()
    idx = np.where(np.isfinite(out))[0]
    if idx.size == 0:
        return out
    if idx.size == 1:
        out[:] = out[idx[0]]
        return out
    xx = np.arange(out.size, dtype=float)
    out[:] = np.interp(xx, idx.astype(float), out[idx])
    return out


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    c = _fill_1d_linear(centers)
    n = c.size
    out = np.full(n + 1, np.nan, dtype=float)
    if n == 0 or np.count_nonzero(np.isfinite(c)) == 0:
        return out
    if n == 1:
        out[0] = c[0] - 0.5
        out[1] = c[0] + 0.5
        return out
    out[1:-1] = 0.5 * (c[:-1] + c[1:])
    out[0] = c[0] - 0.5 * (c[1] - c[0])
    out[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return out


def _sample_idx_with_last(n: int, step: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=int)
    step = max(1, int(step))
    idx = np.arange(0, n, step, dtype=int)
    if idx[-1] != (n - 1):
        idx = np.append(idx, n - 1)
    return idx


def _looks_georeferenced(track_e: np.ndarray, track_n: np.ndarray) -> bool:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    valid = np.isfinite(te) & np.isfinite(tn)
    if np.count_nonzero(valid) < 2:
        return False
    te = te[valid]
    tn = tn[valid]
    med_abs = max(float(np.nanmedian(np.abs(te))), float(np.nanmedian(np.abs(tn))))
    span = max(float(np.nanmax(te) - np.nanmin(te)), float(np.nanmax(tn) - np.nanmin(tn)))
    # Belgische Lambert72 ligt ruwweg in honderdduizenden meters.
    # Tracks met waarden van enkele meters zijn vrijwel zeker relatief en niet gegeorefereerd.
    return bool(med_abs >= 5000.0 and span >= 1.0)


def _plotly_colorbar_controls_post_script(default_scale: str) -> str:
    script = r"""
(function() {
  const gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) return;

  const meta = (gd.layout && gd.layout.meta) ? gd.layout.meta : {};
  const sMin0 = Number(meta.speed_min);
  const sMax0 = Number(meta.speed_max);
  if (!Number.isFinite(sMin0) || !Number.isFinite(sMax0) || sMax0 <= sMin0) return;

  const surfaceIdx = [];
  const meshIdx = [];
  const markerIdx = [];
  for (let i = 0; i < gd.data.length; i++) {
    const t = gd.data[i];
    if (t && String(t.type || '').toLowerCase() === 'surface') {
      surfaceIdx.push(i);
    } else if (
      t &&
      String(t.type || '').toLowerCase() === 'mesh3d' &&
      t.intensity !== undefined
    ) {
      meshIdx.push(i);
    } else if (
      t &&
      String(t.type || '').toLowerCase() === 'scatter3d' &&
      t.marker &&
      t.marker.color !== undefined &&
      t.marker.colorscale !== undefined
    ) {
      markerIdx.push(i);
    }
  }
  if (!surfaceIdx.length && !meshIdx.length && !markerIdx.length) return;

  const scales = ["Turbo", "Viridis", "Plasma", "Cividis", "Jet", "Portland", "RdBu", "Bluered", "YlGnBu"];
  const root = gd.parentElement || gd;
  if (window.getComputedStyle(root).position === 'static') {
    root.style.position = 'relative';
  }

  const panel = document.createElement('div');
  panel.style.cssText = [
    "position:absolute",
    "top:12px",
    "left:12px",
    "z-index:30",
    "background:rgba(255,255,255,0.94)",
    "border:1px solid #a0a0a0",
    "border-radius:8px",
    "padding:8px 10px",
    "font:12px/1.2 Arial,sans-serif",
    "box-shadow:0 2px 8px rgba(0,0,0,0.18)"
  ].join(';');

  const rowStyle = "display:grid;grid-template-columns:110px 120px;gap:8px;align-items:center;margin:4px 0;";
  panel.innerHTML = ''
    + '<div style="font-weight:600;margin-bottom:6px;">Kleurschaal 3D</div>'
    + '<div style="' + rowStyle + '"><label for="m9-scale">Kleurschaal</label><select id="m9-scale"></select></div>'
    + '<div style="' + rowStyle + '"><label for="m9-cmin">Min (m/s)</label><input id="m9-cmin" type="number" step="0.01"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-cmax">Max (m/s)</label><input id="m9-cmax" type="number" step="0.01"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-cbx">Balk X</label><input id="m9-cbx" type="range" min="0.80" max="1.35" step="0.01"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-cblen">Balk lengte</label><input id="m9-cblen" type="range" min="0.25" max="0.95" step="0.01"></div>'
    + '<div style="' + rowStyle + '"><label for="m9-cbth">Balk dikte</label><input id="m9-cbth" type="range" min="8" max="40" step="1"></div>'
    + '<div style="margin-top:6px;"><button id="m9-cbapply" type="button">Toepassen</button> <button id="m9-cbreset" type="button">Reset</button></div>';

  root.appendChild(panel);

  const scaleSel = panel.querySelector('#m9-scale');
  const cminInput = panel.querySelector('#m9-cmin');
  const cmaxInput = panel.querySelector('#m9-cmax');
  const cbxInput = panel.querySelector('#m9-cbx');
  const cblInput = panel.querySelector('#m9-cblen');
  const cbtInput = panel.querySelector('#m9-cbth');
  const applyBtn = panel.querySelector('#m9-cbapply');
  const resetBtn = panel.querySelector('#m9-cbreset');
  if (!scaleSel || !cminInput || !cmaxInput || !cbxInput || !cblInput || !cbtInput || !applyBtn || !resetBtn) return;

  scales.forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    scaleSel.appendChild(opt);
  });

  const defaultScale = String(meta.color_scale || __DEFAULT_SCALE__);
  const defaultCbx = Number(meta.colorbar_x);
  const defaultCbl = Number(meta.colorbar_len);
  const defaultCbt = Number(meta.colorbar_thickness);

  function setDefaults() {
    scaleSel.value = scales.includes(defaultScale) ? defaultScale : scales[0];
    cminInput.value = sMin0.toFixed(3);
    cmaxInput.value = sMax0.toFixed(3);
    cbxInput.value = Number.isFinite(defaultCbx) ? defaultCbx.toFixed(2) : "1.03";
    cblInput.value = Number.isFinite(defaultCbl) ? defaultCbl.toFixed(2) : "0.74";
    cbtInput.value = Number.isFinite(defaultCbt) ? String(Math.round(defaultCbt)) : "18";
  }

  function applyChanges() {
    let cmin = Number(cminInput.value);
    let cmax = Number(cmaxInput.value);
    if (!Number.isFinite(cmin) || !Number.isFinite(cmax)) return;
    if (cmax <= cmin) {
      cmax = cmin + 0.001;
      cmaxInput.value = cmax.toFixed(3);
    }

    const cbx = Number(cbxInput.value);
    const cbl = Number(cblInput.value);
    const cbt = Number(cbtInput.value);
    if (surfaceIdx.length) {
      Plotly.restyle(
        gd,
        {
          colorscale: scaleSel.value,
          cmin: cmin,
          cmax: cmax,
          "colorbar.x": cbx,
          "colorbar.len": cbl,
          "colorbar.thickness": cbt,
        },
        surfaceIdx
      );
    }
    if (meshIdx.length) {
      Plotly.restyle(
        gd,
        {
          colorscale: scaleSel.value,
          cmin: cmin,
          cmax: cmax,
          "colorbar.x": cbx,
          "colorbar.len": cbl,
          "colorbar.thickness": cbt,
        },
        meshIdx
      );
    }
    if (markerIdx.length) {
      Plotly.restyle(
        gd,
        {
          "marker.colorscale": scaleSel.value,
          "marker.cmin": cmin,
          "marker.cmax": cmax,
          "marker.colorbar.x": cbx,
          "marker.colorbar.len": cbl,
          "marker.colorbar.thickness": cbt,
        },
        markerIdx
      );
    }
  }

  setDefaults();
  applyBtn.addEventListener('click', applyChanges);
  resetBtn.addEventListener('click', () => {
    setDefaults();
    applyChanges();
  });
})();
"""
    return script.replace("__DEFAULT_SCALE__", json.dumps(default_scale))


def _build_contour_lines(
    data: Mat3DData,
    n_levels: int = 8,
    max_lines: int = 120,
    min_points: int = 12,
) -> list[dict]:
    if not _MPL_OK:
        return []

    speed = data.speed.copy()
    finite = np.isfinite(speed)
    if np.count_nonzero(finite) < 50:
        return []

    svals = speed[finite]
    smin = float(np.nanpercentile(svals, 10))
    smax = float(np.nanpercentile(svals, 90))
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        return []

    n_levels = max(2, int(n_levels))
    levels = np.linspace(smin, smax, n_levels)

    nc, ns = speed.shape
    xg = np.arange(ns, dtype=float)
    yg = np.arange(nc, dtype=float)

    fig_tmp, ax_tmp = plt.subplots(figsize=(3, 3))
    try:
        cs = ax_tmp.contour(xg, yg, speed, levels=levels)
    except Exception:
        plt.close(fig_tmp)
        return []

    contour_lines: list[dict] = []
    for lev, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if seg is None or len(seg) < min_points:
                continue
            ens_f = np.clip(seg[:, 0], 0, ns - 1)
            cell_f = np.clip(seg[:, 1], 0, nc - 1).round().astype(int)

            e = np.interp(ens_f, xg, data.track_e)
            n = np.interp(ens_f, xg, data.track_n)

            ens_i = np.clip(np.rint(ens_f).astype(int), 0, ns - 1)
            depth = data.depth_abs[cell_f, ens_i]
            z = -depth

            ok = np.isfinite(e) & np.isfinite(n) & np.isfinite(z)
            if np.count_nonzero(ok) < min_points:
                continue

            contour_lines.append(
                {
                    "level": float(lev),
                    "x": e[ok],
                    "y": n[ok],
                    "z": z[ok],
                }
            )

    plt.close(fig_tmp)
    if len(contour_lines) > max_lines:
        step = int(np.ceil(len(contour_lines) / max_lines))
        contour_lines = contour_lines[::step]
    return contour_lines


def read_mat_3d(
    mat_path: str | Path,
    m9_hour_shift: int = -1,
    track_is_ne: bool = False,
    ensemble_step: int = 2,
    cell_step: int = 1,
    reference_path: str | Path | None = None,
    track_epsg: int = 31370,
) -> Mat3DData:
    return read_mat_3d_basic(
        mat_path=mat_path,
        m9_hour_shift=m9_hour_shift,
        track_is_ne=track_is_ne,
        ensemble_step=ensemble_step,
        cell_step=cell_step,
        reference_path=reference_path,
        track_epsg=track_epsg,
    )

    p = Path(mat_path)
    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)

    for key in ("System", "Summary", "BottomTrack", "WaterTrack"):
        if key not in mat:
            raise ValueError(f"{p.name}: structuur '{key}' ontbreekt.")

    sys_obj = mat["System"]
    sum_obj = mat["Summary"]
    bt_obj = mat["BottomTrack"]
    wt_obj = mat["WaterTrack"]
    if not hasattr(sys_obj, "Time"):
        raise ValueError(f"{p.name}: System.Time ontbreekt.")
    if not hasattr(sum_obj, "Track"):
        raise ValueError(f"{p.name}: Summary.Track ontbreekt.")
    if not hasattr(wt_obj, "Velocity"):
        raise ValueError(f"{p.name}: WaterTrack.Velocity ontbreekt.")

    t_sec = np.asarray(sys_obj.Time, dtype=float).reshape(-1)
    track = np.asarray(sum_obj.Track, dtype=float)
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError(f"{p.name}: Summary.Track vorm onverwacht {track.shape}")
    track = track[:, :2]
    if track_is_ne:
        track = track[:, [1, 0]]

    ns = min(len(t_sec), track.shape[0])
    if ns < 2:
        raise ValueError(f"{p.name}: te weinig ensembles ({ns}).")

    t_sec = t_sec[:ns]
    track_e = np.asarray(track[:ns, 0], dtype=float)
    track_n = np.asarray(track[:ns, 1], dtype=float)

    georef_report: list[str] = []
    post_target_bearing: float | None = None
    snap_to_reference_line = False
    if not _looks_georeferenced(track_e, track_n):
        compass_ok_pre: bool | None = None
        if setup_obj is not None:
            compass_ok_pre, _, _ = _compass_status(
                setup=setup_obj,
                system=sys_obj,
                compass=compass_obj,
                ns=ns,
                check_compass=check_compass,
                force_raai_fallback=force_raai_fallback,
                mag_error_threshold=compass_mag_error_threshold,
            )
        ref_anchor_e = reference_anchor_e
        ref_anchor_n = reference_anchor_n
        ref_bearing: float | None = None
        ref_labels: list[str] = []
        if reference_path:
            path_anchor_e, path_anchor_n, path_bearing, path_label = _kmz_or_kml_reference_to_projected(reference_path, track_epsg)
            if ref_anchor_e is None:
                ref_anchor_e = path_anchor_e
            if ref_anchor_n is None:
                ref_anchor_n = path_anchor_n
            ref_bearing = path_bearing
            if path_label:
                ref_labels.append(path_label)
        use_raai_orientation = bool(raai_kmz_path) and (
            relative_track_to_raai or force_raai_fallback or (fallback_on_bad_compass and compass_ok_pre is False)
        )
        if ref_bearing is None and use_raai_orientation:
            try:
                p1, p2 = _parse_kmz_line(raai_kmz_path)
                ref_bearing = _geodesic_bearing_deg(p1, p2)
                ref_labels.append("raai-oriëntatie")
            except Exception as exc:
                georef_report.append(f"Raai-oriëntatie kon niet toegepast worden: {exc}")
        if reference_anchor_e is not None and reference_anchor_n is not None:
            ref_labels.append("RTK anker")
        if ref_anchor_e is not None and ref_anchor_n is not None:
            snap_to_reference_line = bool(ref_bearing is not None)
            post_target_bearing = ref_bearing
            track_e, track_n, georef_report_apply = _apply_reference_georef(
                track_e=track_e,
                track_n=track_n,
                anchor_e=float(ref_anchor_e),
                anchor_n=float(ref_anchor_n),
                target_bearing_deg=ref_bearing,
                snap_to_line=snap_to_reference_line,
            )
            georef_report.extend(georef_report_apply)
            if ref_labels:
                georef_report.insert(0, f"Referentiebron: {' + '.join(ref_labels)}.")

    cstart = _as_1d(getattr(sys_obj, "Cell_Start", None), ns)
    csize = _as_1d(getattr(sys_obj, "Cell_Size", None), ns)
    bed = _as_1d(getattr(bt_obj, "BT_Depth", None), ns)

    vel = _reorder_velocity(np.asarray(getattr(wt_obj, "Velocity")), ns_target=ns)
    nc, ncomp, ns_vel = vel.shape
    if ncomp < 2:
        raise ValueError(f"{p.name}: Velocity heeft minder dan 2 componenten: {vel.shape}")
    ns = min(ns, ns_vel)

    track_e = track_e[:ns]
    track_n = track_n[:ns]
    cstart = cstart[:ns]
    csize = csize[:ns]
    bed = bed[:ns]
    t_sec = t_sec[:ns]
    vel = vel[:, :, :ns]

    u = np.asarray(vel[:, 0, :], dtype=float)
    v = np.asarray(vel[:, 1, :], dtype=float)
    speed = np.sqrt(u * u + v * v)

    row_idx = np.arange(nc, dtype=float).reshape(-1, 1)
    depth_abs = cstart.reshape(1, -1) + (row_idx + 0.5) * csize.reshape(1, -1)
    depth_top_abs = cstart.reshape(1, -1) + row_idx * csize.reshape(1, -1)
    depth_bottom_abs = cstart.reshape(1, -1) + (row_idx + 1.0) * csize.reshape(1, -1)

    valid = np.isfinite(depth_abs) & np.isfinite(speed) & np.isfinite(track_e.reshape(1, -1)) & np.isfinite(
        track_n.reshape(1, -1)
    )
    valid &= depth_abs > 0.0

    bed_ok = np.isfinite(bed).reshape(1, -1)
    valid &= (~bed_ok) | (depth_abs < bed.reshape(1, -1))

    x = np.tile(track_e.reshape(1, -1), (nc, 1))
    y = np.tile(track_n.reshape(1, -1), (nc, 1))
    z = -depth_abs.copy()

    x[~valid] = np.nan
    y[~valid] = np.nan
    z[~valid] = np.nan
    speed[~valid] = np.nan
    depth_abs[~valid] = np.nan
    depth_top_abs[~valid] = np.nan
    depth_bottom_abs[~valid] = np.nan

    ens_step = max(1, int(ensemble_step))
    cell_step = max(1, int(cell_step))
    csel = np.arange(0, ns, ens_step, dtype=int)
    rsel = np.arange(0, nc, cell_step, dtype=int)

    x = x[np.ix_(rsel, csel)]
    y = y[np.ix_(rsel, csel)]
    z = z[np.ix_(rsel, csel)]
    speed = speed[np.ix_(rsel, csel)]
    depth_abs = depth_abs[np.ix_(rsel, csel)]
    depth_top_abs = depth_top_abs[np.ix_(rsel, csel)]
    depth_bottom_abs = depth_bottom_abs[np.ix_(rsel, csel)]

    track_e = track_e[csel]
    track_n = track_n[csel]
    bed = bed[csel]
    t_utc = _m9_time_to_utc(t_sec)
    t_utc = (t_utc + pd.Timedelta(hours=int(m9_hour_shift)))[csel]
    ensemble = (csel + 1).astype(int)
    speed_surface = _surface_speed_from_columns(speed)

    if (not snap_to_reference_line) and post_target_bearing is not None and np.isfinite(post_target_bearing):
        first_xy_post = _first_valid_xy(track_e, track_n)
        if first_xy_post is not None:
            base_e_post, base_n_post = first_xy_post
            rel_e_post = track_e - base_e_post
            rel_n_post = track_n - base_n_post
            measured_post = _track_net_bearing(rel_e_post, rel_n_post)
            measured_post_label = "netto verplaatsingsrichting"
            if not np.isfinite(measured_post):
                measured_post = _track_axis_bearing(rel_e_post, rel_n_post)
                measured_post_label = "hoofd-as"
            if np.isfinite(measured_post):
                candidates_post = [float(post_target_bearing), float((post_target_bearing + 180.0) % 360.0)]
                chosen_post = min(candidates_post, key=lambda cand: abs(_angle_diff_deg(cand, measured_post)))
                delta_post = _angle_diff_deg(chosen_post, measured_post)
                rel_e_post, rel_n_post = _rotate_points(rel_e_post, rel_n_post, -delta_post)
                track_e = rel_e_post + base_e_post
                track_n = rel_n_post + base_n_post
                x = np.tile(track_e.reshape(1, -1), (x.shape[0], 1))
                y = np.tile(track_n.reshape(1, -1), (y.shape[0], 1))
                georef_report.append(
                    f"Geselecteerde track extra geroteerd met {delta_post:+.3f}° op basis van {measured_post_label} naar {chosen_post:.3f}°."
                )

    if np.count_nonzero(np.isfinite(speed)) < 10:
        raise ValueError(f"{p.name}: onvoldoende geldige snelheidspunten.")

    note = ""
    if not _looks_georeferenced(track_e, track_n):
        note = "Track lijkt relatief (mogelijk niet gegeorefereerd); E/N wisselen lost dit niet op."
    elif georef_report:
        note = "Track gegeorefereerd via externe referentie."

    compass_ok: bool | None = None
    compass_reason = ""
    processing_mode = "standard"
    report_lines: tuple[str, ...] = ()
    if setup_obj is not None:
        compass_ok, compass_reason, processing_mode, report_lines = _analyze_compass_and_raai(
            p=p,
            setup=setup_obj,
            system=sys_obj,
            compass=compass_obj,
            summary=sum_obj,
            track_e=np.asarray(track[:ns, 0], dtype=float),
            track_n=np.asarray(track[:ns, 1], dtype=float),
            ns=ns,
            raai_kmz_path=raai_kmz_path,
            check_compass=check_compass,
            fallback_on_bad_compass=fallback_on_bad_compass,
            force_raai_fallback=force_raai_fallback,
            relative_track_to_raai=relative_track_to_raai,
            mag_error_threshold=compass_mag_error_threshold,
        )
        if compass_reason:
            note = f"{note} | {compass_reason}" if note else compass_reason
    if georef_report:
        report_lines = tuple(list(report_lines) + georef_report)

    return Mat3DData(
        path=p,
        x=x,
        y=y,
        z=z,
        speed=speed,
        depth_abs=depth_abs,
        depth_top_abs=depth_top_abs,
        depth_bottom_abs=depth_bottom_abs,
        track_e=track_e,
        track_n=track_n,
        bed_depth=bed,
        track_speed_surface=speed_surface,
        time_utc=t_utc,
        ensemble=ensemble,
        note=note,
        processing_mode=processing_mode,
        compass_ok=compass_ok,
        compass_reason=compass_reason,
        report_lines=report_lines,
    )


def read_mat_3d_basic(
    mat_path: str | Path,
    m9_hour_shift: int = -1,
    track_is_ne: bool = False,
    ensemble_step: int = 2,
    cell_step: int = 1,
    reference_path: str | Path | None = None,
    track_epsg: int = 31370,
) -> Mat3DData:
    p = Path(mat_path)
    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)

    for key in ("System", "Summary", "BottomTrack", "WaterTrack"):
        if key not in mat:
            raise ValueError(f"{p.name}: structuur '{key}' ontbreekt.")

    sys_obj = mat["System"]
    sum_obj = mat["Summary"]
    bt_obj = mat["BottomTrack"]
    wt_obj = mat["WaterTrack"]

    if not hasattr(sys_obj, "Time"):
        raise ValueError(f"{p.name}: System.Time ontbreekt.")
    if not hasattr(sum_obj, "Track"):
        raise ValueError(f"{p.name}: Summary.Track ontbreekt.")
    if not hasattr(wt_obj, "Velocity"):
        raise ValueError(f"{p.name}: WaterTrack.Velocity ontbreekt.")

    t_sec = np.asarray(sys_obj.Time, dtype=float).reshape(-1)
    track = np.asarray(sum_obj.Track, dtype=float)
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError(f"{p.name}: Summary.Track vorm onverwacht {track.shape}")
    track = track[:, :2]
    if track_is_ne:
        track = track[:, [1, 0]]

    ns = min(len(t_sec), track.shape[0])
    if ns < 2:
        raise ValueError(f"{p.name}: te weinig ensembles ({ns}).")

    t_sec = t_sec[:ns]
    track_e = np.asarray(track[:ns, 0], dtype=float)
    track_n = np.asarray(track[:ns, 1], dtype=float)

    georef_report: list[str] = []
    if not _looks_georeferenced(track_e, track_n) and reference_path:
        ref_anchor_e, ref_anchor_n, path_label = _kmz_or_kml_reference_to_projected(reference_path, track_epsg)
        georef_report.append(f"Referentiebron: {path_label}.")
        track_e, track_n, georef_report_apply = _apply_point_reference_georef(
            track_e=track_e,
            track_n=track_n,
            anchor_e=ref_anchor_e,
            anchor_n=ref_anchor_n,
        )
        georef_report.extend(georef_report_apply)

    cstart = _as_1d(getattr(sys_obj, "Cell_Start", None), ns)
    csize = _as_1d(getattr(sys_obj, "Cell_Size", None), ns)
    bed = _as_1d(getattr(bt_obj, "BT_Depth", None), ns)

    vel = _reorder_velocity(np.asarray(getattr(wt_obj, "Velocity")), ns_target=ns)
    nc, ncomp, ns_vel = vel.shape
    if ncomp < 2:
        raise ValueError(f"{p.name}: Velocity heeft minder dan 2 componenten: {vel.shape}")
    ns = min(ns, ns_vel)

    track_e = track_e[:ns]
    track_n = track_n[:ns]
    cstart = cstart[:ns]
    csize = csize[:ns]
    bed = bed[:ns]
    t_sec = t_sec[:ns]
    vel = vel[:, :, :ns]

    u = np.asarray(vel[:, 0, :], dtype=float)
    v = np.asarray(vel[:, 1, :], dtype=float)
    speed = np.sqrt(u * u + v * v)

    row_idx = np.arange(nc, dtype=float).reshape(-1, 1)
    depth_abs = cstart.reshape(1, -1) + (row_idx + 0.5) * csize.reshape(1, -1)
    depth_top_abs = cstart.reshape(1, -1) + row_idx * csize.reshape(1, -1)
    depth_bottom_abs = cstart.reshape(1, -1) + (row_idx + 1.0) * csize.reshape(1, -1)

    valid = np.isfinite(depth_abs) & np.isfinite(speed) & np.isfinite(track_e.reshape(1, -1)) & np.isfinite(
        track_n.reshape(1, -1)
    )
    valid &= depth_abs > 0.0

    bed_ok = np.isfinite(bed).reshape(1, -1)
    valid &= (~bed_ok) | (depth_abs < bed.reshape(1, -1))

    x = np.tile(track_e.reshape(1, -1), (nc, 1))
    y = np.tile(track_n.reshape(1, -1), (nc, 1))
    z = -depth_abs.copy()

    x[~valid] = np.nan
    y[~valid] = np.nan
    z[~valid] = np.nan
    speed[~valid] = np.nan
    depth_abs[~valid] = np.nan
    depth_top_abs[~valid] = np.nan
    depth_bottom_abs[~valid] = np.nan

    ens_step = max(1, int(ensemble_step))
    cell_step = max(1, int(cell_step))
    csel = np.arange(0, ns, ens_step, dtype=int)
    rsel = np.arange(0, nc, cell_step, dtype=int)

    x = x[np.ix_(rsel, csel)]
    y = y[np.ix_(rsel, csel)]
    z = z[np.ix_(rsel, csel)]
    speed = speed[np.ix_(rsel, csel)]
    depth_abs = depth_abs[np.ix_(rsel, csel)]
    depth_top_abs = depth_top_abs[np.ix_(rsel, csel)]
    depth_bottom_abs = depth_bottom_abs[np.ix_(rsel, csel)]

    track_e = track_e[csel]
    track_n = track_n[csel]
    bed = bed[csel]
    t_utc = _m9_time_to_utc(t_sec)
    t_utc = (t_utc + pd.Timedelta(hours=int(m9_hour_shift)))[csel]
    ensemble = (csel + 1).astype(int)
    speed_surface = _surface_speed_from_columns(speed)

    if np.count_nonzero(np.isfinite(speed)) < 10:
        raise ValueError(f"{p.name}: onvoldoende geldige snelheidspunten.")

    note = ""
    if not _looks_georeferenced(track_e, track_n):
        note = "Track lijkt relatief (mogelijk niet gegeorefereerd); E/N wisselen lost dit niet op."
    elif georef_report:
        note = "Track gegeorefereerd via externe referentie."

    return Mat3DData(
        path=p,
        x=x,
        y=y,
        z=z,
        speed=speed,
        depth_abs=depth_abs,
        depth_top_abs=depth_top_abs,
        depth_bottom_abs=depth_bottom_abs,
        track_e=track_e,
        track_n=track_n,
        bed_depth=bed,
        track_speed_surface=speed_surface,
        time_utc=t_utc,
        ensemble=ensemble,
        note=note,
        processing_mode="standard",
        compass_ok=None,
        compass_reason="",
        report_lines=tuple(georef_report),
    )


def read_mat_3d(
    mat_path: str | Path,
    m9_hour_shift: int = -1,
    track_is_ne: bool = False,
    ensemble_step: int = 2,
    cell_step: int = 1,
    measurement_mode: str = "stationair",
    reference_path: str | Path | None = None,
    left_bank_lon: float | None = None,
    left_bank_lat: float | None = None,
    right_bank_lon: float | None = None,
    right_bank_lat: float | None = None,
    track_epsg: int = 31370,
) -> Mat3DData:
    mode = str(measurement_mode).strip().lower()
    if mode == "stationair" and not reference_path:
        raise ValueError("Stationaire meting vereist een vast referentiepunt via KMZ/KML of kaartklik.")
    use_reference = reference_path if mode == "stationair" else None
    data = read_mat_3d_basic(
        mat_path=mat_path,
        m9_hour_shift=m9_hour_shift,
        track_is_ne=track_is_ne,
        ensemble_step=ensemble_step,
        cell_step=cell_step,
        reference_path=use_reference,
        track_epsg=track_epsg,
    )

    if mode == "dwarsprofiel":
        missing = [
            name
            for name, value in (
                ("left_bank_lon", left_bank_lon),
                ("left_bank_lat", left_bank_lat),
                ("right_bank_lon", right_bank_lon),
                ("right_bank_lat", right_bank_lat),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Dwarsprofiel vereist linker/rechteroever in WGS84. Ontbreekt: {', '.join(missing)}.")
        anchor_e, anchor_n, target_bearing, chosen_label = _profile_reference_from_banks(
            left_lon=float(left_bank_lon),
            left_lat=float(left_bank_lat),
            right_lon=float(right_bank_lon),
            right_lat=float(right_bank_lat),
            track_e=data.track_e,
            track_n=data.track_n,
            track_epsg=track_epsg,
        )
        new_track_e, new_track_n, profile_report = _align_track_to_profile(
            track_e=data.track_e,
            track_n=data.track_n,
            anchor_e=anchor_e,
            anchor_n=anchor_n,
            target_bearing_deg=target_bearing,
        )
        data.track_e = new_track_e
        data.track_n = new_track_n
        data.x = np.tile(new_track_e.reshape(1, -1), (data.x.shape[0], 1))
        data.y = np.tile(new_track_n.reshape(1, -1), (data.y.shape[0], 1))
        extra = [f"Profielmodus: {chosen_label}."] + profile_report
        data.processing_mode = "dwarsprofiel"
        data.note = "Dwarsprofiel uitgelijnd tussen linker- en rechteroever."
        data.report_lines = tuple(list(data.report_lines) + extra)
        return data

    if mode != "stationair":
        raise ValueError(f"Onbekend measurement_mode: {measurement_mode}")
    if use_reference:
        t0 = pd.Timestamp(data.time_utc[0])
        elapsed_min = np.array([(pd.Timestamp(t) - t0).total_seconds() / 60.0 for t in data.time_utc], dtype=float)
        data.track_e = elapsed_min
        data.track_n = np.zeros_like(elapsed_min)
        data.x = np.tile(data.track_e.reshape(1, -1), (data.x.shape[0], 1))
        data.y = np.zeros_like(data.x)
        data.processing_mode = "stationair"
        data.note = "Stationaire meting op vaste positie, uitgezet als tijdslijn."
        data.report_lines = tuple(list(data.report_lines) + ["Stationaire tijdslijn actief: x-as = minuten sinds start."])
    return data


def _nan_stat(values: np.ndarray, reducer: str) -> float | None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if reducer == "min":
        return float(np.nanmin(arr))
    if reducer == "max":
        return float(np.nanmax(arr))
    if reducer == "mean":
        return float(np.nanmean(arr))
    raise ValueError(f"Onbekende reducer: {reducer}")


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.zeros(0, dtype=float)
    out = np.full(arr.shape[1], np.nan, dtype=float)
    for i in range(arr.shape[1]):
        col = arr[:, i]
        ok = np.isfinite(col)
        if np.any(ok):
            out[i] = float(np.nanmean(col[ok]))
    return out


def _cumulative_track_distance(track_e: np.ndarray, track_n: np.ndarray) -> np.ndarray:
    te = np.asarray(track_e, dtype=float).reshape(-1)
    tn = np.asarray(track_n, dtype=float).reshape(-1)
    n = min(len(te), len(tn))
    out = np.zeros(n, dtype=float)
    for i in range(1, n):
        if np.isfinite(te[i - 1]) and np.isfinite(tn[i - 1]) and np.isfinite(te[i]) and np.isfinite(tn[i]):
            out[i] = out[i - 1] + float(np.hypot(te[i] - te[i - 1], tn[i] - tn[i - 1]))
        else:
            out[i] = out[i - 1]
    return out


def build_mat_preview_summary(
    mat_path: str | Path,
    m9_hour_shift: int = -1,
    track_is_ne: bool = False,
    ensemble_step: int = 2,
    cell_step: int = 1,
) -> MatPreviewSummary:
    data = read_mat_3d_basic(
        mat_path=mat_path,
        m9_hour_shift=m9_hour_shift,
        track_is_ne=track_is_ne,
        ensemble_step=ensemble_step,
        cell_step=cell_step,
        reference_path=None,
    )
    mat = sio.loadmat(str(Path(mat_path)), squeeze_me=True, struct_as_record=False)
    sum_obj = mat.get("Summary", None)
    discharge_mean: float | None = None
    if sum_obj is not None and hasattr(sum_obj, "Total_Q"):
        total_q = np.asarray(getattr(sum_obj, "Total_Q"), dtype=float).reshape(-1)
        discharge_mean = _nan_stat(total_q, "mean")

    start_time = pd.Timestamp(data.time_utc[0]) if len(data.time_utc) else None
    stop_time = pd.Timestamp(data.time_utc[-1]) if len(data.time_utc) else None
    if len(data.time_utc):
        t0 = pd.Timestamp(data.time_utc[0])
        elapsed_min = np.array([(pd.Timestamp(t) - t0).total_seconds() / 60.0 for t in data.time_utc], dtype=float)
    else:
        elapsed_min = np.arange(len(data.track_e), dtype=float)
    mean_speed_series = _nanmean_axis0(data.speed)
    distance_m = _cumulative_track_distance(data.track_e, data.track_n)
    return MatPreviewSummary(
        path=data.path,
        track_e=np.asarray(data.track_e, dtype=float).copy(),
        track_n=np.asarray(data.track_n, dtype=float).copy(),
        distance_m=distance_m,
        elapsed_min=elapsed_min,
        bed_depth=np.asarray(data.bed_depth, dtype=float).copy(),
        surface_speed=np.asarray(data.track_speed_surface, dtype=float).copy(),
        mean_speed_series=mean_speed_series,
        start_time=start_time,
        stop_time=stop_time,
        speed_min=_nan_stat(data.speed, "min"),
        speed_max=_nan_stat(data.speed, "max"),
        speed_mean=_nan_stat(data.speed, "mean"),
        discharge_mean=discharge_mean,
        ensembles=int(len(data.ensemble)),
        note=data.note,
    )


def _format_preview_time(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return "n/a"
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _format_preview_value(value: float | None, ndigits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{int(ndigits)}f}"


def build_3d_figure(
    datasets: list[Mat3DData],
    opacity: float = 0.92,
    use_cell_coloring: bool = False,
    show_contours: bool = True,
    contour_levels: int = 8,
    contour_max_lines: int = 120,
    color_scale: str = "Turbo",
    colorbar_x: float = 1.03,
    colorbar_len: float = 0.74,
    colorbar_thickness: int = 18,
) -> go.Figure:
    if not datasets:
        raise ValueError("Geen datasets beschikbaar voor figuur.")

    speeds = [d.speed[np.isfinite(d.speed)] for d in datasets if np.count_nonzero(np.isfinite(d.speed)) > 0]
    if not speeds:
        raise ValueError("Geen geldige snelheden gevonden.")

    smin = min(float(np.nanpercentile(s, 2)) for s in speeds)
    smax = max(float(np.nanpercentile(s, 98)) for s in speeds)
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        smin = min(float(np.nanmin(s)) for s in speeds)
        smax = max(float(np.nanmax(s)) for s in speeds)

    fig = go.Figure()
    stationair_view = bool(datasets) and all(d.processing_mode == "stationair" for d in datasets)
    track_colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#ff7f00",
        "#984ea3",
        "#a65628",
        "#f781bf",
        "#999999",
    ]

    tr_e = [d.track_e[np.isfinite(d.track_e)] for d in datasets if np.count_nonzero(np.isfinite(d.track_e)) > 0]
    tr_n = [d.track_n[np.isfinite(d.track_n)] for d in datasets if np.count_nonzero(np.isfinite(d.track_n)) > 0]
    if tr_e and tr_n:
        all_e = np.concatenate(tr_e)
        all_n = np.concatenate(tr_n)
        ctr_e = float(np.nanmedian(all_e))
        ctr_n = float(np.nanmedian(all_n))
        span = max(float(np.nanmax(all_e) - np.nanmin(all_e)), float(np.nanmax(all_n) - np.nanmin(all_n)), 1.0)
    else:
        ctr_e = 0.0
        ctr_n = 0.0
        span = 100.0
    label_offset_cap = min(4.0, max(1.2, 0.04 * span))

    for i, d in enumerate(datasets):
        col = track_colors[i % len(track_colors)]
        grp = d.path.name
        t_iso = np.array([pd.Timestamp(t).tz_convert("UTC").isoformat() for t in d.time_utc], dtype=object)

        nr, nc = d.speed.shape
        surf_text = np.empty((nr, nc), dtype=object)
        for c in range(nc):
            ens = int(d.ensemble[c]) if c < len(d.ensemble) else (c + 1)
            ts = t_iso[c] if c < len(t_iso) else ""
            for r in range(nr):
                sp = d.speed[r, c]
                dep = abs(float(d.z[r, c])) if np.isfinite(d.z[r, c]) else np.nan
                sp_txt = f"{float(sp):.3f}" if np.isfinite(sp) else "n/a"
                dep_txt = f"{dep:.3f}" if np.isfinite(dep) else "n/a"
                if stationair_view:
                    coord_txt = f"Tijd sinds start={float(d.x[r, c]):.2f} min<br>Vaste positie"
                else:
                    coord_txt = f"E={float(d.x[r, c]):.3f}<br>N={float(d.y[r, c]):.3f}"
                surf_text[r, c] = (
                    f"<b>{d.path.name} waterkolom</b><br>"
                    f"ensemble={ens}<br>"
                    f"{ts}<br>"
                    f"{coord_txt}<br>"
                    f"Diepte={dep_txt} m<br>"
                    f"Snelheid={sp_txt} m/s<br>"
                    f"Verwerking={html.escape(d.processing_mode)}"
                )
                if d.note:
                    surf_text[r, c] += f"<br>Note={html.escape(d.note)}"

        track_text: list[str] = []
        for k in range(len(d.track_e)):
            ens = int(d.ensemble[k]) if k < len(d.ensemble) else (k + 1)
            ts = t_iso[k] if k < len(t_iso) else ""
            sp = d.track_speed_surface[k] if k < len(d.track_speed_surface) else np.nan
            sp_txt = f"{float(sp):.3f}" if np.isfinite(sp) else "n/a"
            if stationair_view:
                coord_txt = f"Tijd sinds start={float(d.track_e[k]):.2f} min<br>Vaste positie"
            else:
                coord_txt = f"E={float(d.track_e[k]):.3f}<br>N={float(d.track_n[k]):.3f}"
            track_text.append(
                f"<b>Track oppervlak</b><br>"
                f"ensemble={ens}<br>"
                f"{ts}<br>"
                f"{coord_txt}<br>"
                f"Snelheid (bovenste cel)={sp_txt} m/s<br>"
                f"Verwerking={html.escape(d.processing_mode)}"
            )
            if d.note:
                track_text[-1] += f"<br>Note={html.escape(d.note)}"

        if use_cell_coloring:
            e_edges = _edges_from_centers(d.track_e)
            n_edges = _edges_from_centers(d.track_n)

            xv: list[float] = []
            yv: list[float] = []
            zv: list[float] = []
            iv: list[int] = []
            jv: list[int] = []
            kv: list[int] = []
            face_intensity: list[float] = []
            txt: list[str] = []

            nr, nc = d.speed.shape
            ncols = min(nc, max(0, len(e_edges) - 1), max(0, len(n_edges) - 1))
            for c in range(ncols):
                e0 = float(e_edges[c])
                e1 = float(e_edges[c + 1])
                n0 = float(n_edges[c])
                n1 = float(n_edges[c + 1])
                if not (np.isfinite(e0) and np.isfinite(e1) and np.isfinite(n0) and np.isfinite(n1)):
                    continue
                for r in range(nr):
                    sp = float(d.speed[r, c])
                    z_top = -float(d.depth_top_abs[r, c]) if np.isfinite(d.depth_top_abs[r, c]) else np.nan
                    z_bot = -float(d.depth_bottom_abs[r, c]) if np.isfinite(d.depth_bottom_abs[r, c]) else np.nan
                    if not (np.isfinite(sp) and np.isfinite(z_top) and np.isfinite(z_bot)):
                        continue
                    if z_top <= z_bot:
                        continue

                    base = len(xv)
                    xv.extend([e0, e1, e1, e0])
                    yv.extend([n0, n1, n1, n0])
                    zv.extend([z_top, z_top, z_bot, z_bot])
                    cell_text = surf_text[r, c]
                    txt.extend([cell_text, cell_text, cell_text, cell_text])

                    iv.extend([base + 0, base + 0])
                    jv.extend([base + 1, base + 2])
                    kv.extend([base + 2, base + 3])
                    face_intensity.extend([sp, sp])

            if iv:
                fig.add_trace(
                    go.Mesh3d(
                        x=xv,
                        y=yv,
                        z=zv,
                        i=iv,
                        j=jv,
                        k=kv,
                        intensity=face_intensity,
                        intensitymode="cell",
                        colorscale=color_scale,
                        cmin=smin,
                        cmax=smax,
                        flatshading=True,
                        opacity=float(opacity),
                        showscale=(i == 0),
                        colorbar=dict(
                            title="Snelheid (m/s)",
                            x=float(colorbar_x),
                            len=float(colorbar_len),
                            thickness=int(colorbar_thickness),
                        ),
                        text=txt,
                        hoverinfo="text",
                        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
                        name=f"{d.path.name} meetcellen",
                        legendgroup=grp,
                        showlegend=True,
                    )
                )
        else:
            fig.add_trace(
                go.Surface(
                    x=d.x,
                    y=d.y,
                    z=d.z,
                    surfacecolor=d.speed,
                    hovertext=surf_text,
                    hoverinfo="text",
                    colorscale=color_scale,
                    cmin=smin,
                    cmax=smax,
                    opacity=float(opacity),
                    showscale=(i == 0),
                    colorbar=dict(
                        title="Snelheid (m/s)",
                        x=float(colorbar_x),
                        len=float(colorbar_len),
                        thickness=int(colorbar_thickness),
                    ),
                    name=f"{d.path.name} waterkolom",
                    legendgroup=grp,
                    showlegend=True,
                )
            )

        z_track = np.zeros_like(d.track_e, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=d.track_e,
                y=d.track_n,
                z=z_track,
                mode="lines",
                hovertext=track_text,
                hoverinfo="text",
                line=dict(color=col, width=6),
                name=f"{d.path.name} track",
                legendgroup=grp,
                showlegend=True,
            )
        )

        valid_track = np.isfinite(d.track_e) & np.isfinite(d.track_n)
        if np.count_nonzero(valid_track) >= 2:
            idx = np.where(valid_track)[0]
            local_e = d.track_e[idx]
            local_n = d.track_n[idx]
            local_span = max(
                float(np.nanmax(local_e) - np.nanmin(local_e)),
                float(np.nanmax(local_n) - np.nanmin(local_n)),
                0.25,
            )
            local_label_offset = min(label_offset_cap, max(0.6, 0.35 * local_span))
            k_mid = int(idx[len(idx) // 2])
            k_prev = int(idx[max(0, len(idx) // 2 - 1)])
            k_next = int(idx[min(len(idx) - 1, len(idx) // 2 + 1)])

            a_e = float(d.track_e[k_mid])
            a_n = float(d.track_n[k_mid])
            compact_label = d.processing_mode in {"raai_projected", "raai_fallback", "stationair"} or local_span <= 3.0
            t_e = float(d.track_e[k_next] - d.track_e[k_prev])
            t_n = float(d.track_n[k_next] - d.track_n[k_prev])
            p_e, p_n = -t_n, t_e
            nrm = float(np.hypot(p_e, p_n))
            if nrm < 1e-9:
                p_e, p_n = a_e - ctr_e, a_n - ctr_n
                nrm = float(np.hypot(p_e, p_n))
            if nrm < 1e-9:
                p_e, p_n = 1.0, 0.0
                nrm = 1.0
            p_e /= nrm
            p_n /= nrm

            outward = (a_e - ctr_e) * p_e + (a_n - ctr_n) * p_n
            if outward < 0.0:
                p_e *= -1.0
                p_n *= -1.0
            if i % 2 == 1:
                p_e *= -1.0
                p_n *= -1.0

            if d.processing_mode == "stationair":
                lbl_e = a_e
                lbl_n = 0.18 * ((i % 3) - 1)
                lbl_z = 0.7 + 0.18 * (i // 3)
                fig.add_trace(
                    go.Scatter3d(
                        x=[a_e, lbl_e],
                        y=[a_n, lbl_n],
                        z=[0.0, lbl_z],
                        mode="lines",
                        line=dict(color=col, width=3, dash="dot"),
                        name=f"{d.path.name} label-lijn",
                        legendgroup=grp,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[lbl_e],
                        y=[lbl_n],
                        z=[lbl_z],
                        mode="text",
                        text=[d.path.stem],
                        textposition="top center",
                        textfont=dict(color=col, size=12),
                        name=f"{d.path.name} label",
                        legendgroup=grp,
                        showlegend=False,
                        hovertemplate=f"{html.escape(d.path.name)}<extra></extra>",
                    )
                )
            elif compact_label:
                lbl_e = a_e + p_e * min(0.45, local_label_offset)
                lbl_n = a_n + p_n * min(0.45, local_label_offset)
                lbl_z = 0.22
                fig.add_trace(
                    go.Scatter3d(
                        x=[lbl_e],
                        y=[lbl_n],
                        z=[lbl_z],
                        mode="markers+text",
                        marker=dict(size=5, color=col),
                        text=[d.path.stem],
                        textposition="top center",
                        name=f"{d.path.name} label",
                        legendgroup=grp,
                        showlegend=False,
                        hovertemplate=f"{html.escape(d.path.name)}<extra></extra>",
                    )
                )
            else:
                lbl_e = a_e + p_e * local_label_offset
                lbl_n = a_n + p_n * local_label_offset
                lbl_z = 0.5

                fig.add_trace(
                    go.Scatter3d(
                        x=[a_e, lbl_e],
                        y=[a_n, lbl_n],
                        z=[0.0, lbl_z],
                        mode="lines",
                        line=dict(color=col, width=4, dash="dot"),
                        name=f"{d.path.name} label-lijn",
                        legendgroup=grp,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[lbl_e],
                        y=[lbl_n],
                        z=[lbl_z],
                        mode="markers+text",
                        marker=dict(size=5, color=col),
                        text=[d.path.stem],
                        textposition="top center",
                        name=f"{d.path.name} label",
                        legendgroup=grp,
                        showlegend=False,
                        hovertemplate=f"{html.escape(d.path.name)}<extra></extra>",
                    )
                )

        if np.count_nonzero(np.isfinite(d.bed_depth)) > 2:
            fig.add_trace(
                go.Scatter3d(
                    x=d.track_e,
                    y=d.track_n,
                    z=-d.bed_depth,
                    mode="lines",
                    line=dict(color="rgba(100,70,20,0.8)", width=3, dash="dot"),
                    name=f"{d.path.name} bodem",
                    legendgroup=grp,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if show_contours and _MPL_OK:
            lines = _build_contour_lines(
                data=d,
                n_levels=contour_levels,
                max_lines=contour_max_lines,
            )
            for ln in lines:
                fig.add_trace(
                    go.Scatter3d(
                        x=ln["x"],
                        y=ln["y"],
                        z=ln["z"],
                        mode="lines",
                        line=dict(color="rgba(255,255,255,0.65)", width=2),
                        name=f"{d.path.name} contour",
                        legendgroup=grp,
                        showlegend=False,
                        hovertemplate=f"Contour snelheid ~ {ln['level']:.3f} m/s<extra></extra>",
                    )
                )

    fig.update_layout(
        title="3D verticale waterkolom en stroming",
        template="plotly_white",
        scene=dict(
            xaxis_title=("Tijd sinds start (min)" if stationair_view else "Easting (m)"),
            yaxis_title=("Vaste positie" if stationair_view else "Northing (m)"),
            zaxis_title="Diepte (m, negatief)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.75)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=10, r=10, b=10, t=45),
        meta=dict(
            speed_min=float(smin),
            speed_max=float(smax),
            color_scale=str(color_scale),
            colorbar_x=float(colorbar_x),
            colorbar_len=float(colorbar_len),
            colorbar_thickness=int(colorbar_thickness),
        ),
    )
    return fig


def _speed_to_kml_color(speed: float, smin: float, smax: float) -> str:
    if not np.isfinite(speed):
        return "ff808080"  # aabbggrr
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        t = 0.5
    else:
        t = float(np.clip((speed - smin) / (smax - smin), 0.0, 1.0))

    if mpl_cm is not None:
        r_f, g_f, b_f, a_f = mpl_cm.turbo(t)
        r = int(np.clip(255 * r_f, 0, 255))
        g = int(np.clip(255 * g_f, 0, 255))
        b = int(np.clip(255 * b_f, 0, 255))
        a = int(np.clip(255 * a_f, 0, 255))
    else:
        r = int(255 * t)
        g = int(255 * (1.0 - abs(2.0 * t - 1.0)))
        b = int(255 * (1.0 - t))
        a = 255

    return f"{a:02x}{b:02x}{g:02x}{r:02x}"  # KML uses aabbggrr


def _dataset_to_tri_mesh(
    d: Mat3DData,
    ensemble_step: int = 1,
    cell_step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    ridx = _sample_idx_with_last(d.x.shape[0], cell_step)
    cidx = _sample_idx_with_last(d.x.shape[1], ensemble_step)
    if ridx.size < 2 or cidx.size < 2:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)

    x = d.x[np.ix_(ridx, cidx)]
    y = d.y[np.ix_(ridx, cidx)]
    z = d.z[np.ix_(ridx, cidx)]

    nr, nc = x.shape
    verts = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1))).astype(float)
    valid = np.isfinite(verts).all(axis=1)

    faces: list[list[int]] = []
    for r in range(nr - 1):
        row0 = r * nc
        row1 = (r + 1) * nc
        for c in range(nc - 1):
            v0 = row0 + c
            v1 = v0 + 1
            v2 = row1 + c
            v3 = v2 + 1
            if valid[v0] and valid[v1] and valid[v2]:
                faces.append([v0, v1, v2])
            if valid[v2] and valid[v1] and valid[v3]:
                faces.append([v2, v1, v3])

    if not faces:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)

    faces_a = np.asarray(faces, dtype=np.int64)
    used = np.unique(faces_a.ravel())
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    verts = verts[used]
    faces_a = remap[faces_a]
    return verts, faces_a


def _write_collada_ge(vertices: np.ndarray, faces: np.ndarray, out_dae: str | Path) -> None:
    if vertices.size == 0 or faces.size == 0:
        raise ValueError("Lege mesh kan niet naar DAE worden geschreven.")

    pos_flat = " ".join(f"{v:.6f}" for v in vertices.reshape(-1))

    p_data: list[int] = []
    for f in faces:
        p_data.extend([int(f[0]), int(f[1]), int(f[2])])
    p_flat = " ".join(str(v) for v in p_data)

    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    dae_txt = f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor><authoring_tool>M9 Waterkolom Export</authoring_tool></contributor>
    <created>{now}</created>
    <modified>{now}</modified>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_effects>
    <effect id="mat-effect">
      <profile_COMMON>
        <technique sid="common">
          <lambert>
            <diffuse><color>0.30 0.60 0.90 1</color></diffuse>
          </lambert>
        </technique>
        <extra>
          <technique profile="GOOGLEEARTH">
            <double_sided>1</double_sided>
          </technique>
        </extra>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="mat-material" name="mat-material"><instance_effect url="#mat-effect"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="mesh-geometry" name="mesh-geometry">
      <mesh>
        <source id="mesh-positions">
          <float_array id="mesh-positions-array" count="{vertices.shape[0] * 3}">{pos_flat}</float_array>
          <technique_common>
            <accessor source="#mesh-positions-array" count="{vertices.shape[0]}" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="mesh-vertices">
          <input semantic="POSITION" source="#mesh-positions"/>
        </vertices>
        <triangles count="{faces.shape[0]}" material="mat-material">
          <input semantic="VERTEX" source="#mesh-vertices" offset="0"/>
          <p>{p_flat}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="mesh-node" name="mesh-node">
        <instance_geometry url="#mesh-geometry">
          <bind_material>
            <technique_common>
              <instance_material symbol="mat-material" target="#mat-material"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
"""
    outp = Path(out_dae)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(dae_txt, encoding="utf-8")


def build_dae_export(
    datasets: list[Mat3DData],
    out_dae: str | Path,
    ensemble_step: int = 1,
    cell_step: int = 1,
) -> tuple[float, float, float]:
    if not datasets:
        raise ValueError("Geen datasets beschikbaar voor DAE.")

    verts_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    offset = 0

    for d in datasets:
        v, f = _dataset_to_tri_mesh(d, ensemble_step=ensemble_step, cell_step=cell_step)
        if v.size == 0 or f.size == 0:
            continue
        verts_all.append(v)
        faces_all.append(f + offset)
        offset += v.shape[0]

    if not verts_all:
        raise ValueError("Geen geldige triangulatie voor DAE (te veel NaN of te weinig data).")

    verts = np.vstack(verts_all)
    faces = np.vstack(faces_all)

    origin_e = float(np.nanmedian(verts[:, 0]))
    origin_n = float(np.nanmedian(verts[:, 1]))
    verts_local = verts.copy()
    verts_local[:, 0] -= origin_e
    verts_local[:, 1] -= origin_n

    _write_collada_ge(verts_local, faces, out_dae)
    z_min = float(np.nanmin(verts_local[:, 2]))
    model_altitude = max(2.0, -z_min + 2.0)
    return origin_e, origin_n, model_altitude


def build_kmz_model(
    dae_path: str | Path,
    out_kmz: str | Path,
    origin_e: float,
    origin_n: float,
    track_epsg: int = 31370,
    model_altitude_m: float = 2.0,
) -> None:
    if not _PROJ_OK:
        raise RuntimeError("pyproj niet beschikbaar; KMZ model export vereist coordinatentransformatie.")

    dae_path = Path(dae_path)
    if not dae_path.exists():
        raise FileNotFoundError(f"DAE niet gevonden: {dae_path}")

    tr = Transformer.from_crs(f"EPSG:{int(track_epsg)}", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(float(origin_e), float(origin_n))
    if not np.isfinite(lon) or not np.isfinite(lat) or abs(float(lon)) > 180 or abs(float(lat)) > 90:
        raise ValueError(
            f"Ongeldige geolocatie voor KMZ model: lon={lon}, lat={lat}. Controleer Track EPSG ({track_epsg})."
        )

    dae_name = dae_path.name
    dae_arcname = f"models/{dae_name}"
    doc_kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{html.escape(dae_name)}</name>
    <Placemark>
      <name>3D waterkolom model</name>
      <Model>
        <altitudeMode>relativeToGround</altitudeMode>
        <Location>
          <longitude>{float(lon):.8f}</longitude>
          <latitude>{float(lat):.8f}</latitude>
          <altitude>{float(model_altitude_m):.3f}</altitude>
        </Location>
        <Orientation>
          <heading>0</heading>
          <tilt>0</tilt>
          <roll>0</roll>
        </Orientation>
        <Scale>
          <x>1</x><y>1</y><z>1</z>
        </Scale>
        <Link><href>{html.escape(dae_arcname)}</href></Link>
      </Model>
    </Placemark>
  </Document>
</kml>
"""

    outp = Path(out_kmz)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outp, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", doc_kml.encode("utf-8"))
        zf.write(dae_path, arcname=dae_arcname)


def build_kml_export(
    datasets: list[Mat3DData],
    out_kml: str | Path,
    track_epsg: int = 31370,
    vertical_exaggeration: float = 4.0,
    ensemble_step: int = 8,
    cell_step: int = 2,
) -> None:
    if not _PROJ_OK:
        raise RuntimeError("pyproj niet beschikbaar; KML export vereist coordinatentransformatie.")
    if not datasets:
        raise ValueError("Geen datasets beschikbaar voor KML.")

    speeds = [d.speed[np.isfinite(d.speed)] for d in datasets if np.count_nonzero(np.isfinite(d.speed)) > 0]
    if not speeds:
        raise ValueError("Geen geldige snelheden gevonden voor KML.")

    smin = min(float(np.nanpercentile(s, 2)) for s in speeds)
    smax = max(float(np.nanpercentile(s, 98)) for s in speeds)
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        smin = min(float(np.nanmin(s)) for s in speeds)
        smax = max(float(np.nanmax(s)) for s in speeds)

    tr = Transformer.from_crs(f"EPSG:{int(track_epsg)}", "EPSG:4326", always_xy=True)

    ens_step = max(1, int(ensemble_step))
    c_step = max(1, int(cell_step))
    z_scale = max(0.1, float(vertical_exaggeration))

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append("<Document>")
    lines.append("<name>M9 3D waterkolom</name>")
    lines.append("<description>Track + verticale waterkolom lijnen (diepte als negatieve altitude).</description>")
    lines.append(
        "<Style id=\"trackStyle\"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>"
    )

    for d in datasets:
        name = html.escape(d.path.name)
        lines.append(f"<Folder><name>{name}</name>")

        lon, lat = tr.transform(d.track_e, d.track_n)
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        ok_track = np.isfinite(lon) & np.isfinite(lat)
        if np.count_nonzero(ok_track) < 2:
            lines.append("</Folder>")
            continue

        coords_track = " ".join(f"{lon[i]:.8f},{lat[i]:.8f},0" for i in np.where(ok_track)[0])
        lines.append("<Placemark>")
        lines.append(f"<name>{name} track</name>")
        lines.append("<styleUrl>#trackStyle</styleUrl>")
        lines.append("<LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>")
        lines.append(f"<coordinates>{coords_track}</coordinates>")
        lines.append("</LineString>")
        lines.append("</Placemark>")

        n_ens = min(len(d.track_e), d.speed.shape[1], len(d.ensemble), len(d.time_utc))
        for j in range(0, n_ens, ens_step):
            if not np.isfinite(lon[j]) or not np.isfinite(lat[j]):
                continue

            col_depth = d.depth_abs[:, j]
            col_speed = d.speed[:, j]
            valid = np.isfinite(col_depth) & np.isfinite(col_speed)
            valid_idx = np.where(valid)[0]
            if valid_idx.size < 2:
                continue

            valid_idx = valid_idx[::c_step]
            if valid_idx.size < 2:
                continue

            sp_col = float(np.nanmean(col_speed[valid_idx]))
            kml_color = _speed_to_kml_color(sp_col, smin, smax)

            coords = []
            for r in valid_idx:
                alt = -float(col_depth[r]) * z_scale
                coords.append(f"{lon[j]:.8f},{lat[j]:.8f},{alt:.3f}")
            coord_str = " ".join(coords)

            ts = pd.Timestamp(d.time_utc[j]).tz_convert("UTC").isoformat()
            desc = (
                f"bestand={name}; ensemble={int(d.ensemble[j])}; tijd={ts}; "
                f"E={float(d.track_e[j]):.3f}; N={float(d.track_n[j]):.3f}; "
                f"gem_snelheid={sp_col:.3f} m/s"
            )
            desc = html.escape(desc)

            lines.append("<Placemark>")
            lines.append(f"<name>{name} kolom ens {int(d.ensemble[j])}</name>")
            lines.append(f"<description>{desc}</description>")
            lines.append("<Style>")
            lines.append(f"<LineStyle><color>{kml_color}</color><width>2</width></LineStyle>")
            lines.append("</Style>")
            lines.append("<LineString><tessellate>0</tessellate><altitudeMode>relativeToGround</altitudeMode>")
            lines.append(f"<coordinates>{coord_str}</coordinates>")
            lines.append("</LineString>")
            lines.append("</Placemark>")

        lines.append("</Folder>")

    lines.append("</Document>")
    lines.append("</kml>")

    outp = Path(out_kml)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines), encoding="utf-8")


def build_osm_alternative_map(
    datasets: list[Mat3DData],
    out_html: str | Path,
    track_epsg: int = 31370,
    marker_step: int = 20,
    three_d_html_name: str | None = None,
) -> None:
    if not _MAP_OK:
        raise RuntimeError("folium/pyproj niet beschikbaar.")
    if not datasets:
        raise ValueError("Geen datasets beschikbaar voor OSM-kaart.")

    tr = Transformer.from_crs(f"EPSG:{int(track_epsg)}", "EPSG:4326", always_xy=True)
    lat_all: list[np.ndarray] = []
    lon_all: list[np.ndarray] = []
    geo_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for d in datasets:
        lon, lat = tr.transform(d.track_e, d.track_n)
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        if np.count_nonzero(ok) < 2:
            raise ValueError(f"{d.path.name}: onvoldoende geldige coordinaatpunten voor OSM.")
        geo_cache.append((lat, lon, ok))
        lat_all.append(lat[ok])
        lon_all.append(lon[ok])

    center = [float(np.nanmedian(np.concatenate(lat_all))), float(np.nanmedian(np.concatenate(lon_all)))]
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

    if three_d_html_name:
        link = html.escape(str(three_d_html_name))
        popup_html = (
            f'<b>3D figuur</b><br>'
            f'<a href="{link}" target="_blank">Open 3D HTML</a>'
        )
        folium.Marker(
            location=center,
            tooltip="Open 3D figuur",
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(m)

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#17becf", "#a65628", "#f781bf"]
    step = max(1, int(marker_step))

    for i, d in enumerate(datasets):
        color = colors[i % len(colors)]
        lat, lon, ok = geo_cache[i]
        grp = folium.FeatureGroup(name=f"{d.path.name} ({int(np.count_nonzero(ok))} pt)", show=True)

        folium.PolyLine(np.column_stack([lat[ok], lon[ok]]).tolist(), color=color, weight=4, opacity=0.9).add_to(grp)

        n = min(len(lat), len(d.ensemble), len(d.time_utc), len(d.track_e), len(d.track_n), len(d.track_speed_surface))
        for k in range(0, n, step):
            if not ok[k]:
                continue
            ts = pd.Timestamp(d.time_utc[k]).tz_convert("UTC").isoformat()
            sp = d.track_speed_surface[k]
            sp_txt = f"{float(sp):.3f}" if np.isfinite(sp) else "n/a"
            tip = (
                f"<b>{d.path.name}</b><br>"
                f"ensemble={int(d.ensemble[k])}<br>"
                f"{ts}<br>"
                f"E={float(d.track_e[k]):.3f} N={float(d.track_n[k]):.3f}<br>"
                f"lat={float(lat[k]):.7f} lon={float(lon[k]):.7f}<br>"
                f"Snelheid (bovenste cel)={sp_txt} m/s"
            )
            folium.CircleMarker(
                location=[float(lat[k]), float(lon[k])],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                opacity=0.9,
                tooltip=folium.Tooltip(tip, sticky=True),
            ).add_to(grp)

        grp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    outp = Path(out_html)
    outp.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(outp))
    add_interactive_html_saver(outp)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M9 MAT multi 3D waterkolom")
        self.geometry("1180x820")

        self.var_out = tk.StringVar(value=str(Path.cwd() / "m9_waterkolom_3d.html"))
        self.var_m9_shift = tk.StringVar(value="-1")
        self.var_track_is_ne = tk.BooleanVar(value=False)
        self.var_ens_step = tk.IntVar(value=2)
        self.var_cell_step = tk.IntVar(value=1)
        self.var_opacity = tk.DoubleVar(value=0.92)
        self.var_use_cell_coloring = tk.BooleanVar(value=False)
        self.var_show_contours = tk.BooleanVar(value=True)
        self.var_contour_levels = tk.IntVar(value=8)
        self.var_contour_max_lines = tk.IntVar(value=120)
        self.var_colorbar_x = tk.DoubleVar(value=1.03)
        self.var_colorbar_len = tk.DoubleVar(value=0.74)
        self.var_colorbar_thickness = tk.IntVar(value=18)
        self.var_open_after = tk.BooleanVar(value=True)
        self.var_colorscale = tk.StringVar(value="Turbo")
        self.var_make_osm_alt = tk.BooleanVar(value=True)
        self.var_osm_add_3d_link = tk.BooleanVar(value=True)
        self.var_track_epsg = tk.StringVar(value="31370")
        self.var_measurement_mode = tk.StringVar(value="stationair")
        self.var_reference_kmz = tk.StringVar(value="")
        self.var_left_bank_lon = tk.StringVar(value="")
        self.var_left_bank_lat = tk.StringVar(value="")
        self.var_right_bank_lon = tk.StringVar(value="")
        self.var_right_bank_lat = tk.StringVar(value="")
        self.var_osm_marker_step = tk.IntVar(value=20)
        self.var_make_kml = tk.BooleanVar(value=True)
        self.var_kml_vertical_exag = tk.DoubleVar(value=4.0)
        self.var_kml_ens_step = tk.IntVar(value=8)
        self.var_kml_cell_step = tk.IntVar(value=2)
        self.var_make_dae = tk.BooleanVar(value=True)
        self.var_make_kmz_model = tk.BooleanVar(value=True)
        self.var_dae_ens_step = tk.IntVar(value=1)
        self.var_dae_cell_step = tk.IntVar(value=1)
        self._generated_reference_path: Path | None = None
        self._preview_window: tk.Toplevel | None = None
        self._loaded_preview: list[MatPreviewSummary] = []
        self._loaded_signature: tuple[object, ...] | None = None

        self._mat_files: list[Path] = []
        self._build()
        for var in (self.var_m9_shift, self.var_track_is_ne, self.var_ens_step, self.var_cell_step):
            var.trace_add("write", self._on_load_input_changed)

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(3, weight=1)

        files_box = ttk.LabelFrame(frm, text="MAT-bestanden", padding=10)
        files_box.grid(row=0, column=0, sticky="nsew")

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

        opt = ttk.LabelFrame(frm, text="3D opties", padding=10)
        opt.grid(row=1, column=0, sticky="we", pady=(10, 0))

        ttk.Label(opt, text="M9 tijdshift (uren):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            opt,
            textvariable=self.var_m9_shift,
            values=["-2", "-1", "0", "+1", "+2"],
            width=6,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Checkbutton(opt, text="Summary.Track is N,E (swap)", variable=self.var_track_is_ne).grid(
            row=0, column=2, sticky="w", padx=(14, 0)
        )

        ttk.Label(opt, text="Ensemble step:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_ens_step, width=8).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Cell step:").grid(row=1, column=2, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_cell_step, width=8).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Opacity (0-1):").grid(row=1, column=4, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_opacity, width=8).grid(row=1, column=5, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Kleurschaal:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            opt,
            textvariable=self.var_colorscale,
            values=["Turbo", "Viridis", "Plasma", "Cividis", "Jet"],
            width=10,
            state="readonly",
        ).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Checkbutton(opt, text="Toon stromingscontouren", variable=self.var_show_contours).grid(
            row=2, column=2, sticky="w", padx=(14, 0), pady=(8, 0)
        )
        ttk.Checkbutton(opt, text="Meetcellen als vakjes (geen interpolatie)", variable=self.var_use_cell_coloring).grid(
            row=2, column=3, sticky="w", padx=(14, 0), pady=(8, 0)
        )

        ttk.Label(opt, text="Contour niveaus:").grid(row=2, column=4, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_contour_levels, width=8).grid(row=2, column=5, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Max contourlijnen/file:").grid(row=2, column=6, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_contour_max_lines, width=8).grid(row=2, column=7, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Kleurschaal X:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_colorbar_x, width=8).grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(opt, text="Kleurschaal lengte:").grid(row=3, column=2, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_colorbar_len, width=8).grid(row=3, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(opt, text="Kleurschaal dikte:").grid(row=3, column=4, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_colorbar_thickness, width=8).grid(
            row=3, column=5, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        ttk.Checkbutton(opt, text="Maak OSM alternatiefkaart", variable=self.var_make_osm_alt).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Checkbutton(opt, text="Zet 3D-link op OSM kaart", variable=self.var_osm_add_3d_link).grid(
            row=4, column=2, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(opt, text="Track EPSG:").grid(row=4, column=4, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_track_epsg, width=8).grid(row=4, column=5, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(opt, text="OSM marker step:").grid(row=4, column=6, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_osm_marker_step, width=8).grid(row=4, column=7, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(opt, text="Metingstype:").grid(row=4, column=8, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Radiobutton(opt, text="Stationair", variable=self.var_measurement_mode, value="stationair").grid(
            row=4, column=9, sticky="w", padx=(6, 0), pady=(8, 0)
        )
        ttk.Radiobutton(opt, text="Dwarsprofiel", variable=self.var_measurement_mode, value="dwarsprofiel").grid(
            row=4, column=10, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        ttk.Checkbutton(opt, text="Maak KML export", variable=self.var_make_kml).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(opt, text="KML verticale exag:").grid(row=5, column=5, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_kml_vertical_exag, width=8).grid(row=5, column=6, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(opt, text="KML ens step:").grid(row=5, column=7, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_kml_ens_step, width=8).grid(row=5, column=8, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(opt, text="KML cell step:").grid(row=6, column=5, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_kml_cell_step, width=8).grid(row=6, column=6, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Checkbutton(opt, text="Maak DAE mesh export", variable=self.var_make_dae).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Checkbutton(opt, text="Maak KMZ model (aanbevolen voor GE)", variable=self.var_make_kmz_model).grid(
            row=6, column=2, columnspan=3, sticky="w", pady=(8, 0)
        )
        ttk.Label(opt, text="DAE ens step:").grid(row=6, column=7, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_dae_ens_step, width=8).grid(row=6, column=8, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(opt, text="DAE cell step:").grid(row=7, column=5, sticky="e", padx=(14, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_dae_cell_step, width=8).grid(row=7, column=6, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(opt, text="Stationair ref KMZ/KML punt:").grid(row=7, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_reference_kmz, width=58).grid(row=7, column=1, columnspan=4, sticky="we", padx=(6, 0), pady=(8, 0))
        ttk.Button(opt, text="Kies KMZ/KML...", command=self._pick_reference_kmz).grid(row=7, column=7, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Button(opt, text="Klik punt op kaart...", command=self._pick_reference_map).grid(row=7, column=8, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(opt, text="Linkeroever lon/lat:").grid(row=8, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_left_bank_lon, width=16).grid(row=8, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_left_bank_lat, width=16).grid(row=8, column=2, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Button(opt, text="Klik linkeroever...", command=self._pick_left_bank_map).grid(row=8, column=7, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(opt, text="Rechteroever lon/lat:").grid(row=9, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_right_bank_lon, width=16).grid(row=9, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Entry(opt, textvariable=self.var_right_bank_lat, width=16).grid(row=9, column=2, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Button(opt, text="Klik rechteroever...", command=self._pick_right_bank_map).grid(row=9, column=7, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Button(opt, text="Klik profiel (2 punten)...", command=self._pick_profile_line_map).grid(
            row=9, column=8, columnspan=2, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        out_box = ttk.LabelFrame(frm, text="Output HTML", padding=10)
        out_box.grid(row=2, column=0, sticky="we", pady=(10, 0))
        out_box.grid_columnconfigure(0, weight=1)

        ttk.Entry(out_box, textvariable=self.var_out, width=112).grid(row=0, column=0, sticky="we")
        ttk.Button(out_box, text="Opslaan als...", command=self._pick_out).grid(row=0, column=1, padx=(8, 0))
        ttk.Checkbutton(out_box, text="Open na export", variable=self.var_open_after).grid(row=1, column=0, sticky="w", pady=(8, 0))

        bot = ttk.Frame(frm)
        bot.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        bot.grid_columnconfigure(0, weight=1)
        bot.grid_rowconfigure(2, weight=1)

        btns = ttk.Frame(bot)
        btns.grid(row=0, column=0, sticky="we")
        ttk.Button(btns, text="Lees data in", command=self._load_data).pack(side="left")
        self.btn_process = ttk.Button(btns, text="Verwerk de data", command=self._process_data, state="disabled")
        self.btn_process.pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Sluiten", command=self.destroy).pack(side="right")

        ttk.Label(bot, text="Log").grid(row=1, column=0, sticky="w")
        self.txt = tk.Text(bot, height=14, wrap="word")
        self.txt.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

    def _log(self, s: str) -> None:
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _set_process_enabled(self, enabled: bool) -> None:
        if enabled:
            self.btn_process.state(["!disabled"])
        else:
            self.btn_process.state(["disabled"])

    def _close_preview_window(self) -> None:
        if self._preview_window is not None and self._preview_window.winfo_exists():
            self._preview_window.destroy()
        self._preview_window = None

    def _invalidate_loaded_data(self, close_preview: bool = True) -> None:
        self._loaded_preview = []
        self._loaded_signature = None
        self._set_process_enabled(False)
        if close_preview:
            self._close_preview_window()

    def _on_load_input_changed(self, *_args: object) -> None:
        self._invalidate_loaded_data()

    def _current_load_signature(self) -> tuple[object, ...]:
        return (
            tuple(str(p.resolve()).lower() for p in self._mat_files),
            str(self.var_m9_shift.get()).strip(),
            bool(self.var_track_is_ne.get()),
            int(self.var_ens_step.get()),
            int(self.var_cell_step.get()),
        )

    def _show_preview_window(self, previews: list[MatPreviewSummary]) -> None:
        self._close_preview_window()
        win = tk.Toplevel(self)
        win.title("Ingelezen data")
        win.geometry("1280x860")
        try:
            win.state("zoomed")
        except Exception:
            pass
        win.transient(self)
        self._preview_window = win

        def _on_close() -> None:
            self._preview_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        outer = ttk.Frame(win, padding=10)
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        notebook = ttk.Notebook(outer)
        notebook.grid(row=0, column=0, sticky="nsew")

        def _make_tab(title: str) -> ttk.Frame:
            frm = ttk.Frame(notebook, padding=8)
            frm.pack_propagate(False)
            notebook.add(frm, text=title)
            return frm

        track_box = _make_tab("Track")
        depth_box = _make_tab("Diepteprofiel")
        speed_box = _make_tab("Snelheden tijd")
        speed_dist_box = _make_tab("Snelheden afstand")
        table_box = _make_tab("Samenvatting")
        table_box.grid_columnconfigure(0, weight=1)
        table_box.grid_rowconfigure(0, weight=1)

        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#17becf", "#a65628", "#f781bf"]
        if _MPL_OK and _MPL_TK_OK and FigureCanvasTkAgg is not None and plt is not None:
            fig_track = plt.Figure(figsize=(8.5, 5.8), dpi=100, tight_layout=True)
            ax_track = fig_track.add_subplot(111)
            all_e: list[np.ndarray] = []
            all_n: list[np.ndarray] = []
            for i, item in enumerate(previews):
                ok = np.isfinite(item.track_e) & np.isfinite(item.track_n)
                if np.count_nonzero(ok) == 0:
                    continue
                e = item.track_e[ok]
                n = item.track_n[ok]
                col = colors[i % len(colors)]
                ax_track.plot(e, n, color=col, linewidth=1.8, label=item.path.stem)
                ax_track.scatter([e[0]], [n[0]], color=col, s=20)
                ax_track.scatter([e[-1]], [n[-1]], color=col, s=26, marker="x")
                all_e.append(e)
                all_n.append(n)
            ax_track.set_title("Ruwe track uit MAT")
            ax_track.set_xlabel("Track X")
            ax_track.set_ylabel("Track Y")
            ax_track.grid(True, alpha=0.25)
            if all_e and all_n:
                span_e = float(np.nanmax(np.concatenate(all_e)) - np.nanmin(np.concatenate(all_e)))
                span_n = float(np.nanmax(np.concatenate(all_n)) - np.nanmin(np.concatenate(all_n)))
                if span_e > 0.0 and span_n > 0.0:
                    ax_track.set_aspect("equal", adjustable="datalim")
            if len(previews) <= 8:
                ax_track.legend(loc="best", fontsize=8)
            canvas_track = FigureCanvasTkAgg(fig_track, master=track_box)
            canvas_track.draw()
            canvas_track.get_tk_widget().pack(fill="both", expand=True)

            fig_depth = plt.Figure(figsize=(8.5, 5.8), dpi=100, tight_layout=True)
            ax_depth = fig_depth.add_subplot(111)
            for i, item in enumerate(previews):
                n = min(len(item.distance_m), len(item.bed_depth))
                if n == 0:
                    continue
                x = np.asarray(item.distance_m[:n], dtype=float)
                y = np.asarray(item.bed_depth[:n], dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(ok) == 0:
                    continue
                ax_depth.plot(x[ok], y[ok], color=colors[i % len(colors)], linewidth=1.8, label=item.path.stem)
            ax_depth.set_title("Bodemdiepte langs de meting")
            ax_depth.set_xlabel("Cumulatieve afstand langs track (m)")
            ax_depth.set_ylabel("Diepte (m)")
            ax_depth.grid(True, alpha=0.25)
            ax_depth.invert_yaxis()
            if len(previews) <= 8:
                ax_depth.legend(loc="best", fontsize=8)
            canvas_depth = FigureCanvasTkAgg(fig_depth, master=depth_box)
            canvas_depth.draw()
            canvas_depth.get_tk_widget().pack(fill="both", expand=True)

            fig_speed = plt.Figure(figsize=(8.5, 5.8), dpi=100, tight_layout=True)
            ax_speed = fig_speed.add_subplot(111)
            for i, item in enumerate(previews):
                col = colors[i % len(colors)]
                n_surface = min(len(item.elapsed_min), len(item.surface_speed))
                if n_surface > 0:
                    x_surface = np.asarray(item.elapsed_min[:n_surface], dtype=float)
                    y_surface = np.asarray(item.surface_speed[:n_surface], dtype=float)
                    ok_surface = np.isfinite(x_surface) & np.isfinite(y_surface)
                    if np.count_nonzero(ok_surface) > 0:
                        ax_speed.plot(
                            x_surface[ok_surface],
                            y_surface[ok_surface],
                            color=col,
                            linewidth=1.8,
                            label=f"{item.path.stem} oppervlak",
                        )
                n_mean = min(len(item.elapsed_min), len(item.mean_speed_series))
                if n_mean > 0:
                    x_mean = np.asarray(item.elapsed_min[:n_mean], dtype=float)
                    y_mean = np.asarray(item.mean_speed_series[:n_mean], dtype=float)
                    ok_mean = np.isfinite(x_mean) & np.isfinite(y_mean)
                    if np.count_nonzero(ok_mean) > 0:
                        ax_speed.plot(
                            x_mean[ok_mean],
                            y_mean[ok_mean],
                            color=col,
                            linewidth=1.4,
                            linestyle="--",
                            label=f"{item.path.stem} kolomgem",
                        )
            ax_speed.set_title("Snelheid in de tijd")
            ax_speed.set_xlabel("Tijd sinds start (min)")
            ax_speed.set_ylabel("Snelheid (m/s)")
            ax_speed.grid(True, alpha=0.25)
            if len(previews) <= 4:
                ax_speed.legend(loc="best", fontsize=8)
            canvas_speed = FigureCanvasTkAgg(fig_speed, master=speed_box)
            canvas_speed.draw()
            canvas_speed.get_tk_widget().pack(fill="both", expand=True)

            fig_speed_dist = plt.Figure(figsize=(8.5, 5.8), dpi=100, tight_layout=True)
            ax_speed_dist = fig_speed_dist.add_subplot(111)
            for i, item in enumerate(previews):
                col = colors[i % len(colors)]
                n_surface = min(len(item.distance_m), len(item.surface_speed))
                if n_surface > 0:
                    x_surface = np.asarray(item.distance_m[:n_surface], dtype=float)
                    y_surface = np.asarray(item.surface_speed[:n_surface], dtype=float)
                    ok_surface = np.isfinite(x_surface) & np.isfinite(y_surface)
                    if np.count_nonzero(ok_surface) > 0:
                        ax_speed_dist.plot(
                            x_surface[ok_surface],
                            y_surface[ok_surface],
                            color=col,
                            linewidth=1.8,
                            label=f"{item.path.stem} oppervlak",
                        )
                n_mean = min(len(item.distance_m), len(item.mean_speed_series))
                if n_mean > 0:
                    x_mean = np.asarray(item.distance_m[:n_mean], dtype=float)
                    y_mean = np.asarray(item.mean_speed_series[:n_mean], dtype=float)
                    ok_mean = np.isfinite(x_mean) & np.isfinite(y_mean)
                    if np.count_nonzero(ok_mean) > 0:
                        ax_speed_dist.plot(
                            x_mean[ok_mean],
                            y_mean[ok_mean],
                            color=col,
                            linewidth=1.4,
                            linestyle="--",
                            label=f"{item.path.stem} kolomgem",
                        )
            ax_speed_dist.set_title("Snelheid tov afstandsas")
            ax_speed_dist.set_xlabel("Cumulatieve afstand langs track (m)")
            ax_speed_dist.set_ylabel("Snelheid (m/s)")
            ax_speed_dist.grid(True, alpha=0.25)
            if len(previews) <= 4:
                ax_speed_dist.legend(loc="best", fontsize=8)
            canvas_speed_dist = FigureCanvasTkAgg(fig_speed_dist, master=speed_dist_box)
            canvas_speed_dist.draw()
            canvas_speed_dist.get_tk_widget().pack(fill="both", expand=True)
        else:
            ttk.Label(
                track_box,
                text="Matplotlib Tk-backend niet beschikbaar; track preview kan hier niet getoond worden.",
                justify="left",
                wraplength=420,
            ).pack(fill="both", expand=True)
            ttk.Label(
                depth_box,
                text="Matplotlib Tk-backend niet beschikbaar; diepteprofiel kan hier niet getoond worden.",
                justify="left",
                wraplength=420,
            ).pack(fill="both", expand=True)
            ttk.Label(
                speed_box,
                text="Matplotlib Tk-backend niet beschikbaar; stroomsnelheden kunnen hier niet getoond worden.",
                justify="left",
                wraplength=420,
            ).pack(fill="both", expand=True)
            ttk.Label(
                speed_dist_box,
                text="Matplotlib Tk-backend niet beschikbaar; snelheden tov afstandas kunnen hier niet getoond worden.",
                justify="left",
                wraplength=420,
            ).pack(fill="both", expand=True)

        cols = ("bestand", "start", "stop", "vmin", "vgem", "vmax", "debiet")
        tree = ttk.Treeview(table_box, columns=cols, show="headings", height=max(6, min(18, len(previews) + 1)))
        headers = {
            "bestand": "Bestand",
            "start": "Start",
            "stop": "Stop",
            "vmin": "V min",
            "vgem": "V gem",
            "vmax": "V max",
            "debiet": "Debiet",
        }
        widths = {"bestand": 180, "start": 145, "stop": 145, "vmin": 70, "vgem": 70, "vmax": 70, "debiet": 85}
        anchors = {"bestand": "w", "start": "center", "stop": "center", "vmin": "e", "vgem": "e", "vmax": "e", "debiet": "e"}
        for col in cols:
            tree.heading(col, text=headers[col])
            tree.column(col, width=widths[col], anchor=anchors[col], stretch=(col in {"bestand", "start", "stop"}))
        for item in previews:
            tree.insert(
                "",
                "end",
                values=(
                    item.path.name,
                    _format_preview_time(item.start_time),
                    _format_preview_time(item.stop_time),
                    _format_preview_value(item.speed_min),
                    _format_preview_value(item.speed_mean),
                    _format_preview_value(item.speed_max),
                    _format_preview_value(item.discharge_mean),
                ),
            )
        tree.grid(row=0, column=0, sticky="nsew")
        ysb = ttk.Scrollbar(table_box, orient="vertical", command=tree.yview)
        ysb.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=ysb.set)

        ttk.Label(
            outer,
            text=(
                "Debiet in de preview = gemiddelde van Summary.Total_Q uit de MAT. "
                "In 'Stroomsnelheden' is volle lijn = oppervlak en stippellijn = kolomgemiddelde. "
                "De afstandsas is de cumulatieve afstand langs de ruwe track. "
                "Export gebeurt pas via 'Verwerk de data'."
            ),
            justify="left",
        ).grid(row=1, column=0, sticky="we", pady=(8, 0))

    def _load_data(self) -> None:
        try:
            if not self._mat_files:
                messagebox.showerror("Input", "Voeg eerst een of meer MAT-bestanden toe.")
                return

            m9_shift = int(self.var_m9_shift.get())
            track_is_ne = bool(self.var_track_is_ne.get())
            ens_step = max(1, int(self.var_ens_step.get()))
            cell_step = max(1, int(self.var_cell_step.get()))

            previews: list[MatPreviewSummary] = []
            failed: list[str] = []
            self._log("MAT-bestanden inlezen voor preview...")
            for p in self._mat_files:
                try:
                    item = build_mat_preview_summary(
                        mat_path=p,
                        m9_hour_shift=m9_shift,
                        track_is_ne=track_is_ne,
                        ensemble_step=ens_step,
                        cell_step=cell_step,
                    )
                    previews.append(item)
                    self._log(
                        f"  OK preview: {p.name} | start={_format_preview_time(item.start_time)} | "
                        f"stop={_format_preview_time(item.stop_time)} | "
                        f"v(min/gem/max)={_format_preview_value(item.speed_min)}/"
                        f"{_format_preview_value(item.speed_mean)}/"
                        f"{_format_preview_value(item.speed_max)} m/s | "
                        f"debiet={_format_preview_value(item.discharge_mean)} m3/s"
                    )
                    if item.note:
                        self._log(f"      note: {item.note}")
                except Exception as exc:
                    failed.append(f"{p.name}: {exc}")
                    self._log(f"  ERROR preview: {p.name} -> {exc}")

            if not previews:
                raise ValueError("Geen geldig MAT-bestand kunnen inlezen.")

            self._loaded_preview = previews
            self._loaded_signature = self._current_load_signature()
            self._set_process_enabled(True)
            self._show_preview_window(previews)

            if failed:
                messagebox.showwarning(
                    "Preview met waarschuwingen",
                    f"{len(previews)} bestand(en) ingelezen.\n{len(failed)} bestand(en) faalden.\nZie log voor details.",
                )
            else:
                messagebox.showinfo("Data ingelezen", f"{len(previews)} bestand(en) ingelezen.")
        except Exception as exc:
            messagebox.showerror("Fout bij inlezen", str(exc))
            self._log(f"ERROR inlezen: {exc}")

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
        self._invalidate_loaded_data()

    def _remove_selected(self) -> None:
        sel = list(self.listbox.curselection())
        if not sel:
            return
        idxs = set(sel)
        self._mat_files = [p for i, p in enumerate(self._mat_files) if i not in idxs]
        self._refresh_listbox()
        self._invalidate_loaded_data()

    def _clear_files(self) -> None:
        self._mat_files = []
        self._refresh_listbox()
        self._invalidate_loaded_data()

    def _pick_out(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Kies output HTML",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("All", "*.*")],
        )
        if p:
            self.var_out.set(p)

    def _pick_reference_kmz(self) -> None:
        p = filedialog.askopenfilename(
            title="Kies absolute puntreferentie KMZ/KML",
            filetypes=[("KMZ/KML", "*.kmz *.kml"), ("KMZ", "*.kmz"), ("KML", "*.kml"), ("All", "*.*")],
        )
        if p:
            self.var_reference_kmz.set(p)

    def _store_generated_reference(self, selection: dict[str, object], target_var: tk.StringVar, label: str) -> None:
        mode = str(selection["mode"])
        coords = [(float(lon), float(lat)) for lon, lat in selection["coords"]]
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outp = Path(tempfile.gettempdir()) / f"adcp_reference_{label}_{mode}_{stamp}.kml"
        name = "ADCP referentiepunt" if mode == "point" else "ADCP raai"
        _write_reference_kml(outp, coords, mode, name)
        self._generated_reference_path = outp
        target_var.set(str(outp))
        self._log(f"Kaartselectie opgeslagen als {label}: {outp}")

    def _pick_reference_map(self) -> None:
        try:
            center = _default_reference_center(self.var_reference_kmz.get().strip() or None)
            sel = _launch_reference_picker(initial_center=center, initial_mode="point")
            if not sel:
                self._log("Kaartselectie geannuleerd of verlopen.")
                return
            if str(sel.get("mode", "")) != "point":
                messagebox.showerror("Kaartselectie", "Alleen een puntreferentie wordt nog ondersteund.")
                return
            self._store_generated_reference(sel, self.var_reference_kmz, "absolute_referentie")
        except Exception as exc:
            messagebox.showerror("Kaartselectie", f"Kaartselectie mislukt:\n{exc}")

    def _pick_bank_point(self, side: str) -> None:
        try:
            center = _default_reference_center(self.var_reference_kmz.get().strip() or None)
            sel = _launch_reference_picker(initial_center=center, initial_mode="point")
            if not sel:
                self._log("Kaartselectie geannuleerd of verlopen.")
                return
            lon, lat = sel["coords"][0]
            if side == "left":
                self.var_left_bank_lon.set(f"{float(lon):.7f}")
                self.var_left_bank_lat.set(f"{float(lat):.7f}")
                self._log(f"Linkeroever gekozen: lon={float(lon):.7f}, lat={float(lat):.7f}")
            else:
                self.var_right_bank_lon.set(f"{float(lon):.7f}")
                self.var_right_bank_lat.set(f"{float(lat):.7f}")
                self._log(f"Rechteroever gekozen: lon={float(lon):.7f}, lat={float(lat):.7f}")
        except Exception as exc:
            messagebox.showerror("Kaartselectie", f"Kaartselectie mislukt:\n{exc}")

    def _pick_left_bank_map(self) -> None:
        self._pick_bank_point("left")

    def _pick_right_bank_map(self) -> None:
        self._pick_bank_point("right")

    def _pick_profile_line_map(self) -> None:
        try:
            center = _default_reference_center(self.var_reference_kmz.get().strip() or None)
            sel = _launch_reference_picker(initial_center=center, initial_mode="line")
            if not sel:
                self._log("Kaartselectie geannuleerd of verlopen.")
                return
            if str(sel.get("mode", "")) != "line":
                messagebox.showerror("Kaartselectie", "Profiel vereist exact 2 klikpunten.")
                return
            (lon1, lat1), (lon2, lat2) = sel["coords"]
            self.var_left_bank_lon.set(f"{float(lon1):.7f}")
            self.var_left_bank_lat.set(f"{float(lat1):.7f}")
            self.var_right_bank_lon.set(f"{float(lon2):.7f}")
            self.var_right_bank_lat.set(f"{float(lat2):.7f}")
            self._log(
                f"Profiellijn gekozen: linkeroever lon={float(lon1):.7f}, lat={float(lat1):.7f} | "
                f"rechteroever lon={float(lon2):.7f}, lat={float(lat2):.7f}"
            )
        except Exception as exc:
            messagebox.showerror("Kaartselectie", f"Kaartselectie mislukt:\n{exc}")

    def _process_data(self) -> None:
        try:
            if not _PLOTLY_OK:
                messagebox.showerror("Dependency", "plotly niet beschikbaar. Installeer: pip install plotly")
                return
            if not self._mat_files:
                messagebox.showerror("Input", "Voeg eerst een of meer MAT-bestanden toe.")
                return

            out_html = self.var_out.get().strip()
            if not out_html:
                messagebox.showerror("Output", "Geef een output HTML-bestand op.")
                return

            m9_shift = int(self.var_m9_shift.get())
            track_is_ne = bool(self.var_track_is_ne.get())
            ens_step = max(1, int(self.var_ens_step.get()))
            cell_step = max(1, int(self.var_cell_step.get()))
            current_signature = self._current_load_signature()
            if not self._loaded_preview or self._loaded_signature != current_signature:
                raise ValueError(
                    "Klik eerst op 'Lees data in' met de huidige bestanden en inleesopties voordat je gaat verwerken."
                )
            opacity = float(self.var_opacity.get())
            opacity = min(1.0, max(0.05, opacity))
            use_cell_coloring = bool(self.var_use_cell_coloring.get())
            show_contours = bool(self.var_show_contours.get())
            contour_levels = max(2, int(self.var_contour_levels.get()))
            contour_max_lines = max(10, int(self.var_contour_max_lines.get()))
            colorscale = self.var_colorscale.get().strip() or "Turbo"
            colorbar_x = float(self.var_colorbar_x.get())
            colorbar_x = min(1.35, max(0.80, colorbar_x))
            colorbar_len = float(self.var_colorbar_len.get())
            colorbar_len = min(0.95, max(0.25, colorbar_len))
            colorbar_thickness = max(8, int(self.var_colorbar_thickness.get()))
            make_osm_alt = bool(self.var_make_osm_alt.get())
            osm_add_3d_link = bool(self.var_osm_add_3d_link.get())
            track_epsg = int(self.var_track_epsg.get())
            measurement_mode = self.var_measurement_mode.get().strip() or "stationair"
            reference_kmz = self.var_reference_kmz.get().strip() or None
            left_bank_lon = float(self.var_left_bank_lon.get()) if self.var_left_bank_lon.get().strip() else None
            left_bank_lat = float(self.var_left_bank_lat.get()) if self.var_left_bank_lat.get().strip() else None
            right_bank_lon = float(self.var_right_bank_lon.get()) if self.var_right_bank_lon.get().strip() else None
            right_bank_lat = float(self.var_right_bank_lat.get()) if self.var_right_bank_lat.get().strip() else None
            osm_marker_step = max(1, int(self.var_osm_marker_step.get()))
            make_kml = bool(self.var_make_kml.get())
            kml_vertical_exag = max(0.1, float(self.var_kml_vertical_exag.get()))
            kml_ens_step = max(1, int(self.var_kml_ens_step.get()))
            kml_cell_step = max(1, int(self.var_kml_cell_step.get()))
            make_dae = bool(self.var_make_dae.get())
            make_kmz_model = bool(self.var_make_kmz_model.get())
            dae_ens_step = max(1, int(self.var_dae_ens_step.get()))
            dae_cell_step = max(1, int(self.var_dae_cell_step.get()))

            if measurement_mode == "stationair" and not reference_kmz:
                raise ValueError("Stationaire meting vereist een vast punt via KMZ/KML of kaartklik.")
            if measurement_mode == "dwarsprofiel":
                missing = [
                    name
                    for name, value in (
                        ("left_bank_lon", left_bank_lon),
                        ("left_bank_lat", left_bank_lat),
                        ("right_bank_lon", right_bank_lon),
                        ("right_bank_lat", right_bank_lat),
                    )
                    if value is None
                ]
                if missing:
                    raise ValueError(f"Dwarsprofiel vereist beide oevers in WGS84. Ontbreekt: {', '.join(missing)}.")

            datasets: list[Mat3DData] = []
            failed: list[str] = []

            self._log("MAT-bestanden verwerken...")
            self._log(f"Metingstype: {measurement_mode}")
            if measurement_mode == "stationair" and reference_kmz:
                self._log(f"Absolute referentie KMZ/KML: {reference_kmz}")
            if measurement_mode == "dwarsprofiel":
                self._log(
                    f"Linkeroever WGS84: lon={left_bank_lon if left_bank_lon is not None else 'n/a'}, "
                    f"lat={left_bank_lat if left_bank_lat is not None else 'n/a'}"
                )
                self._log(
                    f"Rechteroever WGS84: lon={right_bank_lon if right_bank_lon is not None else 'n/a'}, "
                    f"lat={right_bank_lat if right_bank_lat is not None else 'n/a'}"
                )
            for p in self._mat_files:
                try:
                    d = read_mat_3d(
                        p,
                        m9_hour_shift=m9_shift,
                        track_is_ne=track_is_ne,
                        ensemble_step=ens_step,
                        cell_step=cell_step,
                        measurement_mode=measurement_mode,
                        reference_path=reference_kmz,
                        left_bank_lon=left_bank_lon,
                        left_bank_lat=left_bank_lat,
                        right_bank_lon=right_bank_lon,
                        right_bank_lat=right_bank_lat,
                        track_epsg=track_epsg,
                    )
                    datasets.append(d)
                    self._log(
                        f"  OK: {p.name} | ensembles={d.x.shape[1]} | cells={d.x.shape[0]} | "
                        f"speed pts={np.count_nonzero(np.isfinite(d.speed))} | verwerking={d.processing_mode}"
                    )
                    if d.note:
                        self._log(f"      note: {d.note}")
                    for line in d.report_lines:
                        self._log(f"      {line}")
                except Exception as e:
                    failed.append(f"{p.name}: {e}")
                    self._log(f"  ERROR: {p.name} -> {e}")

            if not datasets:
                raise ValueError("Geen geldig MAT-bestand kunnen verwerken.")

            georef_ok = all(_looks_georeferenced(d.track_e, d.track_n) for d in datasets)
            if not georef_ok:
                self._log(
                    "WAARSCHUWING: minstens een dataset heeft een relatieve/niet-gegeorefereerde track. "
                    "OSM/KML/KMZ-locatie wordt overgeslagen; dit is geen simpel E/N-wisselprobleem."
                )

            self._log("3D figuur bouwen...")
            fig = build_3d_figure(
                datasets=datasets,
                opacity=opacity,
                use_cell_coloring=use_cell_coloring,
                show_contours=show_contours,
                contour_levels=contour_levels,
                contour_max_lines=contour_max_lines,
                color_scale=colorscale,
                colorbar_x=colorbar_x,
                colorbar_len=colorbar_len,
                colorbar_thickness=colorbar_thickness,
            )
            outp = Path(out_html)
            outp.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(
                str(outp),
                include_plotlyjs="cdn",
                full_html=True,
                config={"responsive": True},
                post_script=_plotly_colorbar_controls_post_script(colorscale),
            )
            add_interactive_html_saver(outp)
            self._log(f"Klaar: {outp}")

            osm_path: Path | None = None
            if make_osm_alt:
                if not georef_ok:
                    self._log("OSM alternatiefkaart overgeslagen: track is relatief of niet gegeorefereerd.")
                elif _MAP_OK:
                    osm_path = outp.with_name(outp.stem + "_openstreetmap.html")
                    self._log("OSM alternatiefkaart bouwen...")
                    build_osm_alternative_map(
                        datasets=datasets,
                        out_html=osm_path,
                        track_epsg=track_epsg,
                        marker_step=osm_marker_step,
                        three_d_html_name=(outp.name if osm_add_3d_link else None),
                    )
                    self._log(f"OSM kaart: {osm_path}")
                else:
                    self._log("WAARSCHUWING: folium/pyproj niet beschikbaar, OSM alternatief overgeslagen.")

            kml_path: Path | None = None
            if make_kml:
                if not georef_ok:
                    self._log("KML export overgeslagen: track is relatief of niet gegeorefereerd.")
                elif _PROJ_OK:
                    kml_path = outp.with_suffix(".kml")
                    self._log("KML export bouwen...")
                    build_kml_export(
                        datasets=datasets,
                        out_kml=kml_path,
                        track_epsg=track_epsg,
                        vertical_exaggeration=kml_vertical_exag,
                        ensemble_step=kml_ens_step,
                        cell_step=kml_cell_step,
                    )
                    self._log(f"KML: {kml_path}")
                else:
                    self._log("WAARSCHUWING: pyproj niet beschikbaar, KML export overgeslagen.")

            dae_path: Path | None = None
            kmz_model_path: Path | None = None
            if make_dae:
                dae_path = outp.with_suffix(".dae")
                self._log("DAE export bouwen...")
                origin_e, origin_n, model_altitude_m = build_dae_export(
                    datasets=datasets,
                    out_dae=dae_path,
                    ensemble_step=dae_ens_step,
                    cell_step=dae_cell_step,
                )
                self._log(f"DAE: {dae_path}")

                if make_kmz_model:
                    if not georef_ok:
                        self._log("KMZ model overgeslagen: track is relatief of niet gegeorefereerd.")
                    elif _PROJ_OK:
                        kmz_model_path = outp.with_name(outp.stem + "_model.kmz")
                        self._log("KMZ model bouwen...")
                        build_kmz_model(
                            dae_path=dae_path,
                            out_kmz=kmz_model_path,
                            origin_e=origin_e,
                            origin_n=origin_n,
                            track_epsg=track_epsg,
                            model_altitude_m=model_altitude_m,
                        )
                        self._log(f"KMZ model: {kmz_model_path}")
                    else:
                        self._log("WAARSCHUWING: pyproj niet beschikbaar, KMZ model overgeslagen.")

            if bool(self.var_open_after.get()):
                webbrowser.open(outp.resolve().as_uri())
                if osm_path is not None:
                    webbrowser.open(osm_path.resolve().as_uri())
                if kml_path is not None:
                    webbrowser.open(kml_path.resolve().as_uri())
                if dae_path is not None:
                    webbrowser.open(dae_path.resolve().as_uri())
                if kmz_model_path is not None:
                    webbrowser.open(kmz_model_path.resolve().as_uri())

            if failed:
                messagebox.showwarning(
                    "Klaar met waarschuwingen",
                    f"3D HTML gemaakt met {len(datasets)} bestand(en).\n"
                    f"{len(failed)} bestand(en) faalden.\nZie log voor details.",
                )
            else:
                messagebox.showinfo("Klaar", f"3D HTML gemaakt met {len(datasets)} bestand(en).")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
