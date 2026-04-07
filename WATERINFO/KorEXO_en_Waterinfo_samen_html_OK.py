#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KorEXO_en_Waterinfo_samen_html_v4.py

Wijzigingen t.o.v. v3:
- GEEN legenda-weergave (showlegend=False); mouse-over blijft volledige info geven.
- UI in de HTML om een tijdshift (± minuten) toe te passen op ENKEL de KorEXO-reeksen.
  (Client-side: verschuift de x-as waarden voor traces met meta=="KorEXO", reset-knop aanwezig.)

Overig: zelfde KorEXO- en Waterinfo-parsers, multi-bestand selectie, interactieve HTML met checkboxen,
Y-as bereik-bediening en X-range tools.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import html
import re
from typing import List, Dict, Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# GUI
import tkinter as tk
from tkinter import filedialog, messagebox


# ============== KOR-EXO PARSER (uit werkende logica) ==============
MEAN_TOKEN = "MEAN VALUE"
STD_TOKEN  = "STANDARD DEVIATION"
SER_TOKEN  = "SENSOR SERIAL NUMBER"
NON_VALUE_COL_PREFIXES = ("Date", "Time", "Site Name")

def _try_open_text(path: Path):
    """
    Probeer veelvoorkomende encodings, incl. UTF-16(LE).
    """
    for enc in ("utf-16", "utf-16le", "utf-8-sig", "utf-8"):
        try:
            return path.open("r", encoding=enc, errors="ignore")
        except Exception:
            continue
    return path.open("r")

def _is_mean_or_std(cells: List[str]) -> bool:
    joined = ",".join(cells).upper()
    return (MEAN_TOKEN in joined) or (STD_TOKEN in joined)

def _is_serial_header(cells: List[str]) -> bool:
    joined = ",".join(cells).upper()
    return SER_TOKEN in joined

def _row_is_data(cells: List[str]) -> bool:
    if not cells:
        return False
    c0 = cells[0].strip()
    # dd/mm/yyyy (verdere velden volgen)
    return bool(re.match(r"^\d{1,2}/\d{1,2}/\d{4}(\s*|[,;\t].*)$", c0))

def _split_param_unit(label: str) -> tuple[str, str]:
    """
    Haal "parameter" en "unit" uit labels zoals "Turbidity (FNU) [12345]".
    """
    clean = re.sub(r"\s*\[[^\]]+\]\s*$", "", label).strip()
    m = re.search(r"\(([^)]+)\)\s*$", clean)
    if m:
        unit = m.group(1).strip()
        param = clean[:m.start()].strip()
    else:
        unit = ""
        param = clean
    return param, unit

def parse_korexo_csv(path: str | Path) -> pd.DataFrame:
    """
    Parseert KorEXO export met meerdere blokken (Measurement Data File Export).
    Retourneert DataFrame met: Timestamp, Value, Series, parameter, unit, source="KorEXO".
    """
    p = Path(path)
    rows_long: List[Dict[str, Any]] = []

    with _try_open_text(p) as f:
        # delimiter sniff (soms ';' i.p.v. ',')
        sample = f.read(4096)
        f.seek(0)
        delim = ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t"])
            delim = dialect.delimiter
        except Exception:
            if sample.count(";") > sample.count(","):
                delim = ";"
            elif "\t" in sample and sample.count("\t") > 2:
                delim = "\t"

        reader = csv.reader(f, delimiter=delim)
        block_idx = -1
        current_serials: List[str] | None = None
        current_header:  List[str] | None = None
        site_name_idx: int | None = None

        for raw_cells in reader:
            cells = [c.strip() for c in raw_cells]
            if not any(cells):
                continue
            if _is_mean_or_std(cells):
                continue

            if _is_serial_header(cells):
                block_idx += 1
                current_serials = cells
                current_header  = None
                site_name_idx   = None
                continue

            if current_serials is not None and current_header is None and not _is_mean_or_std(cells):
                current_header = cells
                for i, h in enumerate(current_header):
                    if h.startswith("Site Name"):
                        site_name_idx = i
                        break
                continue

            if current_header is not None and _row_is_data(cells):
                # samengestelde kolomnamen (header + [serial])
                headers = []
                n = max(len(current_header), len(current_serials or []))
                for i in range(n):
                    h = current_header[i] if i < len(current_header) else f"col{i}"
                    s = (current_serials[i] if (current_serials and i < len(current_serials)) else "").strip()
                    if s and not h.startswith(NON_VALUE_COL_PREFIXES):
                        headers.append(f"{h} [{s}]")
                    else:
                        headers.append(h)

                if len(cells) < len(headers):
                    cells = cells + [""] * (len(headers) - len(cells))
                elif len(cells) > len(headers):
                    cells = cells[:len(headers)]

                date_str = cells[0]
                time_str = cells[1] if len(cells) > 1 else "00:00:00"
                ts = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")

                site = ""
                if site_name_idx is not None and site_name_idx < len(cells):
                    site = cells[site_name_idx]

                for i, lab in enumerate(headers):
                    if lab.startswith(NON_VALUE_COL_PREFIXES):
                        continue
                    val_str = cells[i].replace(",", ".")
                    try:
                        val = float(val_str)
                    except Exception:
                        continue

                    param, unit = _split_param_unit(lab)

                    rows_long.append({
                        "Timestamp": ts,
                        "Series": lab,          # originele kolomtitel + [serial]
                        "Value": val,
                        "Block": block_idx,
                        "Site": site,
                        "parameter": param,
                        "unit": unit,
                        "source": "KorEXO"
                    })

    df = pd.DataFrame.from_records(rows_long)
    if df.empty:
        return df
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values(["Series", "Timestamp"]).reset_index(drop=True)
    return df


# ============== WATERINFO PARSERS ==============
def parse_multistation_csv(path: str | Path) -> pd.DataFrame:
    """
    Multi-station CSV van Waterinfo (met blokken #station_name; ... #rows;N + #Timestamp;Value;...).
    Retourneert: Timestamp, Value, station_name, station_no, parameter, unit
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\r\n") for ln in f]

    recs: list[dict] = []
    cur = {"station_name": None, "station_no": None,
           "stationparameter_name": None, "parametertype_name": None,
           "ts_unitname": None, "rows": None}
    in_data = False
    rows_needed: int | None = None
    rows_got = 0

    for ln in lines:
        if not ln:
            continue
        if ln.startswith("#"):
            if ln.startswith("#Timestamp;"):
                in_data = True
                rows_got = 0
                continue
            if ";" in ln:
                key, val = ln[1:].split(";", 1)
                key = key.strip(); val = val.strip()
                cur[key] = val
                if key == "rows":
                    try: rows_needed = int(val)
                    except Exception: rows_needed = None
            continue

        if in_data:
            parts = ln.split(";")
            if len(parts) >= 2:
                t_iso = parts[0].strip()
                v_raw = parts[1].strip().replace(",", ".")
                try: value = float(v_raw) if v_raw != "" else float("nan")
                except Exception: value = float("nan")
                ts = pd.to_datetime(t_iso, errors="coerce")
                recs.append({
                    "Timestamp": ts,
                    "Value": value,
                    "station_name": cur.get("station_name"),
                    "station_no": cur.get("station_no"),
                    "parameter": cur.get("stationparameter_name") or cur.get("parametertype_name"),
                    "unit": cur.get("ts_unitname"),
                })
                rows_got += 1
                if rows_needed is not None and rows_got >= rows_needed:
                    in_data = False

    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return df
    df = df.dropna(subset=["Timestamp"])
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"]).sort_values(["station_name","Timestamp"]).reset_index(drop=True)
    return df

def parse_simple_waterinfo_csv(path: str | Path) -> pd.DataFrame:
    """
    Fallback voor eenvoudige CSV's:
    - kopregels kunnen #... bevatten
    - kolommen: Timestamp;Value (of met komma als scheiding)
    - decimale komma wordt ondersteund
    Station/parameter/unit worden (indien niet in headers) uit bestandsnaam afgeleid.
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    header_info = {}
    data_rows = []
    sep = ";" if txt.count(";") >= txt.count(",") else ","
    for ln in lines:
        if ln.startswith("#"):
            if sep in ln[1:]:
                key, val = ln[1:].split(sep, 1)
                header_info[key.strip()] = val.strip()
            continue
        if "Timestamp" in ln and sep in ln:
            # skip kolomkop
            continue
        parts = ln.split(sep)
        if len(parts) >= 2:
            t_iso = parts[0].strip()
            v_raw = parts[1].strip().replace(",", ".")
            try: value = float(v_raw) if v_raw != "" else float("nan")
            except Exception: value = float("nan")
            ts = pd.to_datetime(t_iso, errors="coerce")
            data_rows.append((ts, value))

    if not data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(data_rows, columns=["Timestamp","Value"])
    df = df.dropna(subset=["Timestamp"])
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    station  = header_info.get("station_name") or p.stem
    param    = header_info.get("stationparameter_name") or header_info.get("parametertype_name") or p.stem
    unit     = header_info.get("ts_unitname") or ""
    df["station_name"] = station
    df["parameter"] = param
    df["unit"] = unit
    return df.sort_values("Timestamp").reset_index(drop=True)


# ============== PLOT / HTML ==============
def build_plot(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        raise ValueError("Geen data om te plotten.")

    # Y-as label generiek
    unit_str = ""
    if "unit" in df and df["unit"].notna().any():
        units = df["unit"].dropna().astype(str).unique().tolist()
        unit_str = units[0] if len(units) == 1 else ""
    y_label = f"Waarde ({unit_str})" if unit_str else "Waarde"
    title   = "KorEXO + Waterinfo – gecombineerde grafiek"

    fig = go.Figure()
    added = 0

    for series, g in df.groupby("Series", dropna=False):
        g = g.sort_values("Timestamp")
        if g.empty:
            continue

        src  = str(g["source"].dropna().iloc[0]) if "source" in g and g["source"].notna().any() else ""
        par  = str(g["parameter"].dropna().iloc[0]) if "parameter" in g and g["parameter"].notna().any() else ""
        unit = str(g["unit"].dropna().iloc[0]) if "unit" in g and g["unit"].notna().any() else ""

        hover_parts = []
        if src:   hover_parts.append(f"<b>Bron:</b> {html.escape(src)}")
        hover_parts.append(f"<b>Reeks:</b> {html.escape(str(series))}")
        hover_parts.append("<b>Tijd:</b> %{x}")
        hover_parts.append("<b>Waarde:</b> %{y}")
        if par:
            if unit: hover_parts.append(f"<b>Parameter:</b> {html.escape(par)} ({html.escape(unit)})")
            else:    hover_parts.append(f"<b>Parameter:</b> {html.escape(par)}")
        hover = "<br>".join(hover_parts) + "<br><extra></extra>"

        fig.add_trace(go.Scattergl(
            x=g["Timestamp"], y=g["Value"],
            mode="markers" if src == "KorEXO" else "lines",
            line=dict(width=1.8),
            marker=dict(size=5, symbol="circle") if src == "KorEXO" else None,
            name=str(series),
            hovertemplate=hover,
            meta=src  # <-- bron labelen zodat JS enkel KorEXO kan shiften
        ))
        added += 1

    if added == 0:
        raise ValueError("Er zijn geen reeksen toegevoegd.")

    fig.update_layout(
        title=title,
        xaxis_title="Tijd",
        yaxis_title=y_label,
        hovermode="closest",
        dragmode="zoom",
        margin=dict(l=60, r=20, t=60, b=60),
        showlegend=False  # <-- legenda verbergen
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=1,  label="1u",  step="hour", stepmode="backward"),
            dict(count=6,  label="6u",  step="hour", stepmode="backward"),
            dict(count=12, label="12u", step="hour", stepmode="backward"),
            dict(count=1,  label="1d",  step="day",  stepmode="backward"),
            dict(count=3,  label="3d",  step="day",  stepmode="backward"),
            dict(step="all", label="Alles")
        ]),
        showspikes=True, spikemode="across"
    )
    fig.update_yaxes(showspikes=True, spikemode="across")
    fig.update_traces(hoverlabel=dict(bgcolor="white", bordercolor="black", font_size=12, font_family="Arial"))

    # “Alle / Geen / Alleen deze” dropdown
    n = len(list(fig.data))
    if n > 0:
        buttons = [
            dict(label="Alle", method="update", args=[{"visible": [True]*n}]),
            dict(label="Geen", method="update", args=[{"visible": [False]*n}]),
        ]
        for i, tr in enumerate(fig.data):
            vis = [False]*n; vis[i] = True
            buttons.append(dict(label=f"Alleen: {tr.name[:32]}", method="update", args=[{"visible": vis}]))
        fig.update_layout(updatemenus=[dict(type="dropdown", direction="down",
                                            x=1.0, xanchor="right", y=1.12, yanchor="top",
                                            buttons=buttons, showactive=False)])

    return fig

TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  body {{ margin:0; font-family: Arial, Helvetica, sans-serif; }}
  #container {{ display:flex; flex-direction: row; height: 100vh; }}
  #controls {{ width: 360px; min-width: 240px; max-width: 45vw; overflow:auto; border-right:1px solid #ddd; padding:12px; box-sizing: border-box; }}
  #plotwrap {{ flex:1; min-width:0; }}
  #plot {{ width:100%; height:100%; }}
  .section {{ margin-bottom: 14px; }}
  .row {{ display:flex; align-items:center; gap:8px; margin:6px 0; flex-wrap:wrap; }}
  .row input[type="number"] {{ width:110px; padding:4px; }}
  .btn {{ padding:6px 10px; border:1px solid #777; background:#f7f7f7; cursor:pointer; border-radius:4px; }}
  .btn:hover {{ background:#eee; }}
  .station-item {{ display:flex; align-items:center; margin:4px 0; }}
  .station-item input {{ margin-right:8px; }}
</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<div id="container">
  <div id="controls">
    <div class="section">
      <h3 style="margin:0;">Reeksen</h3>
      <div class="row">
        <button class="btn" id="btnAll">Alle</button>
        <button class="btn" id="btnNone">Geen</button>
      </div>
      <div id="stations"></div>
    </div>

    <div class="section">
      <h3 style="margin:0;">Y-as bereik</h3>
      <div class="row">
        <label for="ymin">Min:</label>
        <input id="ymin" type="number" step="any" placeholder="auto"/>
        <label for="ymax">Max:</label>
        <input id="ymax" type="number" step="any" placeholder="auto"/>
      </div>
      <div class="row">
        <button class="btn" id="btnApplyY">Toepassen</button>
        <button class="btn" id="btnAutoY">Auto</button>
      </div>
    </div>

    <div class="section">
      <h3 style="margin:0;">Tijdshift EXO (minuten)</h3>
      <div class="row">
        <label for="shiftMin">Δt (min):</label>
        <input id="shiftMin" type="number" step="1" value="0" />
      </div>
      <div class="row">
        <button class="btn" id="btnApplyShift">Shift toepassen</button>
        <button class="btn" id="btnResetShift">Reset EXO</button>
      </div>
      <div style="font-size:12px; color:#555; margin-top:4px;">
        Positief = later, negatief = vroeger. Enkel reeksen met bron "KorEXO" worden verschoven.
      </div>
    </div>
  </div>

  <div id="plotwrap"><div id="plot"></div></div>
</div>

<script>
  const fig = {fig_json};
  const config = {{
    responsive: true,
    displaylogo: false,
    modeBarButtonsToAdd: ['drawline','drawopenpath','eraseshape']
  }};
  const gd = document.getElementById('plot');

  Plotly.newPlot(gd, fig.data, fig.layout, config).then(() => {{
    // --- Bewaar originele X voor alle traces (voor reset & her-shift) ---
    const origX = gd.data.map(tr => (tr.x || []).map(x => new Date(x)));

    // --- Reeksen checkboxen ---
    const stationsDiv = document.getElementById('stations');
    const names = gd.data.map(tr => tr.name || 'Reeks');
    const vis = gd.data.map(tr => (typeof tr.visible === 'undefined') ? true : (tr.visible === true));

    names.forEach((nm, idx) => {{
      const row = document.createElement('div');
      row.className = 'station-item';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.id = 'st_' + idx;
      cb.checked = !!vis[idx];
      cb.addEventListener('change', () => {{
        Plotly.restyle(gd, {{visible: cb.checked}}, [idx]);
      }});
      const lab = document.createElement('label');
      lab.htmlFor = cb.id;
      lab.textContent = nm;
      row.appendChild(cb);
      row.appendChild(lab);
      stationsDiv.appendChild(row);
    }});

    document.getElementById('btnAll').addEventListener('click', () => {{
      const n = gd.data.length; const arr = Array(n).fill(true);
      Plotly.update(gd, {{visible: arr}});
      for (let i=0;i<n;i++) {{ const cb = document.getElementById('st_'+i); if (cb) cb.checked = true; }}
    }});
    document.getElementById('btnNone').addEventListener('click', () => {{
      const n = gd.data.length; const arr = Array(n).fill(false);
      Plotly.update(gd, {{visible: arr}});
      for (let i=0;i<n;i++) {{ const cb = document.getElementById('st_'+i); if (cb) cb.checked = false; }}
    }});

    // --- Y-as bediening ---
    const yminEl = document.getElementById('ymin');
    const ymaxEl = document.getElementById('ymax');

    document.getElementById('btnApplyY').addEventListener('click', () => {{
      const ymin = yminEl.value === '' ? null : Number(yminEl.value);
      const ymax = ymaxEl.value === '' ? null : Number(ymaxEl.value);
      const relayout = {{}};
      if (ymin !== null && !Number.isNaN(ymin) && ymax !== null && !Number.isNaN(ymax)) {{
        relayout['yaxis.autorange'] = false;
        relayout['yaxis.range'] = [ymin, ymax];
      }} else if ((ymin === null || Number.isNaN(ymin)) && (ymax === null || Number.isNaN(ymax))) {{
        relayout['yaxis.autorange'] = true;
      }} else {{
        const curr = gd.layout.yaxis.range || [null, null];
        const low  = (ymin !== null && !Number.isNaN(ymin)) ? ymin : curr[0];
        const high = (ymax !== null && !Number.isNaN(ymax)) ? ymax : curr[1];
        relayout['yaxis.autorange'] = false;
        relayout['yaxis.range'] = [low, high];
      }}
      Plotly.relayout(gd, relayout);
    }});

    document.getElementById('btnAutoY').addEventListener('click', () => {{
      Plotly.relayout(gd, {{'yaxis.autorange': true}});
      yminEl.value = '';
      ymaxEl.value = '';
    }});

    // --- Tijdshift EXO (meta == "KorEXO") ---
    function applyShift(mins) {{
      const ms = Number(mins) * 60000;
      const updates = [];
      const idxs = [];
      for (let i = 0; i < gd.data.length; i++) {{
        const tr = gd.data[i];
        if (tr.meta === 'KorEXO') {{
          const shifted = origX[i].map(d => new Date(d.getTime() + ms));
          updates.push({{ x: [shifted] }});
          idxs.push(i);
        }}
      }}
      // Batch restyle om knipperen te vermijden
      for (let k = 0; k < idxs.length; k++) {{
        Plotly.restyle(gd, updates[k], [idxs[k]]);
      }}
    }}

    function resetEXO() {{
      const updates = [];
      const idxs = [];
      for (let i = 0; i < gd.data.length; i++) {{
        const tr = gd.data[i];
        if (tr.meta === 'KorEXO') {{
          updates.push({{ x: [origX[i]] }});
          idxs.push(i);
        }}
      }}
      for (let k = 0; k < idxs.length; k++) {{
        Plotly.restyle(gd, updates[k], [idxs[k]]);
      }}
    }}

    document.getElementById('btnApplyShift').addEventListener('click', () => {{
      const val = Number(document.getElementById('shiftMin').value || '0');
      if (!Number.isFinite(val)) return;
      applyShift(val);
    }});
    document.getElementById('btnResetShift').addEventListener('click', () => {{
      document.getElementById('shiftMin').value = 0;
      resetEXO();
    }});
  }});
</script>
</body>
</html>
"""

def write_interactive_html(fig: go.Figure, out_path: Path) -> None:
    fig_json_str = pio.to_json(fig, pretty=False)
    title_text = fig.layout.title.text if getattr(fig.layout.title, "text", None) else "Plot"
    html_txt = TEMPLATE.format(title=html.escape(title_text), fig_json=fig_json_str)
    Path(out_path).write_text(html_txt, encoding="utf-8")


# ============== MAIN (2 dialogs) ==============
def main() -> None:
    root = tk.Tk(); root.withdraw()

    kor_paths = filedialog.askopenfilenames(
        title="Kies één of meerdere KorEXO CSV-exporten",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    water_paths = filedialog.askopenfilenames(
        title="Kies één of meerdere Waterinfo CSV's (multi-station of eenvoudig)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    frames: list[pd.DataFrame] = []
    problems: list[str] = []

    # KorEXO
    for pth in kor_paths:
        try:
            dfk = parse_korexo_csv(pth)
        except Exception as e:
            problems.append(f"KorEXO fout in {Path(pth).name}: {e}")
            continue
        if dfk is None or dfk.empty:
            problems.append(f"KorEXO leeg of onbruikbaar: {Path(pth).name}")
            continue
        dfk = dfk.copy()
        dfk["source"] = "KorEXO"
        frames.append(dfk[["Timestamp","Value","Series","parameter","unit","source"]])

    # Waterinfo
    for pth in water_paths:
        try:
            dfw = parse_multistation_csv(pth)
            if dfw.empty:
                dfw = parse_simple_waterinfo_csv(pth)
        except Exception as e:
            problems.append(f"Waterinfo fout in {Path(pth).name}: {e}")
            continue
        if dfw is None or dfw.empty:
            problems.append(f"Waterinfo leeg of onbruikbaar: {Path(pth).name}")
            continue

        dfw = dfw.copy()
        dfw["source"] = "Waterinfo"
        def mk_series(row):
            st = row.get("station_name") or "Station"
            pa = row.get("parameter") or "Parameter"
            un = row.get("unit") or ""
            return f"{st} – {pa}" + (f" ({un})" if un else "")
        dfw["Series"] = dfw.apply(mk_series, axis=1)
        frames.append(dfw[["Timestamp","Value","Series","parameter","unit","source"]])

    if not frames:
        message = "Geen bruikbare data gevonden."
        if problems:
            message += "\\n\\nDetails:\\n" + "\\n".join(problems)
        messagebox.showerror("Fout", message)
        sys.exit(1)

    big = pd.concat(frames, axis=0, ignore_index=True)
    big = big.dropna(subset=["Timestamp","Value"])
    # --- Tijdzone-normalisatie: alles naar Europe/Brussels (tz-naive) ---
    big['Timestamp'] = pd.to_datetime(big['Timestamp'], errors='coerce', utc=True)
    try:
        big['Timestamp'] = big['Timestamp'].dt.tz_convert('Europe/Brussels').dt.tz_localize(None)
    except Exception:
        big['Timestamp'] = big['Timestamp'].dt.tz_localize(None)
    big['Series'] = big['Series'].astype(str)

    big = big.sort_values(["Series","Timestamp"]).reset_index(drop=True)

    try:
        fig = build_plot(big)
    except Exception as e:
        messagebox.showerror("Plot-fout", str(e))
        sys.exit(1)

    out_path_str = filedialog.asksaveasfilename(
        title="Bewaar gecombineerde interactieve HTML grafiek",
        defaultextension=".html",
        filetypes=[("HTML file", "*.html")]
    )
    if not out_path_str:
        sys.exit(0)

    try:
        write_interactive_html(fig, Path(out_path_str))
    except Exception as e:
        messagebox.showerror("Opslaan mislukt", str(e))
        sys.exit(1)

    if problems:
        messagebox.showinfo("Klaar (met opmerkingen)",
            f"HTML grafiek bewaard:\\n{out_path_str}\\n\\nOpmerkingen:\\n" + "\\n".join(problems))
    else:
        messagebox.showinfo("Klaar", f"HTML grafiek bewaard:\\n{out_path_str}")

if __name__ == "__main__":
    main()
