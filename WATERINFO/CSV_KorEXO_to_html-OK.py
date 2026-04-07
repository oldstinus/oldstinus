
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV_KorEXO_to_html.py

Leest YSI KorEXO "Measurement Data File Export" CSV's (meestal UTF-16LE)
met meerdere meetblokken. Elk blok start met een regel:
  ",,,SENSOR SERIAL NUMBER,..."
gevolgd door een kolomtitelregel.
Daarna volgen meetrijen. Aan het einde van elk blok staan meestal samenvattingen:
  ",,,MEAN VALUE:,..."
  ",,,STANDARD DEVIATION:,..."
— deze worden genegeerd.

Output: een standalone interactieve HTML-grafiek (Plotly via CDN) met:
- Alleen lijnen (geen markers)
- Hover per lijn (alleen de lijn onder de muis)
- Checkboxen: Alle / Geen + per reeks
- Y-as bereik instellen (Toepassen) + Auto
- X-as range-selector + range-slider
- figuur-JSON via plotly.io.to_json

Gebruik:
  python CSV_KorEXO_to_html.py  # GUI-bestandskeuze en opslaan
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

# GUI (bestand kiezen en opslaan)
import tkinter as tk
from tkinter import filedialog, messagebox


MEAN_TOKEN = "MEAN VALUE"
STD_TOKEN  = "STANDARD DEVIATION"
SER_TOKEN  = "SENSOR SERIAL NUMBER"

NON_VALUE_COL_PREFIXES = ("Date", "Time", "Site Name")

def _try_open_text(path: Path):
    """
    Open tekstbestand met juiste encoding. KorEXO export is doorgaans UTF-16.
    Valt terug op UTF-8 bij mislukking.
    """
    for enc in ("utf-16", "utf-16le", "utf-8-sig", "utf-8"):
        try:
            return path.open("r", encoding=enc, errors="ignore")
        except Exception:
            continue
    # Laat fout bubbelen met standaard encoding
    return path.open("r")

def _is_mean_or_std(cells: List[str]) -> bool:
    # Match op een van de cellen de tokens
    joined = ",".join(cells).upper()
    return (MEAN_TOKEN in joined) or (STD_TOKEN in joined)

def _is_serial_header(cells: List[str]) -> bool:
    joined = ",".join(cells).upper()
    return SER_TOKEN in joined

def _row_is_data(cells: List[str]) -> bool:
    """
    Heuristiek: dataregel start meestal met "M/D/YYYY" en "HH:MM:SS".
    We testen iig op datum aan het begin (kolom 0).
    """
    if not cells:
        return False
    c0 = cells[0].strip()
    # datum patroon 1-2/1-2/4
    return bool(re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", c0))

def parse_korexo_csv(path: str | Path) -> pd.DataFrame:
    """
    Parseert KorEXO export met meerdere blokken.
    Retourneert een 'lange' DataFrame met kolommen:
      Timestamp (datetime64), Series (str), Value (float), Block (int), Site (str)
    waarbij 'Series' de samengestelde kolomtitel is (header + [serial indien bekend]).
    """
    p = Path(path)
    rows_long: List[Dict[str, Any]] = []

    with _try_open_text(p) as f:
        reader = csv.reader(f)
        block_idx = -1
        current_serials: List[str] | None = None
        current_header:  List[str] | None = None
        site_name_idx: int | None = None

        for raw_cells in reader:
            # CSV kan lege cellen bevatten; normaliseer naar strings
            cells = [c.strip() for c in raw_cells]

            if not any(cells):
                # lege regel -> ga door
                continue

            if _is_mean_or_std(cells):
                # expliciet overslaan
                continue

            if _is_serial_header(cells):
                # start nieuw blok
                block_idx += 1
                current_serials = cells
                current_header  = None
                site_name_idx   = None
                continue

            # Als we een header verwachten, neem de eerste niet-mean/std regel NA serials
            if current_serials is not None and current_header is None and not _is_mean_or_std(cells):
                current_header = cells
                # Zoek site-name kolom (optioneel)
                for i, h in enumerate(current_header):
                    if h.startswith("Site Name"):
                        site_name_idx = i
                        break
                continue

            # Data?
            if current_header is not None and _row_is_data(cells):
                # Maak samengestelde kolomnamen
                headers = []
                n = max(len(current_header), len(current_serials or []))
                for i in range(n):
                    h = current_header[i] if i < len(current_header) else f"col{i}"
                    s = (current_serials[i] if (current_serials and i < len(current_serials)) else "").strip()
                    # voeg serial toe indien aanwezig en niet leeg (en niet "Date", "Time", etc.)
                    if s and not h.startswith(NON_VALUE_COL_PREFIXES):
                        headers.append(f"{h} [{s}]")
                    else:
                        headers.append(h)

                # Bewaak lengte cells t.o.v. headers
                # (vul aan met lege strings of cutoff als te lang)
                if len(cells) < len(headers):
                    cells = cells + [""] * (len(headers) - len(cells))
                elif len(cells) > len(headers):
                    cells = cells[:len(headers)]

                # Bouw datum/tijd
                date_str = cells[0]
                time_str = cells[1] if len(cells) > 1 else "00:00:00"
                # fractie seconden negeren voor nu
                ts = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")

                # Site naam (optioneel)
                site = ""
                if site_name_idx is not None and site_name_idx < len(cells):
                    site = cells[site_name_idx]

                # Zet alle waarde-kolommen naar long records
                for i, lab in enumerate(headers):
                    if lab.startswith(NON_VALUE_COL_PREFIXES):
                        continue
                    val_str = cells[i]
                    # Komma naar punt indien aanwezig
                    val_str = val_str.replace(",", ".")
                    try:
                        val = float(val_str)
                    except Exception:
                        # skip niet-numeriek
                        continue
                    rows_long.append({
                        "Timestamp": ts,
                        "Series": lab,
                        "Value": val,
                        "Block": block_idx,
                        "Site": site
                    })

            # Andere regels buiten data laten we ongemoeid

    df = pd.DataFrame.from_records(rows_long)
    if df.empty:
        return df

    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values(["Series", "Timestamp"]).reset_index(drop=True)

    # Diagnostiek
    print(f"[INFO] Rijen (long) ingelezen: {len(df)}")
    print(f"[INFO] Aantal reeksen: {df['Series'].nunique()}")
    print(f"[INFO] Blokken gevonden: {df['Block'].nunique()}")

    return df


# =============== Plot (alleen lijnen, hover per trace) ===============
def build_plot(df: pd.DataFrame) -> go.Figure:
    """
    Maakt een interactieve multi-reeks lijnplot.
    Verwacht kolommen: Timestamp, Value, Series (str), optioneel Site, Block.
    """
    if df.empty:
        raise ValueError("Geen data doorgegeven aan build_plot().")

    # y-as label en titel bepalen
    y_label = "Waarde"
    title   = "KorEXO metingen – meerdere reeksen"

    fig = go.Figure()

    # Eén lijntracé per 'Series'
    added = 0
    for series, g in df.groupby("Series", dropna=False):
        g = g.sort_values("Timestamp")
        if g.empty:
            continue

        hover = (
            "<b>Reeks:</b> " + html.escape(str(series)) + "<br>" +
            "<b>Tijd:</b> %{x}<br>" +
            "<b>Waarde:</b> %{y}<br>" +
            "<extra></extra>"
        )

        fig.add_trace(go.Scattergl(
            x=g["Timestamp"],
            y=g["Value"],
            mode="lines+markers",
            marker=dict(size=4, symbol="circle", opacity=0.8),
            line=dict(width=1.6),
            name=str(series),
            hovertemplate=hover
        ))
        added += 1

    if added == 0:
        raise ValueError("Er zijn geen reeksen toegevoegd.")

    # Interactie & layout
    fig.update_layout(
        title=title,
        xaxis_title="Tijd",
        yaxis_title=y_label,
        hovermode="closest",
        dragmode="zoom",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=60, r=20, t=60, b=60)
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1u",  step="hour", stepmode="backward"),
                dict(count=6,  label="6u",  step="hour", stepmode="backward"),
                dict(count=12, label="12u", step="hour", stepmode="backward"),
                dict(count=1,  label="1d",  step="day",  stepmode="backward"),
                dict(count=3,  label="3d",  step="day",  stepmode="backward"),
                dict(step="all", label="Alles")
            ]
        ),
        showspikes=True, spikemode="across"
    )
    fig.update_yaxes(showspikes=True, spikemode="across")

    fig.update_traces(
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=12,
            font_family="Arial"
        )
    )

    # Dropdown “Alle / Geen / Alleen deze”
    n = len(list(fig.data))
    if n > 0:
        buttons = [
            dict(label="Alle", method="update", args=[{"visible": [True]*n}]),
            dict(label="Geen", method="update", args=[{"visible": [False]*n}]),
        ]
        for i, tr in enumerate(fig.data):
            vis = [False]*n
            vis[i] = True
            buttons.append(dict(
                label=f"Alleen: {tr.name[:32]}",
                method="update",
                args=[{"visible": vis}]
            ))
        fig.update_layout(
            updatemenus=[dict(
                type="dropdown", direction="down",
                x=1.0, xanchor="right", y=1.12, yanchor="top",
                buttons=buttons, showactive=False
            )]
        )

    return fig


# =============== HTML TEMPLATE (Plotly via CDN) + WRITER ===============
TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  body {{ margin:0; font-family: Arial, Helvetica, sans-serif; }}
  #container {{ display:flex; flex-direction: row; height: 100vh; }}
  #controls {{ width: 340px; min-width: 240px; max-width: 45vw; overflow:auto; border-right:1px solid #ddd; padding:12px; box-sizing: border-box; }}
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
<!-- Plotly vanaf CDN (stabiel) -->
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
    // Reeks-checkboxen
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

    // Alle/Geen
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

    // Y-as bediening
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
  }});
</script>
</body>
</html>
"""

def write_interactive_html(fig: go.Figure, out_path: Path) -> None:
    """
    Schrijf standalone HTML (Plotly via CDN) met:
      - Reeksvinkjes (Alle/Geen)
      - Y-as min/max Apply + Auto
    """
    fig_json_str = pio.to_json(fig, pretty=False)
    title_text = fig.layout.title.text if getattr(fig.layout.title, "text", None) else "Plot"

    html_txt = TEMPLATE.format(
        title=html.escape(title_text),
        fig_json=fig_json_str
    )
    Path(out_path).write_text(html_txt, encoding="utf-8")


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    in_path = filedialog.askopenfilename(
        title="Kies KorEXO CSV-export",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not in_path:
        sys.exit(0)

    try:
        df = parse_korexo_csv(in_path)
    except Exception as e:
        messagebox.showerror("Fout bij inlezen", str(e))
        sys.exit(1)

    if df.empty:
        messagebox.showwarning("Geen data", "Geen bruikbare rijen gevonden (controleer blokken en kolommen).")
        sys.exit(0)

    try:
        fig = build_plot(df)
    except Exception as e:
        messagebox.showerror("Plot-fout", str(e))
        sys.exit(1)

    out_path_str = filedialog.asksaveasfilename(
        title="Bewaar interactieve HTML grafiek",
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

    messagebox.showinfo("Klaar", f"HTML grafiek bewaard:\n{out_path_str}")


if __name__ == "__main__":
    main()
