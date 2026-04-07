#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV_waterinfo_multi_to_html.py

Leest een multi-station CSV met blokken van de vorm:
  #station_name;...
  #station_no;...
  #stationparameter_name;...
  #parametertype_name;...
  #ts_name;...
  #ts_unitname;...
  #rows;N
  #Timestamp;Value;Quality Code;Absolute Value;AV Quality Code
  <N data-rijen met ';' als scheiding>

Produceert een interactieve HTML-grafiek met:
- Alleen lijnen (geen markers/kolommen)
- Hover per lijn (alleen de lijn onder de muis)
- Checkboxen (Alle/Geen + per reeks toggle)
- Y-as min/max instellen (Toepassen) + Auto
- X-as range-selector en range-slider
- Plotly via CDN (stabiel) + figuur als JSON via plotly.io.to_json
"""

from __future__ import annotations

import sys
from pathlib import Path
import html

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# GUI (bestand kiezen en opslaan)
import tkinter as tk
from tkinter import filedialog, messagebox


# =============== Parsing ===============
def parse_multistation_csv(path: str | Path) -> pd.DataFrame:
    """
    Parseert een CSV met meerdere stations (zie module-comment).
    Retourneert DataFrame met kolommen:
      Timestamp (datetime64, TZ indien aanwezig),
      Value (float),
      station_name (str),
      station_no (str),
      parameter (str)  [stationparameter_name of parametertype_name],
      unit (str)       [ts_unitname]
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\r\n") for ln in f]

    recs: list[dict] = []
    cur = {
        "station_name": None,
        "station_no": None,
        "stationparameter_name": None,
        "parametertype_name": None,
        "ts_unitname": None,
        "rows": None,
    }
    in_data = False
    rows_needed: int | None = None
    rows_got = 0

    for ln in lines:
        if not ln:
            continue

        if ln.startswith("#"):
            # Kolomkop data -> volgende regels zijn data
            if ln.startswith("#Timestamp;"):
                in_data = True
                rows_got = 0
                continue

            # Algemene headerregel: key;value
            if ";" in ln:
                key, val = ln[1:].split(";", 1)
                key = key.strip()
                val = val.strip()
                cur[key] = val
                if key == "rows":
                    try:
                        rows_needed = int(val)
                    except Exception:
                        rows_needed = None
            continue

        # Datarij
        if in_data:
            parts = ln.split(";")
            if len(parts) >= 2:
                t_iso = parts[0].strip()
                v_raw = parts[1].strip().replace(",", ".")  # decimale komma -> punt

                # parse value
                try:
                    value = float(v_raw) if v_raw != "" else float("nan")
                except Exception:
                    value = float("nan")

                # parse timestamp (met TZ indien aanwezig)
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

                # Sluit blok af wanneer N rijen bereikt
                if rows_needed is not None and rows_got >= rows_needed:
                    in_data = False  # wachten op nieuwe headers

    df = pd.DataFrame.from_records(recs)

    # Diagnostiek in console
    print(f"[INFO] Rijen ingelezen (bruto): {len(df)}")
    if len(df) == 0:
        return df

    # Drop ongeldige timestamps en waarden
    before = len(df)
    df = df.dropna(subset=["Timestamp"])
    print(f"[INFO] Rijen met geldige Timestamp: {len(df)} (dropte {before - len(df)})")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Value"])
    print(f"[INFO] Rijen met geldige Value: {len(df)} (dropte {before - len(df)})")

    # Sorteer voor nette lijnen
    df = df.sort_values(["station_name", "Timestamp"]).reset_index(drop=True)

    # Samenvatting per station
    n_stations = df["station_name"].nunique(dropna=False)
    print(f"[INFO] Aantal stations aangetroffen: {n_stations}")
    for s, g in df.groupby("station_name", dropna=False):
        sname = s if pd.notna(s) else "Onbekend station"
        if not g.empty:
            print(f"   - {sname}: {len(g)} rijen, tijd {g['Timestamp'].min()} → {g['Timestamp'].max()}")

    return df


# =============== Plot (alleen lijnen, hover per trace) ===============
def build_plot(df: pd.DataFrame) -> go.Figure:
    """
    Maakt een interactieve multi-station lijnplot.
    Verwacht kolommen: Timestamp, Value, station_name, parameter, unit (optioneel station_no).
    """
    if df.empty:
        raise ValueError("Geen data doorgegeven aan build_plot().")

    # y-as label en titel bepalen
    param = df["parameter"].dropna().astype(str)
    unit  = df["unit"].dropna().astype(str)
    param_str = param.iloc[0] if len(param) else "Parameter"
    unit_str  = unit.iloc[0]  if len(unit)  else ""
    y_label = f"{param_str} ({unit_str})" if unit_str else param_str
    title   = f"{y_label} – meerdere stations"

    fig = go.Figure()

    # Eén lijntracé per station (geen markers/kolommen)
    added = 0
    for station, g in df.groupby("station_name", dropna=False):
        g = g.sort_values("Timestamp")
        if g.empty:
            continue

        station_label = str(station) if pd.notna(station) else "Onbekend station"
        # parameter en unit (voor hover)
        p = str(g["parameter"].dropna().iloc[0]) if "parameter" in g and not g["parameter"].dropna().empty else param_str
        u = str(g["unit"].dropna().iloc[0]) if "unit" in g and not g["unit"].dropna().empty else unit_str

        hover = (
            "<b>Station:</b> " + station_label + "<br>" +
            "<b>Tijd:</b> %{x}<br>" +
            "<b>Waarde:</b> %{y}<br>" +
            "<b>Parameter:</b> " + p + (f" ({u})" if u else "") + "<br>" +
            "<extra></extra>"   # leeg extra-blokje => geen extra legendabox
        )

        fig.add_trace(go.Scattergl(
            x=g["Timestamp"],
            y=g["Value"],
            mode="lines",             # alleen lijnen
            line=dict(width=1.8),
            name=station_label,
            hovertemplate=hover
        ))
        added += 1

    if added == 0:
        raise ValueError("Er zijn geen reeksen toegevoegd (mogelijk alle groepen leeg).")

    # Interactie & layout
    fig.update_layout(
        title=title,
        xaxis_title="Tijd",
        yaxis_title=y_label,
        hovermode="closest",                # enkel hover voor trace onder de muis
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

    # Hoverlabel-styling (beter leesbaar)
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
  #controls {{ width: 320px; min-width: 240px; max-width: 45vw; overflow:auto; border-right:1px solid #ddd; padding:12px; box-sizing: border-box; }}
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
    fig_json_str = pio.to_json(fig, pretty=False)  # JSON string (veilig voor numpy/datetime)
    title_text = fig.layout.title.text if getattr(fig.layout.title, "text", None) else "Plot"

    html_txt = TEMPLATE.format(
        title=html.escape(title_text),
        fig_json=fig_json_str
    )
    Path(out_path).write_text(html_txt, encoding="utf-8")


# =============== GUI-main ===============
def main() -> None:
    root = tk.Tk()
    root.withdraw()

    in_path = filedialog.askopenfilename(
        title="Kies multi-station CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not in_path:
        sys.exit(0)

    try:
        df = parse_multistation_csv(in_path)
    except Exception as e:
        messagebox.showerror("Fout bij inlezen", str(e))
        sys.exit(1)

    if df.empty:
        messagebox.showwarning("Geen data", "Er zijn geen bruikbare rijen gevonden (controleer headers/kolommen).")
        sys.exit(0)

    if df["Value"].isna().all():
        messagebox.showwarning("Geen waarden", "Alle waarden zijn NaN; controleer decimale scheiding en kolomvolgorde.")
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
