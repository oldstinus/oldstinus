#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV_waterinfo_multi_files_to_html_dual_axis_NO_LEGEND_vars_as_name.py

Doel:
- Meerdere Waterinfo CSV-bestanden (met multi-header) inlezen.
- Interactieve Plotly-grafiek met één of twee Y-assen (links/rechts).
- GEEN Plotly-legende op de grafiek zelf.
- Reeksen in-/uitschakelen via checkboxen in de kantlijn.
- Trace-naam (en dus ook de "legende"-tekst in de kantlijn) en hover-info:
  station_name_station_no_stationparameter_name_ts_unitname

Voorbeeld trace-naam:
  Weert SF/Zeeschelde_zes39c-SF-1066_PPFD1_micromol/sec/sq m

Compatibel met Waterinfo CSV's met multi-header, zoals:

#station_name;Weert SF/Zeeschelde
#station_no;zes39c-SF-1066
#stationparameter_name;PPFD1
#parametertype_name;PPFD
#ts_name;O.01d
#ts_unitname;micromol/sec/sq m
#rows;26752
#Timestamp;Value;Quality Code;Absolute Value;AV Quality Code
...
"""

from __future__ import annotations
from pathlib import Path
import sys
import html

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# GUI
import tkinter as tk
from tkinter import filedialog, messagebox


# ---------- Parser voor één CSV ----------

def parse_multistation_csv(path: str | Path) -> pd.DataFrame:
    """
    Leest één Waterinfo-CSV met multi-header en datablok.
    Geeft een DataFrame terug met o.a.:

    Timestamp, Value, station_name, station_no,
    stationparameter_name, parametertype_name, ts_unitname,
    parameter (alias voor stationparameter_name of parametertype_name),
    unit (alias voor ts_unitname).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\r\n") for ln in f]

    recs = []
    cur = {
        "station_name": None,
        "station_no": None,
        "stationparameter_name": None,
        "parametertype_name": None,
        "ts_unitname": None,
        "rows": None,
    }
    in_data = False
    rows_needed = None
    rows_got = 0

    for ln in lines:
        if not ln:
            continue

        if ln.startswith("#"):
            # Start van het datablok
            if ln.startswith("#Timestamp;"):
                in_data = True
                rows_got = 0
                continue

            # Headerregel "#key;value"
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

        # Datablok
        if in_data:
            parts = ln.split(";")
            if len(parts) >= 2:
                t_iso = parts[0].strip()
                v_raw = parts[1].strip().replace(",", ".")
                try:
                    value = float(v_raw) if v_raw != "" else float("nan")
                except Exception:
                    value = float("nan")

                ts = pd.to_datetime(t_iso, errors="coerce")

                recs.append({
                    "Timestamp": ts,
                    "Value": value,
                    "station_name": cur.get("station_name"),
                    "station_no": cur.get("station_no"),
                    "stationparameter_name": cur.get("stationparameter_name"),
                    "parametertype_name": cur.get("parametertype_name"),
                    # "parameter" = preferent stationparameter_name, anders parametertype_name
                    "parameter": cur.get("stationparameter_name") or cur.get("parametertype_name"),
                    "ts_unitname": cur.get("ts_unitname"),
                    # unit blijft alias voor ts_unitname (voor bestaande code)
                    "unit": cur.get("ts_unitname"),
                })
                rows_got += 1

                if rows_needed is not None and rows_got >= rows_needed:
                    in_data = False

    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return df

    df = df.dropna(subset=["Timestamp"]).copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    df = df.sort_values(["station_name", "Timestamp"]).reset_index(drop=True)
    return df


# ---------- Meerdere CSV's laden ----------

def load_many_csv(paths: list[Path]) -> pd.DataFrame:
    """
    Leest meerdere CSV-bestanden in en voegt ze samen.
    Kolom 'source_file' bewaart de bestandsnaam (voor debug, maar niet meer in de namen).
    """
    frames = []
    for p in paths:
        try:
            dfi = parse_multistation_csv(p)
        except Exception as e:
            print(f"[WARN] Fout bij inlezen {p}: {e}")
            continue

        if not dfi.empty:
            dfi["source_file"] = p.name
            frames.append(dfi)

    if not frames:
        return pd.DataFrame(columns=[
            "Timestamp", "Value", "station_name", "station_no",
            "stationparameter_name", "parametertype_name",
            "parameter", "ts_unitname", "unit", "source_file"
        ])

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["parameter", "station_name", "Timestamp"]).reset_index(drop=True)
    return df


# ---------- Parameteraanwijzer (Tk) ----------

def choose_secondary_parameter(root: tk.Tk, params: list[str]) -> str | None:
    """
    Laat de gebruiker optioneel één parameter kiezen die op Y2 komt.
    Return None voor 'Geen tweede as'.
    """
    if not params:
        return None

    sel = {"value": None}
    win = tk.Toplevel(root)
    win.title("Kies parameter voor tweede Y-as (optioneel)")
    win.grab_set()
    win.resizable(False, False)

    tk.Label(
        win,
        text="Plaats één parameter op de rechter Y-as.\n"
             "Kies een parameter of klik op 'Geen tweede as'.",
        justify="left",
        padx=8, pady=8
    ).pack(anchor="w")

    lb = tk.Listbox(win, height=min(12, len(params)), exportselection=False)
    for p in params:
        lb.insert(tk.END, p)
    lb.pack(fill="both", expand=True, padx=8)

    btnframe = tk.Frame(win)
    btnframe.pack(fill="x", pady=8)

    def ok():
        idxs = lb.curselection()
        if idxs:
            sel["value"] = params[idxs[0]]
        win.destroy()

    def none():
        sel["value"] = None
        win.destroy()

    tk.Button(btnframe, text="OK", width=12, command=ok).pack(side="left", padx=8)
    tk.Button(btnframe, text="Geen tweede as", width=14, command=none).pack(side="right", padx=8)

    root.wait_window(win)
    return sel["value"]


# ---------- Plot bouwen ----------

def build_plot(df: pd.DataFrame,
               secondary_param: str | None = None) -> go.Figure:
    """
    Bouwt een Plotly-figuur met:
    - Y1 (links) en optioneel Y2 (rechts).
    - Trace-namen = station_name_station_no_stationparameter_name_ts_unitname.
    - Hovertekst zonder bestandsnaam.

    secondary_param:
      None → alles op Y1
      str  → reeksen met deze parameter komen op Y2 (rechts)
    """
    if df.empty:
        raise ValueError("Lege dataset.")

    # Unieke param/units → labels
    params_ordered = df["parameter"].dropna().astype(str)

    units_map = (
        df.dropna(subset=["parameter"])
          .drop_duplicates(subset=["parameter"], keep="first")
          .set_index("parameter")["unit"]
          .to_dict()
    )

    # Y2 label
    if secondary_param:
        u2 = units_map.get(secondary_param, None)
        y2_label = f"{secondary_param} ({u2})" if (u2 and str(u2).strip()) else secondary_param
    else:
        y2_label = None

    # Y1 label (eerste niet-secondary param)
    y1_param = None
    for p in params_ordered:
        if secondary_param is None or p != secondary_param:
            y1_param = p
            break
    if y1_param is None and not params_ordered.empty:
        y1_param = params_ordered.iloc[0]
    u1 = units_map.get(y1_param, None) if y1_param else None
    y1_label = f"{y1_param} ({u1})" if (y1_param and u1 and str(u1).strip()) else (y1_param or "Waarde")

    title_extra = ""
    n_param = df["parameter"].nunique(dropna=True)
    n_unit = df["unit"].nunique(dropna=True)
    if n_param > 1 or n_unit > 1:
        title_extra = " – meerdere parameters/eenheden"

    title = f"Tijdreeks – Y1: {y1_label}" + (f" | Y2: {y2_label}" if y2_label else "") + title_extra

    fig = go.Figure()

    # Groepering per trace (bestandsnaam wordt NIET meer gebruikt in de naam, maar wel nog als kolom)
    group_cols = ["station_name", "source_file", "parameter", "unit"]

    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("Timestamp")
        if g.empty:
            continue

        # Metadata ophalen uit de groep
        station_label = str(g["station_name"].dropna().iloc[0]) if "station_name" in g and not g["station_name"].dropna().empty else ""
        station_no    = str(g["station_no"].dropna().iloc[0]) if "station_no" in g and not g["station_no"].dropna().empty else ""
        # stationparameter_name kan ontbreken → fallback op 'parameter' of 'parametertype_name'
        sp_name = ""
        if "stationparameter_name" in g and not g["stationparameter_name"].dropna().empty:
            sp_name = str(g["stationparameter_name"].dropna().iloc[0])
        elif "parameter" in g and not g["parameter"].dropna().empty:
            sp_name = str(g["parameter"].dropna().iloc[0])
        elif "parametertype_name" in g and not g["parametertype_name"].dropna().empty:
            sp_name = str(g["parametertype_name"].dropna().iloc[0])

        # Eenheid uit ts_unitname (zoals in de header)
        if "ts_unitname" in g and not g["ts_unitname"].dropna().empty:
            unit_name = str(g["ts_unitname"].dropna().iloc[0])
        elif "unit" in g and not g["unit"].dropna().empty:
            unit_name = str(g["unit"].dropna().iloc[0])
        else:
            unit_name = ""

        # Unieke trace-naam volgens jouw specificatie:
        # station_name_station_no_stationparameter_name_ts_unitname
        trace_name = f"{station_label}_{station_no}_{sp_name}_{unit_name}"

        # Hovertemplate (muisover)
        hover = (
            f"<b>{station_label}</b><br>"
            f"Stationnr: {station_no}<br>"
            f"Parameter: {sp_name}<br>"
            f"Eenheid: {unit_name}<br>"
            "<b>Tijd:</b> %{x}<br>"
            "<b>Waarde:</b> %{y}<br>"
            "<extra></extra>"
        )

        # Y2 indien juiste parameter
        p = str(g["parameter"].dropna().iloc[0]) if "parameter" in g and not g["parameter"].dropna().empty else ""
        use_y2 = (secondary_param is not None and p == secondary_param)

        fig.add_trace(go.Scattergl(
            x=g["Timestamp"],
            y=g["Value"],
            mode="lines",
            line=dict(width=1.8),
            name=trace_name,
            hovertemplate=hover,
            yaxis="y2" if use_y2 else "y1",
        ))

    # Layout: géén legende op de grafiek, wel hover en tools
    fig.update_layout(
        title=title,
        showlegend=False,  # <-- BELANGRIJK: geen Plotly-legende
        xaxis=dict(
            title="Tijd",
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
        ),
        yaxis=dict(title=y1_label, showspikes=True, spikemode="across"),
        hovermode="closest",
        dragmode="zoom",
        margin=dict(l=70, r=70 if secondary_param else 20, t=70, b=70)
    )

    if secondary_param:
        fig.update_layout(
            yaxis2=dict(
                title=y2_label,
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    fig.update_traces(
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=12,
            font_family="Arial"
        )
    )

    # Dropdown "Alle/Geen/Alleen deze" (optioneel, stoort de eis niet)
    n = len(list(fig.data))
    if n > 0:
        buttons = [
            dict(label="Alle", method="update", args=[{"visible": [True] * n}]),
            dict(label="Geen", method="update", args=[{"visible": [False] * n}]),
        ]
        for i, tr in enumerate(fig.data):
            vis = [False] * n
            vis[i] = True
            buttons.append(dict(
                label=f"Alleen: {tr.name[:40]}",
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


# ---------- HTML template met Y1 & Y2 bediening en kantlijn-reeksen ----------

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
  .subtle {{ color:#666; font-size:12px; }}
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
      <h3 style="margin:0;">Y1-as (links)</h3>
      <div class="row">
        <label for="y1min">Min:</label>
        <input id="y1min" type="number" step="any" placeholder="auto"/>
        <label for="y1max">Max:</label>
        <input id="y1max" type="number" step="any" placeholder="auto"/>
      </div>
      <div class="row">
        <button class="btn" id="btnApplyY1">Toepassen</button>
        <button class="btn" id="btnAutoY1">Auto</button>
      </div>
    </div>

    <div class="section">
      <h3 style="margin:0;">Y2-as (rechts)</h3>
      <div class="row">
        <label for="y2min">Min:</label>
        <input id="y2min" type="number" step="any" placeholder="auto"/>
        <label for="y2max">Max:</label>
        <input id="y2max" type="number" step="any" placeholder="auto"/>
      </div>
      <div class="row">
        <button class="btn" id="btnApplyY2">Toepassen</button>
        <button class="btn" id="btnAutoY2">Auto</button>
      </div>
      <div class="subtle" id="y2hint"></div>
    </div>
  </div>

  <div id="plotwrap"><div id="plot"></div></div>
</div>

<script>
  const fig = {fig_json};
  const hasY2 = {has_y2};
  const y2Label = {y2_label_json};

  const config = {{
    responsive: true,
    displaylogo: false,
    modeBarButtonsToAdd: ['drawline','drawopenpath','eraseshape']
  }};
  const gd = document.getElementById('plot');

  Plotly.newPlot(gd, fig.data, fig.layout, config).then(() => {{
    const stationsDiv = document.getElementById('stations');
    const names = gd.data.map(tr => tr.name || 'Reeks');
    const vis = gd.data.map(tr => (typeof tr.visible === 'undefined') ? true : (tr.visible === true));

    // Checkbox-lijst: gebruikt trace.name, dus nu bv.
    // Weert SF/Zeeschelde_zes39c-SF-1066_PPFD1_micromol/sec/sq m
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

    // Y1 bediening
    const y1minEl = document.getElementById('y1min');
    const y1maxEl = document.getElementById('y1max');
    document.getElementById('btnApplyY1').addEventListener('click', () => {{
      const ymin = y1minEl.value === '' ? null : Number(y1minEl.value);
      const ymax = y1maxEl.value === '' ? null : Number(y1maxEl.value);
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
    document.getElementById('btnAutoY1').addEventListener('click', () => {{
      Plotly.relayout(gd, {{'yaxis.autorange': true}});
      y1minEl.value = ''; y1maxEl.value = '';
    }});

    // Y2 bediening
    document.getElementById('y2hint').textContent = hasY2 ? ('Tweede as actief: ' + (y2Label || '')) : 'Geen tweede as ingesteld.';
    const y2minEl = document.getElementById('y2min');
    const y2maxEl = document.getElementById('y2max');
    document.getElementById('btnApplyY2').addEventListener('click', () => {{
      const ymin = y2minEl.value === '' ? null : Number(y2minEl.value);
      const ymax = y2maxEl.value === '' ? null : Number(y2maxEl.value);
      const relayout = {{}};
      if (ymin !== null && !Number.isNaN(ymin) && ymax !== null && !Number.isNaN(ymax)) {{
        relayout['yaxis2.autorange'] = false;
        relayout['yaxis2.range'] = [ymin, ymax];
      }} else if ((ymin === null || Number.isNaN(ymin)) && (ymax === null || Number.isNaN(ymax))) {{
        relayout['yaxis2.autorange'] = true;
      }} else {{
        const curr = (gd.layout.yaxis2 && gd.layout.yaxis2.range) || [null, null];
        const low  = (ymin !== null && !Number.isNaN(ymin)) ? ymin : curr[0];
        const high = (ymax !== null && !Number.isNaN(ymax)) ? ymax : curr[1];
        relayout['yaxis2.autorange'] = false;
        relayout['yaxis2.range'] = [low, high];
      }}
      Plotly.relayout(gd, relayout);
    }});
    document.getElementById('btnAutoY2').addEventListener('click', () => {{
      Plotly.relayout(gd, {{'yaxis2.autorange': true}});
      y2minEl.value = ''; y2maxEl.value = '';
    }});
  }});
</script>
</body>
</html>
"""

def write_interactive_html(fig: go.Figure, out_path: Path, has_y2: bool, y2_label: str | None) -> None:
    fig_json_str = pio.to_json(fig, pretty=False)
    title_text = fig.layout.title.text if getattr(fig.layout.title, "text", None) else "Plot"
    html_txt = TEMPLATE.format(
        title=html.escape(title_text),
        fig_json=fig_json_str,
        has_y2="true" if has_y2 else "false",
        y2_label_json=("null" if not y2_label else '"' + html.escape(y2_label) + '"')
    )
    Path(out_path).write_text(html_txt, encoding="utf-8")


# ---------- Main ----------

def main() -> None:
    root = tk.Tk()
    root.withdraw()

    # CSV-bestanden selecteren
    file_paths = filedialog.askopenfilenames(
        title="Selecteer één of meerdere Waterinfo-CSV’s",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file_paths:
        sys.exit(0)
    paths = [Path(p) for p in file_paths]

    # Data inlezen
    try:
        df = load_many_csv(paths)
    except Exception as e:
        messagebox.showerror("Fout bij inlezen", str(e))
        sys.exit(1)

    if df.empty or df["Value"].isna().all():
        messagebox.showwarning("Geen data", "Geen bruikbare waarden gevonden in de geselecteerde bestanden.")
        sys.exit(0)

    # Optioneel: parameter op Y2
    unique_params = sorted([p for p in df["parameter"].dropna().astype(str).unique()])
    secondary_param = choose_secondary_parameter(root, unique_params)

    # Figuur bouwen
    try:
        fig = build_plot(df, secondary_param=secondary_param)
    except Exception as e:
        messagebox.showerror("Plot-fout", str(e))
        sys.exit(1)

    # Bestandsnaam voor HTML
    out_path_str = filedialog.asksaveasfilename(
        title="Bewaar interactieve HTML-grafiek",
        defaultextension=".html",
        filetypes=[("HTML file", "*.html")]
    )
    if not out_path_str:
        sys.exit(0)

    # Label voor Y2 doorgeven
    y2_label = None
    if secondary_param and hasattr(fig.layout, "yaxis2"):
        y2_label = fig.layout.yaxis2.title.text

    try:
        write_interactive_html(fig, Path(out_path_str), has_y2=bool(secondary_param), y2_label=y2_label)
    except Exception as e:
        messagebox.showerror("Opslaan mislukt", str(e))
        sys.exit(1)

    msg = f"HTML-grafiek bewaard:\n{out_path_str}"
    if secondary_param:
        msg += f"\nTweede Y-as (rechts): {secondary_param}"
    messagebox.showinfo("Klaar", msg)


if __name__ == "__main__":
    main()
