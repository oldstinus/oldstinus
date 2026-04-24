import os
import sys
import importlib.util
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime, timedelta
from pathlib import Path

try:
    import chardet
except Exception:
    chardet = None


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


# ------------------------------------------------------
# Bestand selectie: meerdere .dis bestanden toegestaan
# ------------------------------------------------------
def select_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Selecteer één of meerdere .dis bestanden",
        filetypes=[("Discharge Files", "*.dis"), ("All Files", "*.*")]
    )
    if not file_paths:
        messagebox.showerror("Geen bestanden", "Selecteer minstens één .dis bestand.")
        sys.exit()
    return list(file_paths)


# ------------------------------------------------------
# Detecteert encoding
# ------------------------------------------------------
def detect_encoding(file_path, num_bytes=10000):
    with open(file_path, 'rb') as f:
        raw = f.read(num_bytes)
    if chardet is not None:
        result = chardet.detect(raw)
        encoding = result.get('encoding')
        if encoding:
            return encoding

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            raw.decode(encoding)
            return encoding
        except Exception:
            continue
    return "latin-1"


# ------------------------------------------------------
# Duur omzetten naar timedelta
# ------------------------------------------------------
def parse_duration(s):
    parts = s.strip().split(':')
    if len(parts) == 3:
        h, m, sec = map(int, parts)
    elif len(parts) == 2:
        h = 0
        m, sec = map(int, parts)
    else:
        raise ValueError(f"Onbekend Duration-formaat: {s}")
    return timedelta(hours=h, minutes=m, seconds=sec)


# ------------------------------------------------------
# Verwerken van één .dis bestand → DataFrame
# ------------------------------------------------------
def process_single_dis(file_path):
    encoding = detect_encoding(file_path)
    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=53,
        engine="python",
        on_bad_lines="warn",
        encoding=encoding,
        dtype=str
    )

    df.columns = [c.strip() for c in df.columns]

    required = ['Start Date', 'Start Time', 'Duration',
                'Total Q (m3/s)', 'Mean Speed (m/s)',
                'Transect', 'File name']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' ontbreekt in {file_path}")

    df.dropna(subset=required, inplace=True)

    df['Start Datetime'] = pd.to_datetime(
        df['Start Date'] + " " + df['Start Time'],
        format='%m/%d/%Y %I:%M:%S %p',
        errors='coerce'
    )
    df.dropna(subset=['Start Datetime'], inplace=True)

    df['Duration'] = df['Duration'].apply(parse_duration)
    df['Aangrijpingstijd'] = df['Start Datetime'] + df['Duration'] / 2
    df.dropna(subset=['Aangrijpingstijd'], inplace=True)

    df['Total Q (m3/s)'] = pd.to_numeric(df['Total Q (m3/s)'], errors='coerce')
    df['Mean Speed (m/s)'] = pd.to_numeric(df['Mean Speed (m/s)'], errors='coerce')
    df.dropna(subset=['Total Q (m3/s)', 'Mean Speed (m/s)'], inplace=True)

    df['Source File'] = os.path.basename(file_path)

    return df


# ------------------------------------------------------
# Combineert meerdere bestanden
# ------------------------------------------------------
def process_multiple(files):
    frames = []
    for f in files:
        try:
            df = process_single_dis(f)
            frames.append(df)
        except Exception as e:
            messagebox.showerror("Fout in bestand", f"{f}\n\n{e}")
            sys.exit()

    full = pd.concat(frames, ignore_index=True)
    full.sort_values("Aangrijpingstijd", inplace=True)

    print("Totaal aantal records:", len(full))
    return full


# ------------------------------------------------------
# Plotten
# ------------------------------------------------------
def plot_data(df, files):
    input_dir = os.path.dirname(files[0])
    base = "combined" if len(files) > 1 else os.path.splitext(os.path.basename(files[0]))[0]
    output_html = os.path.join(input_dir, f"{base}_interactive_plot.html")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Total Q
    fig.add_trace(
        go.Scatter(
            x=df['Aangrijpingstijd'],
            y=df['Total Q (m3/s)'],
            mode='lines+markers',
            name='Total Q (m³/s)',
            marker=dict(color='blue'),
            customdata=df[['Transect', 'File name', 'Source File']].values,
            hovertemplate="<b>Transect:</b> %{customdata[0]}<br>"
                          "<b>File:</b> %{customdata[2]}<br>"
                          "<b>Tijd:</b> %{x}<br>"
                          "<b>Total Q:</b> %{y} m³/s<br>"
                          "<extra></extra>"
        ),
        secondary_y=False
    )

    # Mean Speed
    fig.add_trace(
        go.Scatter(
            x=df['Aangrijpingstijd'],
            y=df['Mean Speed (m/s)'],
            mode='lines+markers',
            name='Mean Speed (m/s)',
            marker=dict(color='red'),
            customdata=df[['Transect', 'File name', 'Source File']].values,
            hovertemplate="<b>Transect:</b> %{customdata[0]}<br>"
                          "<b>File:</b> %{customdata[2]}<br>"
                          "<b>Tijd:</b> %{x}<br>"
                          "<b>Mean Speed:</b> %{y} m/s<br>"
                          "<extra></extra>"
        ),
        secondary_y=True
    )

    # Layout
    fig.update_layout(
        title="Total Q en Mean Speed over tijd (gecombineerd)",
        xaxis_title="Aangrijpingstijd",
        hovermode="closest"
    )

    fig.update_yaxes(
        title_text="Total Q (m³/s)",
        title_font=dict(color="blue"),
        tickfont=dict(color="blue"),
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Mean Speed (m/s)",
        title_font=dict(color="red"),
        tickfont=dict(color="red"),
        secondary_y=True
    )

    fig.update_xaxes(showgrid=True, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridcolor='LightGray')

    fig.write_html(output_html, include_plotlyjs='cdn')
    add_interactive_html_saver(output_html)

    messagebox.showinfo("Succes", f"Interactieve grafiek opgeslagen:\n{output_html}")
    print("Opgeslagen:", output_html)


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    files = select_files()
    df = process_multiple(files)

    if df.empty:
        messagebox.showwarning("Geen data", "Geen bruikbare data gevonden.")
        sys.exit()

    plot_data(df, files)


if __name__ == "__main__":
    main()
