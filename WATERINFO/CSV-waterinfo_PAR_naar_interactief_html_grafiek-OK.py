import pandas as pd
import plotly.express as px
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

def load_ppfd_csv(path: str) -> pd.DataFrame:
    # Bepaal kolomnamen en sla commentregels (#) over
    df = pd.read_csv(
        path,
        sep=";",
        comment="#",
        names=["Timestamp", "Value", "QualityCode", "AbsValue", "AVQuality"],
        skip_blank_lines=True,
        engine="python"
    )
    # Decimale komma -> punt en naar float
    df["Value"] = (
        df["Value"].astype(str).str.replace(",", ".", regex=False)
        .replace({"": None, "nan": None})
        .astype(float)
    )
    # Parse ISO8601 (behoud tijdzone)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    # Drop lege tijdstempels/waarden
    df = df.dropna(subset=["Timestamp", "Value"]).reset_index(drop=True)
    return df

def make_plot(df: pd.DataFrame):
    fig = px.line(
        df, x="Timestamp", y="Value",
        title="Fotonenstroomdichtheid (PPFD)",
        labels={"Timestamp": "Tijd", "Value": "PPFD (µmol m⁻² s⁻¹)"}
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    return fig

def main():
    root = tk.Tk()
    root.withdraw()

    csv_path = filedialog.askopenfilename(
        title="Kies PPFD CSV",
        filetypes=[("CSV files","*.csv"), ("All files","*.*")]
    )
    if not csv_path:
        sys.exit(0)

    try:
        df = load_ppfd_csv(csv_path)
    except Exception as e:
        messagebox.showerror("Fout bij inlezen", str(e))
        sys.exit(1)

    if df.empty:
        messagebox.showwarning("Geen data", "Geen bruikbare rijen gevonden in het CSV-bestand.")
        sys.exit(0)

    fig = make_plot(df)

    html_path = filedialog.asksaveasfilename(
        title="Bewaar interactieve HTML grafiek",
        defaultextension=".html",
        filetypes=[("HTML file","*.html")]
    )
    if not html_path:
        sys.exit(0)

    try:
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    except Exception as e:
        messagebox.showerror("Fout bij opslaan", str(e))
        sys.exit(1)

    messagebox.showinfo("Klaar", f"HTML grafiek bewaard:\n{html_path}")

if __name__ == "__main__":
    main()
