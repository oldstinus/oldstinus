import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import csv
from io import StringIO

def load_cr8_file(path: str) -> pd.DataFrame:
    """Lees een .cr8 tekstbestand met header op regel 9 en data vanaf regel 10"""
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    header_line = lines[8]   # regel 9 (0-based index)
    data_lines = lines[9:]   # data vanaf regel 10

    # detecteer delimiter (komma of puntkomma)
    sample_text = header_line + (data_lines[0] if data_lines else "")
    dialect = csv.Sniffer().sniff(sample_text)
    delimiter = dialect.delimiter

    # bouw CSV tekst
    csv_text = header_line + "".join(data_lines)

    df = pd.read_csv(StringIO(csv_text), sep=delimiter, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]

    if "Absolute Value" not in df.columns:
        raise ValueError(f"Kolommen gevonden: {df.columns.tolist()}, geen 'Absolute Value' aanwezig.")

    # omrekening naar spanning
    coef = -158.93
    df["U_in_mV"]  = 1000 * df["Absolute Value"] / coef
    df["U_abs_mV"] = df["U_in_mV"].abs()
    return df

def main():
    root = tk.Tk()
    root.withdraw()

    cr8_path = filedialog.askopenfilename(
        title="Kies CR8-bestand met Absolute Value",
        filetypes=[("CR8 of tekstbestand", "*.cr8 *.txt"), ("Alle bestanden", "*.*")]
    )
    if not cr8_path:
        sys.exit(0)

    try:
        df = load_cr8_file(cr8_path)
    except Exception as e:
        messagebox.showerror("Fout bij inlezen", str(e))
        sys.exit(1)

    if df.empty:
        messagebox.showwarning("Geen data", "Geen bruikbare rijen gevonden in het bestand.")
        sys.exit(0)

    save_path = filedialog.asksaveasfilename(
        title="Bewaar CSV met spanning in mV",
        defaultextension=".csv",
        filetypes=[("CSV-bestand", "*.csv")]
    )
    if not save_path:
        sys.exit(0)

    try:
        df.to_csv(save_path, index=False, encoding="utf-8")
    except Exception as e:
        messagebox.showerror("Fout bij opslaan", str(e))
        sys.exit(1)

    messagebox.showinfo("Klaar", f"CSV bewaard met spanning in mV:\n{save_path}")

if __name__ == "__main__":
    main()
