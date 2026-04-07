import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

# Globale variabelen voor de geselecteerde bestanden
file1_path = None
file2_path = None

def update_date_range():
    """
    Leest beide bestanden in (met toepassing van de opgegeven offset)
    en stelt de begin- en einddatum/tijd in op basis van de vroegste en laatste timestamp.
    """
    global file1_path, file2_path
    try:
        try:
            offset1_seconds = float(offset1_entry.get().strip())
        except Exception:
            offset1_seconds = 0
        try:
            offset2_seconds = float(offset2_entry.get().strip())
        except Exception:
            offset2_seconds = 0

        df1 = pd.read_csv(file1_path, delimiter=",", skiprows=[0, 2, 3], decimal=".")
        df1['TIMESTAMP'] = pd.to_datetime(df1['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
        df1['TIMESTAMP'] += pd.to_timedelta(offset1_seconds, unit='s')
        
        df2 = pd.read_csv(file2_path, delimiter=",", skiprows=[0, 2, 3], decimal=".")
        df2['TIMESTAMP'] = pd.to_datetime(df2['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
        df2['TIMESTAMP'] += pd.to_timedelta(offset2_seconds, unit='s')
        
        min_time = min(df1['TIMESTAMP'].min(), df2['TIMESTAMP'].min())
        max_time = max(df1['TIMESTAMP'].max(), df2['TIMESTAMP'].max())
        
        start_entry.delete(0, tk.END)
        start_entry.insert(0, min_time.strftime("%Y-%m-%d %H:%M:%S"))
        end_entry.delete(0, tk.END)
        end_entry.insert(0, max_time.strftime("%Y-%m-%d %H:%M:%S"))
        
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het bijwerken van de datum/tijd:\n{e}")

def select_file1():
    global file1_path
    file1_path = filedialog.askopenfilename(
        title="Selecteer Bestand 1",
        filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
    )
    if file1_path:
        file1_label.config(text=f"Bestand 1:\n{file1_path}")
    if file1_path and file2_path:
        update_date_range()

def select_file2():
    global file2_path
    file2_path = filedialog.askopenfilename(
        title="Selecteer Bestand 2",
        filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
    )
    if file2_path:
        file2_label.config(text=f"Bestand 2:\n{file2_path}")
    if file1_path and file2_path:
        update_date_range()

def get_filtered_data():
    """
    Leest beide bestanden in, past de offsets toe, filtert op het tijdsinterval
    en retourneert de gecombineerde DataFrame met een extra kolom 'Source'.
    """
    try:
        offset1 = float(offset1_entry.get().strip())
    except Exception:
        offset1 = 0
    try:
        offset2 = float(offset2_entry.get().strip())
    except Exception:
        offset2 = 0

    start_str = start_entry.get().strip()
    end_str = end_entry.get().strip()
    try:
        start_dt = pd.to_datetime(start_str, format="%Y-%m-%d %H:%M:%S")
        end_dt = pd.to_datetime(end_str, format="%Y-%m-%d %H:%M:%S")
    except Exception:
        messagebox.showerror("Fout", "Ongeldig datum/tijd formaat.\nGebruik: jjjj-mm-dd uu:mm:ss")
        return None

    df_list = []
    try:
        df1 = pd.read_csv(file1_path, delimiter=",", skiprows=[0,2,3], decimal=".")
        df1['TIMESTAMP'] = pd.to_datetime(df1['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
        df1['TIMESTAMP'] += pd.to_timedelta(offset1, unit='s')
        df1 = df1[(df1['TIMESTAMP'] >= start_dt) & (df1['TIMESTAMP'] <= end_dt)]
        df1['Source'] = "Bestand 1"
        df_list.append(df1)
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het inlezen van Bestand 1:\n{file1_path}\n\n{e}")
        return None

    try:
        df2 = pd.read_csv(file2_path, delimiter=",", skiprows=[0,2,3], decimal=".")
        df2['TIMESTAMP'] = pd.to_datetime(df2['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
        df2['TIMESTAMP'] += pd.to_timedelta(offset2, unit='s')
        df2 = df2[(df2['TIMESTAMP'] >= start_dt) & (df2['TIMESTAMP'] <= end_dt)]
        df2['Source'] = "Bestand 2"
        df_list.append(df2)
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het inlezen van Bestand 2:\n{file2_path}\n\n{e}")
        return None

    if not df_list:
        messagebox.showinfo("Info", "Geen data gevonden in het opgegeven tijdsinterval.")
        return None

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def plot_data():
    if not file1_path or not file2_path:
        messagebox.showerror("Fout", "Selecteer beide bestanden!")
        return

    df = get_filtered_data()
    if df is None or df.empty:
        return

    plt.figure(figsize=(10, 6))
    for source, group in df.groupby("Source"):
        marker, color = ('o', 'blue') if source == "Bestand 1" else ('s', 'red')
        plt.plot(group['TIMESTAMP'], group['Value1'], marker=marker, linestyle='-', color=color, label=source)

    plt.xlabel("Tijd (uu:mm:ss)")
    plt.ylabel("Value1 (Afstand tot water)")
    plt.title("Radar: Afstand tot water in de tijd")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def export_data():
    if not file1_path or not file2_path:
        messagebox.showerror("Fout", "Selecteer beide bestanden!")
        return

    df = get_filtered_data()
    if df is None or df.empty:
        return

    export_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV bestanden", "*.csv"), ("Alle bestanden", "*.*")]
    )
    if not export_path:
        return

    try:
        df.to_csv(export_path, index=False)
        messagebox.showinfo("Succes", f"Data succesvol geëxporteerd naar:\n{export_path}")
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het exporteren van data:\n{e}")

def compare_series():
    """
    Vergelijkt beide reeksen in een XY-grafiek door de data van Bestand 2 per seconde te verschuiven (van -4000 tot +4000 s)
    en voor elke verschuiving de lineaire R² te berekenen. De optimale verschuiving wordt gekozen en
    in de grafiek wordt de trendlijn met vergelijking getoond. Tevens geeft de grafiektitel de namen van de bestanden weer,
    en wordt er aangegeven dat datafile Bestand 2 aangepast moet worden met de optimale tijdverschuiving.
    """
    if not file1_path or not file2_path:
        messagebox.showerror("Fout", "Selecteer beide bestanden!")
        return

    try:
        offset1 = float(offset1_entry.get().strip())
    except Exception:
        offset1 = 0
    try:
        offset2 = float(offset2_entry.get().strip())
    except Exception:
        offset2 = 0
    start_str = start_entry.get().strip()
    end_str = end_entry.get().strip()
    try:
        start_dt = pd.to_datetime(start_str, format="%Y-%m-%d %H:%M:%S")
        end_dt = pd.to_datetime(end_str, format="%Y-%m-%d %H:%M:%S")
    except Exception:
        messagebox.showerror("Fout", "Ongeldig datum/tijd formaat.")
        return

    # Inlezen van Bestand 1 (referentiereeks)
    try:
        df1 = pd.read_csv(file1_path, delimiter=",", skiprows=[0,2,3], decimal=".")
        df1['TIMESTAMP'] = pd.to_datetime(df1['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
        df1['TIMESTAMP'] += pd.to_timedelta(offset1, unit='s')
        df1 = df1[(df1['TIMESTAMP'] >= start_dt) & (df1['TIMESTAMP'] <= end_dt)]
        df1 = df1.sort_values('TIMESTAMP')
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij inlezen Bestand 1:\n{e}")
        return

    # Inlezen van Bestand 2 (te verschuiven reeks)
    try:
        df2 = pd.read_csv(file2_path, delimiter=",", skiprows=[0,2,3], decimal=".")
        df2['TIMESTAMP'] = pd.to_datetime(df2['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
        df2['TIMESTAMP'] += pd.to_timedelta(offset2, unit='s')
        df2 = df2[(df2['TIMESTAMP'] >= start_dt) & (df2['TIMESTAMP'] <= end_dt)]
        df2 = df2.sort_values('TIMESTAMP')
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij inlezen Bestand 2:\n{e}")
        return

    best_r2 = -np.inf
    best_shift = None
    best_merged = None

    # Zoek naar optimale verschuiving van Bestand 2 (van -4000 tot +4000 seconden)
    for shift in range(-4000, 4001):
        df2_shifted = df2.copy()
        df2_shifted['TIMESTAMP'] += pd.to_timedelta(shift, unit='s')
        merged = pd.merge_asof(df1, df2_shifted, on='TIMESTAMP', direction='nearest', tolerance=pd.Timedelta(seconds=2), suffixes=('_1', '_2'))
        merged = merged.dropna(subset=['Value1_1', 'Value1_2'])
        if merged.empty:
            continue
        x = merged['Value1_1'].values
        y = merged['Value1_2'].values
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        if r2 > best_r2:
            best_r2 = r2
            best_shift = shift
            best_merged = merged.copy()

    if best_shift is None:
        messagebox.showinfo("Info", "Geen overeenkomende data gevonden met de gegeven tolerantie.")
        return

    # Bereken regressie op de best gematchte data
    x = best_merged['Value1_1'].values
    y = best_merged['Value1_2'].values
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept

    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)
    title = f"Vergelijking: {file1_name} vs. {file2_name}\nOptimale verschuiving: {best_shift} s, R² = {best_r2:.3f}"

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='purple', label="Gematchte data")
    plt.plot(x_line, y_line, color='green', linestyle='--', label=f"Trendlijn: y = {slope:.3f}x + {intercept:.3f}")
    plt.xlabel("Value1 (Bestand 1)")
    plt.ylabel("Value1 (Bestand 2, verschoven)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    messagebox.showinfo("Optimale verschuiving",
                        f"Datafile '{file2_name}' moet worden aangepast met een tijdverschuiving van {best_shift} seconden voor de beste overeenkomst (R² = {best_r2:.3f}).")

# Hoofd-GUI opzetten
root = tk.Tk()
root.title("Radar Data Plotter")
root.geometry("600x600")

# Frame voor bestandsselectie en offset-invoer
file_frame = tk.Frame(root)
file_frame.pack(padx=10, pady=10, fill='x')

file1_button = tk.Button(file_frame, text="Selecteer Bestand 1", command=select_file1)
file1_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
file1_label = tk.Label(file_frame, text="Bestand 1:\nNog niet geselecteerd", wraplength=400, justify="left")
file1_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

offset1_label = tk.Label(file_frame, text="Tijdverschuiving Bestand 1 (seconden):")
offset1_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
offset1_entry = tk.Entry(file_frame, width=10)
offset1_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
offset1_entry.insert(0, "0")

file2_button = tk.Button(file_frame, text="Selecteer Bestand 2", command=select_file2)
file2_button.grid(row=2, column=0, padx=5, pady=5, sticky="w")
file2_label = tk.Label(file_frame, text="Bestand 2:\nNog niet geselecteerd", wraplength=400, justify="left")
file2_label.grid(row=2, column=1, padx=5, pady=5, sticky="w")

offset2_label = tk.Label(file_frame, text="Tijdverschuiving Bestand 2 (seconden):")
offset2_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
offset2_entry = tk.Entry(file_frame, width=10)
offset2_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
offset2_entry.insert(0, "0")

# Frame voor datum-/tijdinvoer, plot-, export- en vergelijkknoppen
input_frame = tk.Frame(root)
input_frame.pack(padx=10, pady=10, fill='x')

start_label = tk.Label(input_frame, text="Begin Datum/Tijd (jjjj-mm-dd uu:mm:ss):")
start_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
start_entry = tk.Entry(input_frame, width=25)
start_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
start_entry.insert(0, "2025-02-13 10:57:30")  # Wordt overschreven bij update_date_range()

end_label = tk.Label(input_frame, text="Eind Datum/Tijd (jjjj-mm-dd uu:mm:ss):")
end_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
end_entry = tk.Entry(input_frame, width=25)
end_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
end_entry.insert(0, "2025-02-13 10:58:15")  # Wordt overschreven bij update_date_range()

plot_button = tk.Button(input_frame, text="Plot Data", command=plot_data)
plot_button.grid(row=2, column=0, columnspan=2, pady=10)

export_button = tk.Button(input_frame, text="Exporteer CSV", command=export_data)
export_button.grid(row=3, column=0, columnspan=2, pady=10)

compare_button = tk.Button(input_frame, text="Vergelijk Reeksen (XY)", command=compare_series)
compare_button.grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
