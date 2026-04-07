import os

import tkinter as tk
from tkinter import filedialog, messagebox
import re


import simplekml


# Zet debug uit (False) als u klaar bent met testen
DEBUG = False

def dms_to_dd(dms_str):
    """
    Converteert een DMS-string (bijv. '51° 12' 42.47"') naar decimale graden (float).
    """
    pattern = r'(\d+)°\s+(\d+)\'\s+([\d\.]+)"'
    match = re.search(pattern, dms_str.strip())
    if not match:
        return None
    deg = float(match.group(1))
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    return deg + minutes/60 + seconds/3600

def parse_file(file_path):
    """
    Leest het TXT-bestand (tab-delimited), slaat de eerste twee regels (header + eenheden) over.
    Kolom 16 = latitude (DMS), kolom 17 = longitude (DMS).
    """
    coords = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Sla de eerste twee regels over (header + eenheden)
        header_line = f.readline()
        units_line = f.readline()

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if DEBUG:
                print("DEBUG: Ruwe regel:", repr(line))
                print(f"DEBUG: Aantal kolommen = {len(parts)}")
                for i, col in enumerate(parts):
                    print(f"  kolom {i}: '{col}'")
                print("Stop debug na deze ene regel.")
                return []

            if len(parts) < 18:
                continue

            # Hier zijn de indices aangepast op basis van de debug-output:
            lat_dms = parts[16].strip()
            lon_dms = parts[17].strip()

            lat_dd = dms_to_dd(lat_dms)
            lon_dd = dms_to_dd(lon_dms)

            if lat_dd is not None and lon_dd is not None:
                coords.append((lon_dd, lat_dd))

    return coords

def export_to_kml(coordinates, output_file):
    """
    Maakt een KML-bestand aan met een LineString die het gemeten traject weergeeft.
    """
    kml = simplekml.Kml()
    linestring = kml.newlinestring(name="M9 Metingen")
    linestring.coords = coordinates
    linestring.style.linestyle.width = 4
    linestring.style.linestyle.color = simplekml.Color.red
    kml.save(output_file)

def open_file():
    file_path = filedialog.askopenfilename(
        title="Selecteer TXT-bestand (M9 metingen)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if file_path:
        coords = parse_file(file_path)
        if coords:
            output_file = filedialog.asksaveasfilename(
                title="Opslaan als KML",
                defaultextension=".kml",
                filetypes=[("KML files", "*.kml")]
            )
            if output_file:
                export_to_kml(coords, output_file)
                messagebox.showinfo("Succes", f"KML-bestand opgeslagen als:\n{output_file}")
        else:
            messagebox.showerror("Fout", "Geen geldige DMS-coördinaten gevonden of debug-mode actief.")

# GUI
root = tk.Tk()
root.title("M9 Data -> KML Converter")

btn = tk.Button(root, text="Selecteer TXT -> Exporteer naar KML", command=open_file, padx=20, pady=10)
btn.pack(padx=20, pady=20)

root.mainloop()
