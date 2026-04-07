import sys
import tkinter as tk
from tkinter import filedialog, messagebox

# Controleer of alle benodigde modules geïmporteerd kunnen worden
required_modules = ['pandas', 'pyproj', 'folium']
missing_modules = []

for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    missing = ', '.join(missing_modules)
    message = f"De volgende vereiste modules zijn niet geïnstalleerd: {missing}\n\nInstalleer ze met:\n\npip install {' '.join(missing_modules)}"
    raise ImportError(message)

import pandas as pd
from pyproj import Transformer
import folium
from folium.plugins import MarkerCluster
import os

def select_file():
    file_path = filedialog.askopenfilename(
        title="Selecteer het gegevensbestand",
        filetypes=(("Tekstbestanden", "*.txt *.csv"), ("Alle bestanden", "*.*"))
    )
    if file_path:
        try:
            # Lees het bestand met pandas
            df = pd.read_csv(file_path, sep=';', header=None, 
                             names=["type1", "type2", "x", "y", "val1", "val2", "val3", "tijd", "datum"])
            
            # Controleer of de benodigde kolommen aanwezig zijn
            required_columns = {'x', 'y', 'val1', 'type1', 'type2', 'tijd', 'datum'}
            if not required_columns.issubset(df.columns):
                messagebox.showerror("Fout", f"Het bestand bevat niet de vereiste kolommen: {', '.join(required_columns)}.")
                return
            
            # Debug: Toon de eerste paar rijen van de dataframe
            print("Eerste paar rijen van de data:")
            print(df.head())
            
            # Initialiseer de transformer: van EPSG:31370 naar EPSG:4326
            transformer = Transformer.from_crs("epsg:31370", "epsg:4326", always_xy=True)
            
            # Transformeer de coördinaten (y, x) naar (lon, lat) - x en y omgewisseld
            df['lon'], df['lat'] = transformer.transform(df['y'].values, df['x'].values)
            
            # Debug: Toon de getransformeerde coördinaten
            print("Getransformeerde coördinaten (lat, lon):")
            print(df[['lat', 'lon']].head())
            
            # Voeg een testpunt toe voor Kortrijk om de transformatie te verifiëren
            test_x = 184032.2136
            test_y = 87362.9854
            test_lon, test_lat = transformer.transform(test_y, test_x)  # y en x omgewisseld
            print(f"Test coördinaat: x={test_x}, y={test_y} -> lon={test_lon}, lat={test_lat}")
            
            # Plot de interactieve kaart
            plot_interactive_map(df, file_path)
            
        except Exception as e:
            messagebox.showerror("Fout", f"Er is een fout opgetreden: {e}")

def plot_interactive_map(df, file_path):
    try:
        # Bepaal het gemiddelde van de coördinaten voor de initiële locatie van de kaart
        avg_lat = df['lat'].mean()
        avg_lon = df['lon'].mean()
        
        # Debug: Toon de gemiddelde lat en lon
        print(f"Gemiddelde lat: {avg_lat}, Gemiddelde lon: {avg_lon}")
        
        # Maak een Folium kaart met MarkerCluster
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10, tiles='OpenStreetMap')
        marker_cluster = MarkerCluster().add_to(m)
        
        # Voeg punten toe aan de kaart met labels
        for _, row in df.iterrows():
            # Maak de popup HTML met de val1 waarde in bold en " mT.A.W."
            popup_html = f"<b>{row['val1']}</b> mT.A.W."
            
            # Maak de tooltip HTML met type1, type2, val1 in bold en datum en tijd
            tooltip_html = (
                f"Type1: {row['type1']}<br>"
                f"Type2: {row['type2']}<br>"
                f"<b>{row['val1']}</b> mT.A.W.<br>"
                f"Datum: {row['datum']}<br>"
                f"Tijd: {row['tijd']}"
            )
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,  # Verhoog de radius voor betere zichtbaarheid
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, parse_html=True),
                tooltip=folium.Tooltip(tooltip_html, sticky=True)
            ).add_to(marker_cluster)
        
        # Bepaal de directory van het invoerbestand
        input_dir = os.path.dirname(file_path)
        
        # Sla de interactieve kaart op als HTML in dezelfde directory
        output_file = os.path.join(input_dir, "topo_data_map.html")
        m.save(output_file)
        
        messagebox.showinfo("Succes", f"Interactieve kaart succesvol opgeslagen als {output_file}")
        print(f"Interactieve kaart succesvol opgeslagen als {output_file}")
        
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het plotten van de interactieve kaart: {e}")

def main():
    # Initialiseer de GUI
    root = tk.Tk()
    root.title("Topo Data Exporteren op Interactieve Kaart")
    root.geometry("400x200")
    
    # Voeg een knop toe om het bestand te selecteren
    select_button = tk.Button(root, text="Selecteer Gegevensbestand", command=select_file, height=2, width=25)
    select_button.pack(pady=50)
    
    # Start de GUI loop
    root.mainloop()

if __name__ == "__main__":
    # Debug: Toon Python-executable en versie
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Sys path:", sys.path)
    
    main()
