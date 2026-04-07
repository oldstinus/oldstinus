import sys
import tkinter as tk
from tkinter import filedialog, messagebox

# Controleer of alle benodigde modules geïmporteerd kunnen worden
required_modules = ['pandas', 'pyproj', 'matplotlib', 'cartopy', 'folium']
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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
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
            if not {'x', 'y'}.issubset(df.columns):
                messagebox.showerror("Fout", "Het bestand bevat niet de vereiste x en y kolommen.")
                return
            
            # Debug: Toon de eerste paar rijen van de dataframe
            print("Eerste paar rijen van de data:")
            print(df.head())
            
            # Initialiseer de transformer: van EPSG:31370 naar EPSG:4326
            transformer = Transformer.from_crs("epsg:31370", "epsg:4326", always_xy=True)
            
            # Controleer of de optie om x en y te wisselen is aangevinkt
            swap = swap_var.get()
            if swap:
                print("Waarschuwing: X en Y worden gewisseld tijdens de transformatie.")
                df['lon'], df['lat'] = transformer.transform(df['y'].values, df['x'].values)
            else:
                df['lon'], df['lat'] = transformer.transform(df['x'].values, df['y'].values)
            
            # Debug: Toon de getransformeerde coördinaten
            print("Getransformeerde coördinaten (lat, lon):")
            print(df[['lat', 'lon']].head())
            
            # Plot de statische kaart
            plot_static_map(df, file_path)
            
            # Plot de interactieve kaart
            plot_interactive_map(df, file_path)
            
        except Exception as e:
            messagebox.showerror("Fout", f"Er is een fout opgetreden: {e}")

def plot_static_map(df, file_path):
    try:
        # Maak een nieuwe figuur met Cartopy
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=4.0, central_latitude=50.5))
        
        # Voeg land, kusten en grenzen toe
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        
        # Stel het extent in op België
        ax.set_extent([2.5, 6.5, 49.5, 51.5], crs=ccrs.PlateCarree())
        
        # Plot de punten
        ax.scatter(df['lon'], df['lat'], color='red', s=10, transform=ccrs.PlateCarree(), label='Topo punten')
        
        # Voeg een legenda toe
        plt.legend(loc='upper right')
        
        # Voeg titels toe
        plt.title('Topo Data Plot')
        
        # Bepaal de directory van het invoerbestand
        input_dir = os.path.dirname(file_path)
        
        # Sla de figuur op in dezelfde directory als het invoerbestand
        output_file = os.path.join(input_dir, "topo_data_map.png")
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        messagebox.showinfo("Succes", f"Statische kaart succesvol opgeslagen als {output_file}")
        
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het plotten van de statische kaart: {e}")

def plot_interactive_map(df, file_path):
    try:
        # Bepaal het gemiddelde van de coördinaten voor de initiële locatie van de kaart
        avg_lat = df['lat'].mean()
        avg_lon = df['lon'].mean()
        
        # Debug: Toon de gemiddelde lat en lon
        print(f"Gemiddelde lat: {avg_lat}, Gemiddelde lon: {avg_lon}")
        
        # Maak een Folium kaart
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=8, tiles='OpenStreetMap')
        
        # Voeg punten toe aan de kaart
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=f"{row['type1']} {row['type2']}<br>Datum: {row['datum']} Tijd: {row['tijd']}"
            ).add_to(m)
        
        # Bepaal de directory van het invoerbestand
        input_dir = os.path.dirname(file_path)
        
        # Sla de interactieve kaart op als HTML in dezelfde directory
        output_file = os.path.join(input_dir, "topo_data_map.html")
        m.save(output_file)
        
        messagebox.showinfo("Succes", f"Interatieve kaart succesvol opgeslagen als {output_file}")
        
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het plotten van de interactieve kaart: {e}")

def main():
    # Initialiseer de GUI
    root = tk.Tk()
    root.title("Topo Data Exporteren op Kaart")
    root.geometry("400x250")
    
    # Voeg een checkbox toe om x en y te wisselen indien nodig
    global swap_var
    swap_var = tk.BooleanVar()
    swap_check = tk.Checkbutton(root, text="Wissel X en Y coördinaten", variable=swap_var)
    swap_check.pack(pady=10)
    
    # Voeg een knop toe om het bestand te selecteren
    select_button = tk.Button(root, text="Selecteer Gegevensbestand", command=select_file, height=2, width=25)
    select_button.pack(pady=20)
    
    # Start de GUI loop
    root.mainloop()

if __name__ == "__main__":
    # Debug: Toon Python-executable en versie
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Sys path:", sys.path)
    
    main()
