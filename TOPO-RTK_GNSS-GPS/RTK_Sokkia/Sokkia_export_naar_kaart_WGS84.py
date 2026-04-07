import xml.etree.ElementTree as ET
from tkinter import Tk, filedialog, Label, Button, Toplevel, messagebox, ttk, Scale, HORIZONTAL
from pyproj import Transformer
import pandas as pd
import numpy as np
import os
import folium
import webbrowser
from datetime import datetime

# --- 1. Globale Variabelen en Projectie ---
# Transformer voor LB72 (EPSG:31370) naar WGS84 (EPSG:4326)
try:
    transformer = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
except Exception:
    messagebox.showerror("Pyproj Fout", "Fout bij het initialiseren van pyproj. Geen kaartvisualisatie mogelijk.")
    transformer = None

# Globale DataFrame om de data te bewaren en te manipuleren
global_df = pd.DataFrame()
# Globale variabele voor de Treeview widget (nodig voor selectie)
tree_widget = None

# --- 2. MAXML Lezer Functie (met GPS-Tijd) ---
def lees_maxml(file_path):
    """
    Leest het MAXML-bestand, extraheert de 'auto_topo' punten, GPS-tijd en coördinaten.
    """
    punten_data = []
    namespaces = {'tps': 'tps'}
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        for dp in root.findall('.//tps:DesignPoint', namespaces):
            station = dp.find('tps:Station', namespaces)
            
            if station is not None and station.find('tps:PointFlags', namespaces) is not None:
                flags = station.find('tps:PointFlags', namespaces).text
                
                if 'auto_topo' in flags:
                    # Haal de data op
                    name_elem = station.find('tps:Name', namespaces)
                    neh_elem = station.find('tps:Position/tps:NEH', namespaces)
                    code_elem = dp.find('tps:Code/tps:CodeDescription', namespaces)
                    gps_time_elem = station.find('tps:GPSTime', namespaces) # GPS Tijd toevoegen!
                    
                    if name_elem is not None and neh_elem is not None:
                        Noord = float(neh_elem.find('tps:North', namespaces).text)
                        Oost = float(neh_elem.find('tps:East', namespaces).text)
                        Hoogte = float(neh_elem.find('tps:Height', namespaces).text)
                        
                        gps_time_str = gps_time_elem.text if gps_time_elem is not None else "N/A"
                        
                        # Converteer LB72 (Oost, Noord) naar WGS84 (Longitude, Latitude)
                        if transformer:
                            lon, lat = transformer.transform(Oost, Noord)
                        else:
                            lon, lat = np.nan, np.nan
                        
                        punten_data.append({
                            'ID': dp.get('id'),
                            'Naam': name_elem.text,
                            'Code': code_elem.get('idRef') if code_elem is not None else "Onbekend",
                            'GPSTijd': gps_time_str, # GPS Tijd toegevoegd
                            'Oost_LB72': Oost,
                            'Noord_LB72': Noord,
                            'Hoogte_m': Hoogte,
                            'Latitude': lat,
                            'Longitude': lon
                        })
        
        return pd.DataFrame(punten_data)
        
    except FileNotFoundError:
        messagebox.showerror("Fout", "Bestand niet gevonden.")
    except ET.ParseError:
        messagebox.showerror("Fout", "Fout bij het parsen van het XML-bestand. Is het een geldig MAXML bestand?")
    except Exception as e:
        messagebox.showerror("Fout", f"Een onverwachte fout is opgetreden: {e}")
        
    return pd.DataFrame()

# --- 3. Kaart Visualisatie Functie (Folium) ---
def generate_folium_map(df, selected_ids=None, marker_radius=5):
    """
    Genereert en opent een interactieve Folium kaart.
    Geselecteerde punten worden rood gemarkeerd, niet-geselecteerde blauw.
    """
    if df.empty or transformer is None:
        messagebox.showinfo("Informatie", "Geen data om te visualiseren of conversie is mislukt.")
        return

    # Filter NaN waarden uit voor correcte centrering
    df_valid = df.dropna(subset=['Latitude', 'Longitude'])
    if df_valid.empty:
        messagebox.showinfo("Informatie", "Geen geldige WGS84 coördinaten gevonden voor visualisatie.")
        return

    # Bereken het midden van de punten voor de initiële kaart focus
    gemiddelde_lat = df_valid['Latitude'].mean()
    gemiddelde_lon = df_valid['Longitude'].mean()
    
    # Maak de Folium kaart aan
    m = folium.Map(location=[gemiddelde_lat, gemiddelde_lon], zoom_start=15, 
                   tiles='OpenStreetMap')

    # Voeg een Marker toe voor elk punt
    for index, rij in df_valid.iterrows():
        is_selected = rij['ID'] in selected_ids if selected_ids else False
        
        # Bepaal kleur en straal
        color = '#FF0000' if is_selected else '#0000FF' # Rood voor geselecteerd, Blauw voor de rest
        radius = marker_radius * 1.5 if is_selected else marker_radius
        
        # HTML Pop-up inhoud
        popup_html = f"""
        <b>Naam:</b> {rij['Naam']}<br>
        <b>Code:</b> {rij['Code']}<br>
        <b>GPS Tijd:</b> {rij['GPSTijd']}<br>
        <b>LB72 Oost:</b> {rij['Oost_LB72']:.3f} m<br>
        <b>LB72 Noord:</b> {rij['Noord_LB72']:.3f} m<br>
        <b>Hoogte:</b> {rij['Hoogte_m']:.3f} m
        """
        
        folium.CircleMarker(
            location=[rij['Latitude'], rij['Longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    # Sla de kaart op als een tijdelijk HTML-bestand
    kaart_pad = "autotopo_visualisatie_kaart.html"
    m.save(kaart_pad)
    
    # Open de kaart in de standaard webbrowser
    webbrowser.open('file://' + os.path.realpath(kaart_pad))
    
    # Geef feedback
    if selected_ids:
        messagebox.showinfo("Kaart Update", f"Kaart geopend. {len(selected_ids)} punt(en) zijn rood gemarkeerd.")
    else:
        messagebox.showinfo("Kaart Update", "Kaart geopend. Alle punten zijn blauw gemarkeerd.")

# --- 4. Manipulatie en Export Functies ---
def get_selected_ids(tree):
    geselecteerde_items = tree.selection()
    return [tree.item(item, 'values')[0] for item in geselecteerde_items]

def delete_selected_points():
    global global_df, tree_widget
    
    if tree_widget is None: return
    geselecteerde_ids = get_selected_ids(tree_widget)
        
    if not geselecteerde_ids:
        messagebox.showinfo("Informatie", "Selecteer eerst één of meerdere rijen in de tabel.")
        return
        
    aantal_verwijderd = len(geselecteerde_ids)
    global_df = global_df[~global_df['ID'].isin(geselecteerde_ids)]
    
    for item in tree_widget.selection():
        tree_widget.delete(item)
        
    messagebox.showinfo("Klaar", f"{aantal_verwijderde} punt(en) succesvol verwijderd.")

def export_data_gui():
    """Opent de export dialoog om de overgebleven data op te slaan."""
    if global_df.empty:
        messagebox.showinfo("Informatie", "Geen data om te exporteren.")
        return
        
    # ASCII Export
    bestand_naam_ascii = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("ASCII Tabel", "*.txt"), ("Alle bestanden", "*.*")],
        title="Sla overgebleven data op als ASCII Tabel"
    )
    if bestand_naam_ascii:
        # GPS Tijd toegevoegd aan de exportkolommen
        kolommen_ascii = ['Naam', 'Code', 'GPSTijd', 'Oost_LB72', 'Noord_LB72', 'Hoogte_m']
        global_df[kolommen_ascii].to_csv(bestand_naam_ascii, sep='\t', index=False, float_format='%.5f')
        messagebox.showinfo("Klaar", f"Data geëxporteerd naar ASCII: {os.path.basename(bestand_naam_ascii)}")

    # MAXML Export (Verbeterde, maar nog steeds vereenvoudigde structuur)
    bestand_naam_maxml = filedialog.asksaveasfilename(
        defaultextension=".mxl",
        filetypes=[("MAGNET Field MAXML", "*.mxl")],
        title="Sla overgebleven data op als MAXML (Vereenvoudigd)"
    )
    
    if bestand_naam_maxml:
        try:
            # Gebruik het oorspronkelijke bestand om de structuur te behouden (job info, etc.)
            original_tree = ET.parse("ASPER MOBIEL1.mxl") 
            root = original_tree.getroot()
            namespaces = {'tps': 'tps'}

            # 1. Haal de bestaande DesignPoints op
            design_points_container = root.find('.//tps:DesignPoints', namespaces)
            if design_points_container is None:
                # Als de container niet bestaat, voeg deze toe (zeer zeldzaam in jobfiles)
                project_info = root.find('.//tps:ProjectInfo', namespaces)
                if project_info is not None:
                     design_points_container = ET.SubElement(project_info, '{tps}DesignPoints')
                else:
                    # Noodoplossing als zelfs ProjectInfo ontbreekt
                    design_points_container = ET.Element('{tps}DesignPoints')
                    root.append(design_points_container)


            # 2. Verwijder alle DesignPoints uit de container
            for dp in design_points_container.findall('.//tps:DesignPoint', namespaces):
                design_points_container.remove(dp)

            # 3. Voeg ALLEEN de overgebleven punten toe
            # Dit vereist het reconstrueren van de XML-structuur voor elk punt.
            # We gebruiken nu de originele structuur (Root, ProjectInfo, DesignPoints) en injecteren de nieuwe punten.
            for index, row in global_df.iterrows():
                # Reconstructie van DesignPoint
                dp = ET.Element('{tps}DesignPoint', {'id': row['ID'], 'csIdRef': 'COS1'})
                
                # Code
                code = ET.SubElement(dp, '{tps}Code')
                ET.SubElement(code, '{tps}CodeDescription', {'idRef': row['Code']})
                
                # Station
                station = ET.SubElement(dp, '{tps}Station', {'id': f"STA{row['ID'][3:]}" if len(row['ID']) > 3 else 'STA000'})
                ET.SubElement(station, '{tps:Name}', namespaces).text = row['Naam']
                
                # Positie
                position = ET.SubElement(station, '{tps}Position')
                neh = ET.SubElement(position, '{tps}NEH')
                ET.SubElement(neh, '{tps}North').text = str(row['Noord_LB72'])
                ET.SubElement(neh, '{tps}East').text = str(row['Oost_LB72'])
                ET.SubElement(neh, '{tps}Height').text = str(row['Hoogte_m'])

                # GPS Tijd toevoegen
                if row['GPSTijd'] != "N/A":
                    ET.SubElement(station, '{tps}GPSTime').text = row['GPSTijd']
                
                # Vlag
                ET.SubElement(station, '{tps}PointFlags').text = 'auto_topo'
                
                design_points_container.append(dp) # Voeg het DesignPoint toe aan de container

            # Schrijf de aangepaste tree naar het bestand
            ET.register_namespace('tps', 'tps')
            original_tree.write(bestand_naam_maxml, encoding='UTF-8', xml_declaration=True)

            messagebox.showinfo("Klaar", f"Data geëxporteerd naar MAXML (Verbeterde structuur): {os.path.basename(bestand_naam_maxml)}")

        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij MAXML export. Controleer de originele file 'ASPER MOBIEL1.mxl'. Fout: {e}")

# --- 5. GUI Functie (Tkinter) ---
def start_gui():
    """
    Initialiseert de Tkinter GUI voor bestandsselectie en manipulatie.
    """
    global global_df, tree_widget
    root = Tk()
    root.title("MAGNET Field Auto-Topo Data Tool")

    # --- Functie om de data in de Treeview te laden ---
    def load_data_to_tree(df):
        global global_df
        global_df = df
        
        for item in tree.get_children():
            tree.delete(item)
            
        if not df.empty:
            for index, row in df.iterrows():
                # GPS Tijd toegevoegd aan de Treeview
                tree.insert('', 'end', values=(row['ID'], row['Naam'], row['Code'], row['GPSTijd'],
                                               f"{row['Oost_LB72']:.3f}", 
                                               f"{row['Noord_LB72']:.3f}", 
                                               f"{row['Hoogte_m']:.3f}"))
            status_label.config(text=f"Totaal {len(df)} 'auto_topo' punten geladen.")
        else:
            status_label.config(text="Geen 'auto_topo' punten gevonden of geladen.")
            
    # --- Handler voor Bestandsselectie ---
    def selecteer_bestand(file_path=None):
        nonlocal initial_file_path
        
        if file_path is None:
            file_path = filedialog.askopenfilename(
                defaultextension=".mxl",
                filetypes=[("MAGNET Field MAXML", "*.mxl"), ("Alle bestanden", "*.*")],
                title="Selecteer het MAXML-bestand"
            )
        
        if file_path:
            df = lees_maxml(file_path)
            load_data_to_tree(df)
            initial_file_path = file_path 
            
            # Open direct de map/controle venster na laden (optioneel)
            open_visualisatie_control_venster(root)
            
    # --- Visualisatie Controle Venster ---
    def open_visualisatie_control_venster(parent):
        if global_df.empty:
            messagebox.showinfo("Informatie", "Er zijn geen data geladen om te visualiseren.")
            return

        control_venster = Toplevel(parent)
        control_venster.title("Kaart Visualisatie Controle")
        initial_radius = 5
        
        def update_map(radius=initial_radius):
            selected_ids = get_selected_ids(tree_widget)
            generate_folium_map(global_df, selected_ids=selected_ids, marker_radius=radius)

        size_slider = Scale(control_venster, label="Pas Grootte Punten aan (Radius/ptn):", 
                            from_=1, to=15, resolution=1, 
                            orient=HORIZONTAL, command=update_map)
        size_slider.set(initial_radius)
        size_slider.pack(pady=10, padx=10, fill='x')
        
        # Knop om de kaart te regenereren
        Button(control_venster, text="Toon/Markeer Punten op Kaart", 
               command=lambda: update_map(radius=size_slider.get()),
               bg='yellow').pack(pady=5, padx=10, fill='x')
               
        # Automatisch de kaart openen bij het openen van het controlevenster
        update_map(radius=size_slider.get())
        
        # Event voor veranderingen in de hoofd-treeview (selectie)
        def on_treeview_select(event):
            # Update de kaart bij elke selectie in de tabel
            update_map(radius=size_slider.get())

        # Bind de selectie-update aan het Treeview widget (belangrijk voor interactie!)
        tree_widget.bind('<<TreeviewSelect>>', on_treeview_select)


    # --- GUI Elementen van het Hoofdvenster ---
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=10, padx=10, fill='x')
    
    Button(button_frame, text="Selecteer MAXML-bestand", command=lambda: selecteer_bestand(None),
           bg='lightblue', fg='black').pack(side='left', padx=5)
    
    initial_file_path = "ASPER MOBIEL1.mxl"
    
    # FIX: Gebruik Geuploade File knop
    Button(button_frame, text="Gebruik geüploade file", 
            command=lambda: selecteer_bestand(initial_file_path),
            bg='lightgreen', fg='black').pack(side='left', padx=5)

    Button(button_frame, text="Open Kaart Controle (ptn)", command=lambda: open_visualisatie_control_venster(root),
           bg='yellow', fg='black').pack(side='left', padx=5)

    Button(button_frame, text="Delete Geselecteerde Punten", 
           command=delete_selected_points,
           bg='red', fg='white').pack(side='left', padx=5)
           
    Button(button_frame, text="Export Data", 
           command=export_data_gui,
           bg='orange', fg='black').pack(side='left', padx=5)
    
    status_label = Label(root, text="Laad een MAXML bestand om te beginnen.", fg='gray')
    status_label.pack(pady=5)
    
    # --- Data Tabel (Treeview) ---
    tree_frame = ttk.Frame(root)
    tree_frame.pack(pady=10, padx=10, fill='both', expand=True)

    scrollbar = ttk.Scrollbar(tree_frame)
    scrollbar.pack(side='right', fill='y')

    # GPS Tijd toegevoegd aan de kolommen
    kolommen = ('ID', 'Naam', 'Code', 'GPS Tijd', 'Oost (LB72)', 'Noord (LB72)', 'Hoogte (m)')
    tree = ttk.Treeview(tree_frame, columns=kolommen, show='headings', selectmode='extended', yscrollcommand=scrollbar.set)
    scrollbar.config(command=tree.yview)
    tree_widget = tree # Stel de globale variabele in

    # Kolominstellingen
    tree.heading('ID', text='ID')
    tree.heading('Naam', text='Naam')
    tree.heading('Code', text='Code')
    tree.heading('GPS Tijd', text='GPS Tijd') # Nieuwe kolom
    tree.heading('Oost (LB72)', text='Oost (LB72)')
    tree.heading('Noord (LB72)', text='Noord (LB72)')
    tree.heading('Hoogte (m)', text='Hoogte (m)')

    # Kolombreedtes
    tree.column('ID', width=0, stretch=False) 
    tree.column('Naam', width=80)
    tree.column('Code', width=80)
    tree.column('GPS Tijd', width=150) # Nieuwe kolom breedte
    tree.column('Oost (LB72)', width=100, anchor='e')
    tree.column('Noord (LB72)', width=100, anchor='e')
    tree.column('Hoogte (m)', width=80, anchor='e')

    tree.pack(fill='both', expand=True)
    
    # Laad de geüploade file direct indien deze bestaat (fix)
    if os.path.exists(initial_file_path):
        selecteer_bestand(initial_file_path)

    root.mainloop()

if __name__ == '__main__':
    start_gui()