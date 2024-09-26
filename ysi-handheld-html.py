# # Controleer welke versie van Python wordt gebruikt
# reticulate::use_python("C:/Users/claeysst/AppData/Local/Programs/Python/Python312/python.exe", required = TRUE)
# # Navigeer naar de locatie waar je de virtual environment wilt maken
# cd C:\Users\claeysst\AppData\Local\Programs\Python\Python312
# 
# # Maak een nieuwe virtual environment aan
# python -m venv my_r_venv
# 
# # Activeer de virtual environment (Windows)
# C:\Users\claeysst\AppData\Local\Programs\Python\Python312\my_r_venv\Scripts\activate.bat
# 
# # (Voor Mac/Linux zou dit zijn: source my_r_venv/bin/activate)
Sys.setenv(RETICULATE_PYTHON = "C:/Users/claeysst/Documents/.virtualenvs/my_r_venv/Scripts/python.exe")
library(reticulate)

# Gebruik dubbele backslashes in het pad
use_virtualenv("C:\\Users\\claeysst\\AppData\\Local\\Programs\\Python\\Python312\\my_r_venv", required = TRUE)

# Controleer de Python-configuratie
py_config()
pd <- import("pandas")
library(reticulate)

# Specificeer het pad naar de virtual environment (pas het pad aan naar jouw venv locatie)
reticulate::use_virtualenv("C:/Users/claeysst/AppData/Local/Programs/Python/Python312/my_r_venv/scripts", required = TRUE)

# Controleer of de configuratie juist is
py_config()



import pandas as pd
import plotly.graph_objects as go
from tkinter import filedialog, Tk, Label, Button
from tkinter.simpledialog import askstring
from PIL import Image
from datetime import datetime

# Functie om het CSV-bestand te kiezen via een GUI
def load_file():
    Tk().withdraw()  # GUI venster verbergen
    file_path = filedialog.askopenfilename(
        title="Selecteer een CSV-bestand",
        filetypes=[("CSV bestanden", "*.csv")]
    )
    return file_path

# Functie om start- en einddatum in te voeren
def get_date_input(prompt):
    date_str = askstring("Invoer", f"Voer {prompt} in (formaat: YYYY-MM-DD HH:MM:SS):")
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

# Functie om de data in te lezen
def read_and_clean_data(file_path):
    df = pd.read_csv(file_path, delimiter='.', decimal=',', skiprows=1)
    df.columns = ['Date Time', 'Temp (C)', 'SpCond (uS)', 'Sal (ppt)', 'Depth (m)', 'pH', 'Turbid (NTU)', 'Battery']
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d/%m/%y %H:%M:%S')
    return df

# Functie om de grafiek te maken en op te slaan
def create_plot(df, start_date, end_date, logo_path):
    fig = go.Figure()

    # Voeg data toe aan de linker y-as (default y-as)
    columns_left_y = ['SpCond (uS)', 'Turbid (NTU)']
    for column in columns_left_y:
        fig.add_trace(go.Scatter(x=df['Date Time'], y=df[column], mode='lines', name=column, yaxis="y"))

    # Voeg data toe aan de rechter y-as
    columns_right_y = ['Temp (C)', 'Sal (ppt)', 'Depth (m)', 'pH', 'Battery']
    for column in columns_right_y:
        fig.add_trace(go.Scatter(x=df['Date Time'], y=df[column], mode='lines', name=column, yaxis="y2"))

    # Layout aanpassen voor dubbele y-assen en hover functionaliteit
    fig.update_layout(
        title="Data Overzicht met Dubbele Y-as en Hover Info",
        xaxis_title="Tijd",
        yaxis=dict(
            title="SpCond en Turbiditeit",
            side="left"
        ),
        yaxis2=dict(
            title="Temperatuur, Saliniteit, Diepte, pH, Batterij",
            overlaying="y",
            side="right"
        ),
        xaxis_range=[start_date, end_date],  # Alleen de zichtbare range beperken
        hovermode='x unified',  # Dit zorgt ervoor dat alle waarden op de tijdlijn worden weergegeven
        images=[dict(
            source=Image.open(logo_path),
            xref="paper", yref="paper",
            x=1, y=1,
            sizex=0.2, sizey=0.2,
            xanchor="right", yanchor="bottom"
        )]
    )

    # Opslaan als HTML
    output_file = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML bestanden", "*.html")])
    fig.write_html(output_file)
    print(f"De interactieve grafiek is opgeslagen als {output_file}")

# Hoofdprogramma voor GUI-besturing
def main():
    # Bestandsselectie via GUI
    file_path = load_file()
    if not file_path:
        print("Geen bestand geselecteerd.")
        return
    
    # Invoer van start- en einddatum
    start_date = get_date_input("startdatum")
    end_date = get_date_input("einddatum")

    # CSV-bestand inlezen
    df = read_and_clean_data(file_path)

    # Logo kiezen
    logo_path = filedialog.askopenfilename(
        title="Selecteer een logo-afbeelding",
        filetypes=[("PNG bestanden", "*.png")]
    )
    
    if not logo_path:
        print("Geen logo geselecteerd.")
        return

    # Plot maken en opslaan
    create_plot(df, start_date, end_date, logo_path)

if __name__ == "__main__":
    main()
