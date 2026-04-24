import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime, timedelta
import sys
import os  # Voor padmanipulatie

def select_file():
    """
    Open een dialoogvenster voor de gebruiker om een ADCP StreamPro bestand te selecteren.
    Retourneert het geselecteerde bestandspad.
    """
    root = tk.Tk()
    root.withdraw()  # Verberg het hoofdvenster
    file_path = filedialog.askopenfilename(
        title="Selecteer ADCP StreamPro bestand",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not file_path:
        messagebox.showerror("Geen Bestand Geselecteerd", "Selecteer alstublieft een ADCP StreamPro bestand om door te gaan.")
        sys.exit()
    return file_path

def parse_duration(duration_str):
    """
    Parse een duurstring in het formaat 'H:MM:SS' of 'MM:SS' naar een timedelta object.
    """
    parts = duration_str.strip().split(':')
    if len(parts) == 3:
        try:
            hours, minutes, seconds = map(int, parts)
        except ValueError:
            raise ValueError(f"Ongeldig duurformaat: {duration_str}")
    elif len(parts) == 2:
        try:
            hours = 0
            minutes, seconds = map(int, parts)
        except ValueError:
            raise ValueError(f"Ongeldig duurformaat: {duration_str}")
    else:
        raise ValueError(f"Ongeldig duurformaat: {duration_str}")
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def normalize_column_names(columns):
    """
    Normaliseert kolomnamen door spaties te strippen en naar kleine letters te converteren.
    """
    return [col.strip().lower() for col in columns]

def find_column(df_columns, target):
    """
    Vindt de kolomnaam in df_columns die overeenkomt met het target, ongeacht hoofdletters of extra spaties.
    Retourneert de originele kolomnaam als gevonden, anders None.
    """
    normalized_columns = normalize_column_names(df_columns)
    target_normalized = target.strip().lower()
    for original, normalized in zip(df_columns, normalized_columns):
        if normalized == target_normalized:
            return original
    return None

def process_data(file_path):
    """
    Verwerkt het ADCP StreamPro bestand en retourneert een DataFrame met berekende 'Aangrijpingstijd'.
    """
    try:
        # Gebruik UTF-8 codering
        encoding = 'utf-8'
        
        # Lees het bestand met de gekozen codering
        # Gebruik header=0 om de eerste rij als kolomnamen te gebruiken
        # Gebruik skiprows=[1] om alleen de tweede rij (eenheden) over te slaan
        df = pd.read_csv(
            file_path,
            sep='\t',  # Tab-delimited
            header=0,   # Eerste rij bevat kolomnamen
            skiprows=[1],  # Sla alleen de tweede rij (eenheden) over
            engine='python',
            on_bad_lines='warn',  # Waarschuw over slechte lijnen en sla ze over
            encoding=encoding,
            dtype=str  # Lees alle data als strings om inconsistenties te behandelen
        )
        
        # Verwijder eventuele spaties in kolomnamen en standaardiseer namen
        df.columns = [col.strip() for col in df.columns]
        
        # Debugging: Print de kolomnamen met repr om verborgen tekens te detecteren
        print("Kolomnamen na lezen:")
        print([repr(col) for col in df.columns])
        
        # Definieer de vereiste kolommen
        required_columns = [
            'Transect', 'Start Bank', '# Ens.', 'Start Time', 'Total Q', 'Delta Q',
            'Top Q', 'Meas. Q', 'Bottom Q', 'Left Q', 'Left Dist.', 'Right Q',
            'Right Dist.', 'Width', 'Total Area', 'Q/Area', 'Boat Speed',
            'Flow Speed', 'Flow Dir.', 'End Time', 'Duration', 'Start Ens.',
            'End Ens.', 'Velocity', 'Depth'
        ]
        
        # Zoek naar de vereiste kolommen, ongeacht hoofdletters of extra spaties
        column_mapping = {}
        missing_columns = []
        for col in required_columns:
            found_col = find_column(df.columns, col)
            if found_col:
                column_mapping[found_col] = col
            else:
                missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"Vereiste kolommen niet gevonden: {missing_columns}")
        
        # Hernoem de kolommen naar de vereiste namen
        df.rename(columns=column_mapping, inplace=True)
        
        # Verwijder rijen waar vereiste velden ontbreken
        df.dropna(subset=required_columns, inplace=True)
        
        # Voeg een vaste datum toe aan de tijden
        fixed_date = '03/10/2024'  # DD/MM/YYYY
        
        # Combineer 'Start Time' en 'End Time' met de vaste datum
        df['Start Datetime'] = pd.to_datetime(
            fixed_date + ' ' + df['Start Time'],
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'  # Converteer fouten naar NaT
        )
        
        df['End Datetime'] = pd.to_datetime(
            fixed_date + ' ' + df['End Time'],
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'  # Converteer fouten naar NaT
        )
        
        # Verwijder rijen met ongeldige datetime
        df = df.dropna(subset=['Start Datetime', 'End Datetime'])
        
        # Bereken 'Aangrijpingstijd' als het middenpunt tussen 'Start Datetime' en 'End Datetime'
        df['Aangrijpingstijd'] = df['Start Datetime'] + (df['End Datetime'] - df['Start Datetime']) / 2
        
        # Converteer relevante kolommen naar numeriek
        numeric_columns = [
            'Total Q', 'Delta Q', 'Top Q', 'Meas. Q', 'Bottom Q', 'Left Q',
            'Right Q', 'Width', 'Total Area', 'Q/Area', 'Boat Speed',
            'Flow Speed', 'Velocity', 'Depth'
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Verwijder rijen met NaN in de numerieke kolommen die je wilt plotten
        df = df.dropna(subset=['Total Q', 'Flow Speed'])
        
        # Sorteer de DataFrame op 'Aangrijpingstijd'
        df.sort_values('Aangrijpingstijd', inplace=True)
        
        # Voeg de input bestand naam toe als een kolom voor hover informatie
        file_name = os.path.basename(file_path)
        df['File name'] = file_name
        
        # Debugging: Inspecteer de DataFrame
        print("\nDataFrame Naadkijken:")
        print(df.head())
        print("\nDatatypes:")
        print(df.dtypes)
        
        return df
    except Exception as e:
        messagebox.showerror("Fout bij het Verwerken van het Bestand", f"Er is een fout opgetreden bij het verwerken van het bestand:\n{e}")
        sys.exit()

def plot_data(df, input_file_path):
    """
    Plots 'Total Q' en 'Flow Speed' tegen 'Aangrijpingstijd' met Plotly.
    Slaat de grafiek op als een interactieve HTML file in dezelfde directory als het input bestand.
    """
    try:
        # Bepaal de directory van het input bestand
        input_dir = os.path.dirname(input_file_path)
        
        # Bepaal de basisnaam van het input bestand zonder extensie
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        
        # Stel de output HTML bestandsnaam samen
        output_html = os.path.join(input_dir, f"{base_name}_interactive_plot.html")
        
        # Maak een figuur met een subplot grid en een secundaire y-as
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Voeg Total Q toe aan de primaire y-as
        fig.add_trace(
            go.Scatter(
                x=df['Aangrijpingstijd'],
                y=df['Total Q'],
                mode='lines+markers',
                name='Total Q (m³/s)',
                marker=dict(color='blue'),
                hovertemplate=(
                    "<b>Transect:</b> %{customdata[0]}<br>" +
                    "<b>Bestand:</b> %{customdata[1]}<br>" +
                    "<b>Tijd:</b> %{x}<br>" +
                    "<b>Total Q:</b> %{y} m³/s<br>" +
                    "<extra></extra>"
                ),
                customdata=df[['Transect', 'File name']].values
            ),
            secondary_y=False
        )
        
        # Voeg Flow Speed toe aan de secundaire y-as
        fig.add_trace(
            go.Scatter(
                x=df['Aangrijpingstijd'],
                y=df['Flow Speed'],
                mode='lines+markers',
                name='Flow Speed (m/s)',
                marker=dict(color='red'),
                hovertemplate=(
                    "<b>Transect:</b> %{customdata[0]}<br>" +
                    "<b>Bestand:</b> %{customdata[1]}<br>" +
                    "<b>Tijd:</b> %{x}<br>" +
                    "<b>Flow Speed:</b> %{y} m/s<br>" +
                    "<extra></extra>"
                ),
                customdata=df[['Transect', 'File name']].values
            ),
            secondary_y=True
        )
        
        # Update de layout van de figuur
        fig.update_layout(
            title='Total Q en Flow Speed Over Tijd',
            xaxis_title='Aangrijpingstijd (Tijd)',
            legend=dict(x=0.01, y=0.99),
            hovermode='closest'
        )
        
        # Update de y-assen titels en kleuren
        fig.update_yaxes(
            title_text="Total Q (m³/s)",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            secondary_y=False
        )
        
        fig.update_yaxes(
            title_text="Flow Speed (m/s)",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            secondary_y=True
        )
        
        # Voeg grid toe via layout
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', secondary_y=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', secondary_y=True)
        
        # Sla de figuur op als een interactieve HTML file
        fig.write_html(output_html, include_plotlyjs='cdn')
        print(f"Interatieve grafiek opgeslagen als {output_html}")
        messagebox.showinfo("Succes", f"Interatieve grafiek succesvol opgeslagen als:\n{output_html}")
    except Exception as e:
        messagebox.showerror("Fout bij het Plotten van de Data", f"Er is een fout opgetreden bij het plotten van de data:\n{e}")
        sys.exit()

def main():
    """
    Hoofdfunctie om het script uit te voeren.
    """
    file_path = select_file()
    df = process_data(file_path)
    if df.empty:
        messagebox.showwarning("Geen Data", "Geen geldige data gevonden om te plotten.")
        sys.exit()
    plot_data(df, file_path)

if __name__ == "__main__":
    main()
