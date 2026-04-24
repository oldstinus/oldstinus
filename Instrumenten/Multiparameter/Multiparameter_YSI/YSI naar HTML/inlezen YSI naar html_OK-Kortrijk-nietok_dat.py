import pandas as pd
import plotly.graph_objects as go
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from tkcalendar import DateEntry
import re  # Voor het valideren van bestandsnamen
from PIL import Image, ImageTk, __version__ as PILLOW_VERSION
import base64
from collections import defaultdict
from io import BytesIO
import logging

# Stel logging in
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pillow_version_at_least(major, minor, patch=0):
    version = tuple(map(int, PILLOW_VERSION.split('.')[:3]))
    return version >= (major, minor, patch)

def main():
    # Controleer of Pillow versie voldoet
    if not pillow_version_at_least(8, 0, 0):
        messagebox.showerror("Dependency Error", "Pillow versie 8.0.0 of hoger is vereist.")
        return

    # Hoofdvenster maken
    root = tk.Tk()
    root.title("CSV Plotter")

    # Variabelen voor GUI
    delimiter_var = tk.StringVar(value='.')
    decimal_var = tk.StringVar(value=',')
    time_column_var = tk.StringVar()
    creator_name_var = tk.StringVar()
    creation_date_var = tk.StringVar(value=datetime.today().strftime('%d/%m/%Y'))
    logo_path_var = tk.StringVar(value=r'C:\Users\claeysst\Desktop\werkfiles\Scripts & 3D & config\Logo\comilogo_waterbouwkundiglab__Is-Wetenschap.jpg')  # Standaard logo pad
    location_name_var = tk.StringVar()  # Variabele voor locatie naam
    project_number_var = tk.StringVar(value='24_059')  # Standaard projectnummer

    # Dictionary om de asselectie per kolom bij te houden
    axis_selection_vars = {}  # Key: kolomnaam, Value: StringVar met 'None', 'Primair', 'Secundair'

    # Functie om bestand te selecteren
    def select_file():
        file_path = filedialog.askopenfilename(
            title="Selecteer het CSV-bestand",
            filetypes=(("CSV-bestanden", "*.csv"), ("Alle bestanden", "*.*"))
        )
        if file_path:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, file_path)
            load_columns(file_path)

    # Functie om logo te selecteren
    def select_logo():
        logo_path = filedialog.askopenfilename(
            title="Selecteer het logo-bestand",
            filetypes=(("Afbeeldingsbestanden", "*.png;*.jpg;*.jpeg;*.gif"), ("Alle bestanden", "*.*"))
        )
        if logo_path:
            logo_entry.delete(0, tk.END)
            logo_entry.insert(0, logo_path)
            logo_path_var.set(logo_path)  # Update de StringVar

    # Functie om kolommen te laden en weer te geven
    def load_columns(file_path):
        logging.info(f"Proberen kolommen te laden uit {file_path}")
        delimiter = delimiter_var.get()
        decimal_sign = decimal_var.get()
        try:
            # Lees de eerste twee rijen om kolomnamen en eenheden te krijgen
            data = pd.read_csv(file_path, delimiter=delimiter, nrows=2, header=None)
            parameter_names = data.iloc[0].tolist()
            units = data.iloc[1].tolist()

            # Combineer parameter namen en eenheden voor duidelijkheid en uniciteit
            combined_names = []
            counts = defaultdict(int)
            for param, unit in zip(parameter_names, units):
                param_str = str(param).strip() if not pd.isna(param) else ''
                unit_str = str(unit).strip() if not pd.isna(unit) else ''
                if unit_str == '':
                    combined_name = param_str
                else:
                    combined_name = f"{param_str} ({unit_str})"

                # Controleer of de gecombineerde naam al bestaat
                counts[combined_name] += 1
                if counts[combined_name] > 1:
                    # Voeg een suffix toe om uniek te blijven
                    unique_name = f"{combined_name}_{counts[combined_name]-1}"
                else:
                    unique_name = combined_name
                combined_names.append(unique_name)

            # Maak een mapping van parameter namen naar eenheden (optioneel)
            parameter_units = dict(zip(combined_names, units))
            load_columns.parameter_units = parameter_units  # Opslaan voor later gebruik

            # Lees de rest van de data vanaf de derde rij met unieke kolomnamen
            data = pd.read_csv(file_path, delimiter=delimiter, skiprows=2, names=combined_names)

            # Vervang decimaal teken in alle kolommen die objecten zijn (strings)
            # Dit voorkomt onbedoelde vervangingen in numerieke kolommen
            numeric_columns = data.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                data[col] = data[col].str.replace(decimal_sign, '.', regex=False)

            # Sla de data op in een attribuut voor later gebruik
            load_columns.data = data

            # Vul de tijdkolom optie
            if combined_names:
                time_column_var.set(combined_names[0])  # Standaard de eerste kolom
            else:
                messagebox.showerror("Fout", "Geen kolommen gevonden in het CSV-bestand.")
                return

            time_menu['menu'].delete(0, 'end')
            for col in combined_names:
                time_menu['menu'].add_command(label=col, command=lambda c=col: update_time_column(c))

            # Verwijder eerdere widgets
            for widget in column_frame.winfo_children():
                widget.destroy()
            axis_selection_vars.clear()

            # Voeg opties toe voor elke kolom (behalve de tijdkolom)
            for col, display_name in zip(combined_names, combined_names):
                if col != time_column_var.get():
                    frame = tk.Frame(column_frame)
                    frame.pack(anchor='w', pady=2)
                    tk.Label(frame, text=display_name).pack(side=tk.LEFT)
                    axis_var = tk.StringVar(value='None')
                    axis_selection_vars[col] = axis_var
                    axis_menu = tk.OptionMenu(frame, axis_var, 'None', 'Primair', 'Secundair')
                    axis_menu.pack(side=tk.LEFT)

            # Voeg een preview van de data toe
            show_data_preview()

            # Update de start- en einddatum en tijd
            update_datetime_widgets()
            logging.info("Kolommen succesvol geladen en verwerkt.")
        except pd.errors.ParserError as pe:
            logging.error(f"Parser Fout: {pe}")
            messagebox.showerror("Parser Fout", f"Fout bij het parseren van het CSV-bestand: {pe}")
        except FileNotFoundError:
            logging.error("Bestand Niet Gevonden.")
            messagebox.showerror("Bestand Niet Gevonden", "Het geselecteerde bestand bestaat niet.")
        except Exception as e:
            logging.error(f"Algemene Fout: {e}")
            messagebox.showerror("Fout", f"Kan kolommen niet laden: {e}")

    # Functie om een data preview te tonen
    def show_data_preview():
        # Verwijder eerdere preview
        for widget in preview_frame.winfo_children():
            widget.destroy()
        try:
            preview_data = load_columns.data.head(10).to_string()
            preview_text = tk.Text(preview_frame, height=10, width=100)
            preview_text.pack()
            preview_text.insert(tk.END, preview_data)
            preview_text.config(state='disabled')
        except Exception as e:
            logging.warning(f"Kan data preview niet tonen: {e}")
            # Als er een fout is bij het tonen van de preview, negeer het en ga verder

    # Functie om de tijdkolom te updaten
    def update_time_column(col):
        time_column_var.set(col)
        update_datetime_widgets()

    # Functie om de start- en einddatum en tijd widgets bij te werken
    def update_datetime_widgets():
        try:
            data = load_columns.data.copy()
            time_column = time_column_var.get()
            if time_column not in data.columns:
                messagebox.showerror("Fout", f"Tijdkolom '{time_column}' bestaat niet in de data.")
                return
            # Converteer de tijdkolom naar datetime
            data[time_column] = pd.to_datetime(data[time_column], dayfirst=True, errors='coerce')
            # Verwijder eventuele NaT waarden
            data = data.dropna(subset=[time_column])
            if data.empty:
                messagebox.showwarning("Waarschuwing", "De tijdkolom bevat geen valide datums.")
                return
            min_datetime = data[time_column].min()
            max_datetime = data[time_column].max()
            # Stel de startdatum en tijd in
            start_date_entry.set_date(min_datetime.date())
            start_hour_spinbox.delete(0, tk.END)
            start_hour_spinbox.insert(0, f"{min_datetime.hour:02}")
            start_minute_spinbox.delete(0, tk.END)
            start_minute_spinbox.insert(0, f"{min_datetime.minute:02}")
            start_second_spinbox.delete(0, tk.END)
            start_second_spinbox.insert(0, f"{min_datetime.second:02}")
            # Stel de einddatum en tijd in
            end_date_entry.set_date(max_datetime.date())
            end_hour_spinbox.delete(0, tk.END)
            end_hour_spinbox.insert(0, f"{max_datetime.hour:02}")
            end_minute_spinbox.delete(0, tk.END)
            end_minute_spinbox.insert(0, f"{max_datetime.minute:02}")
            end_second_spinbox.delete(0, tk.END)
            end_second_spinbox.insert(0, f"{max_datetime.second:02}")
        except Exception as e:
            logging.error(f"Fout bij het instellen van start- en einddatum: {e}")
            messagebox.showerror("Fout", f"Kan start- en einddatum niet instellen: {e}")

    # Functie om de grafiek te genereren
    def generate_plot():
        file_path = file_entry.get()
        delimiter = delimiter_var.get()
        decimal_sign = decimal_var.get()
        time_column = time_column_var.get()
        creator_name = creator_name_var.get()
        creation_date = creation_date_var.get()
        logo_path = logo_entry.get()
        location_name = location_name_var.get()
        project_number = project_number_var.get()

        if not file_path:
            messagebox.showwarning("Waarschuwing", "Selecteer een CSV-bestand.")
            return

        selected_columns_primary = [col for col, var in axis_selection_vars.items() if var.get() == 'Primair']
        selected_columns_secondary = [col for col, var in axis_selection_vars.items() if var.get() == 'Secundair']
        if not selected_columns_primary and not selected_columns_secondary:
            messagebox.showwarning("Waarschuwing", "Selecteer minstens één kolom om te plotten.")
            return

        try:
            data = load_columns.data.copy()
            parameter_units = load_columns.parameter_units  # Haal de units mapping op
            # Vervang decimaal teken in alle kolommen die objecten zijn (strings)
            numeric_columns = data.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                data[col] = data[col].str.replace(decimal_sign, '.', regex=False)

            # Tijdkolom converteren
            try:
                # Aangepast datum-tijdformaat
                data[time_column] = pd.to_datetime(data[time_column], dayfirst=True, errors='coerce')
                if data[time_column].isnull().all():
                    messagebox.showerror("Fout", "Kan de tijdkolom niet converteren naar datetime-formaat. Controleer het datum-tijdformaat.")
                    return
            except Exception as e:
                messagebox.showerror("Fout", f"Er is een fout opgetreden bij het parsen van de tijdkolom: {e}")
                return

            # Bouw de start- en einddatetime op basis van de widgets
            start_date = start_date_entry.get_date()
            start_hour = int(start_hour_spinbox.get())
            start_minute = int(start_minute_spinbox.get())
            start_second = int(start_second_spinbox.get())
            start_datetime = datetime.combine(start_date, datetime.min.time()).replace(
                hour=start_hour, minute=start_minute, second=start_second
            )

            end_date = end_date_entry.get_date()
            end_hour = int(end_hour_spinbox.get())
            end_minute = int(end_minute_spinbox.get())
            end_second = int(end_second_spinbox.get())
            end_datetime = datetime.combine(end_date, datetime.min.time()).replace(
                hour=end_hour, minute=end_minute, second=end_second
            )

            # Filter data op basis van start- en einddatum
            data = data[(data[time_column] >= start_datetime) & (data[time_column] <= end_datetime)]

            # Converteer numerieke kolommen naar float
            for col in selected_columns_primary + selected_columns_secondary:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # Controleer of er data over is na filteren
            if data.empty:
                messagebox.showwarning("Waarschuwing", "Geen data beschikbaar binnen de geselecteerde periode.")
                return

            # Plotly interactieve grafiek maken
            fig = go.Figure()
            # Kleurcodering voor assen
            primary_color = 'blue'
            secondary_color = 'green'

            # Traces voor primaire y-as
            for param in selected_columns_primary:
                try:
                    y_values = data[param]
                    unit = parameter_units.get(param, '')  # Haal de eenheid op
                    # Voeg eenheid en (Primair) toe aan de trace-naam
                    param_str = str(param)
                    unit_str = str(unit) if not pd.isna(unit) else ''
                    trace_name = f'{param_str} (Primair)'
                    if unit_str.strip() != '':
                        trace_name = f'{param_str} ({unit_str}) (Primair)'
                    fig.add_trace(go.Scatter(
                        x=data[time_column],
                        y=y_values,
                        mode='lines',
                        name=trace_name,
                        yaxis='y1',
                        line=dict(),
                        hoverinfo='x+y+name'
                    ))
                except Exception as e:
                    logging.warning(f"Kan kolom '{param}' niet plotten: {e}")
                    messagebox.showwarning("Waarschuwing", f"Kan kolom '{param}' niet plotten: {e}")
                    continue

            # Traces voor secundaire y-as
            for param in selected_columns_secondary:
                try:
                    y_values = data[param]
                    unit = parameter_units.get(param, '')  # Haal de eenheid op
                    # Voeg eenheid en (Secundair) toe aan de trace-naam
                    param_str = str(param)
                    unit_str = str(unit) if not pd.isna(unit) else ''
                    trace_name = f'{param_str} (Secundair)'
                    if unit_str.strip() != '':
                        trace_name = f'{param_str} ({unit_str}) (Secundair)'
                    fig.add_trace(go.Scatter(
                        x=data[time_column],
                        y=y_values,
                        mode='lines',
                        name=trace_name,
                        yaxis='y2',
                        line=dict(),
                        hoverinfo='x+y+name'
                    ))
                except Exception as e:
                    logging.warning(f"Kan kolom '{param}' niet plotten: {e}")
                    messagebox.showwarning("Waarschuwing", f"Kan kolom '{param}' niet plotten: {e}")
                    continue

            # Bereken de minimale en maximale waarden voor de assen (niet meer nodig voor dynamische schaal)
            # x_min = start_datetime
            # x_max = end_datetime

            # Verwijder de vaste ranges en stel autorange in
            # Y-as titels instellen met eenheden indien mogelijk
            y1_title = "Waarden (Primair)"
            y1_units = set(str(parameter_units.get(param, '')) for param in selected_columns_primary)
            y1_units = {unit for unit in y1_units if unit.strip() != ''}
            if len(y1_units) == 1:
                y1_unit = y1_units.pop()
                y1_title += f" [{y1_unit}]"
            elif len(y1_units) > 1:
                y1_title += " [Verschillende eenheden]"
            # Geen eenheden toevoegen als er geen of meerdere verschillende eenheden zijn

            y2_title = "Waarden (Secundair)"
            y2_units = set(str(parameter_units.get(param, '')) for param in selected_columns_secondary)
            y2_units = {unit for unit in y2_units if unit.strip() != ''}
            if len(y2_units) == 1:
                y2_unit = y2_units.pop()
                y2_title += f" [{y2_unit}]"
            elif len(y2_units) > 1:
                y2_title += " [Verschillende eenheden]"
            # Geen eenheden toevoegen als er geen of meerdere verschillende eenheden zijn

            # Layout instellen zonder vaste ranges en met autorange
            fig.update_layout(
                xaxis_title="Tijd",
                yaxis_title=y1_title,
                yaxis=dict(
                    title=dict(
                        text=y1_title,
                        font=dict(
                            family="Arial Black, sans-serif",
                            size=14,
                            color=primary_color
                        )
                    ),
                    side="left",
                    tickfont=dict(
                        family="Arial",
                        size=12,
                        color=primary_color
                    ),
                    autorange=True
                ),
                yaxis2=dict(
                    title=dict(
                        text=y2_title,
                        font=dict(
                            family="Arial Black, sans-serif",
                            size=14,
                            color=secondary_color
                        )
                    ),
                    overlaying='y',
                    side='right',
                    tickfont=dict(
                        family="Arial",
                        size=12,
                        color=secondary_color
                    ),
                    autorange=True
                ),
                legend_title=dict(
                    text="Parameters",
                    font=dict(
                        family="Arial Black, sans-serif",
                        size=14,
                        color="Black"
                    )
                ),
                legend=dict(
                    itemsizing='constant'
                ),
                hovermode="x unified",
                hoverlabel=dict(
                    font_size=14,
                    font_family="Arial"
                )
            )

            # Voeg informatiebox toe boven de grafiek
            annotations = []
            info_text = f"Projectnummer: {project_number}<br>Naam: {creator_name}<br>Datum: {creation_date}<br>Locatie: {location_name}<br>Periode: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')} - {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}"
            annotations.append(dict(
                x=0.5,
                y=1.15,
                xref='paper',
                yref='paper',
                text=info_text,
                showarrow=False,
                font=dict(size=16, color='black'),
                xanchor='center',
                yanchor='top'
            ))
            # Voeg logo toe als het is geselecteerd
            if logo_path and os.path.exists(logo_path):
                try:
                    with Image.open(logo_path) as img:
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        encoded_image = base64.b64encode(buffered.getvalue()).decode()
                    fig.add_layout_image(
                        dict(
                            source=f'data:image/png;base64,{encoded_image}',
                            xref="paper", yref="paper",
                            x=0.99, y=1.15,  # Position boven de plot area
                            sizex=0.3, sizey=0.3,  # 3x groter
                            xanchor="right", yanchor="top"
                        )
                    )
                except Exception as e:
                    logging.warning(f"Kan logo niet toevoegen aan de grafiek: {e}")
                    messagebox.showwarning("Waarschuwing", f"Kan logo niet toevoegen aan de grafiek: {e}")

            # Voeg annotaties toe aan de layout (alleen info_text)
            fig.update_layout(annotations=annotations)

            # Interactieve periode-selector toevoegen
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )

            # Bestandsnaam opbouwen met geselecteerde periode en locatie naam
            start_datetime_str = start_datetime.strftime('%Y%m%d_%H%M%S')
            end_datetime_str = end_datetime.strftime('%Y%m%d_%H%M%S')
            # Maak locatie naam veilig voor gebruik in bestandsnaam
            safe_location_name = re.sub(r'[^\w\-_\. ]', '_', location_name.strip())[:50]  # Limiteer tot 50 tekens
            filename = f"{start_datetime_str}_{end_datetime_str}_{safe_location_name}.html"

            # De grafiek opslaan als HTML-bestand in dezelfde map als het CSV-bestand
            output_dir = os.path.dirname(file_path)
            output_file_path = os.path.join(output_dir, filename)
            fig.write_html(output_file_path)
            messagebox.showinfo("Succes", f"De interactieve grafiek is opgeslagen als {output_file_path}")
            logging.info(f"Grafiek succesvol opgeslagen als {output_file_path}")
        except Exception as e:
            logging.error(f"Fout bij het genereren van de grafiek: {e}")
            messagebox.showerror("Fout", f"Er is een fout opgetreden: {e}")

    # GUI-elementen

    # Titelkader (Title Frame)
    title_frame = tk.Frame(root, bd=2, relief=tk.RIDGE, padx=10, pady=10)
    title_frame.pack(pady=10, fill="x")

    # Projectnummer invoer
    project_frame = tk.Frame(title_frame)
    project_frame.pack(side=tk.LEFT, padx=10)
    tk.Label(project_frame, text="Projectnummer:").pack(anchor='w')
    project_entry = tk.Entry(project_frame, textvariable=project_number_var, width=20)
    project_entry.pack(anchor='w')

    # Creator en locatie info in title frame
    info_frame = tk.Frame(title_frame)
    info_frame.pack(side=tk.LEFT, padx=10)
    tk.Label(info_frame, text="Naam maker:").grid(row=0, column=0, sticky='w')
    creator_entry_title = tk.Entry(info_frame, textvariable=creator_name_var, width=20)
    creator_entry_title.grid(row=0, column=1, sticky='w')
    tk.Label(info_frame, text="Locatie:").grid(row=1, column=0, sticky='w')
    location_entry_title = tk.Entry(info_frame, textvariable=location_name_var, width=20)
    location_entry_title.grid(row=1, column=1, sticky='w')
    tk.Label(info_frame, text="Datum aanmaak:").grid(row=2, column=0, sticky='w')
    creation_date_entry_title = tk.Entry(info_frame, textvariable=creation_date_var, width=12)
    creation_date_entry_title.grid(row=2, column=1, sticky='w')

    # Bestandselectie
    file_frame = tk.Frame(root)
    file_frame.pack(pady=5)
    tk.Label(file_frame, text="CSV-bestand:").pack(side=tk.LEFT)
    file_entry = tk.Entry(file_frame, width=50)
    file_entry.pack(side=tk.LEFT)
    tk.Button(file_frame, text="Bladeren...", command=select_file).pack(side=tk.LEFT)

    # Delimiter selectie
    delimiter_frame = tk.Frame(root)
    delimiter_frame.pack(pady=5)
    tk.Label(delimiter_frame, text="Kies delimiter:").pack(side=tk.LEFT)
    delimiter_options = ['.', ',', ';', '\t', '|', ' ']
    delimiter_menu = tk.OptionMenu(delimiter_frame, delimiter_var, *delimiter_options)
    delimiter_menu.pack(side=tk.LEFT)

    # Decimaal teken selectie
    decimal_frame = tk.Frame(root)
    decimal_frame.pack(pady=5)
    tk.Label(decimal_frame, text="Kies decimaal teken:").pack(side=tk.LEFT)
    decimal_options = [',', '.']
    decimal_menu = tk.OptionMenu(decimal_frame, decimal_var, *decimal_options)
    decimal_menu.pack(side=tk.LEFT)

    # Tijdkolom selectie
    time_column_frame = tk.Frame(root)
    time_column_frame.pack(pady=5)
    tk.Label(time_column_frame, text="Selecteer tijdkolom:").pack(side=tk.LEFT)
    time_menu = tk.OptionMenu(time_column_frame, time_column_var, '')
    time_menu.pack(side=tk.LEFT)

    # Startdatum en tijd invoer
    start_datetime_frame = tk.Frame(root)
    start_datetime_frame.pack(pady=5)
    tk.Label(start_datetime_frame, text="Startdatum:").pack(side=tk.LEFT)
    start_date_entry = DateEntry(start_datetime_frame, date_pattern='dd/mm/yyyy')
    start_date_entry.pack(side=tk.LEFT)
    tk.Label(start_datetime_frame, text="Tijd (HH:MM:SS):").pack(side=tk.LEFT)
    start_hour_spinbox = tk.Spinbox(start_datetime_frame, from_=0, to=23, width=2, format="%02.0f")
    start_hour_spinbox.pack(side=tk.LEFT)
    tk.Label(start_datetime_frame, text=":").pack(side=tk.LEFT)
    start_minute_spinbox = tk.Spinbox(start_datetime_frame, from_=0, to=59, width=2, format="%02.0f")
    start_minute_spinbox.pack(side=tk.LEFT)
    tk.Label(start_datetime_frame, text=":").pack(side=tk.LEFT)
    start_second_spinbox = tk.Spinbox(start_datetime_frame, from_=0, to=59, width=2, format="%02.0f")
    start_second_spinbox.pack(side=tk.LEFT)

    # Einddatum en tijd invoer
    end_datetime_frame = tk.Frame(root)
    end_datetime_frame.pack(pady=5)
    tk.Label(end_datetime_frame, text="Einddatum:").pack(side=tk.LEFT)
    end_date_entry = DateEntry(end_datetime_frame, date_pattern='dd/mm/yyyy')
    end_date_entry.pack(side=tk.LEFT)
    tk.Label(end_datetime_frame, text="Tijd (HH:MM:SS):").pack(side=tk.LEFT)
    end_hour_spinbox = tk.Spinbox(end_datetime_frame, from_=0, to=23, width=2, format="%02.0f")
    end_hour_spinbox.pack(side=tk.LEFT)
    tk.Label(end_datetime_frame, text=":").pack(side=tk.LEFT)
    end_minute_spinbox = tk.Spinbox(end_datetime_frame, from_=0, to=59, width=2, format="%02.0f")
    end_minute_spinbox.pack(side=tk.LEFT)
    tk.Label(end_datetime_frame, text=":").pack(side=tk.LEFT)
    end_second_spinbox = tk.Spinbox(end_datetime_frame, from_=0, to=59, width=2, format="%02.0f")
    end_second_spinbox.pack(side=tk.LEFT)

    # Logo selectie
    logo_frame = tk.Frame(root)
    logo_frame.pack(pady=5)
    tk.Label(logo_frame, text="Logo-bestand:").pack(side=tk.LEFT)
    logo_entry = tk.Entry(logo_frame, textvariable=logo_path_var, width=40)
    logo_entry.pack(side=tk.LEFT)
    tk.Button(logo_frame, text="Bladeren...", command=select_logo).pack(side=tk.LEFT)

    # Kolomselectie met opties voor as
    column_frame = tk.LabelFrame(root, text="Selecteer kolommen en assen:")
    column_frame.pack(pady=5, fill="both", expand="yes")
    # Widgets worden dynamisch toegevoegd in load_columns()

    # Preview Frame voor Data
    preview_frame = tk.LabelFrame(root, text="Data Preview (Eerste 10 Rijen):")
    preview_frame.pack(pady=5, fill="both", expand="yes")

    # Plot knop
    plot_button = tk.Button(root, text="Genereer Grafiek", command=generate_plot)
    plot_button.pack(pady=10)

    # Als er een standaard logo pad is, probeer het te laden
    if logo_path_var.get() and os.path.exists(logo_path_var.get()):
        logo_entry.insert(0, logo_path_var.get())

    root.mainloop()

if __name__ == "__main__":
    main()
