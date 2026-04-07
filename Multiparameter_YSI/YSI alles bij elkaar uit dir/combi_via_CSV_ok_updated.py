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
    if not pillow_version_at_least(8, 0, 0):
        messagebox.showerror("Dependency Error", "Pillow versie 8.0.0 of hoger is vereist.")
        return

    root = tk.Tk()
    root.title("CSV Plotter")

    # GUI-variabelen
    delimiter_var = tk.StringVar(value='.')
    decimal_var = tk.StringVar(value=',')
    time_column_var = tk.StringVar()
    creator_name_var = tk.StringVar()
    creation_date_var = tk.StringVar(value=datetime.today().strftime('%d/%m/%Y'))
    logo_path_var = tk.StringVar()
    location_name_var = tk.StringVar()  # Locatie naam
    project_number_var = tk.StringVar()  # Projectnummer

    # Dictionary voor asselectie (per kolom)
    axis_selection_vars = {}  # Key: kolomnaam, Value: StringVar ('None', 'Primair', 'Secundair')

    def select_file():
        file_paths = filedialog.askopenfilenames(
            title="Selecteer CSV-bestanden",
            filetypes=(("CSV-bestanden", "*.csv"), ("Alle bestanden", "*.*"))
        )
        if file_paths:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, '; '.join(file_paths))
            load_columns(list(file_paths))

    def select_logo():
        logo_path = filedialog.askopenfilename(
            title="Selecteer het logo-bestand",
            filetypes=(("Afbeeldingsbestanden", "*.png;*.jpg;*.jpeg;*.gif"), ("Alle bestanden", "*.*"))
        )
        if logo_path:
            logo_entry.delete(0, tk.END)
            logo_entry.insert(0, logo_path)

    # Lees de CSV-bestanden in en maak unieke kolomnamen
    def load_columns(file_paths):
        logging.info(f"Proberen kolommen te laden uit: {file_paths}")
        delimiter = delimiter_var.get()
        decimal_sign = decimal_var.get()
        try:
            # Bepaal kolomnamen en eenheden uit het eerste bestand (eerste twee rijen)
            data_temp = pd.read_csv(file_paths[0], delimiter=delimiter, nrows=2, header=None)
            parameter_names = data_temp.iloc[0].tolist()
            units = data_temp.iloc[1].tolist()

            # Maak unieke, gecombineerde kolomnamen
            combined_names = []
            counts = defaultdict(int)
            for param, unit in zip(parameter_names, units):
                param_str = str(param).strip() if not pd.isna(param) else ''
                unit_str = str(unit).strip() if not pd.isna(unit) else ''
                combined_name = param_str if unit_str == '' else f"{param_str} ({unit_str})"
                counts[combined_name] += 1
                if counts[combined_name] > 1:
                    unique_name = f"{combined_name}_{counts[combined_name]-1}"
                else:
                    unique_name = combined_name
                combined_names.append(unique_name)

            # Sla de mapping parameter -> unit op
            parameter_units = dict(zip(combined_names, units))
            load_columns.parameter_units = parameter_units

            # Lees elk bestand vanaf de derde rij en voeg een kolom 'SourceFile' toe
            data_list = []
            for file_path in file_paths:
                df = pd.read_csv(file_path, delimiter=delimiter, skiprows=2, names=combined_names)
                # Vervang decimaal teken in tekstkolommen
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].str.replace(decimal_sign, '.', regex=False)
                # Voeg bronbestandsnaam toe
                df['SourceFile'] = os.path.basename(file_path)
                data_list.append((file_path, df))
            load_columns.data = data_list

            # Stel de tijdkolom standaard in (eerste kolom)
            if combined_names:
                time_column_var.set(combined_names[0])
            else:
                messagebox.showerror("Fout", "Geen kolommen gevonden in de CSV-bestanden.")
                return

            time_menu['menu'].delete(0, 'end')
            for col in combined_names:
                time_menu['menu'].add_command(label=col, command=lambda c=col: update_time_column(c))

            # Maak de kolomselectie voor asinstellingen
            for widget in column_frame.winfo_children():
                widget.destroy()
            axis_selection_vars.clear()
            for col in combined_names:
                if col != time_column_var.get():
                    frame = tk.Frame(column_frame)
                    frame.pack(anchor='w', pady=2)
                    tk.Label(frame, text=col).pack(side=tk.LEFT)
                    axis_var = tk.StringVar(value='Primair')
                    axis_selection_vars[col] = axis_var
                    axis_menu = tk.OptionMenu(frame, axis_var, 'None', 'Primair', 'Secundair')
                    axis_menu.pack(side=tk.LEFT)

            show_data_preview()
            update_datetime_widgets()
            logging.info("Kolommen succesvol geladen.")
        except Exception as e:
            logging.error(f"Fout bij het laden van kolommen: {e}")
            messagebox.showerror("Fout", f"Kan kolommen niet laden: {e}")

    def show_data_preview():
        for widget in preview_frame.winfo_children():
            widget.destroy()
        try:
            if hasattr(load_columns, 'data') and load_columns.data:
                first_file, df = load_columns.data[0]
                preview_data = f"Preview van: {first_file}\n" + df.head(10).to_string()
            else:
                preview_data = "Geen data geladen."
            preview_text = tk.Text(preview_frame, height=10, width=100)
            preview_text.pack()
            preview_text.insert(tk.END, preview_data)
            preview_text.config(state='disabled')
        except Exception as e:
            logging.warning(f"Kan data preview niet tonen: {e}")

    def update_time_column(col):
        time_column_var.set(col)
        update_datetime_widgets()

    def update_datetime_widgets():
        try:
            time_column = time_column_var.get()
            if not hasattr(load_columns, 'data') or not load_columns.data:
                return
            all_times = pd.Series(dtype='datetime64[ns]')
            for _, df in load_columns.data:
                df[time_column] = pd.to_datetime(df[time_column], dayfirst=True, errors='coerce')
                all_times = pd.concat([all_times, df[time_column].dropna()])
            if all_times.empty:
                messagebox.showwarning("Waarschuwing", "Geen valide datums in de ingelezen bestanden.")
                return
            min_datetime = all_times.min()
            max_datetime = all_times.max()
            start_date_entry.set_date(min_datetime.date())
            start_hour_spinbox.delete(0, tk.END)
            start_hour_spinbox.insert(0, f"{min_datetime.hour:02}")
            start_minute_spinbox.delete(0, tk.END)
            start_minute_spinbox.insert(0, f"{min_datetime.minute:02}")
            start_second_spinbox.delete(0, tk.END)
            start_second_spinbox.insert(0, f"{min_datetime.second:02}")
            end_date_entry.set_date(max_datetime.date())
            end_hour_spinbox.delete(0, tk.END)
            end_hour_spinbox.insert(0, f"{max_datetime.hour:02}")
            end_minute_spinbox.delete(0, tk.END)
            end_minute_spinbox.insert(0, f"{max_datetime.minute:02}")
            end_second_spinbox.delete(0, tk.END)
            end_second_spinbox.insert(0, f"{max_datetime.second:02}")
        except Exception as e:
            logging.error(f"Fout bij het instellen van de datum/tijd: {e}")
            messagebox.showerror("Fout", f"Kan start- en einddatum niet instellen: {e}")

    # Genereer de grafiek met de samengevoegde data en geef aan waar een nieuwe file begint
    def generate_plot():
        file_paths_str = file_entry.get()
        time_column = time_column_var.get()
        creator_name = creator_name_var.get()
        creation_date = creation_date_var.get()
        logo_path = logo_entry.get()
        location_name = location_name_var.get()
        project_number = project_number_var.get()

        if not file_paths_str:
            messagebox.showwarning("Waarschuwing", "Selecteer minstens één CSV-bestand.")
            return

        selected_columns_primary = [col for col, var in axis_selection_vars.items() if var.get() == 'Primair']
        selected_columns_secondary = [col for col, var in axis_selection_vars.items() if var.get() == 'Secundair']
        if not selected_columns_primary and not selected_columns_secondary:
            messagebox.showwarning("Waarschuwing", "Selecteer minstens één kolom om te plotten.")
            return

        try:
            # Bouw start- en einddatetime op basis van de widget-invoer
            start_date = start_date_entry.get_date()
            start_datetime = datetime.combine(start_date, datetime.min.time()).replace(
                hour=int(start_hour_spinbox.get()),
                minute=int(start_minute_spinbox.get()),
                second=int(start_second_spinbox.get())
            )
            end_date = end_date_entry.get_date()
            end_datetime = datetime.combine(end_date, datetime.min.time()).replace(
                hour=int(end_hour_spinbox.get()),
                minute=int(end_minute_spinbox.get()),
                second=int(end_second_spinbox.get())
            )

            # Combineer alle data en filter op de geselecteerde periode
            merged_df = pd.concat([df for _, df in load_columns.data], ignore_index=True)
            merged_df[time_column] = pd.to_datetime(merged_df[time_column], dayfirst=True, errors='coerce')
            merged_df = merged_df[(merged_df[time_column] >= start_datetime) & (merged_df[time_column] <= end_datetime)]
            merged_df.sort_values(by=time_column, inplace=True)

            fig = go.Figure()
            primary_color = 'blue'
            secondary_color = 'green'

            # Voeg traces toe voor de geselecteerde kolommen
            for param in selected_columns_primary:
                try:
                    y_values = pd.to_numeric(merged_df[param], errors='coerce')
                    if y_values.isna().all():
                        logging.warning(f"Parameter '{param}' bevat geen geldige numerieke waarden en wordt niet geplot.")
                        continue
                    unit = str(load_columns.parameter_units.get(param, ''))
                    trace_name = f"{param} (Primair)"
                    if unit:
                        trace_name = f"{param} ({unit}) (Primair)"
                    fig.add_trace(go.Scatter(
                        x=merged_df[time_column],
                        y=y_values,
                        mode='lines',
                        name=trace_name,
                        yaxis='y1',
                        hoverinfo='x+y+name'
                    ))
                except Exception as e:
                    logging.warning(f"Kan kolom '{param}' niet plotten: {e}")
                    continue

            for param in selected_columns_secondary:
                try:
                    y_values = pd.to_numeric(merged_df[param], errors='coerce')
                    if y_values.isna().all():
                        logging.warning(f"Parameter '{param}' bevat geen geldige numerieke waarden en wordt niet geplot.")
                        continue
                    unit = str(load_columns.parameter_units.get(param, ''))
                    trace_name = f"{param} (Secundair)"
                    if unit:
                        trace_name = f"{param} ({unit}) (Secundair)"
                    fig.add_trace(go.Scatter(
                        x=merged_df[time_column],
                        y=y_values,
                        mode='lines',
                        name=trace_name,
                        yaxis='y2',
                        hoverinfo='x+y+name'
                    ))
                except Exception as e:
                    logging.warning(f"Kan kolom '{param}' niet plotten: {e}")
                    continue

            # Voeg verticale lijnen en annotaties toe waar een nieuwe file begint.
            # Dit doen we door te kijken waar de waarde in 'SourceFile' verandert.
            merged_df['FileChange'] = merged_df['SourceFile'].ne(merged_df['SourceFile'].shift())
            file_boundaries = merged_df[merged_df['FileChange']]
            for idx, row in file_boundaries.iterrows():
                boundary_time = row[time_column]
                file_name = row['SourceFile']
                # Voeg een verticale lijn toe
                fig.add_shape(
                    dict(
                        type="line",
                        x0=boundary_time,
                        y0=0,
                        x1=boundary_time,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(color="red", width=2, dash="dot")
                    )
                )
                # Voeg een annotatie toe
                fig.add_annotation(
                    dict(
                        x=boundary_time,
                        y=1.05,
                        xref="x",
                        yref="paper",
                        text=f"Start: {file_name}",
                        showarrow=False,
                        font=dict(color="red", size=10)
                    )
                )

            # Stel as-titels in
            y1_title = "Waarden (Primair)"
            y1_units = {str(load_columns.parameter_units.get(param, '')).strip() for param in selected_columns_primary if str(load_columns.parameter_units.get(param, '')).strip()}
            if len(y1_units) == 1:
                y1_title += f" [{y1_units.pop()}]"
            elif len(y1_units) > 1:
                y1_title += " [Verschillende eenheden]"

            y2_title = "Waarden (Secundair)"
            y2_units = {str(load_columns.parameter_units.get(param, '')).strip() for param in selected_columns_secondary if str(load_columns.parameter_units.get(param, '')).strip()}
            if len(y2_units) == 1:
                y2_title += f" [{y2_units.pop()}]"
            elif len(y2_units) > 1:
                y2_title += " [Verschillende eenheden]"

            fig.update_layout(
                xaxis_title="Tijd",
                yaxis=dict(
                    title=dict(text=y1_title, font=dict(family="Arial Black, sans-serif", size=14, color=primary_color)),
                    side="left",
                    tickfont=dict(family="Arial", size=12, color=primary_color),
                    autorange=True
                ),
                yaxis2=dict(
                    title=dict(text=y2_title, font=dict(family="Arial Black, sans-serif", size=14, color=secondary_color)),
                    overlaying='y',
                    side='right',
                    tickfont=dict(family="Arial", size=12, color=secondary_color),
                    autorange=True
                ),
                legend_title=dict(text="Parameters", font=dict(family="Arial Black, sans-serif", size=14, color="Black")),
                legend=dict(itemsizing='constant'),
                hovermode="x unified",
                hoverlabel=dict(font_size=14, font_family="Arial")
            )

            # Informatiebox
            info_text = (
                f"Projectnummer: {project_number}<br>"
                f"Naam: {creator_name}<br>"
                f"Datum: {creation_date}<br>"
                f"Locatie: {location_name}<br>"
                f"Periode: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')} - {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}"
            )
            fig.update_layout(annotations=[dict(
                x=0.5, y=1.15, xref='paper', yref='paper',
                text=info_text,
                showarrow=False,
                font=dict(size=16, color='black'),
                xanchor='center',
                yanchor='top'
            )])

            # Voeg logo toe indien geselecteerd
            if logo_path:
                try:
                    with Image.open(logo_path) as img:
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        encoded_image = base64.b64encode(buffered.getvalue()).decode()
                    fig.add_layout_image(dict(
                        source=f'data:image/png;base64,{encoded_image}',
                        xref="paper", yref="paper",
                        x=0.99, y=1.15,
                        sizex=0.3, sizey=0.3,
                        xanchor="right", yanchor="top"
                    ))
                except Exception as e:
                    logging.warning(f"Kan logo niet toevoegen: {e}")
                    messagebox.showwarning("Waarschuwing", f"Kan logo niet toevoegen: {e}")

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

            # Bestandsnaam voor opslag
            start_str = start_datetime.strftime('%Y%m%d_%H%M%S')
            end_str = end_datetime.strftime('%Y%m%d_%H%M%S')
            safe_loc = re.sub(r'[^\w\-_\. ]', '_', location_name.strip())[:50]
            filename = f"{start_str}_{end_str}_{safe_loc}.html"
            output_dir = os.path.dirname(file_entry.get().split(';')[0])
            output_file_path = os.path.join(output_dir, filename)
            fig.write_html(output_file_path)
            messagebox.showinfo("Succes", f"De interactieve grafiek is opgeslagen als {output_file_path}")
            logging.info(f"Grafiek succesvol opgeslagen als {output_file_path}")
        except Exception as e:
            logging.error(f"Fout bij grafiekgeneratie: {e}")
            messagebox.showerror("Fout", f"Er is een fout opgetreden: {e}")

    # Exportfunctie: combineer de data uit alle bestanden (voeg 'SourceFile' al toe)
    def export_filtered_data():
        file_paths_str = file_entry.get()
        if not file_paths_str:
            messagebox.showwarning("Waarschuwing", "Selecteer minstens één CSV-bestand.")
            return

        time_column = time_column_var.get()
        try:
            start_date = start_date_entry.get_date()
            start_datetime = datetime.combine(start_date, datetime.min.time()).replace(
                hour=int(start_hour_spinbox.get()),
                minute=int(start_minute_spinbox.get()),
                second=int(start_second_spinbox.get())
            )
            end_date = end_date_entry.get_date()
            end_datetime = datetime.combine(end_date, datetime.min.time()).replace(
                hour=int(end_hour_spinbox.get()),
                minute=int(end_minute_spinbox.get()),
                second=int(end_second_spinbox.get())
            )

            combined_list = []
            for file_path, df in load_columns.data:
                local_df = df.copy()
                local_df[time_column] = pd.to_datetime(local_df[time_column], dayfirst=True, errors='coerce')
                local_df = local_df[(local_df[time_column] >= start_datetime) & (local_df[time_column] <= end_datetime)]
                # 'SourceFile' staat al in de data
                combined_list.append(local_df)
            if not combined_list:
                messagebox.showwarning("Waarschuwing", "Geen data beschikbaar voor export.")
                return
            result_df = pd.concat(combined_list, ignore_index=True)

            export_path = filedialog.asksaveasfilename(
                title="Exporteer data naar CSV",
                defaultextension=".csv",
                filetypes=(("CSV-bestanden", "*.csv"), ("Alle bestanden", "*.*"))
            )
            if export_path:
                result_df.to_csv(export_path, index=False)
                messagebox.showinfo("Succes", f"De data is succesvol geëxporteerd naar {export_path}")
        except Exception as e:
            logging.error(f"Fout bij exporteren: {e}")
            messagebox.showerror("Fout", f"Er is een fout opgetreden tijdens het exporteren: {e}")

    # GUI-elementen opbouwen
    title_frame = tk.Frame(root, bd=2, relief=tk.RIDGE, padx=10, pady=10)
    title_frame.pack(pady=10, fill="x")

    project_frame = tk.Frame(title_frame)
    project_frame.pack(side=tk.LEFT, padx=10)
    tk.Label(project_frame, text="Projectnummer:").pack(anchor='w')
    project_entry = tk.Entry(project_frame, textvariable=project_number_var, width=20)
    project_entry.pack(anchor='w')

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

    file_frame = tk.Frame(root)
    file_frame.pack(pady=5)
    tk.Label(file_frame, text="CSV-bestanden:").pack(side=tk.LEFT)
    file_entry = tk.Entry(file_frame, width=50)
    file_entry.pack(side=tk.LEFT)
    tk.Button(file_frame, text="Bladeren...", command=select_file).pack(side=tk.LEFT)

    delimiter_frame = tk.Frame(root)
    delimiter_frame.pack(pady=5)
    tk.Label(delimiter_frame, text="Kies delimiter:").pack(side=tk.LEFT)
    delimiter_options = ['.', ',', ';', '\t', '|', ' ']
    delimiter_menu = tk.OptionMenu(delimiter_frame, delimiter_var, *delimiter_options)
    delimiter_menu.pack(side=tk.LEFT)

    decimal_frame = tk.Frame(root)
    decimal_frame.pack(pady=5)
    tk.Label(decimal_frame, text="Kies decimaal teken:").pack(side=tk.LEFT)
    decimal_options = [',', '.']
    decimal_menu = tk.OptionMenu(decimal_frame, decimal_var, *decimal_options)
    decimal_menu.pack(side=tk.LEFT)

    time_column_frame = tk.Frame(root)
    time_column_frame.pack(pady=5)
    tk.Label(time_column_frame, text="Selecteer tijdkolom:").pack(side=tk.LEFT)
    time_menu = tk.OptionMenu(time_column_frame, time_column_var, '')
    time_menu.pack(side=tk.LEFT)

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

    logo_frame = tk.Frame(root)
    logo_frame.pack(pady=5)
    tk.Label(logo_frame, text="Logo-bestand:").pack(side=tk.LEFT)
    logo_entry = tk.Entry(logo_frame, textvariable=logo_path_var, width=40)
    logo_entry.pack(side=tk.LEFT)
    tk.Button(logo_frame, text="Bladeren...", command=select_logo).pack(side=tk.LEFT)

    column_frame = tk.LabelFrame(root, text="Selecteer kolommen en assen:")
    column_frame.pack(pady=5, fill="both", expand="yes")

    preview_frame = tk.LabelFrame(root, text="Data Preview (Eerste 10 Rijen):")
    preview_frame.pack(pady=5, fill="both", expand="yes")

    plot_button = tk.Button(root, text="Genereer Grafiek", command=generate_plot)
    plot_button.pack(pady=10)
    
    export_button = tk.Button(root, text="Exporteer Data", command=export_filtered_data)
    export_button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
