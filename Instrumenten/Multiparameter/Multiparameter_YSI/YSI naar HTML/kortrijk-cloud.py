# kortrijk-cloud.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, time
import re  # Voor het valideren van bestandsnamen
from PIL import Image
import base64
from collections import defaultdict
from io import BytesIO
import logging

# Stel logging in
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pillow_version_at_least(major, minor, patch=0):
    version = tuple(map(int, Image.__version__.split('.')[:3]))
    return version >= (major, minor, patch)

def main():
    st.set_page_config(page_title="CSV Plotter", layout="wide")
    st.title("CSV Plotter")

    # Controleer of Pillow versie voldoet
    if not pillow_version_at_least(8, 0, 0):
        st.error("Pillow versie 8.0.0 of hoger is vereist.")
        return

    # Invoer voor projectinformatie
    st.header("Projectinformatie")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        project_number = st.text_input("Projectnummer:")
    with col2:
        creator_name = st.text_input("Naam maker:")
    with col3:
        location_name = st.text_input("Locatie:")
    with col4:
        creation_date = st.date_input("Datum aanmaak:", value=datetime.today())

    st.markdown("---")

    # Bestand uploaden
    st.header("Bestandselectie")
    uploaded_file = st.file_uploader("Selecteer een CSV-bestand", type=["csv"])

    if uploaded_file is not None:
        # Selecteer delimiter en decimaal teken
        col1, col2 = st.columns(2)
        with col1:
            delimiter = st.selectbox("Kies delimiter:", options=['.', ',', ';', '\t', '|', ' '], index=0)  # delimiter = "."
        with col2:
            decimal_sign = st.selectbox("Kies decimaal teken:", options=[',', '.'], index=0)  # decimaal teken = ","

        try:
            # Lees de eerste twee rijen om kolomnamen en eenheden te krijgen
            data_preview = pd.read_csv(uploaded_file, delimiter=delimiter, nrows=2, header=None)
            parameter_names = data_preview.iloc[0].tolist()
            units = data_preview.iloc[1].tolist()

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

            # Lees de rest van de data vanaf de derde rij met unieke kolomnamen
            uploaded_file.seek(0)  # Reset de bestandslezer
            data = pd.read_csv(uploaded_file, delimiter=delimiter, skiprows=2, names=combined_names)

            # Vervang decimaal teken in alle kolommen die objecten zijn (strings)
            # Dit voorkomt onbedoelde vervangingen in numerieke kolommen
            numeric_columns = data.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                data[col] = data[col].str.replace(decimal_sign, '.', regex=False)

            st.subheader("Data Preview (Eerste 10 Rijen)")
            st.dataframe(data.head(10))

            # Selecteer de tijdkolom
            time_column = st.selectbox("Selecteer tijdkolom:", options=combined_names)

            # Automatisch bepalen van start- en eindtijd
            try:
                data[time_column] = pd.to_datetime(data[time_column], dayfirst=True, errors='coerce')
                min_datetime = data[time_column].min()
                max_datetime = data[time_column].max()

                if pd.isnull(min_datetime) or pd.isnull(max_datetime):
                    st.error("De geselecteerde tijdkolom bevat geen geldige datums.")
                    return

                start_datetime = min_datetime
                end_datetime = max_datetime

                st.info(f"Automatisch bepaalde starttijd: {start_datetime}")
                st.info(f"Automatisch bepaalde eindtijd: {end_datetime}")
            except Exception as e:
                st.error(f"Fout bij het bepalen van start- en eindtijd: {e}")
                return

            st.markdown("---")

            # Logo uploaden
            st.header("Logo Uploaden (optioneel)")
            uploaded_logo = st.file_uploader("Selecteer een logo-bestand", type=["png", "jpg", "jpeg", "gif"])

            st.markdown("---")

            # Kolomselectie voor assen
            st.header("Selecteer kolommen en assen")
            axis_selection = {}
            cols = st.columns(3)
            for i, col in enumerate(combined_names):
                if col != time_column:
                    with cols[i % 3]:
                        axis = st.selectbox(f"Assen voor {col}:", options=['None', 'Primair', 'Secundair'], key=col)
                        axis_selection[col] = axis

            st.markdown("---")

            # Plot knop
            if st.button("Genereer Grafiek"):
                selected_columns_primary = [col for col, axis in axis_selection.items() if axis == 'Primair']
                selected_columns_secondary = [col for col, axis in axis_selection.items() if axis == 'Secundair']

                if not selected_columns_primary and not selected_columns_secondary:
                    st.warning("Selecteer minstens één kolom om te plotten.")
                else:
                    try:
                        # Filter data op basis van start- en einddatum
                        data_filtered = data[(data[time_column] >= start_datetime) & (data[time_column] <= end_datetime)]

                        if data_filtered.empty:
                            st.warning("Geen data beschikbaar binnen de geselecteerde periode.")
                        else:
                            # Converteer numerieke kolommen naar float
                            for col in selected_columns_primary + selected_columns_secondary:
                                data_filtered[col] = pd.to_numeric(data_filtered[col], errors='coerce')

                            # Plotly interactieve grafiek maken
                            fig = go.Figure()

                            # Kleurcodering voor assen
                            primary_color = 'blue'
                            secondary_color = 'green'

                            # Traces voor primaire y-as
                            for param in selected_columns_primary:
                                y_values = data_filtered[param]
                                unit = parameter_units.get(param, '')
                                trace_name = f"{param} (Primair)" if unit.strip() == '' else f"{param} ({unit}) (Primair)"
                                fig.add_trace(go.Scatter(
                                    x=data_filtered[time_column],
                                    y=y_values,
                                    mode='lines',
                                    name=trace_name,
                                    yaxis='y1',
                                    hoverinfo='x+y+name'
                                ))

                            # Traces voor secundaire y-as
                            for param in selected_columns_secondary:
                                y_values = data_filtered[param]
                                unit = parameter_units.get(param, '')
                                trace_name = f"{param} (Secundair)" if unit.strip() == '' else f"{param} ({unit}) (Secundair)"
                                fig.add_trace(go.Scatter(
                                    x=data_filtered[time_column],
                                    y=y_values,
                                    mode='lines',
                                    name=trace_name,
                                    yaxis='y2',
                                    hoverinfo='x+y+name'
                                ))

                            # Layout instellen
                            fig.update_layout(
                                xaxis_title="Tijd",
                                yaxis_title="Waarden (Primair)",
                                yaxis2=dict(
                                    title="Waarden (Secundair)",
                                    overlaying='y',
                                    side='right'
                                ),
                                legend_title="Parameters",
                                hovermode="x unified",
                                autosize=True,
                                margin=dict(l=50, r=50, t=100, b=50)
                            )

                            # Voeg informatiebox toe boven de grafiek
                            info_text = (


                                f"Projectnummer: {project_number}<br>"
                                f"Naam: {creator_name}<br>"
                                f"Datum: {creation_date.strftime('%d/%m/%Y')}<br>"
                                f"Locatie: {location_name}<br>"
                                f"Periode: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')} - {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}"
                            )
                            fig.add_annotation(dict(
                                x=0.5,
                                y=1.15,
                                xref='paper',
                                yref='paper',
                                text=info_text,
                                showarrow=False,
                                font=dict(size=14, color='black'),
                                xanchor='center',
                                yanchor='top'
                            ))

                            # Voeg logo toe als het is geselecteerd
                            if uploaded_logo is not None:
                                try:
                                    logo = Image.open(uploaded_logo)
                                    buffered = BytesIO()
                                    logo.save(buffered, format="PNG")
                                    encoded_image = base64.b64encode(buffered.getvalue()).decode()
                                    fig.add_layout_image(
                                        dict(
                                            source=f"data:image/png;base64,{encoded_image}",
                                            xref="paper", yref="paper",
                                            x=0.99, y=1.15,
                                            sizex=0.3, sizey=0.3,
                                            xanchor="right", yanchor="top"
                                        )
                                    )
                                except Exception as e:
                                    st.warning(f"Kan logo niet toevoegen aan de grafiek: {e}")

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

                            st.plotly_chart(fig, use_container_width=True)

                            # Opslaan als HTML-bestand
                            filename = f"{start_datetime.strftime('%Y%m%d_%H%M%S')}_{end_datetime.strftime('%Y%m%d_%H%M%S')}_{re.sub(r'[^\w\-_\. ]', '_', location_name.strip())[:50]}.html"
                            output_dir = "output_html"
                            os.makedirs(output_dir, exist_ok=True)
                            output_file_path = os.path.join(output_dir, filename)
                            fig.write_html(output_file_path)
                            st.success(f"De interactieve grafiek is opgeslagen als `{filename}`.")

                            # Download link voor het HTML-bestand
                            with open(output_file_path, "rb") as file:
                                btn = st.download_button(
                                    label="Download HTML Grafiek",
                                    data=file,
                                    file_name=filename,
                                    mime="text/html"
                                )
                    except Exception as e:
                        st.error(f"Er is een fout opgetreden bij het genereren van de grafiek: {e}")
        except Exception as e:
            st.error(f"Er is een fout opgetreden bij het verwerken van het bestand: {e}")

    # Voeg een knop toe om de app te sluiten
    st.sidebar.markdown("---")
    if st.sidebar.button("Sluit App"):
        # Gebruik JavaScript om het tabblad te sluiten
        st.markdown(
            """
            <script>
            window.close();
            </script>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    if __name__ == "__main__":
        main()
