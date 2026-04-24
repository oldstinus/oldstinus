import os
import json
import tkinter as tk
import pandas as pd
import plotly.graph_objects as go
from tkinter import filedialog, messagebox
from bs4 import BeautifulSoup

def extract_json_from_text(text, start_index):
    """
    Zoekt vanaf start_index naar een JSON-object of -array door de haakjes te balanceren.
    Geeft de gevonden JSON-string terug.
    """
    if text[start_index] not in ['{', '[']:
        return None
    opening = text[start_index]
    closing = '}' if opening == '{' else ']'
    stack = []
    for i in range(start_index, len(text)):
        char = text[i]
        if char == opening:
            stack.append(opening)
        elif char == closing:
            if stack:
                stack.pop()
                if not stack:
                    return text[start_index:i+1]
    return None

def extract_plot_data_from_file(filepath):
    """
    Extraheert Plotly grafiekdata uit een HTML-bestand.
    Zoekt in de script-tags naar een Plotly.newPlot-aanroep en extraheert de JSON-data.
    Geeft voor elke trace een tuple (x, y, naam) terug.
    """
    plot_data_list = []
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        script_tags = soup.find_all("script")
        for script in script_tags:
            if "Plotly.newPlot" in script.text:
                pos = script.text.find("Plotly.newPlot(")
                if pos == -1:
                    continue
                pos = script.text.find("(", pos)
                if pos == -1:
                    continue
                # Zoek het eerste teken '{' of '[' na de "("
                json_start = script.text.find("{", pos)
                alt_start = script.text.find("[", pos)
                if json_start == -1 or (alt_start != -1 and alt_start < json_start):
                    json_start = alt_start
                if json_start == -1:
                    continue
                json_str = extract_json_from_text(script.text, json_start)
                if not json_str:
                    continue
                try:
                    data_obj = json.loads(json_str)
                except Exception:
                    try:
                        data_obj = json.loads(json_str.replace("'", "\""))
                    except Exception:
                        continue
                # Indien de JSON een dict bevat met een "data"-key, gebruik die; anders gaan we ervan uit dat het direct de traces betreft.
                if isinstance(data_obj, dict) and "data" in data_obj:
                    traces = data_obj["data"]
                elif isinstance(data_obj, list):
                    traces = data_obj
                else:
                    continue
                for trace in traces:
                    x = trace.get("x", [])
                    y = trace.get("y", [])
                    name = trace.get("name", os.path.basename(filepath))
                    if x and y:
                        plot_data_list.append((x, y, name))
    return plot_data_list

def merge_plot_data(file_list):
    """
    Voegt de grafiekdata van alle geselecteerde HTML-bestanden samen.
    """
    merged_traces = []
    for file in file_list:
        traces = extract_plot_data_from_file(file)
        merged_traces.extend(traces)
    return merged_traces

def plot_merged_data_interactive(merged_traces):
    """
    Bouwt een interactieve Plotly-grafiek met de gecombineerde traces.
    Indien de trace “conductiviteit” in de naam voorkomt, wordt deze op de tweede y-as getoond.
    Retourneert de figuur zodat deze eventueel als HTML opgeslagen kan worden.
    """
    fig = go.Figure()
    for x, y, name in merged_traces:
        trace_config = {}
        if "conductiviteit" in name.lower():
            trace_config["yaxis"] = "y2"
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name, **trace_config))
    fig.update_layout(
        title="Gecombineerde Grafiek uit HTML-bestanden",
        xaxis_title="Tijd",
        yaxis=dict(title="Waarde"),
        yaxis2=dict(
            title="Conductiviteit",
            overlaying='y',
            side='right'
        )
    )
    fig.show()
    return fig

def export_to_csv(merged_traces, save_path):
    """
    Exporteert de gecombineerde data naar een CSV-bestand.
    In plaats van ervan uit te gaan dat alle traces dezelfde x-as delen,
    wordt nu de unie van alle x-waarden gebruikt. Voor elke trace wordt met een mapping gecontroleerd
    of er per tijdstip een y-waarde aanwezig is; ontbrekende waarden worden opgevuld.
    """
    if not merged_traces:
        messagebox.showwarning("Geen data", "Geen grafiekdata om te exporteren.")
        return

    # Stel de unie van alle x-waarden samen
    all_time = sorted({t for (x, y, name) in merged_traces for t in x})
    data = {"Tijd": all_time}
    for x, y, name in merged_traces:
        mapping = dict(zip(x, y))
        # Voor elk tijdstip in de unie: als er een waarde is, gebruik deze, anders een lege string
        col_data = [mapping.get(t, '') for t in all_time]
        data[name] = col_data
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, sep=';')
    messagebox.showinfo("Export succesvol", f"Data geëxporteerd naar {save_path}")

def main():
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(
        title="Selecteer HTML-bestanden",
        filetypes=[("HTML bestanden", "*.html")]
    )
    if not files:
        messagebox.showwarning("Geen bestanden", "Er zijn geen bestanden geselecteerd.")
        return
    merged_traces = merge_plot_data(files)
    if not merged_traces:
        messagebox.showwarning("Geen data", "Geen grafiekdata gevonden in de geselecteerde HTML-bestanden.")
        return

    if messagebox.askyesno("Interactie grafiek weergeven", "Wil je de interactieve grafiek weergeven?"):
        fig = plot_merged_data_interactive(merged_traces)
        if messagebox.askyesno("Grafiek opslaan", "Wil je de interactieve grafiek opslaan als HTML-bestand?"):
            save_path = filedialog.asksaveasfilename(
                title="Opslaan als HTML",
                defaultextension=".html",
                filetypes=[("HTML bestanden", "*.html")]
            )
            if save_path:
                fig.write_html(save_path)
                messagebox.showinfo("Opslaan succesvol", f"Grafiek succesvol opgeslagen als {save_path}")

    if messagebox.askyesno("Data exporteren", "Wil je de gecombineerde data exporteren naar een CSV-bestand?"):
        save_path = filedialog.asksaveasfilename(
            title="Opslaan als CSV",
            defaultextension=".csv",
            filetypes=[("CSV bestanden", "*.csv")]
        )
        if save_path:
            export_to_csv(merged_traces, save_path)

if __name__ == "__main__":
    main()
