import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector
from datetime import datetime, timedelta
import re
import chardet
from scipy.stats import linregress
import numpy as np
import io

plt.ion()


def select_directory(title="Selecteer Directory"):
    """Toon een dialoog om een directory te kiezen."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)


def get_offset_input(label):
    root = tk.Tk()
    root.withdraw()
    try:
        val = simpledialog.askfloat(f"Voer {label} offset in",
                                    f"Voer de {label} offset in:")
        return val if val is not None else 0.0
    except ValueError:
        messagebox.showerror("Ongeldige Invoer", "Voer een geldig nummer in.")
        return 0.0


def get_file_creation_time(file_path):
    try:
        ctime = os.path.getctime(file_path)
        return datetime.fromtimestamp(ctime)
    except Exception as e:
        print(f"Fout bij het verkrijgen van aanmaakdatum voor {file_path}: {e}")
        return datetime.now()


def dataset_selection_gui():
    """Toon een eerste venster om te kiezen welke datasets verwerkt worden."""
    root = tk.Tk()
    root.title("Kies datasets")

    use_wave_var = tk.BooleanVar(value=False)
    use_mon_var = tk.BooleanVar(value=False)
    use_ref_var = tk.BooleanVar(value=False)

    ttk.Checkbutton(root, text="HF druksensor gegevens",
                    variable=use_wave_var).pack(anchor="w", padx=10, pady=2)
    ttk.Checkbutton(root, text="Diver gegevens",
                    variable=use_mon_var).pack(anchor="w", padx=10, pady=2)
    ttk.Checkbutton(root, text="Referentiedruk (DDR) gegevens",
                    variable=use_ref_var).pack(anchor="w", padx=10, pady=2)

    def proceed():
        root.destroy()

    ttk.Button(root, text="Verder", command=proceed).pack(pady=10)
    root.mainloop()
    return use_wave_var.get(), use_mon_var.get(), use_ref_var.get()


def process_wave_file(file_path, pressure_offset=0):
    print(f"Bestand inlezen: {file_path}")
    try:
        data_numeric = pd.read_csv(file_path, encoding='latin1',
                                   skiprows=10, header=None)
        filtered = data_numeric[data_numeric[1] == 'C1']
        measurements = (filtered[2].astype(float) * 1000) + pressure_offset
        print(f"Meetwaarden gevonden: {len(measurements)}")
        return measurements
    except Exception as e:
        print(f"Fout bij het verwerken van {file_path}: {e}")
        return None


def detect_file_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read()
            return chardet.detect(rawdata)['encoding']
    except Exception as e:
        print(f"Fout bij het detecteren van encoding voor {file_path}: {e}")
        return 'utf-8'


def read_mon_file(file_path):
    encoding = detect_file_encoding(file_path)
    data_lines = []
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if 'END OF DATA' in line:
                    break
                elif line_num >= 54:
                    data_lines.append(line)
    except Exception as e:
        print(f"Fout bij het lezen van het .mon bestand: {e}")
    return data_lines


def parse_mon_data(data_lines, pressure_offset=0, time_offset=0):
    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3 and "/" in parts[0] and ":" in parts[1]:
            try:
                date_str = parts[0] + " " + parts[1]
                dt_obj = datetime.strptime(date_str,
                                           "%Y/%m/%d %H:%M:%S.%f") + timedelta(seconds=time_offset)
                pressure = float(parts[2].replace(',', '.')) + pressure_offset
                data.append({'Datetime': dt_obj, 'Pressure': pressure})
            except ValueError as ve:
                print(f"Fout bij het parsen van regel: {line}. Error: {ve}")
    return pd.DataFrame(data)


def convert_pressure_units(pressure_pa):
    mmh2o = pressure_pa / 9.80665
    mmhg = pressure_pa / 133.322
    bar = pressure_pa / 100000
    return pressure_pa, mmh2o, mmhg, bar


def display_pressure_summary(df_wave, df_mon, df_reference):
    """Toon een samenvatting van de gemeten drukken."""
    try:
        summary_rows = []

        if not df_wave.empty:
            wave_start = df_wave['Pressure'].iloc[0]
            wave_end = df_wave['Pressure'].iloc[-1]
            wave_max = df_wave['Pressure'].max()
            summary_rows.extend([
                ('HF Druksonde Start', wave_start),
                ('HF Druksonde Eind', wave_end),
                ('HF Druksonde Max', wave_max)
            ])

        if not df_mon.empty:
            mon_start = df_mon['Pressure'].iloc[0]
            mon_end = df_mon['Pressure'].iloc[-1]
            mon_max = df_mon['Pressure'].max()
            summary_rows.extend([
                ('Diver Druk Start', mon_start),
                ('Diver Druk Eind', mon_end),
                ('Diver Druk Max', mon_max)
            ])

        if not df_reference.empty:
            ref_start = df_reference['Pressure'].iloc[0]
            ref_end = df_reference['Pressure'].iloc[-1]
            ref_max = df_reference['Pressure'].max()
            summary_rows.extend([
                ('Referentie Sensor Start', ref_start),
                ('Referentie Sensor Eind', ref_end),
                ('Referentie Sensor Max', ref_max)
            ])

        if not summary_rows:
            messagebox.showinfo("Geen Data", "Geen data om samen te vatten.")
            return

        summary_data = {
            'Druksoort': [],
            'Druk (Pa)': [],
            'Druk (mmH₂O)': [],
            'Druk (mmHg)': [],
            'Druk (bar)': []
        }

        for soort, druk in summary_rows:
            summary_data['Druksoort'].append(soort)
            summary_data['Druk (Pa)'].append(druk)
            summary_data['Druk (mmH₂O)'].append(convert_pressure_units(druk)[1])
            summary_data['Druk (mmHg)'].append(convert_pressure_units(druk)[2])
            summary_data['Druk (bar)'].append(convert_pressure_units(druk)[3])

        df_summary = pd.DataFrame(summary_data)

        summary_window = tk.Toplevel()
        summary_window.title("Druk Samenvatting")

        tree = ttk.Treeview(summary_window, columns=(
            "Druksoort", "Druk (Pa)", "Druk (mmH₂O)", "Druk (mmHg)", "Druk (bar)"),
            show='headings')
        for col in tree['columns']:
            tree.heading(col, text=col)

        for _, row in df_summary.iterrows():
            tree.insert("", "end", values=(
                row['Druksoort'],
                f"{row['Druk (Pa)']:.2f}",
                f"{row['Druk (mmH₂O)']:.2f}",
                f"{row['Druk (mmHg)']:.2f}",
                f"{row['Druk (bar)']:.5f}"
            ))

        tree.pack(expand=True, fill='both')
        ttk.Button(summary_window, text="Sluiten",
                   command=summary_window.destroy).pack(pady=10)
    except Exception as e:
        print(f"Fout bij het tonen van druk samenvatting: {e}")


def create_time_selection_gui(df_wave, df_mon, df_reference,
                              selected_directory,
                              update_plot_callback,
                              export_and_close_callback,
                              show_wave=True,
                              show_mon=True,
                              show_ref=True):
    root = tk.Tk()
    root.title("Tijdselectie voor Grafiek")

    time_values = []
    if not df_wave.empty:
        time_values.extend([df_wave['Datetime'].min(), df_wave['Datetime'].max()])
    if not df_mon.empty:
        time_values.extend([df_mon['Datetime'].min(), df_mon['Datetime'].max()])
    if not df_reference.empty:
        time_values.extend([df_reference['Datetime'].min(), df_reference['Datetime'].max()])

    if not time_values:
        messagebox.showerror("Geen Data", "Er is geen data om te tonen.")
        return

    min_time = min(time_values)
    max_time = max(time_values)

    ttk.Label(root, text="Begin tijd (YYYY-MM-DD HH:MM:SS):").grid(
        row=0, column=0, padx=5, pady=5, sticky='e')
    start_time_entry = ttk.Entry(root, width=25)
    start_time_entry.insert(0, min_time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Eind tijd (YYYY-MM-DD HH:MM:SS):").grid(
        row=1, column=0, padx=5, pady=5, sticky='e')
    end_time_entry = ttk.Entry(root, width=25)
    end_time_entry.insert(0, max_time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time_entry.grid(row=1, column=1, padx=5, pady=5)

    use_wave_var = tk.BooleanVar(value=show_wave)
    use_mon_var = tk.BooleanVar(value=show_mon)
    use_ref_var = tk.BooleanVar(value=show_ref)

    row_idx = 2
    if not df_wave.empty:
        ttk.Checkbutton(root, text="Gebruik HF gegevens",
                        variable=use_wave_var).grid(row=row_idx, column=0, sticky='w', padx=5)
        row_idx += 1
    if not df_mon.empty:
        ttk.Checkbutton(root, text="Gebruik Diver gegevens",
                        variable=use_mon_var).grid(row=row_idx, column=0, sticky='w', padx=5)
        row_idx += 1
    if not df_reference.empty:
        ttk.Checkbutton(root, text="Gebruik Referentie gegevens",
                        variable=use_ref_var).grid(row=row_idx, column=0, sticky='w', padx=5)
        row_idx += 1

    def update_plot():
        try:
            start_time = datetime.strptime(start_time_entry.get(),
                                           "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_entry.get(),
                                         "%Y-%m-%d %H:%M:%S")
            if start_time >= end_time:
                messagebox.showerror("Ongeldige Tijd",
                                     "Begin tijd moet voor eind tijd zijn.")
                return
            update_plot_callback(start_time, end_time,
                                 use_wave_var.get(), use_mon_var.get(), use_ref_var.get())
        except ValueError:
            messagebox.showerror("Ongeldige Invoer",
                                 "Gebruik het juiste formaat: YYYY-MM-DD HH:MM:SS")

    def export_and_close():
        try:
            start_time = datetime.strptime(start_time_entry.get(),
                                           "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_entry.get(),
                                         "%Y-%m-%d %H:%M:%S")
            if start_time >= end_time:
                messagebox.showerror("Ongeldige Tijd",
                                     "Begin tijd moet voor eind tijd zijn.")
                return
            export_and_close_callback(start_time, end_time,
                                      use_wave_var.get(), use_mon_var.get(), use_ref_var.get())
            root.destroy()
        except ValueError:
            messagebox.showerror("Ongeldige Invoer",
                                 "Gebruik het juiste formaat: YYYY-MM-DD HH:MM:SS")

    ttk.Button(root, text="Update Grafiek",
               command=update_plot).grid(row=row_idx, column=0, padx=5, pady=10, sticky='e')
    ttk.Button(root, text="Export Data en Sluiten",
               command=export_and_close).grid(row=row_idx, column=1, padx=5, pady=10, sticky='w')

    root.mainloop()


def plot_combined_graph(df_wave, df_mon, df_reference,
                        start_time=None, end_time=None,
                        show_wave=True, show_mon=True, show_ref=True):
    """Interactive tijdreeksplot met RectangleSelector en slider."""
    try:
        current_shift = {'value': 0.0}

        def draw_lines(ax_plot):
            if start_time and end_time:
                mask_wave = (df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)
                mask_mon = (df_mon['Datetime'] >= start_time) & (df_mon['Datetime'] <= end_time)
                mask_ref = (df_reference['Datetime'] >= start_time) & (df_reference['Datetime'] <= end_time)
                df_wave_plot = df_wave.loc[mask_wave].copy()
                df_mon_shift = df_mon.loc[mask_mon].copy()
                df_ref_plot = df_reference.loc[mask_ref].copy()
            else:
                df_wave_plot = df_wave.copy()
                df_mon_shift = df_mon.copy()
                df_ref_plot = df_reference.copy()

            if show_mon and not df_mon_shift.empty:
                df_mon_shift['Datetime'] = df_mon_shift['Datetime'] + timedelta(seconds=current_shift['value'])

            if not df_wave_plot.empty:
                df_wave_plot = df_wave_plot[df_wave_plot['Pressure'] <= 2100]
            if not df_mon_shift.empty:
                df_mon_shift = df_mon_shift[df_mon_shift['Pressure'] <= 2100]
            if not df_ref_plot.empty:
                df_ref_plot = df_ref_plot[df_ref_plot['Pressure'] <= 2100]

            if show_wave and not df_wave_plot.empty:
                ax_plot.plot(df_wave_plot['Datetime'], df_wave_plot['Pressure'],
                             label='HF druksonde', color='red', marker='o',
                             markersize=2, linewidth=0.5, picker=5)
            if show_mon and not df_mon_shift.empty:
                ax_plot.plot(df_mon_shift['Datetime'], df_mon_shift['Pressure'],
                             label='Diver', color='blue', marker='x',
                             markersize=2, linewidth=0.5, picker=5)
            if show_ref and not df_ref_plot.empty:
                ax_plot.plot(df_ref_plot['Datetime'], df_ref_plot['Pressure'],
                             label='Referentie', color='green', marker='s',
                             markersize=4, linestyle='--', picker=5)

            ax_plot.set_xlabel('Tijd')
            ax_plot.set_ylabel('Druk (Pa)')
            ax_plot.set_title('Tijdreeks drukmetingen')
            ax_plot.legend()
            ax_plot.grid(True)

            if start_time and end_time:
                ax_plot.set_xlim(start_time, end_time)

            ys = []
            for dfp, show in [(df_wave_plot, show_wave), (df_mon_shift, show_mon), (df_ref_plot, show_ref)]:
                if show and not dfp.empty:
                    ys.append(dfp['Pressure'].dropna().values)
            if ys:
                arr = np.concatenate(ys)
                y0 = arr.min(); y1 = arr.max()
                span = y1 - y0
                if span == 0:
                    ax_plot.set_ylim(y0 * 0.95, y1 * 1.05)
                else:
                    ax_plot.set_ylim(y0 - 0.05*span, y1 + 0.05*span)

        def onselect(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if None in (x1, x2, y1, y2):
                return
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            if show_wave and not df_wave.empty:
                mask = (df_wave['Datetime'] >= x_min) & (df_wave['Datetime'] <= x_max) & \
                       (df_wave['Pressure'] >= y_min) & (df_wave['Pressure'] <= y_max)
                df_wave.loc[mask, 'Pressure'] = np.nan

            if show_mon and not df_mon.empty:
                shift = current_shift['value']
                adj_min = x_min - timedelta(seconds=shift)
                adj_max = x_max - timedelta(seconds=shift)
                mask = (df_mon['Datetime'] >= adj_min) & (df_mon['Datetime'] <= adj_max) & \
                       (df_mon['Pressure'] >= y_min) & (df_mon['Pressure'] <= y_max)
                df_mon.loc[mask, 'Pressure'] = np.nan

            if show_ref and not df_reference.empty:
                mask = (df_reference['Datetime'] >= x_min) & (df_reference['Datetime'] <= x_max) & \
                       (df_reference['Pressure'] >= y_min) & (df_reference['Pressure'] <= y_max)
                df_reference.loc[mask, 'Pressure'] = np.nan

            ax_plot.clear()
            draw_lines(ax_plot)
            fig.canvas.draw_idle()

        def on_slider(val):
            current_shift['value'] = val
            ax_plot.clear()
            draw_lines(ax_plot)
            fig.canvas.draw_idle()

        fig, (ax_plot, ax_slider) = plt.subplots(
            nrows=2, ncols=1,
            gridspec_kw={'height_ratios': [9, 1]},
            figsize=(10, 6),
            constrained_layout=True
        )

        draw_lines(ax_plot)

        rect = RectangleSelector(ax_plot, onselect, useblit=True,
                                 button=[1], minspanx=5, minspany=5,
                                 spancoords='data', interactive=True)

        slider = Slider(ax_slider, 'Shift Diver (sec)', -60.0, 60.0,
                        valinit=current_shift['value'], valstep=0.5)
        slider.on_changed(on_slider)

        plt.show(block=False)
        plt.pause(0.001)
    except Exception as e:
        print(f"Fout in plot_combined_graph: {e}")


def plot_xy_regression_with_slider(df_wave, df_mon,
                                   start_time=None, end_time=None,
                                   show_wave=True, show_mon=True):
    try:
        if not show_wave or not show_mon or df_wave.empty or df_mon.empty:
            messagebox.showinfo(
                "Regressie niet mogelijk",
                "Voor regressie zijn zowel HF- als Divergegevens nodig.")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)

        if start_time and end_time:
            mask_wave = (df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)
            mask_mon = (df_mon['Datetime'] >= start_time) & (df_mon['Datetime'] <= end_time)
            df_wave_filtered = df_wave[mask_wave]
            df_mon_filtered = df_mon[mask_mon]
        else:
            df_wave_filtered = df_wave
            df_mon_filtered = df_mon

        merged_df = pd.merge_asof(df_wave_filtered.sort_values('Datetime'),
                                  df_mon_filtered.sort_values('Datetime'),
                                  on='Datetime', suffixes=('_hf', '_diver'))

        merged_df = merged_df[(merged_df['Pressure_hf'] <= 2100) &
                              (merged_df['Pressure_diver'] <= 2100)]

        x = merged_df['Pressure_hf']
        y = merged_df['Pressure_diver']

        ax.scatter(x, y, color='blue', label='Data')

        if len(x) > 0 and len(y) > 0:
            slope, intercept, r_value, _, _ = linregress(x, y)
            line = slope * x + intercept
            r_squared = r_value**2
            ax.plot(x, line, color='red',
                    label=f'Lineaire Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r_squared:.4f}')

        ax.set_xlabel('HF druksonde druk (Pa)')
        ax.set_ylabel('Diver druk (Pa)')
        ax.set_title('X-Y Plot van HF druksonde druk tegen Diver druk')
        ax.legend()
        ax.grid(True)

        ax_shift = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider_shift = Slider(ax_shift, 'Tijd Shift (sec)', -60, 60, valinit=0, valstep=0.5)

        def update_regression(val):
            try:
                shift = slider_shift.val
                shifted_datetimes = df_mon['Datetime'] + timedelta(seconds=shift)
                shifted_df_mon = df_mon.copy()
                shifted_df_mon['Datetime'] = shifted_datetimes

                if start_time and end_time:
                    mask_wave = (df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)
                    mask_mon = (shifted_df_mon['Datetime'] >= start_time) & (shifted_df_mon['Datetime'] <= end_time)
                    df_wave_shifted = df_wave[mask_wave]
                    df_mon_shifted = shifted_df_mon[mask_mon]
                else:
                    df_wave_shifted = df_wave
                    df_mon_shifted = shifted_df_mon

                merged = pd.merge_asof(df_wave_shifted.sort_values('Datetime'),
                                       df_mon_shifted.sort_values('Datetime'),
                                       on='Datetime', suffixes=('_hf', '_diver'))

                merged = merged[(merged['Pressure_hf'] <= 2100) &
                                (merged['Pressure_diver'] <= 2100)]

                x_new = merged['Pressure_hf']
                y_new = merged['Pressure_diver']

                valid_mask = (~np.isnan(x_new)) & (~np.isnan(y_new))
                x_valid = x_new[valid_mask]
                y_valid = y_new[valid_mask]

                if len(x_valid) > 0 and len(y_valid) > 0:
                    slope, intercept, r_value, _, _ = linregress(x_valid, y_valid)
                    line = slope * x_valid + intercept
                    r_squared = r_value**2

                    ax.clear()
                    ax.scatter(x_valid, y_valid, color='blue', label='Data')
                    ax.plot(x_valid, line, color='red',
                            label=f'Lineaire Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r_squared:.4f}')
                    ax.set_xlabel('HF druksonde druk (Pa)')
                    ax.set_ylabel('Diver druk (Pa)')
                    ax.set_title('X-Y Plot van HF druksonde druk tegen Diver druk')
                    ax.legend()
                    ax.grid(True)
                    fig.canvas.draw_idle()
            except Exception as e:
                print(f"Fout in update_regression: {e}")

        slider_shift.on_changed(update_regression)

        plt.show(block=False)
        plt.pause(0.001)
    except Exception as e:
        print(f"Fout in plot_xy_regression_with_slider: {e}")


def export_data(df_wave, df_mon, df_reference, selected_directory,
                start_time, end_time,
                use_wave=True, use_mon=True, use_ref=True):
    try:
        mask_wave = (df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)
        mask_mon = (df_mon['Datetime'] >= start_time) & (df_mon['Datetime'] <= end_time)

        df_wave_filtered = df_wave[mask_wave]
        df_mon_filtered = df_mon[mask_mon]
        mask_ref = (df_reference['Datetime'] >= start_time) & (df_reference['Datetime'] <= end_time)
        df_ref_filtered = df_reference[mask_ref]

        df_wave_filtered = df_wave_filtered[df_wave_filtered['Pressure'] <= 2100]
        df_mon_filtered = df_mon_filtered[df_mon_filtered['Pressure'] <= 2100]
        df_ref_filtered = df_ref_filtered[df_ref_filtered['Pressure'] <= 2100]

        merged_df = pd.merge_asof(df_wave_filtered.sort_values('Datetime'),
                                  df_mon_filtered.sort_values('Datetime'),
                                  on='Datetime', suffixes=('_hf', '_diver'))

        if use_wave:
            hf_filename = os.path.join(selected_directory, 'hf_druksonde_data_filtered.csv')
            df_wave_filtered.to_csv(hf_filename, index=False)
            print(f"HF druksonde data geëxporteerd naar {hf_filename}")

        if use_mon:
            diver_filename = os.path.join(selected_directory, 'diver_druk_data_filtered.csv')
            df_mon_filtered.to_csv(diver_filename, index=False)
            print(f"Diver druk data geëxporteerd naar {diver_filename}")

        if use_ref:
            ref_filename = os.path.join(selected_directory, 'ref_druk_data_filtered.csv')
            df_ref_filtered.to_csv(ref_filename, index=False)
            print(f"Referentiedruk data geëxporteerd naar {ref_filename}")
        if use_wave and use_mon:
            combined_filename = os.path.join(selected_directory, 'combined_data_filtered.csv')
            merged_df.to_csv(combined_filename, index=False)
            print(f"Gecombineerde data geëxporteerd naar {combined_filename}")

        messagebox.showinfo(
            "Export Succesvol",
            f"Data succesvol geëxporteerd naar:\n{selected_directory}")
    except Exception as e:
        print(f"Fout bij het exporteren van data: {e}")
        messagebox.showerror(
            "Export Fout",
            f"Er is een fout opgetreden bij het exporteren van de data:\n{e}")


def load_reference_sensor_data(file_path, base_datetime=None):
    """Lees een DDR referentiedrukbestand. Dit bestand heeft vaak in de eerste
    regel de bestandsnaam met datum en tijd en vanaf de derde regel de kolommen
    ``time[sec],data``. Wanneer alleen relatieve tijd beschikbaar is, wordt deze
    omgezet naar echte datums gebaseerd op ``base_datetime`` of de datum in het
    bestand zelf.
    """
    try:
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            raise ValueError("Bestand is leeg")

        # Probeer een datumtijd te halen uit de eerste regel of uit de bestandsnaam
        first_line = lines[0]
        dt_match = re.search(r"(\d{8}_\d{6})", first_line)
        if not dt_match:
            fname = os.path.basename(file_path)
            dt_match = re.search(r"(\d{8}_\d{6})", fname)

        file_dt = None
        if dt_match:
            try:
                file_dt = datetime.strptime(dt_match.group(1), "%Y%m%d_%H%M%S")
            except Exception:
                file_dt = None

        header_idx = 0
        for i, line in enumerate(lines):
            if "," in line and any(key in line.lower() for key in ["time", "seconde", "druk", "data"]):
                header_idx = i
                break

        if header_idx + 1 >= len(lines):
            raise ValueError("Onvoldoende datarijen")

        csv_content = "\n".join(lines[header_idx:])
        df_reference = pd.read_csv(io.StringIO(csv_content))

        if df_reference.empty or df_reference.shape[1] < 2:
            raise ValueError("Te weinig kolommen in referentiedrukbestand")

        time_col = df_reference.columns[0]
        pressure_col = df_reference.columns[1]

        df_reference['Pressure'] = pd.to_numeric(df_reference[pressure_col], errors='coerce')

        if file_dt is None and base_datetime is None:
            base_dt = datetime.now()
        else:
            base_dt = file_dt if file_dt is not None else base_datetime

        df_reference['Datetime'] = base_dt + pd.to_timedelta(df_reference[time_col], unit='s')

        df_reference = df_reference[['Datetime', 'Pressure']].dropna()
        return df_reference
    except Exception as e:
        print(f"Fout bij het lezen van referentiedrukbestand: {e}")
        return pd.DataFrame()


def main():
    use_wave, use_mon, use_ref = dataset_selection_gui()
    if not any([use_wave, use_mon, use_ref]):
        messagebox.showwarning("Geen keuze", "Geen datasets geselecteerd.")
        return

    df_wave = pd.DataFrame()
    df_mon = pd.DataFrame()
    df_reference = pd.DataFrame()
    selected_directory = ""

    if use_wave:
        selected_directory = select_directory("Selecteer HF druksensor directory")
        if selected_directory:
            files = [f for f in os.listdir(selected_directory)
                     if f.lower().endswith('.csv') and not f.endswith('_filtered.csv')
                     and f != 'combined_data_filtered.csv']
            if not files:
                messagebox.showwarning("Geen CSV-bestanden",
                                       "Geen CSV-bestanden gevonden in de geselecteerde directory (geen ongefilterde .csv bestanden).")
                use_wave = False
            else:
                all_measurements = []
                all_timestamps = []
                wave_pressure_offset = get_offset_input('HF druksonde druk')
                for file in files:
                    file_path = os.path.join(selected_directory, file)
                    measurements = process_wave_file(file_path, wave_pressure_offset)
                    if measurements is not None:
                        creation_time = get_file_creation_time(file_path)
                        timestamps = [creation_time + timedelta(seconds=i/8)
                                      for i in range(len(measurements))]
                        all_measurements.extend(measurements)
                        all_timestamps.extend(timestamps)
                df_wave = pd.DataFrame({'Datetime': all_timestamps, 'Pressure': all_measurements})
        else:
            messagebox.showwarning("Geen Directory Geselecteerd", "Er is geen directory geselecteerd.")
            use_wave = False

    if use_mon:
        mon_file = filedialog.askopenfilename(title="Selecteer een .mon bestand (Diver)",
                                              filetypes=[("MON bestanden", "*.mon")])
        if mon_file:
            mon_pressure_offset = get_offset_input('Diver druk')
            mon_time_offset = get_offset_input('Diver tijd (in seconden)')
            data_lines = read_mon_file(mon_file)
            if data_lines:
                df_mon = parse_mon_data(data_lines, mon_pressure_offset, mon_time_offset)
                if df_mon.empty:
                    messagebox.showerror("Geen Data", "Geen geldige data geparsed uit het .mon bestand.")
                    use_mon = False
            else:
                messagebox.showerror("Geen Data", "Geen data gevonden in het .mon bestand.")
                use_mon = False
        else:
            messagebox.showwarning("Geen Bestand Geselecteerd", "Er is geen .mon bestand geselecteerd.")
            use_mon = False

    if use_ref:
        ref_file = filedialog.askopenfilename(title="Selecteer DDR referentiedruk bestand",
                                              filetypes=[("CSV bestanden", "*.csv"), ("Alle bestanden", "*.*")])
        if ref_file:
            base_dt = None
            if not df_wave.empty:
                base_dt = df_wave['Datetime'].min()
            elif not df_mon.empty:
                base_dt = df_mon['Datetime'].min()
            df_reference = load_reference_sensor_data(ref_file, base_dt)
            if df_reference.empty:
                messagebox.showerror("Geen Data", "Kon referentiedrukbestand niet inlezen.")
                use_ref = False
        else:
            messagebox.showwarning("Geen bestand geselecteerd", "Er is geen DDR bestand geselecteerd.")
            use_ref = False

    if df_wave.empty and df_mon.empty and df_reference.empty:
        messagebox.showerror("Geen Data", "Er kon geen data worden geladen.")
        return

    display_pressure_summary(df_wave, df_mon, df_reference)

    def update_plot_with_time_range(start, end, show_wave_choice, show_mon_choice, show_ref_choice):
        plot_combined_graph(df_wave, df_mon, df_reference,
                            start, end, show_wave_choice, show_mon_choice, show_ref_choice)
        plot_xy_regression_with_slider(df_wave, df_mon,
                                       start, end, show_wave_choice, show_mon_choice)

    def export_and_close(start, end, show_wave_choice, show_mon_choice, show_ref_choice):
        # Alleen data exporteren en daarna de GUI sluiten
        export_data(
            df_wave,
            df_mon,
            df_reference,
            selected_directory,
            start,
            end,
            show_wave_choice,
            show_mon_choice,
            show_ref_choice,
        )

    create_time_selection_gui(df_wave, df_mon, df_reference,
                              selected_directory,
                              update_plot_with_time_range,
                              export_and_close,
                              show_wave=use_wave,
                              show_mon=use_mon,
                              show_ref=use_ref)


if __name__ == "__main__":
    main()
