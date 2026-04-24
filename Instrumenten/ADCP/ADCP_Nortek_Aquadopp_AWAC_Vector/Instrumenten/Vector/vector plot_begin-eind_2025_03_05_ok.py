import logging
import re
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from tkcalendar import DateEntry
except ImportError as exc:
    raise ImportError(
        "De module tkcalendar is nodig. Installeer deze met 'pip install tkcalendar'."
    ) from exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


DAT_COLUMN_NAMES = [
    "Burst_counter",
    "Ensemble_counter",
    "Velocity_Beam1",
    "Velocity_Beam2",
    "Velocity_Beam3",
    "Amplitude_Beam1",
    "Amplitude_Beam2",
    "Amplitude_Beam3",
    "SNR_Beam1",
    "SNR_Beam2",
    "SNR_Beam3",
    "Correlation_Beam1",
    "Correlation_Beam2",
    "Correlation_Beam3",
    "Pressure",
    "Analog_input1",
    "Analog_input2",
    "Checksum",
]


SEN_COLUMN_NAMES = [
    "Month",
    "Day",
    "Year",
    "Hour",
    "Minute",
    "Second",
    "Error_code",
    "Status_code",
    "Battery_voltage",
    "Soundspeed",
    "Heading",
    "Pitch",
    "Roll",
    "Temperature",
    "Analog_input",
    "Checksum",
]


SEN_MERGE_COLUMNS = [
    "Error_code",
    "Status_code",
    "Battery_voltage",
    "Soundspeed",
    "Heading",
    "Pitch",
    "Roll",
    "Temperature",
    "Analog_input",
    "SEN_Checksum",
]


SAMPLING_RATE_PATTERN = re.compile(r"sampling\s*[-_:]?\s*rate.*?(\d+(?:\.\d+)?)\s*hz", re.IGNORECASE)


def angle_in_range(angle, lower, upper):
    if lower <= upper:
        return lower <= angle <= upper
    return angle >= lower or angle <= upper


def parse_sampling_rate_from_hdr(hdr_file_path):
    if not hdr_file_path:
        return None

    hdr_path = Path(hdr_file_path)
    if not hdr_path.exists():
        return None

    lines = hdr_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        match = SAMPLING_RATE_PATTERN.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def get_transformation_matrix_from_hdr(hdr_file_path):
    try:
        lines = Path(hdr_file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        logging.error("Fout bij het lezen van het hdr bestand: %s", exc)
        raise

    for index, line in enumerate(lines):
        if "Transformation matrix" not in line:
            continue

        parts = line.split("Transformation matrix", maxsplit=1)[-1].strip().split()
        row1 = [float(value) for value in parts] if parts else []
        row2 = [float(value) for value in lines[index + 1].strip().split()]
        row3 = [float(value) for value in lines[index + 2].strip().split()]

        if len(row1) < 3:
            needed = 3 - len(row1)
            row1.extend(row2[:needed])
            row2 = row2[needed:]

        matrix = np.array([row1, row2, row3], dtype=float)
        logging.info("Transformatie matrix ingelezen: %s", matrix)
        return matrix

    raise ValueError("Transformatie matrix niet gevonden in het hdr bestand.")


def resolve_related_files(selected_path):
    base_path = Path(selected_path).with_suffix("")
    return {
        "dat_file": base_path.with_suffix(".dat"),
        "sen_file": base_path.with_suffix(".sen"),
        "hdr_file": base_path.with_suffix(".hdr"),
    }


def read_data_file(file_path, column_names):
    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=column_names,
        comment="#",
        engine="python",
    )


def create_datetime_column(data_sen):
    sen = data_sen.copy()

    for column in ["Month", "Day", "Year", "Hour", "Minute", "Second"]:
        sen[column] = pd.to_numeric(sen[column], errors="coerce")

    year = sen["Year"].copy()
    year = np.where(year < 100, np.where(year < 80, year + 2000, year + 1900), year)
    sen["Year"] = year

    sen["Datetime"] = pd.to_datetime(
        {
            "year": sen["Year"],
            "month": sen["Month"],
            "day": sen["Day"],
            "hour": sen["Hour"],
            "minute": sen["Minute"],
            "second": sen["Second"],
        },
        errors="coerce",
    )
    return sen


def build_dat_datetime_index(data_dat, data_sen, hdr_file_path):
    sen_datetimes = data_sen["Datetime"].dropna().reset_index(drop=True)
    if sen_datetimes.empty:
        raise ValueError("Geen geldige tijdstempels in het .sen bestand gevonden.")

    if len(data_dat) == len(data_sen):
        return pd.to_datetime(data_sen["Datetime"]).reset_index(drop=True)

    sampling_rate_hz = parse_sampling_rate_from_hdr(hdr_file_path)
    if not sampling_rate_hz or sampling_rate_hz <= 0:
        sampling_rate_hz = max(1.0, len(data_dat) / max(len(sen_datetimes), 1))
        logging.info("Sampling rate niet in hdr gevonden, geschat op %.3f Hz.", sampling_rate_hz)
    else:
        logging.info("Sampling rate uit hdr: %.3f Hz.", sampling_rate_hz)

    offsets = pd.to_timedelta(np.arange(len(data_dat), dtype=np.float64) / sampling_rate_hz, unit="s")
    return pd.Series(pd.to_datetime(sen_datetimes.iloc[0]) + offsets)


def merge_sen_metadata(data_dat, data_sen):
    sen = data_sen.rename(columns={"Checksum": "SEN_Checksum"}).copy()
    sen_columns = ["Datetime"] + [column for column in SEN_MERGE_COLUMNS if column in sen.columns]

    if len(data_dat) == len(sen):
        for column in sen_columns:
            if column == "Datetime":
                continue
            data_dat[column] = sen[column].reset_index(drop=True)
        return data_dat

    left = data_dat.sort_values("Datetime").reset_index(drop=True)
    right = sen[sen_columns].sort_values("Datetime").reset_index(drop=True)
    return pd.merge_asof(left, right, on="Datetime", direction="backward")


def load_vector_dataset(dat_file_path, sen_file_path, hdr_file_path):
    data_dat = read_data_file(dat_file_path, DAT_COLUMN_NAMES)
    data_sen = create_datetime_column(read_data_file(sen_file_path, SEN_COLUMN_NAMES))

    data_dat["Datetime"] = build_dat_datetime_index(data_dat, data_sen, hdr_file_path)
    data_dat = merge_sen_metadata(data_dat, data_sen)

    checksum_mask = pd.to_numeric(data_dat["Checksum"], errors="coerce").fillna(1) == 0
    if "SEN_Checksum" in data_dat.columns:
        checksum_mask &= pd.to_numeric(data_dat["SEN_Checksum"], errors="coerce").fillna(0) == 0

    data_valid = data_dat.loc[checksum_mask].copy()
    data_valid = data_valid.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    if data_valid.empty:
        raise ValueError("Geen geldige metingen over na checksum/tijd filtering.")

    return data_valid


def longest_true_block(mask):
    best_start = None
    best_end = None
    current_start = None

    for index, value in enumerate(mask):
        if value and current_start is None:
            current_start = index
        if not value and current_start is not None:
            if best_start is None or index - current_start > best_end - best_start:
                best_start = current_start
                best_end = index
            current_start = None

    if current_start is not None:
        end_index = len(mask)
        if best_start is None or end_index - current_start > best_end - best_start:
            best_start = current_start
            best_end = end_index

    if best_start is None:
        return None
    return best_start, best_end


def detect_submerged_interval(dataframe):
    result = {
        "detected": False,
        "mask": pd.Series(True, index=dataframe.index),
        "threshold": None,
        "baseline": None,
        "start_dt": pd.to_datetime(dataframe["Datetime"].min()),
        "end_dt": pd.to_datetime(dataframe["Datetime"].max()),
        "reason": "Geen drukfilter toegepast.",
    }

    if "Pressure" not in dataframe.columns:
        result["reason"] = "Geen Pressure kolom gevonden."
        return result

    pressure = pd.to_numeric(dataframe["Pressure"], errors="coerce")
    valid_pressure = pressure.dropna()
    if valid_pressure.empty:
        result["reason"] = "Pressure bevat geen numerieke waarden."
        return result

    p05 = float(valid_pressure.quantile(0.05))
    p95 = float(valid_pressure.quantile(0.95))
    dynamic_range = p95 - p05

    if not np.isfinite(dynamic_range) or dynamic_range <= max(abs(p95) * 0.05, 0.02):
        result["reason"] = "Drukverschil te klein voor automatische lucht/water detectie."
        return result

    threshold = p05 + 0.15 * dynamic_range
    wet_mask = pressure > threshold
    block = longest_true_block(wet_mask.fillna(False).to_numpy())

    if block is None:
        result["reason"] = "Geen onderwaterblok gevonden op basis van druk."
        return result

    start_idx, end_idx = block
    trimmed_mask = pd.Series(False, index=dataframe.index)
    trimmed_mask.iloc[start_idx:end_idx] = True

    wet_data = dataframe.loc[trimmed_mask]
    if wet_data.empty:
        result["reason"] = "Automatische waterselectie leverde geen data op."
        return result

    result.update(
        {
            "detected": True,
            "mask": trimmed_mask,
            "threshold": threshold,
            "baseline": p05,
            "start_dt": pd.to_datetime(wet_data["Datetime"].min()),
            "end_dt": pd.to_datetime(wet_data["Datetime"].max()),
            "reason": "Langste onderwaterblok op basis van Pressure geselecteerd.",
        }
    )
    return result


def transform_velocities(data_filtered, transformation_matrix):
    beam_velocities = data_filtered[["Velocity_Beam1", "Velocity_Beam2", "Velocity_Beam3"]].to_numpy(dtype=float)
    enu_velocities = beam_velocities.dot(transformation_matrix.T)
    data_filtered = data_filtered.copy()
    data_filtered["Velocity_East"] = enu_velocities[:, 0]
    data_filtered["Velocity_North"] = enu_velocities[:, 1]
    data_filtered["Velocity_Up"] = enu_velocities[:, 2]
    return data_filtered


def calculate_resultant_speed_direction(data_filtered, pos_range, neg_range):
    data_filtered = data_filtered.copy()
    data_filtered["Resultant_Speed"] = np.sqrt(
        data_filtered["Velocity_East"] ** 2 + data_filtered["Velocity_North"] ** 2
    )
    data_filtered["Direction"] = (
        np.degrees(np.arctan2(data_filtered["Velocity_North"], data_filtered["Velocity_East"])) + 360
    ) % 360
    data_filtered["Compass_Bearing_Flow"] = (90 - data_filtered["Direction"]) % 360

    def get_sign(angle):
        if angle_in_range(angle, neg_range[0], neg_range[1]):
            return -1
        if angle_in_range(angle, pos_range[0], pos_range[1]):
            return 1
        return 1

    signs = np.vectorize(get_sign)(data_filtered["Compass_Bearing_Flow"])
    data_filtered["Velocity_Channel"] = signs * data_filtered["Resultant_Speed"]

    if "Heading" in data_filtered.columns:
        data_filtered["Compass_Heading"] = pd.to_numeric(data_filtered["Heading"], errors="coerce") % 360

    return data_filtered


def reorder_columns(data_filtered):
    sen_columns = [column for column in SEN_MERGE_COLUMNS if column in data_filtered.columns]
    derived_columns = [
        "In_Water",
        "Velocity_East",
        "Velocity_North",
        "Velocity_Up",
        "Resultant_Speed",
        "Direction",
        "Compass_Bearing_Flow",
        "Compass_Heading",
        "Velocity_Channel",
    ]
    derived_columns = [column for column in derived_columns if column in data_filtered.columns]
    columns_order = ["Datetime"] + DAT_COLUMN_NAMES + sen_columns + derived_columns
    existing_columns = [column for column in columns_order if column in data_filtered.columns]
    return data_filtered[existing_columns]


def save_to_csv(dataframe, output_path, description):
    dataframe.to_csv(output_path, index=False)
    logging.info("%s opgeslagen in: %s", description, output_path)


def build_output_stem(dat_file_path, start_dt, end_dt):
    stem = Path(dat_file_path).stem
    time_label = f"{start_dt:%Y%m%d_%H%M%S}_{end_dt:%Y%m%d_%H%M%S}"
    return f"{stem}_{time_label}"


def configure_time_axis(axis):
    axis.grid(True)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))


def visualize_data(data_filtered, data_dir, output_stem):
    fig, axes = plt.subplots(8, 1, figsize=(16, 24), sharex=True, constrained_layout=True)

    axes[0].plot(data_filtered["Datetime"], data_filtered["Velocity_East"], color="tab:red")
    axes[0].set_title("Velocity East")
    axes[0].set_ylabel("m/s")

    axes[1].plot(data_filtered["Datetime"], data_filtered["Velocity_North"], color="tab:green")
    axes[1].set_title("Velocity North")
    axes[1].set_ylabel("m/s")

    axes[2].plot(data_filtered["Datetime"], data_filtered["Velocity_Up"], color="tab:blue")
    axes[2].set_title("Velocity Up")
    axes[2].set_ylabel("m/s")

    axes[3].plot(data_filtered["Datetime"], data_filtered["Resultant_Speed"], label="Resultant", color="tab:purple")
    axes[3].plot(data_filtered["Datetime"], data_filtered["Velocity_Channel"], label="Channel +/-", color="tab:cyan")
    axes[3].set_title("Resultant and Channel Velocity")
    axes[3].set_ylabel("m/s")
    axes[3].legend(loc="upper right")

    if "Pressure" in data_filtered.columns:
        axes[4].plot(data_filtered["Datetime"], data_filtered["Pressure"], color="tab:orange")
    axes[4].set_title("Pressure")
    axes[4].set_ylabel("Pressure")

    if "Heading" in data_filtered.columns:
        axes[5].plot(data_filtered["Datetime"], data_filtered["Heading"], label="Heading", color="tab:brown")
    if "Compass_Bearing_Flow" in data_filtered.columns:
        axes[5].plot(
            data_filtered["Datetime"],
            data_filtered["Compass_Bearing_Flow"],
            label="Flow bearing",
            color="tab:gray",
        )
    axes[5].set_title("Compass / Heading")
    axes[5].set_ylabel("deg")
    axes[5].legend(loc="upper right")

    if "Pitch" in data_filtered.columns:
        axes[6].plot(data_filtered["Datetime"], data_filtered["Pitch"], color="tab:pink")
    axes[6].set_title("Pitch")
    axes[6].set_ylabel("deg")

    if "Roll" in data_filtered.columns:
        axes[7].plot(data_filtered["Datetime"], data_filtered["Roll"], color="tab:olive")
    axes[7].set_title("Roll")
    axes[7].set_ylabel("deg")
    axes[7].set_xlabel("Datetime")

    for axis in axes:
        configure_time_axis(axis)

    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plot_path = Path(data_dir) / f"{output_stem}_timeseries.png"
    fig.savefig(plot_path, dpi=150)
    logging.info("Plots opgeslagen in: %s", plot_path)
    plt.show()


def visualize_speed_direction_rose(data_filtered, data_dir, output_stem):
    figure = plt.figure(figsize=(8, 8))
    axis = figure.add_subplot(111, projection="polar")
    axis.set_theta_zero_location("N")

    angles = np.deg2rad(data_filtered["Compass_Bearing_Flow"])
    speeds = data_filtered["Resultant_Speed"]
    scatter = axis.scatter(angles, speeds, c=speeds, cmap="viridis", alpha=0.75)
    axis.set_title("Snelheid-richting roos")
    figure.colorbar(scatter, label="Snelheid (m/s)")

    rose_plot_path = Path(data_dir) / f"{output_stem}_rose.png"
    figure.savefig(rose_plot_path, dpi=150)
    logging.info("Snelheid-richting-roos opgeslagen in: %s", rose_plot_path)
    plt.show()


def visualize_orientation_data(data_filtered, data_dir, output_stem):
    columns_to_plot = [column for column in ["Heading", "Pitch", "Roll", "Compass_Bearing_Flow"] if column in data_filtered.columns]
    if not columns_to_plot:
        return

    fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    if len(columns_to_plot) == 1:
        axes = [axes]

    colors = ["tab:brown", "tab:pink", "tab:olive", "tab:gray"]
    titles = {
        "Heading": "Heading",
        "Pitch": "Pitch",
        "Roll": "Roll",
        "Compass_Bearing_Flow": "Flow bearing",
    }

    for axis, column, color in zip(axes, columns_to_plot, colors):
        axis.plot(data_filtered["Datetime"], data_filtered[column], color=color)
        axis.set_title(titles[column])
        axis.set_ylabel("deg")
        configure_time_axis(axis)

    axes[-1].set_xlabel("Datetime")
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plot_path = Path(data_dir) / f"{output_stem}_orientation.png"
    fig.savefig(plot_path, dpi=150)
    logging.info("Orientatieplots opgeslagen in: %s", plot_path)
    plt.show()


def set_spinbox_value(widget, value):
    widget.delete(0, tk.END)
    widget.insert(0, f"{int(value):02d}")


def set_datetime_widgets(date_widget, hour_widget, minute_widget, second_widget, dt_value):
    date_widget.set_date(dt_value.date())
    set_spinbox_value(hour_widget, dt_value.hour)
    set_spinbox_value(minute_widget, dt_value.minute)
    set_spinbox_value(second_widget, dt_value.second)


def get_datetime_from_widgets(date_widget, hour_widget, minute_widget, second_widget):
    return datetime.strptime(
        f"{date_widget.get()} {int(hour_widget.get()):02d}:{int(minute_widget.get()):02d}:{int(second_widget.get()):02d}",
        "%Y-%m-%d %H:%M:%S",
    )


def get_user_inputs_via_gui():
    root = tk.Tk()
    root.title("Vector verwerking")

    loaded_state = {}
    user_inputs = {}

    selected_file_var = tk.StringVar()
    dat_file_var = tk.StringVar()
    sen_file_var = tk.StringVar()
    hdr_file_var = tk.StringVar()
    dataset_info_var = tk.StringVar(value="Nog geen dataset geladen.")
    water_info_var = tk.StringVar(value="Nog geen onderwaterinterval bepaald.")

    def update_loaded_labels():
        dat_label.config(text=Path(dat_file_var.get()).name if dat_file_var.get() else "Niet gevonden")
        sen_label.config(text=Path(sen_file_var.get()).name if sen_file_var.get() else "Niet gevonden")
        hdr_label.config(text=Path(hdr_file_var.get()).name if hdr_file_var.get() else "Niet gevonden")

    def preload_dataset():
        try:
            dataset = load_vector_dataset(dat_file_var.get(), sen_file_var.get(), hdr_file_var.get() or None)
            water_detection = detect_submerged_interval(dataset)
            dataset = dataset.copy()
            dataset["In_Water"] = water_detection["mask"].to_numpy()

            full_start = pd.to_datetime(dataset["Datetime"].min())
            full_end = pd.to_datetime(dataset["Datetime"].max())
            default_start = water_detection["start_dt"] if water_detection["detected"] else full_start
            default_end = water_detection["end_dt"] if water_detection["detected"] else full_end

            set_datetime_widgets(start_date, start_hour, start_minute, start_second, default_start)
            set_datetime_widgets(end_date, end_hour, end_minute, end_second, default_end)

            dataset_info_var.set(
                f"Volledig bereik: {full_start:%Y-%m-%d %H:%M:%S} t/m {full_end:%Y-%m-%d %H:%M:%S} | {len(dataset)} geldige records"
            )

            if water_detection["detected"]:
                water_info_var.set(
                    "Onderwater bereik: "
                    f"{water_detection['start_dt']:%Y-%m-%d %H:%M:%S} t/m {water_detection['end_dt']:%Y-%m-%d %H:%M:%S} "
                    f"| drukdrempel ~ {water_detection['threshold']:.3f}"
                )
            else:
                water_info_var.set(f"Onderwaterdetectie: {water_detection['reason']}")

            loaded_state["dataset"] = dataset
            loaded_state["water_detection"] = water_detection
        except Exception as exc:
            loaded_state.clear()
            messagebox.showerror("Fout", f"Fout bij laden van de dataset: {exc}")

    def select_vector_file():
        filename = filedialog.askopenfilename(
            title="Selecteer een Vector bestand",
            filetypes=[("Vector files", "*.dat *.sen *.hdr"), ("All files", "*.*")],
        )
        if not filename:
            return

        related_files = resolve_related_files(filename)
        missing_required = [key for key in ["dat_file", "sen_file"] if not related_files[key].exists()]
        if missing_required:
            readable_missing = ", ".join(item.replace("_file", "") for item in missing_required)
            messagebox.showerror(
                "Bestanden ontbreken",
                f"Automatisch gekoppelde bestanden ontbreken voor dezelfde basename: {readable_missing}.",
            )
            return

        selected_file_var.set(filename)
        dat_file_var.set(str(related_files["dat_file"]))
        sen_file_var.set(str(related_files["sen_file"]))
        hdr_file_var.set(str(related_files["hdr_file"]) if related_files["hdr_file"].exists() else "")

        update_loaded_labels()
        preload_dataset()

    ttk.Label(root, text="Kies 1 Vector bestand; .dat/.sen/.hdr met dezelfde naam worden automatisch gekoppeld.").grid(
        row=0, column=0, columnspan=4, padx=5, pady=(8, 5), sticky="w"
    )

    ttk.Button(root, text="Kies Vector bestand...", command=select_vector_file).grid(
        row=1, column=0, padx=5, pady=5, sticky="w"
    )
    ttk.Label(root, textvariable=selected_file_var, width=90).grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="w")

    ttk.Label(root, text=".dat:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
    dat_label = ttk.Label(root, text="Niet geladen")
    dat_label.grid(row=2, column=1, padx=5, pady=2, sticky="w")

    ttk.Label(root, text=".sen:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
    sen_label = ttk.Label(root, text="Niet geladen")
    sen_label.grid(row=3, column=1, padx=5, pady=2, sticky="w")

    ttk.Label(root, text=".hdr:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
    hdr_label = ttk.Label(root, text="Niet geladen")
    hdr_label.grid(row=4, column=1, padx=5, pady=2, sticky="w")

    ttk.Label(root, textvariable=dataset_info_var, wraplength=900).grid(
        row=5, column=0, columnspan=4, padx=5, pady=(8, 2), sticky="w"
    )
    ttk.Label(root, textvariable=water_info_var, wraplength=900).grid(
        row=6, column=0, columnspan=4, padx=5, pady=(0, 10), sticky="w"
    )

    ttk.Label(root, text="Startdatum:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
    start_date = DateEntry(root, date_pattern="yyyy-mm-dd")
    start_date.grid(row=7, column=1, padx=5, pady=5, sticky="w")

    ttk.Label(root, text="Starttijd (hh:mm:ss):").grid(row=8, column=0, padx=5, pady=5, sticky="w")
    start_hour = tk.Spinbox(root, from_=0, to=23, width=3, format="%02.0f")
    start_hour.grid(row=8, column=1, padx=(5, 0), pady=5, sticky="w")
    start_minute = tk.Spinbox(root, from_=0, to=59, width=3, format="%02.0f")
    start_minute.grid(row=8, column=1, padx=(40, 0), pady=5, sticky="w")
    start_second = tk.Spinbox(root, from_=0, to=59, width=3, format="%02.0f")
    start_second.grid(row=8, column=1, padx=(80, 0), pady=5, sticky="w")

    ttk.Label(root, text="Einddatum:").grid(row=9, column=0, padx=5, pady=5, sticky="w")
    end_date = DateEntry(root, date_pattern="yyyy-mm-dd")
    end_date.grid(row=9, column=1, padx=5, pady=5, sticky="w")

    ttk.Label(root, text="Eindtijd (hh:mm:ss):").grid(row=10, column=0, padx=5, pady=5, sticky="w")
    end_hour = tk.Spinbox(root, from_=0, to=23, width=3, format="%02.0f")
    end_hour.grid(row=10, column=1, padx=(5, 0), pady=5, sticky="w")
    end_minute = tk.Spinbox(root, from_=0, to=59, width=3, format="%02.0f")
    end_minute.grid(row=10, column=1, padx=(40, 0), pady=5, sticky="w")
    end_second = tk.Spinbox(root, from_=0, to=59, width=3, format="%02.0f")
    end_second.grid(row=10, column=1, padx=(80, 0), pady=5, sticky="w")

    ttk.Label(root, text="Positieve stroming (graden t.o.v. noorden):").grid(
        row=11, column=0, columnspan=4, padx=5, pady=(10, 2), sticky="w"
    )
    ttk.Label(root, text="Van:").grid(row=12, column=0, padx=5, pady=5, sticky="w")
    pos_min = tk.Spinbox(root, from_=0, to=359, width=5)
    pos_min.delete(0, tk.END)
    pos_min.insert(0, "246")
    pos_min.grid(row=12, column=1, padx=(5, 0), pady=5, sticky="w")
    ttk.Label(root, text="Tot:").grid(row=12, column=2, padx=5, pady=5, sticky="w")
    pos_max = tk.Spinbox(root, from_=0, to=359, width=5)
    pos_max.delete(0, tk.END)
    pos_max.insert(0, "67")
    pos_max.grid(row=12, column=3, padx=(5, 0), pady=5, sticky="w")

    ttk.Label(root, text="Negatieve stroming (graden t.o.v. noorden):").grid(
        row=13, column=0, columnspan=4, padx=5, pady=(10, 2), sticky="w"
    )
    ttk.Label(root, text="Van:").grid(row=14, column=0, padx=5, pady=5, sticky="w")
    neg_min = tk.Spinbox(root, from_=0, to=359, width=5)
    neg_min.delete(0, tk.END)
    neg_min.insert(0, "67")
    neg_min.grid(row=14, column=1, padx=(5, 0), pady=5, sticky="w")
    ttk.Label(root, text="Tot:").grid(row=14, column=2, padx=5, pady=5, sticky="w")
    neg_max = tk.Spinbox(root, from_=0, to=359, width=5)
    neg_max.delete(0, tk.END)
    neg_max.insert(0, "246")
    neg_max.grid(row=14, column=3, padx=(5, 0), pady=5, sticky="w")

    ttk.Label(root, text="Weergave type:").grid(row=15, column=0, padx=5, pady=(10, 2), sticky="w")
    plot_type = tk.StringVar(value="tijd")
    ttk.Radiobutton(root, text="Tijdgrafiek", variable=plot_type, value="tijd").grid(
        row=15, column=1, padx=5, pady=5, sticky="w"
    )
    ttk.Radiobutton(root, text="Roos", variable=plot_type, value="roos").grid(
        row=15, column=2, padx=5, pady=5, sticky="w"
    )

    def submit():
        try:
            if "dataset" not in loaded_state:
                messagebox.showerror("Fout", "Laad eerst een Vector dataset.")
                return
            if not dat_file_var.get() or not sen_file_var.get():
                messagebox.showerror("Fout", "De .dat en .sen bestanden zijn verplicht.")
                return
            if not hdr_file_var.get():
                messagebox.showerror("Fout", "Het .hdr bestand is nodig voor de transformatie matrix.")
                return

            start_dt = get_datetime_from_widgets(start_date, start_hour, start_minute, start_second)
            end_dt = get_datetime_from_widgets(end_date, end_hour, end_minute, end_second)
            if start_dt >= end_dt:
                messagebox.showerror("Fout", "De startdatum/tijd moet voor de einddatum/tijd liggen.")
                return

            user_inputs["start_dt"] = start_dt
            user_inputs["end_dt"] = end_dt
            user_inputs["dat_file"] = dat_file_var.get()
            user_inputs["sen_file"] = sen_file_var.get()
            user_inputs["hdr_file"] = hdr_file_var.get()
            user_inputs["pos_range"] = (float(pos_min.get()), float(pos_max.get()))
            user_inputs["neg_range"] = (float(neg_min.get()), float(neg_max.get()))
            user_inputs["plot_type"] = plot_type.get()
            user_inputs["dataset"] = loaded_state["dataset"]
            user_inputs["water_detection"] = loaded_state["water_detection"]
            root.destroy()
        except Exception as exc:
            messagebox.showerror("Fout", f"Er is een fout opgetreden: {exc}")

    ttk.Button(root, text="Start verwerking", command=submit).grid(row=16, column=0, columnspan=4, pady=15)

    root.mainloop()
    return user_inputs


def main():
    inputs = get_user_inputs_via_gui()
    if not inputs:
        logging.error("Niet alle vereiste inputs zijn verkregen. Script wordt beindigd.")
        return

    start_dt = inputs["start_dt"]
    end_dt = inputs["end_dt"]
    dat_file = inputs["dat_file"]
    hdr_file = inputs["hdr_file"]
    pos_range = inputs["pos_range"]
    neg_range = inputs["neg_range"]
    plot_type = inputs["plot_type"]

    logging.info("Geselecteerde startdatum en -tijd: %s", start_dt)
    logging.info("Geselecteerde einddatum en -tijd: %s", end_dt)
    logging.info("Positieve stroming: van %s deg tot %s deg", pos_range[0], pos_range[1])
    logging.info("Negatieve stroming: van %s deg tot %s deg", neg_range[0], neg_range[1])
    logging.info("Gekozen weergave: %s", plot_type)

    data_all = inputs["dataset"].copy()
    water_detection = inputs["water_detection"]

    if "In_Water" in data_all.columns:
        removed_count = int((~data_all["In_Water"]).sum())
        if removed_count > 0:
            logging.info("Automatisch %s records buiten water verwijderd.", removed_count)
            logging.info("Waterdetectie: %s", water_detection["reason"])

    mask = (data_all["Datetime"] >= start_dt) & (data_all["Datetime"] <= end_dt)
    if "In_Water" in data_all.columns:
        mask &= data_all["In_Water"]
    data_filtered = data_all.loc[mask].copy()

    if data_filtered.empty:
        logging.warning("Geen data beschikbaar na tijdselectie en luchtfilter.")
        messagebox.showwarning("Geen data", "Geen data beschikbaar na tijdselectie en luchtfilter.")
        return

    try:
        transformation_matrix = get_transformation_matrix_from_hdr(hdr_file)
    except Exception as exc:
        logging.error("Fout bij het inlezen van de transformatie matrix: %s", exc)
        messagebox.showerror("Fout", f"Fout bij het inlezen van de transformatie matrix: {exc}")
        return

    data_filtered = transform_velocities(data_filtered, transformation_matrix)
    data_filtered = calculate_resultant_speed_direction(data_filtered, pos_range, neg_range)
    data_filtered = reorder_columns(data_filtered)

    logging.info(
        "Eerste 5 rijen:\n%s",
        data_filtered[
            [
                column
                for column in [
                    "Datetime",
                    "Velocity_East",
                    "Velocity_North",
                    "Velocity_Up",
                    "Resultant_Speed",
                    "Compass_Bearing_Flow",
                    "Heading",
                    "Pitch",
                    "Roll",
                    "Velocity_Channel",
                ]
                if column in data_filtered.columns
            ]
        ].head(),
    )

    data_dir = Path(dat_file).parent
    output_stem = build_output_stem(dat_file, start_dt, end_dt)
    output_csv = data_dir / f"{output_stem}_ENU_filtered.csv"

    try:
        save_to_csv(data_filtered, output_csv, "Gefilterde ENU-snelheden met HRP en kompas")
    except Exception as exc:
        logging.error("Fout bij het opslaan van csv: %s", exc)
        messagebox.showerror("Fout", f"Fout bij het opslaan van csv: {exc}")
        return

    if plot_type == "tijd":
        visualize_data(data_filtered, data_dir, output_stem)
    else:
        visualize_speed_direction_rose(data_filtered, data_dir, output_stem)
        visualize_orientation_data(data_filtered, data_dir, output_stem)


if __name__ == "__main__":
    main()
