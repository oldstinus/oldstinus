import os
import struct
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import logging
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeConverter:
    @staticmethod
    def convert_time(seconds_since_base):
        base_date = datetime(1984, 3, 1)
        try:
            # Vectorize the conversion to handle arrays
            vectorized_convert = np.vectorize(
                lambda sec: (base_date + timedelta(seconds=sec)).strftime("%d/%m/%y %H:%M:%S")
            )
            return vectorized_convert(seconds_since_base)
        except Exception as e:
            logging.error(f"Time conversion error: {e}")
            # Return an array of "Invalid Time" strings matching the input shape
            if isinstance(seconds_since_base, np.ndarray):
                return np.full_like(seconds_since_base, "Invalid Time", dtype=object)
            else:
                return "Invalid Time"

def YSI6SeriesParse(filename, mode):
    """
    YSI6SeriesParse Parser for YSI 6 series MultiParameter data logger files.

    Parameters:
        filename (str): Path to the input .DAT file.
        mode (str): Toolbox data type mode.

    Returns:
        dict: sample_data containing sample data.
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string.")

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} does not exist")

    # Read the entire file as bytes
    try:
        with open(filename, 'rb') as f:
            data = f.read()
    except Exception as e:
        raise IOError(f"Error reading file {filename}: {e}")

    # Read the record format from the header
    header = readHeader(data)

    # Parse all of the records
    records = readRecords(header, data)

    # Initialize sample_data dictionary
    sample_data = {
        'toolbox_input_file': filename,
        'meta': {
            'instrument_make': 'YSI',
            'instrument_model': '6 Series',
            'instrument_serial_no': '',
            'instrument_sample_interval': np.median(np.diff(records['time']) * 24 * 3600),
            'featureType': mode
        },
        'dimensions': [],
        'variables': []
    }

    # Define dimensions
    # Assuming netcdf3ToMatlabType and imosParameters are implemented elsewhere
    time_type = netcdf3ToMatlabType(imosParameters('TIME', 'type'))
    sample_data['dimensions'].append({
        'name': 'TIME',
        'typeCastFunc': float,
        'data': records['time'].astype(float)  # Store as array of floats
    })

    # Define variables with initial values
    variables_info = [
        {'name': 'TIMESERIES', 'data': 1, 'dimensions': []},
        {'name': 'LATITUDE', 'data': np.nan, 'dimensions': []},
        {'name': 'LONGITUDE', 'data': np.nan, 'dimensions': []},
        {'name': 'NOMINAL_DEPTH', 'data': np.nan, 'dimensions': []}
    ]

    for var in variables_info:
        var_type = netcdf3ToMatlabType(imosParameters(var['name'], 'type'))
        var_entry = {
            'name': var['name'],
            'typeCastFunc': eval(var_type),
            'data': eval(var_type)(var['data']),
            'dimensions': var['dimensions']
        }
        sample_data['variables'].append(var_entry)

    # Define fields and coordinates
    fields = [key for key in records.keys() if key != 'time']
    coordinates = 'TIME LATITUDE LONGITUDE NOMINAL_DEPTH'

    # Initialize variables list
    for field in fields:
        field_data = records[field]

        # Determine dimensions
        if field.lower() in ['latitude', 'longitude']:
            dimensions = []
        else:
            dimensions = [0]  # Assuming 'TIME' is the first dimension

        var_entry = {}
        var_entry['dimensions'] = dimensions

        # Map MATLAB fields to variable names and perform necessary conversions
        if field == 'temperature':
            var_entry['name'] = 'TEMP'
            var_entry['data'] = imos_cast(field_data, 'TEMP')
        elif field == 'cond':
            var_entry['name'] = 'CNDC'
            var_entry['data'] = imos_cast(field_data / 10.0, 'CNDC')
        elif field == 'spcond':
            var_entry['name'] = 'SPEC_CNDC'
            var_entry['data'] = imos_cast(field_data / 10.0, 'SPEC_CNDC')
        elif field == 'tds':
            var_entry['name'] = 'TDS'
            var_entry['data'] = imos_cast(field_data, 'TDS')
        elif field == 'salinity':
            var_entry['name'] = 'PSAL'
            var_entry['data'] = imos_cast(field_data, 'PSAL')
        elif field == 'ph':
            var_entry['name'] = 'ACID'
            var_entry['data'] = imos_cast(field_data, 'ACID')
        elif field == 'orp':
            var_entry['name'] = 'ORP'
            var_entry['data'] = imos_cast(field_data, 'ORP')
        elif field == 'depth':
            var_entry['name'] = 'DEPTH'
            var_entry['data'] = imos_cast(field_data, 'DEPTH')
        elif field == 'bp':
            var_entry['name'] = 'PRES'
            var_entry['data'] = imos_cast(field_data / 1.45037738, 'PRES')
        elif field == 'battery':
            var_entry['name'] = 'BAT_VOLT'
            var_entry['data'] = imos_cast(field_data, 'BAT_VOLT')
        elif field == 'chlorophyll':
            var_entry['name'] = 'CPHL'
            var_entry['data'] = imos_cast(field_data, 'CPHL')
            var_entry['comment'] = getCPHLcomment('unknown', '470nm', 'above 630nm')
        elif field == 'latitude':
            sample_data['variables'][1]['data'] = imos_cast(field_data, 'LATITUDE')
            continue  # Already handled
        elif field == 'longitude':
            sample_data['variables'][2]['data'] = imos_cast(field_data, 'LONGITUDE')
            continue  # Already handled
        elif field == 'turbidity':
            var_entry['name'] = 'TURB'
            var_entry['data'] = imos_cast(field_data, 'TURB')
            var_entry['comment'] = 'Turbidity from 6136 sensor.'
        elif field == 'odo':
            var_entry['name'] = 'DOXS'
            var_entry['data'] = imos_cast(field_data, 'DOXS')
            var_entry['comment'] = 'Dissolved oxygen saturation from ROX optical sensor.'
        elif field == 'odo2':
            var_entry['name'] = 'DOXY'
            var_entry['data'] = imos_cast(field_data, 'DOXY')
            var_entry['comment'] = 'Dissolved oxygen from ROX optical sensor.'
        else:
            # Handle unexpected fields if necessary
            continue

        var_entry['coordinates'] = coordinates
        var_entry['typeCastFunc'] = netcdf3ToMatlabType(imosParameters(var_entry['name'], 'type'))

        sample_data['variables'].append(var_entry)

    return sample_data

def readHeader(data):
    """
    Reads the file header and returns the header information in a dictionary.

    Parameters:
        data (bytes): The binary data from the file.

    Returns:
        dict: header information including record format, start, and length.
    """
    header = {
        'recordFmt': []
    }

    # Find the first record sync byte (0x42)
    try:
        idx = data.index(66)  # 66 is ASCII for 'B'
    except ValueError:
        raise ValueError("Sync byte 0x42 not found in data.")

    while True:
        entry = data[idx:idx+15]
        if len(entry) < 15:
            break  # Not enough data for another entry

        if entry[0] != 66:  # 0x42
            break

        header['recordFmt'].append(entry[3])  # MATLAB is 1-based
        idx += 15

    header['recordStart'] = idx
    header['recordLength'] = 1 + (len(header['recordFmt']) + 1) * 4

    return header

def readRecords(header, data):
    """
    Reads the records contained in the given bytes.

    Parameters:
        header (dict): Header information including record format and length.
        data (bytes): The binary data from the file.

    Returns:
        dict: Parsed records with time and various sensor measurements.
    """
    records = {}
    record_length = header['recordLength']
    record_start = header['recordStart']
    record_fmt = header['recordFmt']
    rNum = 0

    # Determine endianness
    cpu_endianness = sys.byteorder  # 'little' or 'big'

    # Slice the data to start reading records
    data = data[record_start:]

    while len(data) >= record_length:
        record = data[:record_length]
        data = data[record_length:]
        rNum += 1

        if record[0] != 68:  # 0x44
            # Missing sync byte; find the next sync byte (0x44)
            try:
                next_sync = record.index(68, 1)
                data = record[next_sync:] + data
            except ValueError:
                # No sync byte found in the current record
                continue
            continue

        # Unpack time (bytes 1-4)
        time_bytes = record[1:5]
        time_val = byte_cast(time_bytes, 'L', 'I', cpu_endianness)
        records.setdefault('time', []).append(time_val)

        # Unpack the rest of the record as single precision floats
        num_floats = (len(record) - 5) // 4
        vals = byte_cast(record[5:], 'L', 'f', cpu_endianness, count=num_floats)

        # Assign values based on record format
        for fmt, val in zip(record_fmt, vals):
            if fmt == 1:
                records.setdefault('temperature', []).append(val)
            elif fmt == 4:
                records.setdefault('cond', []).append(val)
            elif fmt == 6:
                records.setdefault('spcond', []).append(val)
            elif fmt == 10:
                records.setdefault('tds', []).append(val)
            elif fmt == 12:
                records.setdefault('salinity', []).append(val)
            elif fmt == 18:
                records.setdefault('ph', []).append(val)
            elif fmt == 19:
                records.setdefault('orp', []).append(val)
            elif fmt == 22:
                records.setdefault('depth', []).append(val)
            elif fmt == 24:
                records.setdefault('bp', []).append(val)
            elif fmt == 28:
                records.setdefault('battery', []).append(val)
            elif fmt == 193:
                records.setdefault('chlorophyll', []).append(val)
            elif fmt == 196:
                records.setdefault('latitude', []).append(val)
            elif fmt == 197:
                records.setdefault('longitude', []).append(val)
            elif fmt == 203:
                records.setdefault('turbidity', []).append(val)
            elif fmt == 211:
                records.setdefault('odo', []).append(val)
            elif fmt == 212:
                records.setdefault('odo2', []).append(val)
            # Add more cases if necessary

    # Convert lists to numpy arrays for efficiency
    for key in records:
        records[key] = np.array(records[key])

    return records

def byte_cast(byte_data, endian_indicator, data_type, cpu_endianness, count=1):
    """
    Casts byte data to the specified data type considering endianness.

    Parameters:
        byte_data (bytes): The bytes to cast.
        endian_indicator (str): 'L' for little endian, 'B' for big endian.
        data_type (str): The format character for struct.unpack.
        cpu_endianness (str): 'little' or 'big'.
        count (int): Number of items to unpack.

    Returns:
        int or float or list: The unpacked value(s).
    """
    if endian_indicator == 'L':
        endian = '<'  # Little endian
    else:
        endian = '>'  # Big endian

    format_str = endian + data_type * count
    try:
        unpacked = struct.unpack(format_str, byte_data)
    except struct.error as e:
        raise ValueError(f"Error unpacking data: {e}")

    if count == 1:
        return unpacked[0]
    return unpacked

def netcdf3ToMatlabType(netcdf_type):
    """
    Converts NetCDF3 type to a corresponding Python type casting function.

    Parameters:
        netcdf_type (str): The NetCDF3 type as a string.

    Returns:
        str: A string representing the Python type casting function.
    """
    type_mapping = {
        'float': 'float',
        'double': 'float',
        'int': 'int',
        'short': 'int',
        'byte': 'int',
        'char': 'str',
        # Add more mappings as needed
    }
    return type_mapping.get(netcdf_type.lower(), 'float')  # Default to float

def imosParameters(variable_name, parameter):
    """
    Retrieves IMOS parameters based on variable name and parameter type.

    Parameters:
        variable_name (str): The name of the variable.
        parameter (str): The parameter type (e.g., 'type').

    Returns:
        str: The parameter value.
    """
    # This is a stub. Replace with actual implementation.
    # Example:
    imos_params = {
        'TIME': {'type': 'float'},
        'TIMESERIES': {'type': 'float'},
        'LATITUDE': {'type': 'float'},
        'LONGITUDE': {'type': 'float'},
        'NOMINAL_DEPTH': {'type': 'float'},
        'TEMP': {'type': 'float'},
        'CNDC': {'type': 'float'},
        'SPEC_CNDC': {'type': 'float'},
        'TDS': {'type': 'float'},
        'PSAL': {'type': 'float'},
        'ACID': {'type': 'float'},
        'ORP': {'type': 'float'},
        'DEPTH': {'type': 'float'},
        'PRES': {'type': 'float'},
        'BAT_VOLT': {'type': 'float'},
        'CPHL': {'type': 'float'},
        'TURB': {'type': 'float'},
        'DOXS': {'type': 'float'},
        'DOXY': {'type': 'float'},
        # Add more as needed
    }
    return imos_params.get(variable_name.upper(), {}).get(parameter.lower(), 'float')

def getCPHLcomment(param1, param2, param3):
    """
    Generates a comment for chlorophyll data.

    Parameters:
        param1 (str): First parameter (e.g., 'unknown').
        param2 (str): Second parameter (e.g., '470nm').
        param3 (str): Third parameter (e.g., 'above 630nm').

    Returns:
        str: The generated comment.
    """
    # This is a stub. Replace with actual implementation if needed.
    return f"Chlorophyll measurement parameters: {param1}, {param2}, {param3}."

def imos_cast(data, variable_name):
    """
    Casts data based on the IMOS parameter type.

    Parameters:
        data (numpy.ndarray): The data to cast.
        variable_name (str): The name of the variable.

    Returns:
        numpy.ndarray or other: The casted data.
    """
    cast_type = netcdf3ToMatlabType(imosParameters(variable_name, 'type'))
    if cast_type == 'float':
        return data.astype(float)
    elif cast_type == 'int':
        return data.astype(int)
    elif cast_type == 'str':
        return data.astype(str)
    else:
        return data  # Default case

# ------------------- GUI Implementation -------------------

class YSIParserGUI:
    def __init__(self, master):
        self.master = master
        master.title("YSI 6 Series .DAT Parser")

        # Configure grid layout
        master.columnconfigure(1, weight=1)

        # File selection
        self.file_label = ttk.Label(master, text="Selected File:")
        self.file_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(master, textvariable=self.file_path, width=50, state='readonly')
        self.file_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.EW)

        self.browse_button = ttk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        # Mode selection
        self.mode_label = ttk.Label(master, text="Mode:")
        self.mode_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

        self.mode_var = tk.StringVar()
        self.mode_combobox = ttk.Combobox(master, textvariable=self.mode_var, state='readonly')
        self.mode_combobox['values'] = ('Mode1', 'Mode2', 'Mode3')  # Replace with actual modes
        self.mode_combobox.current(0)
        self.mode_combobox.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)

        # Parse button
        self.parse_button = ttk.Button(master, text="Parse and Export CSV", command=self.parse_and_export)
        self.parse_button.grid(row=2, column=1, padx=10, pady=20)

        # Status label
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(master, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky=tk.W)

    def browse_file(self):
        """Opens a file dialog for the user to select a .DAT file."""
        filetypes = (("DAT files", "*.dat"), ("All files", "*.*"))
        filename = filedialog.askopenfilename(title="Open DAT file", filetypes=filetypes)
        if filename:
            self.file_path.set(filename)
            self.status_var.set("File selected.")

    def parse_and_export(self):
        """Parses the selected file and exports the data to a CSV."""
        filename = self.file_path.get()
        mode = self.mode_var.get()

        if not filename:
            messagebox.showerror("Error", "Please select a .DAT file to parse.")
            return

        try:
            self.status_var.set("Parsing the file...")
            self.master.update_idletasks()
            sample_data = YSI6SeriesParse(filename, mode)
            self.status_var.set("Parsing completed. Exporting to CSV...")

            # Convert sample_data to DataFrame
            df = self.sample_data_to_dataframe(sample_data)

            # Ask user where to save the CSV
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save CSV file"
            )

            if save_path:
                df.to_csv(save_path, index=False)
                self.status_var.set(f"CSV exported successfully to {save_path}.")
                messagebox.showinfo("Success", f"CSV exported successfully to {save_path}.")
            else:
                self.status_var.set("CSV export canceled.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_var.set("Error occurred during parsing.")

    def sample_data_to_dataframe(self, sample_data):
        """
        Converts the sample_data dictionary to a pandas DataFrame.

        Parameters:
            sample_data (dict): The parsed sample data.

        Returns:
            pandas.DataFrame: The resulting DataFrame.
        """
        # Initialize a dictionary to hold DataFrame columns
        data_dict = {}

        # Determine the number of records from 'TIME' dimension
        if sample_data['dimensions']:
            num_records = len(sample_data['dimensions'][0]['data'])
        else:
            num_records = 0
        logging.info(f"Number of records to write: {num_records}")

        # Process dimensions (e.g., TIME)
        for dim in sample_data['dimensions']:
            if dim['name'].upper() == 'TIME':
                # Convert seconds to datetime strings using the convert_time method
                time_seconds = dim['data']
                data_dict['TIME'] = TimeConverter.convert_time(time_seconds)
                logging.debug(f"TIME converted: {data_dict['TIME'][:5]}")  # Show first 5 entries

        # Process variables
        for var in sample_data['variables']:
            var_name = var['name']
            var_data = var['data']
            if isinstance(var_data, np.ndarray):
                data_dict[var_name] = var_data
                logging.debug(f"Variable '{var_name}' added with shape {var_data.shape}")
            else:
                # If var_data is a scalar, replicate it to match the length of TIME
                data_length = len(data_dict['TIME']) if 'TIME' in data_dict else 1
                data_dict[var_name] = [var_data] * data_length
                logging.debug(f"Variable '{var_name}' added as scalar replicated to length {data_length}")

        # Create DataFrame
        df = pd.DataFrame(data_dict)
        logging.debug(f"DataFrame created with columns: {df.columns}")

        return df

# ------------------- Main Execution -------------------

def main():
    root = tk.Tk()
    app = YSIParserGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
