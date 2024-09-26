import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import serial.tools.list_ports
import threading
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import csv
import os
from queue import Queue, Empty
import re

matplotlib.use("TkAgg")


class SerialReader(threading.Thread):
    def __init__(self, port, baudrate, parity, stopbits, data_queue, log_queue, command_queue, stop_event, delimiter, decimal_sep, datetime_format, timeout=1):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.parity = parity
        self.stopbits = stopbits
        self.data_queue = data_queue
        self.log_queue = log_queue
        self.command_queue = command_queue  # Nieuwe commando wachtrij
        self.stop_event = stop_event
        self.serial_conn = None
        self.delimiter = delimiter
        self.decimal_sep = decimal_sep
        self.datetime_format = datetime_format
        self.timeout = timeout  # Configurable timeout

    def run(self):
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout
            )
            self.log_queue.put(f"Seriële verbinding geopend op {self.port} met baudrate {self.baudrate}")
            buffer = ""
            while not self.stop_event.is_set():
                # Verzend commando's indien beschikbaar
                try:
                    command = self.command_queue.get_nowait()
                    if command:
                        if command == 'ESC':
                            self.serial_conn.write(b'\x1b')  # ASCII ESC
                            self.log_queue.put("Verzonden: ESC")
                        elif command == 'BREAK':
                            self.serial_conn.send_break()
                            self.log_queue.put("Verzonden: BREAK")
                        else:
                            self.serial_conn.write((command + '\n').encode('ascii'))
                            self.log_queue.put(f"Verzonden: {command}")
                except Empty:
                    pass

                # Lees inkomende data
                if self.serial_conn.in_waiting:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('ascii', errors='ignore')
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self.log_queue.put(f"Ontvangen: {line}")
                            self.data_queue.put(line)
        except serial.SerialException as e:
            self.log_queue.put(f"Seriële fout: {e}")
        except UnicodeDecodeError as e:
            self.log_queue.put(f"Decodeerfout: {e}")
        except Exception as e:
            self.log_queue.put(f"Fout: {e}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                self.log_queue.put("Seriële verbinding gesloten.")


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RS232 Real-time Data Reader")
        self.geometry("1600x1000")  # Vergroot het venster voor extra componenten

        self.create_widgets()
        self.serial_thread = None
        self.data_queue = Queue()
        self.log_queue = Queue()
        self.command_queue = Queue()  # Nieuwe commando wachtrij
        self.stop_event = threading.Event()
        self.plot_data = {
            'time': [],
            'Temp (C)': [],
            'Cond (uS/cm)': [],
            'Depth (meters)': [],
            'pH': [],
            'Turbid+ (NTU)': [],
            'Battery (volts)': []
        }
        self.running = False

        # Tracking van after IDs
        self.after_ids = []

        # Start processing log messages
        self.after_ids.append(self.after(100, self.process_log_queue))

    def create_widgets(self):
        # Frame voor seriële instellingen
        settings_frame = ttk.LabelFrame(self, text="Seriële Instellingen")
        settings_frame.pack(fill="x", padx=10, pady=5)

        # Poort
        ttk.Label(settings_frame, text="Poort:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.port_var = tk.StringVar()
        self.port_cb = ttk.Combobox(settings_frame, textvariable=self.port_var, values=self.get_serial_ports(), state='readonly', width=30)
        self.port_cb.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Voeg de "Vernieuwen" knop toe voor dynamische detectie
        self.refresh_ports_btn = ttk.Button(settings_frame, text="Vernieuwen", command=self.refresh_serial_ports)
        self.refresh_ports_btn.grid(row=0, column=2, padx=5, pady=5)
        self.create_tooltip(self.refresh_ports_btn, "Vernieuw de lijst met beschikbare seriële poorten.")

        # Baudrate
        ttk.Label(settings_frame, text="Baudrate:").grid(row=0, column=3, padx=5, pady=5, sticky='e')
        self.baud_var = tk.StringVar(value="9600")
        self.baud_entry = ttk.Entry(settings_frame, textvariable=self.baud_var, width=15)
        self.baud_entry.grid(row=0, column=4, padx=5, pady=5, sticky='w')

        # Pariteit
        ttk.Label(settings_frame, text="Pariteit:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.parity_var = tk.StringVar(value="N")
        self.parity_cb = ttk.Combobox(settings_frame, textvariable=self.parity_var, values=["N", "E", "O", "M", "S"], state='readonly', width=30)
        self.parity_cb.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Stopbits
        ttk.Label(settings_frame, text="Stopbits:").grid(row=1, column=2, padx=5, pady=5, sticky='e')
        self.stopbits_var = tk.StringVar(value="1")
        self.stopbits_cb = ttk.Combobox(settings_frame, textvariable=self.stopbits_var, values=["1", "1.5", "2"], state='readonly', width=15)
        self.stopbits_cb.grid(row=1, column=3, padx=5, pady=5, sticky='w')

        # Frame voor Start, Stop en Cancel knoppen
        button_frame = ttk.Frame(settings_frame)
        button_frame.grid(row=2, column=0, columnspan=5, pady=10)

        self.start_btn = ttk.Button(button_frame, text="Start", command=self.start_reading)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_reading, state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        self.cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.cancel_program)
        self.cancel_btn.pack(side='left', padx=5)

        # Frame voor Parsing Instellingen
        parsing_frame = ttk.LabelFrame(self, text="Parsing Instellingen")
        parsing_frame.pack(fill="x", padx=10, pady=5)

        # Datum- en Tijdformaat
        ttk.Label(parsing_frame, text="Datum-Tijd Formaat:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.datetime_format_var = tk.StringVar(value="%d/%m/%Y %H:%M:%S")
        self.datetime_format_entry = ttk.Entry(parsing_frame, textvariable=self.datetime_format_var, width=30)
        self.datetime_format_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(parsing_frame, text='(bijv. "%d/%m/%Y %H:%M:%S")').grid(row=0, column=2, padx=5, pady=5, sticky='w')

        # Delimiter
        ttk.Label(parsing_frame, text="Delimiter:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.delimiter_var = tk.StringVar(value="space")
        self.delimiter_cb = ttk.Combobox(parsing_frame, textvariable=self.delimiter_var, values=["space", "comma", "dot", "semicolon", "tab"], state='readonly', width=30)
        self.delimiter_cb.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Decimaalteken
        ttk.Label(parsing_frame, text="Decimaalteken:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.decimal_sep_var = tk.StringVar(value="point")
        self.decimal_sep_cb = ttk.Combobox(parsing_frame, textvariable=self.decimal_sep_var, values=["comma", "point"], state='readonly', width=30)
        self.decimal_sep_cb.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        # Frame voor Opslaan Instellingen
        save_settings_frame = ttk.LabelFrame(self, text="Opslaan Instellingen")
        save_settings_frame.pack(fill="x", padx=10, pady=5)

        # Selectie van Directory
        ttk.Label(save_settings_frame, text="Opslaan in Directory:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.directory_var = tk.StringVar()
        self.directory_cb = ttk.Combobox(save_settings_frame, textvariable=self.directory_var, values=self.get_default_directories(), state='readonly', width=50)
        self.directory_cb.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.directory_cb.current(0)  # Selecteer standaard directory

        self.browse_dir_btn = ttk.Button(save_settings_frame, text="Bladeren", command=self.browse_directory)
        self.browse_dir_btn.grid(row=0, column=2, padx=5, pady=5)

        # Bestandsnaam
        ttk.Label(save_settings_frame, text="Bestandsnaam:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.filename_var = tk.StringVar()
        self.filename_entry = ttk.Entry(save_settings_frame, textvariable=self.filename_var, width=50)
        self.filename_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Opslaan knop
        self.save_btn = ttk.Button(save_settings_frame, text="Opslaan", command=self.save_data, state='disabled')
        self.save_btn.grid(row=1, column=2, padx=5, pady=5)

        # Voeg een nieuw frame toe voor commando's
        command_frame = ttk.LabelFrame(self, text="Commando's Verzenden")
        command_frame.pack(fill="x", padx=10, pady=5)

        # Invoerveld voor commando
        ttk.Label(command_frame, text="Commando:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.command_var = tk.StringVar()
        self.command_entry = ttk.Entry(command_frame, textvariable=self.command_var, width=50)
        self.command_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Verzendknop
        self.send_btn = ttk.Button(command_frame, text="Verzend", command=self.send_command, state='disabled')
        self.send_btn.grid(row=0, column=2, padx=5, pady=5)
        self.create_tooltip(self.send_btn, "Verzend het ingevoerde commando naar het instrument.")

        # **Toevoegen van ESC en BREAK knoppen**
        # Send ESC Knop
        self.send_esc_btn = ttk.Button(command_frame, text="Verzend ESC", command=self.send_esc, state='disabled')
        self.send_esc_btn.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.create_tooltip(self.send_esc_btn, "Verzend het ESC (Escape) commando naar het instrument.")

        # Send BREAK Knop
        self.send_break_btn = ttk.Button(command_frame, text="Verzend BREAK", command=self.send_break, state='disabled')
        self.send_break_btn.grid(row=1, column=2, padx=5, pady=5, sticky='w')
        self.create_tooltip(self.send_break_btn, "Verzend het BREAK commando naar het instrument.")

        # Frame voor Data Log
        log_frame = ttk.LabelFrame(self, text="Data Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_text = tk.Text(log_frame, height=15, state='disabled', wrap='word')
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Scrollbar voor Data Log
        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side='right', fill='y')

        # Grafiek
        plot_frame = ttk.LabelFrame(self, text="Real-time Grafiek")
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.ax.set_xlabel("Tijd (HH:MM:SS)")
        self.ax.set_ylabel("Waarden")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def get_serial_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def get_default_directories(self):
        # Definieer hier de standaard directories die je wilt aanbieden
        home = os.path.expanduser("~")
        default_dirs = [
            home,
            os.path.join(home, "Documents"),
            os.path.join(home, "Desktop")
        ]
        return default_dirs

    def browse_directory(self):
        selected_dir = filedialog.askdirectory(initialdir=self.directory_var.get() or os.path.expanduser("~"))
        if selected_dir:
            current_dirs = list(self.directory_cb['values'])
            if selected_dir not in current_dirs:
                current_dirs.append(selected_dir)
                self.directory_cb['values'] = current_dirs
            self.directory_var.set(selected_dir)

    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.config(state='normal')
                self.log_text.insert('end', message + "\n")
                self.log_text.see('end')
                self.log_text.config(state='disabled')
        except Empty:
            pass
        # Plan de volgende check alleen als de applicatie nog niet is gesloten
        if self.winfo_exists():
            after_id = self.after(100, self.process_log_queue)
            self.after_ids.append(after_id)

    def refresh_serial_ports(self):
        ports = self.get_serial_ports()
        self.port_cb['values'] = ports
        if ports:
            current_port = self.port_var.get()
            if current_port not in ports:
                self.port_var.set(ports[0])
        else:
            self.port_var.set('')
        self.log_queue.put("Lijst met seriële poorten vernieuwd.")

    def start_reading(self):
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("Geen Poort", "Selecteer een seriële poort.")
            return
        try:
            baudrate = int(self.baud_var.get())
            if baudrate not in [110, 300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]:
                messagebox.showwarning("Ongeldige Baudrate", "Voer een gangbare baudrate in.")
                return
        except ValueError:
            messagebox.showwarning("Ongeldige Baudrate", "Voer een geldige baudrate in.")
            return
        parity_str = self.parity_var.get()
        stopbits_str = self.stopbits_var.get()

        # Pariteit instellen
        parity = serial.PARITY_NONE
        if parity_str == "E":
            parity = serial.PARITY_EVEN
        elif parity_str == "O":
            parity = serial.PARITY_ODD
        elif parity_str == "M":
            parity = serial.PARITY_MARK
        elif parity_str == "S":
            parity = serial.PARITY_SPACE

        # Stopbits instellen
        stopbits = serial.STOPBITS_ONE
        if stopbits_str == "1.5":
            stopbits = serial.STOPBITS_ONE_POINT_FIVE
        elif stopbits_str == "2":
            stopbits = serial.STOPBITS_TWO

        # Parsing instellingen
        datetime_format = self.datetime_format_var.get()
        delimiter = self.delimiter_var.get()
        decimal_sep = self.decimal_sep_var.get()

        # Valideer het datetime_format
        try:
            # Gebruik een voorbeelddatum om het formaat te valideren
            datetime.strptime("26/09/2024 10:39:53", datetime_format)
        except ValueError as ve:
            messagebox.showerror("Ongeldig Formaat", f"Het opgegeven datum-tijd formaat is ongeldig:\n{ve}")
            return

        # Converteer delimiter naar daadwerkelijke scheidingsteken
        delimiter_mapping = {
            "space": " ",
            "comma": ",",
            "dot": ".",
            "semicolon": ";",
            "tab": "\t"
        }
        delimiter_char = delimiter_mapping.get(delimiter, " ")

        # Converteer decimal separator
        decimal_mapping = {
            "comma": ",",
            "point": "."
        }
        decimal_char = decimal_mapping.get(decimal_sep, ".")

        # Reset data en plot
        self.data_queue = Queue()
        self.plot_data = {
            'time': [],
            'Temp (C)': [],
            'Cond (uS/cm)': [],
            'Depth (meters)': [],
            'pH': [],
            'Turbid+ (NTU)': [],
            'Battery (volts)': []
        }
        self.ax.clear()
        self.ax.set_xlabel("Tijd (HH:MM:SS)")
        self.ax.set_ylabel("Waarden")
        self.ax.grid(True)
        self.canvas.draw()

        # Start de seriële thread
        self.stop_event.clear()
        self.serial_thread = SerialReader(
            port=port,
            baudrate=baudrate,
            parity=parity,
            stopbits=stopbits,
            data_queue=self.data_queue,
            log_queue=self.log_queue,
            command_queue=self.command_queue,  # Pass command_queue
            stop_event=self.stop_event,
            delimiter=delimiter_char,
            decimal_sep=decimal_char,
            datetime_format=datetime_format
        )
        self.serial_thread.start()

        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.save_btn.config(state='disabled')
        self.send_btn.config(state='normal')  # Activeer de verzendknop
        self.send_esc_btn.config(state='normal')  # Activeer de ESC knop
        self.send_break_btn.config(state='normal')  # Activeer de BREAK knop

        self.log_queue.put("Starten met het lezen van seriële data...")
        # Start de plot update
        after_id = self.after(100, self.update_plot)
        self.after_ids.append(after_id)

    def stop_reading(self):
        if self.serial_thread and self.serial_thread.is_alive():
            self.stop_event.set()
            self.serial_thread.join()

        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.save_btn.config(state='normal')
        self.send_btn.config(state='disabled')  # Deactiveer de verzendknop
        self.send_esc_btn.config(state='disabled')  # Deactiveer de ESC knop
        self.send_break_btn.config(state='disabled')  # Deactiveer de BREAK knop

        self.log_queue.put("Lezen van seriële data gestopt.")

    def send_command(self):
        command = self.command_var.get().strip()
        if not command:
            messagebox.showwarning("Geen Commando", "Voer een commando in om te verzenden.")
            return
        if not self.running:
            messagebox.showwarning("Niet Verbonden", "Start eerst de seriële verbinding voordat je commando's verstuurt.")
            return
        # Voeg het commando toe aan de command_queue
        self.command_queue.put(command)
        self.log_queue.put(f"Commando toegevoegd aan wachtrij: {command}")
        self.command_var.set("")  # Clear het invoerveld

    def send_esc(self):
        if not self.running:
            messagebox.showwarning("Niet Verbonden", "Start eerst de seriële verbinding voordat je ESC verzendt.")
            return
        # Voeg het ESC commando toe aan de command_queue
        self.command_queue.put('ESC')
        self.log_queue.put("ESC commando toegevoegd aan wachtrij.")
    
    def send_break(self):
        if not self.running:
            messagebox.showwarning("Niet Verbonden", "Start eerst de seriële verbinding voordat je BREAK verzendt.")
            return
        # Voeg het BREAK commando toe aan de command_queue
        self.command_queue.put('BREAK')
        self.log_queue.put("BREAK commando toegevoegd aan wachtrij.")

    def cancel_program(self):
        if self.running:
            self.stop_reading()
        # Annuleer alle geplande after callbacks
        for after_id in self.after_ids:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
        self.destroy()

    def update_plot(self):
        if not self.running:
            return

        while not self.data_queue.empty():
            line = self.data_queue.get()
            parsed = self.parse_line(line)
            if parsed:
                timestamp, data_values = parsed
                self.plot_data['time'].append(timestamp)
                self.plot_data['Temp (C)'].append(data_values.get('Temp (C)', 0))
                self.plot_data['Cond (uS/cm)'].append(data_values.get('Cond (uS/cm)', 0))
                self.plot_data['Depth (meters)'].append(data_values.get('Depth (meters)', 0))
                self.plot_data['pH'].append(data_values.get('pH', 0))
                self.plot_data['Turbid+ (NTU)'].append(data_values.get('Turbid+ (NTU)', 0))
                self.plot_data['Battery (volts)'].append(data_values.get('Battery (volts)', 0))

        self.ax.clear()
        self.ax.set_xlabel("Tijd (HH:MM:SS)")
        self.ax.set_ylabel("Waarden")
        self.ax.grid(True)

        if self.plot_data['time']:
            # Limiteer het aantal weergegeven punten om performance te verbeteren
            max_points = 100
            times = [t.strftime("%H:%M:%S") for t in self.plot_data['time'][-max_points:]]
            self.ax.plot(times, self.plot_data['Temp (C)'][-max_points:], label='Temp (C)')
            self.ax.plot(times, self.plot_data['Cond (uS/cm)'][-max_points:], label='Cond (uS/cm)')
            self.ax.plot(times, self.plot_data['Depth (meters)'][-max_points:], label='Depth (meters)')
            self.ax.plot(times, self.plot_data['pH'][-max_points:], label='pH')
            self.ax.plot(times, self.plot_data['Turbid+ (NTU)'][-max_points:], label='Turbid+ (NTU)')
            self.ax.plot(times, self.plot_data['Battery (volts)'][-max_points:], label='Battery (volts)')
            self.ax.legend(loc='upper left', fontsize='small', ncol=2)
            self.fig.autofmt_xdate()

        self.canvas.draw()
        # Plan de volgende update alleen als de applicatie nog niet is gestopt
        if self.running:
            after_id = self.after(100, self.update_plot)
            self.after_ids.append(after_id)

    def parse_line(self, line):
        try:
            # Splits op het geselecteerde delimiter
            if self.serial_thread.delimiter == " ":
                parts = line.split()  # Splits op elke hoeveelheid whitespace
            else:
                parts = line.split(self.serial_thread.delimiter)

            # Controleer of er voldoende onderdelen zijn
            if len(parts) < 8:
                self.log_queue.put(f"Onvoldoende gegevens in lijn: {line}")
                return None

            # Combineer de eerste twee onderdelen voor datetime_str
            datetime_str = f"{parts[0].strip()} {parts[1].strip()}"
            data_strs = parts[2:8]  # Temp, Cond, Depth, pH, Turbid+, Battery

            # Parse datetime
            timestamp = datetime.strptime(datetime_str, self.serial_thread.datetime_format)

            # Parse data waarden
            data_values = {}
            labels = ['Temp (C)', 'Cond (uS/cm)', 'Depth (meters)', 'pH', 'Turbid+ (NTU)', 'Battery (volts)']
            for i, label in enumerate(labels):
                data = data_strs[i].strip()
                if self.serial_thread.decimal_sep == ",":
                    data = data.replace(',', '.')
                data_values[label] = float(data) if data else 0.0

            return timestamp, data_values
        except Exception as e:
            self.log_queue.put(f"Parseerfout: {e} in lijn: {line}")
            return None

    def save_data(self):
        if not self.plot_data.get('time'):
            messagebox.showwarning("Geen Data", "Er is geen data om op te slaan.")
            return

        # Haal directory en filename op
        directory = self.directory_var.get()
        filename = self.filename_var.get().strip()

        if not directory:
            messagebox.showwarning("Geen Directory", "Selecteer een directory om de data op te slaan.")
            return

        if not filename:
            messagebox.showwarning("Geen Bestandsnaam", "Voer een bestandsnaam in om de data op te slaan.")
            return

        # Voeg .csv extensie toe indien niet aanwezig
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        file_path = os.path.join(directory, filename)

        # Controleer of het bestand al bestaat
        if os.path.exists(file_path):
            overwrite = messagebox.askyesno("Bestand Bestaat", f"Het bestand {filename} bestaat al. Wil je het overschrijven?")
            if not overwrite:
                return

        # Valideer de bestandsnaam
        if not self.is_valid_filename(filename):
            messagebox.showerror("Ongeldige Bestandsnaam", "De bestandsnaam bevat ongeldige tekens.")
            return

        try:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                # Schrijf header
                headers = ["Tijd (d/m/y H:M:S)", "Temp (C)", "Cond (uS/cm)", "Depth (meters)", "pH", "Turbid+ (NTU)", "Battery (volts)"]
                writer.writerow(headers)
                # Schrijf data
                for i in range(len(self.plot_data['time'])):
                    row = [
                        self.plot_data['time'][i].strftime(self.serial_thread.datetime_format),
                        self.plot_data['Temp (C)'][i],
                        self.plot_data['Cond (uS/cm)'][i],
                        self.plot_data['Depth (meters)'][i],
                        self.plot_data['pH'][i],
                        self.plot_data['Turbid+ (NTU)'][i],
                        self.plot_data['Battery (volts)'][i]
                    ]
                    writer.writerow(row)
            messagebox.showinfo("Succes", f"Data succesvol opgeslagen naar {file_path}")
            self.log_queue.put(f"Data opgeslagen naar {file_path}")
        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij opslaan: {e}")

    def is_valid_filename(self, filename):
        # Windows verboden tekens: \ / : * ? " < > |
        # UNIX heeft minder beperkingen, maar het is goed om deze te controleren
        return not re.search(r'[\\/:*?"<>|]', filename)

    def create_tooltip(self, widget, text):
        tooltip = Tooltip(widget, text)


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip or not self.text:
            return
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


if __name__ == "__main__":
    app = Application()
    app.mainloop()
