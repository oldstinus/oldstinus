import tkinter as tk
from tkinter import ttk, messagebox
import serial
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class GPSLoggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aaronia GPS Logger Communicatie en Data Visualisatie")

        # Seriële communicatie parameters
        self.serial_port = None
        self.baudrate = tk.StringVar(value="625000")  # Aangepaste baudrate voor de GPS Logger
        self.port = tk.StringVar(value="COM1")  # Standaard poort
        self.timeout = tk.IntVar(value=1)

        # Maak de GUI layout
        self.create_gui()

    def create_gui(self):
        # Frame voor seriële communicatie parameters
        param_frame = ttk.LabelFrame(self.root, text="RS232 Communicatie Instellingen", padding="10")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(param_frame, text="Poort:").grid(row=0, column=0, sticky=tk.W)
        self.port_entry = ttk.Entry(param_frame, textvariable=self.port, width=15)
        self.port_entry.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(param_frame, text="Baudrate:").grid(row=1, column=0, sticky=tk.W)
        self.baudrate_entry = ttk.Entry(param_frame, textvariable=self.baudrate, width=15)
        self.baudrate_entry.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(param_frame, text="Timeout:").grid(row=2, column=0, sticky=tk.W)
        self.timeout_entry = ttk.Entry(param_frame, textvariable=self.timeout, width=15)
        self.timeout_entry.grid(row=2, column=1, sticky=tk.W)

        # Open seriële verbinding knop
        self.open_button = ttk.Button(param_frame, text="Open Verbinding", command=self.open_connection)
        self.open_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Frame voor terminal commando's
        terminal_frame = ttk.LabelFrame(self.root, text="Commando Terminal", padding="10")
        terminal_frame.grid(row=1, column=0, padx=10, pady=10, sticky=(tk.W, tk.E))

        self.command_entry = ttk.Entry(terminal_frame, width=40)
        self.command_entry.grid(row=0, column=0, padx=5, pady=5)
        self.command_button = ttk.Button(terminal_frame, text="Verstuur Commando", command=self.send_command)
        self.command_button.grid(row=0, column=1, padx=5, pady=5)

        # Frame voor data visualisatie
        plot_frame = ttk.LabelFrame(self.root, text="Data Visualisatie", padding="10")
        plot_frame.grid(row=2, column=0, padx=10, pady=10, sticky=(tk.W, tk.E))

        # Matplotlib figure en canvas voor visualisatie
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("GPS Logger Data Visualisatie")
        self.ax.set_xlabel("Tijd (s)")
        self.ax.set_ylabel("Sensor Waarde")
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)

    def open_connection(self):
        """Open de RS232 seriële verbinding."""
        try:
            self.serial_port = serial.Serial(
                port=self.port.get(),
                baudrate=self.baudrate.get(),
                timeout=self.timeout.get(),
                bytesize=8,
                stopbits=2,
                parity='N',
                xonxoff=0,
                rtscts=0
            )
            messagebox.showinfo("Info", f"Verbonden met {self.port.get()} op {self.baudrate.get()} baud.")
            self.start_reading()
        except Exception as e:
            messagebox.showerror("Error", f"Verbinding mislukt: {e}")

    def close_connection(self):
        """Sluit de seriële verbinding."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            messagebox.showinfo("Info", "Verbinding gesloten.")

    def send_command(self):
        """Verstuur een commando naar de GPS Logger."""
        if self.serial_port and self.serial_port.is_open:
            command = self.command_entry.get()
            self.serial_port.write(command.encode())
            self.command_entry.delete(0, tk.END)  # Maak het invoerveld leeg
            messagebox.showinfo("Info", f"Commando '{command}' verstuurd.")
        else:
            messagebox.showerror("Error", "Verbinding is niet geopend.")

    def start_reading(self):
        """Start een nieuwe thread om gegevens van de seriële poort te lezen."""
        self.data = []
        self.time_stamps = []
        self.start_time = time.time()
        self.read_thread = threading.Thread(target=self.read_data)
        self.read_thread.daemon = True
        self.read_thread.start()

    def read_data(self):
        """Lees data van de seriële poort en werk de grafiek bij."""
        while True:
            if self.serial_port and self.serial_port.is_open:
                data = self.serial_port.readline()
                if data:
                    # Verwerk de ontvangen data
                    decoded_data = data.decode().strip()
                    timestamp = time.time() - self.start_time
                    self.data.append(float(decoded_data))  # Voorbeeld: verwacht numerieke data
                    self.time_stamps.append(timestamp)
                    self.update_plot()
            time.sleep(0.1)

    def update_plot(self):
        """Werk de grafiek bij met nieuwe data."""
        if self.data:
            self.ax.clear()
            self.ax.plot(self.time_stamps, self.data, label="Ontvangen Data")
            self.ax.set_title("GPS Logger Data Visualisatie")
            self.ax.set_xlabel("Tijd (s)")
            self.ax.set_ylabel("Sensor Waarde")
            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = GPSLoggerApp(root)
    root.mainloop()
