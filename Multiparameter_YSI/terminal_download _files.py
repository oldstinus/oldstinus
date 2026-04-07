import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import serial
import serial.tools.list_ports
import threading
import time

class SerialTerminalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YSI Multiparameter Communicator")
        self.geometry("800x600")
        self.serial_port = None
        self.running = False
        self.create_widgets()
    
    def create_widgets(self):
        # Frame voor verbindingsinstellingen
        conn_frame = ttk.LabelFrame(self, text="Verbindingsinstellingen")
        conn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(conn_frame, text="COM-poort:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.com_port_var = tk.StringVar()
        self.com_port_combo = ttk.Combobox(conn_frame, textvariable=self.com_port_var, width=15)
        self.com_port_combo.grid(row=0, column=1, padx=5, pady=5)
        self.refresh_com_ports()
        
        ttk.Label(conn_frame, text="Baudrate:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.baudrate_var = tk.StringVar(value="9600")
        self.baudrate_combo = ttk.Combobox(conn_frame, textvariable=self.baudrate_var, 
                                           values=["300", "600", "1200", "2400", "4800", "9600", "19200", "38400", "57600", "115200"],
                                           width=10)
        self.baudrate_combo.grid(row=0, column=3, padx=5, pady=5)
        
        self.connect_button = ttk.Button(conn_frame, text="Verbinden", command=self.connect_serial)
        self.connect_button.grid(row=0, column=4, padx=5, pady=5)
        
        # Terminal display
        terminal_frame = ttk.LabelFrame(self, text="Terminal")
        terminal_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.terminal_text = scrolledtext.ScrolledText(terminal_frame, state="disabled", wrap="word")
        self.terminal_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Command invoer en verstuur knop
        cmd_frame = ttk.Frame(self)
        cmd_frame.pack(fill="x", padx=5, pady=5)
        self.cmd_entry = ttk.Entry(cmd_frame)
        self.cmd_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.cmd_entry.bind("<Return>", self.send_command)
        self.send_button = ttk.Button(cmd_frame, text="Verstuur", command=self.send_command)
        self.send_button.pack(side="left", padx=5, pady=5)
        
        # Download knop voor automatische Kermit filetransfer (PC6000)
        self.download_button = ttk.Button(self, text="Download Data (Kermit)", command=self.download_data)
        self.download_button.pack(pady=5)
    
    def refresh_com_ports(self):
        """Vult de combobox met beschikbare COM-poorten."""
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        self.com_port_combo['values'] = port_list
        if port_list:
            self.com_port_combo.current(0)
    
    def connect_serial(self):
        """Probeert een seriële verbinding op te zetten."""
        port = self.com_port_var.get()
        try:
            baudrate = int(self.baudrate_var.get())
        except ValueError:
            messagebox.showerror("Fout", "Ongeldige baudrate.")
            return

        try:
            self.serial_port = serial.Serial(port, baudrate, timeout=0.5)
            self.running = True
            self.log(f"Verbonden met {port} op {baudrate} baud.")
            # Start een thread om continu de seriële data te lezen.
            self.read_thread = threading.Thread(target=self.read_serial, daemon=True)
            self.read_thread.start()
        except Exception as e:
            messagebox.showerror("Fout", f"Kan niet verbinden: {e}")
    
    def read_serial(self):
        """Leest continu data van de seriële poort en toont dit in de terminal."""
        while self.running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if data:
                        self.log(data.decode(errors='replace'))
                time.sleep(0.1)
            except Exception as e:
                self.log(f"Fout tijdens lezen: {e}")
                break
    
    def send_command(self, event=None):
        """Verstuurt een commando dat in de invoer staat naar het apparaat."""
        cmd = self.cmd_entry.get()
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write((cmd + "\r\n").encode())
                self.log(f"> {cmd}")
                self.cmd_entry.delete(0, tk.END)
            except Exception as e:
                self.log(f"Fout tijdens versturen: {e}")
        else:
            messagebox.showwarning("Niet verbonden", "Eerst verbinden met een COM-poort.")
    
    def log(self, message):
        """Voegt een regel toe aan de terminal."""
        self.terminal_text.configure(state="normal")
        self.terminal_text.insert(tk.END, message + "\n")
        self.terminal_text.configure(state="disabled")
        self.terminal_text.see(tk.END)
    
    def wait_for_prompt(self, prompt, timeout=10):
        """
        Wacht tot de binnenkomende data de meegegeven prompt bevat.
        Geeft de volledige buffer terug als de prompt herkend wordt,
        of None als de timeout verloopt.
        """
        buffer = ""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.serial_port.in_waiting:
                data = self.serial_port.read(self.serial_port.in_waiting).decode(errors="replace")
                buffer += data
                if prompt in buffer:
                    return buffer
            time.sleep(0.1)
        return None
    
    def go_to_main_menu(self):
        """
        Probeert ervoor te zorgen dat het apparaat in het hoofdmenu zit.
        Er wordt een aantal keren een ESC-teken met newline verstuurd en gecontroleerd
        op bekende tekstfragmenten (bv. 'Select option' of 'Filename') om zeker te zijn dat
        het hoofdmenu bereikt is.
        """
        self.log("Proberen in het hoofdmenu te komen...")
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            self.log(f"Poging {attempt}...")
            self.serial_port.reset_input_buffer()
            self.serial_port.write(b'\x1b\r\n')
            time.sleep(1)
            
            buffer = ""
            timeout = time.time() + 3  # 3 seconden wachten per poging
            while time.time() < timeout:
                if self.serial_port.in_waiting:
                    buffer += self.serial_port.read(self.serial_port.in_waiting).decode(errors='replace')
                if "Select option" in buffer or "Filename" in buffer:
                    self.log("Hoofdmenu gedetecteerd.")
                    return True
                time.sleep(0.1)
            self.log(f"Na poging {attempt} geen hoofdmenu gevonden.")
        self.log("Hoofdmenu niet gedetecteerd na meerdere pogingen.")
        return False
    
    def download_data(self):
        """
        Start de automatische Kermit filetransfer in een aparte thread.
        Hierbij wordt eerst geprobeerd in het hoofdmenu te komen en daarna
        worden de menu's automatisch doorlopen:
          1. Verstuur 'kermit'
          2. Wacht op het "File details"‑menu en stuur dan '1' (view file)
          3. Wacht op het "Time window"‑menu en stuur dan '1' (Proceed)
          4. Verzamel vervolgens de daadwerkelijke filedata
        """
        if self.serial_port and self.serial_port.is_open:
            threading.Thread(target=self.download_data_thread, daemon=True).start()
        else:
            messagebox.showwarning("Niet verbonden", "Eerst verbinden met een COM-poort.")
    
    def download_data_thread(self):
        try:
            # Zorg ervoor dat we in het hoofdmenu zitten.
            if not self.go_to_main_menu():
                self.log("Hoofdmenu niet gevonden. Transfer afgebroken.")
                return
            
            # 1. Verstuur het kermit-commando
            self.serial_port.write(b'kermit\r\n')
            self.log("> kermit (Start filetransfer via Kermit)")
            
            # 2. Wacht op het File details-menu
            details = self.wait_for_prompt("File details", timeout=10)
            if not details:
                self.log("File details menu niet gevonden. Transfer afgebroken.")
                return
            self.log("File details menu ontvangen.")
            
            # Verstuur '1' om het bestand te selecteren (pas dit aan indien een ander commando nodig is)
            self.serial_port.write(b'1\r\n')
            self.log("> 1 (File geselecteerd)")
            
            # 3. Wacht op het Time window-menu
            time_window = self.wait_for_prompt("Time window", timeout=10)
            if not time_window:
                self.log("Time window menu niet gevonden. Transfer afgebroken.")
                return
            self.log("Time window menu ontvangen.")
            
            # Verstuur '1' om door te gaan (Proceed)
            self.serial_port.write(b'1\r\n')
            self.log("> 1 (Proceed)")
            
            # 4. Verzamel de daadwerkelijke filedata.
            self.log("Ontvang bestandgegevens...")
            received_data = b""
            last_data_time = time.time()
            timeout_after_last_data = 3  # wacht 3 seconden na de laatste data
            while True:
                if self.serial_port.in_waiting:
                    chunk = self.serial_port.read(self.serial_port.in_waiting)
                    received_data += chunk
                    last_data_time = time.time()
                else:
                    if time.time() - last_data_time > timeout_after_last_data:
                        break
                time.sleep(0.1)
            
            self.log("Bestandgegevens volledig ontvangen.")
            self.after(0, self.save_file, received_data)
        except Exception as e:
            self.log(f"Fout tijdens filetransfer: {e}")
    
    def save_file(self, data):
        file_path = filedialog.asksaveasfilename(title="Sla bestand op", defaultextension=".dat",
                                                 filetypes=[("Data Files", "*.dat"), ("All Files", "*.*")])
        if file_path:
            try:
                with open(file_path, "wb") as f:
                    f.write(data)
                self.log(f"Bestand opgeslagen als: {file_path}")
            except Exception as e:
                self.log(f"Fout bij opslaan bestand: {e}")
        else:
            self.log("Opslaan geannuleerd.")
    
    def on_close(self):
        """Wordt aangeroepen bij afsluiten van de applicatie."""
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.destroy()

if __name__ == "__main__":
    app = SerialTerminalApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
