import os
import csv
import zipfile
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# -------------------------------
# Parser-functies
# -------------------------------

def parse_two_line_file(content: str, filename: str):
    """
    Parseer een twee-regelig Seaguard Aanderaa tekstbestand.
    Returns: (station_name, timestamp_str_out, conductivity_value, temperature_value)

    - Regel 1, kolom 3: temperatuur (°C)
    - Regel 2, kolom 3: conductiviteit (µS/cm)
    - Tijdstempel uit regel 1, kolom 2 -> formaat dd/mm/yyyyhh:mm:ss (zonder spatie)
    - Station name uit regel 1, kolom 1
    """
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Bestand {filename} bevat minder dan 2 regels.")

    parts1 = lines[0].split("|")  # regel 1
    parts2 = lines[1].split("|")  # regel 2
    if len(parts1) < 3 or len(parts2) < 3:
        raise ValueError(f"Onvoldoende kolommen in {filename} (verwacht minstens 3 per regel).")

    station_name = parts1[0].strip()
    ts_raw = parts1[1].strip()  # bv. '2025-08-26 17:05:00.00000'
    ts_main = ts_raw.split(".")[0]  # strip fracties

    # Probeer te parsen uit de regel; zo niet, val terug op bestandsnaam (station_yyyymmddhhmmss)
    dt = None
    if ts_main:
        try:
            dt = datetime.strptime(ts_main, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = None

    if dt is None:
        base = os.path.basename(filename)
        # verwacht patroon: station_yyyymmddhhmmss.ext
        stamp = base.split("_")[-1].split(".")[0]
        dt = datetime.strptime(stamp, "%Y%m%d%H%M%S")

    timestamp_out = dt.strftime("%d/%m/%Y%H:%M:%S")  # dd/mm/yyyyhh:mm:ss (geen spatie)
    temperature = parts1[2].strip()
    conductivity = parts2[2].strip()
    return station_name, timestamp_out, conductivity, temperature


def iter_files_from_inputs(paths):
    """
    Genereer (virtuele_pad, content_bytes) uit:
    - mappen (alle .txt, recursief)
    - .zip (alle .txt binnen de zip)
    - losse .txt
    """
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    if fn.lower().endswith(".txt"):
                        full = os.path.join(root, fn)
                        with open(full, "rb") as f:
                            yield full, f.read()
        elif os.path.isfile(path):
            if path.lower().endswith(".zip"):
                with zipfile.ZipFile(path, "r") as zf:
                    for zi in zf.infolist():
                        if not zi.is_dir() and zi.filename.lower().endswith(".txt"):
                            with zf.open(zi) as f:
                                yield zi.filename, f.read()
            elif path.lower().endswith(".txt"):
                with open(path, "rb") as f:
                    yield path, f.read()


def build_csv_from_inputs(inputs, output_csv_path, on_progress=None):
    """
    Bouw één CSV met blokken per station:
      Rij 1: Station Name
      Rij 2: Date, Time, Value [µS/cm], Quality, Tags, Comments
      Rij 3+: dd/mm/yyyy, hh:mm:ss, <cond>, NA, conductivity, <temp>
    """
    data_by_station = {}
    total = 0

    # Eerst alles inlezen/parsen
    for vpath, content in iter_files_from_inputs(inputs):
        total += 1
        try:
            txt = content.decode("utf-8", errors="replace")
            station, ts_out, cond, temp = parse_two_line_file(txt, vpath)
            data_by_station.setdefault(station, []).append((ts_out, cond, temp))
            if on_progress:
                on_progress(f"Ingelezen: {os.path.basename(vpath)}")
        except Exception as e:
            if on_progress:
                on_progress(f"Overgeslagen (patroon mismatch): {os.path.basename(vpath)} — {e}")

    if not data_by_station:
        raise RuntimeError("Geen geldige bestanden gevonden die aan het 2-regelspatroon voldoen.")

    # sorteer per station op tijd
    for st in data_by_station:
        data_by_station[st].sort(key=lambda r: datetime.strptime(r[0], "%d/%m/%Y%H:%M:%S"))

    # wegschrijven
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",")
        first = True
        for station, rows in data_by_station.items():
            if not first:
                w.writerow([])  # lege lijn tussen blokken
            first = False
            # Rij 1: Station Name
            w.writerow([station])
            # Rij 2: header
            w.writerow(["Date", "Time", "Value [µS/cm]", "Quality", "Tags", "Comments"])
            # Data
            for ts_out, cond, temp in rows:
                date_part = ts_out[:10]      # dd/mm/yyyy
                time_part = ts_out[10:]      # hh:mm:ss
                w.writerow([date_part, time_part, cond, "NA", "conductivity", temp])

    return output_csv_path, total, {k: len(v) for k, v in data_by_station.items()}


# -------------------------------
# GUI-app
# -------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Seaguard Aanderaa → CSV (dir of zip/txt)")
        self.geometry("720x520")

        self.input_mode = tk.StringVar(value="dir")  # "dir" of "files"
        self.selected_paths = []  # lijst van paden (dir of bestanden)
        self.output_csv = tk.StringVar(value="")

        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        # Keuze input mode
        lbl_mode = ttk.Label(frm, text="Invoerbron:")
        lbl_mode.grid(row=0, column=0, sticky="w")

        rb_dir = ttk.Radiobutton(frm, text="Directory", variable=self.input_mode, value="dir")
        rb_files = ttk.Radiobutton(frm, text="Bestanden (ZIP/TXT)", variable=self.input_mode, value="files")
        rb_dir.grid(row=0, column=1, sticky="w")
        rb_files.grid(row=0, column=2, sticky="w")

        # Knoppen selecteren
        btn_sel = ttk.Button(frm, text="Selecteer…", command=self.select_inputs)
        btn_sel.grid(row=0, column=3, padx=8, sticky="w")

        # Lijst van gekozen paden
        self.lst = tk.Listbox(frm, height=6)
        self.lst.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(6, 6))
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(3, weight=1)

        # Output selecteren
        lbl_out = ttk.Label(frm, text="Output CSV:")
        lbl_out.grid(row=2, column=0, sticky="w")
        ent_out = ttk.Entry(frm, textvariable=self.output_csv)
        ent_out.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(0, 6))
        btn_out = ttk.Button(frm, text="Opslaan als…", command=self.select_output)
        btn_out.grid(row=2, column=3, sticky="e")

        # Run knop
        btn_run = ttk.Button(frm, text="Start conversie", command=self.run_conversion)
        btn_run.grid(row=3, column=3, sticky="e", pady=8)

        # Log venster
        lbl_log = ttk.Label(frm, text="Log / voortgang:")
        lbl_log.grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.txt_log = tk.Text(frm, height=12)
        self.txt_log.grid(row=5, column=0, columnspan=4, sticky="nsew")
        frm.rowconfigure(5, weight=2)

    def log(self, msg: str):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.update_idletasks()

    def select_inputs(self):
        mode = self.input_mode.get()
        if mode == "dir":
            d = filedialog.askdirectory(title="Kies een directory met .txt bestanden")
            if d:
                self.selected_paths = [d]
                self.refresh_listbox()
        else:
            files = filedialog.askopenfilenames(
                title="Kies ZIP en/of TXT bestanden",
                filetypes=[("ZIP of TXT", "*.zip *.txt"), ("ZIP", "*.zip"), ("TXT", "*.txt"), ("Alle bestanden", "*.*")]
            )
            if files:
                self.selected_paths = list(files)
                self.refresh_listbox()

    def refresh_listbox(self):
        self.lst.delete(0, tk.END)
        for p in self.selected_paths:
            self.lst.insert(tk.END, p)

    def select_output(self):
        f = filedialog.asksaveasfilename(
            title="CSV opslaan als…",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if f:
            self.output_csv.set(f)

    def run_conversion(self):
        if not self.selected_paths:
            messagebox.showwarning("Ontbrekende invoer", "Selecteer eerst een directory of bestanden.")
            return
        if not self.output_csv.get():
            messagebox.showwarning("Ontbrekende uitvoer", "Kies eerst een output CSV-bestand.")
            return

        self.txt_log.delete("1.0", tk.END)
        self.log("Conversie gestart…")

        try:
            out, total, per_station = build_csv_from_inputs(
                self.selected_paths,
                self.output_csv.get(),
                on_progress=self.log
            )
        except Exception as e:
            self.log(f"Fout: {e}")
            messagebox.showerror("Conversie mislukt", str(e))
            return

        self.log("Conversie klaar.")
        self.log(f"Totaal ingelezen kandidaten: {total}")
        for st, n in per_station.items():
            self.log(f"Station '{st}': {n} records")
        messagebox.showinfo("Succes", f"CSV geschreven:\n{out}")

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
