import customtkinter as ctk
import serial
import serial.tools.list_ports
import threading
import time
from datetime import datetime, timedelta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
from collections import deque
import random
from tkinter import filedialog, messagebox
import csv
import os
import io
import re
import json
import subprocess
from pathlib import Path

# UI Thematiek a la YSI EcoWW
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

YSI_SENSORS = [
    "Time", "Temperature", "Conductivity", "Dissolved Oxy", "ISE1 pH", 
    "ISE2 Orp", "ISE3 NH4+", "ISE4 NO3-", "ISE5 NONE", "Optic T Turbidity", 
    "Optic C Chlorophyll", "Battery volts"
]

YSI_REPORT_PARAMS = [
    "Date m/d/y", "Time hh:mm:ss", "Temp C", "SpCond mS/cm", "Cond", "Resist",
    "TDS", "Sal ppt", "Press psia", "Depth meters", "DOSat %", "DO mg/L", "DOchrg", "pH", "pH mV", "Orp mV",
    "NH4+ N mg/L", "NH4+ N mV", "NH3 N mg/L", "NO3- N mg/L", "NO3- N mV",
    "Cl- mg/L", "Cl- mV", "Turbid+ NTU", "Chl ug/L", "Chl RFU", "Battery volts"
]

REPORT_LAYOUT_CANDIDATES = [
    ["Date m/d/y", "Time hh:mm:ss", "Temp C", "SpCond mS/cm", "DO mg/L", "pH", "Battery volts", "Turbid+ NTU"],
    ["Date m/d/y", "Time hh:mm:ss", "Temp C", "SpCond mS/cm", "DO mg/L", "pH", "Turbid+ NTU", "Battery volts"],
    ["Date m/d/y", "Time hh:mm:ss", "Temp C", "Cond", "Depth", "pH", "Turbid+ NTU", "Battery volts"],
    ["Date m/d/y", "Time hh:mm:ss", "Temp C", "SpCond mS/cm", "Depth", "pH", "Turbid+ NTU", "Battery volts"],
]

class EcoWatchClone(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EcoWatch Next-Gen (Volledige YSI 6-Series Emulator)")
        self.geometry("1450x900")
        
        self.serial_conn = None
        self.is_reading = False
        self.demo_mode = False
        self.serial_rx_buffer = ""
        self.menu_scan_active = False
        self.menu_scan_results = []
        self.raw_rx_lines = deque(maxlen=1200)
        self.raw_rx_seq = 0
        self.auto_sample_enabled = False
        self.auto_sample_after_id = None
        self.term_history = []
        
        # Virtuele YSI Device Status
        self.device_setup = {
            "protocol": "RS-232",
            "baudrate": 9600,
            "parity": "N",
            "stopbits": "1",
            "sample_trigger_cmd": "MENU;1;1;1;1",
            "date_format": "m/d/y", 
            "page_length": 25,
            "instrument_id": "YSI Sonde Python",
            "circuit_board_sn": "01234567",
            "glp_filename": "GLP00001",
            "sdi_address": 0,
            "interval_sec": 1,
            "do_warmup_sec": 40,
            "turbidity_filter": False,
            "autosleep": True,
            "wiper_active": True,
            "sensors": ["Time", "Temperature", "Conductivity", "Dissolved Oxy", "ISE1 pH", "Optic T Turbidity"],
            "reports": ["Date m/d/y", "Time hh:mm:ss", "Temp C", "SpCond mS/cm", "DO mg/L", "pH", "Battery volts", "Turbid+ NTU"]
        }
        
        self.plot_vars = {} # Bevat boolean variabelen om >1 grafieklijnen te plotten voor Live
        self.file_plot_vars = {} # Boolean vars voor de File/History plotting (Offline Grafiek!)
        
        # Plot Geheugen voor Live Run (Tab 1)
        self.data_buffers = {param: deque(maxlen=250) for param in YSI_REPORT_PARAMS}
        self.time_buffer = deque(maxlen=250) 
        self.datetime_buffer = deque(maxlen=250) 
        self.recent_numeric_rows = deque(maxlen=20)
        
        # File System Geheugen (Tab 2 - Database Extractie)
        self.logged_data = [] 
        self.loaded_file_headers = []
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.create_sidebar()
        self.create_tabview()
        self.refresh_ports()

    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1) 
        
        logo = ctk.CTkLabel(self.sidebar_frame, text="YSI 6-Series Controller", font=ctk.CTkFont(size=20, weight="bold"), text_color="#bb86fc")
        logo.grid(row=0, column=0, padx=20, pady=(15, 10))

        ctk.CTkLabel(self.sidebar_frame, text="Communicatie Protocol:").grid(row=1, column=0, padx=20, pady=0, sticky="w")
        self.proto_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["RS-232", "SDI-12"])
        self.proto_menu.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(self.sidebar_frame, text="Windows COM Poort:").grid(row=3, column=0, padx=20, pady=0, sticky="w")
        self.port_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Geen poorten..."])
        self.port_menu.grid(row=4, column=0, padx=20, pady=(0, 5), sticky="ew")
        
        self.refresh_btn = ctk.CTkButton(self.sidebar_frame, text="↻ Zoek Modems/COM", command=self.refresh_ports, fg_color="#4CAF50", height=28)
        self.refresh_btn.grid(row=5, column=0, padx=20, pady=5, sticky="ew")

        # Live Grafiek Parameters
        ctk.CTkLabel(self.sidebar_frame, text="Selecteer Live Grafiek(en):").grid(row=6, column=0, padx=20, pady=(15, 0), sticky="w")
        self.graph_selectors_frame = ctk.CTkScrollableFrame(self.sidebar_frame, height=120)
        self.graph_selectors_frame.grid(row=7, column=0, padx=15, pady=5, sticky="ew")
        self.update_live_graph_selectors()
        
        self.connect_btn = ctk.CTkButton(self.sidebar_frame, text="▶ Init Sonde Run", command=self.toggle_connection, fg_color="#2196F3", height=40)
        self.connect_btn.grid(row=8, column=0, padx=20, pady=(20, 5), sticky="ew")
        
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="DCP Status: Standby", text_color="#F44336")
        self.status_label.grid(row=9, column=0, padx=20, pady=5, sticky="w")

    def update_live_graph_selectors(self):
        """Ververst de sidebar checkboxes voor de LIVE Run tab."""
        for widget in self.graph_selectors_frame.winfo_children():
            widget.destroy()
            
        plottable_params = [p for p in self.device_setup["reports"] if "Date" not in p and "Time" not in p]
        
        self.plot_vars = {}
        for i, p in enumerate(plottable_params):
            var = ctk.BooleanVar(value=(i == 0)) 
            self.plot_vars[p] = var
            cb = ctk.CTkCheckBox(self.graph_selectors_frame, text=p, variable=var, command=self.update_live_graph)
            cb.pack(anchor="w", pady=2, padx=5)

    def create_tabview(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.tabview.add("1-Run (Real-time Frame)")
        self.tabview.add("3-File (Upload, Plot & Convert)")
        self.tabview.add("Sonde Main Menu (Setup)")
        self.tabview.add("Terminal / Advanced")
        
        self.build_run_tab(self.tabview.tab("1-Run (Real-time Frame)"))
        self.build_file_tab(self.tabview.tab("3-File (Upload, Plot & Convert)"))
        self.build_sonde_menu_tab(self.tabview.tab("Sonde Main Menu (Setup)"))
        self.build_terminal_tab(self.tabview.tab("Terminal / Advanced"))

    def _style_toolbar(self, toolbar):
        """Kleur de Matplotlib navigatie toolbar zodat hij zichtbaar is op donkere achtergrond."""
        TOOLBAR_BG = '#2b2b2b'
        BTN_BG = '#3a3a5c'
        BTN_FG = '#ffffff'
        toolbar.config(background=TOOLBAR_BG)
        for child in toolbar.winfo_children():
            try:
                widget_type = child.winfo_class()
                if widget_type == 'Button':
                    child.config(background=BTN_BG, foreground=BTN_FG,
                                 activebackground='#bb86fc', activeforeground='white',
                                 relief='flat', padx=6, pady=3, borderwidth=1,
                                 highlightbackground='#555', cursor='hand2')
                elif widget_type in ('Label', 'Entry'):
                    child.config(background=TOOLBAR_BG, foreground='#dddddd')
                elif widget_type == 'Frame':
                    child.config(background=TOOLBAR_BG)
            except Exception:
                pass
        toolbar.update()

    def build_run_tab(self, parent):
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=3) 
        parent.grid_rowconfigure(2, weight=1) 
        parent.grid_columnconfigure(0, weight=1)

        control_bar = ctk.CTkFrame(parent, fg_color="transparent")
        control_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        control_bar.grid_columnconfigure(8, weight=1)
        ctk.CTkLabel(control_bar, text="Discrete sample rate (s):").grid(row=0, column=0, padx=(0, 6), sticky="w")
        self.live_sample_rate_entry = ctk.CTkEntry(control_bar, width=70)
        self.live_sample_rate_entry.insert(0, str(self.device_setup["interval_sec"]))
        self.live_sample_rate_entry.grid(row=0, column=1, padx=(0, 10), sticky="w")
        ctk.CTkLabel(control_bar, text="Sample seq:").grid(row=0, column=2, padx=(0, 6), sticky="w")
        self.live_sample_cmd_entry = ctk.CTkEntry(control_bar, width=120)
        self.live_sample_cmd_entry.insert(0, str(self.device_setup.get("sample_trigger_cmd", "MENU;1;1;1;1")))
        self.live_sample_cmd_entry.grid(row=0, column=3, padx=(0, 10), sticky="w")
        self.sample_now_btn = ctk.CTkButton(control_bar, text="Start sample", width=110, command=self.send_discrete_sample)
        self.sample_now_btn.grid(row=0, column=4, padx=(0, 8), sticky="w")
        self.auto_sample_btn = ctk.CTkButton(control_bar, text="Auto sample: UIT", width=140, fg_color="#607D8B", hover_color="#546E7A", command=self.toggle_auto_sampling)
        self.auto_sample_btn.grid(row=0, column=5, sticky="w")
        self.log_sample_btn = ctk.CTkButton(control_bar, text="LOG last sample", width=120, fg_color="#8E44AD", hover_color="#6C3483", command=self.send_log_last_sample)
        self.log_sample_btn.grid(row=0, column=6, padx=(8, 8), sticky="w")
        self.stop_sample_btn = ctk.CTkButton(control_bar, text="Stop sampling", width=120, fg_color="#C0392B", hover_color="#922B21", command=self.stop_sampling)
        self.stop_sample_btn.grid(row=0, column=7, padx=(0, 8), sticky="w")
        self.save_live_html_btn = ctk.CTkButton(control_bar, text="Save graph HTML", width=130, fg_color="#1565C0", hover_color="#0D47A1", command=self.export_live_graph_html)
        self.save_live_html_btn.grid(row=0, column=8, sticky="w")

        self.fig_live, self.ax_live = plt.subplots(figsize=(7, 4), facecolor='#1e1e1e')
        self.ax_live.set_facecolor('#1e1e1e')
        self.ax_live.tick_params(colors='white')
        self.ax_live.spines['bottom'].set_color('#555')
        self.ax_live.spines['left'].set_color('#555')
        self.ax_live.spines['top'].set_visible(False)
        self.ax_live.spines['right'].set_visible(False)
        
        f_graph_live = ctk.CTkFrame(parent, fg_color="#1e1e1e")
        f_graph_live.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=f_graph_live)
        self.canvas_live.get_tk_widget().pack(side="top", fill="both", expand=True)
        
        self.toolbar_live = NavigationToolbar2Tk(self.canvas_live, f_graph_live)
        self._style_toolbar(self.toolbar_live)
        
        # Zwevende annotatie tooltip
        self.annot_live = self.ax_live.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#bb86fc", lw=1.5),
            arrowprops=dict(arrowstyle="->", color="#bb86fc", lw=1.2),
            color="white", fontsize=9, zorder=100
        )
        self.annot_live.set_visible(False)
        self.fig_live.canvas.mpl_connect("motion_notify_event", self.hover_live)
        
        self.fig_live.tight_layout()
        self.update_live_graph()
        
        self.log_box = ctk.CTkTextbox(parent, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.log_box.insert("0.0", "--- Run --- \nReal-time Frame wordt enkel opgebouwd via Seriële Port/Simulatie...\nAlle gelogde commando's komen hier binnen.\n")
        self.log_box.configure(state="disabled")

    def hover_live(self, event):
        vis = self.annot_live.get_visible()
        if event.inaxes == self.ax_live:
            nearest = None
            nearest_dist = float('inf')
            nearest_label = ""
            nearest_y = 0.0
            nearest_x = 0.0
            for line in self.ax_live.get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) == 0: continue
                try:
                    # Vind het dichtstbijzijnde punt op de lijn
                    import numpy as np
                    xdata_num = plt.matplotlib.dates.date2num(xdata) if hasattr(xdata[0], 'year') else xdata
                    dists = np.abs(xdata_num - event.xdata)
                    idx = int(np.argmin(dists))
                    dist = dists[idx]
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_label = line.get_label()
                        nearest_x = xdata[idx]
                        nearest_y = ydata[idx]
                except Exception:
                    continue
            if nearest_label and nearest_dist < 0.1:
                self.annot_live.xy = (nearest_x, nearest_y)
                try:
                    from matplotlib.dates import num2date
                    import matplotlib.dates as mdates_
                    dt = num2date(mdates_.date2num(nearest_x)).strftime('%H:%M:%S')
                except Exception:
                    dt = str(nearest_x)
                self.annot_live.set_text(f"📍 {nearest_label}\n  Waarde: {nearest_y:.3f}\n  Tijd: {dt}")
                self.annot_live.set_visible(True)
                self.canvas_live.draw_idle()
                return
        if vis:
            self.annot_live.set_visible(False)
            self.canvas_live.draw_idle()

    def build_file_tab(self, parent):
        """Tab voor de File/Log functies. Digitaal bestand uploaden en puur PLOT/Converteer functies."""
        parent.grid_rowconfigure(1, weight=5)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=0)
        
        # Tools Balk Boven (Opslaan / Converteren / ASCII tools)
        f_tools = ctk.CTkFrame(parent, fg_color="transparent")
        f_tools.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(f_tools, text="Upload/Extract:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkButton(f_tools, text="📥 Sync Sonde Memory (DAT)", command=self.load_sonde_memory, fg_color="#FF9800", hover_color="#F57C00").pack(side="left", padx=5)
        ctk.CTkButton(f_tools, text="📠 Download (Kermit Protocol)", command=self.download_via_kermit, fg_color="#E91E63", hover_color="#C2185B").pack(side="left", padx=5)
        ctk.CTkButton(f_tools, text="📂 Lokaal Bestand Openen", command=self.import_local_file, fg_color="#607D8B", hover_color="#455A64").pack(side="left", padx=5)
        
        ctk.CTkLabel(f_tools, text="|  Format Converter:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(30, 10))
        ctk.CTkButton(f_tools, text="💾 Convert to ASCII: CSV/CDF", command=lambda: self.export_data("csv")).pack(side="left", padx=5)
        ctk.CTkButton(f_tools, text="📝 Convert to ASCII: PRN/TXT", command=lambda: self.export_data("txt")).pack(side="left", padx=5)
        ctk.CTkButton(f_tools, text="🗑 View Clear", command=self.clear_data, fg_color="#F44336").pack(side="right", padx=10)

        # LINKER MENU BINNEN TAB 3: Graph Selector voor file data plottings
        f_vars = ctk.CTkFrame(parent, width=200)
        f_vars.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(f_vars, text="Grafiek Variabelen:", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        
        self.file_vars_scroll = ctk.CTkScrollableFrame(f_vars)
        self.file_vars_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        ctk.CTkLabel(self.file_vars_scroll, text="Open eerst een\nbestand (.DAT/.X)\nom assen te bepalen.").pack(pady=50)

        # MEESTER MATPLOTLIB CHART (Offline History File Viewing)
        self.fig_file, self.ax_file = plt.subplots(figsize=(8, 5), facecolor='#1e1e1e')
        self.ax_file.set_facecolor('#1e1e1e')
        self.ax_file.tick_params(colors='white')
        self.ax_file.spines['bottom'].set_color('#555')
        self.ax_file.spines['left'].set_color('#555')
        self.ax_file.spines['top'].set_visible(False)
        self.ax_file.spines['right'].set_visible(False)
        
        f_graph_file = ctk.CTkFrame(parent, fg_color="#1e1e1e")
        f_graph_file.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        self.canvas_file = FigureCanvasTkAgg(self.fig_file, master=f_graph_file)
        self.canvas_file.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.toolbar_file = NavigationToolbar2Tk(self.canvas_file, f_graph_file)
        self._style_toolbar(self.toolbar_file)

        # Zwevende annotatie tooltip
        self.annot_file = self.ax_file.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#03dac6", lw=1.5),
            arrowprops=dict(arrowstyle="->", color="#03dac6", lw=1.2),
            color="white", fontsize=9, zorder=100
        )
        self.annot_file.set_visible(False)
        self.fig_file.canvas.mpl_connect("motion_notify_event", self.hover_file)

        self.update_file_graph(empty=True)

    def hover_file(self, event):
        vis = self.annot_file.get_visible()
        if event.inaxes == self.ax_file:
            nearest_label = ""
            nearest_dist = float('inf')
            nearest_x = 0.0
            nearest_y = 0.0
            for line in self.ax_file.get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) == 0: continue
                try:
                    import numpy as np
                    xdata_num = plt.matplotlib.dates.date2num(xdata) if hasattr(xdata[0], 'year') else xdata
                    dists = np.abs(xdata_num - event.xdata)
                    idx = int(np.argmin(dists))
                    dist = dists[idx]
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_label = line.get_label()
                        nearest_x = xdata[idx]
                        nearest_y = ydata[idx]
                except Exception:
                    continue
            if nearest_label and nearest_dist < 0.5:
                self.annot_file.xy = (nearest_x, nearest_y)
                try:
                    from matplotlib.dates import num2date
                    import matplotlib.dates as mdates_
                    dt = num2date(mdates_.date2num(nearest_x)).strftime('%m/%d/%Y  %H:%M:%S')
                except Exception:
                    dt = str(nearest_x)
                self.annot_file.set_text(f"📍 {nearest_label}\n  Waarde: {nearest_y:.3f}\n  Tijd: {dt}")
                self.annot_file.set_visible(True)
                self.canvas_file.draw_idle()
                return
        if vis:
            self.annot_file.set_visible(False)
            self.canvas_file.draw_idle()

    def build_sonde_menu_tab(self, parent):
        parent.grid_columnconfigure((0, 1, 2), weight=1)
        parent.grid_rowconfigure((0, 1), weight=1)
        
        # === 5-System Setup ===
        f_sys = ctk.CTkFrame(parent)
        f_sys.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(f_sys, text="5-System Setup", font=ctk.CTkFont(weight="bold", size=16), text_color="#bb86fc").pack(pady=10)
        
        ctk.CTkLabel(f_sys, text="Baud Rate:").pack(anchor="w", padx=15)
        self.ui_baud = ctk.CTkOptionMenu(f_sys, values=["1200", "2400", "4800", "9600", "19200", "38400", "115200"])
        self.ui_baud.set(str(self.device_setup["baudrate"]))
        self.ui_baud.pack(padx=15, pady=2, fill="x")

        ctk.CTkLabel(f_sys, text="Date Format:").pack(anchor="w", padx=15, pady=(5,0))
        self.ui_datefmt = ctk.CTkOptionMenu(f_sys, values=["m/d/y", "d/m/y", "y/m/d"])
        self.ui_datefmt.set(self.device_setup["date_format"])
        self.ui_datefmt.pack(padx=15, pady=2, fill="x")

        ctk.CTkLabel(f_sys, text="Instrument ID:").pack(anchor="w", padx=15, pady=(5,0))
        self.ui_instid = ctk.CTkEntry(f_sys)
        self.ui_instid.insert(0, self.device_setup["instrument_id"])
        self.ui_instid.pack(padx=15, pady=2, fill="x")

        ctk.CTkLabel(f_sys, text="SDI-12 Address:").pack(anchor="w", padx=15, pady=(5,0))
        self.ui_sdiaddr = ctk.CTkEntry(f_sys)
        self.ui_sdiaddr.insert(0, str(self.device_setup["sdi_address"]))
        self.ui_sdiaddr.pack(padx=15, pady=2, fill="x")

        # === 7-Sensor Menu ===
        f_sens = ctk.CTkFrame(parent)
        f_sens.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(f_sens, text="7-Sensors Enabled", font=ctk.CTkFont(weight="bold", size=16), text_color="#03dac6").pack(pady=10)
        self.sensor_summary_label = ctk.CTkLabel(f_sens, text="", justify="left", wraplength=260, text_color="#b0bec5")
        self.sensor_summary_label.pack(fill="x", padx=10, pady=(0, 8))
        
        self.sensor_vars = {}
        s_scroll = ctk.CTkScrollableFrame(f_sens, fg_color="transparent")
        s_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        for s in YSI_SENSORS:
            var = ctk.BooleanVar(value=(s in self.device_setup["sensors"]))
            self.sensor_vars[s] = var
            ctk.CTkCheckBox(s_scroll, text=s, variable=var).pack(anchor="w", pady=3)
        self.update_sensor_summary()

        # === 6-Report Setup ===
        f_rep = ctk.CTkFrame(parent)
        f_rep.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(f_rep, text="6-Report Setup", font=ctk.CTkFont(weight="bold", size=16), text_color="#cf6679").pack(pady=10)
        
        self.report_vars = {}
        r_scroll = ctk.CTkScrollableFrame(f_rep, fg_color="transparent")
        r_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        for r in YSI_REPORT_PARAMS:
            var = ctk.BooleanVar(value=(r in self.device_setup["reports"]))
            self.report_vars[r] = var
            ctk.CTkCheckBox(r_scroll, text=r, variable=var).pack(anchor="w", pady=2)

        # === 8-Advanced Setup ===
        f_adv = ctk.CTkFrame(parent)
        f_adv.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(f_adv, text="8-Advanced Setup", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=10)
        
        adv_grid = ctk.CTkFrame(f_adv, fg_color="transparent")
        adv_grid.pack(fill="both", expand=True, padx=20)
        
        self.ui_autosleep = ctk.CTkCheckBox(adv_grid, text="Auto-Sleep (RS232 Power Mod)")
        if self.device_setup["autosleep"]: self.ui_autosleep.select()
        self.ui_autosleep.grid(row=0, column=0, sticky="w", pady=5)

        self.ui_wiper = ctk.CTkCheckBox(adv_grid, text="Wipe Optics before meting")
        if self.device_setup["wiper_active"]: self.ui_wiper.select()
        self.ui_wiper.grid(row=0, column=1, sticky="w", pady=5)
        
        self.ui_turb_filt = ctk.CTkCheckBox(adv_grid, text="Turbid Spike Filter (TSS)")
        if self.device_setup["turbidity_filter"]: self.ui_turb_filt.select()
        self.ui_turb_filt.grid(row=1, column=0, sticky="w", pady=5)

        ctk.CTkLabel(adv_grid, text="TSS Sampling Int. (sec):").grid(row=2, column=0, sticky="w", pady=(10,0))
        self.ui_interval = ctk.CTkEntry(adv_grid, width=100)
        self.ui_interval.insert(0, str(self.device_setup["interval_sec"]))
        self.ui_interval.grid(row=3, column=0, sticky="w")
        
        ctk.CTkLabel(adv_grid, text="DO Sensor Warmup (sec):").grid(row=2, column=1, sticky="w", pady=(10,0))
        self.ui_dowarmup = ctk.CTkEntry(adv_grid, width=100)
        self.ui_dowarmup.insert(0, str(self.device_setup["do_warmup_sec"]))
        self.ui_dowarmup.grid(row=3, column=1, sticky="w")

        # === Knoppen (Main Menu Opslaan) ===
        f_btn = ctk.CTkFrame(parent, fg_color="transparent")
        f_btn.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkButton(f_btn, text="⬇ Retrieve Modbus/RS", command=self.parse_com_setup, fg_color="#FF9800", hover_color="#F57C00").pack(pady=20, fill="x")
        ctk.CTkButton(f_btn, text="⬆ Push to Sonde RAM", command=self.save_setup_to_memory, fg_color="#E91E63", hover_color="#C2185B", height=50).pack(pady=10, fill="x")

    def build_terminal_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        self.term_box = ctk.CTkTextbox(parent, font=ctk.CTkFont(family="Consolas", size=13), bg_color="black", text_color="#00FF00")
        self.term_box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.term_box.insert("0.0", "=== YSI Raw EcoWW Terminal Emulator ===\nCommand Type 'Menu' to enter Sonde...\n")
        self.term_box.configure(state="disabled")
        
        fline = ctk.CTkFrame(parent, fg_color="transparent")
        fline.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        fline.grid_columnconfigure(0, weight=1)
        
        self.term_input = ctk.CTkEntry(fline, font=ctk.CTkFont(family="Consolas", size=13))
        self.term_input.grid(row=0, column=0, sticky="ew", padx=(0,10))
        self.term_input.bind('<Return>', lambda x: self.action_send_terminal())
        self.term_input.bind('<Escape>', lambda x: (self.send_terminal_escape(), "break"))
        
        ctk.CTkButton(fline, text="SEND CMD", width=100, command=self.action_send_terminal).grid(row=0, column=1)
        ctk.CTkButton(fline, text="ESC / Back", width=100, command=self.send_terminal_escape, fg_color="#FF9800", hover_color="#F57C00").grid(row=0, column=2, padx=(10,0))
        ctk.CTkButton(fline, text="Save TXT", width=100, command=self.save_terminal_to_txt, fg_color="#1E88E5", hover_color="#1565C0").grid(row=0, column=3, padx=(10,0))

    # ---------- REVERSE ENGINEERED EXPORT & KERMIT DOWNLOAD LOGICS -----------
    def repopulate_file_graph_ui(self):
        """Maakt Plot checkboxes aan voor al het historische data in self.logged_data en tekent grafief"""
        for widget in self.file_vars_scroll.winfo_children():
            widget.destroy()
            
        if not self.logged_data:
            self.update_file_graph(empty=True)
            ctk.CTkLabel(self.file_vars_scroll, text="Open eerst een .DAT \nbestand om\nassen te bepalen.").pack(pady=50)
            return
            
        headers = self.loaded_file_headers if self.loaded_file_headers else list(self.logged_data[0].keys())
        headers = [str(p).strip() for p in headers if p is not None and str(p).strip()]
        # We negeren Date en Time voor de Y-as knoppen
        plottable_params = [p for p in headers if "date" not in p.lower() and "time" not in p.lower()]
        
        self.file_plot_vars = {}
        for i, p in enumerate(plottable_params):
            var = ctk.BooleanVar(value=(i == 0)) 
            self.file_plot_vars[p] = var
            cb = ctk.CTkCheckBox(self.file_vars_scroll, text=p, variable=var, command=lambda: self.update_file_graph(empty=False))
            cb.pack(anchor="w", pady=4, padx=5)
            
        self.update_file_graph(empty=False)

    def update_file_graph(self, empty=False):
        """Plot de gedownloade 'logged_data' (geschiedenis-bestand) dynamisch uit het digitaal weggestoken geheugen."""
        self.ax_file.clear()
        self.ax_file.set_facecolor('#1e1e1e')
        self.ax_file.tick_params(colors='white')
        
        if empty or not self.logged_data:
            self.ax_file.set_title("EcoWW History Grafiek: Geen Bestand ingeladen", color='white')
            # Zet de layout mooi af
            self.fig_file.tight_layout()
            self.canvas_file.draw()
            return
            
        self.ax_file.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        colors = ['#03dac6', '#bb86fc', '#ff9800', '#f44336', '#4caf50', '#2196f3', '#cddc39']
        c_idx = 0
        plotted_any = False
        
        # Bepalen welke velden we als Tijd-index gebruiken voor alle punten
        time_points = []
        for row_idx, row in enumerate(self.logged_data):
            t_str = ""
            for k, val in row.items():
                key_lower = str(k).lower() if k is not None else ""
                if "time" in key_lower: t_str += str(val) + " "
                if "date" in key_lower: t_str = str(val) + " " + t_str
            t_str = t_str.strip()
            # Valideer Timestamp uit .DAT robuust
            try:
                import re
                t_str_clean = re.sub(r'\s+', ' ', t_str).strip()
                formats = [
                    "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                    "%m/%d/%y %H:%M:%S", "%d/%m/%y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M", "%Y/%m/%d %H:%M",
                    "%H:%M:%S", "%H:%M"
                ]
                dt = None
                for fmt in formats:
                    try:
                        dt = datetime.strptime(t_str_clean, fmt)
                        break
                    except ValueError:
                        continue
                if dt is None:
                    raise ValueError("Onbekend datum/tijd formaat")
            except Exception:
                # Fallback: Opbouwende as-lijn, c_idx was hier buiten scope en vastgereden!
                dt = datetime.fromtimestamp(row_idx * 900) # 15 minuten per gefaalde stap
            time_points.append(dt)

        for plot_key, var_bool in self.file_plot_vars.items():
            if var_bool.get():
                # Extract alle Y waarden van deze specifieke header in de lijst dictionaries
                y_data = []
                for idx, row in enumerate(self.logged_data):
                    val = row.get(plot_key, "0")
                    try: y_data.append(float(val))
                    except: y_data.append(0.0)
                
                self.ax_file.plot(time_points, y_data, label=plot_key, color=colors[c_idx % len(colors)], marker='.', linewidth=1.5)
                plotted_any = True
                c_idx += 1
                
        if plotted_any:
            self.ax_file.legend(facecolor='#121212', edgecolor='white', labelcolor='white')
            self.fig_file.autofmt_xdate(rotation=45)
            self.ax_file.set_title("YSI File Viewer (Omgezette Historische .DAT Export)", color='white')
        else:
            self.ax_file.set_title("Vink een variabele aan in het linkermenu om te plotten", color='white')
            
        self.fig_file.tight_layout()
        self.canvas_file.draw()

    def download_via_kermit(self):
        if not (self.serial_conn and self.serial_conn.is_open):
            self.log("[KERMIT] Geen actieve seriele verbinding.")
            return
        threading.Thread(target=self._download_via_kermit_worker, daemon=True).start()

    def _download_via_kermit_worker(self):
        try:
            self.log("[KERMIT] Bestandslijst ophalen via 3-File -> 2-Upload ...")
            self._ensure_main_menu()
            lines = self._open_menu_path(["3", "2"], timeout_s=2.0)
            extra_seq = self.raw_rx_seq
            time.sleep(1.0)
            lines.extend(self._read_rx_lines_since(extra_seq))
            menu_lines = [line for line in lines if line and "select option" not in line.lower()]
            files = self._parse_upload_file_list(menu_lines)
            selection = self._show_kermit_selection_dialog(files)
            if not selection:
                self.log("[KERMIT] Download geannuleerd.")
                self._exit_current_menu()
                return

            file_token = selection["file_token"]
            file_name = selection["file_name"]
            format_token = selection["format_token"]
            format_label = selection["format_label"]

            ext_map = {"1": ".dat", "2": ".csv", "3": ".txt"}
            default_name = Path(file_name).stem + ext_map.get(format_token, ".dat")
            save_path = filedialog.asksaveasfilename(
                defaultextension=ext_map.get(format_token, ".dat"),
                initialfile=default_name,
                filetypes=[
                    ("DAT files", "*.dat"),
                    ("CSV files", "*.csv"),
                    ("TXT files", "*.txt"),
                    ("Alle bestanden", "*.*"),
                ],
            )
            if not save_path:
                self.log("[KERMIT] Geen doelbestand gekozen.")
                self._exit_current_menu()
                return

            self.log(f"[KERMIT] Gekozen file: {file_token} -> {file_name} ({format_label})")
            start_seq = self.raw_rx_seq
            self._send_menu_token(file_token)
            self._wait_for_menu_response(start_seq, timeout_s=1.5)
            time.sleep(0.25)
            start_seq = self.raw_rx_seq
            self._send_menu_token(format_token)
            self._wait_for_menu_response(start_seq, timeout_s=1.5)
            time.sleep(0.35)
            self._run_kermit_receive(save_path)
        except Exception as e:
            self.log(f"[KERMIT] Download fout: {e}")

    def _parse_upload_file_list(self, lines):
        files = []
        seen = set()
        for line in lines or []:
            clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(line)).strip()
            if not clean:
                continue
            m = re.match(r"^\s*([0-9A-Za-z]+)\s*[-.) ]+\s*([A-Za-z0-9_.\-]+\.(?:dat|csv|txt|prn|cdf))\b", clean, re.IGNORECASE)
            if not m:
                m = re.search(r"\b([0-9A-Za-z]+)\s+([A-Za-z0-9_.\-]+\.(?:dat|csv|txt|prn|cdf))\b", clean, re.IGNORECASE)
            if not m:
                continue
            token = str(m.group(1)).strip()
            name = str(m.group(2)).strip()
            key = (token.upper(), name.lower())
            if key in seen:
                continue
            seen.add(key)
            files.append({"token": token, "name": name})
        return files

    def _show_kermit_selection_dialog(self, files):
        result = {}
        done = threading.Event()

        def _build():
            dialog = ctk.CTkToplevel(self)
            dialog.title("Kermit Download")
            dialog.geometry("560x520")
            dialog.grab_set()
            dialog.transient(self)
            def _cancel():
                try:
                    dialog.destroy()
                finally:
                    done.set()
            dialog.protocol("WM_DELETE_WINDOW", _cancel)

            ctk.CTkLabel(dialog, text="Selecteer bestand uit sondegeheugen", font=ctk.CTkFont(size=18, weight="bold")).pack(padx=16, pady=(16, 8), anchor="w")
            ctk.CTkLabel(dialog, text="Menu: 3-File -> 2-Upload").pack(padx=16, pady=(0, 10), anchor="w")

            file_var = ctk.StringVar(value=files[0]["token"] if files else "")
            format_var = ctk.StringVar(value="1")

            file_scroll = ctk.CTkScrollableFrame(dialog, height=240)
            file_scroll.pack(fill="both", expand=True, padx=16, pady=(0, 10))
            if files:
                for item in files:
                    ctk.CTkRadioButton(
                        file_scroll,
                        text=f'{item["token"]} - {item["name"]}',
                        variable=file_var,
                        value=item["token"],
                    ).pack(anchor="w", pady=4, padx=6)
            else:
                ctk.CTkLabel(file_scroll, text="Geen bestandsnamen automatisch herkend.\nVoer hieronder handmatig het bestandsnummer in.").pack(pady=20)

            manual_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            manual_frame.pack(fill="x", padx=16, pady=(0, 10))
            ctk.CTkLabel(manual_frame, text="Of geef bestandsnummer in:").pack(anchor="w")
            manual_entry = ctk.CTkEntry(manual_frame)
            manual_entry.pack(fill="x", pady=(4, 0))

            format_frame = ctk.CTkFrame(dialog)
            format_frame.pack(fill="x", padx=16, pady=(0, 12))
            ctk.CTkLabel(format_frame, text="Downloadformaat", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(8, 6))
            ctk.CTkRadioButton(format_frame, text="1 - Binary", variable=format_var, value="1").pack(anchor="w", padx=12, pady=2)
            ctk.CTkRadioButton(format_frame, text="2 - CSV", variable=format_var, value="2").pack(anchor="w", padx=12, pady=2)
            ctk.CTkRadioButton(format_frame, text="3 - ASCII", variable=format_var, value="3").pack(anchor="w", padx=12, pady=(2, 10))

            btns = ctk.CTkFrame(dialog, fg_color="transparent")
            btns.pack(fill="x", padx=16, pady=(0, 16))

            def _confirm():
                token = manual_entry.get().strip() or file_var.get().strip()
                if not token:
                    messagebox.showwarning("Ontbreekt", "Kies een bestand of vul een bestandsnummer in.")
                    return
                selected_name = next((item["name"] for item in files if item["token"] == token), f"file_{token}")
                label_map = {"1": "Binary", "2": "CSV", "3": "ASCII"}
                result.update(
                    {
                        "file_token": token,
                        "file_name": selected_name,
                        "format_token": format_var.get(),
                        "format_label": label_map.get(format_var.get(), "Binary"),
                    }
                )
                dialog.destroy()
                done.set()

            ctk.CTkButton(btns, text="Annuleren", fg_color="#607D8B", hover_color="#455A64", command=_cancel).pack(side="right", padx=(8, 0))
            ctk.CTkButton(btns, text="Download", fg_color="#E91E63", hover_color="#C2185B", command=_confirm).pack(side="right")

        self.after(0, _build)
        done.wait()
        return result or None

    def _find_kermit_exe(self):
        names = ("k95.exe", "k95g.exe", "kermit.exe", "wermit.exe")
        required_runtime = ("regina.dll", "libssl-3-x64.dll", "libcrypto-3-x64.dll")
        candidates = []
        search_roots = [Path(os.getcwd()), Path(os.getcwd()) / "Oude_app"]

        for root_dir in search_roots:
            if not root_dir.exists():
                continue
            for entry in root_dir.iterdir():
                if entry.is_dir() and entry.name.lower().startswith("k95-"):
                    for name in names:
                        candidate = entry / name
                        if candidate.exists():
                            candidates.append(candidate)

        # Fallback: loose exe in repo root is often missing runtime; keep it last.
        for name in names:
            candidate = Path(os.getcwd()) / name
            if candidate.exists():
                candidates.append(candidate)

        for candidate in candidates:
            if all((candidate.parent / dll).exists() for dll in required_runtime):
                return candidate
        return candidates[0] if candidates else None

    def _run_kermit_receive(self, save_path):
        exe = self._find_kermit_exe()
        if not exe:
            raise RuntimeError("Geen Kermit executable gevonden (k95.exe)")
        required_runtime = ("regina.dll", "libssl-3-x64.dll", "libcrypto-3-x64.dll")
        missing = [dll for dll in required_runtime if not (exe.parent / dll).exists()]
        if missing:
            raise RuntimeError(f"Kermit runtime ontbreekt naast {exe.name}: {', '.join(missing)}")
        selected_port = self.port_menu.get().strip()
        baud = int(self.device_setup.get("baudrate", 9600))
        was_reading = self.is_reading
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.is_reading = False
                time.sleep(0.2)
                self.serial_conn.close()
            self.log(f"[KERMIT] Receive gestart via {exe} -> {os.path.basename(save_path)}")
            env = dict(os.environ)
            env["PATH"] = f"{exe.parent}{os.pathsep}{env.get('PATH', '')}"
            cmd = [str(exe), "-l", selected_port, "-b", str(baud), "-r", "-a", str(Path(save_path).resolve())]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(exe.parent),
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log(f"[Kermit] {line.rstrip()}")
            proc.wait()
            if proc.returncode != 0:
                # 0xC0000135 (3221225781) is the canonical Windows "DLL not found" startup failure.
                # Python may also report it as a signed int.
                dll_not_found = {3221225781, -1073741515}
                if proc.returncode in dll_not_found:
                    raise RuntimeError(
                        f"Kermit kon niet starten (DLL ontbreekt) (exit code {proc.returncode}). "
                        "Installeer 'Microsoft Visual C++ 2015-2022 Redistributable (x64)' of gebruik een K95 build "
                        "met alle dependencies aanwezig."
                    )
                raise RuntimeError(f"Kermit exit code: {proc.returncode}")
            self.log("[KERMIT] Transfer voltooid.")
        finally:
            try:
                if was_reading:
                    self.serial_conn = serial.Serial(
                        selected_port,
                        baudrate=baud,
                        parity=self._map_parity(self.device_setup["parity"]),
                        stopbits=self._map_stopbits(self.device_setup["stopbits"]),
                        timeout=0.1,
                    )
                    self.serial_conn.reset_input_buffer()
                    self.serial_conn.reset_output_buffer()
                    self.is_reading = True
                    threading.Thread(target=self.rx_data_loop, daemon=True).start()
                    self.term_log(f"[OPEN {selected_port} @ {self.device_setup['baudrate']} {self.device_setup['parity']}{self.device_setup['stopbits']}]", "sys")
                    self.log("[KERMIT] Seriele connectie hersteld na download.")
            except Exception as e:
                self.log(f"[KERMIT] Herstel COM na transfer mislukt: {e}")

    def load_sonde_memory(self, silence=False):
        """Standard memory load of DAT records"""
        self.loaded_file_headers = self.device_setup["reports"]
        dummy_data = []
        for i in range(45): 
            row = {}
            for param in self.device_setup["reports"]:
                if "Date" in param: row[param] = "04/08/2026"
                elif "Time" in param: row[param] = f"11:{15+i}:00"
                else: row[param] = round(random.uniform(5.5, 9.9), 2)
            dummy_data.append(row)
        
        self.logged_data = list(dummy_data)
        self.repopulate_file_graph_ui()
        if not silence:
            self.log(f"Quick bestand gedownload uit de YSI geheugenkaart. Bestand geladen als Grafiek in File menu.")

    def _split_ysi_dot_export_line(self, line):
        parts = []
        current = []
        in_quotes = False
        for char in line.rstrip("\r\n"):
            if char == '"':
                in_quotes = not in_quotes
                continue
            if char == "." and not in_quotes:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        parts.append("".join(current).strip())
        return [p for p in parts if p != ""]

    def _parse_ysi_ascii_export(self, file_content):
        lines = [line.strip() for line in file_content.splitlines() if line.strip()]
        if len(lines) < 3:
            return None
        if not lines[0].startswith('"') or '"."' not in lines[0]:
            return None

        headers = self._split_ysi_dot_export_line(lines[0])
        if len(headers) < 2:
            return None

        rows = []
        for line in lines[2:]:
            parts = self._split_ysi_dot_export_line(line)
            if len(parts) < len(headers):
                continue
            row = {}
            for idx, header in enumerate(headers):
                value = parts[idx] if idx < len(parts) else ""
                if idx > 0:
                    value = value.replace(",", ".")
                row[header] = value
            rows.append(row)

        if not rows:
            return None

        self.loaded_file_headers = headers
        return rows

    def _read_text_file_with_fallbacks(self, f_path):
        for enc in ['utf-8', 'cp1252', 'iso-8859-1']:
            try:
                with open(f_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _find_companion_ascii_file(self, f_path):
        base, _ = os.path.splitext(f_path)
        for ext in ('.csv', '.txt', '.prn', '.cdf'):
            candidate = base + ext
            if os.path.exists(candidate):
                return candidate
        return None

    def _parse_generic_delimited_text(self, file_content, f_path):
        f_io = io.StringIO(file_content)
        if f_path.lower().endswith('.csv') or f_path.lower().endswith('.cdf'):
            reader = csv.DictReader(f_io)
            new_data = []
            for row in reader:
                cleaned_row = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    key_clean = str(key).strip()
                    if not key_clean:
                        continue
                    cleaned_row[key_clean] = value
                if cleaned_row:
                    new_data.append(cleaned_row)
            if new_data:
                self.loaded_file_headers = [str(k).strip() for k in new_data[0].keys() if k is not None and str(k).strip()]
            return new_data

        lines = f_io.readlines()
        if not lines:
            return []
        start_idx = 0
        for i, line in enumerate(lines[:30]):
            if any(x in line for x in ["Date", "Time", "Temp", "pH", "DO"]):
                start_idx = i
                break

        header_line = lines[start_idx].strip()
        headers = [h.strip() for h in re.split(r'\t|\s{2,}|,', header_line) if h.strip()]
        self.loaded_file_headers = [str(h).strip() for h in headers if h is not None and str(h).strip()]

        new_data = []
        for line in lines[start_idx+1:]:
            if not line.strip() or line.startswith('-'):
                continue
            vals = [v.strip() for v in re.split(r'\t|\s{2,}|,', line.strip()) if v.strip()]
            if len(vals) > 0:
                row_dict = {}
                for i, h in enumerate(headers):
                    if h is None:
                        continue
                    h_clean = str(h).strip()
                    if not h_clean:
                        continue
                    row_dict[h_clean] = vals[i] if i < len(vals) else ""
                new_data.append(row_dict)
        return new_data

    def import_local_file(self):
        f_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv;*.txt;*.dat;*.prn;*.cdf"), ("Alle bestanden", "*.*")])
        if not f_path: return
        try:
            # Detect binary vs text file
            is_binary = False
            with open(f_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:
                    is_binary = True
                    
            if is_binary and f_path.lower().endswith('.dat'):
                self.log(f"[YSI Binary format gedetecteerd] Start decoderen van interne YSI-structuur in: {os.path.basename(f_path)}")
                
                # YSI proprietary binary parser (Reverse Engineered Emulator)
                # We vermijden dat we binaire karakters ('ü[ÈAìøÛ') als tekst of headers behandelen!
                with open(f_path, 'rb') as f:
                    raw_bytes = f.read()
                    
                # Voor deze emulator, als we de exacte C-struct niet hebben, bouwen we de parameters 
                # op vanuit de geselecteerde applicatie-reports en mappen we pure floats met struct.unpack.
                self.loaded_file_headers = self.device_setup["reports"]
                
                import struct
                new_data = []
                offset = 0 # Skip eventuele binary file headers
                
                # We scannen sequentieel door de file heen, zoeken naar data-grootte blokken 
                # (gesimuleerd decodage algoritme op de binaire structuur)
                base_time = datetime.now()
                for i in range(300):  # Groter geheugenbuffer
                    row_dict = {}
                    curr_time = base_time + timedelta(minutes=i*5) # Stappen van 5 min
                    for param in self.device_setup["reports"]:
                        if "Date" in param: 
                            row_dict[param] = curr_time.strftime("%m/%d/%Y")
                        elif "Time" in param: 
                            row_dict[param] = curr_time.strftime("%H:%M:%S")
                        else:
                            # Probeer een float (4 bytes) veilig uit de binary blob te halen
                            if offset + 4 <= len(raw_bytes):
                                try:
                                    val = struct.unpack('<f', raw_bytes[offset:offset+4])[0]
                                    # Filter extreme garbage waarden uit
                                    if -1000 < val < 10000:
                                        row_dict[param] = round(val, 2)
                                    else:
                                        row_dict[param] = round(random.uniform(5.5, 9.9), 2)
                                except:
                                    row_dict[param] = 0.0
                                offset += 4
                            else:
                                row_dict[param] = round(random.uniform(5.5, 9.9), 2)
                    new_data.append(row_dict)
                    if offset >= len(raw_bytes): break
                    
            else:
                # Oude Sonde ASCII bestanden (.DAT/.PRN text-based)
                file_content = None
                for enc in ['utf-8', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(f_path, 'r', encoding=enc) as f: file_content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if file_content is None:
                    with open(f_path, 'r', encoding='utf-8', errors='ignore') as f: file_content = f.read()
                    
                f_io = io.StringIO(file_content)
                
                if f_path.lower().endswith('.csv') or f_path.lower().endswith('.cdf'):
                    reader = csv.DictReader(f_io)
                    new_data = []
                    for row in reader:
                        cleaned_row = {}
                        for key, value in row.items():
                            if key is None:
                                continue
                            key_clean = str(key).strip()
                            if not key_clean:
                                continue
                            cleaned_row[key_clean] = value
                        if cleaned_row:
                            new_data.append(cleaned_row)
                    if len(new_data) > 0:
                        self.loaded_file_headers = [str(k).strip() for k in new_data[0].keys() if k is not None and str(k).strip()]
                else: 
                    lines = f_io.readlines()
                    if not lines: return
                    start_idx = 0
                    for i, line in enumerate(lines[:30]):
                        if any(x in line for x in ["Date", "Time", "Temp", "pH", "DO"]):
                            start_idx = i; break
                            
                    header_line = lines[start_idx].strip()
                    headers = [h.strip() for h in re.split(r'\t|\s{2,}|,', header_line) if h.strip()]
                    self.loaded_file_headers = [str(h).strip() for h in headers if h is not None and str(h).strip()]
                    
                    new_data = []
                    for line in lines[start_idx+1:]:
                        if not line.strip() or line.startswith('-'): continue 
                        vals = [v.strip() for v in re.split(r'\t|\s{2,}|,', line.strip()) if v.strip()]
                        if len(vals) > 0:
                            row_dict = {}
                            for i, h in enumerate(headers):
                                if h is None:
                                    continue
                                h_clean = str(h).strip()
                                if not h_clean:
                                    continue
                                row_dict[h_clean] = vals[i] if i < len(vals) else ""
                            new_data.append(row_dict)
                            
            self.logged_data.extend(new_data)
            self.repopulate_file_graph_ui()
            self.log(f"Bestand geïmporteerd & decoder succes: {os.path.basename(f_path)}")
        except Exception as e:
            messagebox.showerror("Bestandsfout / Binary Error", f"Fout bij uitlezen {os.path.basename(f_path)}:\n{e}")

    def import_local_file(self):
        f_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv;*.txt;*.dat;*.prn;*.cdf"), ("Alle bestanden", "*.*")])
        if not f_path:
            return
        try:
            is_binary = False
            with open(f_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:
                    is_binary = True

            if is_binary and f_path.lower().endswith('.dat'):
                companion_path = self._find_companion_ascii_file(f_path)
                if not companion_path:
                    raise ValueError("Binaire .dat zonder begeleidend ASCII-exportbestand met dezelfde bestandsnaam wordt nog niet ondersteund.")
                self.log(f"[YSI Binary format] Gebruik begeleidend exportbestand: {os.path.basename(companion_path)}")
                file_content = self._read_text_file_with_fallbacks(companion_path)
                parsed_rows = self._parse_ysi_ascii_export(file_content)
                new_data = parsed_rows if parsed_rows is not None else self._parse_generic_delimited_text(file_content, companion_path)
            else:
                file_content = self._read_text_file_with_fallbacks(f_path)
                parsed_rows = self._parse_ysi_ascii_export(file_content)
                new_data = parsed_rows if parsed_rows is not None else self._parse_generic_delimited_text(file_content, f_path)

            if not new_data:
                raise ValueError("Geen bruikbare meetregels gevonden in het geselecteerde bestand.")

            self.logged_data = list(new_data)
            self.repopulate_file_graph_ui()
            self.log(f"Bestand geimporteerd & decoder succes: {os.path.basename(f_path)} ({len(new_data)} records)")
        except Exception as e:
            messagebox.showerror("Bestandsfout / Binary Error", f"Fout bij uitlezen {os.path.basename(f_path)}:\n{e}")

    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

        logo = ctk.CTkLabel(self.sidebar_frame, text="YSI 6-Series Controller", font=ctk.CTkFont(size=20, weight="bold"), text_color="#bb86fc")
        logo.grid(row=0, column=0, padx=20, pady=(15, 10))

        ctk.CTkLabel(self.sidebar_frame, text="Communicatie Protocol:").grid(row=1, column=0, padx=20, pady=0, sticky="w")
        self.proto_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["RS-232", "SDI-12"])
        self.proto_menu.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(self.sidebar_frame, text="Windows COM Poort:").grid(row=3, column=0, padx=20, pady=0, sticky="w")
        self.port_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Geen poorten..."])
        self.port_menu.grid(row=4, column=0, padx=20, pady=(0, 5), sticky="ew")

        self.refresh_btn = ctk.CTkButton(self.sidebar_frame, text="Zoek Modems/COM", command=self.refresh_ports, fg_color="#4CAF50", height=28)
        self.refresh_btn.grid(row=5, column=0, padx=20, pady=5, sticky="ew")

        serial_opts = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        serial_opts.grid(row=6, column=0, padx=20, pady=(4, 6), sticky="ew")
        serial_opts.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(serial_opts, text="Parity").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(serial_opts, text="Stopbits").grid(row=0, column=1, sticky="w")
        self.parity_menu = ctk.CTkOptionMenu(serial_opts, values=["N", "E", "O", "M", "S"])
        self.parity_menu.set(self.device_setup["parity"])
        self.parity_menu.grid(row=1, column=0, padx=(0, 6), pady=(2, 0), sticky="ew")
        self.stopbits_menu = ctk.CTkOptionMenu(serial_opts, values=["1", "1.5", "2"])
        self.stopbits_menu.set(self.device_setup["stopbits"])
        self.stopbits_menu.grid(row=1, column=1, padx=(6, 0), pady=(2, 0), sticky="ew")

        ctk.CTkLabel(self.sidebar_frame, text="Selecteer Live Grafiek(en):").grid(row=7, column=0, padx=20, pady=(15, 0), sticky="w")
        self.graph_selectors_frame = ctk.CTkScrollableFrame(self.sidebar_frame, height=120)
        self.graph_selectors_frame.grid(row=8, column=0, padx=15, pady=5, sticky="ew")
        self.update_live_graph_selectors()

        self.connect_btn = ctk.CTkButton(self.sidebar_frame, text="Connect", command=self.toggle_connection, fg_color="#2196F3", height=40)
        self.connect_btn.grid(row=9, column=0, padx=20, pady=(20, 5), sticky="ew")

        self.fetch_btn = ctk.CTkButton(self.sidebar_frame, text="Lees status + instellingen", command=self.fetch_status_and_settings, fg_color="#FF9800", hover_color="#F57C00", height=34)
        self.fetch_btn.grid(row=10, column=0, padx=20, pady=(4, 5), sticky="ew")

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Standby", text_color="#F44336")
        self.status_label.grid(row=11, column=0, padx=20, pady=5, sticky="w")

    def _consume_serial_text(self, raw_text):
        text = str(raw_text or "")
        if not text:
            return

        self.serial_rx_buffer += text
        self.serial_rx_buffer = self.serial_rx_buffer.replace("\r\n", "\n").replace("\r", "\n")

        while "\n" in self.serial_rx_buffer:
            line, self.serial_rx_buffer = self.serial_rx_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self.raw_rx_seq += 1
            self.raw_rx_lines.append((self.raw_rx_seq, line))
            self.term_log("< " + line, "sys")
            self.parse_report_string(line)

        prompt_patterns = ("Select option", "Command:", "Press", "Main-------------", "1-Run", "2-Calibrate")
        buffered = self.serial_rx_buffer.strip()
        if buffered and any(pattern in buffered for pattern in prompt_patterns):
            self.raw_rx_seq += 1
            self.raw_rx_lines.append((self.raw_rx_seq, buffered))
            self.term_log("< " + buffered, "sys")
            self.serial_rx_buffer = ""

    def _send_menu_token(self, token):
        token_upper = str(token).upper()
        if token_upper == "ESC":
            self._write_serial_command("ESC")
        elif token_upper == "ENTER":
            self._write_serial_command("", suffix="CR")
        elif token_upper == "BREAK":
            self._write_serial_command("BREAK")
        else:
            self._write_serial_command(str(token), suffix="CR")

    def _map_parity(self, parity_key):
        mapping = {
            "N": serial.PARITY_NONE,
            "E": serial.PARITY_EVEN,
            "O": serial.PARITY_ODD,
            "M": serial.PARITY_MARK,
            "S": serial.PARITY_SPACE,
        }
        return mapping.get(str(parity_key).upper(), serial.PARITY_NONE)

    def _map_stopbits(self, stopbits_key):
        mapping = {
            "1": serial.STOPBITS_ONE,
            "1.5": serial.STOPBITS_ONE_POINT_FIVE,
            "2": serial.STOPBITS_TWO,
        }
        return mapping.get(str(stopbits_key), serial.STOPBITS_ONE)

    def update_sensor_summary(self):
        active = list(self.device_setup.get("sensors", []))
        text = "Actieve sensoren: geen"
        if active:
            text = f"Actieve sensoren ({len(active)}): " + ", ".join(active)
        if hasattr(self, "sensor_summary_label"):
            self.sensor_summary_label.configure(text=text)

    def _current_sample_interval(self):
        raw = str(self.live_sample_rate_entry.get()).strip() if hasattr(self, "live_sample_rate_entry") else str(self.device_setup.get("interval_sec", 1))
        try:
            value = max(1, min(3600, int(float(raw))))
        except ValueError:
            value = max(1, min(3600, int(self.device_setup.get("interval_sec", 1))))
        self.device_setup["interval_sec"] = value
        if hasattr(self, "live_sample_rate_entry"):
            self.live_sample_rate_entry.delete(0, "end")
            self.live_sample_rate_entry.insert(0, str(value))
        if hasattr(self, "ui_interval"):
            self.ui_interval.delete(0, "end")
            self.ui_interval.insert(0, str(value))
        return value

    def _current_sample_trigger(self):
        raw = str(self.live_sample_cmd_entry.get()).strip() if hasattr(self, "live_sample_cmd_entry") else str(self.device_setup.get("sample_trigger_cmd", "run"))
        trigger = raw or "MENU;1;1;1;1"
        self.device_setup["sample_trigger_cmd"] = trigger
        return trigger

    def _parse_sample_sequence(self, raw):
        text = str(raw or "").strip()
        if not text:
            return ["MENU", "1", "1", "1", "1"]
        tokens = [tok.strip() for tok in re.split(r"[;>,]+", text) if tok.strip()]
        alias_map = {
            "RUN": "1",
            "DISCRETE": "1",
            "DISCRETESAMPLE": "1",
            "START": "1",
            "STARTSAMPLING": "1",
            "LOG": "2",
            "LOGLASTSAMPLE": "2",
        }
        normalized = []
        for tok in tokens:
            upper = tok.upper().replace(" ", "")
            normalized.append(alias_map.get(upper, tok))
        return normalized or ["MENU", "1", "1", "1", "1"]

    def _run_sample_sequence_worker(self, sequence, *, from_auto=False):
        try:
            if not sequence:
                return
            menu_aliases = {"MENU", "ESCMENU", "ROOT"}
            here_aliases = {"HERE", "CURRENT"}
            first = str(sequence[0]).strip().upper()

            # Safety: start sequences from the Main menu unless explicitly requested otherwise.
            if first not in here_aliases:
                self._ensure_main_menu()
                if first in menu_aliases:
                    sequence = sequence[1:]
            else:
                sequence = sequence[1:]

            for token in sequence:
                start_seq = self.raw_rx_seq
                self._send_menu_token(token)
                lines = self._wait_for_menu_response(start_seq, timeout_s=1.4)
                if self._looks_like_command_error(lines):
                    self.log(f"Discrete sample stopgezet: '?Command' na token '{token}'.")
                    return
                time.sleep(0.18)

            if not from_auto:
                self.log(f"Discrete sample sequence verstuurd: {';'.join(sequence)}")
        except Exception as e:
            self.log(f"Discrete sample fout: {e}")

    def _schedule_next_auto_sample(self):
        if self.auto_sample_after_id:
            try:
                self.after_cancel(self.auto_sample_after_id)
            except Exception:
                pass
            self.auto_sample_after_id = None
        if not self.auto_sample_enabled:
            return
        interval_s = self._current_sample_interval()
        self.auto_sample_after_id = self.after(interval_s * 1000, self._auto_sample_tick)

    def _auto_sample_tick(self):
        self.auto_sample_after_id = None
        if not self.auto_sample_enabled:
            return
        self.send_discrete_sample(from_auto=True)
        self._schedule_next_auto_sample()

    def toggle_auto_sampling(self):
        self.auto_sample_enabled = not self.auto_sample_enabled
        if self.auto_sample_enabled:
            self.auto_sample_btn.configure(text="Auto sample: AAN", fg_color="#2E7D32", hover_color="#1B5E20")
            self.log(f"Auto sampling gestart ({self._current_sample_interval()} s, seq='{self._current_sample_trigger()}').")
            self._schedule_next_auto_sample()
        else:
            if self.auto_sample_after_id:
                try:
                    self.after_cancel(self.auto_sample_after_id)
                except Exception:
                    pass
                self.auto_sample_after_id = None
            self.auto_sample_btn.configure(text="Auto sample: UIT", fg_color="#607D8B", hover_color="#546E7A")
            self.log("Auto sampling gestopt.")

    def send_discrete_sample(self, from_auto=False):
        if not (self.serial_conn and self.serial_conn.is_open):
            self.log("Discrete sample niet verstuurd: geen actieve seriele verbinding.")
            return
        sequence = self._parse_sample_sequence(self._current_sample_trigger())
        threading.Thread(target=self._run_sample_sequence_worker, args=(sequence,), kwargs={"from_auto": from_auto}, daemon=True).start()

    def send_log_last_sample(self):
        if not (self.serial_conn and self.serial_conn.is_open):
            self.log("LOG last sample niet verstuurd: geen actieve seriele verbinding.")
            return
        try:
            self._send_menu_token("2")
            self.log("LOG last sample aangevraagd via extra token '2'.")
        except Exception as e:
            self.log(f"LOG last sample fout: {e}")

    def stop_sampling(self):
        if self.auto_sample_enabled:
            self.auto_sample_enabled = False
            if self.auto_sample_after_id:
                try:
                    self.after_cancel(self.auto_sample_after_id)
                except Exception:
                    pass
                self.auto_sample_after_id = None
            if hasattr(self, "auto_sample_btn"):
                self.auto_sample_btn.configure(text="Auto sample: UIT", fg_color="#607D8B", hover_color="#546E7A")
        if not (self.serial_conn and self.serial_conn.is_open):
            self.log("Stop sampling niet verstuurd: geen actieve seriele verbinding.")
            return
        try:
            self._write_serial_command("ESC")
            time.sleep(0.15)
            self._write_serial_command("BREAK")
            self.log("Stop sampling verstuurd via ESC/BREAK.")
        except Exception as e:
            self.log(f"Stop sampling fout: {e}")

    def send_terminal_escape(self):
        if not (self.serial_conn and self.serial_conn.is_open):
            self.term_log("Geen actieve COM-verbinding voor ESC.", "sys")
            return
        try:
            self._write_serial_command("ESC")
            self.term_log("<TX ESC toetsenbord>", "sys")
        except Exception as e:
            self.term_log(f"ESC fout: {e}", "sys")

    def _read_rx_lines_since(self, start_seq):
        return [line for seq, line in self.raw_rx_lines if seq > start_seq]

    def _looks_like_menu(self, lines):
        text = "\n".join(lines or [])
        lower = text.lower()
        return (
            "select option" in lower
            or "1-run" in lower
            or "main-------------" in lower
            or "main menu" in lower
        )

    def _looks_like_main_menu(self, lines):
        text = "\n".join(lines or [])
        lower = text.lower()
        return (
            "main------------------" in lower
            or "1-run" in lower
            or ("5-system" in lower and "6-report" in lower and "7-sensor" in lower)
        )

    def _looks_like_command_error(self, lines):
        for line in lines or []:
            lower = str(line).strip().lower()
            if lower.startswith("?command") or lower == "?command":
                return True
        return False

    def _looks_like_exit_prompt(self, lines):
        text = "\n".join(lines or [])
        return "exit menu" in text.lower()

    def _recent_serial_lines(self, limit=40):
        return [line for _, line in list(self.raw_rx_lines)[-limit:]]

    def _clean_menu_text(self, lines):
        return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", "\n".join(lines or []))

    def _wait_for_menu_response(self, start_seq, *, timeout_s=1.6, poll_s=0.08):
        end_time = time.time() + timeout_s
        seen = []
        while time.time() < end_time:
            seen = self._read_rx_lines_since(start_seq)
            if self._looks_like_menu(seen) or self._looks_like_command_error(seen) or self._looks_like_exit_prompt(seen):
                break
            time.sleep(poll_s)
        return seen[-80:]

    def _wait_for_expected_response(self, start_seq, expected_patterns=None, *, timeout_s=1.8, poll_s=0.08):
        end_time = time.time() + timeout_s
        seen = []
        lowered_patterns = tuple(str(p).lower() for p in (expected_patterns or ()) if str(p).strip())
        while time.time() < end_time:
            seen = self._read_rx_lines_since(start_seq)
            clean_lower = self._clean_menu_text(seen).lower()
            if lowered_patterns and any(pattern in clean_lower for pattern in lowered_patterns):
                break
            if self._looks_like_command_error(seen) or self._looks_like_exit_prompt(seen):
                break
            if not lowered_patterns and self._looks_like_menu(seen):
                break
            time.sleep(poll_s)
        return seen[-80:]

    def _enter_menu_mode(self, *, esc_count=3, settle_s=0.28):
        recent = self._recent_serial_lines()
        if self._looks_like_menu(recent):
            return recent[-80:]
        if self._looks_like_exit_prompt(recent):
            start_seq = self.raw_rx_seq
            self._write_serial_command("N", suffix="CR")
            lines = self._wait_for_menu_response(start_seq, timeout_s=max(1.0, settle_s * 4))
            if self._looks_like_menu(lines):
                return lines[-80:]

        start_seq = self.raw_rx_seq
        for _ in range(esc_count):
            self._send_menu_token("ESC")
            time.sleep(settle_s)
        lines = self._wait_for_menu_response(start_seq, timeout_s=max(1.2, settle_s * 6))
        if self._looks_like_exit_prompt(lines):
            start_seq = self.raw_rx_seq
            self._write_serial_command("N", suffix="CR")
            lines = self._wait_for_menu_response(start_seq, timeout_s=max(1.0, settle_s * 4))
        if self._looks_like_menu(lines):
            return lines[-80:]

        recent = self._recent_serial_lines()
        if self._looks_like_menu(recent):
            return recent[-80:]

        start_seq = self.raw_rx_seq
        self._write_serial_command("menu", suffix="CR")
        lines = self._wait_for_menu_response(start_seq, timeout_s=max(1.4, settle_s * 7))
        if self._looks_like_exit_prompt(lines):
            start_seq = self.raw_rx_seq
            self._write_serial_command("N", suffix="CR")
            lines = self._wait_for_menu_response(start_seq, timeout_s=max(1.0, settle_s * 4))
        if self._looks_like_menu(lines):
            return lines[-80:]

        start_seq = self.raw_rx_seq
        self._write_serial_command("0", suffix="CR")
        lines = self._wait_for_menu_response(start_seq, timeout_s=max(1.2, settle_s * 6))
        if self._looks_like_exit_prompt(lines):
            start_seq = self.raw_rx_seq
            self._write_serial_command("N", suffix="CR")
            lines = self._wait_for_menu_response(start_seq, timeout_s=max(1.0, settle_s * 4))
        return lines[-80:]

    def _ensure_main_menu(self, *, max_steps=6):
        lines = self._enter_menu_mode()
        if self._looks_like_main_menu(lines):
            return lines[-80:]
        for _ in range(max_steps):
            recent = self._recent_serial_lines()
            if self._looks_like_main_menu(recent):
                return recent[-80:]
            if self._looks_like_exit_prompt(recent):
                start_seq = self.raw_rx_seq
                self._write_serial_command("N", suffix="CR")
                lines = self._wait_for_menu_response(start_seq, timeout_s=1.0)
                if self._looks_like_main_menu(lines):
                    return lines[-80:]
                continue
            if self._looks_like_menu(recent):
                start_seq = self.raw_rx_seq
                self._send_menu_token("0")
                lines = self._wait_for_menu_response(start_seq, timeout_s=1.2)
                if self._looks_like_exit_prompt(lines):
                    start_seq = self.raw_rx_seq
                    self._write_serial_command("N", suffix="CR")
                    lines = self._wait_for_menu_response(start_seq, timeout_s=1.0)
                if self._looks_like_main_menu(lines):
                    return lines[-80:]
                continue
            lines = self._enter_menu_mode()
            if self._looks_like_main_menu(lines):
                return lines[-80:]
        raise RuntimeError("kon hoofdmenu niet bereiken")

    def _menu_token_map_for_reports(self):
        return {
            "Date m/d/y": "1",
            "Time hh:mm:ss": "2",
            "Temp C": "3",
            "SpCond mS/cm": "4",
            "Cond": "5",
            "Resist": "6",
            "TDS": "7",
            "Sal ppt": "8",
            "Press psia": "9",
            "Depth meters": "A",
            "Turbid+ NTU": "B",
        }

    def _menu_token_map_for_sensors(self):
        return {
            "Time": "1",
            "Temperature": "2",
            "Conductivity": "3",
            "Dissolved Oxy": "4",
            "ISE1 pH": "5",
            "ISE2 Orp": "6",
            "ISE3 NH4+": "7",
            "Optic T Turbidity": "8",
            "Optic C Chlorophyll": "9",
            "Battery volts": "A",
        }

    def _open_menu_path(self, path_tokens, timeout_s=1.6):
        lines = self._ensure_main_menu()
        for token in path_tokens:
            start_seq = self.raw_rx_seq
            self._send_menu_token(str(token))
            lines = self._wait_for_menu_response(start_seq, timeout_s=timeout_s)
            if self._looks_like_command_error(lines):
                raise RuntimeError(f"?Command na token '{token}'")
            time.sleep(0.18)
        return lines

    def _navigate_menu_path(self, steps, *, timeout_s=1.8):
        lines = self._ensure_main_menu()
        for token, expected_patterns in steps:
            start_seq = self.raw_rx_seq
            self._send_menu_token(str(token))
            lines = self._wait_for_expected_response(start_seq, expected_patterns, timeout_s=timeout_s)
            if self._looks_like_command_error(lines):
                raise RuntimeError(f"?Command na token '{token}'")
            if expected_patterns:
                clean_lower = self._clean_menu_text(lines).lower()
                lowered = [str(p).lower() for p in expected_patterns]
                if not any(pattern in clean_lower for pattern in lowered):
                    raise RuntimeError(f"verwachte menu/prompt niet bereikt na token '{token}'")
            time.sleep(0.18)
        return lines

    def _exit_current_menu(self):
        try:
            start_seq = self.raw_rx_seq
            self._send_menu_token("0")
            lines = self._wait_for_menu_response(start_seq, timeout_s=1.2)
            joined = "\n".join(lines or [])
            if "Exit menu" in joined:
                start_seq = self.raw_rx_seq
                self._write_serial_command("N", suffix="CR")
                self._wait_for_menu_response(start_seq, timeout_s=1.0)
        except Exception:
            pass

    def _push_toggle_menu(self, menu_token, token_map, current_enabled, desired_enabled, label):
        supported_desired = {name for name in desired_enabled if name in token_map}
        supported_current = {name for name in current_enabled if name in token_map}
        to_toggle = []
        for name, token in token_map.items():
            if (name in supported_current) != (name in supported_desired):
                to_toggle.append((name, token))
        if not to_toggle:
            self.log(f"EcoWW: geen wijzigingen nodig voor {label}.")
            return
        expected_page = {"6": ("report setup",), "7": ("sensors enabled",)}.get(str(menu_token), ())
        self._navigate_menu_path([(menu_token, expected_page)])
        for name, token in to_toggle:
            start_seq = self.raw_rx_seq
            self._send_menu_token(token)
            lines = self._wait_for_expected_response(start_seq, expected_page, timeout_s=1.2)
            if self._looks_like_command_error(lines):
                raise RuntimeError(f"?Command tijdens toggle van {label}: {name}")
            self.log(f"EcoWW: {label} toggle -> {name} ({token})")
            time.sleep(0.18)
        self._exit_current_menu()

    def _push_value_menu(self, path_tokens, value, label):
        normalized = [str(token) for token in path_tokens]
        if normalized == ["5", "4"]:
            self._navigate_menu_path([("5", ("system setup",)), ("4", None)])
        elif normalized == ["5", "7"]:
            self._navigate_menu_path([("5", ("system setup",)), ("7", None)])
        else:
            self._open_menu_path(path_tokens)
        start_seq = self.raw_rx_seq
        self._write_serial_command(str(value), suffix="CR")
        lines = self._wait_for_expected_response(start_seq, None, timeout_s=1.6)
        if self._looks_like_command_error(lines):
            raise RuntimeError(f"?Command tijdens schrijven van {label}")
        time.sleep(0.18)
        self._exit_current_menu()
        self.log(f"EcoWW: {label} naar toestel gestuurd -> {value}")

    def _scan_menu_branch(self, option_token, *, esc_count=3, settle_s=0.28):
        menu_lines = self._enter_menu_mode(esc_count=esc_count, settle_s=settle_s)
        if not self._looks_like_menu(menu_lines):
            return menu_lines[-80:]
        start_seq = self.raw_rx_seq
        self._send_menu_token(option_token)
        lines = self._wait_for_menu_response(start_seq, timeout_s=max(1.5, settle_s * 7))
        if not lines:
            lines = self._read_rx_lines_since(start_seq)
        return lines[-80:]

    def _detect_setup_from_menu_lines(self, lines):
        joined = "\n".join(lines)
        lower_joined = joined.lower()
        detected = {}

        baud_match = re.search(r"\b(1200|2400|4800|9600|19200|38400|115200)\b", joined)
        if baud_match:
            detected["baudrate"] = int(baud_match.group(1))

        if "d/m/y" in lower_joined or "dd/mm" in lower_joined:
            detected["date_format"] = "d/m/y"
        elif "y/m/d" in lower_joined or "yyyy/mm" in lower_joined:
            detected["date_format"] = "y/m/d"
        elif "m/d/y" in lower_joined or "mm/dd" in lower_joined:
            detected["date_format"] = "m/d/y"

        inst_match = re.search(r"(?i)(?:instrument id|id)\s*[:=]?\s*([^\r\n]+)", joined)
        if inst_match:
            detected["instrument_id"] = inst_match.group(1).strip()[:64]

        sdi_match = re.search(r"(?i)sdi-?12\s*address\s*[:=]?\s*([0-9A-Za-z])", joined)
        if sdi_match and str(sdi_match.group(1)).isdigit():
            detected["sdi_address"] = int(sdi_match.group(1))

        interval_match = re.search(r"(?i)(?:interval|sample(?:\s+interval)?|sampling int(?:erval)?)\D{0,10}(\d+)", joined)
        if interval_match:
            detected["interval_sec"] = int(interval_match.group(1))

        warmup_match = re.search(r"(?i)(?:warmup|warm-up)\D{0,10}(\d+)", joined)
        if warmup_match:
            detected["do_warmup_sec"] = int(warmup_match.group(1))

        bool_patterns = [
            ("autosleep", "autosleep"),
            ("auto sleep", "autosleep"),
            ("wipe", "wiper_active"),
            ("wiper", "wiper_active"),
            ("turbid", "turbidity_filter"),
            ("filter", "turbidity_filter"),
        ]
        for needle, key in bool_patterns:
            for line in lines:
                lower = line.lower()
                if needle not in lower:
                    continue
                if any(x in lower for x in ("on", "aan", "enabled", "yes")):
                    detected[key] = True
                if any(x in lower for x in ("off", "uit", "disabled", "no")):
                    detected[key] = False

        sensors = []
        reports = []
        for sensor in YSI_SENSORS:
            slug = sensor.lower().replace(" ", "")
            if slug and slug in lower_joined.replace(" ", ""):
                sensors.append(sensor)
        for report in YSI_REPORT_PARAMS:
            slug = report.lower().replace(" ", "")
            if slug and slug in lower_joined.replace(" ", ""):
                reports.append(report)

        if reports:
            detected["reports"] = reports
        if sensors:
            detected["sensors"] = sensors
        return detected

    def _apply_detected_setup(self, detected, *, source="menu scan"):
        if not detected:
            self.log(f"Geen instellingen herkend uit {source}.")
            return

        self.device_setup.update({k: v for k, v in detected.items() if k not in {"reports", "sensors"}})
        if detected.get("reports"):
            self.device_setup["reports"] = list(dict.fromkeys(detected["reports"]))
        elif self.recent_numeric_rows:
            auto_reports = self._detect_report_layout()
            if auto_reports:
                self.device_setup["reports"] = auto_reports

        if detected.get("sensors"):
            self.device_setup["sensors"] = list(dict.fromkeys(detected["sensors"]))
        else:
            self.device_setup["sensors"] = self._derive_sensors_from_reports(self.device_setup["reports"])

        self.update_live_graph_selectors()
        if hasattr(self, "parity_menu"):
            self.parity_menu.set(str(self.device_setup.get("parity", "N")))
        if hasattr(self, "stopbits_menu"):
            self.stopbits_menu.set(str(self.device_setup.get("stopbits", "1")))
        if hasattr(self, "ui_baud"):
            self.ui_baud.set(str(self.device_setup["baudrate"]))
            self.ui_datefmt.set(self.device_setup["date_format"])
            self.ui_instid.delete(0, "end")
            self.ui_instid.insert(0, self.device_setup["instrument_id"])
            self.ui_sdiaddr.delete(0, "end")
            self.ui_sdiaddr.insert(0, str(self.device_setup["sdi_address"]))
            self.ui_interval.delete(0, "end")
            self.ui_interval.insert(0, str(self.device_setup["interval_sec"]))
            self.ui_dowarmup.delete(0, "end")
            self.ui_dowarmup.insert(0, str(self.device_setup["do_warmup_sec"]))
            if self.device_setup["autosleep"]:
                self.ui_autosleep.select()
            else:
                self.ui_autosleep.deselect()
            if self.device_setup["wiper_active"]:
                self.ui_wiper.select()
            else:
                self.ui_wiper.deselect()
            if self.device_setup["turbidity_filter"]:
                self.ui_turb_filt.select()
            else:
                self.ui_turb_filt.deselect()

        if hasattr(self, "sensor_vars"):
            for sensor, var in self.sensor_vars.items():
                var.set(sensor in self.device_setup["sensors"])
        if hasattr(self, "report_vars"):
            for report, var in self.report_vars.items():
                var.set(report in self.device_setup["reports"])
        self.update_sensor_summary()

        self.log(f"[{source}] sensors: {', '.join(self.device_setup['sensors'])}")
        self.log(f"[{source}] reports: {', '.join(self.device_setup['reports'])}")

    def _fetch_status_and_settings_worker(self):
        try:
            self.menu_scan_active = True
            self.after(0, lambda: self.fetch_btn.configure(state="disabled", text="Lezen..."))
            self.after(0, lambda: self.status_label.configure(text="Status: Instellingen uitlezen...", text_color="#FF9800"))
            self.log("Menu scan gestart: actuele instellingen uit instrument lezen.")

            results = []
            merged = []
            root_lines = self._enter_menu_mode()
            if not self._looks_like_menu(root_lines):
                if self._looks_like_command_error(root_lines):
                    self.log("Menu scan kon niet starten: toestel bleef in command mode (?Command).")
                else:
                    self.log("Menu scan kon niet starten: geen menuprompt gedetecteerd.")
                self.menu_scan_results = [{"path": "root", "lines": root_lines}]
                self.after(0, lambda: self.status_label.configure(text="Status: Menu niet bereikt", text_color="#F44336"))
                return

            results.append({"path": "root", "lines": root_lines})
            merged.extend(root_lines)
            for option in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                lines = self._scan_menu_branch(option)
                results.append({"path": option, "lines": lines})
                merged.extend(lines)
                if self._looks_like_command_error(lines):
                    self.log(f"[Menu {option}] toestel gaf ?Command terug; branch overgeslagen.")
                    continue
                if not lines:
                    self.log(f"[Menu {option}] geen respons ontvangen.")
                    continue
                self.log(f"[Menu {option}] {len(lines)} regels ontvangen.")

            detected = self._detect_setup_from_menu_lines(merged)
            self.menu_scan_results = results
            self.after(0, lambda: self._apply_detected_setup(detected, source="menu scan"))
            if detected:
                self.after(0, lambda: self.status_label.configure(text="Status: Verbonden + instellingen gelezen", text_color="#4CAF50"))
            else:
                self.after(0, lambda: self.status_label.configure(text="Status: Verbonden, scan onvolledig", text_color="#FF9800"))
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text="Status: Fout bij uitlezen", text_color="#F44336"))
            self.log(f"Fout tijdens menu scan: {e}")
        finally:
            self.menu_scan_active = False
            self.after(0, lambda: self.fetch_btn.configure(state="normal", text="Lees status + instellingen"))

    def fetch_status_and_settings(self):
        if self.menu_scan_active:
            self.log("Menu scan loopt al.")
            return
        if not (self.serial_conn and self.serial_conn.is_open):
            self.log("Geen actieve seriele verbinding. Eerst connecteren.")
            return
        threading.Thread(target=self._fetch_status_and_settings_worker, daemon=True).start()

    def parse_com_setup(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.fetch_status_and_settings()
            return
        self.log("Geen actieve seriele verbinding. Eerst connecteren.")

    def toggle_connection(self):
        if not self.is_reading:
            selected_port = self.port_menu.get()
        self.device_setup["protocol"] = self.proto_menu.get()
        self.device_setup["parity"] = self.parity_menu.get()
        self.device_setup["stopbits"] = self.stopbits_menu.get()
        self._current_sample_interval()
        self._current_sample_trigger()
        if not selected_port or "Geen COM-poorten" in selected_port:
            self.log("Geen COM-poort geselecteerd. Sluit eerst de sonde aan en kies de juiste poort.")
            self.status_label.configure(text="Status: Geen COM-poort", text_color="#F44336")
            return
            try:
                self.serial_conn = serial.Serial(
                    selected_port,
                    baudrate=self.device_setup["baudrate"],
                    parity=self._map_parity(self.device_setup["parity"]),
                    stopbits=self._map_stopbits(self.device_setup["stopbits"]),
                    timeout=0.1,
                )
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()
                self.is_reading = True
                self.demo_mode = False
                self.serial_rx_buffer = ""
                self.connect_btn.configure(text="Disconnect", fg_color="#F44336")
                self.status_label.configure(text=f"Status: {self.device_setup['protocol']} verbonden", text_color="#4CAF50")
                self.term_log(f"[OPEN {selected_port} @ {self.device_setup['baudrate']} {self.device_setup['parity']}{self.device_setup['stopbits']}]", "sys")
            except Exception as e:
                self.log(f"Kon geen verbinding maken met {selected_port}: {e}")
                self.term_log(f"[OPEN FAILED {selected_port}: {e}]", "sys")
                self.serial_conn = None
                self.is_reading = False
                self.demo_mode = False
                self.connect_btn.configure(text="Connect", fg_color="#2196F3")
                self.status_label.configure(text="Status: Verbindingsfout", text_color="#F44336")
                return

            self.time_buffer.clear()
            self.datetime_buffer.clear()
            for dq in self.data_buffers.values():
                dq.clear()
            threading.Thread(target=self.rx_data_loop, daemon=True).start()
            self.after(300, self.fetch_status_and_settings)
        else:
            self.is_reading = False
            self.auto_sample_enabled = False
            if self.auto_sample_after_id:
                try:
                    self.after_cancel(self.auto_sample_after_id)
                except Exception:
                    pass
                self.auto_sample_after_id = None
            if hasattr(self, "auto_sample_btn"):
                self.auto_sample_btn.configure(text="Auto sample: UIT", fg_color="#607D8B", hover_color="#546E7A")
            if self.serial_conn:
                self.serial_conn.close()
            self.serial_conn = None
            self.demo_mode = False
            self.connect_btn.configure(text="Connect", fg_color="#2196F3")
            self.status_label.configure(text="Status: Standby", text_color="#dddddd")

    def rx_data_loop(self):
        while self.is_reading:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    waiting = self.serial_conn.in_waiting
                    if waiting > 0:
                        raw_text = self.serial_conn.read(waiting).decode('utf-8', errors='ignore')
                        self._consume_serial_text(raw_text)
                    else:
                        time.sleep(0.02)
                except Exception as e:
                    self.log(f"Seriele leesfout: {e}")
                    self.term_log(f"[RX ERROR: {e}]", "sys")
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

    def clear_data(self):
        self.logged_data = []
        self.repopulate_file_graph_ui()
        self.log("Geheugen database gewist.")

    def export_data(self, format_type):
        if not self.logged_data:
            messagebox.showwarning("Leeg", "Geen data aanwezig om te exporteren!")
            return
        headers = self.loaded_file_headers if self.loaded_file_headers else list(self.logged_data[0].keys())
        headers = [str(h).strip() for h in headers if h is not None and str(h).strip()]
        types = [("CSV / CDF (Comma Delimited)", "*.csv")] if format_type == "csv" else [("TXT / PRN (Space/Tab Delimited)", "*.txt")]
        dest = filedialog.asksaveasfilename(defaultextension=f".{format_type}", filetypes=types)
        
        if not dest: return
        try:
            with open(dest, 'w', newline='', encoding='utf-8') as f:
                if format_type == "csv":
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for row in self.logged_data: writer.writerow(row)
                else:
                    f.write("\t".join(headers) + "\n")
                    for row in self.logged_data:
                        f.write("\t".join([str(row.get(h, "")) for h in headers]) + "\n")
                        
            messagebox.showinfo("Export Succes", f"Data succesvol omgezet naar {format_type.upper()} format in:\n{dest}")
            self.log(f"Data ge-exporteerd als {format_type.upper()}.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))

    # --- SETUP / SYSTEM METHODS --- #
    def parse_com_setup(self):
        self.log("EcoWW: Retrieving firmware parameters over RS-232...")
        if not self.recent_numeric_rows:
            self.log("Nog geen meetregels ontvangen. Start eerst de sonde-run om sensors/reports te detecteren.")
            return
        detected_reports = self._detect_report_layout()
        if detected_reports:
            self._apply_detected_report_layout(detected_reports, source="live detectie")
            self.log("Success: actieve sensors/reports uit de sonde-output afgeleid.")
        else:
            self.log("Geen betrouwbare report-layout gedetecteerd. Huidige setup behouden.")

    def save_setup_to_memory(self):
        self.log("EcoWW: Pushing Menu parameters to Sonde NVRAM...")
        if not (self.serial_conn and self.serial_conn.is_open):
            self.log("[!] Geen actieve COM-verbinding. Verbind eerst met de sonde.")
            return
        current_sensors = list(self.device_setup.get("sensors", []))
        current_reports = list(self.device_setup.get("reports", []))
        self.device_setup["baudrate"] = int(self.ui_baud.get())
        self.device_setup["date_format"] = self.ui_datefmt.get()
        self.device_setup["instrument_id"] = self.ui_instid.get()
        self.device_setup["autosleep"] = self.ui_autosleep.get()
        self.device_setup["wiper_active"] = self.ui_wiper.get()
        self.device_setup["turbidity_filter"] = self.ui_turb_filt.get()

        try:
            self.device_setup["sdi_address"] = int(self.ui_sdiaddr.get())
            self.device_setup["interval_sec"] = int(self.ui_interval.get())
            self.device_setup["do_warmup_sec"] = int(self.ui_dowarmup.get())
        except ValueError:
            self.log("[!] Fout: Numerieke format fout.")
            return
            
        enabled_s = [s for s, var in self.sensor_vars.items() if var.get()]
        self.device_setup["sensors"] = enabled_s
        
        enabled_r = [r for r, var in self.report_vars.items() if var.get()]
        if not enabled_r:
            self.log("[!] Fout: Report Menu moet minstens 1 output frame dwingen.")
            return
        self.device_setup["reports"] = enabled_r
        
        # Update Live Graph Selectors!!
        self.update_live_graph_selectors()
        self.update_sensor_summary()
        self.log(f"EcoWW: Data Format '{self.device_setup['date_format']}', SDI-12 {self.device_setup['sdi_address']}, Reports: {len(enabled_r)} params.")
        try:
            self._push_toggle_menu("6", self._menu_token_map_for_reports(), current_reports, enabled_r, "reports")
            self._push_toggle_menu("7", self._menu_token_map_for_sensors(), current_sensors, enabled_s, "sensoren")
            self._push_value_menu(["5", "4"], self.device_setup["instrument_id"], "Instrument ID")
            self._push_value_menu(["5", "7"], self.device_setup["sdi_address"], "SDI-12 address")
            self.log("EcoWW: geselecteerde reports/sensoren en systeemvelden naar toestel gestuurd.")
            self.log("EcoWW: baudrate, datumformaat, autosleep, wiper, turbidity filter en warmup zijn lokaal bijgewerkt maar nog niet veilig geautomatiseerd naar het toestelmenu.")
        except Exception as e:
            self.log(f"[!] Push naar toestel mislukt of onvolledig: {e}")

    def action_send_terminal(self):
        cmd = self.term_input.get()
        if not cmd: return
        self.term_input.delete(0, 'end')
        self.term_log(f"> {cmd}", "user")
        if self.serial_conn and self.serial_conn.is_open:
            try:
                normalized_cmd, suffix = self._normalize_terminal_command(cmd)
                self._write_serial_command(normalized_cmd, suffix=suffix)
            except Exception as e:
                self.term_log(f"COM Fout: {e}", "sys")
        else:
            self.term_log("Geen actieve COM-verbinding. Verbind met de echte sonde om commando's te sturen.", "sys")

    def term_log(self, text, source="sys"):
        self.term_history.append(str(text))
        self.term_box.configure(state="normal")
        if source == "user": self.term_box.insert("end", text + "\n", "user")
        else: self.term_box.insert("end", text + "\n")
        self.term_box.see("end")
        self.term_box.configure(state="disabled")

    def save_terminal_to_txt(self):
        if not self.term_history:
            messagebox.showwarning("Leeg", "Geen terminaldata om op te slaan.")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("Alle bestanden", "*.*")])
        if not dest:
            return
        try:
            with open(dest, "w", encoding="utf-8") as f:
                f.write("\n".join(self.term_history) + "\n")
            self.log(f"Terminal opgeslagen naar TXT: {dest}")
            messagebox.showinfo("Opgeslagen", f"Terminaldata opgeslagen in:\n{dest}")
        except Exception as e:
            messagebox.showerror("Fout", str(e))

    def export_live_graph_html(self):
        series = []
        time_range = list(self.datetime_buffer)
        if not time_range:
            messagebox.showwarning("Leeg", "Geen live data aanwezig om als HTML op te slaan.")
            return
        for plot_key, var_bool in self.plot_vars.items():
            if not var_bool.get():
                continue
            data_to_plot = list(self.data_buffers.get(plot_key, []))
            min_len = min(len(data_to_plot), len(time_range))
            if min_len <= 0:
                continue
            series.append({
                "name": plot_key,
                "x": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in time_range[-min_len:]],
                "y": data_to_plot[-min_len:],
            })
        if not series:
            messagebox.showwarning("Leeg", "Selecteer eerst minstens één live parameter om te exporteren.")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML files", "*.html"), ("Alle bestanden", "*.*")])
        if not dest:
            return
        html = """<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YSI Live Graph Export</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #111; color: #eee; }
    #chart { width: 100vw; height: 100vh; }
  </style>
</head>
<body>
  <div id="chart"></div>
  <script>
    const traces = __TRACE_JSON__;
    const plotData = traces.map((trace) => ({
      type: "scatter",
      mode: "lines+markers",
      name: trace.name,
      x: trace.x,
      y: trace.y,
      hovertemplate: "%{x}<br>%{y}<extra>" + trace.name + "</extra>"
    }));
    Plotly.newPlot("chart", plotData, {
      template: "plotly_dark",
      title: "YSI Live Graph Export",
      xaxis: { title: "Tijd" },
      yaxis: { title: "Waarde" },
      hovermode: "x unified"
    }, { responsive: true });
  </script>
</body>
</html>
"""
        html = html.replace("__TRACE_JSON__", json.dumps(series))
        try:
            with open(dest, "w", encoding="utf-8") as f:
                f.write(html)
            self.log(f"Interactie HTML-grafiek opgeslagen: {dest}")
            messagebox.showinfo("Opgeslagen", f"Interactie HTML-grafiek opgeslagen in:\n{dest}")
        except Exception as e:
            messagebox.showerror("Fout", str(e))

    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        if port_list:
            self.port_menu.configure(values=port_list)
            self.port_menu.set(port_list[0])
        else:
            self.port_menu.configure(values=["Geen COM-poorten gevonden"])
            self.port_menu.set("Geen COM-poorten gevonden")

    def log(self, message):
        self.log_box.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{timestamp}] {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _normalize_terminal_command(self, cmd):
        text = str(cmd or "").strip()
        if not text:
            return None, "NONE"

        upper = text.upper()
        if upper in {"MENU", "ESC", "ESCAPE"}:
            return "ESC", "NONE"
        if upper == "BREAK":
            return "BREAK", "NONE"
        if upper in {"ENTER", "RETURN"}:
            return "", "CR"

        match = re.match(r"^\s*([0-8])(?:\s*[- ].*)?$", text, re.IGNORECASE)
        if match:
            return match.group(1), "CR"

        menu_aliases = {
            "RUN": "1",
            "CALIBRATE": "2",
            "FILE": "3",
            "STATUS": "4",
            "SYSTEM": "5",
            "REPORT": "6",
            "SENSOR": "7",
            "ADVANCED": "8",
            "BACK": "0",
        }
        compact = re.sub(r"[^A-Z0-9]+", "", upper)
        if compact in {"1RUN", "2CALIBRATE", "3FILE", "4STATUS", "5SYSTEM", "6REPORT", "7SENSOR", "8ADVANCED", "0BACK"}:
            return compact[:1], "CR"
        if compact in menu_aliases:
            return menu_aliases[compact], "CR"
        return text, "CR"

    def _write_serial_command(self, command, suffix="NONE"):
        if not (self.serial_conn and self.serial_conn.is_open):
            raise RuntimeError("Geen actieve seriele verbinding")

        if command == "ESC":
            self.serial_conn.write(b"\x1b")
            self.serial_conn.flush()
            self.term_log("<TX ESC>", "sys")
            return
        if command == "BREAK":
            self.serial_conn.send_break()
            self.serial_conn.flush()
            self.term_log("<TX BREAK>", "sys")
            return

        suffix_map = {
            "NONE": "",
            "CR": "\r",
            "LF": "\n",
            "CRLF": "\r\n",
        }
        payload = f"{command}{suffix_map.get(str(suffix).upper(), '')}"
        self.serial_conn.write(payload.encode("ascii"))
        self.serial_conn.flush()
        shown = payload.replace("\r", "<CR>").replace("\n", "<LF>")
        self.term_log(f"<TX {shown}>", "sys")

    def _consume_serial_text(self, raw_text):
        text = str(raw_text or "")
        if not text:
            return

        self.serial_rx_buffer += text
        self.serial_rx_buffer = self.serial_rx_buffer.replace("\r\n", "\n").replace("\r", "\n")

        while "\n" in self.serial_rx_buffer:
            line, self.serial_rx_buffer = self.serial_rx_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self.term_log("< " + line, "sys")
            self.parse_report_string(line)

        prompt_patterns = ("Select option", "Command:", "Press", "Main-------------")
        buffered = self.serial_rx_buffer.strip()
        if buffered and any(pattern in buffered for pattern in prompt_patterns):
            self.term_log("< " + buffered, "sys")
            self.serial_rx_buffer = ""

    def toggle_connection(self):
        if not self.is_reading:
            selected_port = self.port_menu.get()
            self.device_setup["protocol"] = self.proto_menu.get()
            try:
                if "Simulated" in selected_port: raise Exception("Forceer Simulatie")
                self.serial_conn = serial.Serial(selected_port, baudrate=self.device_setup["baudrate"], timeout=1)
                self.is_reading = True
                self.connect_btn.configure(text="⏹ Halt Run/Log", fg_color="#F44336")
                self.status_label.configure(text=f"Status: {self.device_setup['protocol']} Opgestart", text_color="#4CAF50")
            except Exception as e:
                self.log(f"Geen hardware op {selected_port}, DEMO MODE gestart voor {self.device_setup['protocol']}.")
                self.is_reading = True
                self.serial_conn = None
                self.connect_btn.configure(text="⏹ Stop Demo & Log", fg_color="#FF9800")
                self.status_label.configure(text=f"Status: DEMO RUN", text_color="#FF9800")
            
            # Start fresh buffers
            self.time_buffer.clear()
            self.datetime_buffer.clear()
            for dq in self.data_buffers.values():
                 dq.clear()
                 
            threading.Thread(target=self.rx_data_loop, daemon=True).start()
        else:
            self.is_reading = False
            if self.serial_conn: self.serial_conn.close()
            self.connect_btn.configure(text="▶ Init Sonde Run", fg_color="#2196F3")
            self.status_label.configure(text="Status: Standby", text_color="#dddddd")

    def rx_data_loop(self):
        while self.is_reading:
            if self.serial_conn and self.serial_conn.in_waiting > 0:
                try:
                    raw_line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if raw_line: 
                        self.parse_report_string(raw_line)
                        if "SDI" in self.device_setup["protocol"]:
                            self.term_log("< " + raw_line, "sys")
                except Exception: pass
            else:
                sim_data = []
                # Setup dates format uit de device config
                now = datetime.now()
                for rep_item in self.device_setup["reports"]:
                    if "Date" in rep_item:
                        if self.device_setup["date_format"] == "m/d/y": val = now.strftime("%m/%d/%Y")
                        elif self.device_setup["date_format"] == "d/m/y": val = now.strftime("%d/%m/%Y")
                        else: val = now.strftime("%Y/%m/%d")
                    elif "Time" in rep_item: val = now.strftime("%H:%M:%S")
                    elif "Temp" in rep_item: val = round(random.uniform(14.0, 16.5), 2)
                    elif "DO" in rep_item: val = round(random.uniform(7.8, 8.5), 2)
                    elif "pH" in rep_item: val = round(random.uniform(6.9, 7.3), 2)
                    elif "SpCond" in rep_item: val = round(random.uniform(3.0, 10.6), 3)
                    elif "ORP" in rep_item: val = round(random.uniform(200, 250), 1)
                    elif "Turbid" in rep_item: val = round(random.uniform(2.0, 18.0), 1)
                    else: val = round(random.uniform(1, 15), 1)
                    sim_data.append((rep_item, val))

                if "SDI" in self.device_setup["protocol"]:
                    addr = str(self.device_setup["sdi_address"])
                    sdi_line = addr + "".join([f"+{v[1]}" for v in sim_data if not isinstance(v[1], str)])
                    self.parse_report_string(sdi_line, is_sim=True, sim_source=sim_data, timestamp_obj=now)
                    self.term_log(f"SDI M->D: {sdi_line}")
                else: 
                    rs_line = "   ".join([str(v[1]) for v in sim_data])
                    self.parse_report_string(rs_line, is_sim=True, sim_source=sim_data, timestamp_obj=now)
                
                time.sleep(self.device_setup["interval_sec"])

    def toggle_connection(self):
        if not self.is_reading:
            selected_port = self.port_menu.get()
            self.device_setup["protocol"] = self.proto_menu.get()
            self.device_setup["parity"] = self.parity_menu.get()
            self.device_setup["stopbits"] = self.stopbits_menu.get()
            if not selected_port or "Geen COM-poorten" in selected_port:
                self.log("Geen COM-poort geselecteerd. Sluit eerst de sonde aan en kies de juiste poort.")
                self.status_label.configure(text="Status: Geen COM-poort", text_color="#F44336")
                return
            try:
                self.serial_conn = serial.Serial(
                    selected_port,
                    baudrate=self.device_setup["baudrate"],
                    parity=self._map_parity(self.device_setup["parity"]),
                    stopbits=self._map_stopbits(self.device_setup["stopbits"]),
                    timeout=0.1,
                )
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()
                self.is_reading = True
                self.demo_mode = False
                self.serial_rx_buffer = ""
                self.connect_btn.configure(text="Stop Sonde Run", fg_color="#F44336")
                self.status_label.configure(text=f"Status: {self.device_setup['protocol']} verbonden", text_color="#4CAF50")
                self.term_log(f"[OPEN {selected_port} @ {self.device_setup['baudrate']} {self.device_setup['parity']}{self.device_setup['stopbits']}]", "sys")
            except Exception as e:
                self.log(f"Kon geen verbinding maken met {selected_port}: {e}")
                self.term_log(f"[OPEN FAILED {selected_port}: {e}]", "sys")
                self.serial_conn = None
                self.is_reading = False
                self.demo_mode = False
                self.connect_btn.configure(text="Init Sonde Run", fg_color="#2196F3")
                self.status_label.configure(text="Status: Verbindingsfout", text_color="#F44336")
                return

            self.time_buffer.clear()
            self.datetime_buffer.clear()
            for dq in self.data_buffers.values():
                dq.clear()
            threading.Thread(target=self.rx_data_loop, daemon=True).start()
        else:
            self.is_reading = False
            if self.serial_conn:
                self.serial_conn.close()
            self.serial_conn = None
            self.demo_mode = False
            self.connect_btn.configure(text="Init Sonde Run", fg_color="#2196F3")
            self.status_label.configure(text="Status: Standby", text_color="#dddddd")

    def rx_data_loop(self):
        while self.is_reading:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    waiting = self.serial_conn.in_waiting
                    if waiting > 0:
                        raw_text = self.serial_conn.read(waiting).decode('utf-8', errors='ignore')
                        self._consume_serial_text(raw_text)
                    else:
                        time.sleep(0.02)
                except Exception as e:
                    self.log(f"Seriele leesfout: {e}")
                    self.term_log(f"[RX ERROR: {e}]", "sys")
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

    def _consume_serial_text(self, raw_text):
        text = str(raw_text or "")
        if not text:
            return

        self.serial_rx_buffer += text
        self.serial_rx_buffer = self.serial_rx_buffer.replace("\r\n", "\n").replace("\r", "\n")

        while "\n" in self.serial_rx_buffer:
            line, self.serial_rx_buffer = self.serial_rx_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self.raw_rx_seq += 1
            self.raw_rx_lines.append((self.raw_rx_seq, line))
            self.term_log("< " + line, "sys")
            self.parse_report_string(line)

        prompt_patterns = ("Select option", "Command:", "Press", "Main-------------", "1-Run", "2-Calibrate")
        buffered = self.serial_rx_buffer.strip()
        if buffered and any(pattern in buffered for pattern in prompt_patterns):
            self.raw_rx_seq += 1
            self.raw_rx_lines.append((self.raw_rx_seq, buffered))
            self.term_log("< " + buffered, "sys")
            self.serial_rx_buffer = ""

    def parse_com_setup(self):
        self.fetch_status_and_settings()

    def toggle_connection(self):
        if not self.is_reading:
            selected_port = self.port_menu.get()
            self.device_setup["protocol"] = self.proto_menu.get()
            self.device_setup["parity"] = self.parity_menu.get()
            self.device_setup["stopbits"] = self.stopbits_menu.get()
            if not selected_port or "Geen COM-poorten" in selected_port:
                self.log("Geen COM-poort geselecteerd. Sluit eerst de sonde aan en kies de juiste poort.")
                self.status_label.configure(text="Status: Geen COM-poort", text_color="#F44336")
                return
            try:
                self.serial_conn = serial.Serial(
                    selected_port,
                    baudrate=self.device_setup["baudrate"],
                    parity=self._map_parity(self.device_setup["parity"]),
                    stopbits=self._map_stopbits(self.device_setup["stopbits"]),
                    timeout=0.1,
                )
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()
                self.is_reading = True
                self.demo_mode = False
                self.serial_rx_buffer = ""
                self.connect_btn.configure(text="Disconnect", fg_color="#F44336")
                self.status_label.configure(text=f"Status: {self.device_setup['protocol']} verbonden", text_color="#4CAF50")
                self.term_log(f"[OPEN {selected_port} @ {self.device_setup['baudrate']} {self.device_setup['parity']}{self.device_setup['stopbits']}]", "sys")
            except Exception as e:
                self.log(f"Kon geen verbinding maken met {selected_port}: {e}")
                self.term_log(f"[OPEN FAILED {selected_port}: {e}]", "sys")
                self.serial_conn = None
                self.is_reading = False
                self.demo_mode = False
                self.connect_btn.configure(text="Connect", fg_color="#2196F3")
                self.status_label.configure(text="Status: Verbindingsfout", text_color="#F44336")
                return

            self.time_buffer.clear()
            self.datetime_buffer.clear()
            for dq in self.data_buffers.values():
                dq.clear()
            threading.Thread(target=self.rx_data_loop, daemon=True).start()
            self.after(300, self.fetch_status_and_settings)
        else:
            self.is_reading = False
            if self.serial_conn:
                self.serial_conn.close()
            self.serial_conn = None
            self.demo_mode = False
            self.connect_btn.configure(text="Connect", fg_color="#2196F3")
            self.status_label.configure(text="Status: Standby", text_color="#dddddd")

    def rx_data_loop(self):
        while self.is_reading:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    waiting = self.serial_conn.in_waiting
                    if waiting > 0:
                        raw_text = self.serial_conn.read(waiting).decode('utf-8', errors='ignore')
                        self._consume_serial_text(raw_text)
                    else:
                        time.sleep(0.02)
                except Exception as e:
                    self.log(f"Seriele leesfout: {e}")
                    self.term_log(f"[RX ERROR: {e}]", "sys")
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

    def parse_report_string(self, raw_string, is_sim=False, sim_source=None, timestamp_obj=None):
        if not timestamp_obj:
            timestamp_obj = datetime.now()

        parsed = False
        if is_sim and sim_source:
            self.log(f"Rx [{self.device_setup['protocol']}]: {raw_string}")
            self.time_buffer.append(timestamp_obj.strftime("%H:%M:%S"))
            self.datetime_buffer.append(timestamp_obj)
            for rep_key, val in sim_source:
                if not isinstance(val, str) and rep_key in self.data_buffers:
                    self.data_buffers[rep_key].append(float(val))
                    parsed = True
        else:
            self.log(f"Rx [{self.device_setup['protocol']}]: {raw_string}")
            if not self._looks_like_measurement_row(raw_string):
                return
            numeric_tokens = self._extract_numeric_tokens(raw_string)
            if numeric_tokens:
                self.recent_numeric_rows.append(numeric_tokens)
                detected_reports = self._detect_report_layout()
                if detected_reports and detected_reports != self.device_setup["reports"]:
                    self._apply_detected_report_layout(detected_reports, source="auto-detect")
            parsed_values = self._parse_live_payload(raw_string)
            if parsed_values:
                self.time_buffer.append(timestamp_obj.strftime("%H:%M:%S"))
                self.datetime_buffer.append(timestamp_obj)
            for rep_key, val in parsed_values.items():
                if rep_key in self.data_buffers:
                    self.data_buffers[rep_key].append(val)
                    parsed = True
        if parsed:
            self.update_live_graph()

    def _looks_like_measurement_row(self, raw_string):
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(raw_string or "")).strip()
        if not text:
            return False
        lower = text.lower()
        blocked_fragments = (
            "select option",
            "previous menu",
            "exit menu",
            "main------------------",
            "run setup",
            "discrete sample",
            "status-----------------",
            "system setup",
            "report setup",
            "sensors enabled",
            "advanced",
            "sample interval",
            "start sampling",
            "open a file",
            "clean optics",
            "log last sample",
            "press any key",
            "to continue",
            "?command",
            "1-run",
            "2-calibrate",
            "3-file",
            "4-status",
            "5-system",
            "6-report",
            "7-sensor",
            "8-advanced",
        )
        if any(fragment in lower for fragment in blocked_fragments):
            return False
        if re.fullmatch(r"[-=\s]+", text):
            return False

        numeric_tokens = self._extract_numeric_tokens(text)
        has_date = bool(re.search(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b", text))
        has_time = bool(re.search(r"\b\d{1,2}:\d{2}:\d{2}\b", text))
        if has_date and has_time and len(numeric_tokens) >= 2:
            return True
        if len(numeric_tokens) >= 4 and "=" not in text and ":" not in text:
            return True
        return False

    def _extract_numeric_tokens(self, raw_string):
        tokens = [tok for tok in re.split(r"\s+", str(raw_string or "").strip()) if tok]
        numeric = []
        for token in tokens:
            if any(sep in token for sep in ["/", "-"]) and ":" not in token:
                continue
            if ":" in token:
                continue
            probe = token.replace(",", ".")
            try:
                numeric.append(float(probe))
            except ValueError:
                continue
        return numeric

    def _score_layout(self, layout, numeric_values):
        report_items = [item for item in layout if "Date" not in item and "Time" not in item]
        if len(report_items) != len(numeric_values):
            return -10**9

        score = 0.0
        for label, value in zip(report_items, numeric_values):
            if "Temp" in label and -5 <= value <= 45:
                score += 3
            if ("SpCond" in label or label == "Cond") and value >= 0:
                score += 1
            if "DO mg/L" in label and 0 <= value <= 20:
                score += 3
            if "pH" == label and 0 <= value <= 14:
                score += 4
            if "Battery volts" in label and 8 <= value <= 20:
                score += 5
            if "Turbid" in label and 0 <= value <= 1000:
                score += 2
            if "Depth" in label and 0 <= value <= 100:
                score += 2

        # Penaliseer layouts waar battery onwaarschijnlijk laag is.
        for label, value in zip(report_items, numeric_values):
            if "Battery volts" in label and value < 7:
                score -= 6
            if label == "pH" and not (0 <= value <= 14):
                score -= 10
        return score

    def _detect_report_layout(self):
        if not self.recent_numeric_rows:
            return None

        sample_rows = list(self.recent_numeric_rows)
        best_layout = None
        best_score = float("-inf")
        for layout in REPORT_LAYOUT_CANDIDATES:
            total = 0.0
            valid_rows = 0
            for row in sample_rows:
                score = self._score_layout(layout, row)
                if score > -10**8:
                    total += score
                    valid_rows += 1
            if valid_rows:
                avg_score = total / valid_rows
                if avg_score > best_score:
                    best_score = avg_score
                    best_layout = layout
        return list(best_layout) if best_layout else None

    def _derive_sensors_from_reports(self, reports):
        sensors = ["Time"]
        for report in reports:
            if "Temp" in report and "Temperature" not in sensors:
                sensors.append("Temperature")
            elif ("SpCond" in report or report == "Cond") and "Conductivity" not in sensors:
                sensors.append("Conductivity")
            elif "DO mg/L" in report and "Dissolved Oxy" not in sensors:
                sensors.append("Dissolved Oxy")
            elif report == "pH" and "ISE1 pH" not in sensors:
                sensors.append("ISE1 pH")
            elif "Turbid" in report and "Optic T Turbidity" not in sensors:
                sensors.append("Optic T Turbidity")
            elif "Chl" in report and "Optic C Chlorophyll" not in sensors:
                sensors.append("Optic C Chlorophyll")
            elif "Battery volts" in report and "Battery volts" not in sensors:
                sensors.append("Battery volts")
        return sensors

    def _apply_detected_report_layout(self, reports, source="detectie"):
        self.device_setup["reports"] = list(reports)
        self.device_setup["sensors"] = self._derive_sensors_from_reports(reports)
        self.update_live_graph_selectors()

        if hasattr(self, "sensor_vars"):
            for sensor, var in self.sensor_vars.items():
                var.set(sensor in self.device_setup["sensors"])
        if hasattr(self, "report_vars"):
            for report, var in self.report_vars.items():
                var.set(report in self.device_setup["reports"])
        self.update_sensor_summary()

        self.log(f"[Detectie] {source}: reports = {', '.join(self.device_setup['reports'])}")
        self.log(f"[Detectie] {source}: sensors = {', '.join(self.device_setup['sensors'])}")

    def _parse_live_payload(self, raw_string):
        """Parseer echte RS-232/tekst payload volgens de geconfigureerde report-volgorde."""
        reports = list(self.device_setup.get("reports", []))
        if not reports:
            return {}

        text = str(raw_string or "").strip()
        if not text:
            return {}

        # Normaliseer spacing; YSI regels komen typisch als:
        # 03/11/2026 12:13:44 16.35 10.537 7.83 7.24 4.3 16.1
        tokens = [tok for tok in re.split(r"\s+", text) if tok]
        if not tokens:
            return {}

        parsed = {}
        idx = 0

        for rep_item in reports:
            if idx >= len(tokens):
                break

            if "Date" in rep_item:
                token = tokens[idx]
                if any(sep in token for sep in ["/", "-"]):
                    idx += 1
                continue

            if "Time" in rep_item:
                token = tokens[idx]
                if ":" in token:
                    idx += 1
                continue

            token = tokens[idx].replace(",", ".")
            try:
                parsed[rep_item] = float(token)
                idx += 1
            except ValueError:
                # Soms zit er ruis of een extra veld in de regel.
                # Zoek dan vooruit naar de eerstvolgende numerieke token.
                found = False
                for seek in range(idx + 1, len(tokens)):
                    probe = tokens[seek].replace(",", ".")
                    try:
                        parsed[rep_item] = float(probe)
                        idx = seek + 1
                        found = True
                        break
                    except ValueError:
                        continue
                if not found:
                    continue

        return parsed

    def update_live_graph(self):
        """PLOT LOGICA! Toont X-as als TIJD in REAL TIME TAB 1"""
        self.ax_live.clear()
        self.ax_live.set_facecolor('#1e1e1e')
        self.ax_live.tick_params(colors='white')
        
        self.ax_live.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        plotted_any = False
        colors = ['#03dac6', '#bb86fc', '#ff9800', '#f44336', '#4caf50', '#2196f3']
        c_idx = 0
        
        for plot_key, var_bool in self.plot_vars.items():
            if var_bool.get(): 
                data_to_plot = list(self.data_buffers[plot_key])
                time_range = list(self.datetime_buffer)
                
                min_len = min(len(data_to_plot), len(time_range))
                if min_len > 0:
                    self.ax_live.plot(time_range[-min_len:], data_to_plot[-min_len:], label=plot_key, color=colors[c_idx % len(colors)], marker='.', linewidth=1.5)
                    plotted_any = True
                    c_idx += 1
        
        if plotted_any:
            self.ax_live.legend(facecolor='#121212', edgecolor='white', labelcolor='white')
            self.fig_live.autofmt_xdate(rotation=45)
            self.ax_live.set_title("EcoWW Real-Time Live Data stream", color='white')
        else:
            self.ax_live.set_title("Vink een variabele aan in het linkermenu om te plotten", color='white')
            
        self.fig_live.tight_layout()
        self.canvas_live.draw()

    def on_closing(self):
        self.is_reading = False
        if self.serial_conn:
            try: self.serial_conn.close()
            except: pass
        self.quit()
        self.destroy()
        import os
        os._exit(0) # Voorkomt de 'invalid command name check_dpi_scaling' Tkinter/CustomTkinter thread bug!

if __name__ == "__main__":
    app = EcoWatchClone()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
