import cv2
import easyocr
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError("Kan de video niet openen.")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count / self.fps

        self.values = []
        self.reader = easyocr.Reader(['en'])   # OCR voor Engels
        self.roi = None                       # (x, y, w, h) in originele framecoördinaten
        self.tracker = None                   # CSRT-tracker
        # Regex-patroon voor vier cijfers, punt, één cijfer (bijv. "0123.4")
        self.pattern = re.compile(r'^\d{4}\.\d$')  

    def set_roi(self, x, y, w, h):
        """Stel de ROI in via de opgegeven coördinaten."""
        self.roi = (x, y, w, h)
        print(f"ROI ingesteld: {self.roi}")

    def init_tracker(self, init_frame):
        """Initialiseer de CSRT-tracker met het opgegeven frame en de ingestelde ROI."""
        if self.roi is None:
            return
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            # In sommige OpenCV-versies zit de CSRT-tracker in de 'legacy' namespace
            self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracker.init(init_frame, self.roi)
        print("Tracker geïnitialiseerd.")

    def rotate_image(self, image, angle):
        """Roteer een afbeelding (subframe) rond het middelpunt met opgegeven hoek in graden."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        # Positieve hoek = tegen de klok in rotatie in OpenCV
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def try_ocr_with_angle(self, subframe, angle):
        """Voer OCR uit op de subframe nadat deze is geroteerd met de gegeven hoek.
           Retourneer de gedetecteerde waarde (float) als die het patroon XXXX.X matcht,
           anders None."""
        rotated = self.rotate_image(subframe, angle)

        # Preprocessing
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # OCR
        result = self.reader.readtext(thresh, detail=0)
        if result:
            for text in result:
                # Verwijder alle niet-cijfers en punten
                cleaned_text = ''.join(ch for ch in text.replace(',', '.') if ch.isdigit() or ch == '.')
                # Controleer of het patroon XXXX.X overeenkomt
                if self.pattern.match(cleaned_text):
                    try:
                        return float(cleaned_text)
                    except ValueError:
                        return None
        return None

    def extract_value_from_frame(self, frame):
        """Knip de ROI uit, test meerdere rotatiehoeken en
           retourneer de eerste geldige waarde die aan het patroon voldoet."""
        if frame is None or self.roi is None:
            return np.nan

        x, y, w, h = self.roi
        # Subframe uitknippen
        subframe = frame[int(y):int(y+h), int(x):int(x+w)]

        # Doorloop een reeks hoeken (bijv. -10 tot +10 in stappen van 1 graad)
        for angle in range(-10, 11, 1):
            value = self.try_ocr_with_angle(subframe, angle)
            if value is not None:
                return value

        # Geen enkele hoek leverde een geldige match op
        return np.nan

    def extract_values(self, start_frame, end_frame, spacing):
        """Lees frames van start_frame tot end_frame, met 'spacing' als interval.
           Update de tracker continu en pas per frame de OCR toe op de (eventueel) 
           geroteerde subframe."""
        # Initialiseer de tracker op het eerste frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = self.cap.read()
        if not ret:
            raise IOError("Kon het eerste frame niet lezen voor trackerinitialisatie.")

        if self.roi is not None and self.tracker is None:
            self.init_tracker(first_frame)

        # Reset pointer naar start_frame voor de lus
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_num in range(start_frame, end_frame, spacing):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Tracker bijwerken
            if self.tracker is not None:
                success, new_roi = self.tracker.update(frame)
                if success:
                    # Rond de coördinaten af naar integers
                    self.roi = tuple(map(int, new_roi))
                else:
                    print("Tracking mislukt, blijf oude ROI gebruiken.")

            # OCR uitvoeren op basis van de (eventueel) bijgewerkte ROI
            value = self.extract_value_from_frame(frame)
            timestamp = frame_num / self.fps
            print(f"Tijd: {timestamp:.2f} s, Waarde: {value}")
            self.values.append((timestamp, value))

        self.cap.release()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video naar Data Digitalisatie (Tracking + Dynamische Rotatie)")
        self.geometry("700x750")
        self.video_processor = None
        self.photo = None
        self.scale_x = 1
        self.scale_y = 1

        # Knoppen
        self.load_btn = tk.Button(self, text="Laad Video", command=self.load_video)
        self.load_btn.pack(pady=5)

        # Canvas voor video preview + ROI-selectie
        self.video_canvas = tk.Canvas(self, width=400, height=300, bg='grey')
        self.video_canvas.pack(pady=5)
        self.video_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.video_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.roi_start = None
        self.roi_rect = None

        # ROI-velden
        self.roi_frame = tk.Frame(self)
        self.roi_frame.pack(pady=5)
        tk.Label(self.roi_frame, text="X:").grid(row=0, column=0)
        self.x_entry = tk.Entry(self.roi_frame, width=5)
        self.x_entry.grid(row=0, column=1, padx=5)
        tk.Label(self.roi_frame, text="Y:").grid(row=0, column=2)
        self.y_entry = tk.Entry(self.roi_frame, width=5)
        self.y_entry.grid(row=0, column=3, padx=5)
        tk.Label(self.roi_frame, text="Breedte:").grid(row=1, column=0)
        self.w_entry = tk.Entry(self.roi_frame, width=5)
        self.w_entry.grid(row=1, column=1, padx=5)
        tk.Label(self.roi_frame, text="Hoogte:").grid(row=1, column=2)
        self.h_entry = tk.Entry(self.roi_frame, width=5)
        self.h_entry.grid(row=1, column=3, padx=5)

        # Verwerkknop
        self.process_btn = tk.Button(self, text="Verwerk Video", command=self.process_video, state=tk.DISABLED)
        self.process_btn.pack(pady=5)

        # Status
        self.status_label = tk.Label(self, text="Status: Wachtend op invoer")
        self.status_label.pack(pady=5)

    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Selecteer Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            try:
                self.video_processor = VideoProcessor(file_path)
                msg = (f"Video geladen: {os.path.basename(file_path)}\n"
                       f"Duur: {self.video_processor.duration:.2f} sec")
                messagebox.showinfo("Succes", msg)
                self.status_label.config(text="Status: Video geladen. Selecteer ROI.")

                # Preview eerste frame
                self.video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_processor.cap.read()
                if ret:
                    orig_h, orig_w = frame.shape[:2]
                    preview_w, preview_h = 400, 300
                    self.scale_x = orig_w / preview_w
                    self.scale_y = orig_h / preview_h

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    # Pillow >= 9.1.0: gebruik Image.Resampling.LANCZOS
                    image = image.resize((preview_w, preview_h), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(image)
                    self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                # Reset pointer
                self.video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception as e:
                messagebox.showerror("Fout", str(e))

    def on_mouse_down(self, event):
        self.roi_start = (event.x, event.y)
        if self.roi_rect:
            self.video_canvas.delete(self.roi_rect)
            self.roi_rect = None

    def on_mouse_drag(self, event):
        if not self.roi_start:
            return
        if self.roi_rect:
            self.video_canvas.delete(self.roi_rect)
        self.roi_rect = self.video_canvas.create_rectangle(
            self.roi_start[0], self.roi_start[1], event.x, event.y,
            outline='red', width=2
        )

    def on_mouse_up(self, event):
        if not self.roi_start:
            return
        x0, y0 = self.roi_start
        x1, y1 = event.x, event.y
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)

        # Canvas -> originele framecoördinaten
        x_orig = int(x * self.scale_x)
        y_orig = int(y * self.scale_y)
        w_orig = int(w * self.scale_x)
        h_orig = int(h * self.scale_y)

        self.x_entry.delete(0, tk.END)
        self.x_entry.insert(0, str(x_orig))
        self.y_entry.delete(0, tk.END)
        self.y_entry.insert(0, str(y_orig))
        self.w_entry.delete(0, tk.END)
        self.w_entry.insert(0, str(w_orig))
        self.h_entry.delete(0, tk.END)
        self.h_entry.insert(0, str(h_orig))

        if self.video_processor:
            self.video_processor.set_roi(x_orig, y_orig, w_orig, h_orig)
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: ROI ingesteld.")
        self.roi_start = None

    def process_video(self):
        if not self.video_processor:
            messagebox.showerror("Fout", "Geen video geladen.")
            return

        start_time = simpledialog.askfloat("Starttijd", "Starttijd (s):",
                                           minvalue=0, 
                                           maxvalue=self.video_processor.duration)
        if start_time is None:
            messagebox.showerror("Fout", "Ongeldige starttijd.")
            return

        end_time = simpledialog.askfloat("Eindtijd", "Eindtijd (s):",
                                         minvalue=start_time, 
                                         maxvalue=self.video_processor.duration)
        if end_time is None:
            messagebox.showerror("Fout", "Ongeldige eindtijd.")
            return

        spacing = simpledialog.askinteger("Spacing", "Spacing tussen frames (in frames):",
                                          minvalue=1)
        if spacing is None:
            messagebox.showerror("Fout", "Ongeldige spacing.")
            return

        start_frame = int(start_time * self.video_processor.fps)
        end_frame = int(end_time * self.video_processor.fps)

        self.status_label.config(text="Status: Verwerken van video gestart...")
        self.update_idletasks()

        try:
            self.video_processor.extract_values(start_frame, end_frame, spacing)
            self.status_label.config(text="Status: Verwerking voltooid.")
        except Exception as e:
            messagebox.showerror("Fout", f"Fout tijdens verwerking: {str(e)}")
            self.status_label.config(text="Status: Fout tijdens verwerking.")
            return

        if not self.video_processor.values:
            messagebox.showwarning("Waarschuwing", "Geen data geëxtraheerd uit de video.")
            return

        df = pd.DataFrame(self.video_processor.values, columns=["Tijd (s)", "Druk (bar)"])
        df.dropna(inplace=True)

        if df.empty:
            messagebox.showwarning("Waarschuwing", "Geen geldige drukwaarden gevonden in de video.")
            return

        csv_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV Files", "*.csv")])
        if csv_path:
            df.to_csv(csv_path, index=False)
            messagebox.showinfo("Succes", f"Data opgeslagen als {os.path.basename(csv_path)}")

        plt.figure(figsize=(10, 6))
        plt.plot(df["Tijd (s)"], df["Druk (bar)"], marker='o')
        plt.title("Druk versus Tijd")
        plt.xlabel("Tijd (s)")
        plt.ylabel("Druk (bar)")
        plt.grid(True)
        plt.show()

        self.status_label.config(text="Status: Data opgeslagen en grafiek weergegeven.")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
