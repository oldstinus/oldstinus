import cv2
import easyocr
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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
        self.reader = easyocr.Reader(['en'])  # Initialiseer EasyOCR voor Engels
        self.roi = None  # ROI in originele framecoördinaten (x, y, w, h)
        self.tracker = None

    def set_roi(self, x, y, w, h):
        """Stel de ROI in via de opgegeven coördinaten."""
        self.roi = (x, y, w, h)
        print(f"ROI ingesteld: {self.roi}")

    def init_tracker(self, init_frame):
        """Initialiseer de tracker met het opgegeven frame en de ingestelde ROI."""
        if self.roi is None:
            return
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracker.init(init_frame, self.roi)
        print("Tracker geïnitialiseerd.")

    def extract_value_from_frame(self, frame, roi):
        """Extraheer OCR-waarde uit het gegeven frame op basis van de ROI."""
        if frame is None:
            return np.nan

        x, y, w, h = roi
        roi_frame = frame[int(y):int(y+h), int(x):int(x+w)]

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        result = self.reader.readtext(thresh, detail=0)
        print(f"OCR resultaat: {result}")

        if result:
            try:
                for text in result:
                    cleaned_text = ''.join(filter(lambda c: c.isdigit() or c == '.', text.replace(',', '.')))
                    if cleaned_text:
                        return float(cleaned_text)
                return np.nan
            except ValueError:
                return np.nan
        else:
            return np.nan

    def extract_values(self, start_frame, end_frame, spacing):
        # Zet pointer op start_frame en lees het eerste frame voor trackerinitialisatie
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = self.cap.read()
        if not ret:
            raise IOError("Kon het eerste frame niet lezen voor trackerinitialisatie.")
        if self.roi is not None and self.tracker is None:
            self.init_tracker(first_frame)
        # Reset pointer naar start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, end_frame, spacing):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Update de tracker als deze is geïnitialiseerd
            if self.tracker is not None:
                success, new_roi = self.tracker.update(frame)
                if success:
                    self.roi = tuple(map(int, new_roi))
                else:
                    print("Tracking mislukt, gebruik vorige ROI.")
            # Extraheer waarde met de huidige ROI
            value = self.extract_value_from_frame(frame, self.roi) if self.roi else np.nan
            timestamp = frame_num / self.fps
            print(f"Tijd: {timestamp:.2f} sec, Waarde: {value}")
            self.values.append((timestamp, value))
        self.cap.release()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video naar Data Digitalisatie")
        self.geometry("650x650")
        self.video_processor = None
        self.photo = None
        self.scale_x = 1
        self.scale_y = 1

        # Knoppen
        self.load_btn = tk.Button(self, text="Laad Video", command=self.load_video)
        self.load_btn.pack(pady=10)

        # Canvas voor video preview en ROI-selectie
        self.video_canvas = tk.Canvas(self, width=400, height=300, bg='grey')
        self.video_canvas.pack(pady=10)
        self.video_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.video_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.roi_start = None
        self.roi_rect = None

        # ROI Invoer Velden
        self.roi_frame = tk.Frame(self)
        self.roi_frame.pack(pady=10)
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

        # Verwerk Video knop
        self.process_btn = tk.Button(self, text="Verwerk Video", command=self.process_video, state=tk.DISABLED)
        self.process_btn.pack(pady=10)

        # Status Label
        self.status_label = tk.Label(self, text="Status: Wachtend op invoer")
        self.status_label.pack(pady=10)

    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Selecteer Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            try:
                self.video_processor = VideoProcessor(file_path)
                messagebox.showinfo("Succes", f"Video geladen: {os.path.basename(file_path)}\nDuur: {self.video_processor.duration:.2f} sec")
                self.status_label.config(text="Status: Video geladen. Sleep met de muis om de ROI te selecteren.")

                # Haal het eerste frame op en toon een preview op de canvas
                self.video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_processor.cap.read()
                if ret:
                    # Bewaar originele frame afmetingen voor schaalcorrectie
                    orig_h, orig_w = frame.shape[:2]
                    preview_w, preview_h = 400, 300
                    self.scale_x = orig_w / preview_w
                    self.scale_y = orig_h / preview_h

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    image = image.resize((preview_w, preview_h), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(image)
                    self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                # Reset pointer zodat verwerking later vanaf het begin plaatsvindt
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
        self.roi_rect = self.video_canvas.create_rectangle(self.roi_start[0], self.roi_start[1], event.x, event.y, outline='red', width=2)

    def on_mouse_up(self, event):
        if not self.roi_start:
            return
        x0, y0 = self.roi_start
        x1, y1 = event.x, event.y
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        # Converteer canvas-ROI naar originele framecoördinaten
        x_orig = int(x * self.scale_x)
        y_orig = int(y * self.scale_y)
        w_orig = int(w * self.scale_x)
        h_orig = int(h * self.scale_y)
        # Werk ROI-invoer bij
        self.x_entry.delete(0, tk.END)
        self.x_entry.insert(0, str(x_orig))
        self.y_entry.delete(0, tk.END)
        self.y_entry.insert(0, str(y_orig))
        self.w_entry.delete(0, tk.END)
        self.w_entry.insert(0, str(w_orig))
        self.h_entry.delete(0, tk.END)
        self.h_entry.insert(0, str(h_orig))
        # Stel ROI in voor de video_processor
        if self.video_processor:
            self.video_processor.set_roi(x_orig, y_orig, w_orig, h_orig)
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: ROI ingesteld en tracker geactiveerd.")
        self.roi_start = None

    def process_video(self):
        if not self.video_processor:
            messagebox.showerror("Fout", "Geen video geladen.")
            return

        start_time = simpledialog.askfloat("Input", "Starttijd (seconden):", minvalue=0, maxvalue=self.video_processor.duration)
        if start_time is None:
            messagebox.showerror("Fout", "Ongeldige starttijd.")
            return
        end_time = simpledialog.askfloat("Input", "Eindtijd (seconden):", minvalue=start_time, maxvalue=self.video_processor.duration)
        if end_time is None:
            messagebox.showerror("Fout", "Ongeldige eindtijd.")
            return
        spacing = simpledialog.askinteger("Input", "Spacing tussen frames (in frames):", minvalue=1)
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

        csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
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
