# -*- coding: utf-8 -*-
"""
Offerte-PDF → TXT met KERAS-OCR (géén Tesseract/Poppler)
- Velden: Omschrijving (='Beschrijving', met leidend nummer als prefix), Hoeveelheid (='Aantal'), Prijs_per_stuk (='Prijs')
- GUI (Tkinter) voor bestandskeuze en opslaan
- Pipeline:
    1) Render PDF-pagina's via PyMuPDF (fitz) → images
    2) OCR per pagina met keras-ocr
    3) Woorden → regels (top→down, left→right)
    4) Regels → blokken → (Omschrijving, Hoeveelheid, Prijs)

Benodigd:
    pip install keras-ocr pymupdf pillow numpy opencv-python-headless shapely tensorflow
"""

import io
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 1=verberg INFO, 2=verberg INFO+WARNING, 3=verberg alles behalve FATAL
import re
import sys
from pathlib import Path
from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image

# --- PyMuPDF voor renderen ---
import fitz  # PyMuPDF

# --- Keras-OCR ---
import keras_ocr


# ----------------------------- Regex-profielen --------------------------------
# Kop/voet-ruis verwijderen (NL offerte lay-out; voeg gerust regels toe)
HEADER_RE = re.compile("|".join([
    r"^\s*RS Components\s*$",
    r"^\s*Uw winkelwagen op RS Online\s*$",
    r"^\s*Beschrijving\s+Aantal\s+Prijs\s+Totaal\s*$",  # Beschrijving = Omschrijving
    r"^\s*Pagina\s+\d+\s+van\s+\d+\s*$",
    r"^\s*Aflevertermijn.*$",
    r"^\s*Totaal excl\.\s*B\.T\.W\..*$",
    r"^\s*Levering\s+.*$",
    r"^\s*B\.T\.W\..*$",
    r"^\s*Totaal incl\.\s*B\.T\.W\..*$",
    r"^\s*Algemene voorwaarden.*$",
    r"^\s*Contact.*$",
], re.IGNORECASE))

# Hoeveelheid + prijs op 1 regel, euro optioneel (OCR vergist zich soms)
# qty = integer; price: 1.234,56 of 12,34 of 12.34
PRICE_TOKEN = r"(?:\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2})|\d+)"
QTY_PRICE_RE = re.compile(
    rf"^\s*(\d{{1,5}})\s+({PRICE_TOKEN})\s*(?:€|eur)?\s*$",
    re.IGNORECASE
)

# Pure numerieke productcode (scheidt vaak beschrijving ↔ meta)
PURE_NUM_RE = re.compile(r"^\s*\d{5,}\s*$")

# Beschrijvingsruis die we vermijden
SKIP_IN_DESC_RE = re.compile(r"(RoHS-status|Each|Pack of|Set$)", re.IGNORECASE)

# Heuristiek voor leidend cijferblok dat we als prefix willen (OCR → geen bold info)
LEADING_NUM_HEUR_RE = re.compile(r"^\s*([0-9][0-9\.\-\)\( ]{0,10})")


# ----------------------------- Data-structuren --------------------------------
@dataclass
class OCRWord:
    text: str
    x_center: float
    y_center: float
    y_top: float
    y_bottom: float

@dataclass
class OCRLine:
    text: str
    y: float  # lijnpositie (centrum)
    x_left: float
    x_right: float


# ----------------------------- Hulpfuncties -----------------------------------
def render_pdf_to_images(pdf_path: Path, dpi: int = 300):
    """Render elke PDF-pagina naar een PIL.Image via PyMuPDF."""
    images = []
    doc = fitz.open(str(pdf_path))
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def keras_ocr_read_images(images):
    """Run keras-ocr pipeline over een lijst van PIL images; retourneer per pagina predictions."""
    pipeline = keras_ocr.pipeline.Pipeline()  # laadt modellen bij de 1e keer (kan even duren)
    # Keras-OCR verwacht filepaths of np arrays; we geven arrays
    np_images = [np.array(img) for img in images]
    predictions = pipeline.recognize(np_images)  # lijst per pagina: [(text, box), ...]
    return predictions

def words_to_lines(preds_page, y_tol_frac=0.015):
    """
    Groepeer (text, box)-woorden tot regels.
    y_tol_frac = verticale tolerantie t.o.v. afbeeldingshoogte
    """
    if not preds_page:
        return []

    # Haal hoogte / breedte uit de boxen
    # box = 4 punten [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    ys = [p[1][0][1] for p in preds_page] + [p[1][2][1] for p in preds_page]
    xs = [p[1][0][0] for p in preds_page] + [p[1][2][0] for p in preds_page]
    img_h = max(ys) - min(ys) if ys else 1
    tol = max(3.0, y_tol_frac * img_h)

    words = []
    for (txt, box) in preds_page:
        if not txt:
            continue
        txt = txt.strip()
        if not txt:
            continue
        xs_ = [pt[0] for pt in box]
        ys_ = [pt[1] for pt in box]
        x_center = float(np.mean(xs_))
        y_center = float(np.mean(ys_))
        words.append(OCRWord(text=txt, x_center=x_center, y_center=y_center,
                             y_top=min(ys_), y_bottom=max(ys_)))

    # Sorteer woorden grofweg op y, dan x
    words.sort(key=lambda w: (w.y_center, w.x_center))

    # Cluster per 'lijn' op basis van y_center met tolerantie
    lines = []
    current = []
    current_y = None

    for w in words:
        if current_y is None:
            current = [w]
            current_y = w.y_center
            continue
        if abs(w.y_center - current_y) <= tol:
            current.append(w)
            current_y = (current_y * (len(current)-1) + w.y_center) / len(current)
        else:
            # sluit vorige lijn af
            current.sort(key=lambda ww: ww.x_center)
            line_text = " ".join(ww.text for ww in current)
            lines.append(OCRLine(text=line_text, y=current_y,
                                 x_left=min(ww.x_center for ww in current),
                                 x_right=max(ww.x_center for ww in current)))
            # start nieuwe
            current = [w]
            current_y = w.y_center

    if current:
        current.sort(key=lambda ww: ww.x_center)
        line_text = " ".join(ww.text for ww in current)
        lines.append(OCRLine(text=line_text, y=current_y,
                             x_left=min(ww.x_center for ww in current),
                             x_right=max(ww.x_center for ww in current)))

    # Sorteer lijnen top → bottom (kleinste y eerst)
    lines.sort(key=lambda ln: ln.y)
    return lines

def normalize_lines(lines_text):
    out = []
    for raw in lines_text:
        ln = raw.replace("\xa0", " ").strip()
        ln = re.sub(r"\s{2,}", " ", ln)
        out.append(ln)
    return out

def strip_headers(lines):
    return [ln for ln in lines if ln and not HEADER_RE.match(ln)]

def split_blocks(lines):
    """Maak artikelblokken op basis van lege regels en aanwezigheid van '€/eur/prijs'."""
    blocks, buf = [], []
    def is_pricey(s):
        return ("€" in s.lower()) or ("eur" in s.lower()) or re.search(PRICE_TOKEN, s) is not None
    for ln in lines + [""]:
        if ln == "":
            if buf and any(is_pricey(b) for b in buf):
                blocks.append(buf[:])
            buf = []
        else:
            buf.append(ln)
    return blocks

def prefix_leading_number(text_line: str, current_desc: str) -> str:
    """Heuristiek: neem leidend cijferblok uit de éérste beschrijvingslijn en zet vooraan."""
    m = LEADING_NUM_HEUR_RE.match(text_line or "")
    if m:
        token = m.group(1).strip()
        # token niet te kort en niet al aanwezig
        if token and not current_desc.startswith(token):
            return f"{token} — {current_desc}".strip(" —")
    return current_desc

def extract_items_from_lines(lines):
    """
    Van OCR-lijnen → (Omschrijving, Hoeveelheid, Prijs)
    - Omschrijving = eerste 1–3 regels vóór productcode (5+ cijfers), zonder €
    - Hoeveelheid/Prijs = eerste regel die op 'qty price €' lijkt
    - Leidende cijferprefix uit eerste beschrijvingslijn toevoegen
    """
    items = []
    lines = strip_headers(normalize_lines(lines))
    blocks = split_blocks(lines)

    for blk in blocks:
        # Zoek index v/d eerste pure numerieke productcode
        first_num_idx = None
        for i, ln in enumerate(blk):
            if PURE_NUM_RE.match(ln):
                first_num_idx = i
                break
        desc_candidates = blk[:first_num_idx] if first_num_idx is not None else blk[:]
        desc_candidates = [
            ln for ln in desc_candidates
            if ln and "€" not in ln and not SKIP_IN_DESC_RE.search(ln)
        ]

        seen, desc_lines = set(), []
        for ln in desc_candidates:
            if ln not in seen:
                desc_lines.append(ln)
                seen.add(ln)
            if len(desc_lines) >= 3:
                break
        omschrijving = " — ".join(desc_lines).strip(" —")

        hoeveelheid, prijs = "", ""
        # zoek qty/price
        for ln in blk:
            m = QTY_PRICE_RE.match(ln)
            if m:
                hoeveelheid = m.group(1)
                prijs = m.group(2)
                break

        if not omschrijving:
            # fallback: eerste zinvolle regel
            for ln in blk:
                if ln and "€" not in ln and not SKIP_IN_DESC_RE.search(ln):
                    omschrijving = ln
                    break

        # Leidende nummerprefix uit eerste beschrijvingsregel
        if desc_lines:
            omschrijving = prefix_leading_number(desc_lines[0], omschrijving)

        if omschrijving and (hoeveelheid or prijs):
            items.append((omschrijving, hoeveelheid, prijs))

    return items


# ----------------------------- Hoofdlogica ------------------------------------
def process_pdf_with_keras_ocr(pdf_path: Path):
    """PDF → images → OCR → lijnen → items."""
    images = render_pdf_to_images(pdf_path, dpi=300)
    if not images:
        return []
    predictions_pages = keras_ocr_read_images(images)
    all_lines = []
    for preds in predictions_pages:
        lines = words_to_lines(preds, y_tol_frac=0.015)
        all_lines.extend([ln.text for ln in lines])
        all_lines.append("")  # lege regel als pagina-scheiding
    return extract_items_from_lines(all_lines)

def save_as_txt(items, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Omschrijving\tHoeveelheid\tPrijs_per_stuk\n")
        for oms, hvl, pr in items:
            f.write(f"{oms}\t{hvl}\t{pr}\n")


# ----------------------------- GUI -------------------------------------------
def main_gui():
    root = tk.Tk()
    root.withdraw()
    root.update()

    messagebox.showinfo(
        "Offerte → TXT (KERAS-OCR)",
        "Selecteer de offerte-PDF. Dit gebruikt KERAS-OCR (geen Tesseract nodig)."
    )
    pdf_file = filedialog.askopenfilename(
        title="Kies offerte PDF",
        filetypes=[("PDF", "*.pdf"), ("Alle bestanden", "*.*")]
    )
    if not pdf_file:
        messagebox.showwarning("Geannuleerd", "Geen PDF geselecteerd.")
        return

    try:
        items = process_pdf_with_keras_ocr(Path(pdf_file))
    except Exception as e:
        messagebox.showerror("Verwerken mislukt", f"Kon PDF niet verwerken:\n{e}")
        return

    if not items:
        messagebox.showwarning(
            "Geen items gevonden",
            "Geen artikelregels gedetecteerd. Controleer OCR-installatie of PDF-kwaliteit."
        )
        return

    out_file = filedialog.asksaveasfilename(
        title="Bewaar TXT-lijst",
        initialfile=Path(pdf_file).with_suffix(".txt").name,
        defaultextension=".txt",
        filetypes=[("Tekstbestand", "*.txt"), ("Alle bestanden", "*.*")]
    )
    if not out_file:
        messagebox.showwarning("Geannuleerd", "Geen uitvoerbestand gekozen.")
        return

    try:
        save_as_txt(items, Path(out_file))
    except Exception as e:
        messagebox.showerror("Schrijffout", f"Kon uitvoer niet bewaren:\n{e}")
        return

    messagebox.showinfo("Klaar", f"{len(items)} artikellijnen opgeslagen naar:\n{out_file}")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        in_pdf = Path(sys.argv[1])
        out_txt = Path(sys.argv[2]) if len(sys.argv) >= 3 else in_pdf.with_suffix(".txt")
        rows = process_pdf_with_keras_ocr(in_pdf)
        save_as_txt(rows, out_txt)
        print(f"Gereed. {len(rows)} items → {out_txt}")
    else:
        main_gui()
