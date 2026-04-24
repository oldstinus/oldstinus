#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraheer alle ingesloten afbeeldingen/figuren uit een .docx:
- Detecteer echte bestandsindeling (magic bytes) en corrigeer extensie.
- Genereer bestandsnamen met prefix + volgnummer + korte-bijschrift-slug.
- Schrijf manifest.csv met rId, zip-pad, caption, size, sha1, final name.
- Optionele deduplicatie op SHA-1.

Gebruik:
  python extract_docx_images_with_captions.py INPUT.docx [-o OUTDIR]
         [--prefix Fig_] [--dedupe] [--min-bytes 1] [--pad 3]
"""

import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from xml.etree import ElementTree as ET
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


MEDIA_DIR = "word/media/"
RELS_PATH = "word/_rels/document.xml.rels"
DOC_XML = "word/document.xml"

NS = {
    "w":  "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a":  "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r":  "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1(); h.update(b); return h.hexdigest()

def sanitize(s: str) -> str:
    s = s.strip().replace("\u00A0", " ")
    s = SAFE_NAME_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s: s = "x"
    return s

def slug_from_caption(txt: str, max_words=8, max_len=40) -> str:
    if not txt: return ""
    # Pak de eerste zinnige woorden
    words = re.split(r"\s+", txt)
    short = " ".join(words[:max_words]).strip()
    # Verwijder label “Figuur 12:” e.d.
    short = re.sub(r"^(fig(uur)?|figure|afb(eelding)?)[\s.:;-]*\d*\s*[:.-]?\s*", "", short, flags=re.I)
    short = short[:max_len]
    return sanitize(short.lower())

def sniff_ext(data: bytes, fallback_ext: str) -> str:
    # Magic bytes voor rasterformaten
    if data.startswith(b"\x89PNG\r\n\x1a\n"):               return "png"
    if data.startswith(b"\xff\xd8\xff"):                    return "jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"): return "gif"
    if data.startswith(b"BM"):                              return "bmp"
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"): return "tiff"
    if data.startswith(b"RIFF") and b"WEBP" in data[:32]:   return "webp"
    # Vectorformaten: laat de (zip)extensie leidend zijn
    ext = fallback_ext.lower().lstrip(".")
    return ext if ext else "bin"

def next_index_name(prefix: str, idx: int, ext: str, slug: str, pad: int = 3) -> str:
    parts = [f"{prefix}{idx:0{pad}d}"]
    if slug:
        parts.append(slug)
    base = "_".join(parts) + f".{ext}"
    return sanitize(base)

def parse_relationships(zipf: ZipFile):
    """Return dict rId -> target path (e.g., word/media/image1.png)"""
    rels = {}
    try:
        with zipf.open(RELS_PATH) as f:
            tree = ET.parse(f)
        for rel in tree.getroot():
            rId = rel.attrib.get("Id")
            target = rel.attrib.get("Target")
            if rId and target and target.startswith("../media/"):
                rels[rId] = "word/media/" + target.split("../media/")[1]
            elif rId and target and target.startswith("media/"):
                rels[rId] = "word/" + target
    except KeyError:
        pass
    return rels

def extract_captions(zipf: ZipFile):
    """
    Doorzoek document.xml:
      - vind elke <a:blip r:embed="rIdX">
      - bewaar mapping: rId -> (paragraph_text, caption_text_if_any)
    Captionheuristiek:
      * alinea met stijl 'Caption'
      * of alinea die begint met 'Figuur', 'Figure', 'Afbeelding'
      in dezelfde alinea of de eerstvolgende alinea.
    """
    mapping = {}  # rId -> dict with 'para_text', 'caption'
    try:
        with zipf.open(DOC_XML) as f:
            tree = ET.parse(f)
    except KeyError:
        return mapping

    root = tree.getroot()
    paragraphs = root.findall(".//w:p", NS)

    def get_p_text(p):
        texts = p.findall(".//w:t", NS)
        return "".join(t.text or "" for t in texts).strip()

    def p_has_caption_style(p):
        pPr = p.find("./w:pPr", NS)
        if pPr is None: return False
        st = pPr.find("./w:pStyle", NS)
        if st is None: return False
        val = st.attrib.get(f"{{{NS['w']}}}val", "")
        return val.lower() == "caption"

    def looks_like_caption_text(s):
        return bool(re.match(r"^\s*(fig(uur)?|figure|afb(eelding)?)\b", s, flags=re.I))

    for i, p in enumerate(paragraphs):
        # vind alle blips in deze alinea
        blips = p.findall(".//a:blip", NS)
        if not blips:
            continue

        p_text = get_p_text(p)
        # Zoek caption in deze of volgende alinea
        caption = ""
        if p_has_caption_style(p) or looks_like_caption_text(p_text):
            caption = p_text
        else:
            if i + 1 < len(paragraphs):
                p2 = paragraphs[i+1]
                p2_text = get_p_text(p2)
                if p_has_caption_style(p2) or looks_like_caption_text(p2_text):
                    caption = p2_text

        for blip in blips:
            rId = blip.attrib.get(f"{{{NS['r']}}}embed")
            if rId:
                mapping[rId] = {"para_text": p_text, "caption": caption}

    return mapping

def extract_images(docx_path: Path, out_dir: Path, prefix="Afbeelding_", dedupe=False, min_bytes=1, pad=3):
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    seen_hashes = set()
    manifest_rows = []

    with ZipFile(docx_path, "r") as z:
        rels = parse_relationships(z)
        captions = extract_captions(z)

        # verzamel alle media onder word/media/
        media_members = [m for m in z.namelist() if m.startswith(MEDIA_DIR)]
        media_members.sort()

        # ook via relaties (garandeert dat we rId ↔ pad kunnen loggen)
        rids_by_media = {}
        for rId, path in rels.items():
            rids_by_media.setdefault(path, set()).add(rId)

        idx = 1
        for member in media_members:
            try:
                with z.open(member) as f:
                    data = f.read()
            except KeyError:
                continue

            size = len(data)
            if size < min_bytes:
                continue

            # kies een rId (als er is) en bijschrift
            rids = sorted(rids_by_media.get(member, []))
            chosen_rid = rids[0] if rids else ""
            cap_info = captions.get(chosen_rid, {})
            caption_txt = (cap_info.get("caption") or "").strip()
            slug = slug_from_caption(caption_txt)

            # bepaal extensie
            zip_ext = Path(member).suffix.lower().lstrip(".")
            ext = sniff_ext(data, zip_ext)

            # dedupe?
            digest = sha1_bytes(data)
            if dedupe and digest in seen_hashes:
                manifest_rows.append({
                    "index": "",
                    "filename": "",
                    "caption_summary": slug,
                    "caption_full": caption_txt,
                    "rId": ", ".join(rids),
                    "zip_path": member,
                    "size_bytes": size,
                    "sha1": digest,
                    "ext": ext,
                    "note": "duplicate_skipped",
                })
                continue
            seen_hashes.add(digest)

            # bestandsnaam
            name = next_index_name(prefix, idx, ext=ext, slug=slug, pad=pad)
            out_path = out_dir / name
            # voorkom overschrijven bij her-run
            while out_path.exists():
                idx += 1
                name = next_index_name(prefix, idx, ext=ext, slug=slug, pad=pad)
                out_path = out_dir / name

            with open(out_path, "wb") as g:
                g.write(data)

            manifest_rows.append({
                "index": idx,
                "filename": out_path.name,
                "caption_summary": slug,
                "caption_full": caption_txt,
                "rId": ", ".join(rids),
                "zip_path": member,
                "size_bytes": size,
                "sha1": digest,
                "ext": ext,
                "note": "",
            })

            idx += 1
            written += 1

    # manifest
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["index","filename","caption_summary","caption_full","rId","zip_path","size_bytes","sha1","ext","note"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    return written

def main():
    ap = argparse.ArgumentParser(description="Extraheer alle afbeeldingen uit DOCX met correcte extensies en bijschrift in bestandsnaam.")
    ap.add_argument("docx", type=str, help="Pad naar .docx")
    ap.add_argument("-o","--out", type=str, default=None, help="Uitvoermap (default: <DOCX>_media)")
    ap.add_argument("--prefix", type=str, default="Afbeelding_", help="Bestandsprefix (default: Afbeelding_)")
    ap.add_argument("--dedupe", action="store_true", help="Dedupliceer identieke bestanden (SHA-1)")
    ap.add_argument("--min-bytes", type=int, default=1, help="Negeer bestanden kleiner dan N bytes")
    ap.add_argument("--pad", type=int, default=3, help="Nulvulling voor index (default 3 → 001)")
    args = ap.parse_args()

    docx_path = Path(args.docx).expanduser().resolve()
    if not docx_path.exists():
        print(f"Bestand niet gevonden: {docx_path}", file=sys.stderr); sys.exit(1)

    out_dir = Path(args.out).expanduser().resolve() if args.out else docx_path.with_suffix("").parent / f"{docx_path.with_suffix('').name}_media"

    try:
        n = extract_images(docx_path, out_dir, prefix=args.prefix, dedupe=args.dedupe, min_bytes=args.min_bytes, pad=args.pad)
        print(f"Gereed: {n} bestand(en) → {out_dir}")
        print(f"Manifest: {out_dir / 'manifest.csv'}")
        print("Tip: Als er .emf/.wmf staan die je niet kan openen, converteer met Inkscape of ImageMagick naar .png.")
    except BadZipFile:
        print("Dit lijkt geen geldige .docx (ZIP). Sla in Word opnieuw op als .docx en probeer opnieuw.", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Fout: {e}", file=sys.stderr); sys.exit(3)
def gui_main():
    root = tk.Tk()
    root.withdraw()  # Geen hoofdvenster

    docx_path = filedialog.askopenfilename(
        title="Selecteer een Word (.docx) bestand",
        filetypes=[("Word Document", "*.docx")]
    )
    if not docx_path:
        return

    out_dir = filedialog.askdirectory(
        title="Kies uitvoermap (of annuleer voor standaardmap)"
    )
    if not out_dir:
        docx_path_obj = Path(docx_path)
        out_dir = str(docx_path_obj.with_suffix("").parent / f"{docx_path_obj.with_suffix('').name}_media")

    prefix = simpledialog.askstring("Bestandsprefix", "Prefix voor bestandsnamen:", initialvalue="Afbeelding_")
    if not prefix:
        prefix = "Afbeelding_"

    try:
        n = extract_images(
            Path(docx_path),
            Path(out_dir),
            prefix=prefix,
            dedupe=False,
            min_bytes=1,
            pad=3
        )
        messagebox.showinfo("Klaar", f"{n} bestand(en) geëxtraheerd naar:\n{out_dir}\nManifest: manifest.csv")
    except BadZipFile:
        messagebox.showerror("Fout", "Dit lijkt geen geldige .docx (ZIP). Sla in Word opnieuw op als .docx en probeer opnieuw.")
    except Exception as e:
        messagebox.showerror("Fout", str(e))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        gui_main()
    else:
        main()

