from __future__ import annotations

import html
import json
import re
import os
import stat
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote


DOCS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = DOCS_DIR.parent.parent
OUTPUT_HTML = DOCS_DIR / "oldstinus_overzicht.html"
OUTPUT_JSON = DOCS_DIR / "oldstinus_overzicht.json"
LAUNCHERS_DIR = DOCS_DIR / "launchers"

INCLUDED_EXTENSIONS = {".py", ".r", ".m", ".cr300", ".bat", ".md"}
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "QRevPy",
    "openMVG",
    "clone_imos-toolbox",
    "OLD",
    "Stages",
    "Verslagen interventies",
    "output",
    "docs",
}

VENV_DIR_NAMES = {".venv", "venv", "env", ".env", "ENV"}

TOP_LEVEL_LABELS = {
    "ADCP_Nortek_Aquadopp_AWAC_Vector": "ADCP",
    "ADCP_RDI_schepen_stationair": "ADCP",
    "ADCP_Sontek_Riversurveyor_M9_IQ": "ADCP",
    "Drukkamer_HF_DIVER": "Druksensoren",
    "Druksensor_HR_OSSI": "Druksensoren",
    "EXO-verwerking": "Multiparameter",
    "Log-a-level": "Peilmeting",
    "Multiparameter_Aanderaa": "Multiparameter",
    "Multiparameter_Aquatroll": "Multiparameter",
    "Multiparameter_YSI": "Multiparameter",
    "Radar_Vega_via_CR850": "Radar",
    "TOPO-RTK_GNSS-GPS": "GNSS/Topo",
    "WATERINFO": "Waterinfo/HTML",
    "Word-excel-ppt": "Documenten",
    "oldstinus": "Algemeen",
}

PROJECT_LABELS = {
    "ADCP_Nortek_Aquadopp_AWAC_Vector": "Nortek Aquadopp / AWAC / Vector",
    "ADCP_RDI_schepen_stationair": "RDI stationaire ADCP",
    "ADCP_Sontek_Riversurveyor_M9_IQ": "Sontek RiverSurveyor M9 / IQ",
    "Drukkamer_HF_DIVER": "HF druksensor + Diver",
    "Druksensor_HR_OSSI": "HR OSSI",
    "EXO-verwerking": "EXO-verwerking",
    "Log-a-level": "Log-a-level",
    "Multiparameter_Aanderaa": "Aanderaa Seaguard",
    "Multiparameter_Aquatroll": "Aquatroll",
    "Multiparameter_YSI": "YSI",
    "Radar_Vega_via_CR850": "Radar Vega via CR850",
    "TOPO-RTK_GNSS-GPS": "RTK / GNSS / topo",
    "WATERINFO": "Waterinfo en HTML-grafieken",
    "Word-excel-ppt": "Word / Excel / PPT workflows",
    "oldstinus": "Losse hulpscripts",
}


@dataclass
class ScriptInfo:
    rel_path: str
    name: str
    extension: str
    top_level: str
    category: str
    project: str
    subtype: str
    description: str
    input_summary: str
    output_summary: str
    operations: list[str]
    relevance: str
    version_hint: str
    priority_score: int
    is_priority: bool
    project_dir: str
    launcher_rel_path: str
    can_run_directly: bool
    is_old: bool


def is_excluded(path: Path) -> bool:
    return any(
        part in EXCLUDED_DIRS
        or part in VENV_DIR_NAMES
        or part.startswith(".venv_")
        or part == ".tmp"
        or re.fullmatch(r"tmp[a-z0-9_-]{6,}", part, flags=re.I)
        for part in path.parts
    )


def is_old_path(path: Path | str) -> bool:
    parts = path.parts if isinstance(path, Path) else Path(str(path).replace("\\", "/")).parts
    return any(part.lower() == "old" for part in parts)


def root_relative_path(path: Path) -> Path | None:
    try:
        rel = os.path.relpath(path, WORKSPACE_ROOT)
    except ValueError:
        return None
    rel_path = Path(rel)
    if rel == os.curdir:
        return Path(".")
    if rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
        return None
    return rel_path


def is_problem_directory(path: Path) -> bool:
    try:
        attrs = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return True
    if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        return True
    return not os.access(path, os.R_OK)


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(WORKSPACE_ROOT, topdown=True, onerror=lambda _err: None):
        current_dir = Path(dirpath)
        current_rel = root_relative_path(current_dir)
        if current_rel is None:
            dirnames[:] = []
            continue
        filtered_dirnames: list[str] = []
        for dirname in dirnames:
            rel_dir = current_rel / dirname
            full_dir = current_dir / dirname
            if is_excluded(rel_dir):
                continue
            if is_problem_directory(full_dir):
                continue
            filtered_dirnames.append(dirname)
        dirnames[:] = filtered_dirnames
        for filename in filenames:
            path = current_dir / filename
            if path.suffix.lower() not in INCLUDED_EXTENSIONS:
                continue
            rel_path = root_relative_path(path)
            if rel_path is None or is_excluded(rel_path):
                continue
            if not is_relevant_script(path):
                continue
            files.append(path)
    return sorted(files)


def read_text_sample(path: Path, max_chars: int = 120000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return ""


def read_head(path: Path, max_lines: int = 120) -> list[str]:
    text = read_text_sample(path)
    if not text:
        return []
    return text.splitlines()[:max_lines]


def venv_priority(path: Path) -> tuple[int, str]:
    name = path.name
    if name == ".venv_oldstinus":
        return (2, name)
    if name.startswith(".venv_"):
        return (0, name)
    if name in VENV_DIR_NAMES:
        return (1, name)
    return (3, name)


def find_venv_activate(project_dir: Path) -> Path:
    root_resolved = WORKSPACE_ROOT.resolve()
    seen: set[Path] = set()
    for current in [project_dir.resolve(), *project_dir.resolve().parents]:
        if current in seen:
            continue
        seen.add(current)
        if root_relative_path(current) is None and current != root_resolved:
            continue
        candidates: list[Path] = []
        if (current / "pyvenv.cfg").exists():
            candidates.append(current)
        try:
            for child in current.iterdir():
                if child.is_dir() and (child / "pyvenv.cfg").exists():
                    candidates.append(child)
        except OSError:
            candidates = candidates
        if candidates:
            chosen = sorted(candidates, key=venv_priority)[0]
            return chosen / "Scripts" / "activate.bat"
        if current == root_resolved:
            break
    return WORKSPACE_ROOT / ".venv_oldstinus" / "Scripts" / "activate.bat"


def extract_doc_hint(lines: list[str]) -> str:
    joined = "\n".join(lines[:20])
    docstring_match = re.search(r'"""(.*?)"""', joined, re.S)
    if docstring_match:
        extracted = [
            finalize_summary(line)
            for line in docstring_match.group(1).strip().splitlines()
            if is_useful_summary_line(line)
        ]
        for line in extracted:
            if is_purpose_candidate(line):
                return line
        if extracted:
            return extracted[0]
    for line in lines[:20]:
        stripped = line.strip().strip("#'/* ").strip()
        if len(stripped) < 12:
            continue
        if re.match(r"^!/.+", stripped):
            continue
        if re.match(r"^-\*-\s*coding[:=]\s*[-\w.]+(?:\s*-\*-)?$", stripped, flags=re.I):
            continue
        if re.match(r"^coding[:=]\s*[-\w.]+$", stripped, flags=re.I):
            continue
        lowered = stripped.lower()
        if lowered.startswith(("import ", "from ", "library(", "consttable", "def ", "class ", "return ", "if ", "for ", "while ")):
            continue
        if any(token in stripped for token in ("(", ")", "=>", "->", "=", "{", "}", "[", "]")):
            continue
        if stripped.startswith(("!", "@")):
            continue
        if not is_useful_summary_line(stripped):
            continue
        return finalize_summary(stripped)
    return ""


def clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -:#")
    return text[:180].strip()


def finalize_summary(text: str, max_chars: int = 180) -> str:
    text = clean_sentence(text)
    text = re.sub(r"^(input|output|gebruik|usage)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\s+(met|zoals|waarbij)$", "", text, flags=re.I)
    if not text:
        return ""
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text[:max_chars].rstrip()


def is_useful_summary_line(text: str) -> bool:
    cleaned = clean_sentence(text)
    if len(cleaned) < 12:
        return False
    lowered = cleaned.lower()
    if lowered in {"voorbeeld", "gebruik", "output", "input"}:
        return False
    if lowered.startswith(
        (
            "stel logging in",
            "logging.basicconfig",
            "author:",
            "copyright",
            "globale opslag",
            "hier worden de geladen data",
        )
    ):
        return False
    if re.fullmatch(r"[\w .-]+\.(py|r|m|bat|md|html|csv|xlsx|xls|pdf|docx)", cleaned, flags=re.I):
        return False
    return True


def is_purpose_candidate(text: str) -> bool:
    lowered = clean_sentence(text).lower()
    return lowered.startswith(
        (
            "leest ",
            "verwerkt ",
            "maakt ",
            "zet ",
            "vergelijk",
            "combineert ",
            "parseert ",
            "genereert ",
            "converteert ",
            "laat ",
            "reken ",
        )
    )


def extract_module_docstring_lines(text: str) -> list[str]:
    match = re.search(r'(?P<quote>"""|\'\'\')(.*?)(?P=quote)', text, re.S)
    if not match:
        return []
    return [line.strip() for line in match.group(2).splitlines()]


def extract_comment_lines(text: str, max_lines: int = 320) -> list[str]:
    comments: list[str] = []
    for raw_line in text.splitlines()[:max_lines]:
        stripped = raw_line.strip()
        if not stripped.startswith("#"):
            continue
        cleaned = re.sub(r"^#+\s*", "", stripped)
        cleaned = clean_sentence(cleaned)
        if not cleaned:
            continue
        if re.fullmatch(r"[-=_*]{3,}", cleaned):
            continue
        comments.append(cleaned)
    return comments


def combine_summary_parts(parts: list[str]) -> str:
    parts = [clean_sentence(part) for part in parts if clean_sentence(part)]
    if not parts:
        return ""
    if len(parts) == 1:
        return finalize_summary(parts[0])
    return finalize_summary(", ".join(parts[:-1]) + " en " + parts[-1])


def infer_description_from_comments(path: Path, rel_path: str, content: str) -> str:
    text_lower = content.lower()
    comments = extract_comment_lines(content)
    comments_blob = " ".join(comments).lower()

    if "awac" in text_lower and "vector" in text_lower:
        return (
            "Leest AWAC- en Vector-data in, vergelijkt en verwerkt de meetreeksen via GUI, "
            "maakt grafieken en exporteert vergeleken data."
        )

    if comments:
        priority_comments = [
            comment
            for comment in comments
            if any(
                token in comment.lower()
                for token in ("gui", "grafiek", "plot", "vergelijk", "verwerking", "export", "sensor")
            )
        ]
        if priority_comments:
            return finalize_summary(priority_comments[0])

    return ""


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "sectie"


def infer_subtype(rel_path: str) -> str:
    path_lower = rel_path.lower()
    mapping = [
        ("instrumenten\\vector", "Vector"),
        ("instrumenten\\awac", "AWAC"),
        ("instrumenten\\aquadopp", "Aquadopp"),
        ("\\iq\\", "IQ"),
        ("\\bathy\\", "Bathymetrie"),
        ("debieten&snelheden", "Debieten en snelheden"),
        ("gnss_rtk_koppelen_export", "GNSS-koppeling"),
        ("rtk_sokkia", "RTK Sokkia"),
        ("rtk_carlson", "RTK Carlson"),
        ("coordinaten conversie", "Coordinatenconversie"),
        ("ysi naar html", "YSI naar HTML"),
        ("realtime_ysi", "Realtime YSI"),
        ("dat naar csv", "DAT naar CSV"),
        ("nazicht documenten", "Documentnazicht"),
        ("projecten", "Projectspecifiek"),
    ]
    for needle, label in mapping:
        if needle in path_lower:
            return label
    return "Algemeen"


def infer_operations(rel_path: str, lines: list[str]) -> list[str]:
    blob = f"{rel_path.lower()}\n" + "\n".join(lines).lower()
    operations: list[str] = []
    checks = [
        ("GUI", ["tkinter", "ttk", "dateentry"]),
        ("HTML", ["html", "plotly", ".html"]),
        ("CSV inlezen", ["read_csv", ".csv", "csv."]),
        ("MAT/MATLAB", [".mat", "scipy.io", "matlab"]),
        ("Kaart/GIS", ["folium", "geo", "wgs84", "lambert72", "kml", "kmz"]),
        ("Vergelijken", ["vergelijk", "compare", "vs_"]),
        ("Kalibratie", ["kalibr", "referentie"]),
        ("Realtime", ["real-time", "realtime", "callback"]),
        ("OCR/documentextractie", ["ocr", "docx", "pdf", "handwriting"]),
        ("Excel/rapportering", ["openpyxl", "xlsx", "excel", "ppt", "word"]),
        ("Conversie", ["convert", "converter", "transformatie", "naar"]),
        ("Visualisatie", ["matplotlib", "plot", "grafiek"]),
    ]
    for label, needles in checks:
        if any(needle in blob for needle in needles):
            operations.append(label)
    if not operations:
        operations.append("Dataverwerking")
    return operations


def extract_function_names(text: str) -> list[str]:
    names = re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.M)
    ordered: list[str] = []
    for name in names:
        if name.startswith("_"):
            continue
        if name not in ordered:
            ordered.append(name)
    return ordered[:6]


def infer_description_from_content(path: Path, rel_path: str, text: str) -> str:
    text_lower = text.lower()
    functions = extract_function_names(text)
    stem_lower = path.stem.lower().replace("-", " ").replace("_", " ")

    if "askopenfilename" in text_lower or "askdirectory" in text_lower:
        if "plotly" in text_lower and "html" in text_lower:
            return "Laat bestanden kiezen via GUI en exporteert een interactieve HTML-weergave."
        if "matplotlib" in text_lower:
            return "Laat meetbestanden kiezen via GUI en visualiseert de resultaten."
    if "folium" in text_lower or "lambert72" in text_lower or "wgs84" in text_lower:
        return "Leest meetpunten in, zet coordinaten om en maakt kaartoutput."
    if "openpyxl" in text_lower or ".xlsx" in text_lower or "workbook" in text_lower:
        return "Leest Excelbestanden in, verwerkt tabellen en schrijft aangepaste output weg."
    if "docx" in text_lower or "pdf" in text_lower or "ocr" in text_lower:
        return "Leest documenten in, extraheert inhoud en ondersteunt documentverwerking."
    if "read_csv" in text_lower and "plotly" in text_lower and "html" in text_lower:
        return "Leest csv-data in en zet die om naar een interactieve HTML-grafiek."
    if "read_csv" in text_lower and "matplotlib" in text_lower:
        return "Leest csv-data in en maakt een grafische visualisatie."
    if "scipy.io" in text_lower or ".mat" in text_lower:
        return "Leest MAT-bestanden in en verwerkt meetreeksen of ruimtelijke data."
    if "serial" in text_lower or "real-time" in text_lower or "realtime" in text_lower:
        return "Verwerkt realtime meetdata en stuurt de visualisatie of logging aan."
    if "reference" in text_lower or "referentie" in text_lower or "kalibr" in text_lower:
        return "Vergelijkt meetdata met referenties en ondersteunt kalibratie."
    if functions:
        verbs = {
            "load": "laadt data in",
            "read": "leest data in",
            "parse": "parseert bestanden",
            "compare": "vergelijkt datasets",
            "plot": "maakt grafieken",
            "export": "exporteert resultaten",
            "convert": "zet data om",
            "process": "verwerkt data",
            "merge": "combineert reeksen",
        }
        for name in functions:
            for prefix, phrase in verbs.items():
                if name.lower().startswith(prefix):
                    return f"Script dat {phrase} voor {path.stem.replace('_', ' ').replace('-', ' ')}."
    if "waterinfo" in rel_path.lower():
        return "Verwerkt Waterinfo-export en zet die om naar grafieken of html-output."
    if "ysi" in stem_lower:
        return "Verwerkt YSI-metingen en zet die om naar html, csv of visualisaties."
    return ""


def infer_description(path: Path, rel_path: str, lines: list[str], content: str) -> str:
    doc_lines = extract_module_docstring_lines(content)
    from_doc = docstring_summary(
        doc_lines,
        (
            "leest ",
            "verwerkt ",
            "maakt ",
            "zet ",
            "vergelijk",
            "combineert ",
            "parseert ",
            "genereert ",
            "converteert ",
            "laat ",
            "reken ",
        ),
    )
    if from_doc:
        return from_doc

    comment_hint = infer_description_from_comments(path, rel_path, content)
    if comment_hint:
        return comment_hint

    hint = extract_doc_hint(lines)
    if hint:
        return hint

    content_hint = infer_description_from_content(path, rel_path, content)
    if content_hint:
        return content_hint

    stem = path.stem.replace("_", " ").replace("-", " ")
    stem_lower = stem.lower()
    if "ysi" in stem_lower and "html" in stem_lower:
        return "Zet YSI-metingen om naar interactieve HTML-grafieken."
    if "waterinfo" in stem_lower and "html" in stem_lower:
        return "Leest Waterinfo-export in en bouwt een interactieve HTML-visualisatie."
    if "waterinfo" in stem_lower and "par" in stem_lower:
        return "Zet Waterinfo PAR-data om naar een interactieve grafiek."
    if "waterinfo" in stem_lower and "parameters" in stem_lower and "2 assen" in stem_lower:
        return "Maakt een interactieve Waterinfo-grafiek met twee y-assen."
    if "waterinfo" in stem_lower and "parameters" in stem_lower:
        return "Zet Waterinfo-parameterbestanden om naar interactieve HTML-grafieken."
    if "exo" in stem_lower and "html" in stem_lower:
        return "Leest EXO-data in en exporteert een interactieve HTML-weergave."
    if "compare" in stem_lower or "vergelijk" in stem_lower:
        return "Vergelijkt datasets of toestellen en exporteert verschillen/samenvattingen."
    if "ocr" in stem_lower or "docx" in stem_lower or "pdf" in stem_lower:
        return "Leest documenten in, extraheert inhoud en ondersteunt documentbewerking."
    if "radar" in stem_lower:
        return "Leest radarloggerdata in en zet die om naar bruikbare reeksen."
    if "bathy" in stem_lower:
        return "Verwerkt bathymetrie en exporteert punten, kaarten of gekoppelde datasets."
    if "debiet" in stem_lower or "snelheid" in stem_lower:
        return "Verwerkt debieten, snelheden of tijdreeksen uit instrumentdata."
    if "rtk" in stem_lower or "gnss" in stem_lower or "topo" in stem_lower:
        return "Leest GNSS/topodata in, vergelijkt posities en maakt kaartoutput."
    if "diver" in stem_lower or "druk" in stem_lower:
        return "Combineert drukreeksen, offsets en visualisaties voor druksensoren."
    if "csv" in stem_lower and "gui" in stem_lower:
        return "Generieke GUI om CSV-data in te lezen en snel te visualiseren."
    return f"Script voor {stem.strip()}."


def collect_dialog_extensions(text: str, function_names: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for function_name in function_names:
        for match in re.finditer(function_name, text, flags=re.I):
            snippet = text[match.start():match.start() + 350]
            for ext in re.findall(r"\*\.[A-Za-z0-9]+", snippet):
                ext_lower = ext.lower()
                if ext_lower not in found:
                    found.append(ext_lower)
    return found


def describe_extension_kind(ext: str, rel_path: str) -> str:
    rel_lower = rel_path.lower()
    if ext == "*.csv":
        if "korexo" in rel_lower or "exo" in rel_lower:
            return "KorEXO CSV-export"
        if "waterinfo" in rel_lower:
            return "Waterinfo CSV-export"
        if "ysi" in rel_lower:
            return "YSI CSV-export"
        return "CSV-bestanden"
    if ext == "*.mat":
        if "m9" in rel_lower:
            return "Sontek M9 MAT-bestanden"
        return "MAT-bestanden"
    if ext == "*.dat":
        if "ysi" in rel_lower:
            return "YSI DAT-bestanden"
        return "DAT-bestanden"
    if ext in {"*.xlsx", "*.xls"}:
        return "Excel-bestanden"
    if ext == "*.pdf":
        return "PDF-documenten"
    if ext == "*.docx":
        return "Word-documenten"
    if ext in {"*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tif", "*.tiff"}:
        return "Logo- of afbeeldingsbestanden"
    if ext == "*.html":
        return "HTML-bestanden"
    if ext == "*.kml":
        return "KML-bestanden"
    if ext == "*.dae":
        return "DAE-bestanden"
    if ext == "*.txt":
        return "Tekstbestanden"
    return ext.replace("*.", "").upper() + "-bestanden"


def unique_labels(labels: list[str]) -> list[str]:
    ordered: list[str] = []
    for label in labels:
        if label and label not in ordered:
            ordered.append(label)
    return ordered


def summarize_labels(labels: list[str]) -> str:
    labels = unique_labels(labels)
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} en {labels[1]}"
    return f"{', '.join(labels[:-1])} en {labels[-1]}"


def reduce_output_labels(labels: list[str]) -> list[str]:
    labels = unique_labels(labels)
    pairs = {
        "Interactieve HTML-grafiek": "HTML-bestanden",
        "CSV-export": "CSV-bestanden",
        "KML-export": "KML-bestanden",
        "DAE 3D-model": "DAE-bestanden",
        "Excel-output": "Excel-bestanden",
    }
    for preferred, generic in pairs.items():
        if preferred in labels and generic in labels:
            labels.remove(generic)
    return labels


def docstring_summary(doc_lines: list[str], prefixes: tuple[str, ...]) -> str:
    for raw_line in doc_lines:
        if not is_useful_summary_line(raw_line):
            continue
        line = finalize_summary(raw_line)
        lowered = line.lower()
        if any(lowered.startswith(prefix) for prefix in prefixes):
            return line
    return ""


def infer_input_summary(path: Path, rel_path: str, content: str) -> str:
    doc_lines = extract_module_docstring_lines(content)

    text_lower = content.lower()
    if "awac" in text_lower and "vector" in text_lower and all(token in text_lower for token in (".dat", ".sen", ".hdr", ".csv")):
        return "Vector DAT/SEN/HDR en AWAC CSV."
    input_kinds = [
        describe_extension_kind(ext, rel_path)
        for ext in collect_dialog_extensions(content, ("askopenfilename", "askopenfilenames"))
    ]
    if "askdirectory(" in text_lower:
        input_kinds.append("Map met meetbestanden")
    if not input_kinds and ("read_csv" in text_lower or "csv.reader" in text_lower or ".csv" in text_lower):
        input_kinds.append(describe_extension_kind("*.csv", rel_path))
    if not input_kinds and (".mat" in text_lower or "loadmat" in text_lower or "scipy.io" in text_lower):
        input_kinds.append(describe_extension_kind("*.mat", rel_path))
    if "read_excel" in text_lower or "openpyxl" in text_lower or ".xlsx" in text_lower:
        input_kinds.append("Excel-bestanden")
    if ".pdf" in text_lower or "pdfplumber" in text_lower or "pypdf" in text_lower:
        input_kinds.append("PDF-documenten")
    if ".docx" in text_lower or "docx" in text_lower:
        input_kinds.append("Word-documenten")
    if "image.open" in text_lower or any(ext in text_lower for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp")):
        input_kinds.append("Logo- of afbeeldingsbestanden")
    if "serial." in text_lower or "pyserial" in text_lower:
        input_kinds.append("Seriele live meetdata")

    summary = summarize_labels(input_kinds[:3])
    if summary:
        return finalize_summary(summary)
    from_doc = docstring_summary(doc_lines, ("leest ", "parseert ", "verwacht ", "selecteer ", "kies "))
    if from_doc:
        return from_doc
    return "Meetbestanden of exports volgens het script."


def infer_output_summary(path: Path, rel_path: str, content: str) -> str:
    doc_lines = extract_module_docstring_lines(content)
    from_doc = docstring_summary(doc_lines, ("output:", "produceert ", "exporteert ", "schrijf ", "schrijft ", "slaat ", "bewaar "))
    if from_doc:
        return from_doc

    text_lower = content.lower()
    if "awac" in text_lower and "vector" in text_lower and ("matplotlib" in text_lower or "plt.subplots" in text_lower):
        if "to_csv(" in text_lower:
            return "Grafieken in GUI en CSV-export."
        return "Grafieken in GUI."
    output_kinds: list[str] = []
    save_exts = collect_dialog_extensions(content, ("asksaveasfilename",))
    if save_exts:
        output_kinds.extend(describe_extension_kind(ext, rel_path) for ext in save_exts)
    save_exts_set = set(save_exts)
    if (
        "fig.write_html" in text_lower
        or "pio.write_html" in text_lower
        or ("write_text" in text_lower and ".html" in text_lower)
        or "*.html" in save_exts_set
    ):
        output_kinds.append("Interactieve HTML-grafiek")
    if (
        "to_csv(" in text_lower
        or "csv.writer(" in text_lower
        or "*.csv" in save_exts_set
    ):
        output_kinds.append("CSV-export")
    if any(token in text_lower for token in ("to_excel", ".xlsx", "workbook.save", "save_workbook")):
        output_kinds.append("Excel-output")
    if any(token in text_lower for token in ("savefig", ".png", ".jpg", ".jpeg")) and "write_html" not in text_lower:
        output_kinds.append("Afbeeldingsoutput")
    if ".kml" in text_lower or "simplekml" in text_lower:
        output_kinds.append("KML-export")
    if ".dae" in text_lower:
        output_kinds.append("DAE 3D-model")
    if "folium" in text_lower and "html" in text_lower:
        output_kinds.append("HTML-kaart")
    if "messagebox.showinfo" in text_lower and "opgeslagen" in text_lower and not output_kinds:
        output_kinds.append("Bestandsoutput in projectmap of gekozen doelmap")

    summary = summarize_labels(reduce_output_labels(output_kinds)[:3])
    if summary:
        return finalize_summary(summary)
    if "tkinter" in text_lower or "matplotlib" in text_lower or "plotly" in text_lower:
        return "Visualisatie in GUI of browser."
    return "Verwerkte output volgens het script."


def infer_relevance(path: Path) -> tuple[str, int, bool]:
    name = path.name.lower()
    score = 0
    tags: list[str] = []
    if "super" in name:
        score += 50
        tags.append("SUPER")
    if re.search(r"(^|[^a-z])ok([^a-z]|$)", name):
        score += 40
        tags.append("OK")
    if "gui" in name:
        score += 10
        tags.append("GUI")
    if "html" in name:
        score += 8
        tags.append("HTML")
    if "real-time" in name or "realtime" in name:
        score += 8
        tags.append("Realtime")
    version_score = extract_version_score(name)
    score += version_score
    priority = score >= 40
    return (", ".join(tags) if tags else "standaard"), score, priority


def is_relevant_script(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() == ".md":
        return False
    if "super" in name:
        return True
    if re.search(r"(^|[^a-z])ok([^a-z]|$)", name):
        return True
    if re.search(r"versie[_ -]?\d{4}[_-]\d{2}[_-]\d{2}", name):
        return True
    if re.search(r"v\d+(?:-v\d+)*", name):
        return True
    if re.search(r"\d{4}[_-]\d{2}[_-]\d{2}", name) and any(
        token in name for token in ("gui", "html", "koppelen", "compare", "plot", "bathy")
    ):
        return True
    return False


def extract_version_score(name: str) -> int:
    score = 0
    if re.search(r"v\d+", name):
        score += 6
    if re.search(r"\d{4}[_-]\d{2}[_-]\d{2}", name):
        score += 12
    if re.search(r"versie[_ -]?\d{4}[_-]\d{2}[_-]\d{2}", name):
        score += 14
    return score


def extract_version_hint(name: str) -> str:
    found: list[str] = []
    for match in re.findall(r"v\d+(?:-v\d+)*", name, flags=re.I):
        found.append(match)
    for match in re.findall(r"\d{4}[_-]\d{2}[_-]\d{2}", name):
        found.append(match.replace("_", "-"))
    for match in re.findall(r"versie[_ -]?\d{4}[_-]\d{2}[_-]\d{2}", name, flags=re.I):
        found.append(match.replace("_", "-"))
    return ", ".join(dict.fromkeys(found))


def classify_file(path: Path) -> ScriptInfo:
    rel_path_obj = root_relative_path(path)
    project_dir_obj = root_relative_path(path.parent)
    if rel_path_obj is None or project_dir_obj is None:
        raise ValueError(f"Pad ligt buiten ROOT: {path}")
    rel_path = rel_path_obj.as_posix().replace("/", "\\")
    rel_parts = Path(rel_path.replace("\\", "/")).parts
    top_level = rel_parts[0] if len(rel_parts) > 1 else WORKSPACE_ROOT.name
    content = read_text_sample(path)
    lines = read_head(path)
    relevance, score, is_priority = infer_relevance(path)
    version_hint = extract_version_hint(path.name)
    project_dir = project_dir_obj.as_posix().replace("/", "\\")
    launcher_rel_path = f"launchers\\{slugify(rel_path)}.bat"
    return ScriptInfo(
        rel_path=rel_path,
        name=path.name,
        extension=path.suffix.lower(),
        top_level=top_level,
        category=TOP_LEVEL_LABELS.get(top_level, "Overig"),
        project=PROJECT_LABELS.get(top_level, top_level),
        subtype=infer_subtype(rel_path),
        description=infer_description(path, rel_path, lines, content),
        input_summary=infer_input_summary(path, rel_path, content),
        output_summary=infer_output_summary(path, rel_path, content),
        operations=infer_operations(rel_path, lines),
        relevance=relevance,
        version_hint=version_hint or "-",
        priority_score=score,
        is_priority=is_priority,
        project_dir=project_dir,
        launcher_rel_path=launcher_rel_path,
        can_run_directly=path.suffix.lower() == ".py",
        is_old=is_old_path(rel_path),
    )


def rel_href(from_dir: Path, to_path: Path) -> str:
    if root_relative_path(to_path) is None:
        raise ValueError(f"Pad ligt buiten ROOT: {to_path}")
    rel = os.path.relpath(to_path, from_dir)
    return quote(Path(rel).as_posix(), safe="/#():")


def vscode_href(path: Path) -> str:
    return f"vscode://file/{quote(path.resolve().as_posix(), safe='/:')}"


def badge(text: str, cls: str = "") -> str:
    cls_attr = f' class="badge {cls}"' if cls else ' class="badge"'
    return f"<span{cls_attr}>{html.escape(text)}</span>"


def render_card(item: ScriptInfo) -> str:
    file_link = rel_href(OUTPUT_HTML.parent, WORKSPACE_ROOT / item.rel_path)
    launcher_link = (LAUNCHERS_DIR / Path(item.launcher_rel_path).name).resolve().as_uri()
    editor_link = vscode_href(WORKSPACE_ROOT / item.rel_path)
    ops = "".join(badge(op) for op in item.operations[:4])
    emphasis = badge(item.relevance, "accent") if item.is_priority else badge(item.relevance)
    version = badge(item.version_hint) if item.version_hint != "-" else ""
    project_label = item.project_dir if item.project_dir not in ("", ".") else WORKSPACE_ROOT.name
    if item.can_run_directly:
        action_links = (
            f'<a class="action-btn" href="{editor_link}">Open in VS Code</a>'
            f'<a class="action-link" href="{launcher_link}" target="_blank" rel="noopener">Start via launcher (.bat)</a>'
        )
    else:
        action_links = (
            f'<a class="action-btn" href="{editor_link}">Open in VS Code</a>'
            f'<a class="action-link" href="{file_link}">Bestand</a>'
        )
    return f"""
    <article class="card">
      <div class="card-top">
        <h4><a href="{file_link}">{html.escape(item.name)}</a></h4>
        <div class="badges">{emphasis}{version}</div>
      </div>
      <div class="summary">
        <div><strong>Toepassing:</strong> {html.escape(item.description)}</div>
        <div><strong>Input:</strong> {html.escape(item.input_summary)}</div>
        <div><strong>Output:</strong> {html.escape(item.output_summary)}</div>
      </div>
      <div class="actions">{action_links}</div>
      <div class="meta">{ops}</div>
      <div class="path">Project: {html.escape(project_label)}<br>Script: {html.escape(item.rel_path)}</div>
    </article>
    """


def generate_launcher(item: ScriptInfo) -> None:
    launcher_path = LAUNCHERS_DIR / Path(item.launcher_rel_path).name
    project_dir = WORKSPACE_ROOT / item.project_dir
    script_path = WORKSPACE_ROOT / item.rel_path
    project_q = str(project_dir)
    script_q = str(script_path)
    code_path = r"C:\Users\claeysst\AppData\Local\Programs\Microsoft VS Code\Code.exe"
    venv_activate = find_venv_activate(project_dir)
    lines = [
        "@echo off",
        "setlocal",
        f'set "PROJECT_DIR={project_q}"',
        f'set "SCRIPT_PATH={script_q}"',
        f'set "CODE_EXE={code_path}"',
        f'set "VENV_ACTIVATE={venv_activate}"',
        "",
        'if exist "%CODE_EXE%" (',
        '  start "" "%CODE_EXE%" --new-window "%PROJECT_DIR%" "%SCRIPT_PATH%"',
        ") else (",
        '  echo VS Code niet gevonden op "%CODE_EXE%".',
        ")",
        "",
    ]
    if item.can_run_directly:
        lines.extend(
            [
                'if exist "%VENV_ACTIVATE%" (',
                '  start "oldstinus-script" cmd /K "cd /d ""%PROJECT_DIR%"" && call ""%VENV_ACTIVATE%"" && python ""%SCRIPT_PATH%"""',
                ") else (",
                '  start "oldstinus-script" cmd /K "cd /d ""%PROJECT_DIR%"" && python ""%SCRIPT_PATH%"""',
                ")",
            ]
        )
    else:
        lines.extend(
            [
                'if exist "%VENV_ACTIVATE%" (',
                '  start "oldstinus-script" cmd /K "cd /d ""%PROJECT_DIR%"" && call ""%VENV_ACTIVATE%"" && echo Klaar in projectmap: %PROJECT_DIR% && echo Bestand: %SCRIPT_PATH%"',
                ") else (",
                '  start "oldstinus-script" cmd /K "cd /d ""%PROJECT_DIR%"" && echo Klaar in projectmap: %PROJECT_DIR% && echo Bestand: %SCRIPT_PATH%"',
                ")",
            ]
        )
    launcher_path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def build_html(items: list[ScriptInfo]) -> str:
    main_items = [item for item in items if not item.is_old]
    old_items = [item for item in items if item.is_old]
    counts = Counter(item.category for item in main_items)
    operation_counts = Counter(op for item in main_items for op in item.operations)
    project_counts = Counter(item.project for item in main_items)
    priority_items = sorted(
        [item for item in main_items if item.is_priority],
        key=lambda item: (-item.priority_score, item.rel_path.lower()),
    )[:24]

    by_category: dict[str, dict[str, list[ScriptInfo]]] = defaultdict(lambda: defaultdict(list))
    docs_ops: list[ScriptInfo] = []
    by_operation: dict[str, list[ScriptInfo]] = defaultdict(list)
    for item in main_items:
        by_category[item.category][f"{item.project} :: {item.subtype}"].append(item)
        for operation in item.operations:
            by_operation[operation].append(item)
        if any(op in item.operations for op in ("OCR/documentextractie", "Excel/rapportering", "HTML")):
            docs_ops.append(item)

    category_blocks: list[str] = []
    for category in sorted(by_category):
        project_sections: list[str] = []
        for key in sorted(by_category[category]):
            group_items = sorted(
                by_category[category][key],
                key=lambda item: (-item.priority_score, item.name.lower()),
            )[:18]
            project, subtype = key.split(" :: ", 1)
            group_id = slugify(f"group-{category}-{project}-{subtype}")
            project_sections.append(
                f"""
                <section class="group" id="{group_id}">
                  <div class="group-head">
                    <h3>{html.escape(project)}</h3>
                    <span>{html.escape(subtype)}</span>
                  </div>
                  <div class="grid">
                    {''.join(render_card(item) for item in group_items)}
                  </div>
                </section>
                """
            )
        category_blocks.append(
            f"""
            <section class="category" id="{slugify(f'category-{category}')}">
              <div class="section-head">
                <h2>{html.escape(category)}</h2>
                <span>{counts[category]} scripts</span>
              </div>
              {''.join(project_sections)}
            </section>
            """
        )

    operation_blocks: list[str] = []
    for operation in sorted(by_operation):
        op_items = sorted(
            by_operation[operation],
            key=lambda item: (-item.priority_score, item.project.lower(), item.name.lower()),
        )[:18]
        operation_blocks.append(
            f"""
            <section class="category" id="{slugify(f'operation-{operation}')}">
              <div class="section-head">
                <h2>{html.escape(operation)}</h2>
                <span>{operation_counts[operation]} scripts</span>
              </div>
              <div class="grid">{''.join(render_card(item) for item in op_items)}</div>
            </section>
            """
        )

    doc_cards = "".join(
        render_card(item)
        for item in sorted(docs_ops, key=lambda item: (-item.priority_score, item.rel_path.lower()))[:18]
    )
    old_cards = "".join(
        render_card(item)
        for item in sorted(old_items, key=lambda item: (item.category.lower(), item.project.lower(), -item.priority_score, item.rel_path.lower()))[:36]
    )

    summary_cards = "".join(
        f'<div class="stat"><strong>{count}</strong><span>{html.escape(category)}</span></div>'
        for category, count in sorted(counts.items())
    )

    priority_html = "".join(render_card(item) for item in priority_items)
    category_options = "".join(
        f'<option value="#{slugify(f"category-{category}")}">{html.escape(category)} ({counts[category]})</option>'
        for category in sorted(counts)
    )
    project_options = "".join(
        f'<option value="#{slugify(f"group-{item.category}-{item.project}-{item.subtype}")}">{html.escape(item.project)} / {html.escape(item.subtype)}</option>'
        for item in sorted(
            {
                (item.category, item.project, item.subtype): item
                for item in main_items
            }.values(),
            key=lambda item: (item.category.lower(), item.project.lower(), item.subtype.lower()),
        )
    )
    operation_options = "".join(
        f'<option value="#{slugify(f"operation-{operation}")}">{html.escape(operation)} ({operation_counts[operation]})</option>'
        for operation in sorted(operation_counts)
    )
    quick_links = "".join(
        f'<a href="#{slugify(f"category-{category}")}" data-jump-filter="#{slugify(f"category-{category}")}">{html.escape(category)}</a>'
        for category in sorted(counts)
    )

    return f"""<!DOCTYPE html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>oldstinus overzicht</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --paper: #fffaf3;
      --ink: #17202a;
      --muted: #59636e;
      --line: #d8cfc0;
      --accent: #045d56;
      --accent-soft: #dff2ee;
      --warm: #b7652b;
      --shadow: 0 14px 40px rgba(23, 32, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(183, 101, 43, 0.10), transparent 28%),
        linear-gradient(180deg, #f7f1e8 0%, #f3ede4 45%, #efe8dd 100%);
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .shell {{
      width: min(1380px, calc(100vw - 32px));
      margin: 24px auto 56px;
    }}
    .hero {{
      background: linear-gradient(140deg, rgba(4,93,86,0.95), rgba(23,32,42,0.92));
      color: white;
      border-radius: 28px;
      padding: 32px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 12px;
      font-size: clamp(2rem, 5vw, 4.2rem);
      line-height: 0.98;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      max-width: 880px;
      margin: 0;
      color: rgba(255,255,255,0.88);
      font-size: 1.02rem;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 14px;
      margin-top: 24px;
    }}
    .stat {{
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 20px;
      padding: 14px 16px;
      backdrop-filter: blur(4px);
    }}
    .stat strong {{
      display: block;
      font-size: 1.7rem;
      margin-bottom: 4px;
    }}
    .stat span {{
      color: rgba(255,255,255,0.84);
      font-size: 0.92rem;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin: 20px 0 28px;
      padding: 18px 20px;
      background: rgba(255,250,243,0.92);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      position: sticky;
      top: 10px;
      z-index: 10;
    }}
    .toolbar input {{
      flex: 1 1 280px;
      min-width: 220px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 0.98rem;
      background: white;
    }}
    .toolbar select {{
      flex: 0 1 240px;
      min-width: 200px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 0.95rem;
      background: white;
      color: var(--ink);
    }}
    .legend {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .quick-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .quick-links a {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.1);
      color: white;
      border: 1px solid rgba(255,255,255,0.14);
    }}
    .nav-block {{
      margin-top: 24px;
      padding: 18px 20px;
      background: rgba(255,250,243,0.92);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .nav-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .nav-grid label {{
      display: block;
      color: var(--muted);
      font-size: 0.86rem;
      margin-bottom: 6px;
    }}
    .section-head, .group-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin: 30px 0 14px;
    }}
    .section-head h2, .group-head h3 {{
      margin: 0;
    }}
    .section-head h2 {{
      font-size: 1.7rem;
    }}
    .group-head h3 {{
      font-size: 1.15rem;
    }}
    .section-head span, .group-head span {{
      color: var(--warm);
      font-size: 0.95rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }}
    .card h4 {{
      margin: 0;
      font-size: 0.98rem;
      line-height: 1.3;
      max-width: 100%;
      overflow-wrap: anywhere;
      word-break: break-word;
      hyphens: auto;
    }}
    .summary {{
      margin: 12px 0;
      color: var(--muted);
      min-height: 92px;
      font-size: 0.92rem;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .summary div + div {{
      margin-top: 6px;
    }}
    .summary strong {{
      color: var(--ink);
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 10px 0 12px;
      align-items: center;
    }}
    .action-btn, .action-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      padding: 8px 12px;
      border-radius: 12px;
      font-size: 0.86rem;
      line-height: 1.2;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      text-decoration: none;
    }}
    .action-btn {{
      background: var(--accent-soft);
      color: var(--accent);
      border-color: rgba(4,93,86,0.18);
      font-weight: 600;
    }}
    .meta, .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .badges {{
      justify-content: flex-end;
      flex: 0 0 auto;
      max-width: 42%;
    }}
    .card-top > h4 {{
      flex: 1 1 auto;
      min-width: 0;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 5px 9px;
      border-radius: 999px;
      background: #efe7d8;
      color: #5f4b34;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .badge.accent {{
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
    }}
    .path {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px dashed var(--line);
      color: #756b5d;
      font-family: Consolas, monospace;
      font-size: 0.74rem;
      line-height: 1.35;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .category {{
      margin-top: 40px;
    }}
    .note {{
      margin-top: 34px;
      background: rgba(255,250,243,0.88);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      color: var(--muted);
    }}
    .hidden {{ display: none !important; }}
    .view-hidden {{ display: none !important; }}
    @media (max-width: 720px) {{
      .hero, .toolbar, .card {{ border-radius: 18px; }}
      .shell {{ width: min(100vw - 18px, 1380px); margin: 10px auto 40px; }}
      .hero {{ padding: 24px 18px; }}
      .toolbar {{ padding: 14px; position: static; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>oldstinus overzicht</h1>
      <p>
        Automatisch gegenereerde catalogus van scripts per toesteltype, per toestel en voor
        document- en dataverwerking. Bestanden met <strong>OK</strong>, <strong>SUPER</strong> en
        expliciete versies krijgen extra gewicht zodat de meest bruikbare varianten eerst zichtbaar zijn.
      </p>
      <div class="stats">{summary_cards}</div>
      <div class="quick-links">{quick_links}</div>
    </section>

    <section class="toolbar">
      <input id="search" type="search" placeholder="Zoek op toestel, script, operatie of pad">
      <div class="legend">Filtert live op bestandsnaam, uitleg en pad.</div>
    </section>

    <section class="nav-block">
      <div class="section-head">
        <h2>Snel navigeren</h2>
        <span>Per instrumenttype, toestel of bewerking</span>
      </div>
      <div class="nav-grid">
        <div>
          <label for="jump-category">Instrumenttype</label>
          <select id="jump-category">
            <option value="">Kies instrumenttype</option>
            {category_options}
          </select>
        </div>
        <div>
          <label for="jump-project">Toestel / project</label>
          <select id="jump-project">
            <option value="">Kies toestel of project</option>
            {project_options}
          </select>
        </div>
        <div>
          <label for="jump-operation">Type bewerking</label>
          <select id="jump-operation">
            <option value="">Kies bewerking</option>
            {operation_options}
          </select>
        </div>
      </div>
      <div class="quick-links" style="margin-top:14px;">
        <button id="show-all" type="button">Terug naar volledig overzicht</button>
      </div>
    </section>

    <section data-filter-section>
      <div class="section-head">
        <h2>Prioritaire scripts</h2>
        <span>OK / SUPER / versies eerst</span>
      </div>
      <div class="grid">{priority_html}</div>
    </section>

    <section class="category" data-filter-section>
      <div class="section-head">
        <h2>Documenten en andere bewerkingen</h2>
        <span>OCR, HTML, Excel en extractie</span>
      </div>
      <div class="grid">{doc_cards}</div>
    </section>

    <section class="category" id="bewerkingen" data-filter-section>
      <div class="section-head">
        <h2>Per bewerking</h2>
        <span>Zelfde scripts hergroepeerd per taak</span>
      </div>
      {''.join(operation_blocks)}
    </section>

    <section class="category" id="category-old" data-filter-section>
      <div class="section-head">
        <h2>Old</h2>
        <span>{len(old_items)} scripts uit old/OLD-mappen</span>
      </div>
      <div class="grid">{old_cards}</div>
    </section>

    {''.join(category_blocks)}

    <section class="note">
      De scan focust op operationele scripts en filtert zware omgevingen of vendor-code uit
      zoals <code>.venv</code>, <code>node_modules</code>, <code>QRevPy</code> en grote legacy-clones.
      De pagina wordt opnieuw opgebouwd door <code>Word-excel-ppt/docs/generate_oldstinus_catalog.py</code>.
    </section>
  </div>

  <script>
    const input = document.getElementById('search');
    const cards = [...document.querySelectorAll('.card')];
    const selects = ['jump-category', 'jump-project', 'jump-operation']
      .map(id => document.getElementById(id))
      .filter(Boolean);
    const quickJumpLinks = [...document.querySelectorAll('[data-jump-filter]')];
    const filterSections = [...document.querySelectorAll('[data-filter-section], .category, .group')];
    const showAllButton = document.getElementById('show-all');
    const clearSelects = () => {{
      selects.forEach(select => {{
        select.value = '';
      }});
    }};
    const showAllSections = () => {{
      filterSections.forEach(section => section.classList.remove('view-hidden'));
    }};
    const showOnlyTarget = (target) => {{
      const keep = new Set();
      let current = target;
      while (current) {{
        if (current.matches && (current.matches('.group') || current.matches('.category') || current.hasAttribute('data-filter-section'))) {{
          keep.add(current);
        }}
        current = current.parentElement;
      }}
      if (target.querySelectorAll) {{
        target.querySelectorAll('.group, .category, [data-filter-section]').forEach(section => {{
          keep.add(section);
        }});
      }}
      filterSections.forEach(section => {{
        section.classList.toggle('view-hidden', !keep.has(section));
      }});
    }};
    const jumpTo = (value) => {{
      if (!value) return;
      const target = document.querySelector(value);
      if (target) {{
        showOnlyTarget(target);
        target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }}
    }};
    selects.forEach(select => {{
      select.addEventListener('change', () => {{
        const chosen = select.value;
        if (!chosen) {{
          showAllSections();
          return;
        }}
        selects.forEach(other => {{
          if (other !== select) other.value = '';
        }});
        jumpTo(chosen);
      }});
    }});
    quickJumpLinks.forEach(link => {{
      link.addEventListener('click', (event) => {{
        event.preventDefault();
        clearSelects();
        jumpTo(link.dataset.jumpFilter);
      }});
    }});
    showAllButton.addEventListener('click', () => {{
      clearSelects();
      showAllSections();
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }});
    input.addEventListener('input', () => {{
      const needle = input.value.trim().toLowerCase();
      cards.forEach(card => {{
        const text = card.innerText.toLowerCase();
        card.classList.toggle('hidden', needle && !text.includes(needle));
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    items = [classify_file(path) for path in iter_source_files()]
    items.sort(key=lambda item: (-item.priority_score, item.category, item.project, item.rel_path.lower()))
    LAUNCHERS_DIR.mkdir(parents=True, exist_ok=True)
    for existing in LAUNCHERS_DIR.glob("*.bat"):
        existing.unlink()
    for item in items:
        generate_launcher(item)
    OUTPUT_JSON.write_text(
        json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUTPUT_HTML.write_text(build_html(items), encoding="utf-8")
    print(f"Gegenereerd: {OUTPUT_HTML}")
    print(f"Metadata: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
