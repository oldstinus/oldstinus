#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WL Agent (OpenAI API) — Directory + Images + Patch + ALLOW/KEEP gate

Features:
- Scans directory tree with ignore patterns
- Reads selected files (or auto-select based on your instruction)
- Optionally loads images and sends them to the model (base64)
- Produces a minimal patch proposal (KEEP)
- Shows unified diff
- Applies changes only after explicit ALLOW

Usage examples:
  python wl_agent.py --root . --instruction "Fix bug in parsing timestamps in file X, keep changes minimal" --targets src/main.py
  python wl_agent.py --root . --instruction "Interpret screenshot and adjust plotting code accordingly" --targets plot.py --images screenshot.png
"""

import os
import sys
import json
import base64
import fnmatch
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from openai import OpenAI

console = Console()


# ----------------------------
# Config / defaults
# ----------------------------
DEFAULT_IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", "dist", "build", ".idea", ".vscode"
}
DEFAULT_IGNORE_GLOBS = {
    "*.pyc", "*.pyo", "*.pyd", "*.exe", "*.dll", "*.so", "*.dylib",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.mp4", "*.mov",
    "*.zip", "*.7z", "*.tar", "*.gz", "*.rar",
    "*.pdf", "*.docx", "*.pptx", "*.xlsx",
}
TEXT_FILE_MAX_BYTES = 1_500_000  # avoid sending huge files


@dataclass
class FileBlob:
    path: str
    content: str


def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY ontbreekt. Zet env var OPENAI_API_KEY.")
    return OpenAI(api_key=key)


def is_ignored(path: Path, root: Path,
               ignore_dirs: set,
               ignore_globs: set) -> bool:
    rel = path.relative_to(root).as_posix()

    parts = rel.split("/")
    for p in parts:
        if p in ignore_dirs:
            return True

    for g in ignore_globs:
        if fnmatch.fnmatch(path.name, g) or fnmatch.fnmatch(rel, g):
            return True

    return False


def scan_tree(root: Path,
              ignore_dirs: set = DEFAULT_IGNORE_DIRS,
              ignore_globs: set = DEFAULT_IGNORE_GLOBS,
              max_files: int = 2000) -> List[str]:
    files = []
    for p in root.rglob("*"):
        if len(files) >= max_files:
            break
        if p.is_file() and not is_ignored(p, root, ignore_dirs, ignore_globs):
            files.append(p.relative_to(root).as_posix())
    return sorted(files)


def read_text_file(path: Path) -> str:
    # best effort: utf-8 then fallback
    b = path.read_bytes()
    if len(b) > TEXT_FILE_MAX_BYTES:
        return f"<<FILE TOO LARGE ({len(b)} bytes) - omitted>>"
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="replace")


def load_files(root: Path, rel_paths: List[str]) -> List[FileBlob]:
    blobs: List[FileBlob] = []
    for rp in rel_paths:
        p = root / rp
        if not p.exists() or not p.is_file():
            continue
        blobs.append(FileBlob(path=rp, content=read_text_file(p)))
    return blobs


def guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def load_images_as_base64(root: Path, image_paths: List[str]) -> List[Dict]:
    """
    Returns list of dicts containing:
      { "path": "...", "mime": "image/png", "base64": "..." }
    """
    out = []
    for ip in image_paths:
        p = (root / ip).resolve() if not Path(ip).is_absolute() else Path(ip).resolve()
        if not p.exists():
            console.print(f"[yellow]Image not found:[/] {ip}")
            continue
        mime = guess_mime(p)
        if not mime.startswith("image/"):
            console.print(f"[yellow]Not an image file, skipped:[/] {ip} ({mime})")
            continue
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        out.append({"path": str(p), "mime": mime, "base64": b64})
    return out


def unified_diff(a: str, b: str, fromfile: str, tofile: str) -> str:
    import difflib
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    diff = difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile)
    return "".join(diff)


# ----------------------------
# LLM call (Responses API)
# ----------------------------
def llm_propose_patch(client: OpenAI,
                      model: str,
                      system_prompt: str,
                      instruction: str,
                      tree: List[str],
                      files: List[FileBlob],
                      images: List[Dict],
                      keep_minimal: bool = True) -> Dict:
    """
    Ask model to return a JSON object with:
      {
        "summary": "...",
        "rationale": "...",
        "changes": [
          {"path": "rel/path.py", "new_content": "..." }
        ]
      }
    """
    keep_line = "KEEP: make the smallest possible change set. No refactors unless required." if keep_minimal else ""
    tree_text = "\n".join(tree[:1500])  # prevent insane token use

    file_payload = []
    for fb in files:
        file_payload.append({
            "path": fb.path,
            "content": fb.content
        })

    # Build input as a single structured message
    # (We keep it simple: JSON-in / JSON-out contract)
    user_content = {
        "instruction": instruction,
        "keep": keep_line,
        "project_tree": tree_text,
        "files": file_payload,
        "images": [{"path": im["path"], "mime": im["mime"], "base64": im["base64"]} for im in images]
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content)}
        ]
    )

    text = resp.output_text.strip()

    # Robust JSON extraction: allow leading/trailing commentary
    # Try direct JSON parse first, then find first {...} block.
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception as e:
                raise RuntimeError(f"Model output is not valid JSON.\n\nRaw:\n{text[:4000]}") from e
        raise RuntimeError(f"Model output is not valid JSON.\n\nRaw:\n{text[:4000]}")


# ----------------------------
# Apply patches with ALLOW gate
# ----------------------------
def apply_changes(root: Path, changes: List[Dict], dry_run: bool = False) -> Tuple[int, List[str]]:
    applied = 0
    written_paths = []
    for ch in changes:
        rel = ch.get("path")
        new_content = ch.get("new_content")
        if not rel or new_content is None:
            continue

        p = root / rel
        if not p.exists():
            console.print(f"[yellow]Target file does not exist, will be created:[/] {rel}")

        old_content = read_text_file(p) if p.exists() else ""
        diff = unified_diff(old_content, new_content, fromfile=f"a/{rel}", tofile=f"b/{rel}")

        console.rule(f"[bold]DIFF: {rel}[/bold]")
        if diff.strip():
            console.print(Syntax(diff, "diff", word_wrap=False))
        else:
            console.print("[green]No changes (diff empty).[/green]")

        if dry_run:
            continue

        if Confirm.ask(f"ALLOW: apply changes to [bold]{rel}[/bold]?"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new_content, encoding="utf-8")
            applied += 1
            written_paths.append(rel)
        else:
            console.print(f"[yellow]Skipped:[/] {rel}")

    return applied, written_paths


# ----------------------------
# Main CLI
# ----------------------------
def parse_args(argv: List[str]) -> Dict:
    # Minimal manual parser (no extra deps)
    args = {
        "root": ".",
        "instruction": None,
        "targets": [],
        "images": [],
        "model": "gpt-4.1-mini",
        "dry_run": False,
        "no_keep": False
    }
    it = iter(argv)
    for a in it:
        if a in ("--root",):
            args["root"] = next(it)
        elif a in ("--instruction", "-i"):
            args["instruction"] = next(it)
        elif a in ("--targets", "-t"):
            # comma-separated or repeated
            val = next(it)
            args["targets"] += [x.strip() for x in val.split(",") if x.strip()]
        elif a in ("--images",):
            val = next(it)
            args["images"] += [x.strip() for x in val.split(",") if x.strip()]
        elif a in ("--model",):
            args["model"] = next(it)
        elif a in ("--dry-run",):
            args["dry_run"] = True
        elif a in ("--no-keep",):
            args["no_keep"] = True
        else:
            # allow positional targets
            if a.strip():
                args["targets"].append(a.strip())
    return args


def main():
    args = parse_args(sys.argv[1:])
    root = Path(args["root"]).resolve()

    if args["instruction"] is None:
        console.print(Panel.fit("Geef een instructie (wat moet er aangepast worden).", title="WL Agent"))
        args["instruction"] = Prompt.ask("Instruction")

    # Scan
    console.print(f"[bold]Root:[/] {root}")
    tree = scan_tree(root)
    console.print(f"[bold]Files in tree:[/] {len(tree)}")

    # Determine targets
    targets = args["targets"]
    if not targets:
        console.print(Panel(
            "Je hebt geen --targets opgegeven.\n"
            "Ik kan enkel patchen op files die jij opgeeft.\n\n"
            "Geef nu 1 of meer relatieve paden (comma-separated).",
            title="Targets nodig"
        ))
        t = Prompt.ask("Targets")
        targets = [x.strip() for x in t.split(",") if x.strip()]

    # Normalize targets relative to root when possible
    rel_targets = []
    for t in targets:
        tp = Path(t)
        if tp.is_absolute():
            try:
                rel_targets.append(tp.relative_to(root).as_posix())
            except Exception:
                # if outside root: store as absolute; but patch apply will use root/rel, so warn
                console.print(f"[yellow]Target outside root; best is to use relative path:[/] {t}")
                rel_targets.append(t)
        else:
            rel_targets.append(tp.as_posix())

    # Load selected files
    files = load_files(root, rel_targets)
    if not files:
        console.print("[red]Geen target files ingelezen. Check je paden.[/red]")
        sys.exit(2)

    # Load images (optional)
    images = load_images_as_base64(root, args["images"]) if args["images"] else []

    # ALLOW gate: generate proposal
    console.print(Panel(
        "Ik ga nu een patchvoorstel genereren via jouw OpenAI API.\n"
        "Daarna toon ik diffs en vraag ik per file ALLOW om toe te passen.",
        title="Plan"
    ))
    if not Confirm.ask("ALLOW: patchvoorstel genereren?"):
        console.print("[yellow]Afgebroken.[/yellow]")
        sys.exit(0)

    # System prompt (contains your required line)
    system_prompt = (
        "You are a local code agent used inside Visual Studio Code.\n"
        "Describe what this custom agent does and when to use it.\n\n"
        "Hard rules:\n"
        "1) Never modify files without explicit ALLOW.\n"
        "2) Always propose a patch first.\n"
        "3) KEEP changes minimal.\n"
        "4) If images are provided, interpret them carefully and technically.\n\n"
        "Output must be VALID JSON with keys: summary, rationale, changes.\n"
        "changes is a list of objects {path, new_content}.\n"
        "Do not include Markdown fences."
    )

    client = get_client()
    proposal = llm_propose_patch(
        client=client,
        model=args["model"],
        system_prompt=system_prompt,
        instruction=args["instruction"],
        tree=tree,
        files=files,
        images=images,
        keep_minimal=(not args["no_keep"])
    )

    # Show proposal
    summary = proposal.get("summary", "")
    rationale = proposal.get("rationale", "")
    changes = proposal.get("changes", [])

    console.rule("[bold]MODEL OUTPUT[/bold]")
    console.print(Panel(Text(summary or "(no summary)"), title="Summary"))
    console.print(Panel(Text(rationale or "(no rationale)"), title="Rationale"))
    console.print(f"[bold]Files to change:[/] {len(changes)}")

    if not changes:
        console.print("[yellow]Geen wijzigingen voorgesteld. Stop.[/yellow]")
        sys.exit(0)

    # ALLOW gate: apply phase (global)
    if args["dry_run"]:
        console.print("[cyan]Dry-run: geen files worden geschreven.[/cyan]")
    else:
        if not Confirm.ask("ALLOW: ga ik nu diffs tonen en per file vragen om toe te passen?"):
            console.print("[yellow]Afgebroken.[/yellow]")
            sys.exit(0)

    applied, written = apply_changes(root, changes, dry_run=args["dry_run"])

    console.rule("[bold]RESULT[/bold]")
    console.print(f"[bold]Applied:[/] {applied}")
    if written:
        console.print("[green]Written files:[/green]")
        for w in written:
            console.print(f"  - {w}")


if __name__ == "__main__":
    main()
