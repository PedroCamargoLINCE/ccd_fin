"""
Converte notebooks/train_all.py -> notebooks/train_all.ipynb
preservando células # %% e # %% [markdown].
Uso: python notebooks/build_notebook.py
"""
from __future__ import annotations
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SRC = HERE / "train_all.py"
DST = HERE / "train_all.ipynb"


def parse_cells(text: str) -> list[tuple[str, str]]:
    """Retorna lista de (cell_type, source) a partir de marcadores # %%."""
    cells: list[tuple[str, str]] = []
    current_type = "code"
    buf: list[str] = []
    md_re = re.compile(r"^#\s*%%\s*\[markdown\]\s*$")
    code_re = re.compile(r"^#\s*%%\s*$")

    def flush():
        if not buf:
            return
        src = "\n".join(buf).rstrip() + "\n"
        if current_type == "markdown":
            # remove leading '# ' from each line
            md_lines = []
            for ln in src.splitlines():
                if ln.startswith("# "):
                    md_lines.append(ln[2:])
                elif ln == "#":
                    md_lines.append("")
                else:
                    md_lines.append(ln)
            src = "\n".join(md_lines).strip() + "\n"
        cells.append((current_type, src))

    for line in text.splitlines():
        if md_re.match(line):
            flush()
            buf = []
            current_type = "markdown"
        elif code_re.match(line):
            flush()
            buf = []
            current_type = "code"
        else:
            buf.append(line)
    flush()
    # drop leading empty cells (module docstring area)
    while cells and not cells[0][1].strip():
        cells.pop(0)
    return cells


def build():
    text = SRC.read_text(encoding="utf-8")
    cells_data = parse_cells(text)
    nb = nbf.v4.new_notebook()
    nb.cells = []
    for ctype, src in cells_data:
        if ctype == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {
        "display_name": "Python (ccd)",
        "language": "python",
        "name": "ccd",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, str(DST))
    print(f"wrote {DST}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
