"""
Parse CSV or JSON tabular data and render a self-contained HTML document (dark blue theme).

JSON: expands arrays of objects (including nested paths like data.Items) into rows.
If several such arrays exist, emits multiple tables. Otherwise falls back to a key–value table.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
import sys
from typing import Any, List, Optional, Tuple


def _configure_csv_large_fields() -> None:
    """Python's csv default per-field limit is 128KiB; raise it for large cells."""
    for limit in (sys.maxsize, 2**31 - 1, 2**30, 64 * 1024 * 1024):
        try:
            csv.field_size_limit(limit)
            return
        except (OverflowError, OSError, ValueError):
            continue
    # Should not reach: keep module import working
    csv.field_size_limit(min(131072 * 1024, 2**31 - 1))


_configure_csv_large_fields()

# Limits for uploads
MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
MAX_ROWS = 50_000

# (section_title, columns, rows)
TableSection = Tuple[str, List[str], List[List[Any]]]


def _cell_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def parse_csv_bytes(raw: bytes) -> Tuple[List[str], List[List[Any]]]:
    text = raw.decode("utf-8-sig")
    if not text.strip():
        raise ValueError("CSV file is empty")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect=dialect)
    rows_list = [list(row) for row in reader]
    if not rows_list:
        raise ValueError("CSV has no rows")
    header = [h.strip() if isinstance(h, str) else str(h) for h in rows_list[0]]
    if len(rows_list) == 1:
        return header, []
    data_rows: List[List[Any]] = []
    for row in rows_list[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        data_rows.append(row)
    return header, data_rows


def _try_parse_json_array_of_dicts_from_cell(cell: Any) -> Optional[List[dict]]:
    """If cell is a string that parses to a non-empty JSON array of objects, return it."""
    if cell is None:
        return None
    s = cell.strip() if isinstance(cell, str) else str(cell).strip()
    if len(s) < 2 or not s.startswith("["):
        return None
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or len(parsed) == 0:
        return None
    if not all(isinstance(x, dict) for x in parsed):
        return None
    return parsed


def _csv_row_best_json_array_column(
    row: List[Any],
    headers: List[str],
) -> Tuple[Optional[int], Optional[List[dict]]]:
    """Pick one column whose cell is a JSON array of objects (prefer Items/items/data, else longest)."""
    candidates: List[Tuple[int, List[dict], int]] = []
    for i, cell in enumerate(row):
        arr = _try_parse_json_array_of_dicts_from_cell(cell)
        if arr is not None:
            candidates.append((i, arr, len(arr)))
    if not candidates:
        return None, None
    name_pref = {"items", "item", "data", "rows", "records", "results", "values"}
    for i, arr, _ln in candidates:
        if i < len(headers):
            hn = headers[i].strip().lower()
            if hn in name_pref:
                return i, arr
    best = max(candidates, key=lambda x: x[2])
    return best[0], best[1]


def expand_csv_with_embedded_json_arrays(
    headers: List[str],
    rows: List[List[Any]],
) -> Tuple[List[str], List[List[Any]]]:
    """
    When a column holds a JSON string like [{...},{...}] (e.g. exported Items), expand to one row per
    object and append other CSV columns to each row (same as JSON 'explode' behavior).
    If no such cell exists, return headers/rows unchanged.
    """
    if not rows:
        return headers, rows

    w = len(headers)
    expanded: List[List[Any]] = []
    out_headers: Optional[List[str]] = None

    for row in rows:
        if len(row) < w:
            row = list(row) + [""] * (w - len(row))
        elif len(row) > w:
            row = row[:w]

        idx, arr = _csv_row_best_json_array_column(row, headers)
        if arr is None or idx is None:
            return headers, rows

        inner_cols, inner_rows = _json_list_of_dicts(arr)
        meta_names = [h for j, h in enumerate(headers) if j != idx]
        meta_vals = [row[j] for j in range(w) if j != idx]
        combined = inner_cols + meta_names

        if out_headers is None:
            out_headers = combined
        elif out_headers != combined:
            return headers, rows

        for ir in inner_rows:
            expanded.append(ir + meta_vals)

    return out_headers if out_headers is not None else headers, expanded


def _json_list_of_dicts(arr: List[dict]) -> Tuple[List[str], List[List[Any]]]:
    columns: List[str] = []
    seen: set = set()
    for row in arr:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)
    rows = [[row.get(c) for c in columns] for row in arr]
    return columns, rows


def _is_array_of_dicts(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v)


def _collect_arrays_of_dicts(obj: Any, path: str = "", depth: int = 0) -> List[Tuple[str, List[dict]]]:
    """DFS: every list of dicts with at least one row, with dotted path label."""
    out: List[Tuple[str, List[dict]]] = []
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if _is_array_of_dicts(v):
                out.append((p, v))
            elif isinstance(v, dict):
                out.extend(_collect_arrays_of_dicts(v, p, depth + 1))
    return out


def _sort_candidates(candidates: List[Tuple[str, List[dict]]]) -> List[Tuple[str, List[dict]]]:
    """Prefer longer arrays, then paths that look like Items / rows / data."""

    def score(item: Tuple[str, List[dict]]) -> Tuple[int, int, int, str]:
        path, arr = item
        pl = path.lower()
        pref = 0
        if "items" in pl or pl.endswith("rows") or pl.endswith("results"):
            pref = 2
        elif "data" in pl:
            pref = 1
        return (-len(arr), pref, -len(path), path)

    return sorted(candidates, key=score)


def _metadata_section(
    root: dict,
    consumed_top_level: set,
) -> Optional[TableSection]:
    rows_kv: List[List[Any]] = []
    for k, v in root.items():
        if k in consumed_top_level:
            continue
        rows_kv.append([k, _cell_str(v)])
    if not rows_kv:
        return None
    return ("Metadata", ["Key", "Value"], rows_kv)


def _consumed_top_level_from_path(path: str) -> set:
    if not path or path == "root":
        return set()
    return {path.split(".")[0]}


def parse_json_object_to_sections(root: dict) -> List[TableSection]:
    """
    Build one or more tables from a JSON object.
    - Finds arrays of objects (e.g. data.Items) and uses each object as a row.
    - If one primary array: optional Metadata table for other top-level keys.
    - If several arrays: one table per array (labeled by path).
    - If no array of objects: single key–value table.
    """
    candidates = _collect_arrays_of_dicts(root)
    candidates = [c for c in candidates if len(c[1]) > 0]

    if not candidates:
        pairs = [[k, _cell_str(v)] for k, v in root.items()]
        return [("Key — value", ["Key", "Value"], pairs)]

    candidates = _sort_candidates(candidates)

    # Multiple distinct arrays → one table each (no single-row blob)
    if len(candidates) > 1:
        sections: List[TableSection] = []
        for path, arr in candidates:
            cols, rows = _json_list_of_dicts(arr)
            label = path if path else "data"
            sections.append((label, cols, rows))
        return sections

    # Single array of objects: main table + optional metadata
    path, arr = candidates[0]
    cols, rows = _json_list_of_dicts(arr)
    consumed = _consumed_top_level_from_path(path)
    sections = []
    meta = _metadata_section(root, consumed)
    if meta:
        sections.append(meta)
    main_title = path if path else "Rows"
    sections.append((main_title, cols, rows))
    return sections


def _first_row_header_candidate(row: List[Any]) -> bool:
    if not row:
        return False
    return all(isinstance(x, str) and not str(x).strip().isdigit() for x in row)


def parse_json_bytes_to_sections(raw: bytes) -> List[TableSection]:
    text = raw.decode("utf-8-sig")
    if not text.strip():
        raise ValueError("JSON file is empty")
    obj = json.loads(text)

    if isinstance(obj, dict):
        return parse_json_object_to_sections(obj)

    if not isinstance(obj, list):
        raise ValueError("JSON root must be an array or object")

    if len(obj) == 0:
        raise ValueError("JSON array is empty")

    if isinstance(obj[0], dict):
        cols, rows = _json_list_of_dicts(obj)
        return [("Rows", cols, rows)]

    if isinstance(obj[0], list):
        if _first_row_header_candidate(obj[0]) and len(obj) > 1:
            headers = [str(x).strip() for x in obj[0]]
            data = obj[1:]
            w = len(headers)
            out: List[List[Any]] = []
            for r in data:
                if not isinstance(r, list):
                    raise ValueError("JSON rows must be arrays of the same shape")
                rr = list(r)
                if len(rr) < w:
                    rr.extend([None] * (w - len(rr)))
                elif len(rr) > w:
                    rr = rr[:w]
                out.append(rr)
            return [("Rows", headers, out)]
        w = max(len(r) for r in obj if isinstance(r, list))
        headers = [f"Column {i + 1}" for i in range(w)]
        out = []
        for r in obj:
            if not isinstance(r, list):
                raise ValueError("JSON rows must be arrays")
            rr = list(r)
            if len(rr) < w:
                rr.extend([None] * (w - len(rr)))
            elif len(rr) > w:
                rr = rr[:w]
            out.append(rr)
        return [("Rows", headers, out)]

    raise ValueError("Unsupported JSON structure: expected array of objects or array of arrays")


def parse_tabular_bytes(raw: bytes, filename: str) -> Tuple[List[TableSection], str]:
    """Returns (list of sections, format_label)."""
    fn = (filename or "").lower()
    if len(raw) > MAX_BYTES:
        raise ValueError(f"File too large (max {MAX_BYTES // (1024 * 1024)} MB)")

    if fn.endswith(".csv") or fn.endswith(".tsv"):
        cols, rows = parse_csv_bytes(raw)
        cols, rows = expand_csv_with_embedded_json_arrays(cols, rows)
        fmt = "csv"
        sections: List[TableSection] = [("Data", cols, rows)]
    elif fn.endswith(".json") or fn.endswith(".jsonl"):
        if fn.endswith(".jsonl"):
            lines = raw.decode("utf-8-sig").strip().splitlines()
            objs = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                objs.append(json.loads(line))
            if not objs:
                raise ValueError("JSONL file has no rows")
            if not isinstance(objs[0], dict):
                raise ValueError("JSONL lines must be JSON objects")
            cols, rows = _json_list_of_dicts(objs)
            sections = [("Rows", cols, rows)]
            fmt = "jsonl"
        else:
            sections = parse_json_bytes_to_sections(raw)
            fmt = "json"
    else:
        stripped = raw.lstrip()
        if stripped.startswith(b"{") or stripped.startswith(b"["):
            try:
                sections = parse_json_bytes_to_sections(raw)
                fmt = "json"
            except (json.JSONDecodeError, ValueError):
                cols, rows = parse_csv_bytes(raw)
                cols, rows = expand_csv_with_embedded_json_arrays(cols, rows)
                sections = [("Data", cols, rows)]
                fmt = "csv"
        else:
            cols, rows = parse_csv_bytes(raw)
            cols, rows = expand_csv_with_embedded_json_arrays(cols, rows)
            sections = [("Data", cols, rows)]
            fmt = "csv"

    total_rows = sum(len(s[2]) for s in sections)
    if total_rows > MAX_ROWS:
        raise ValueError(f"Too many rows (max {MAX_ROWS})")

    return sections, fmt


def _render_one_table(columns: List[str], rows: List[List[Any]]) -> str:
    esc = html.escape
    thead = "".join(f"<th>{esc(_cell_str(c))}</th>" for c in columns)
    body_rows: List[str] = []
    for i, row in enumerate(rows):
        cls = "row-alt" if i % 2 else ""
        cells = "".join(f"<td>{esc(_cell_str(v))}</td>" for v in row)
        body_rows.append(f'<tr class="{cls}">{cells}</tr>')
    tbody = "\n".join(body_rows)
    return f"""    <div class="table-scroll">
      <table>
        <thead><tr>{thead}</tr></thead>
        <tbody>
{tbody}
        </tbody>
      </table>
    </div>"""


def build_dark_blue_multi_section_html(
    sections: List[TableSection],
    *,
    title: str = "Data",
    subtitle: Optional[str] = None,
) -> str:
    """Full HTML document with one or more tables (dark blue theme)."""
    esc = html.escape
    sub = f'<p class="subtitle">{esc(subtitle)}</p>' if subtitle else ""

    blocks: List[str] = []
    total_rows = sum(len(s[2]) for s in sections)
    max_cols = max((len(s[1]) for s in sections), default=0)

    for sec_title, cols, rows in sections:
        blocks.append(
            f'    <h2 class="section-title">{esc(sec_title)}</h2>\n'
            f'    <div class="meta">{len(rows)} row(s) · {len(cols)} column(s)</div>\n'
            f"{_render_one_table(cols, rows)}"
        )

    body_inner = "\n".join(blocks)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg-deep: #0a1628;
      --bg-panel: #0f2137;
      --bg-header: #153a5c;
      --border: #2a5580;
      --text: #e6eef8;
      --text-muted: #8ba3c4;
      --accent: #4a9eff;
      --accent-soft: rgba(74, 158, 255, 0.12);
      --row-alt: rgba(30, 58, 95, 0.45);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: linear-gradient(165deg, var(--bg-deep) 0%, #0d1f35 40%, #0a1628 100%);
      color: var(--text);
      padding: 1.75rem 1.25rem 2.5rem;
    }}
    .wrap {{
      max-width: min(1200px, 100%);
      margin: 0 auto;
    }}
    h1 {{
      font-size: 1.35rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      margin: 0 0 0.35rem 0;
      color: var(--text);
    }}
    h2.section-title {{
      font-size: 1.05rem;
      font-weight: 600;
      color: var(--accent);
      margin: 1.5rem 0 0.35rem 0;
    }}
    h2.section-title:first-of-type {{
      margin-top: 0;
    }}
    .subtitle {{
      margin: 0 0 1.25rem 0;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .meta {{
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-bottom: 0.65rem;
    }}
    .table-scroll {{
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--bg-panel);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.04);
      overflow: auto;
      max-height: min(55vh, 560px);
      margin-bottom: 0.5rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      text-align: left;
      padding: 0.65rem 0.9rem;
      background: linear-gradient(180deg, var(--bg-header) 0%, #0f2842 100%);
      color: var(--text);
      font-weight: 600;
      border-bottom: 2px solid var(--accent);
      white-space: nowrap;
    }}
    tbody td {{
      padding: 0.55rem 0.9rem;
      border-bottom: 1px solid rgba(42, 85, 128, 0.45);
      vertical-align: top;
      word-break: break-word;
    }}
    tbody tr.row-alt td {{
      background: var(--row-alt);
    }}
    tbody tr:hover td {{
      background: var(--accent-soft);
    }}
    tbody tr:last-child td {{
      border-bottom: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{esc(title)}</h1>
    {sub}
    <div class="meta" style="margin-bottom:1rem;">Total: {total_rows} row(s) across {len(sections)} table(s) · up to {max_cols} column(s)</div>
{body_inner}
  </div>
</body>
</html>
"""


def tabular_bytes_to_html(raw: bytes, filename: str) -> Tuple[str, List[str], List[List[Any]], str]:
    """
    Returns (html_document, primary_columns, primary_rows, format_label).
    Primary = first non-Metadata section when present, else first section.
    """
    sections, fmt = parse_tabular_bytes(raw, filename)
    title = "Table preview"
    if filename:
        title = re.sub(r"[^\w.\-]+", " ", filename).strip() or title

    data_sections = [s for s in sections if s[0] != "Metadata"]
    if data_sections:
        _, primary_cols, primary_rows = data_sections[0]
    else:
        _, primary_cols, primary_rows = sections[0]

    html_doc = build_dark_blue_multi_section_html(
        sections,
        title=title,
        subtitle=f"Format: {fmt.upper()}",
    )
    return html_doc, primary_cols, primary_rows, fmt
