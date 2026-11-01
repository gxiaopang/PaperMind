"""Conservative academic structure helpers; extracted facts always retain raw evidence."""

import html
import math
import re
from html.parser import HTMLParser

from app.services.chunking.tokenizer import bounded
from app.services.parsing.mineru_result import ParsedBlock, ParsedDocument


def heading(value: str) -> str:
    value = re.sub(r"^\s*#+\s*", "", value).strip().casefold()
    return re.sub(r"^(?:\d+(?:\.\d+)*[.)]?|[ivxlcdm]+\.)\s+", "", value).strip()


def is_references(block: ParsedBlock) -> bool:
    return block.kind == "reference" or any(
        heading(p) in {"references", "bibliography", "参考文献", "参考资料"}
        for p in block.section_path
    )


def section_role(path: list[str]) -> str:
    # An enclosing review/ablation section determines ownership even when a
    # nested heading happens to say "Model" or "Results".
    for pattern, role in (
        (r"related work|prior work|相关工作", "他人工作综述"),
        (r"ablation|消融", "消融实验变体"),
    ):
        if any(re.search(pattern, heading(part)) for part in path):
            return role
    for part in reversed(path):
        value = heading(part)
        for pattern, role in (
            (r"related work|prior work|background|相关工作|背景", "他人工作综述"),
            (r"ablation|消融", "消融实验变体"),
            (r"conclu|总结|结论", "本文结论"),
            (r"abstract|摘要", "论文摘要"),
            (r"introduct|引言|绪论", "研究背景与本文贡献"),
            (r"experiment|result|evaluation|实验|结果|评估", "本文实验结果"),
            (r"method|approach|model|framework|方法|模型|框架", "本文方法"),
        ):
            if re.search(pattern, value):
                return role
    return "正文论述（归属需结合原文）"


def context_prefix(title: str, path: list[str], role: str, kind: str = "paragraph") -> str:
    # Escape structural delimiters supplied by document text. This is a label,
    # never an instruction or a guarantee that every claim is the authors' own.
    def clean(value: str, budget: int) -> str:
        return bounded(
            " ".join(value.split()).replace("[", "（").replace("]", "）").replace("|", "／"), budget
        )

    return f"[Paper: {clean(title, 96)} | Section: {clean(' > '.join(path) or '正文', 96)} | Type: {kind} | Role: {role}]\n"


def references(document: ParsedDocument) -> list[dict]:
    entries: list[dict] = []
    for block in document.blocks:
        if block.kind == "title" or not is_references(block) or not block.text.strip():
            continue
        matches = list(re.finditer(r"(?:^|\n)\s*(?:\[(\d+)\]|(\d+)\.)\s+", block.text))
        ref_parts: list[tuple[str | None, str]] = [
            (
                f"[{m.group(1) or m.group(2)}]",
                block.text[
                    m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(block.text)
                ].strip(),
            )
            for i, m in enumerate(matches)
        ] or [(None, block.text.strip())]
        for ref_id, raw in ref_parts:
            year = re.search(r"\b(?:19|20)\d{2}\b", raw)
            # Do not guess author/title boundaries for unrecognized styles.
            # Handle the common 'Authors. Title. Venue, year.' format only.
            fields = re.split(r"\.\s+", raw, maxsplit=2)
            authors: list[str] = []
            title = None
            if len(fields) >= 3 and not re.fullmatch(r"[A-Z]", fields[0].strip()):
                authors = [fields[0].strip()]
                title = fields[1].strip()
            entries.append(
                {
                    "ref_id": ref_id,
                    "title": title,
                    "authors": authors,
                    "year": int(year.group()) if year else None,
                    "raw_text": raw,
                    "block_id": block.id,
                    "locations": [loc.model_dump(mode="json") for loc in block.locations],
                }
            )
    return entries


def cited_references(value: str, entries: list[dict]) -> list[dict]:
    ids: set[str] = set()
    for match in re.finditer(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\]", value):
        for part in re.split(r"[,;]", match.group(1)):
            bounds = re.split(r"[–-]", part)
            first, last = int(bounds[0]), int(bounds[-1])
            if 0 <= last - first <= 100:
                ids.update(f"[{i}]" for i in range(first, last + 1))
    return [e for e in entries if e["ref_id"] in ids]


def plain_table(value: str) -> str:
    value = re.sub(r"</(?:td|th)\s*>", " | ", value, flags=re.I)
    value = re.sub(r"</tr\s*>", "\n", value, flags=re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def markdown_table(value: str) -> str:
    """Render every HTML row; keep the source HTML separately for merged cells."""
    if not re.search(r"<table\b", value, re.I):
        return value

    class TableParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.rows: list[list[str]] = []
            self.row: list[str] = []
            self.cell: list[str] | None = None
            self.notes: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag == "tr":
                self.row = []
            elif tag in {"td", "th"}:
                self.cell = []
                spans = [f"{k}={v}" for k, v in attrs if k in {"colspan", "rowspan"} and v != "1"]
                if spans:
                    self.cell.append("[" + ", ".join(spans) + "] ")
            elif tag == "br" and self.cell is not None:
                self.cell.append("<br>")

        def handle_data(self, data: str) -> None:
            if self.cell is not None:
                self.cell.append(data)
            elif data.strip():
                self.notes.append(data.strip())

        def handle_endtag(self, tag: str) -> None:
            if tag in {"td", "th"} and self.cell is not None:
                self.row.append(
                    "".join(self.cell).strip().replace("|", "\\|").replace("\n", "<br>")
                )
                self.cell = None
            elif tag == "tr" and self.row:
                self.rows.append(self.row)
                self.row = []

    parser = TableParser()
    parser.feed(value)
    if not parser.rows:
        return value
    width = max(len(row) for row in parser.rows)
    rows = ["| " + " | ".join(row + [""] * (width - len(row))) + " |" for row in parser.rows]
    rows.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(rows) + ("\n\n" + "\n".join(parser.notes) if parser.notes else "")


def table_summary(value: str, caption: str = "") -> str:
    """Offline fallback exposes headers and observed rows without inventing SOTA."""
    rows = [r.strip() for r in plain_table(value).splitlines() if r.strip()]
    header = rows[0] if rows else ""
    cells = [[cell.strip() for cell in row.strip("| ").split("|")] for row in rows]
    extrema = []
    if cells:
        for column in range(1, len(cells[0])):
            observations = []
            units = set()
            for row in cells[1:]:
                if len(row) <= column:
                    continue
                raw = row[column].strip(" *")
                match = re.fullmatch(r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(%?)", raw)
                if match:
                    try:
                        number = float(match.group(1).replace(",", ""))
                    except ValueError:
                        continue
                    if not math.isfinite(number):
                        continue
                    units.add(match.group(2))
                    observations.append((number, row[0], raw))
            if len(observations) >= 2 and len(units) == 1:
                low, high = min(observations), max(observations)
                extrema.append(
                    f"{cells[0][column]}：数值最小为 {low[2]}（{low[1]}），最大为 {high[2]}（{high[1]}）"
                )
    # Include both ends; final rows often contain the proposed method.
    examples = rows[1:4] + (rows[-3:] if len(rows) > 7 else rows[4:])
    summary = (
        f"表格：{bounded(caption, 100)}\n对比维度：{bounded(header, 120)}\n"
        f"可解析列的全表极值（非优劣判断）：{bounded('；'.join(extrema), 220)}\n"
        f"原始数据行摘录：{bounded(chr(10).join(examples), 450)}\n"
        "仅为检索用摘录；最终数值、最优方向及方法归属须核对完整表格。"
    )
    return bounded(summary, 720)
