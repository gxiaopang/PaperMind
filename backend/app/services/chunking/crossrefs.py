"""Version-local explicit reference edges and conservative acronym definitions."""

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from app.services.parsing.mineru_result import ParsedDocument

if TYPE_CHECKING:
    from app.services.chunking.chunker import ChunkDraft

NUMBER = r"(?:[A-Z](?:\.\d+)*|\d+(?:\.\d+)*[a-z]?)(?![A-Za-z0-9_])(?!\.\d)"
MENTION = re.compile(
    r"(?<!\w)(?P<kind>equations?|eqs?\.?|tables?|figures?|figs?\.?|appendix|appendices|sections?|secs?\.?)"
    r"\s*\(?\s*(?P<number>" + NUMBER + r")\s*\)?"
    r"|(?P<zh>公式|方程|表|图|附录|章节)\s*[（(]?\s*(?P<zn>" + NUMBER + r")",
    re.I,
)


def labels(text: str) -> list[str]:
    found = []
    for match in MENTION.finditer(text):
        kind = (match.group("kind") or match.group("zh")).casefold().rstrip(".")
        number = (match.group("number") or match.group("zn")).casefold()
        if kind.startswith(("eq", "公式", "方程")):
            family = "equation"
        elif kind.startswith(("tab", "表")):
            family = "table"
        elif kind.startswith(("fig", "图")):
            family = "figure"
        elif kind.startswith(("app", "附录")):
            family = "appendix"
        else:
            family = "section"
        found.append(f"{family}:{number}")
    return list(dict.fromkeys(found))


def acronyms(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    # Match the initials against a suffix, avoiding the preceding sentence or
    # phrases such as "we use" becoming part of an expansion.
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z\s-]{3,160})\s*[（(]([A-Z][A-Z0-9-]{1,14})[）)]", text
    ):
        words = match.group(1).strip().split()
        short = match.group(2)
        for start in range(len(words)):
            candidate = words[start:]
            initials = "".join(w[0] for word in candidate for w in word.split("-") if w)
            if initials.upper() == short.replace("-", ""):
                result.setdefault(short, " ".join(candidate))
                break
        if len(result) >= 128:
            break
    return result


def attach_retrieval_index(document: ParsedDocument, drafts: list["ChunkDraft"]) -> None:
    if not drafts:
        return
    by_block: dict[str, set[int]] = defaultdict(set)
    for ordinal, draft in enumerate(drafts):
        if draft.kind in {"document_meta", "figure"}:
            continue
        for span in draft.source_spans or []:
            by_block[span["block_id"]].add(ordinal)

    targets: dict[str, set[int]] = defaultdict(set)
    for block in document.blocks:
        ordinals = by_block.get(block.id, set())
        # Mentions in ordinary prose never define a table/equation target.
        if block.kind in {"caption", "table"}:
            match = MENTION.match(block.text.lstrip())
            for label in labels(match.group(0) if match else ""):
                if label.startswith(("table:", "figure:")):
                    targets[label].update(ordinals)
        if block.kind == "formula":
            numbers = re.findall(r"\\tag\s*\{([^{}]+)\}", block.text)
            numbers += re.findall(r"[（(](\d+(?:\.\d+)*[a-z]?)[）)]\s*\$*\s*$", block.text)
            for number in numbers:
                targets[f"equation:{number.strip().casefold()}"].update(ordinals)
        for part in block.section_path:
            clean = re.sub(r"^\s*#+\s*", "", part).strip()
            for label in labels(clean):
                if label.startswith(("appendix:", "section:")):
                    targets[label].update(ordinals)
            match = re.match(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)(?:[.)]?\s+|$)", clean)
            if match:
                number = match.group(1).casefold()
                family = "appendix" if number[0].isalpha() else "section"
                targets[f"{family}:{number}"].update(ordinals)

    # Papers commonly refer to an appendix subsection as "Section D.1".
    # Only alias actually defined alphabetic appendix paths; missing sections
    # and numbered main-text sections must keep their own identities.
    for label, members in list(targets.items()):
        if label.startswith("appendix:"):
            targets[label.replace("appendix:", "section:", 1)].update(members)

    dictionary: dict[str, str] = {}
    for block in document.blocks:
        if (
            block.indexable
            and block.kind == "text"
            and not any(
                re.search(r"references|bibliography|参考文献", part, re.I)
                for part in block.section_path
            )
        ):
            for short, full in acronyms(block.text).items():
                if len(dictionary) < 128:
                    dictionary.setdefault(short, full)
    # One copy per version; ordinal zero is present even without an L0 overview.
    drafts[0].chunk_metadata["document_acronyms"] = dictionary
    for ordinal, draft in enumerate(drafts):
        edges = []
        for label in labels(draft.text):
            resolved = sorted(targets.get(label, set()) - {ordinal})
            if ordinal in targets.get(label, set()) and not resolved:
                continue
            edges.append({"label": label, "target_ordinals": resolved})
        draft.chunk_metadata["cross_refs"] = edges
        draft.chunk_metadata["defined_labels"] = [
            label for label, members in targets.items() if ordinal in members
        ]
