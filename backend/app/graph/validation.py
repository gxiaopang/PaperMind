import re

CITATION = re.compile(r"\[(\d+)\]")
SAFE_ANSWER = "现有证据不足以可靠回答。请补充相关论文或换一种问法。"


def citation_issues(answer: str, count: int) -> dict:
    numbers = sorted({int(n) for n in CITATION.findall(answer)})
    invalid = [n for n in numbers if n < 1 or n > count]
    uncited = []
    # A citation immediately following punctuation belongs to the preceding sentence.
    normalized = re.sub(r"([。！？.!?])(\s*(?:\[\d+\]\s*)+)", r"\2\1", answer)
    # Decimal points, common scientific abbreviations and inline math do not end
    # sentences. Retain the surrounding prose when checking citation coverage.
    normalized = re.sub(r"(?<=\d)\.(?=\d)", "․", normalized)
    normalized = re.sub(
        r"\b(?:e\.g\.|i\.e\.|et al\.|Fig\.|Eq\.|Sec\.|vs\.)",
        lambda m: m[0].replace(".", "․"),
        normalized,
    )
    normalized = re.sub(r"\\\(.*?\\\)|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", "MATH", normalized)
    lines = normalized.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#") or re.fullmatch(r"[| :\-]+", line):
            continue
        if i + 1 < len(lines) and re.fullmatch(r"[| :\-]+", lines[i + 1].strip()):
            continue
        line = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", line)
        for unit in re.split(r"(?<=[。！？.!?])\s*", line):
            body = CITATION.sub("", unit)
            if re.search(r"[\w\u4e00-\u9fff]", body) and not CITATION.search(unit):
                uncited.append(unit[:160])
    return {
        "invalid_numbers": invalid,
        "uncited_units": uncited[:8],
        "missing_citations": not numbers,
        "empty_answer": not answer.strip(),
    }


def validate_citations(answer: str, count: int) -> tuple[bool, list[int]]:
    issues = citation_issues(answer, count)
    return not any(issues.values()), sorted({int(n) for n in CITATION.findall(answer)})
