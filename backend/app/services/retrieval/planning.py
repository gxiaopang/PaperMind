"""Bounded local routing; no extra model call or model-controlled scope."""

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QueryPlan:
    intent: Literal["overview", "detail", "comparison", "local"]
    queries: tuple[str, ...]
    include_background: bool = False
    metric_focus: bool = False


def route_query(query: str, *, max_queries: int = 3) -> QueryPlan:
    background = bool(
        re.search(r"related work|prior work|background|acknowledg|相关工作|背景|致谢", query, re.I)
    )
    complex_query = bool(
        re.search(
            r"对比|比较|跨章节|跨论文|痛点|compare|comparison|versus|\bvs\.?\b|how .*address|how .*solve",
            query,
            re.I,
        )
    )
    detail = bool(
        re.search(
            r"多少|指标|准确率|消融|超参|得分|实验结果|pass@|accuracy|performance|\bf1\b|\bbleu\b|\bgsm8k\b|\bscores?\b|表\s*\d|table\s*\d|λ|\\lambda",
            query,
            re.I,
        )
    )
    # Numeric model dimensions, optimizer settings and resource budgets often
    # lack an explicit metric name, particularly in English questions.
    detail = detail or bool(
        re.search(
            r"\bhidden dimensionality\b|\b(?:optimizer|beta values?|hyperparameters?)\b"
            r"|\bhow (?:much|many|small|large|long)\b"
            r"|(?:训练|training).{0,30}(?:GPU|多久|时长|duration|hardware)"
            r"|(?:GPU|显存|checkpoint).{0,30}(?:多大|大小|占用|memory|size)"
            r"|(?:参数预算|parameter budget).{0,50}(?:最好|最佳|best)",
            query,
            re.I,
        )
    )
    overview = bool(
        re.search(
            r"概览|概述|总结|核心创新|创新点|核心思想|优缺点|局限|主要贡献|overview|summari[sz]e|main contributions|main idea|limitations|key innovations",
            query,
            re.I,
        )
    )
    intent: Literal["overview", "detail", "comparison", "local"] = (
        "comparison"
        if complex_query
        else "detail"
        if detail
        else "overview"
        if overview
        else "local"
    )
    queries = [query]
    critique = bool(
        re.search(
            r"隐性|缺点|计算开销|失败案例|失效|failure|hidden weakness|computational cost|compute overhead",
            query,
            re.I,
        )
    )
    if critique:
        queries += [
            query + " limitations assumptions failure cases negative results",
            query + " computational cost memory latency scalability overhead",
        ]
    if intent == "comparison" and not critique:
        chinese = bool(re.search(r"[\u3400-\u9fff]", query))
        if re.search(r"痛点|引言|address|solve|introduction", query, re.I):
            queries += [
                "引言中提出的问题、挑战和痛点"
                if chinese
                else "Introduction problem challenges limitations",
                query
                + (" 本文方法 解决机制" if chinese else " proposed method solution mechanism"),
            ]
        else:
            queries += [
                query + (" 方法与假设" if chinese else " methods assumptions"),
                query + (" 实验结果与局限" if chinese else " experimental results limitations"),
            ]
        clauses = [p.strip() for p in re.split(r"[；;？?]|以及|并且", query) if p.strip()]
        if len(clauses) > 1:
            queries[1:] = clauses
    return QueryPlan(intent, tuple(dict.fromkeys(queries))[:max_queries], background, detail)


def fusion_policy(plan: QueryPlan, vector_weight: float, sparse_weight: float, k: int) -> dict:
    """Query-adaptive RRF alpha/k, preserving the configured channel preference."""
    dense_factor, sparse_factor = (0.65, 1.0) if plan.metric_focus else (1.0, 0.75)
    dense, sparse = vector_weight * dense_factor, sparse_weight * sparse_factor
    alpha = dense / (dense + sparse)
    return {
        "alpha": alpha,
        "k": max(1, round(k * (0.5 if plan.metric_focus else 1.5 if len(plan.queries) > 1 else 1))),
        "vector": alpha,
        "bm25": 1 - alpha,
    }


def expand_acronyms(query: str, dictionary: dict[str, list[str]], *, max_bytes: int) -> str:
    additions: list[str] = []
    for short, expansions in dictionary.items():
        if re.search(r"(?<![A-Za-z0-9_-])" + re.escape(short) + r"(?![A-Za-z0-9_-])", query, re.I):
            for full in expansions:
                if full.casefold() not in query.casefold() and full not in additions:
                    candidate = query + " " + " ".join([*additions, full])
                    if len(candidate.encode()) <= max_bytes:
                        additions.append(full)
    return query + (" " + " ".join(additions) if additions else "")


def keywords(query: str) -> list[str]:
    stop = {
        "what",
        "how",
        "the",
        "this",
        "that",
        "with",
        "and",
        "does",
        "paper",
        "table",
        "dataset",
    }
    return list(
        dict.fromkeys(
            token
            for token in re.findall(r"[A-Za-zλΛ][A-Za-z0-9_λΛ@.{}\\-]*|[\u3400-\u9fff]{2,}", query)
            if len(token) > 1 and token.casefold() not in stop
        )
    )[:16]
