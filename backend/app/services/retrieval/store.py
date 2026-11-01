from typing import Any
from uuid import UUID

from fastapi import HTTPException
from langchain_core.documents import Document
from sqlalchemy import Text, case, cast, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from app.core.config import Settings
from app.db.models import Chunk, ChunkParent, Collection, Paper, PaperVersion, Section, User
from app.services.chunking.paper_map import section_std
from app.services.retrieval.expansion import (
    evidence_unit,
    fits_context,
    hit_counts,
    revoke_orphans,
    select_evidence,
)
from app.services.retrieval.planning import keywords, route_query
from app.services.retrieval.types import RetrievalScope


class RetrievalStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], settings: Settings):
        self.sessions, self.settings = sessions, settings

    async def paper_maps(self, scope: RetrievalScope) -> list[dict]:
        """Mount complete abstracts and navigation for active, authorized versions."""
        await self.validate_scope(scope)
        versions = (
            self.candidates(scope, searchable=False).with_only_columns(Chunk.version_id).distinct()
        )
        async with self.sessions() as db:
            rows = (
                await db.execute(
                    select(PaperVersion, Paper.title, Paper.abstract)
                    .join(Paper, Paper.id == PaperVersion.paper_id)
                    .where(PaperVersion.id.in_(versions))
                    .order_by(PaperVersion.paper_id)
                )
            ).all()
            result = []
            for version, title, abstract in rows:
                value = dict(version.paper_map or {})
                if not value:
                    # Old versions remain readable before an operator rechunks.
                    sections = list(
                        await db.scalars(
                            select(Section)
                            .where(
                                Section.user_id == scope.user_id, Section.version_id == version.id
                            )
                            .order_by(Section.path)
                        )
                    )
                    nodes: dict[str, dict] = {}
                    toc: list[dict] = []
                    for section in sorted(sections, key=lambda s: len(s.path)):
                        node = {
                            "section_id": str(section.id),
                            "title": section.title,
                            "section_std": section_std(section.path),
                            "path": section.path,
                            "children": [],
                        }
                        nodes[str(section.id)] = node
                        if len(section.path) <= 2:
                            parent = nodes.get(str(section.parent_id))
                            (parent["children"] if parent else toc).append(node)
                    chunks = (
                        await db.execute(
                            self.candidates(scope, searchable=False)
                            .where(
                                Chunk.version_id == version.id,
                                Chunk.kind.in_(
                                    ["caption", "table", "figure", "abstract", "document_meta"]
                                ),
                            )
                            .order_by(Chunk.ordinal)
                        )
                    ).all()
                    elements: dict[str, dict] = {}
                    abstract_parts: list[str] = []
                    for chunk, _ in chunks:
                        meta = chunk.chunk_metadata or {}
                        if chunk.kind == "abstract":
                            abstract_parts.append(chunk.text)
                        for key in meta.get("defined_labels", []):
                            family, number = key.split(":", 1)
                            if family in {"table", "figure"}:
                                caption = meta.get("table_caption") or chunk.text
                                elements.setdefault(
                                    key,
                                    {
                                        "key": key,
                                        "label": f"{family.title()} {number}",
                                        "caption": caption,
                                        "section_id": str(chunk.section_id)
                                        if chunk.section_id
                                        else None,
                                    },
                                )
                    value = {
                        "schema_version": "papermind.map.v1",
                        "title": title,
                        "abstract": abstract or "\n\n".join(abstract_parts),
                        "toc": toc,
                        "elements": list(elements.values()),
                        "legacy": True,
                    }
                value.update(
                    paper_id=str(version.paper_id), version_id=str(version.id), title=title
                )
                result.append(value)
        return result

    async def read_section(
        self,
        scope: RetrievalScope,
        standard: str,
        *,
        section_id: UUID | None = None,
        mode: str = "full",
    ) -> tuple[list[Document], list[dict]]:
        """Read SQL L1 parents directly; never perform vector search or cut a parent."""
        await self.validate_scope(scope)
        async with self.sessions() as db:
            sections = list(
                await db.scalars(
                    select(Section).where(
                        Section.user_id == scope.user_id,
                        Section.version_id.in_(
                            self.candidates(scope, searchable=False).with_only_columns(
                                Chunk.version_id
                            )
                        ),
                    )
                )
            )
            selected = [
                s
                for s in sections
                if (
                    s.id == section_id
                    if section_id
                    else s.section_std == standard
                    or section_std(s.path) == standard
                    or s.title.casefold() == standard.casefold()
                )
            ]
            # Include descendants when a concrete chapter is requested.
            if section_id:
                roots = selected[:]
                selected += [
                    s
                    for s in sections
                    if s not in selected
                    and any(
                        s.version_id == root.version_id and s.path[: len(root.path)] == root.path
                        for root in roots
                    )
                ]
            rows = (
                await db.execute(
                    self.candidates(scope, searchable=False)
                    .where(
                        Chunk.section_id.in_([s.id for s in selected]),
                        Chunk.kind.not_in(["document_meta", "figure"]),
                    )
                    .order_by(Chunk.paper_id, Chunk.ordinal)
                )
            ).all()
            documents = [self.document(c, title) for c, title in rows]
            parent_ids = {c.parent_id for c, _ in rows if c.parent_id}
            parents = list(
                await db.scalars(
                    select(ChunkParent)
                    .where(
                        ChunkParent.user_id == scope.user_id,
                        ChunkParent.id.in_(parent_ids),
                    )
                    .order_by(ChunkParent.version_id, ChunkParent.ordinal)
                )
            )
            data = []
            if mode == "summary":
                chosen: set[str] = set()
                for parent in parents:
                    children = [
                        d for d in documents if d.metadata.get("parent_id") == str(parent.id)
                    ]
                    for child in children[:1] + children[-1:]:
                        chosen.add(child.metadata["chunk_id"])
                documents = [d for d in documents if d.metadata["chunk_id"] in chosen]
            for parent in parents:
                children = [d for d in documents if d.metadata.get("parent_id") == str(parent.id)]
                data.append(
                    {
                        "parent_id": str(parent.id),
                        "section_id": str(parent.section_id),
                        "text": parent.text
                        if mode == "full"
                        else "\n\n".join(d.page_content for d in children),
                        "complete": mode == "full",
                        "summary_method": None if mode == "full" else "extractive",
                        "source_ids": [d.metadata["chunk_id"] for d in children],
                    }
                )
            for doc in documents:
                siblings = [
                    d.metadata["chunk_id"]
                    for d in documents
                    if d.metadata.get("parent_id") == doc.metadata.get("parent_id")
                ]
                doc.metadata.update(
                    retrieval_role="section",
                    parent_complete=mode == "full",
                    required_cross_ref_ids=siblings,
                    parent_expected_count=len(siblings) if mode == "full" else -1,
                )
        return await self.refresh(scope, documents), data

    async def inspect(self, scope: RetrievalScope, key: str) -> list[Document]:
        """Exact version-local definitions plus the prose that explicitly cites them."""
        await self.validate_scope(scope)
        # Quoted JSON string gives exact matching: table:2 never matches table:20.
        import json

        definition = cast(Chunk.chunk_metadata["defined_labels"], Text).contains(
            json.dumps(key), autoescape=True
        )
        async with self.sessions() as db:
            targets = (
                await db.execute(
                    self.candidates(scope, searchable=False)
                    .where(definition)
                    .order_by(Chunk.paper_id, Chunk.ordinal)
                )
            ).all()
            result = [self.document(c, title) for c, title in targets]
            for target, _ in targets:
                rows = (
                    await db.execute(
                        self.candidates(scope, searchable=False)
                        .where(
                            Chunk.version_id == target.version_id,
                            Chunk.kind.in_(["text", "formula"]),
                            cast(Chunk.chunk_metadata["cross_refs"], Text).contains(
                                json.dumps(key), autoescape=True
                            ),
                        )
                        .order_by(Chunk.ordinal)
                    )
                ).all()
                result.extend(self.document(c, title) for c, title in rows)
        unique = {d.metadata["chunk_id"]: d for d in result}
        for doc in unique.values():
            doc.metadata["retrieval_role"] = "element"
            doc.metadata["required_cross_ref_ids"] = [
                str(c.id) for c, _ in targets if str(c.version_id) == doc.metadata["version_id"]
            ]
        return await self.refresh(scope, list(unique.values()))

    async def validate_scope(self, scope: RetrievalScope) -> None:
        async with self.sessions() as db:
            user = await db.get(User, scope.user_id)
            if user is None or user.status != "active" or user.email_verified_at is None:
                raise HTTPException(403, "账户不可用或邮箱未验证")
            for ids, model in ((scope.collection_ids, Collection), (scope.paper_ids, Paper)):
                if ids is None or not ids:
                    continue
                stmt = select(model.id).where(model.user_id == scope.user_id, model.id.in_(ids))
                if model is Paper:
                    stmt = stmt.where(Paper.deleted_at.is_(None))
                found = set(await db.scalars(stmt))
                if found != set(ids):
                    raise HTTPException(404, "检索范围不存在")

    def candidates(self, scope: RetrievalScope, *, searchable: bool = True):  # type: ignore[no-untyped-def]
        stmt = (
            select(Chunk, Paper.title)
            .join(Paper, (Paper.id == Chunk.paper_id) & (Paper.user_id == Chunk.user_id))
            .join(PaperVersion, PaperVersion.id == Chunk.version_id)
            .join(User, User.id == Chunk.user_id)
            .where(
                User.status == "active",
                User.email_verified_at.is_not(None),
                Chunk.user_id == scope.user_id,
                Paper.user_id == scope.user_id,
                PaperVersion.user_id == scope.user_id,
                Paper.active_version_id == Chunk.version_id,
                Paper.deleted_at.is_(None),
                PaperVersion.status == "ready",
                PaperVersion.embedding_provider == self.settings.embedding_base_url.rstrip("/"),
                PaperVersion.embedding_model == self.settings.embedding_model,
                PaperVersion.embedding_dim == self.settings.embedding_dim,
            )
        )
        if searchable:
            stmt = stmt.where(Chunk.indexable.is_(True), Chunk.embedding.is_not(None))
        if scope.collection_ids is not None:
            stmt = stmt.where(Chunk.collection_id.in_(scope.collection_ids))
        if scope.paper_ids is not None:
            stmt = stmt.where(Chunk.paper_id.in_(scope.paper_ids))
        return stmt

    @staticmethod
    def document(chunk: Chunk, title: str) -> Document:
        return Document(
            id=str(chunk.id),
            page_content=chunk.text,
            metadata={
                "chunk_id": str(chunk.id),
                "user_id": str(chunk.user_id),
                "collection_id": str(chunk.collection_id),
                "paper_id": str(chunk.paper_id),
                "version_id": str(chunk.version_id),
                "ordinal": chunk.ordinal,
                "section_id": str(chunk.section_id) if chunk.section_id else None,
                "parent_id": str(chunk.parent_id) if chunk.parent_id else None,
                "source_spans": chunk.source_spans,
                "chunk_metadata": chunk.chunk_metadata or {},
                "title": title,
                "section_path": chunk.section_path,
                "kind": chunk.kind,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "locations": chunk.locations,
                "embedding_input": chunk.embedding_input,
            },
        )

    async def vector(
        self, scope: RetrievalScope, vector: list[float], limit: int, query: str = ""
    ) -> list[Document]:
        distance = Chunk.embedding.cosine_distance(vector)
        # Every tenant/version/scope predicate is applied before LIMIT. PostgreSQL
        # may use the HNSW index; deterministic tie-breaking also works with exact scans.
        score = (1 - distance) * self.metadata_weight(query)
        stmt = self.candidates(scope).order_by(score.desc(), Chunk.id).limit(limit)
        async with self.sessions() as db:
            return [self.document(c, title) for c, title in (await db.execute(stmt)).all()]

    async def bm25(self, scope: RetrievalScope, query: str, limit: int) -> list[Document]:
        # Match disjunction tokenizes plain input, without a query parser. User text
        # cannot insert field names, Boolean syntax or ownership predicates.
        stmt = (
            self.candidates(scope)
            .where(Chunk.embedding_input.op("|||")(query))
            .order_by(
                (func.paradedb.score(Chunk.id) * self.metadata_weight(query)).desc(), Chunk.id
            )
            .limit(limit)
        )
        async with self.sessions() as db:
            return [self.document(c, title) for c, title in (await db.execute(stmt)).all()]

    async def refresh(self, scope: RetrievalScope, documents: list[Document]) -> list[Document]:
        """Recheck after external I/O: deletion/version switches revoke stale results."""
        if not documents:
            return []
        ids = [UUID(d.metadata["chunk_id"]) for d in documents]
        async with self.sessions() as db:
            rows = (
                await db.execute(self.candidates(scope, searchable=False).where(Chunk.id.in_(ids)))
            ).all()
        available = {str(c.id): self.document(c, title) for c, title in rows}
        result = []
        for old in documents:
            current = available.get(old.metadata["chunk_id"])
            if current:
                for key in (
                    "rrf_score",
                    "ranks",
                    "rerank_score",
                    "hit_rank",
                    "retrieval_role",
                    "context_anchor",
                    "parent_complete",
                    "retrieval_intent",
                    "unresolved_cross_refs",
                    "expansion_incomplete",
                    "required_cross_ref_ids",
                    "merged_parent",
                    "parent_expected_count",
                ):
                    if key in old.metadata:
                        current.metadata[key] = old.metadata[key]
                result.append(current)
        return (
            revoke_orphans(result)
            if any("required_cross_ref_ids" in d.metadata for d in result)
            else result
        )

    def metadata_weight(self, query: str):  # type: ignore[no-untyped-def]
        plan = route_query(query)
        path = func.lower(cast(Chunk.section_path, Text))
        background = or_(
            *(
                path.contains(term)
                for term in (
                    "related work",
                    "prior work",
                    "background",
                    "acknowledg",
                    "相关工作",
                    "背景",
                    "致谢",
                )
            )
        )
        weight: ColumnElement = (
            case((background, self.settings.retrieval_background_weight), else_=1.0)
            if not plan.include_background
            else literal(1.0)
        )
        if plan.metric_focus:
            weight = weight * case(
                (Chunk.kind == "table", self.settings.retrieval_table_weight), else_=1.0
            )
            weight = weight * case(
                (
                    or_(
                        *(
                            path.contains(term)
                            for term in ("experiment", "evaluation", "results", "实验", "结果")
                        )
                    ),
                    self.settings.retrieval_experiment_weight,
                ),
                else_=1.0,
            )
        return weight

    async def metadata(self, scope: RetrievalScope, query: str, limit: int) -> list[Document]:
        terms = keywords(query)
        if not terms:
            return []
        path = func.lower(cast(Chunk.section_path, Text))
        stmt = (
            self.candidates(scope)
            .where(
                or_(
                    Chunk.kind == "table",
                    path.contains("experiment"),
                    path.contains("实验"),
                    path.contains("evaluation"),
                    path.contains("results"),
                    path.contains("结果"),
                ),
                or_(
                    *(
                        func.lower(Chunk.embedding_input).contains(term.casefold(), autoescape=True)
                        for term in terms
                    )
                ),
            )
            .order_by(self.metadata_weight(query).desc(), Chunk.ordinal, Chunk.id)
            .limit(limit)
        )
        async with self.sessions() as db:
            return [self.document(c, title) for c, title in (await db.execute(stmt)).all()]

    async def acronym_dictionary(self, scope: RetrievalScope) -> dict[str, list[str]]:
        stmt = (
            self.candidates(scope, searchable=False)
            .where(Chunk.ordinal == 0)
            .order_by(Chunk.version_id)
            .limit(self.settings.user_max_papers)
        )
        result: dict[str, list[str]] = {}
        async with self.sessions() as db:
            for chunk, _ in (await db.execute(stmt)).all():
                for short, full in (
                    (chunk.chunk_metadata or {}).get("document_acronyms", {}).items()
                ):
                    if full not in result.setdefault(short, []):
                        result[short].append(full)
        return result

    async def overview(self, scope: RetrievalScope) -> list[Document]:
        """L0 plus first/last child in each section; no similarity search."""
        partition = (Chunk.version_id, Chunk.section_id, cast(Chunk.section_path, Text))
        boundaries = (
            self.candidates(scope, searchable=False)
            .where(Chunk.kind.in_(["text", "abstract"]))
            .add_columns(
                func.row_number()
                .over(partition_by=partition, order_by=Chunk.ordinal)
                .label("first"),
                func.row_number()
                .over(partition_by=partition, order_by=Chunk.ordinal.desc())
                .label("last"),
            )
            .subquery()
        )
        boundary_ids = select(boundaries.c.id).where(
            or_(boundaries.c.first == 1, boundaries.c.last == 1)
        )
        stmt = (
            self.candidates(scope, searchable=False)
            .where(or_(Chunk.kind == "document_meta", Chunk.id.in_(boundary_ids)))
            .order_by(
                case((Chunk.kind == "document_meta", 0), else_=1), Chunk.ordinal, Chunk.version_id
            )
            .limit(self.settings.retrieval_max_sources * self.settings.retrieval_max_subqueries)
        )
        async with self.sessions() as db:
            return [self.document(c, title) for c, title in (await db.execute(stmt)).all()]

    async def _reference_unit(
        self,
        db: AsyncSession,
        source: Any,
        anchor: Document,
        members: list[Document],
        *,
        merged: bool,
        expected: dict[str, int],
        incomplete: list[str],
    ) -> Document | None:
        # Ordinals are version-local, and fetched targets are never traversed again.
        targets = {
            ordinal
            for doc in members
            for edge in doc.metadata.get("chunk_metadata", {}).get("cross_refs", [])
            for ordinal in edge.get("target_ordinals", [])
        }
        if len(targets) > self.settings.retrieval_cross_ref_max_chunks:
            return None
        references = []
        if targets:
            rows = (
                await db.execute(source.where(Chunk.ordinal.in_(targets)).order_by(Chunk.ordinal))
            ).all()
            references = [self.document(c, title) for c, title in rows]
            if {d.metadata["ordinal"] for d in references} != targets:
                return None
        return evidence_unit(
            anchor, members, references, merged=merged, expected=expected, incomplete=incomplete
        )

    async def expand_candidates(
        self, scope: RetrievalScope, hits: list[Document], *, overview: bool = False
    ) -> list[Document]:
        """Restore L1 for multiple L2 hits, otherwise one-child windows; one hop only."""
        hits = await self.refresh(scope, hits)
        counts = hit_counts(hits)
        hit_documents = {d.metadata["chunk_id"]: d for d in hits}
        grouped: set[str] = set()
        units: list[Document] = []
        async with self.sessions() as db:
            for anchor in hits:
                meta = anchor.metadata
                parent = meta.get("parent_id")
                if parent in grouped:
                    continue
                source = self.candidates(scope, searchable=False).where(
                    Chunk.version_id == UUID(meta["version_id"]),
                    Chunk.paper_id == UUID(meta["paper_id"]),
                )
                members = [anchor]
                expected: dict[str, int] = {}
                incomplete: list[str] = []
                merged = False
                if parent:
                    count = await db.scalar(
                        select(ChunkParent.child_count).where(
                            ChunkParent.user_id == scope.user_id,
                            ChunkParent.id == UUID(parent),
                            ChunkParent.version_id == UUID(meta["version_id"]),
                        )
                    )
                    if count:
                        expected[parent] = count
                    if not overview and counts[parent] >= self.settings.retrieval_merge_min_hits:
                        rows = (
                            await db.execute(
                                source.where(Chunk.parent_id == UUID(parent))
                                .order_by(Chunk.ordinal)
                                .limit(self.settings.retrieval_parent_max_chunks + 1)
                            )
                        ).all()
                        siblings = [self.document(c, title) for c, title in rows]
                        if (
                            len(siblings) == count
                            and len(siblings) <= self.settings.retrieval_parent_max_chunks
                            and sum(len(d.page_content.encode()) for d in siblings)
                            <= self.settings.retrieval_parent_bytes
                        ):
                            members, merged = siblings, True
                        else:
                            incomplete.append("parent_budget")
                    if not merged and not overview and self.settings.retrieval_neighbor_window:
                        window = self.settings.retrieval_neighbor_window
                        rows = (
                            await db.execute(
                                source.where(
                                    Chunk.section_path == meta["section_path"],
                                    Chunk.section_id == UUID(meta["section_id"])
                                    if meta.get("section_id")
                                    else Chunk.section_id.is_(None),
                                    Chunk.ordinal.between(
                                        meta["ordinal"] - window, meta["ordinal"] + window
                                    ),
                                    Chunk.kind != "document_meta",
                                ).order_by(Chunk.ordinal)
                            )
                        ).all()
                        members = [self.document(c, title) for c, title in rows]
                        # A deletion may have raced the initial refresh.
                        if not any(d.metadata["chunk_id"] == meta["chunk_id"] for d in members):
                            continue
                for doc in members:
                    seed = hit_documents.get(doc.metadata["chunk_id"])
                    if seed is not None:
                        doc.metadata["retrieval_role"] = "hit"
                        for key in ("rrf_score", "ranks"):
                            if key in seed.metadata:
                                doc.metadata[key] = seed.metadata[key]

                unit = await self._reference_unit(
                    db,
                    source,
                    anchor,
                    members,
                    merged=merged,
                    expected=expected,
                    incomplete=incomplete,
                )
                if unit is None:
                    continue
                if (
                    not fits_context(
                        unit,
                        budget=self.settings.retrieval_context_bytes,
                        max_sources=self.settings.retrieval_max_sources,
                    )
                    and len(members) > 1
                ):
                    # Optional neighbors/parent text can introduce many other
                    # references. Fall back to the whole seed and every target
                    # it actually cites; never trim a mandatory target to fit.
                    merged = False
                    unit = await self._reference_unit(
                        db,
                        source,
                        anchor,
                        [anchor],
                        merged=False,
                        expected=expected,
                        incomplete=[*incomplete, "context_budget"],
                    )
                if unit is not None and fits_context(
                    unit,
                    budget=self.settings.retrieval_context_bytes,
                    max_sources=self.settings.retrieval_max_sources,
                ):
                    units.append(unit)
                    if merged and parent:
                        grouped.add(parent)
        return units

    async def finalize_context(
        self, scope: RetrievalScope, units: list[Document]
    ) -> list[Document]:
        documents, expected = select_evidence(
            units,
            budget=self.settings.retrieval_context_bytes,
            max_sources=self.settings.retrieval_max_sources,
        )
        documents = await self.refresh(scope, documents)
        return revoke_orphans(documents, expected)

    async def expand_context(self, scope: RetrievalScope, hits: list[Document]) -> list[Document]:
        """Convenience path without reranking; production reranks the same units."""
        units = await self.expand_candidates(scope, hits)
        return await self.finalize_context(scope, units)

    async def neighbors(
        self, scope: RetrievalScope, documents: list[Document], *, budget: int
    ) -> list[Document]:
        """Optional adjacent blocks, within the identical scope and byte budget."""
        documents = await self.refresh(scope, documents)
        output, seen = [], {d.metadata["chunk_id"] for d in documents}
        async with self.sessions() as db:
            for doc in documents:
                stmt = (
                    self.candidates(scope)
                    .where(
                        Chunk.version_id == UUID(doc.metadata["version_id"]),
                        Chunk.section_path == doc.metadata["section_path"],
                        Chunk.section_id == UUID(doc.metadata["section_id"])
                        if doc.metadata.get("section_id")
                        else Chunk.section_id.is_(None),
                        Chunk.ordinal.in_(
                            [doc.metadata["ordinal"] - 1, doc.metadata["ordinal"] + 1]
                        ),
                    )
                    .order_by(Chunk.ordinal)
                )
                for chunk, title in (await db.execute(stmt)).all():
                    size = len(chunk.text.encode())
                    if str(chunk.id) not in seen and size <= budget:
                        seen.add(str(chunk.id))
                        budget -= size
                        output.append(self.document(chunk, title))
        return output
