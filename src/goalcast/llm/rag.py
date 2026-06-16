"""Tiny retrieval layer over the project knowledge base.

Default: dependency-free keyword retrieval (plenty for a small doc set). If chromadb is
installed it is used instead. Either way, retrieved snippets ground the AI analyst.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from goalcast.config import KB_DIR


@dataclass
class Chunk:
    source: str
    text: str


def _load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    if not KB_DIR.exists():
        return chunks
    for md in sorted(KB_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) > 30:
                chunks.append(Chunk(source=md.name, text=para))
    return chunks


def _keyword_retrieve(query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
    terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}
    scored = []
    for c in chunks:
        words = set(re.findall(r"\w+", c.text.lower()))
        scored.append((len(terms & words), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:k] if s > 0] or [c for _, c in scored[:k]]


class KnowledgeBase:
    def __init__(self) -> None:
        self.chunks = _load_chunks()
        self._collection = self._try_chroma()

    def _try_chroma(self):
        try:
            import chromadb

            client = chromadb.Client()
            col = client.get_or_create_collection("goalcast_kb")
            if col.count() == 0 and self.chunks:
                col.add(
                    ids=[f"c{i}" for i in range(len(self.chunks))],
                    documents=[c.text for c in self.chunks],
                    metadatas=[{"source": c.source} for c in self.chunks],
                )
            return col
        except Exception:  # noqa: BLE001 - chroma optional
            return None

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        if self._collection is not None:
            try:
                res = self._collection.query(query_texts=[query], n_results=k)
                return [
                    Chunk(source=m["source"], text=d)
                    for d, m in zip(res["documents"][0], res["metadatas"][0], strict=True)
                ]
            except Exception:  # noqa: BLE001
                pass
        return _keyword_retrieve(query, self.chunks, k)
