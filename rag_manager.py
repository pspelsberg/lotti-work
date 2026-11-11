"""
Local LanceDB RAG Integration for Lotti Voice Agent.

Connects to a local LanceDB database (default: `C:\\integriti\\lancedb_data`),
embeds meeting transcript snippets or queries via Mistral Embeddings (with a
resilient deterministic local fallback), and retrieves relevant Architecture
Decision Records (ADRs) and engineering documentation.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

try:
    import lancedb
except ImportError:
    lancedb = None  # type: ignore[assignment]

try:
    from mistralai.client import Mistral
except ImportError:
    Mistral = Any  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from mistralai.client import Mistral as MistralClientType
else:
    MistralClientType = Any

logger = logging.getLogger("lotti.rag")


@dataclass(slots=True)
class KnowledgeDocument:
    """Represents an internal Architecture Decision Record (ADR) or document chunk."""

    doc_id: str
    title: str
    category: str
    status: str
    content: str
    tags: str
    distance: Optional[float] = None

    def format_for_working_memory(self) -> str:
        """Format the document concisely for LLM voice working memory."""
        score_str = (
            f" (distance={self.distance:.3f})" if self.distance is not None else ""
        )
        return (
            f"[{self.doc_id}] {self.title} | Status: {self.status} | "
            f"Category: {self.category}{score_str}\n"
            f"Summary: {self.content}"
        )


DEFAULT_SAMPLE_ADRS: List[Dict[str, str]] = [
    {
        "doc_id": "ADR-001",
        "title": "Primary Relational Database Standard (PostgreSQL 16)",
        "category": "Database",
        "status": "Accepted",
        "content": (
            "All transactional microservices must use managed PostgreSQL 16. "
            "Ad-hoc MySQL or MongoDB instances are prohibited in production without "
            "an architecture exemption. Read replicas must be used for reporting workloads."
        ),
        "tags": "database,postgres,postgresql,sql,storage,persistence,adr",
    },
    {
        "doc_id": "ADR-002",
        "title": "Local Vector Search & Embedded RAG with LanceDB",
        "category": "AI & Data",
        "status": "Accepted",
        "content": (
            "On-device and low-latency internal agents (such as Lotti) must use "
            "LanceDB stored on local NVMe storage (C:\\integriti\\lancedb_data on Windows) "
            "to guarantee sub-50ms retrieval latency and keep proprietary ADRs on-premise."
        ),
        "tags": "lancedb,vector,rag,embeddings,ai,latency,search,adr",
    },
    {
        "doc_id": "ADR-003",
        "title": "Service-to-Service Authentication & Zero Trust mTLS",
        "category": "Security",
        "status": "Accepted",
        "content": (
            "Internal APIs must authenticate using short-lived OAuth2 JWTs (max 15-minute TTL) "
            "combined with mTLS across the service mesh. Static API keys in environment "
            "variables are forbidden for inter-service production traffic."
        ),
        "tags": "security,authentication,oauth,jwt,mtls,api,zero-trust,compliance,adr",
    },
    {
        "doc_id": "ADR-004",
        "title": "Asynchronous Event Backbone (Apache Kafka vs Redis Streams)",
        "category": "Architecture",
        "status": "Accepted",
        "content": (
            "Domain events requiring durability and replayability must be published to "
            "Apache Kafka with Avro schemas. Redis is restricted to ephemeral caching "
            "and rate-limiting with a maximum TTL of 24 hours."
        ),
        "tags": "kafka,redis,events,microservice,messaging,cache,architecture,adr",
    },
    {
        "doc_id": "ADR-005",
        "title": "Real-Time Voice & SLA Latency Budget",
        "category": "Performance",
        "status": "Accepted",
        "content": (
            "Real-time voice and customer-facing endpoints must maintain a p95 latency "
            "under 300ms. Synchronous cross-region blocking calls are disallowed in the "
            "critical request path."
        ),
        "tags": "latency,sla,performance,voice,realtime,timeout,adr",
    },
]


class _LocalFallbackSearchQuery:
    """ LanceDB-compatible search query builder used only when `lancedb` is not installed."""

    def __init__(self, rows: List[Dict[str, Any]], query_vector: Sequence[float]) -> None:
        self._rows = rows
        self._query_vector = query_vector
        self._limit = 5

    def limit(self, k: int) -> "_LocalFallbackSearchQuery":
        self._limit = max(1, k)
        return self

    def to_list(self) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for row in self._rows:
            vec = row.get("vector", [])
            if len(vec) == len(self._query_vector):
                # L2 squared distance (matching LanceDB default L2 metric)
                dist = sum((a - b) ** 2 for a, b in zip(vec, self._query_vector))
            else:
                dist = 999.0
            item = dict(row)
            item["_distance"] = float(dist)
            scored.append(item)
        scored.sort(key=lambda r: r["_distance"])
        return scored[: self._limit]


class _LocalFallbackTable:
    """LanceDB-compatible table wrapper backed by a local JSON file when `lancedb` is absent."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rows: List[Dict[str, Any]] = []
        if self._path.exists():
            self._rows = json.loads(self._path.read_text(encoding="utf-8"))

    def save(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = list(rows)
        self._path.write_text(json.dumps(self._rows), encoding="utf-8")

    def search(self, query_vector: Sequence[float]) -> _LocalFallbackSearchQuery:
        return _LocalFallbackSearchQuery(self._rows, query_vector)


class _LocalFallbackDBConnection:
    """LanceDB-compatible connection fallback when `lancedb` wheel is not installed."""

    def __init__(self, uri: str) -> None:
        self._dir = Path(uri)
        self._dir.mkdir(parents=True, exist_ok=True)

    def table_names(self) -> List[str]:
        return [p.stem for p in self._dir.glob("*.lance.json")]

    def open_table(self, name: str) -> _LocalFallbackTable:
        return _LocalFallbackTable(self._dir / f"{name}.lance.json")

    def create_table(
        self, name: str, data: List[Dict[str, Any]], mode: str = "overwrite"
    ) -> _LocalFallbackTable:
        tbl = _LocalFallbackTable(self._dir / f"{name}.lance.json")
        tbl.save(data)
        return tbl


class RAGManager:
    """
    Manages the local LanceDB knowledge base and provides fast async retrieval
    for live meeting transcripts.
    """

    def __init__(
        self,
        db_uri: Path | str,
        table_name: str = "internal_knowledge",
        mistral_client: Optional[MistralClientType] = None,
        embedding_model: str = "mistral-embed",
        embedding_dim: int = 1024,
        trigger_keywords: Optional[Sequence[str]] = None,
    ) -> None:
        self._db_uri = Path(db_uri)
        self._table_name = table_name
        self._mistral_client = mistral_client
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._trigger_keywords = [
            kw.lower() for kw in (trigger_keywords or [])
        ]
        self._db: Optional[Any] = None
        self._table: Optional[Any] = None
        self._lock = asyncio.Lock()

    @property
    def db_uri(self) -> Path:
        """Return the local filesystem path to the LanceDB directory."""
        return self._db_uri

    def _connect_backend(self) -> Any:
        """Connect using native `lancedb.connect` or fallback if package is missing."""
        self._db_uri.mkdir(parents=True, exist_ok=True)
        if lancedb is not None:
            return lancedb.connect(str(self._db_uri))
        logger.warning(
            "Package `lancedb` is not installed; using local fallback vector store at '%s'.",
            self._db_uri,
        )
        return _LocalFallbackDBConnection(str(self._db_uri))

    def connect(self, seed_if_empty: bool = True) -> None:
        """
        Establish a connection to the local LanceDB instance at `C:\\integriti\\lancedb_data`
        (or configured path) and optionally seed initial ADR records if empty.
        """
        self._db = self._connect_backend()
        existing_tables = self._list_table_names()

        if self._table_name in existing_tables:
            self._table = self._db.open_table(self._table_name)
            logger.info(
                "Connected to existing LanceDB table '%s' at '%s'.",
                self._table_name,
                self._db_uri,
            )
        elif seed_if_empty:
            logger.info(
                "LanceDB table '%s' not found at '%s'. Seeding sample ADRs...",
                self._table_name,
                self._db_uri,
            )
            self.seed_documents(DEFAULT_SAMPLE_ADRS)
        else:
            logger.warning(
                "LanceDB table '%s' does not exist yet at '%s'.",
                self._table_name,
                self._db_uri,
            )

    def _list_table_names(self) -> List[str]:
        """Return table names across LanceDB versions (`table_names()` vs `list_tables()`)."""
        if self._db is None:
            return []
        if hasattr(self._db, "list_tables"):
            res = self._db.list_tables()
            if hasattr(res, "tables"):
                return list(res.tables)
            return list(res)
        return list(self._db.table_names())

    @staticmethod
    def compute_local_embedding(text: str, dim: int = 1024) -> List[float]:
        """
        Deterministic local feature-hash embedding (L2-normalized).

        Used when offline or when low-latency local embedding is preferred without
        blocking on a network roundtrip.
        """
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
            "from", "how", "in", "is", "it", "of", "on", "or", "our", "should",
            "that", "the", "this", "to", "use", "we", "what", "which", "with",
        }
        vec = [0.0] * dim
        raw_tokens = re.findall(r"[a-z0-9_]+", text.lower())
        tokens: List[str] = []
        for tok in raw_tokens:
            if tok in stopwords:
                continue
            # Normalize simple English plurals (e.g., apis -> api, databases -> database)
            if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
                tokens.append(tok[:-1])
            tokens.append(tok)

        if not tokens:
            tokens = raw_tokens or ["empty"]

        # Include unigrams and bigrams for richer semantic overlap
        features: List[str] = list(tokens)
        features.extend(
            f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)
        )

        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            hash_int = int.from_bytes(digest, byteorder="little", signed=False)
            idx = hash_int % dim
            sign = 1.0 if ((hash_int >> 32) & 1) == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    async def embed_text_async(
        self, text: str, use_remote_api: bool = False
    ) -> List[float]:
        """
        Generate an embedding vector for `text`.

        By default uses the fast local deterministic embedding so that existing local
        LanceDB tables have consistent vector spaces out-of-the-box, or calls the
        Mistral Embeddings API if `use_remote_api=True` and a client is configured.
        """
        if use_remote_api and self._mistral_client is not None:
            try:
                response = await self._mistral_client.embeddings.create_async(
                    model=self._embedding_model,
                    inputs=[text],
                )
                if response and response.data and response.data[0].embedding:
                    embedding = list(response.data[0].embedding)
                    if len(embedding) == self._embedding_dim:
                        return embedding
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Mistral remote embedding failed (%s); falling back to local embedding.",
                    exc,
                )

        return self.compute_local_embedding(text, dim=self._embedding_dim)

    def seed_documents(self, documents: Sequence[Dict[str, str]]) -> None:
        """Insert or overwrite ADR / documentation records in the local LanceDB table."""
        if self._db is None:
            self._db = self._connect_backend()

        records: List[Dict[str, Any]] = []
        for doc in documents:
            combined_text = (
                f"{doc['doc_id']} {doc['title']} {doc['category']} "
                f"{doc['content']} {doc.get('tags', '')}"
            )
            vector = self.compute_local_embedding(
                combined_text, dim=self._embedding_dim
            )
            records.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "category": doc.get("category", "General"),
                    "status": doc.get("status", "Accepted"),
                    "content": doc["content"],
                    "tags": doc.get("tags", ""),
                    "vector": vector,
                }
            )

        self._table = self._db.create_table(
            self._table_name, data=records, mode="overwrite"
        )
        logger.info(
            "Seeded %d ADR documents into LanceDB table '%s'.",
            len(records),
            self._table_name,
        )

    def should_trigger_rag(self, transcript_snippet: str) -> bool:
        """
        Determine whether a live meeting transcript snippet warrants a RAG lookup.

        Triggers when:
        - The snippet contains a question mark (`?`) or direct question phrasing, OR
        - The snippet mentions "lotti", OR
        - The snippet matches any configured technical/architectural keyword.
        """
        cleaned = transcript_snippet.strip().lower()
        if len(cleaned) < 4:
            return False

        if "?" in cleaned:
            return True

        question_starters = (
            "what is our",
            "what's our",
            "can we use",
            "are we allowed",
            "which database",
            "how do we handle",
            "do we have an adr",
            "hey lotti",
            "lotti,",
        )
        if any(phrase in cleaned for phrase in question_starters):
            return True

        words = set(re.findall(r"[a-z0-9_]+", cleaned))
        return any(kw in words or kw in cleaned for kw in self._trigger_keywords)

    def retrieve_sync(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> List[KnowledgeDocument]:
        """
        Synchronous vector retrieval against the local LanceDB table.
        """
        if not query or not query.strip():
            return []

        if self._table is None:
            self.connect(seed_if_empty=True)
        if self._table is None:
            return []

        try:
            query_vector = self.compute_local_embedding(
                query, dim=self._embedding_dim
            )
            raw_results: List[Dict[str, Any]] = (
                self._table.search(query_vector).limit(top_k).to_list()
            )

            documents: List[KnowledgeDocument] = []
            for row in raw_results:
                distance = row.get("_distance")
                if (
                    score_threshold is not None
                    and distance is not None
                    and float(distance) > score_threshold
                ):
                    continue
                documents.append(
                    KnowledgeDocument(
                        doc_id=str(row.get("doc_id", "UNKNOWN")),
                        title=str(row.get("title", "")),
                        category=str(row.get("category", "")),
                        status=str(row.get("status", "Accepted")),
                        content=str(row.get("content", "")),
                        tags=str(row.get("tags", "")),
                        distance=float(distance) if distance is not None else None,
                    )
                )
            return documents
        except Exception as exc:  # noqa: BLE001
            logger.error("LanceDB retrieval failed for query '%s': %s", query, exc)
            return []

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> List[KnowledgeDocument]:
        """
        Asynchronous wrapper for LanceDB vector retrieval so the real-time
        voice event loop is never blocked by disk I/O.
        """
        async with self._lock:
            return await asyncio.to_thread(
                self.retrieve_sync,
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
            )

    async def retrieve_formatted_context(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> str:
        """
        Retrieve relevant ADRs/docs from LanceDB and return a formatted string
        ready to be injected into the model's working memory or tool output.
        """
        docs = await self.retrieve_async(
            query=query, top_k=top_k, score_threshold=score_threshold
        )
        if not docs:
            return "No matching internal ADRs or architecture constraints found."

        formatted_blocks = [doc.format_for_working_memory() for doc in docs]
        return "\n\n".join(formatted_blocks)
