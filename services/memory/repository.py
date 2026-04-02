from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import threading
from typing import Dict, Iterable, List, Optional

from contracts.memory import SemanticMemoryEntry, TaggedLocation
from contracts.spatial_memory import ObservationEvent, SemanticInstance


TAGGED_LOCATION_VECTOR_NAMESPACE = "tagged_location"
SEMANTIC_MEMORY_VECTOR_NAMESPACE = "semantic_memory"


@dataclass(frozen=True)
class VectorRecord:
    """持久化的向量记录。"""

    namespace: str
    record_id: str
    embedding_model: str
    embedding_dim: int
    source_text: str
    vector_payload: str
    updated_at: str


class MemoryRepository:
    """记忆库的轻量 SQLite 仓储。"""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path.strip() or ":memory:"
        self._lock = threading.RLock()
        if self._db_path != ":memory:":
            db_file = Path(self._db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self._db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def load_tagged_locations(self) -> List[TaggedLocation]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT payload_json
                FROM tagged_locations
                ORDER BY created_at ASC, location_id ASC
                """
            ).fetchall()
        return [TaggedLocation.model_validate_json(str(row["payload_json"])) for row in rows]

    def load_semantic_memories(self) -> List[SemanticMemoryEntry]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT payload_json
                FROM semantic_memories
                ORDER BY created_at ASC, memory_id ASC
                """
            ).fetchall()
        return [SemanticMemoryEntry.model_validate_json(str(row["payload_json"])) for row in rows]

    def load_semantic_instances(self) -> List[SemanticInstance]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT payload_json
                FROM semantic_instances
                ORDER BY updated_at ASC, instance_id ASC
                """
            ).fetchall()
        return [SemanticInstance.model_validate_json(str(row["payload_json"])) for row in rows]

    def load_observation_events(self) -> List[ObservationEvent]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT payload_json
                FROM observation_events
                ORDER BY created_at ASC, event_id ASC
                """
            ).fetchall()
        return [ObservationEvent.model_validate_json(str(row["payload_json"])) for row in rows]

    def load_vector_records(self, namespace: Optional[str] = None) -> Dict[str, VectorRecord]:
        if namespace is None:
            query = """
                SELECT namespace, record_id, embedding_model, embedding_dim, source_text, vector_payload, updated_at
                FROM memory_vectors
            """
            params: Iterable[object] = ()
        else:
            query = """
                SELECT namespace, record_id, embedding_model, embedding_dim, source_text, vector_payload, updated_at
                FROM memory_vectors
                WHERE namespace = ?
            """
            params = (namespace,)
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return {
            str(row["record_id"]): VectorRecord(
                namespace=str(row["namespace"]),
                record_id=str(row["record_id"]),
                embedding_model=str(row["embedding_model"]),
                embedding_dim=int(row["embedding_dim"]),
                source_text=str(row["source_text"]),
                vector_payload=str(row["vector_payload"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        }

    def upsert_tagged_location(self, location: TaggedLocation) -> None:
        payload_json = location.model_dump_json()
        created_at = location.timestamp.isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tagged_locations (
                    location_id,
                    normalized_name,
                    payload_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(location_id) DO UPDATE SET
                    normalized_name = excluded.normalized_name,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    location.location_id,
                    location.normalized_name,
                    payload_json,
                    created_at,
                    created_at,
                ),
            )
            self._conn.commit()

    def upsert_semantic_memory(self, entry: SemanticMemoryEntry) -> None:
        payload_json = entry.model_dump_json()
        created_at = entry.timestamp.isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO semantic_memories (
                    memory_id,
                    kind,
                    title,
                    summary,
                    payload_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    kind = excluded.kind,
                    title = excluded.title,
                    summary = excluded.summary,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    entry.memory_id,
                    entry.kind.value,
                    entry.title,
                    entry.summary,
                    payload_json,
                    created_at,
                    created_at,
                ),
            )
            self._conn.commit()

    def upsert_semantic_instance(self, entry: SemanticInstance) -> None:
        payload_json = entry.model_dump_json()
        created_at = entry.first_seen_ts.isoformat()
        updated_at = entry.last_seen_ts.isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO semantic_instances (
                    instance_id,
                    label,
                    lifecycle_state,
                    payload_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(instance_id) DO UPDATE SET
                    label = excluded.label,
                    lifecycle_state = excluded.lifecycle_state,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    entry.instance_id,
                    entry.label,
                    entry.lifecycle_state.value,
                    payload_json,
                    created_at,
                    updated_at,
                ),
            )
            self._conn.commit()

    def upsert_observation_event(self, entry: ObservationEvent) -> None:
        payload_json = entry.model_dump_json()
        created_at = entry.timestamp.isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO observation_events (
                    event_id,
                    anchor_id,
                    payload_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    anchor_id = excluded.anchor_id,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    entry.event_id,
                    entry.anchor_id,
                    payload_json,
                    created_at,
                    created_at,
                ),
            )
            self._conn.commit()

    def upsert_vector_record(
        self,
        *,
        namespace: str,
        record_id: str,
        embedding_model: str,
        embedding_dim: int,
        source_text: str,
        vector_payload: str,
        updated_at: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_vectors (
                    namespace,
                    record_id,
                    embedding_model,
                    embedding_dim,
                    source_text,
                    vector_payload,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, record_id) DO UPDATE SET
                    embedding_model = excluded.embedding_model,
                    embedding_dim = excluded.embedding_dim,
                    source_text = excluded.source_text,
                    vector_payload = excluded.vector_payload,
                    updated_at = excluded.updated_at
                """,
                (
                    namespace,
                    record_id,
                    embedding_model,
                    embedding_dim,
                    source_text,
                    vector_payload,
                    updated_at,
                ),
            )
            self._conn.commit()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tagged_locations (
                    location_id TEXT PRIMARY KEY,
                    normalized_name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tagged_locations_normalized_name ON tagged_locations(normalized_name)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    memory_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_memories_kind ON semantic_memories(kind)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    namespace TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    source_text TEXT NOT NULL,
                    vector_payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, record_id)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_vectors_namespace ON memory_vectors(namespace)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_instances (
                    instance_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_instances_label ON semantic_instances(label)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_instances_lifecycle_state ON semantic_instances(lifecycle_state)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_events (
                    event_id TEXT PRIMARY KEY,
                    anchor_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observation_events_anchor_id ON observation_events(anchor_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observation_events_created_at ON observation_events(created_at)"
            )
            self._conn.commit()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return
