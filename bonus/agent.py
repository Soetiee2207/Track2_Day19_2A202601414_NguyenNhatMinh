from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class MemoryChunk:
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


class HybridMemoryAgent:
    """
    Hybrid Memory POC:
    - Episodic memory: Qdrant
    - Stable profile + recent activity: Feature Store
    - Semantic chunking: embedding similarity
    """

    def __init__(
        self,
        qdrant_client=None,
        feature_store=None,
        collection_name: str = "episodic_memory",
        similarity_threshold: float = 0.55,
        max_tokens: int = 500,
    ):
        self.qdrant = qdrant_client
        self.feature_store = feature_store
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens

        self.embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Local fallback for demo
        self._local_memories: list[MemoryChunk] = []

    def remember(
        self,
        text: str,
        user_id: str = "u_001",
    ) -> None:
        """Add episodic memory using semantic chunking."""

        if not text.strip():
            return

        units = self._split_into_units(text)
        chunks = self._semantic_chunk(units)

        for chunk_text in chunks:
            embedding = self._embed(chunk_text)

            metadata = {
                "user_id": user_id,
                "text": chunk_text,
                "source": "conversation",
                "created_at": datetime.utcnow().isoformat(),
            }

            if self.qdrant is not None:
                self._upsert_qdrant(
                    embedding,
                    metadata,
                )
            else:
                self._local_memories.append(
                    MemoryChunk(
                        text=chunk_text,
                        embedding=embedding,
                        metadata=metadata,
                    )
                )

    def recall(
        self,
        query: str,
        user_id: str = "u_001",
    ) -> str:
        """Retrieve memories + profile + recent activity."""

        profile = self._get_profile(user_id)
        recent_activity = self._get_recent_activity(user_id)

        query_embedding = self._embed(query)

        if self.qdrant is not None:
            memories = self._search_qdrant(
                query_embedding,
                user_id,
                limit=3,
            )
        else:
            memories = self._search_local(
                query_embedding,
                user_id,
                limit=3,
            )

        return self._assemble_context(
            profile,
            recent_activity,
            memories,
        )

    # ================================================================
    # Semantic Chunking
    # ================================================================

    def _split_into_units(
        self,
        text: str,
    ) -> list[str]:
        """Split text into sentence-like units."""

        normalized = (
            text.replace("?", ".")
            .replace("!", ".")
            .replace("\n", ".")
        )

        return [
            unit.strip()
            for unit in normalized.split(".")
            if unit.strip()
        ]

    def _semantic_chunk(
        self,
        units: list[str],
    ) -> list[str]:
        """Group semantically similar units."""

        if not units:
            return []

        chunks: list[str] = []
        current_units: list[str] = []
        current_embedding: list[float] | None = None

        for unit in units:
            unit_embedding = self._embed(unit)

            if not current_units:
                current_units = [unit]
                current_embedding = unit_embedding
                continue

            similarity = self._cosine_similarity(
                current_embedding,
                unit_embedding,
            )

            current_text = " ".join(current_units)
            candidate_text = f"{current_text} {unit}"

            if (
                similarity >= self.similarity_threshold
                and self._estimate_tokens(candidate_text)
                <= self.max_tokens
            ):
                current_units.append(unit)

                embeddings = [
                    self._embed(item)
                    for item in current_units
                ]

                current_embedding = self._mean_embedding(
                    embeddings
                )

            else:
                chunks.append(
                    " ".join(current_units)
                )

                current_units = [unit]
                current_embedding = unit_embedding

        if current_units:
            chunks.append(
                " ".join(current_units)
            )

        return chunks

    # ================================================================
    # Embedding
    # ================================================================

    def _embed(
        self,
        text: str,
    ) -> list[float]:
        """Generate multilingual semantic embedding."""

        return self.embedder.encode(
            text,
            normalize_embeddings=True,
        ).tolist()

    @staticmethod
    def _cosine_similarity(
        a: list[float],
        b: list[float],
    ) -> float:
        a_vec = np.array(a)
        b_vec = np.array(b)

        denominator = (
            np.linalg.norm(a_vec)
            * np.linalg.norm(b_vec)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a_vec, b_vec)
            / denominator
        )

    @staticmethod
    def _mean_embedding(
        embeddings: list[list[float]],
    ) -> list[float]:
        vector = np.mean(
            np.array(embeddings),
            axis=0,
        )

        norm = np.linalg.norm(vector)

        if norm == 0:
            return vector.tolist()

        return (
            vector / norm
        ).tolist()

    @staticmethod
    def _estimate_tokens(
        text: str,
    ) -> int:
        """Simple token approximation for POC."""

        return max(
            1,
            len(text.split()),
        )

    # ================================================================
    # Qdrant
    # ================================================================

    def _upsert_qdrant(
        self,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        from qdrant_client.models import PointStruct

        point_id = (
            f"{metadata['user_id']}_"
            f"{datetime.utcnow().timestamp()}"
        )

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata,
                )
            ],
        )

    def _search_qdrant(
        self,
        query_embedding: list[float],
        user_id: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
        )

        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(
                            value=user_id
                        ),
                    )
                ]
            ),
            limit=limit,
        )

        return [
            {
                "text": result.payload["text"],
                "score": result.score,
                "metadata": result.payload,
            }
            for result in results
        ]

    # ================================================================
    # Local Search
    # ================================================================

    def _search_local(
        self,
        query_embedding: list[float],
        user_id: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        results = []

        for memory in self._local_memories:
            if memory.metadata["user_id"] != user_id:
                continue

            score = self._cosine_similarity(
                query_embedding,
                memory.embedding,
            )

            results.append(
                {
                    "text": memory.text,
                    "score": score,
                    "metadata": memory.metadata,
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:limit]

    # ================================================================
    # Feature Store
    # ================================================================

    def _get_profile(
        self,
        user_id: str,
    ) -> dict[str, Any]:
        """Get stable user profile."""

        if self.feature_store is not None:
            return self.feature_store.get_profile(
                user_id
            )

        return {
            "preferred_language": "vi/en",
            "reading_speed_wpm": 420,
            "topic_affinity": [
                "cloud",
                "AI",
                "Kubernetes",
            ],
        }

    def _get_recent_activity(
        self,
        user_id: str,
    ) -> dict[str, Any]:
        """Get recent activity features."""

        if self.feature_store is not None:
            return self.feature_store.get_recent_activity(
                user_id
            )

        return {
            "queries_last_hour": 5,
            "top_recent_topics": [
                "Kubernetes",
                "AWS",
                "cloud security",
            ],
        }

    # ================================================================
    # Context Assembly
    # ================================================================

    def _assemble_context(
        self,
        profile: dict[str, Any],
        recent_activity: dict[str, Any],
        memories: list[dict[str, Any]],
    ) -> str:

        topic_affinity = profile.get(
            "topic_affinity",
            [],
        )

        recent_topics = recent_activity.get(
            "top_recent_topics",
            [],
        )

        lines = [
            "USER PROFILE:",
            (
                "- Preferred language: "
                f"{profile.get('preferred_language', 'unknown')}"
            ),
            (
                "- Reading speed: "
                f"{profile.get('reading_speed_wpm', 'unknown')} WPM"
            ),
            (
                "- Topic affinity: "
                f"{', '.join(topic_affinity)}"
            ),
            "",
            "RECENT ACTIVITY:",
            (
                "- Queries in last hour: "
                f"{recent_activity.get('queries_last_hour', 0)}"
            ),
            (
                "- Recent topics: "
                f"{', '.join(recent_topics)}"
            ),
            "",
            "TOP MEMORIES:",
        ]

        if not memories:
            lines.append(
                "- No relevant memories found."
            )

        for index, memory in enumerate(
            memories,
            start=1,
        ):
            lines.append(
                f"{index}. "
                f"{memory['text']} "
                f"(score={memory['score']:.3f})"
            )

        return "\n".join(lines)