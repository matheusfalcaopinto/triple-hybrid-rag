"""
Entity + Relation Extraction (NER/RE) using OpenAI GPT-5.

This module provides a toggleable extraction pipeline that turns chunks into
entities, relations, and mentions, and optionally persists them to PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID, uuid4

import asyncpg
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import ChildChunk, EntityType, RelationType

logger = logging.getLogger(__name__)

# Models that don't support temperature parameter (only default temperature=1.0)
# gpt-5-nano and similar reasoning models have this limitation
MODELS_WITHOUT_TEMPERATURE_SUPPORT = frozenset({
    "gpt-5-nano",
    "o1",
    "o1-preview", 
    "o1-mini",
    "o3",
    "o3-mini",
})


def _model_supports_temperature(model: str) -> bool:
    """Check if the model supports the temperature parameter."""
    # Check exact match first
    if model.lower() in MODELS_WITHOUT_TEMPERATURE_SUPPORT:
        return False
    # Check if model name contains any of the unsupported model prefixes
    model_lower = model.lower()
    for unsupported in MODELS_WITHOUT_TEMPERATURE_SUPPORT:
        if unsupported in model_lower:
            return False
    return True

# Models that support reasoning_effort parameter (to control thinking depth)
MODELS_WITH_REASONING_EFFORT = frozenset({
    "gpt-5-nano",
    "o1",
    "o1-preview",
    "o1-mini",
    "o3",
    "o3-mini",
})

def _model_supports_reasoning_effort(model: str) -> bool:
    """Check if the model supports reasoning_effort parameter."""
    model_lower = model.lower()
    for model_name in MODELS_WITH_REASONING_EFFORT:
        if model_name in model_lower:
            return True
    return False


@dataclass
class EntityExtraction:
    chunk_id: str
    name: str
    canonical_name: str
    entity_type: str
    description: Optional[str] = None
    mention_text: str = ""
    confidence: float = 0.0


@dataclass
class RelationExtraction:
    chunk_id: str
    subject: str
    object: str
    relation_type: str
    confidence: float = 0.0
    evidence: Optional[str] = None


@dataclass
class GraphExtractionResult:
    entities: List[EntityExtraction] = field(default_factory=list)
    relations: List[RelationExtraction] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.relations


class EntityRelationExtractor:
    """
    Extract entities and relations from chunks using OpenAI GPT-5.

    The output can be stored using GraphEntityStore.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.config = config or get_settings()
        self.client = client or AsyncOpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )

    async def extract(self, chunks: List[ChildChunk]) -> GraphExtractionResult:
        if not self.config.rag_entity_extraction_enabled:
            logger.info("Entity extraction disabled; skipping NER/RE")
            return GraphExtractionResult()

        if not chunks:
            return GraphExtractionResult()

        usable_chunks = [c for c in chunks if c.text and c.modality.value == "text"]
        if not usable_chunks:
            return GraphExtractionResult()

        max_chunks = self.config.rag_ner_max_chunks_per_request
        results = GraphExtractionResult()

        for batch in _batched(usable_chunks, max_chunks):
            batch_result = await self._extract_batch(batch)
            results.entities.extend(batch_result.entities)
            results.relations.extend(batch_result.relations)

        return results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    async def _extract_batch(self, chunks: List[ChildChunk]) -> GraphExtractionResult:
        prompt = _build_prompt(chunks, self.config)
        
        # Build API request kwargs - some models don't support temperature
        request_kwargs = {
            "model": self.config.rag_ner_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        
        # Only add temperature if the model supports it
        # gpt-5-nano and o1/o3 reasoning models don't support temperature
        if _model_supports_temperature(self.config.rag_ner_model):
            request_kwargs["temperature"] = self.config.rag_ner_temperature
        else:
            logger.debug(f"Model {self.config.rag_ner_model} doesn't support temperature parameter, skipping")
        
        # Add reasoning_effort for reasoning models (gpt-5-nano, o1, o3)
        # "low" = faster + cheaper (good for NER), "medium" = balanced, "high" = most thorough
        if _model_supports_reasoning_effort(self.config.rag_ner_model):
            reasoning_effort = getattr(self.config, 'rag_ner_reasoning_effort', 'low')
            request_kwargs["reasoning_effort"] = reasoning_effort
            logger.debug(f"Model {self.config.rag_ner_model} supports reasoning_effort, using: {reasoning_effort}")
        
        response = await self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or "{}"
        return _parse_extraction_response(content, self.config)


class GraphEntityStore:
    """
    Persist extracted entities, relations, and mentions to PostgreSQL.
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or get_settings()
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=1,
                max_size=self.config.database_pool_size,
            )
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def store(
        self,
        result: GraphExtractionResult,
        chunks: List[ChildChunk],
        tenant_id: str,
        document_id: UUID,
    ) -> Dict[str, int]:
        if result.is_empty:
            return {"entities": 0, "relations": 0, "mentions": 0}

        chunk_lookup = {str(c.id): c for c in chunks}
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.transaction():
                entity_id_map = await _upsert_entities(
                    conn,
                    result.entities,
                    tenant_id,
                    document_id,
                )
                mention_count = await _insert_mentions(
                    conn,
                    result.entities,
                    chunk_lookup,
                    entity_id_map,
                    document_id,
                )
                relation_count = await _insert_relations(
                    conn,
                    result.relations,
                    entity_id_map,
                    tenant_id,
                    document_id,
                )

        return {
            "entities": len(entity_id_map),
            "relations": relation_count,
            "mentions": mention_count,
        }


_SYSTEM_PROMPT = (
    "You are an information extraction engine. "
    "Extract entities and relations strictly from the provided chunks. "
    "Return valid JSON only."
)


def _build_prompt(chunks: List[ChildChunk], config: RAGConfig) -> str:
    entity_types = ", ".join(config.entity_types_list)
    relation_types = ", ".join(rt.value for rt in RelationType)

    payload = {
        "entity_types": entity_types,
        "relation_types": relation_types,
        "instructions": [
            "Use only the provided entity_types and relation_types.",
            "Return canonical_name as a lowercase, punctuation-stripped version of name.",
            "Use chunk_id exactly as provided.",
            "If unsure, omit the entity/relation.",
        ],
        "chunks": [
            {
                "chunk_id": str(c.id),
                "text": c.text[: config.rag_ner_max_chars_per_chunk],
            }
            for c in chunks
        ],
        "output_schema": {
            "entities": [
                {
                    "chunk_id": "string",
                    "name": "string",
                    "canonical_name": "string",
                    "entity_type": "string",
                    "description": "string|null",
                    "mention_text": "string",
                    "confidence": "number"
                }
            ],
            "relations": [
                {
                    "chunk_id": "string",
                    "subject": "canonical_name",
                    "object": "canonical_name",
                    "relation_type": "string",
                    "confidence": "number",
                    "evidence": "string|null"
                }
            ]
        },
    }

    return json.dumps(payload, ensure_ascii=False)


def _normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\s._-]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _parse_extraction_response(content: str, config: RAGConfig) -> GraphExtractionResult:
    data = _safe_json_loads(content)
    entities: List[EntityExtraction] = []
    relations: List[RelationExtraction] = []

    for raw in data.get("entities", []) or []:
        name = str(raw.get("name", "")).strip()
        canonical = str(raw.get("canonical_name", "")) or _normalize_name(name)
        entity_type = str(raw.get("entity_type", "")).upper()
        if entity_type not in EntityType.__members__:
            entity_type = EntityType.CONCEPT.value
        mention_text = str(raw.get("mention_text", name)).strip()
        entities.append(
            EntityExtraction(
                chunk_id=str(raw.get("chunk_id", "")),
                name=name,
                canonical_name=_normalize_name(canonical),
                entity_type=entity_type,
                description=_optional_str(raw.get("description")),
                mention_text=mention_text,
                confidence=_safe_float(raw.get("confidence"), 0.6),
            )
        )

    for raw in data.get("relations", []) or []:
        relation_type = str(raw.get("relation_type", "")).upper()
        if relation_type not in RelationType.__members__:
            relation_type = RelationType.RELATED_TO.value
        relations.append(
            RelationExtraction(
                chunk_id=str(raw.get("chunk_id", "")),
                subject=_normalize_name(str(raw.get("subject", ""))),
                object=_normalize_name(str(raw.get("object", ""))),
                relation_type=relation_type,
                confidence=_safe_float(raw.get("confidence"), 0.5),
                evidence=_optional_str(raw.get("evidence")),
            )
        )

    return GraphExtractionResult(entities=entities, relations=relations)


def _safe_json_loads(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _batched(items: List[ChildChunk], size: int) -> Iterable[List[ChildChunk]]:
    if size <= 0:
        yield items
        return
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


async def _upsert_entities(
    conn: asyncpg.Connection,
    entities: List[EntityExtraction],
    tenant_id: str,
    document_id: UUID,
) -> Dict[str, UUID]:
    entity_id_map: Dict[str, UUID] = {}
    for entity in entities:
        if not entity.canonical_name:
            continue
        row = await conn.fetchrow(
            """
            INSERT INTO rag_entities (
                id, tenant_id, document_id, entity_type, name, canonical_name, description, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (tenant_id, canonical_name)
            DO UPDATE SET
                name = EXCLUDED.name,
                entity_type = EXCLUDED.entity_type,
                description = COALESCE(EXCLUDED.description, rag_entities.description)
            RETURNING id
            """,
            uuid4(),
            tenant_id,
            document_id,
            entity.entity_type,
            entity.name,
            entity.canonical_name,
            entity.description,
            {"source": "openai-gpt-5"},
        )
        entity_id_map[entity.canonical_name] = row["id"]

    return entity_id_map


async def _insert_mentions(
    conn: asyncpg.Connection,
    entities: List[EntityExtraction],
    chunk_lookup: Dict[str, ChildChunk],
    entity_id_map: Dict[str, UUID],
    document_id: UUID,
) -> int:
    count = 0
    for entity in entities:
        chunk = chunk_lookup.get(entity.chunk_id)
        if not chunk:
            continue
        entity_id = entity_id_map.get(entity.canonical_name)
        if not entity_id:
            continue
        await conn.execute(
            """
            INSERT INTO rag_entity_mentions (
                id, entity_id, parent_chunk_id, child_chunk_id, document_id,
                mention_text, confidence, char_start, char_end
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            uuid4(),
            entity_id,
            chunk.parent_id,
            chunk.id,
            document_id,
            entity.mention_text or entity.name,
            entity.confidence,
            None,
            None,
        )
        count += 1
    return count


async def _insert_relations(
    conn: asyncpg.Connection,
    relations: List[RelationExtraction],
    entity_id_map: Dict[str, UUID],
    tenant_id: str,
    document_id: UUID,
) -> int:
    count = 0
    for relation in relations:
        subject_id = entity_id_map.get(relation.subject)
        object_id = entity_id_map.get(relation.object)
        if not subject_id or not object_id:
            continue
        await conn.execute(
            """
            INSERT INTO rag_relations (
                id, tenant_id, document_id, relation_type, subject_entity_id,
                object_entity_id, confidence, source_parent_id, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            uuid4(),
            tenant_id,
            document_id,
            relation.relation_type,
            subject_id,
            object_id,
            relation.confidence,
            None,
            {"evidence": relation.evidence} if relation.evidence else {},
        )
        count += 1
    return count
