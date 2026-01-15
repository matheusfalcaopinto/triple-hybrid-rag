"""
RAG 2.0 Entity Extraction Module

GPT-5 based Named Entity Recognition (NER) and Relation Extraction:
- Extracts domain-specific entities from parent chunks
- Identifies relationships between entities
- Resolves multimodal entity ambiguities
- Stores triplets in rag_entities and rag_relations tables
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openai import AsyncOpenAI

from voice_agent.config import SETTINGS

logger = logging.getLogger(__name__)


# =============================================================================
# ENTITY TYPES
# =============================================================================

ENTITY_TYPES = [
    "PERSON",           # People names
    "ORGANIZATION",     # Companies, institutions
    "PRODUCT",          # Products, services
    "LOCATION",         # Physical locations
    "DATE",             # Dates, time periods
    "MONEY",            # Monetary amounts
    "PERCENTAGE",       # Percentages
    "CONTRACT",         # Contract references
    "CLAUSE",           # Contract clauses, terms
    "REGULATION",       # Laws, regulations
    "DOCUMENT",         # Referenced documents
    "TECHNOLOGY",       # Technical terms, systems
    "PROCESS",          # Business processes
    "EVENT",            # Events, meetings
    "METRIC",           # KPIs, measurements
]

RELATION_TYPES = [
    "OWNS",             # Ownership relation
    "EMPLOYS",          # Employment relation
    "LOCATED_IN",       # Location relation
    "PART_OF",          # Component/part relation
    "REFERENCES",       # Document references
    "DEPENDS_ON",       # Dependency relation
    "CREATED_BY",       # Authorship
    "DATED",            # Date association
    "VALUED_AT",        # Value/price relation
    "REGULATES",        # Regulatory relation
    "IMPLEMENTS",       # Implementation relation
    "SUPERSEDES",       # Replacement relation
    "RELATED_TO",       # Generic relation
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractedEntity:
    """An entity extracted from text."""
    name: str
    entity_type: str
    canonical_name: Optional[str] = None
    description: Optional[str] = None
    confidence: float = 1.0
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    """A relation between two entities."""
    subject: str  # Subject entity name
    relation_type: str
    object: str  # Object entity name
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of entity/relation extraction."""
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    errors: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None


# =============================================================================
# SYSTEM PROMPT FOR ENTITY EXTRACTION
# =============================================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an expert Named Entity Recognition (NER) and Relation Extraction system.

Analyze the provided text and extract:
1. Named Entities with their types
2. Relations between entities

## Entity Types
{entity_types}

## Relation Types
{relation_types}

## Output Format
Return a JSON object with:
```json
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "ENTITY_TYPE",
      "canonical_name": "Normalized name (optional)",
      "description": "Brief description (optional)",
      "confidence": 0.95
    }}
  ],
  "relations": [
    {{
      "subject": "Subject Entity Name",
      "relation_type": "RELATION_TYPE",
      "object": "Object Entity Name",
      "confidence": 0.9
    }}
  ]
}}
```

## Guidelines
1. Extract ALL relevant entities, including implicit ones
2. Use canonical names to resolve variations (e.g., "IBM" and "International Business Machines")
3. Confidence should reflect how certain you are (0.0-1.0)
4. Relations should only reference entities in the entities list
5. For ambiguous entities, provide a brief description
6. Include document-specific entities like clause names, contract IDs

Return ONLY valid JSON, no explanations."""


# =============================================================================
# ENTITY EXTRACTOR CLASS
# =============================================================================

class EntityExtractor:
    """
    GPT-5 based entity and relation extractor for RAG 2.0.
    
    Processes parent chunks to extract structured knowledge graph data.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        """
        Initialize the entity extractor.
        
        Args:
            model: Model to use for extraction (default from config)
            api_key: OpenAI API key
            base_url: OpenAI base URL
            temperature: Generation temperature
            max_tokens: Max tokens in response
        """
        self.model = model or getattr(SETTINGS, 'rag2_ner_model', 'gpt-4o')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._client: Optional[AsyncOpenAI] = None
        self._api_key = api_key or SETTINGS.openai_api_key
        self._base_url = base_url or SETTINGS.openai_base_url
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-load async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with entity/relation types."""
        entity_list = "\n".join(f"- {t}" for t in ENTITY_TYPES)
        relation_list = "\n".join(f"- {t}" for t in RELATION_TYPES)
        
        return ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
            entity_types=entity_list,
            relation_types=relation_list,
        )
    
    async def extract(
        self,
        text: str,
        context: Optional[str] = None,
        document_title: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract entities and relations from text.
        
        Args:
            text: Text to analyze (typically a parent chunk)
            context: Optional context (e.g., document type)
            document_title: Optional document title for context
            
        Returns:
            ExtractionResult with entities and relations
        """
        if not text or len(text.strip()) < 50:
            return ExtractionResult(entities=[], relations=[])
        
        try:
            # Build messages
            system_prompt = self._build_system_prompt()
            
            user_message = f"Extract entities and relations from this text:\n\n{text}"
            if context:
                user_message = f"Context: {context}\n\n{user_message}"
            if document_title:
                user_message = f"Document: {document_title}\n\n{user_message}"
            
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            
            # Call GPT
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            
            # Parse response
            content = response.choices[0].message.content or ""
            
            return self._parse_response(content)
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return ExtractionResult(
                entities=[],
                relations=[],
                errors=[str(e)],
            )
    
    def _parse_response(self, content: str) -> ExtractionResult:
        """Parse the LLM response into structured data."""
        try:
            data = json.loads(content)
            
            # Parse entities
            entities = []
            for e in data.get("entities", []):
                entity = ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "UNKNOWN"),
                    canonical_name=e.get("canonical_name"),
                    description=e.get("description"),
                    confidence=float(e.get("confidence", 1.0)),
                    metadata=e.get("metadata", {}),
                )
                if entity.name:
                    entities.append(entity)
            
            # Parse relations
            relations = []
            for r in data.get("relations", []):
                relation = ExtractedRelation(
                    subject=r.get("subject", ""),
                    relation_type=r.get("relation_type", "RELATED_TO"),
                    object=r.get("object", ""),
                    confidence=float(r.get("confidence", 1.0)),
                    metadata=r.get("metadata", {}),
                )
                if relation.subject and relation.object:
                    relations.append(relation)
            
            return ExtractionResult(
                entities=entities,
                relations=relations,
                raw_response=content,
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction response: {e}")
            return ExtractionResult(
                entities=[],
                relations=[],
                errors=[f"JSON parse error: {e}"],
                raw_response=content,
            )
    
    async def extract_batch(
        self,
        texts: List[str],
        contexts: Optional[List[Optional[str]]] = None,
        document_title: Optional[str] = None,
        max_concurrent: int = 5,
    ) -> List[ExtractionResult]:
        """
        Extract entities from multiple texts in parallel.
        
        Args:
            texts: List of texts to process
            contexts: Optional list of contexts
            document_title: Document title for all texts
            max_concurrent: Max concurrent API calls
            
        Returns:
            List of ExtractionResult objects
        """
        ctx_list: List[Optional[str]] = contexts if contexts else [None] * len(texts)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_one(text: str, context: Optional[str]) -> ExtractionResult:
            async with semaphore:
                return await self.extract(text, context, document_title)
        
        tasks = [
            process_one(text, ctx)
            for text, ctx in zip(texts, ctx_list)
        ]
        
        return await asyncio.gather(*tasks)


# =============================================================================
# ENTITY STORAGE
# =============================================================================

class EntityStore:
    """
    Stores extracted entities and relations in the database.
    
    Handles:
    - Entity deduplication via canonical_name
    - Relation normalization
    - Entity-chunk mention tracking
    """
    
    def __init__(self, supabase_client: Any):
        """Initialize with Supabase client."""
        self.db = supabase_client
    
    async def store_extraction(
        self,
        result: ExtractionResult,
        org_id: str,
        document_id: str,
        parent_chunk_id: str,
        child_chunk_ids: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Store extracted entities and relations.
        
        Args:
            result: Extraction result
            org_id: Organization ID
            document_id: Document ID
            parent_chunk_id: Source parent chunk ID
            child_chunk_ids: Optional child chunk IDs for mentions
            
        Returns:
            Stats dict with counts
        """
        stats = {
            "entities_created": 0,
            "entities_updated": 0,
            "relations_created": 0,
            "mentions_created": 0,
        }
        
        if not result.entities:
            return stats
        
        # Map entity names to IDs
        entity_id_map: Dict[str, str] = {}
        
        # Store entities
        for entity in result.entities:
            entity_id = await self._upsert_entity(
                entity=entity,
                org_id=org_id,
                document_id=document_id,
            )
            entity_id_map[entity.name] = entity_id
            
            # Also map canonical name if different
            if entity.canonical_name and entity.canonical_name != entity.name:
                entity_id_map[entity.canonical_name] = entity_id
            
            stats["entities_created"] += 1
        
        # Store relations
        for relation in result.relations:
            subject_id = entity_id_map.get(relation.subject)
            object_id = entity_id_map.get(relation.object)
            
            if subject_id and object_id:
                await self._create_relation(
                    subject_id=subject_id,
                    object_id=object_id,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence,
                    org_id=org_id,
                    document_id=document_id,
                    source_parent_id=parent_chunk_id,
                    metadata=relation.metadata,
                )
                stats["relations_created"] += 1
        
        # Create mentions linking entities to chunks
        if child_chunk_ids:
            for entity in result.entities:
                entity_id = entity_id_map.get(entity.name)
                if entity_id:
                    for chunk_id in child_chunk_ids:
                        await self._create_mention(
                            entity_id=entity_id,
                            parent_chunk_id=parent_chunk_id,
                            child_chunk_id=chunk_id,
                            document_id=document_id,
                            mention_text=entity.name,
                            confidence=entity.confidence,
                        )
                        stats["mentions_created"] += 1
        
        return stats
    
    async def _upsert_entity(
        self,
        entity: ExtractedEntity,
        org_id: str,
        document_id: str,
    ) -> str:
        """Insert or update an entity, returning its ID."""
        canonical = entity.canonical_name or entity.name
        
        # Check if entity exists
        existing = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.db.table("rag_entities")
            .select("id")
            .eq("org_id", org_id)
            .eq("canonical_name", canonical)
            .limit(1)
            .execute()
        )
        
        if existing.data:
            return existing.data[0]["id"]
        
        # Create new entity
        entity_id = str(uuid4())
        entity_data = {
            "id": entity_id,
            "org_id": org_id,
            "document_id": document_id,
            "entity_type": entity.entity_type,
            "name": entity.name,
            "canonical_name": canonical,
            "description": entity.description,
            "metadata": entity.metadata,
        }
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.db.table("rag_entities").insert(entity_data).execute()
        )
        
        return entity_id
    
    async def _create_relation(
        self,
        subject_id: str,
        object_id: str,
        relation_type: str,
        confidence: float,
        org_id: str,
        document_id: str,
        source_parent_id: str,
        metadata: dict,
    ) -> str:
        """Create a relation between two entities."""
        relation_id = str(uuid4())
        relation_data = {
            "id": relation_id,
            "org_id": org_id,
            "document_id": document_id,
            "relation_type": relation_type,
            "subject_entity_id": subject_id,
            "object_entity_id": object_id,
            "confidence": confidence,
            "source_parent_id": source_parent_id,
            "metadata": metadata,
        }
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.db.table("rag_relations").insert(relation_data).execute()
        )
        
        return relation_id
    
    async def _create_mention(
        self,
        entity_id: str,
        parent_chunk_id: str,
        child_chunk_id: str,
        document_id: str,
        mention_text: str,
        confidence: float,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
    ) -> str:
        """Create an entity mention record."""
        mention_id = str(uuid4())
        mention_data = {
            "id": mention_id,
            "entity_id": entity_id,
            "parent_chunk_id": parent_chunk_id,
            "child_chunk_id": child_chunk_id,
            "document_id": document_id,
            "mention_text": mention_text,
            "confidence": confidence,
            "char_start": char_start,
            "char_end": char_end,
        }
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.db.table("rag_entity_mentions").insert(mention_data).execute()
        )
        
        return mention_id


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_entity_extractor(**kwargs: Any) -> EntityExtractor:
    """Get a configured entity extractor instance."""
    return EntityExtractor(**kwargs)


def get_entity_store(supabase_client: Any) -> EntityStore:
    """Get an entity store instance."""
    return EntityStore(supabase_client)
