"""GraphRAG Pipeline with Inline Citations, Hybrid Search, and Graph-Enriched Retrieval."""

# Suppress Vertex AI deprecation warnings BEFORE any imports
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
warnings.filterwarnings("ignore", message=".*deprecated.*")

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import vertexai
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import VertexAILLM
from vertexai.generative_models import GenerationConfig
from vertexai.language_models import TextEmbeddingModel

from .config import PipelineConfig
from .database import DatabaseManager
from .logging_config import MarkdownLogger

logger = logging.getLogger(__name__)

# Enable neo4j-graphrag library logging to capture errors
logging.getLogger("neo4j_graphrag").setLevel(logging.DEBUG)


# =============================================================================
# Data Models for Citations
# =============================================================================


@dataclass
class Citation:
    """A citation reference to a source chunk."""

    id: int  # Citation number [1], [2], etc.
    chunk_id: str  # Chunk ID in database
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    text_preview: str = ""  # First ~100 chars of the chunk
    score: float = 0.0  # Retrieval score
    source_type: str = "vector"  # 'vector', 'keyword', or 'hybrid'


@dataclass
class CitedAnswer:
    """An answer with inline citations and source references."""

    answer: str  # Answer with [1], [2] citations
    citations: List[Citation] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    graph_context: List[Dict[str, Any]] = field(default_factory=list)

    def format_citations(self) -> str:
        """Format citations as a reference list."""
        if not self.citations:
            return ""

        lines = ["\n\n---\n**Sources:**"]
        for citation in self.citations:
            source_type = f"({citation.source_type})" if citation.source_type else ""
            doc_info = f" from _{citation.document_title}_" if citation.document_title else ""
            lines.append(f"[{citation.id}]{doc_info}: {citation.text_preview}... {source_type}")

        return "\n".join(lines)


# =============================================================================
# Custom Logging Handler
# =============================================================================


class MarkdownLogHandler(logging.Handler):
    """Custom logging handler that writes to MarkdownLogger."""

    def __init__(self, md_logger: "MarkdownLogger"):
        super().__init__()
        self.md_logger = md_logger
        self.setLevel(logging.DEBUG)

    def emit(self, record: logging.LogRecord):
        """Write log record to markdown logger."""
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                self.md_logger.error(f"[{record.name}] {msg}")
            elif record.levelno >= logging.WARNING:
                self.md_logger.warning(f"[{record.name}] {msg}")
            else:
                self.md_logger.info(f"[{record.name}] {msg}")
        except Exception:
            self.handleError(record)


# =============================================================================
# Progress Callback System
# =============================================================================

_progress_callback: Optional[Callable[[str, int, int], None]] = None


def set_progress_callback(callback: Optional[Callable[[str, int, int], None]]):
    """Set a callback function for progress updates."""
    global _progress_callback
    _progress_callback = callback


def _update_progress(message: str, current: int = 0, total: int = 0):
    """Update progress via callback if set."""
    global _progress_callback
    if _progress_callback:
        _progress_callback(message, current, total)


# =============================================================================
# Custom Vertex AI Embedder
# =============================================================================


class VertexAIEmbedder(Embedder):
    """Custom Vertex AI embedder using gemini-embedding-001 (3072 dimensions)."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if project_id:
            vertexai.init(project=project_id)
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        logger.info(f"VertexAI Embedder initialized: {model_name} (3072 dimensions)")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text string."""
        embeddings = self.model.get_embeddings([text])
        return embeddings[0].values


# =============================================================================
# GraphRAG Pipeline with All Enhancements
# =============================================================================


class GraphRAGPipeline:
    """Production-ready GraphRAG pipeline with citations, hybrid search, and graph enrichment."""

    def __init__(self, config: PipelineConfig):
        """Initialize the GraphRAG pipeline with all enhanced features."""
        self.config = config
        self.db = DatabaseManager(config.neo4j)
        self.md_logger = MarkdownLogger(config.log_dir)
        self._query_llm = None

        # Attach custom handler to neo4j_graphrag logger
        self._md_handler = MarkdownLogHandler(self.md_logger)
        neo4j_logger = logging.getLogger("neo4j_graphrag")
        neo4j_logger.addHandler(self._md_handler)

        # Initialize Vertex AI
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if project_id:
            vertexai.init(project=project_id)
            logger.info(f"Vertex AI initialized with project: {project_id}")

        # Lazy initialization
        self._llm = None
        self._embedder = None
        self._kg_pipeline = None

        # Fulltext index configuration
        self._fulltext_index_name = "chunk_fulltext"
        self._fulltext_index_created = False

        logger.info("GraphRAG Pipeline initialized with enhanced features")

    # =========================================================================
    # LLM Properties
    # =========================================================================

    @property
    def llm(self):
        """Get or create LLM instance for entity extraction (JSON mode)."""
        if self._llm is None:
            generation_config = GenerationConfig(
                temperature=self.config.llm.temperature,
                max_output_tokens=self.config.llm.max_tokens,
                response_mime_type="application/json",
            )
            self._llm = VertexAILLM(
                model_name=self.config.llm.model_name,
                generation_config=generation_config,
            )
            logger.info(f"Vertex AI LLM initialized: {self.config.llm.model_name} (JSON mode)")
        return self._llm

    @property
    def query_llm(self):
        """Get or create LLM instance for queries (natural text mode)."""
        if not hasattr(self, "_query_llm") or self._query_llm is None:
            generation_config = GenerationConfig(
                temperature=0.3,
                max_output_tokens=self.config.llm.max_tokens,
            )
            self._query_llm = VertexAILLM(
                model_name=self.config.llm.model_name,
                generation_config=generation_config,
            )
            logger.info(f"Vertex AI Query LLM initialized: {self.config.llm.model_name}")
        return self._query_llm

    @property
    def embedder(self):
        """Get or create embedder instance."""
        if self._embedder is None:
            self._embedder = VertexAIEmbedder(model_name=self.config.embedding.model_name)
        return self._embedder

    # =========================================================================
    # Fulltext Index Setup (for Hybrid Search)
    # =========================================================================

    def _ensure_fulltext_index(self):
        """Ensure fulltext index exists for hybrid search."""
        if self._fulltext_index_created:
            return

        try:
            # Check if index already exists
            check_query = """
            SHOW INDEXES WHERE name = $index_name
            """
            existing = self.db.execute_query(check_query, {"index_name": self._fulltext_index_name})

            if not existing:
                # Create fulltext index on Chunk.text
                create_query = f"""
                CREATE FULLTEXT INDEX {self._fulltext_index_name} IF NOT EXISTS
                FOR (c:Chunk)
                ON EACH [c.text]
                """
                self.db.execute_write(create_query, {})
                logger.info(f"Created fulltext index: {self._fulltext_index_name}")
                self.md_logger.info(f"Created fulltext index: {self._fulltext_index_name}")
            else:
                logger.info(f"Fulltext index already exists: {self._fulltext_index_name}")

            self._fulltext_index_created = True

        except Exception as e:
            logger.warning(f"Failed to create fulltext index: {e}")
            self.md_logger.warning(f"Fulltext index creation failed: {e}")

    # =========================================================================
    # Knowledge Graph Pipeline
    # =========================================================================

    async def _create_kg_pipeline(self) -> SimpleKGPipeline:
        """Create the official SimpleKGPipeline."""
        if self._kg_pipeline is None:
            schema = self.config.schema
            entity_types = [nt["label"] if isinstance(nt, dict) else nt for nt in schema.node_types]
            relation_types = [
                rt["label"] if isinstance(rt, dict) else rt for rt in schema.relationship_types
            ]

            self._kg_pipeline = SimpleKGPipeline(
                llm=self.llm,
                driver=self.db.driver,
                embedder=self.embedder,
                entities=entity_types,
                relations=relation_types,
                potential_schema=schema.patterns if schema.patterns else None,
                from_pdf=False,
            )
            logger.info("SimpleKGPipeline created with Vertex AI components")

        return self._kg_pipeline

    # =========================================================================
    # Document Ingestion
    # =========================================================================

    async def ingest_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        document_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest text and build knowledge graph."""
        self.md_logger.section("Document Ingestion")
        self.md_logger.info(f"Document: {document_title or 'Untitled'}")
        self.md_logger.info(f"Document ID: {document_id or 'auto-generated'}")
        self.md_logger.info(f"Text length: {len(text):,} characters")

        start_time = datetime.now()
        _update_progress("Initializing...", 0, 0)

        try:
            # Set up vector index
            self.md_logger.subsection("Initializing Indexes")
            _update_progress("Setting up vector index...", 0, 0)
            self.db.setup_vector_index(
                index_name=self.config.vector_index_name,
                dimensions=self.config.vector_dimensions,
            )
            self.md_logger.info(f"Vector index ready: {self.config.vector_index_name}")

            # Set up fulltext index for hybrid search
            _update_progress("Setting up fulltext index...", 0, 0)
            self._ensure_fulltext_index()

            estimated_chunks = max(1, len(text) // 4000)
            self.md_logger.info(f"Estimated chunks: ~{estimated_chunks}")

            pipeline = await self._create_kg_pipeline()

            self.md_logger.subsection("Pipeline Configuration")
            self.md_logger.info(f"LLM: {self.config.llm.model_name}")
            self.md_logger.info(f"Embeddings: {self.config.embedding.model_name} (3072 dimensions)")
            self.md_logger.info(f"Entity Resolution: {self.config.resolution_type}")
            self.md_logger.info("Enhanced Features: Citations, Hybrid Search, Graph Enrichment")

            # Log input preview
            self.md_logger.subsection("Input Text Preview")
            preview = text[:1000] + "..." if len(text) > 1000 else text
            self.md_logger.text_block(preview)

            # Run pipeline
            self.md_logger.subsection("Running SimpleKGPipeline")
            _update_progress(
                f"Processing text (~{estimated_chunks} chunks)...", 0, estimated_chunks
            )

            try:
                result = await pipeline.run_async(text=text)
            except Exception as pipeline_error:
                self.md_logger.error(f"Pipeline error: {pipeline_error}")
                raise

            elapsed = (datetime.now() - start_time).total_seconds()
            self.md_logger.info(f"Pipeline completed in {elapsed:.2f} seconds")

            # Create document node
            doc_id = document_id or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self._create_document_node(doc_id, document_title, metadata)

            # Link chunks to document
            await self._link_chunks_to_document(doc_id)

            # Run entity resolution
            if self.config.perform_entity_resolution:
                _update_progress("Running entity resolution...", 0, 0)
                await self._run_entity_resolution()

            # Get counts
            _update_progress("Querying results...", 0, 0)
            chunk_count = self.db.execute_query("MATCH (c:Chunk) RETURN count(c) as count")
            entity_count = self.db.execute_query("MATCH (e:__Entity__) RETURN count(e) as count")
            rel_count = self.db.execute_query(
                "MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r) as count"
            )

            chunks = chunk_count[0]["count"] if chunk_count else 0
            entities = entity_count[0]["count"] if entity_count else 0
            relationships = rel_count[0]["count"] if rel_count else 0

            await self._log_ingestion_results(doc_id, chunks, entities, relationships)
            self.md_logger.success("Ingestion completed successfully")

            return {
                "document_id": doc_id,
                "status": "success",
                "chunks": chunks,
                "entities": entities,
                "relationships": relationships,
                "elapsed_seconds": elapsed,
            }

        except Exception as e:
            self.md_logger.error(f"Ingestion failed: {e}")
            import traceback

            self.md_logger.text_block(traceback.format_exc())
            raise

    async def _link_chunks_to_document(self, document_id: str):
        """Link all chunks to the document node for better citation tracking."""
        try:
            query = """
            MATCH (d:Document {id: $doc_id})
            MATCH (c:Chunk) WHERE NOT (c)-[:FROM_DOCUMENT]->(:Document)
            MERGE (c)-[:FROM_DOCUMENT]->(d)
            SET c.document_id = $doc_id
            RETURN count(c) as linked_count
            """
            result = self.db.execute_write(query, {"doc_id": document_id})
            if result:
                logger.info(f"Linked chunks to document: {document_id}")
        except Exception as e:
            logger.warning(f"Failed to link chunks to document: {e}")

    async def _create_document_node(
        self,
        doc_id: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ):
        """Create document node in the graph."""
        query = """
        MERGE (d:Document {id: $doc_id})
        SET d.title = $title,
            d.created_at = datetime(),
            d.metadata = $metadata
        """
        self.db.execute_write(
            query,
            {
                "doc_id": doc_id,
                "title": title or doc_id,
                "metadata": json.dumps(metadata or {}),
            },
        )

    async def _run_entity_resolution(self):
        """Run entity resolution to merge duplicate entities."""
        self.md_logger.subsection("Entity Resolution")

        try:
            if self.config.resolution_type == "exact":
                from neo4j_graphrag.experimental.components.resolver import (
                    SinglePropertyExactMatchResolver,
                )

                resolver = SinglePropertyExactMatchResolver(self.db.driver)
                await resolver.run()
                self.md_logger.success("Entity resolution completed (exact match)")

            elif self.config.resolution_type == "fuzzy":
                try:
                    from neo4j_graphrag.experimental.components.resolver import FuzzyMatchResolver

                    resolver = FuzzyMatchResolver(self.db.driver)
                    await resolver.run()
                    self.md_logger.success("Entity resolution completed (fuzzy match)")
                except ImportError:
                    self.md_logger.warning("Fuzzy matching not available")

            elif self.config.resolution_type == "semantic":
                try:
                    from neo4j_graphrag.experimental.components.resolver import (
                        SpaCySemanticMatchResolver,
                    )

                    resolver = SpaCySemanticMatchResolver(self.db.driver)
                    await resolver.run()
                    self.md_logger.success("Entity resolution completed (semantic match)")
                except ImportError:
                    self.md_logger.warning("Semantic matching not available")

        except Exception as e:
            self.md_logger.error(f"Entity resolution failed: {e}")

    async def _log_ingestion_results(
        self, doc_id: str, chunk_count: int, entity_count: int, rel_count: int
    ):
        """Log detailed ingestion results."""
        self.md_logger.subsection("Ingestion Summary")
        self.md_logger.stats(
            {
                "Document ID": doc_id,
                "Total Chunks": chunk_count,
                "Total Entities": entity_count,
                "Total Relationships": rel_count,
            }
        )

        # Log chunks
        self.md_logger.subsection("Created Chunks")
        chunks = self.db.execute_query(
            "MATCH (c:Chunk) RETURN c.id as id, substring(c.text, 0, 200) as preview LIMIT 20"
        )
        if chunks:
            for i, chunk in enumerate(chunks, 1):
                self.md_logger.info(f"**Chunk {i}** (ID: {chunk.get('id', 'N/A')})")
                if chunk.get("preview"):
                    self.md_logger.text_block(chunk["preview"] + "...")

        # Log entities
        self.md_logger.subsection("Extracted Entities")
        entities = self.db.execute_query(
            """
            MATCH (e:__Entity__)
            RETURN e.name as name, labels(e) as labels, e.description as description
            ORDER BY e.name LIMIT 50
        """
        )
        if entities:
            headers = ["Name", "Type", "Description"]
            rows = []
            for e in entities:
                labels = [l for l in e.get("labels", []) if l != "__Entity__"]
                entity_type = labels[0] if labels else "Entity"
                desc = (e.get("description") or "")[:60]
                rows.append([e.get("name", "?"), entity_type, desc])
            self.md_logger.table(headers, rows)

        # Log relationships
        self.md_logger.subsection("Extracted Relationships")
        relationships = self.db.execute_query(
            """
            MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
            RETURN e1.name as source, type(r) as relationship, e2.name as target
            ORDER BY e1.name LIMIT 50
        """
        )
        if relationships:
            headers = ["Source", "Relationship", "Target"]
            rows = [[r["source"], r["relationship"], r["target"]] for r in relationships]
            self.md_logger.table(headers, rows)

    # =========================================================================
    # Query with Inline Citations
    # =========================================================================

    async def query(
        self,
        question: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
        return_context: bool = False,
        include_citations: bool = True,
    ) -> Union[str, Dict[str, Any], CitedAnswer]:
        """Query the knowledge graph with optional inline citations."""
        self.md_logger.section("Query with Citations")
        self.md_logger.info(f"Question: {question}")
        self.md_logger.info(f"Citations enabled: {include_citations}")
        if document_id:
            self.md_logger.info(f"Document filter: {document_id}")

        try:
            # Hybrid search: vector + fulltext
            self.md_logger.subsection("Hybrid Search")
            chunks, citations = await self._hybrid_search(question, document_id, top_k)
            self.md_logger.info(f"Found {len(chunks)} relevant chunks")

            # Log retrieved chunks with citation numbers
            if chunks:
                self.md_logger.subsection("Retrieved Chunks with Citations")
                for i, chunk in enumerate(chunks):
                    cite = citations[i] if i < len(citations) else None
                    score = chunk.get("score", "N/A")
                    source = chunk.get("source", "unknown")
                    text_preview = chunk.get("text", "")[:200] + "..."
                    self.md_logger.info(f"[{i+1}] Chunk (score: {score}, source: {source}):")
                    self.md_logger.text_block(text_preview)

            # Get graph-enriched context using VectorCypher-style traversal
            self.md_logger.subsection("Graph-Enriched Context")
            graph_context = await self._get_graph_enriched_context(question, chunks, document_id)
            self.md_logger.info(f"Found {len(graph_context)} graph connections")

            if graph_context:
                for rel in graph_context[:15]:
                    self.md_logger.info(
                        f"  {rel.get('source', '?')} --[{rel.get('relationship', '?')}]--> {rel.get('target', '?')}"
                    )

            # Build context (with or without citation markers based on include_citations)
            context_with_citations = self._build_cited_context(
                chunks, citations, graph_context, include_citations
            )

            self.md_logger.subsection("Context for LLM")
            self.md_logger.text_block(
                context_with_citations[:2000] + "..."
                if len(context_with_citations) > 2000
                else context_with_citations
            )

            # Generate answer with citations
            if include_citations:
                answer = await self._generate_cited_answer(
                    question, context_with_citations, citations
                )
            else:
                answer = await self._generate_answer(question, context_with_citations)

            self.md_logger.subsection("Generated Answer")
            self.md_logger.text_block(answer)

            if return_context or include_citations:
                cited_answer = CitedAnswer(
                    answer=answer,
                    citations=citations,
                    chunks=chunks,
                    graph_context=graph_context,
                )
                return {
                    "answer": answer
                    + (cited_answer.format_citations() if include_citations else ""),
                    "chunks": chunks,
                    "graph_context": graph_context,
                    "citations": [
                        {
                            "id": c.id,
                            "chunk_id": c.chunk_id,
                            "text_preview": c.text_preview,
                            "score": c.score,
                            "source_type": c.source_type,
                            "document_title": c.document_title,
                        }
                        for c in citations
                    ],
                }

            return answer

        except Exception as e:
            logger.exception("Query failed")
            self.md_logger.error(f"Query failed: {e}")
            raise

    # =========================================================================
    # Hybrid Search (Vector + Fulltext)
    # =========================================================================

    async def _hybrid_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.7,  # Weight for vector search (0.7 = 70% vector, 30% fulltext)
    ) -> Tuple[List[Dict[str, Any]], List[Citation]]:
        """Perform hybrid search combining vector and fulltext search."""
        results = []
        seen_ids = set()

        # 1. Vector Search (primary) - request more results for better coverage
        vector_results = await self._vector_search(query, document_id, top_k * 2)
        self.md_logger.info(f"Vector search returned {len(vector_results)} results")

        for r in vector_results:
            r["search_type"] = "vector"
            r["weighted_score"] = r.get("score", 0) * alpha
            chunk_id = r.get("id")
            if chunk_id and chunk_id not in seen_ids:
                results.append(r)
                seen_ids.add(chunk_id)
                self.md_logger.info(
                    f"  Vector result: score={r.get('score', 0):.3f}, id={chunk_id}"
                )

        # 2. Fulltext Search (secondary)
        fulltext_results = await self._fulltext_search(query, document_id, top_k * 2)
        self.md_logger.info(f"Fulltext search returned {len(fulltext_results)} results")

        for r in fulltext_results:
            chunk_id = r.get("id")
            if chunk_id and chunk_id not in seen_ids:
                r["search_type"] = "fulltext"
                r["weighted_score"] = r.get("score", 0) * (1 - alpha)
                results.append(r)
                seen_ids.add(chunk_id)
                self.md_logger.info(
                    f"  Fulltext result (new): score={r.get('score', 0):.3f}, id={chunk_id}"
                )
            else:
                self.md_logger.info(f"  Fulltext result (duplicate): id={chunk_id}")

        # Sort by weighted score and take top_k
        results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        results = results[:top_k]

        self.md_logger.info(f"Hybrid search final results: {len(results)} chunks")

        # Create citations
        citations = []
        for i, chunk in enumerate(results, 1):
            text_preview = chunk.get("text", "")[:100]
            citations.append(
                Citation(
                    id=i,
                    chunk_id=chunk.get("id", f"chunk_{i}"),
                    document_id=chunk.get("document_id"),
                    document_title=chunk.get("document_title"),
                    text_preview=text_preview,
                    score=chunk.get("score", 0),
                    source_type=chunk.get("search_type", "unknown"),
                )
            )

        return results, citations

    async def _vector_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        try:
            query_embedding = self.embedder.embed_query(query)
            if not query_embedding:
                return []

            if document_id:
                cypher = """
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node, score
                WHERE node.document_id = $document_id
                OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
                RETURN node.text as text, elementId(node) as id, score, 
                       node.document_id as document_id, d.title as document_title,
                       'vector' as source
                ORDER BY score DESC
                """
                params = {
                    "index_name": self.config.vector_index_name,
                    "top_k": top_k * 2,
                    "embedding": query_embedding,
                    "document_id": document_id,
                }
            else:
                cypher = """
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node, score
                OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
                RETURN node.text as text, elementId(node) as id, score,
                       node.document_id as document_id, d.title as document_title,
                       'vector' as source
                ORDER BY score DESC
                """
                params = {
                    "index_name": self.config.vector_index_name,
                    "top_k": top_k,
                    "embedding": query_embedding,
                }

            return self.db.execute_query(cypher, params) or []

        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def _fulltext_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform fulltext search using the fulltext index."""
        try:
            # Ensure fulltext index exists
            self._ensure_fulltext_index()

            # Prepare query for fulltext search (escape special chars)
            search_query = re.sub(r"[^\w\s]", " ", query)
            search_terms = search_query.split()
            lucene_query = " OR ".join(search_terms)

            if document_id:
                cypher = """
                CALL db.index.fulltext.queryNodes($index_name, $query)
                YIELD node, score
                WHERE node.document_id = $document_id
                OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
                RETURN node.text as text, elementId(node) as id, score,
                       node.document_id as document_id, d.title as document_title,
                       'fulltext' as source
                ORDER BY score DESC
                LIMIT $top_k
                """
                params = {
                    "index_name": self._fulltext_index_name,
                    "query": lucene_query,
                    "document_id": document_id,
                    "top_k": top_k,
                }
            else:
                cypher = """
                CALL db.index.fulltext.queryNodes($index_name, $query)
                YIELD node, score
                OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
                RETURN node.text as text, elementId(node) as id, score,
                       node.document_id as document_id, d.title as document_title,
                       'fulltext' as source
                ORDER BY score DESC
                LIMIT $top_k
                """
                params = {
                    "index_name": self._fulltext_index_name,
                    "query": lucene_query,
                    "top_k": top_k,
                }

            return self.db.execute_query(cypher, params) or []

        except Exception as e:
            logger.warning(f"Fulltext search failed: {e}")
            return []

    # =========================================================================
    # Graph-Enriched Context (VectorCypher-style)
    # =========================================================================

    async def _get_graph_enriched_context(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get graph context enriched with entity relationships from chunks."""
        # Extract entities mentioned in chunks
        chunk_entities = await self._extract_entities_from_chunks(chunks)

        # Also extract entities from the question
        question_entities = await self._extract_question_entities(question)

        # Combine and deduplicate
        all_entities = list(set(chunk_entities + question_entities))

        if not all_entities:
            return []

        # Graph traversal query (VectorCypher-style)
        cypher = """
        MATCH (e:__Entity__)-[r]-(other:__Entity__)
        WHERE any(entity IN $entities WHERE 
            toLower(e.name) CONTAINS toLower(entity) OR 
            toLower(other.name) CONTAINS toLower(entity) OR
            toLower(entity) CONTAINS toLower(e.name) OR
            toLower(entity) CONTAINS toLower(other.name)
        )
        RETURN DISTINCT 
            e.name as source, 
            type(r) as relationship,
            other.name as target, 
            r.description as description,
            labels(e) as source_labels,
            labels(other) as target_labels
        LIMIT $limit
        """

        results = self.db.execute_query(cypher, {"entities": all_entities, "limit": limit})
        return results or []

    async def _extract_entities_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract entity names mentioned in the retrieved chunks."""
        if not chunks:
            return []

        # Get all entities from the graph
        entity_query = """
        MATCH (e:__Entity__)
        RETURN e.name as name
        LIMIT 500
        """
        all_entities = self.db.execute_query(entity_query, {})
        if not all_entities:
            return []

        entity_names = [e["name"].lower() for e in all_entities if e.get("name")]

        # Find entities mentioned in chunks
        mentioned = []
        chunk_text = " ".join([c.get("text", "").lower() for c in chunks])

        for entity in all_entities:
            name = entity.get("name", "")
            if name and name.lower() in chunk_text:
                mentioned.append(name)

        return mentioned[:20]  # Limit to avoid query explosion

    async def _extract_question_entities(self, question: str) -> List[str]:
        """Extract entity names from a question using LLM."""
        prompt = f"""Extract key terms from this question that could match entities in a knowledge graph.
Include: company names, technologies, concepts, components, services.
Return ONLY a JSON list of strings.

Question: {question}

JSON list:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            if content.endswith("```"):
                content = content[:-3]

            entities = json.loads(content.strip())
            return [e for e in entities if isinstance(e, str) and len(e) >= 2][:15]

        except Exception as e:
            logger.warning(f"Failed to extract entities: {e}")
            return []

    # =========================================================================
    # Context Building with Citations
    # =========================================================================

    def _build_cited_context(
        self,
        chunks: List[Dict[str, Any]],
        citations: List[Citation],
        graph_context: List[Dict[str, Any]],
        include_citations: bool = True,
    ) -> str:
        """Build context string, optionally with citation markers."""
        sections = []

        # Text content (with or without citation markers)
        sections.append("## Retrieved Text Passages\n")
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if include_citations:
                citation_id = i + 1
                sections.append(f"[{citation_id}] {text}\n")
            else:
                sections.append(f"{text}\n")

        # Graph relationships
        if graph_context:
            sections.append("\n## Knowledge Graph Relationships\n")
            for rel in graph_context:
                src = rel.get("source", "?")
                rel_type = rel.get("relationship", "?")
                tgt = rel.get("target", "?")
                sections.append(f"- {src} --[{rel_type}]--> {tgt}")

        return "\n".join(sections)

    # =========================================================================
    # Answer Generation
    # =========================================================================

    async def _generate_cited_answer(
        self,
        question: str,
        context: str,
        citations: List[Citation],
    ) -> str:
        """Generate an answer with inline citations."""
        citation_instructions = "\n".join(
            [f"[{c.id}] = Source about: {c.text_preview[:50]}..." for c in citations]
        )

        prompt = f"""You are a helpful assistant. Answer the question using the provided context.

IMPORTANT: Include inline citations using [1], [2], etc. to reference the sources.
Each citation number corresponds to a specific source passage.

Available Sources:
{citation_instructions}

Context:
{context}

Question: {question}

Answer (include [1], [2], etc. citations where you use information from sources):"""

        response = self.query_llm.invoke(prompt)
        return response.content

    async def _generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate a simple answer without citations."""
        prompt = f"""You are a helpful assistant. Answer the question using the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

        response = self.query_llm.invoke(prompt)
        return response.content

    # =========================================================================
    # Cleanup
    # =========================================================================

    def close(self):
        """Close the pipeline and cleanup resources."""
        if hasattr(self, "_md_handler"):
            neo4j_logger = logging.getLogger("neo4j_graphrag")
            neo4j_logger.removeHandler(self._md_handler)

        self.md_logger.finalize()
        self.db.close()
        logger.info("Pipeline closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# =============================================================================
# Utility Functions
# =============================================================================


def run_async(coro):
    """Run an async coroutine."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
