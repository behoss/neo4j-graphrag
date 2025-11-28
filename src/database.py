"""Database management for Neo4j GraphRAG using modern driver best practices."""

import logging
from typing import Any, Dict, List, Optional, cast

from neo4j import GraphDatabase, Driver, RoutingControl
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from .config import Neo4jConfig

# Type alias for Neo4j query strings (to satisfy LiteralString requirements)
QueryStr = str

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages Neo4j database connections using modern driver best practices.

    Key improvements over previous version:
    - Uses driver.execute_query() for automatic retry and routing
    - Proper read/write routing for cluster efficiency
    - GQL error handling for granular error management
    - Result consumption for write operations
    - Parameterized queries throughout (no string interpolation)
    """

    def __init__(self, config: Neo4jConfig):
        """Initialize database manager with configuration."""
        self.config = config
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver (lazy initialization, reuse same instance)."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
            )
            # Verify connectivity immediately
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.config.uri}")
        return self._driver

    def close(self):
        """Close the database connection and release resources."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always close connections."""
        self.close()

    def execute_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a read query with automatic retry and read routing.

        Uses driver.execute_query() with routing_=RoutingControl.READ for:
        - Automatic retry on transient failures
        - Routing to read replicas in a cluster
        - Proper result handling
        """
        try:
            records, summary, keys = self.driver.execute_query(
                query,  # type: ignore[arg-type]  # Neo4j driver accepts str at runtime
                parameters_=parameters or {},
                database_=self.config.database,
                routing_=RoutingControl.READ,
            )
            return [record.data() for record in records]
        except Neo4jError as e:
            self._handle_error(e, query)
            raise

    def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a write query with automatic retry.

        Uses driver.execute_query() with routing_=RoutingControl.WRITE for:
        - Automatic retry on transient failures
        - Routing to the leader node in a cluster
        - Proper result consumption

        Returns the query summary counters.
        """
        try:
            records, summary, keys = self.driver.execute_query(
                query,  # type: ignore[arg-type]  # Neo4j driver accepts str at runtime
                parameters_=parameters or {},
                database_=self.config.database,
                routing_=RoutingControl.WRITE,
            )
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
                "labels_added": summary.counters.labels_added,
                "labels_removed": summary.counters.labels_removed,
                "indexes_added": summary.counters.indexes_added,
                "indexes_removed": summary.counters.indexes_removed,
                "constraints_added": summary.counters.constraints_added,
                "constraints_removed": summary.counters.constraints_removed,
            }
        except Neo4jError as e:
            self._handle_error(e, query)
            raise

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a query (backwards compatible method, defaults to read routing)."""
        return self.execute_read(query, parameters)

    def _handle_error(self, error: Neo4jError, query: str):
        """Handle Neo4j errors with GQL status codes for granular error management."""
        # Log the error with full context
        logger.error(f"Neo4j error executing query: {error.message}")
        logger.error(f"Error code: {error.code}")
        logger.error(f"GQL status: {getattr(error, 'gql_status', 'N/A')}")
        logger.debug(f"Query: {query}")

        # Check for specific error types using GQL status
        if hasattr(error, "find_by_gql_status"):
            if error.find_by_gql_status("42001"):
                logger.error("Syntax error in Cypher query")
            elif error.find_by_gql_status("42NFF"):
                logger.error("Permission denied - check user privileges")
            elif error.find_by_gql_status("42N51"):
                logger.error("Schema constraint violation")

    def setup_constraints(self) -> None:
        """Set up database constraints using proper parameterized queries."""
        logger.info("Setting up database constraints...")

        # NOTE: Do NOT add entity_name_unique constraint!
        # The neo4j-graphrag library creates duplicate entities during extraction
        # and then merges them with Entity Resolution.
        constraints = [
            {
                "name": "document_path_unique",
                "query": """
                    CREATE CONSTRAINT document_path_unique IF NOT EXISTS
                    FOR (d:Document)
                    REQUIRE d.path IS UNIQUE
                """,
            },
            {
                "name": "chunk_id_unique",
                "query": """
                    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                    FOR (c:Chunk)
                    REQUIRE c.id IS UNIQUE
                """,
            },
        ]

        for constraint in constraints:
            try:
                result = self.execute_write(constraint["query"])
                if result.get("constraints_added", 0) > 0:
                    logger.info(f"Created constraint: {constraint['name']}")
                else:
                    logger.debug(f"Constraint already exists: {constraint['name']}")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not create constraint {constraint['name']}: {e}")

        logger.info("Database constraints setup complete")

    def setup_indexes(self) -> None:
        """Set up database indexes for performance."""
        logger.info("Setting up database indexes...")

        indexes = [
            {
                "name": "entity_name_index",
                "query": """
                    CREATE INDEX entity_name_index IF NOT EXISTS
                    FOR (e:__Entity__)
                    ON (e.name)
                """,
            },
            {
                "name": "document_id_index",
                "query": """
                    CREATE INDEX document_id_index IF NOT EXISTS
                    FOR (d:Document)
                    ON (d.id)
                """,
            },
            {
                "name": "chunk_document_index",
                "query": """
                    CREATE INDEX chunk_document_index IF NOT EXISTS
                    FOR (c:Chunk)
                    ON (c.document_id)
                """,
            },
        ]

        for index in indexes:
            try:
                result = self.execute_write(index["query"])
                if result.get("indexes_added", 0) > 0:
                    logger.info(f"Created index: {index['name']}")
                else:
                    logger.debug(f"Index already exists: {index['name']}")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not create index {index['name']}: {e}")

        logger.info("Database indexes setup complete")

    def setup_vector_index(
        self,
        index_name: str = "vector_index",
        dimensions: int = 3072,
        similarity_function: str = "cosine",
    ) -> None:
        """Set up vector index for semantic search with correct dimensions."""
        logger.info(f"Setting up vector index '{index_name}' with {dimensions} dimensions...")

        # Check if index exists and has correct dimensions
        try:
            existing = self.execute_read(
                "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name",
                {"name": index_name},
            )
            if existing:
                existing_dims = (
                    existing[0].get("options", {}).get("indexConfig", {}).get("vector.dimensions")
                )
                if existing_dims == dimensions:
                    logger.info(
                        f"Vector index '{index_name}' already exists with correct dimensions ({dimensions})"
                    )
                    return
                else:
                    logger.warning(
                        f"Vector index '{index_name}' has wrong dimensions ({existing_dims}), recreating..."
                    )
                    # Drop using parameterized approach via APOC or direct call
                    # Note: DROP INDEX doesn't support parameters, so we validate the name
                    allowed_index_names = {"vector_index", "chunk_embedding_index"}
                    if index_name in allowed_index_names:
                        self.execute_write(f"DROP INDEX {index_name}")
                    else:
                        raise ValueError(f"Index name '{index_name}' not in allowed list")
        except Neo4jError as e:
            logger.debug(f"Could not check existing index: {e}")

        # Create vector index - dimensions must be literal in CREATE INDEX
        # This is a Neo4j limitation, so we validate the input instead
        if not isinstance(dimensions, int) or dimensions <= 0 or dimensions > 10000:
            raise ValueError(f"Invalid dimensions value: {dimensions}")
        if similarity_function not in ("cosine", "euclidean"):
            raise ValueError(f"Invalid similarity function: {similarity_function}")

        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (c:Chunk)
        ON c.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: '{similarity_function}'
            }}
        }}
        """

        try:
            result = self.execute_write(query)
            if result.get("indexes_added", 0) > 0:
                logger.info(f"Vector index '{index_name}' created with {dimensions} dimensions")
            else:
                logger.debug(f"Vector index '{index_name}' already exists")
        except Neo4jError as e:
            logger.warning(f"Could not create vector index: {e}")

    def initialize(self) -> None:
        """Initialize database with all required constraints and indexes."""
        self.setup_constraints()
        self.setup_indexes()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics using read routing."""
        stats: Dict[str, Any] = {}

        # Count nodes by label
        node_results = self.execute_read(
            """
            MATCH (n)
            WITH labels(n) as labels, count(*) as count
            UNWIND labels as label
            RETURN label, sum(count) as count
            ORDER BY count DESC
        """
        )
        stats["nodes_by_label"] = {r["label"]: r["count"] for r in node_results}

        # Count relationships by type
        rel_results = self.execute_read(
            """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            ORDER BY count DESC
        """
        )
        stats["relationships_by_type"] = {r["type"]: r["count"] for r in rel_results}

        # Total counts
        total_results = self.execute_read(
            """
            MATCH (n)
            WITH count(n) as nodes
            OPTIONAL MATCH ()-[r]->()
            RETURN nodes, count(r) as relationships
        """
        )
        if total_results:
            stats["total_nodes"] = total_results[0]["nodes"]
            stats["total_relationships"] = total_results[0]["relationships"]

        return stats

    def clear_all(self) -> Dict[str, Any]:
        """Clear all data from the database. USE WITH CAUTION!

        Returns the number of nodes and relationships deleted.
        """
        logger.warning("Clearing all data from database...")
        result = self.execute_write("MATCH (n) DETACH DELETE n")
        logger.info(
            f"Database cleared: {result.get('nodes_deleted', 0)} nodes, "
            f"{result.get('relationships_deleted', 0)} relationships deleted"
        )
        return result

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document and all its related entities."""
        logger.info(f"Deleting document: {document_id}")

        query = """
        MATCH (d:Document {id: $document_id})
        OPTIONAL MATCH (d)<-[:PART_OF_DOCUMENT]-(c:Chunk)
        OPTIONAL MATCH (c)<-[:PART_OF_CHUNK]-(e)
        DETACH DELETE d, c, e
        """
        result = self.execute_write(query, {"document_id": document_id})
        logger.info(
            f"Deleted document {document_id}: " f"{result.get('nodes_deleted', 0)} nodes removed"
        )
        return result

    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the database with their statistics."""
        # Query for documents
        docs = self.execute_read(
            """
            MATCH (d:Document)
            WHERE d.id IS NOT NULL AND d.id <> ''
            RETURN d.id as id, d.title as title
            ORDER BY d.title
        """
        )

        # Get total chunk count (neo4j-graphrag doesn't link chunks to documents)
        chunk_result = self.execute_read("MATCH (c:Chunk) RETURN count(c) as count")
        total_chunks = chunk_result[0]["count"] if chunk_result else 0

        # Get total entity count
        entity_result = self.execute_read("MATCH (e:__Entity__) RETURN count(e) as count")
        total_entities = entity_result[0]["count"] if entity_result else 0

        # Assign totals to each document
        for doc in docs:
            doc["chunk_count"] = total_chunks
            doc["entity_count"] = total_entities

        return docs

    def get_document_entities(self, document_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get entities for a specific document."""
        return self.execute_read(
            """
            MATCH (e:__Entity__)
            WHERE $document_id IN e.source_documents
            RETURN e.name as name, labels(e) as labels, e.description as description
            ORDER BY e.name
            LIMIT $limit
            """,
            {"document_id": document_id, "limit": limit},
        )

    def get_entity_relationships(self, entity_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get relationships for a specific entity."""
        return self.execute_read(
            """
            MATCH (e:__Entity__ {name: $entity_name})-[r]-(other:__Entity__)
            RETURN e.name as source, type(r) as relationship,
                   other.name as target, r.description as description
            LIMIT $limit
            """,
            {"entity_name": entity_name, "limit": limit},
        )

    def find_similar_entities(
        self, entity_name: str, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find entities with similar names (for entity resolution)."""
        return self.execute_read(
            """
            MATCH (e:__Entity__)
            WHERE toLower(e.name) CONTAINS toLower($name)
               OR toLower($name) CONTAINS toLower(e.name)
            RETURN e.name as name, labels(e) as labels
            ORDER BY e.name
            LIMIT 20
            """,
            {"name": entity_name},
        )

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database connection."""
        try:
            self.driver.verify_connectivity()
            result = self.execute_read("RETURN 1 as status")
            return {
                "status": "healthy",
                "connected": True,
                "uri": self.config.uri,
                "database": self.config.database,
            }
        except (ServiceUnavailable, SessionExpired) as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "uri": self.config.uri,
            }
