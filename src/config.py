"""Configuration management for GraphRAG pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""

    uri: str
    username: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Create configuration from environment variables."""
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        if not all([uri, username, password]):
            raise ValueError(
                "Missing required Neo4j environment variables: "
                "NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
            )

        return cls(
            uri=str(uri),
            username=str(username),
            password=str(password),
            database=database,
        )


@dataclass
class LLMConfig:
    """LLM configuration for entity extraction and querying."""

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 16000
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls, model_name: str = "gemini-2.5-flash") -> "LLMConfig":
        """Create configuration from environment variables."""
        # API key is optional for Vertex AI (uses ADC instead)
        api_key = os.getenv("GOOGLE_API_KEY")
        return cls(model_name=model_name, api_key=api_key)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    # ⚠️ DO NOT CHANGE: gemini-embedding-001 is required for all embeddings
    model_name: str = "gemini-embedding-001"
    dimensions: int = 3072  # gemini-embedding-001 outputs 3072 dimensions
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls, model_name: str = "gemini-embedding-001") -> "EmbeddingConfig":
        """Create configuration from environment variables."""
        # API key is optional for Vertex AI (uses ADC instead)
        api_key = os.getenv("GOOGLE_API_KEY")
        return cls(model_name=model_name, api_key=api_key)


@dataclass
class SchemaConfig:
    """Knowledge graph schema configuration."""

    node_types: list = field(
        default_factory=lambda: [
            {
                "label": "Person",
                "description": "A human being",
                "properties": [
                    {"name": "name", "type": "STRING", "required": True},
                    {"name": "description", "type": "STRING"},
                ],
            },
            {
                "label": "Organization",
                "description": "A company, institution, or group",
                "properties": [
                    {"name": "name", "type": "STRING", "required": True},
                    {"name": "description", "type": "STRING"},
                    {"name": "industry", "type": "STRING"},
                ],
            },
            {
                "label": "Technology",
                "description": "A technology, tool, or software",
                "properties": [
                    {"name": "name", "type": "STRING", "required": True},
                    {"name": "description", "type": "STRING"},
                    {"name": "category", "type": "STRING"},
                ],
            },
            {
                "label": "Product",
                "description": "A product or service",
                "properties": [
                    {"name": "name", "type": "STRING", "required": True},
                    {"name": "description", "type": "STRING"},
                ],
            },
            {
                "label": "Concept",
                "description": "An abstract concept or idea",
                "properties": [
                    {"name": "name", "type": "STRING", "required": True},
                    {"name": "description", "type": "STRING"},
                ],
            },
            {
                "label": "Location",
                "description": "A geographical location",
                "properties": [
                    {"name": "name", "type": "STRING", "required": True},
                    {"name": "description", "type": "STRING"},
                ],
            },
            {
                "label": "Event",
                "description": "An event or occurrence",
                "properties": [
                    {"name": "name", "type": "STRING", "required": True},
                    {"name": "description", "type": "STRING"},
                    {"name": "date", "type": "STRING"},
                ],
            },
        ]
    )

    relationship_types: list = field(
        default_factory=lambda: [
            {"label": "WORKS_FOR", "description": "Person works for an organization"},
            {"label": "FOUNDED", "description": "Person or organization founded another entity"},
            {"label": "CREATED", "description": "Entity created another entity"},
            {"label": "USES", "description": "Entity uses another entity"},
            {"label": "RELATED_TO", "description": "General relationship between entities"},
            {"label": "PART_OF", "description": "Entity is part of another entity"},
            {"label": "LOCATED_IN", "description": "Entity is located in a location"},
            {"label": "OWNS", "description": "Entity owns another entity"},
            {"label": "MANAGES", "description": "Entity manages another entity"},
            {
                "label": "COLLABORATES_WITH",
                "description": "Entity collaborates with another entity",
            },
            {"label": "COMPETES_WITH", "description": "Entity competes with another entity"},
            {"label": "ACQUIRED", "description": "Entity acquired another entity"},
            {"label": "INVESTED_IN", "description": "Entity invested in another entity"},
            {"label": "DEVELOPED", "description": "Entity developed another entity"},
            {"label": "IMPLEMENTS", "description": "Entity implements another entity"},
        ]
    )

    patterns: list = field(
        default_factory=lambda: [
            ("Person", "WORKS_FOR", "Organization"),
            ("Person", "FOUNDED", "Organization"),
            ("Person", "CREATED", "Technology"),
            ("Person", "CREATED", "Product"),
            ("Organization", "CREATED", "Technology"),
            ("Organization", "CREATED", "Product"),
            ("Organization", "USES", "Technology"),
            ("Organization", "LOCATED_IN", "Location"),
            ("Organization", "OWNS", "Product"),
            ("Organization", "ACQUIRED", "Organization"),
            ("Organization", "COMPETES_WITH", "Organization"),
            ("Organization", "COLLABORATES_WITH", "Organization"),
            ("Technology", "RELATED_TO", "Technology"),
            ("Technology", "PART_OF", "Technology"),
            ("Technology", "IMPLEMENTS", "Concept"),
            ("Product", "USES", "Technology"),
            ("Product", "RELATED_TO", "Product"),
            ("Concept", "RELATED_TO", "Concept"),
            ("Event", "LOCATED_IN", "Location"),
            ("Person", "RELATED_TO", "Event"),
            ("Organization", "RELATED_TO", "Event"),
        ]
    )

    # Schema enforcement options
    additional_node_types: bool = True
    additional_relationship_types: bool = True
    additional_patterns: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    neo4j: Neo4jConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    schema: SchemaConfig = field(default_factory=SchemaConfig)

    # Processing options
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Entity resolution
    perform_entity_resolution: bool = True
    resolution_type: str = "exact"  # "exact", "fuzzy", or "semantic"

    # Logging
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))

    # Vector index
    vector_index_name: str = "vector_index"
    # ⚠️ DO NOT CHANGE: Must match gemini-embedding-001 output dimensions (3072)
    vector_dimensions: int = 3072

    @classmethod
    def from_env(cls, env_file: str = ".env.local") -> "PipelineConfig":
        """Create configuration from environment variables."""
        load_dotenv(env_file)

        return cls(
            neo4j=Neo4jConfig.from_env(),
            llm=LLMConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
        )

    def __post_init__(self):
        """Ensure directories exist."""
        self.log_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
