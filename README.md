# Neo4j GraphRAG

A production-ready Knowledge Graph RAG (Retrieval-Augmented Generation) solution using Neo4j and Google Gemini.

## Features

- **Knowledge Graph Construction**: Automatically extract entities and relationships from text documents
- **Entity Resolution**: Merge duplicate entities using exact, fuzzy, or semantic matching
- **Vector Search**: Semantic similarity search using embeddings
- **Graph-Enhanced RAG**: Combine vector search with graph traversal for better context
- **Production Ready**: Proper constraints, indexes, and error handling

## Architecture

This solution follows the official Neo4j GraphRAG patterns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Ingestion                        │
├─────────────────────────────────────────────────────────────┤
│  Text → Chunks → Entity Extraction → Graph Building         │
│                         ↓                                    │
│              Entity Resolution (Deduplication)               │
│                         ↓                                    │
│              Vector Index + Graph Storage                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        Query                                 │
├─────────────────────────────────────────────────────────────┤
│  Question → Vector Search + Graph Traversal → LLM Answer    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.12+
- Neo4j AuraDB or local Neo4j instance (5.18+)
- Google Gemini API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/behoss/neo4j-graphrag.git
cd neo4j-graphrag
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. Create a `.env.local` file:
```bash
cp .env.example .env.local
```

4. Edit `.env.local` with your credentials:
```env
# Neo4j AuraDB credentials
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Google Gemini API key
GEMINI_API_KEY=your-api-key
```

5. Initialize the database:
```bash
python -m src.cli init
```

## Usage

### Command Line Interface

The CLI provides several commands for managing your knowledge graph:

```bash
# Initialize database with constraints and indexes
python -m src.cli init

# Ingest a document
python -m src.cli ingest files/document.txt --title "My Document"

# List all documents
python -m src.cli list

# Query the knowledge graph
python -m src.cli query

# Show document info
python -m src.cli info my-document

# Delete a document
python -m src.cli delete my-document

# Show database statistics
python -m src.cli stats

# Purge all data (use with caution!)
python -m src.cli purge
```

### Python API

```python
import asyncio
from src.config import PipelineConfig
from src.pipeline import GraphRAGPipeline

# Load configuration from environment
config = PipelineConfig.from_env()

# Create pipeline
with GraphRAGPipeline(config) as pipeline:
    # Initialize database
    pipeline.db.initialize()
    
    # Ingest a document
    result = asyncio.run(pipeline.ingest_text(
        text="Your document text here...",
        document_id="my-doc",
        document_title="My Document",
    ))
    
    print(f"Ingested: {result['entities']} entities, {result['relationships']} relationships")
    
    # Query the knowledge graph
    answer = asyncio.run(pipeline.query("What is this document about?"))
    print(f"Answer: {answer}")
```

### Demo

Run the demo script to see the pipeline in action:

```bash
python demo.py
```

## Configuration

### Schema Configuration

The default schema includes common entity types and relationships. You can customize it in `src/config.py`:

```python
from src.config import SchemaConfig

schema = SchemaConfig(
    node_types=[
        {"label": "Person", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        {"label": "Organization", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        # Add more entity types...
    ],
    relationship_types=[
        {"label": "WORKS_FOR", "description": "Person works for organization"},
        # Add more relationship types...
    ],
    patterns=[
        ("Person", "WORKS_FOR", "Organization"),
        # Add more patterns...
    ],
)
```

### Entity Resolution

Three resolution strategies are available:

1. **Exact Match** (default): Merges entities with identical names
2. **Fuzzy Match**: Uses Levenshtein distance for similar names (requires `rapidfuzz`)
3. **Semantic Match**: Uses embeddings for semantic similarity (requires `spacy`)

Configure in your pipeline:

```python
config = PipelineConfig.from_env()
config.resolution_type = "fuzzy"  # or "exact" or "semantic"
```

## Project Structure

```
neo4j-graphrag/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── database.py         # Neo4j database operations
│   ├── pipeline.py         # Main GraphRAG pipeline
│   ├── logging_config.py   # Logging utilities
│   └── cli.py              # Command-line interface
├── files/                  # Document storage
├── logs/                   # Execution logs
├── cache/                  # Extraction cache
├── demo.py                 # Demo script
├── pyproject.toml          # Project dependencies
└── README.md
```

## Key Improvements Over POC

This production version includes several improvements:

1. **Entity Deduplication**: MERGE queries now match only on entity name, not document_id
2. **Proper Constraints**: Uniqueness constraints prevent duplicate entities
3. **Entity Resolution**: Post-processing step to merge similar entities
4. **Modular Architecture**: Clean separation of concerns
5. **Error Handling**: Graceful fallbacks and detailed logging
6. **Vector Search**: Proper vector index setup and querying
7. **CLI Tools**: Rich command-line interface for all operations

## Troubleshooting

### Duplicate Entities

If you see duplicate entities in your graph:

1. Run entity resolution manually:
```python
from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver

resolver = SinglePropertyExactMatchResolver(driver)
await resolver.run()
```

2. Or use the fuzzy matcher for similar names:
```python
from neo4j_graphrag.experimental.components.resolver import FuzzyMatchResolver

resolver = FuzzyMatchResolver(driver)
await resolver.run()
```

### Connection Issues

Ensure your Neo4j credentials are correct and the database is accessible:

```bash
python -c "from src.config import PipelineConfig; c = PipelineConfig.from_env(); print('OK')"
```

## License

MIT License

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.
