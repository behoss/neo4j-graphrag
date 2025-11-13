# Neo4j GraphRAG with Google Gemini

A document-scoped GraphRAG implementation using Neo4j Aura DB and Google's Gemini AI for intelligent document querying with knowledge graph construction.

## Features

- 🤖 **Automatic Entity Extraction**: Uses Gemini to automatically extract entities and relationships from documents
- 📚 **Document-Scoped Queries**: Each document is isolated - queries only search within the selected document
- 🔍 **Vector + Graph Search**: Combines semantic search with graph traversal for comprehensive answers
- 📊 **Full Logging**: Complete visibility into the extraction and query process
- 🛠️ **Document Management**: Tools to list, inspect, and delete documents

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Neo4j Database                    │
├─────────────────────────────────────────────────────┤
│  Entities (Person, Company, Product, etc.)         │
│  ├─ document_id: "moby-dick"                       │
│  ├─ document_title: "Moby Dick"                    │
│  └─ name, description, type                        │
│                                                      │
│  Chunks (for vector search)                        │
│  ├─ document_id: "moby-dick"                       │
│  ├─ document_title: "Moby Dick"                    │
│  └─ text, embedding                                │
│                                                      │
│  Relationships (connects entities)                 │
│  └─ type, description                              │
└─────────────────────────────────────────────────────┘
```

## Setup

### 1. Install Dependencies

Using UV (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env.local`:
```bash
# Neo4j Aura DB credentials
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Google Gemini API key
GEMINI_API_KEY=your-gemini-api-key
```

## Usage

### 1. Ingest Documents

Add documents to the knowledge graph (one-time, expensive process):

```bash
# Basic ingestion
python ingest.py files/moby_dick.txt "Moby Dick"

# With overwrite (re-ingest existing document)
python ingest.py files/1984.txt "Nineteen Eighty-Four" --overwrite
```

**What happens during ingestion:**
- Text is split into chunks (500 chars, 100 overlap)
- Each chunk is sent to Gemini for entity/relationship extraction
- Entities and relationships are stored in Neo4j with `document_id` tags
- Embeddings are created for semantic search
- All steps are logged to `logs/graphrag_run_*.md`

### 2. Query Documents

Interactive query session with document selection:

```bash
python query.py
```

**Interactive workflow:**
```
============================================================
📚 Available Documents
============================================================

1. Moby Dick
   ID: moby-dick
   Entities: 543, Chunks: 1247

2. Nineteen Eighty-Four
   ID: 1984
   Entities: 321, Chunks: 892

Select document (1-2): 1

============================================================
📚 Querying Document: Moby Dick
============================================================

📊 Document Statistics:
  • Entities: 543
  • Relationships: 234
  • Chunks: 1247

============================================================
GraphRAG is ready! Ask questions about 'Moby Dick'
Type 'exit' to quit
============================================================

❓ Your question: Who is Captain Ahab?
✅ Answer: [AI-generated response based on the document]
```

### 3. Manage Documents

List all documents:
```bash
python doc_manager.py --list
```

Get document details:
```bash
python doc_manager.py --info moby-dick
```

Delete a document:
```bash
python doc_manager.py --delete moby-dick
```

## File Structure

```
neo4j-graphrag/
├── main.py              # Core GraphRAGPipeline class
├── ingest.py            # Document ingestion CLI
├── query.py             # Document query CLI
├── doc_manager.py       # Document management CLI
├── demo.py              # Example/demo script
├── files/               # Place your .txt documents here
│   ├── document1.txt
│   └── document2.txt
└── logs/                # Auto-generated logs
    └── graphrag_run_*.md
```

## How It Works

### GraphRAG Query Process

When you ask a question, the system:

1. **Vector Search**: Finds the 3 most relevant text chunks using semantic similarity (filtered by `document_id`)
2. **Graph Traversal**: Queries the knowledge graph for connected entities and relationships (filtered by `document_id`)
3. **Context Combination**: Merges text chunks with graph connections
4. **LLM Generation**: Sends combined context to Gemini for answer generation

### Document Isolation

- Each document has a unique `document_id` (normalized from title: lowercase, hyphenated)
- All entities, relationships, and chunks are tagged with `document_id`
- Queries filter by `document_id` ensuring no cross-document contamination
- Multiple documents coexist in one Neo4j database

## Examples

### Example 1: Ingest a Book

```bash
# Place your book.txt in the files/ directory
python ingest.py files/pride_and_prejudice.txt "Pride and Prejudice"
```

Output:
```
📚 Ingesting Document
============================================================
Title: Pride and Prejudice
ID: pride-and-prejudice
File: files/pride_and_prejudice.txt
============================================================

📄 Document size: 725000 characters

🔄 Processing document...
✂️ Splitting text into chunks...
✓ Created 1450 chunks

🔄 Processing chunk 1/1450...
[... extraction process ...]

✅ Document ingested successfully!
============================================================
📊 Statistics:
  • Entities: 892
  • Relationships: 456
  • Chunks: 1450

📝 Log file: logs/graphrag_run_20251113_120000.md
```

### Example 2: Query a Document

```bash
python query.py
# Select document 1 (Pride and Prejudice)
# Ask: "What is the relationship between Elizabeth and Mr. Darcy?"
```

### Example 3: Manage Documents

```bash
# List all documents
python doc_manager.py --list

# Get details about a specific document
python doc_manager.py --info pride-and-prejudice

# Delete a document
python doc_manager.py --delete pride-and-prejudice
```

## Logging

Every operation is logged to `logs/graphrag_run_*.md` with:
- Full input text
- LLM prompts and responses
- Extracted entities and relationships (as JSON)
- Cypher queries executed
- Vector search results
- Final answers

## Customization

### Adjust Chunk Size

Edit `ingest.py` or `query.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Increase for larger chunks
    chunk_overlap=100  # Adjust overlap
)
```

### Change Entity Types

Edit the prompt in `main.py` → `extract_entities_and_relationships()`:
```python
"type": "Person|Company|Product|Technology|Location|YourCustomType"
```

### Modify LLM Model

Edit `main.py` → `_initialize_components()`:
```python
llm_config = {
    "model": "gemini-2.5-flash",
    "temperature": 0,
    "max_retries": 2
}
```

## Troubleshooting

### "No documents found"
- Run `python ingest.py` first to add documents
- Check `.env.local` credentials are correct

### "No relevant information found"
- The question might not relate to the document content
- Try rephrasing the question
- Check if the document was ingested successfully

### Connection errors
- Verify Neo4j Aura DB is running
- Check `NEO4J_URI` format: `neo4j+s://xxxxx.databases.neo4j.io`
- Confirm credentials in `.env.local`

### Rate limiting
- Gemini has rate limits - add delays between chunks if needed
- Consider using a higher tier API key for production

## Contributing

Feel free to open issues or submit PRs!

## License

MIT
