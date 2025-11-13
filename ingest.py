"""
Document Ingestion CLI - Ingest documents into Neo4j GraphRAG
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from main import GraphRAGPipeline

load_dotenv(".env.local")


def normalize_document_id(title: str) -> str:
    """Convert document title to normalized ID (lowercase, hyphenated)"""
    # Remove special characters, replace spaces with hyphens, lowercase
    normalized = re.sub(r"[^\w\s-]", "", title.lower())
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")


def ingest_document(file_path: str, document_title: str, overwrite: bool = False):
    """Ingest a document into the knowledge graph"""

    # Validate file
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)

    if not file_path.endswith(".txt"):
        print(f"❌ Error: Only .txt files are supported")
        sys.exit(1)

    # Generate document ID
    document_id = normalize_document_id(document_title)

    print(f"\n📚 Ingesting Document")
    print(f"{'='*60}")
    print(f"Title: {document_title}")
    print(f"ID: {document_id}")
    print(f"File: {file_path}")
    print(f"{'='*60}\n")

    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"📄 Document size: {len(text)} characters\n")

    # Get credentials
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not all([neo4j_uri, neo4j_username, neo4j_password, gemini_api_key]):
        print("❌ Error: Missing required environment variables")
        sys.exit(1)

    # Initialize pipeline
    pipeline = GraphRAGPipeline(
        neo4j_uri=str(neo4j_uri),
        neo4j_username=str(neo4j_username),
        neo4j_password=str(neo4j_password),
        gemini_api_key=str(gemini_api_key),
    )

    # Check if document already exists
    if not overwrite:
        check_query = "MATCH (n {document_id: $document_id}) RETURN count(n) as count"
        result = pipeline.graph.query(check_query, params={"document_id": document_id})
        if result and result[0]["count"] > 0:
            print(
                f"⚠️  Warning: Document '{document_title}' (ID: {document_id}) already exists!"
            )
            response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
            if response not in ["yes", "y"]:
                print("❌ Ingestion cancelled")
                sys.exit(0)
            overwrite = True

    # Delete existing document if overwriting
    if overwrite:
        print(f"\n🗑️  Removing existing document '{document_id}'...")
        delete_query = "MATCH (n {document_id: $document_id}) DETACH DELETE n"
        pipeline.graph.query(delete_query, params={"document_id": document_id})
        print("✓ Existing document removed\n")

    # Process document
    print("🔄 Processing document...\n")

    # Update process_text to accept document metadata
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    pipeline.log_section(f"INGESTING DOCUMENT: {document_title}")
    pipeline.log(f"Document ID: {document_id}")
    pipeline.log(f"File: {file_path}")

    pipeline.log("📝 Full Input Text:")
    pipeline.log_code_block(text, "text")

    pipeline.log("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    pipeline.log(f"✓ Created {len(chunks)} chunks\n")

    all_entities = []
    all_relationships = []

    # TRUE concurrent batch processing (Tier 2: 1,000 RPM)
    import asyncio
    import time
    from concurrent.futures import ThreadPoolExecutor

    BATCH_SIZE = 100  # Process 100 chunks concurrently (Tier 2 can handle 1,000 RPM)
    DELAY_BETWEEN_BATCHES = 2  # 2 seconds between batches

    ingestion_start = time.time()  # Track total time

    def process_chunk_sync(chunk_data):
        """Process a single chunk synchronously (NO LOGGING to avoid mess)"""
        chunk, chunk_num = chunk_data
        try:
            # Extract without logging chunk details
            prompt = f"""Extract entities and relationships from the following text.

Return a JSON object with this structure:
{{
  "entities": [
    {{"name": "Entity Name", "type": "Person|Company|Product|Technology|Location", "description": "brief description"}}
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "type": "RELATIONSHIP_TYPE", "description": "brief description"}}
  ]
}}

Rules:
- Extract only clear, factual entities and relationships
- Use UPPERCASE_WITH_UNDERSCORES for relationship types
- Keep descriptions concise

Text:
{chunk}

JSON:"""

            response = pipeline.llm.invoke(prompt)
            content = str(response.content).strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
            return {
                "success": True,
                "chunk_num": chunk_num,
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
            }
        except Exception as e:
            return {
                "success": False,
                "chunk_num": chunk_num,
                "error": str(e),
            }

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    for batch_idx in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[batch_idx : batch_idx + BATCH_SIZE]
        current_batch_num = (batch_idx // BATCH_SIZE) + 1

        print(
            f"📦 Batch {current_batch_num}/{total_batches} - Processing {len(batch_chunks)} chunks..."
        )
        pipeline.log(
            f"📦 Batch {current_batch_num}/{total_batches} - {len(batch_chunks)} chunks"
        )

        batch_start = time.time()

        # Prepare chunk data with indices
        chunk_data = [
            (chunk, batch_idx + idx) for idx, chunk in enumerate(batch_chunks)
        ]

        # Process batch concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            results = list(executor.map(process_chunk_sync, chunk_data))

        # Collect results
        for result in results:
            if result["success"]:
                all_entities.extend(result["entities"])
                all_relationships.extend(result["relationships"])
            else:
                print(
                    f"❌ Error on chunk {result['chunk_num']+1}: {result.get('error', 'Unknown')}"
                )

        batch_time = time.time() - batch_start
        elapsed = time.time() - start_time
        remaining_chunks = len(chunks) - (batch_idx + len(batch_chunks))

        chunks_processed = batch_idx + len(batch_chunks)
        print(
            f"⏱️  Batch completed in {batch_time:.1f}s ({len(batch_chunks)/batch_time:.1f} chunks/sec)"
        )

        if remaining_chunks > 0:
            # Estimate remaining time
            chunks_per_second = chunks_processed / elapsed
            estimated_remaining = (
                remaining_chunks / chunks_per_second if chunks_per_second > 0 else 0
            )

            print(
                f"📊 Progress: {chunks_processed}/{len(chunks)} | Remaining: ~{estimated_remaining/60:.1f} minutes"
            )

            # Delay between batches (except after the last batch)
            if current_batch_num < total_batches:
                pipeline.log(
                    f"⏸️  Waiting {DELAY_BETWEEN_BATCHES}s before next batch..."
                )
                print(f"⏸️  Rate limit pause: {DELAY_BETWEEN_BATCHES}s...")
                time.sleep(DELAY_BETWEEN_BATCHES)

    pipeline.log_subsection("Summary of All Extractions")
    pipeline.log(f"Total entities: {len(all_entities)}")
    pipeline.log(f"Total relationships: {len(all_relationships)}")

    # SAVE TO CACHE FILE (persistent backup)
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{document_id}_extractions.json"

    combined = {"entities": all_entities, "relationships": all_relationships}

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Cached to: {cache_file}")
    pipeline.log(f"💾 Extractions saved to cache: {cache_file}")

    # Build graph from extractions
    try:
        pipeline.build_graph_from_extractions(
            combined, document_id=document_id, document_title=document_title
        )
    except Exception as e:
        print(f"\n❌ Error building graph: {e}")
        print(f"✅ BUT extractions are saved in: {cache_file}")
        print(f"You can retry by loading from cache")
        pipeline.log(f"❌ Graph building failed: {e}", "ERROR")
        pipeline.log(f"Extractions preserved in cache file: {cache_file}")
        raise

    # Create vector store with document metadata
    pipeline.log_section("SETTING UP VECTOR STORE")
    pipeline.log(f"📊 Creating embeddings for {len(chunks)} chunks...")

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "document_id": document_id,
                "document_title": document_title,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    pipeline.log("Creating vector index...")
    from langchain_neo4j import Neo4jVector

    # Create separate vector store for this document
    Neo4jVector.from_documents(
        documents,
        pipeline.embeddings,
        url=pipeline.neo4j_uri,
        username=pipeline.neo4j_username,
        password=pipeline.neo4j_password,
        index_name="vector_index",
        node_label="Chunk",
        embedding_node_property="embedding",
        text_node_property="text",
    )

    pipeline.log("✅ Vector store updated successfully")

    # Calculate total time
    total_time = time.time() - ingestion_start
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"✅ Document ingested successfully!")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"  • Entities: {len(all_entities)}")
    print(f"  • Relationships: {len(all_relationships)}")
    print(f"  • Chunks: {len(chunks)}")
    print(f"  • Total time: {minutes}m {seconds}s")
    print(f"\n📝 Log file: {pipeline.log_file}")
    print()

    pipeline.log(f"\n⏱️  Total ingestion time: {minutes}m {seconds}s")


def main():
    """Main ingestion CLI"""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ingest.py <file_path> <document_title> [--overwrite]")
        print("\nExample:")
        print('  python ingest.py files/moby_dick.txt "Moby Dick"')
        print('  python ingest.py files/1984.txt "Nineteen Eighty-Four" --overwrite')
        sys.exit(1)

    file_path = sys.argv[1]

    if len(sys.argv) < 3:
        print("❌ Error: Document title is required")
        print("Usage: python ingest.py <file_path> <document_title>")
        sys.exit(1)

    # Parse arguments
    document_title = sys.argv[2]
    overwrite = "--overwrite" in sys.argv

    ingest_document(file_path, document_title, overwrite)


if __name__ == "__main__":
    main()
