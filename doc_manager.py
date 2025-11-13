"""
Document Manager CLI - Manage documents in Neo4j GraphRAG
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
from main import GraphRAGPipeline

load_dotenv(".env.local")


def get_all_documents(pipeline: GraphRAGPipeline) -> List[Dict[str, str]]:
    """Get list of all documents in the database"""
    query = """
    MATCH (n)
    WHERE n.document_id IS NOT NULL
    RETURN DISTINCT n.document_id as id, n.document_title as title
    ORDER BY n.document_title
    """
    results = pipeline.graph.query(query)
    return [{"id": r["id"], "title": r["title"]} for r in results if r["id"]]


def get_document_stats(pipeline: GraphRAGPipeline, document_id: str) -> Dict:
    """Get statistics for a specific document"""
    # Count entities
    entity_query = """
    MATCH (n)
    WHERE n.document_id = $document_id AND NOT 'Chunk' IN labels(n)
    RETURN count(n) as entity_count
    """
    entity_result = pipeline.graph.query(
        entity_query, params={"document_id": document_id}
    )
    entity_count = entity_result[0]["entity_count"] if entity_result else 0

    # Count chunks
    chunk_query = """
    MATCH (n:Chunk)
    WHERE n.document_id = $document_id
    RETURN count(n) as chunk_count
    """
    chunk_result = pipeline.graph.query(
        chunk_query, params={"document_id": document_id}
    )
    chunk_count = chunk_result[0]["chunk_count"] if chunk_result else 0

    # Count relationships
    rel_query = """
    MATCH (a)-[r]->(b)
    WHERE a.document_id = $document_id
    RETURN count(r) as rel_count
    """
    rel_result = pipeline.graph.query(rel_query, params={"document_id": document_id})
    rel_count = rel_result[0]["rel_count"] if rel_result else 0

    return {"entities": entity_count, "chunks": chunk_count, "relationships": rel_count}


def list_documents(pipeline: GraphRAGPipeline):
    """List all documents with statistics"""
    documents = get_all_documents(pipeline)

    if not documents:
        print("\n📚 No documents found in the database.\n")
        return

    print(f"\n{'='*60}")
    print(f"📚 All Documents ({len(documents)} total)")
    print(f"{'='*60}\n")

    for i, doc in enumerate(documents, 1):
        stats = get_document_stats(pipeline, doc["id"])
        print(f"{i}. {doc['title']}")
        print(f"   ID: {doc['id']}")
        print(f"   📊 Entities: {stats['entities']}")
        print(f"   🔗 Relationships: {stats['relationships']}")
        print(f"   📄 Chunks: {stats['chunks']}\n")


def get_document_info(pipeline: GraphRAGPipeline, document_id: str):
    """Get detailed information about a specific document"""
    # Check if document exists
    check_query = "MATCH (n {document_id: $document_id}) RETURN count(n) as count"
    result = pipeline.graph.query(check_query, params={"document_id": document_id})

    if not result or result[0]["count"] == 0:
        print(f"\n❌ Document '{document_id}' not found.\n")
        return

    # Get document title
    title_query = (
        "MATCH (n {document_id: $document_id}) RETURN n.document_title as title LIMIT 1"
    )
    title_result = pipeline.graph.query(
        title_query, params={"document_id": document_id}
    )
    document_title = title_result[0]["title"] if title_result else "Unknown"

    # Get stats
    stats = get_document_stats(pipeline, document_id)

    print(f"\n{'='*60}")
    print(f"📄 Document Information")
    print(f"{'='*60}\n")
    print(f"Title: {document_title}")
    print(f"ID: {document_id}\n")
    print(f"📊 Statistics:")
    print(f"  • Entities: {stats['entities']}")
    print(f"  • Relationships: {stats['relationships']}")
    print(f"  • Chunks: {stats['chunks']}\n")

    # Get entity types breakdown
    entity_types_query = """
    MATCH (n)
    WHERE n.document_id = $document_id AND NOT 'Chunk' IN labels(n)
    RETURN labels(n)[0] as type, count(n) as count
    ORDER BY count DESC
    """
    entity_types = pipeline.graph.query(
        entity_types_query, params={"document_id": document_id}
    )

    if entity_types:
        print(f"📊 Entity Types:")
        for et in entity_types:
            print(f"  • {et['type']}: {et['count']}")
        print()

    # Get sample entities
    sample_query = """
    MATCH (n)
    WHERE n.document_id = $document_id AND NOT 'Chunk' IN labels(n)
    RETURN labels(n)[0] as type, n.name as name
    ORDER BY n.name
    LIMIT 10
    """
    sample_entities = pipeline.graph.query(
        sample_query, params={"document_id": document_id}
    )

    if sample_entities:
        print(f"📝 Sample Entities:")
        for entity in sample_entities:
            print(f"  • {entity['name']} ({entity['type']})")
        print()


def delete_document(pipeline: GraphRAGPipeline, document_id: str):
    """Delete a specific document from the database"""
    # Check if document exists
    check_query = "MATCH (n {document_id: $document_id}) RETURN count(n) as count"
    result = pipeline.graph.query(check_query, params={"document_id": document_id})

    if not result or result[0]["count"] == 0:
        print(f"\n❌ Document '{document_id}' not found.\n")
        return

    # Get document title
    title_query = (
        "MATCH (n {document_id: $document_id}) RETURN n.document_title as title LIMIT 1"
    )
    title_result = pipeline.graph.query(
        title_query, params={"document_id": document_id}
    )
    document_title = title_result[0]["title"] if title_result else "Unknown"

    # Confirm deletion
    print(
        f"\n⚠️  Warning: You are about to delete '{document_title}' (ID: {document_id})"
    )
    print(
        "This will remove all entities, relationships, and chunks for this document.\n"
    )

    response = input("Are you sure? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("❌ Deletion cancelled\n")
        return

    # Delete document
    print(f"\n🗑️  Deleting document '{document_title}'...")
    delete_query = "MATCH (n {document_id: $document_id}) DETACH DELETE n"
    pipeline.graph.query(delete_query, params={"document_id": document_id})

    print(f"✅ Document deleted successfully\n")


def purge_all(pipeline: GraphRAGPipeline):
    """Purge ALL data from the database"""
    print("\n⚠️  WARNING: COMPLETE DATABASE PURGE")
    print("This will delete ALL nodes and relationships in the database.")
    print("This action CANNOT be undone!\n")

    response = input("Type 'DELETE EVERYTHING' to confirm: ").strip()
    if response != "DELETE EVERYTHING":
        print("❌ Purge cancelled\n")
        return

    print("\n🗑️  Purging entire database...")
    pipeline.graph.query("MATCH (n) DETACH DELETE n")
    print("✅ Database completely purged\n")


def main():
    """Main document manager CLI"""
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python doc_manager.py --list")
        print("  python doc_manager.py --info <document_id>")
        print("  python doc_manager.py --delete <document_id>")
        print("  python doc_manager.py --purge-all")
        print("\nExamples:")
        print("  python doc_manager.py --list")
        print("  python doc_manager.py --info moby-dick")
        print("  python doc_manager.py --delete moby-dick")
        print("  python doc_manager.py --purge-all  # DANGER: Deletes everything!\n")
        sys.exit(1)

    command = sys.argv[1]

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

    if command == "--list":
        list_documents(pipeline)
    elif command == "--info":
        if len(sys.argv) < 3:
            print("❌ Error: Document ID required")
            print("Usage: python doc_manager.py --info <document_id>\n")
            sys.exit(1)
        document_id = sys.argv[2]
        get_document_info(pipeline, document_id)
    elif command == "--delete":
        if len(sys.argv) < 3:
            print("❌ Error: Document ID required")
            print("Usage: python doc_manager.py --delete <document_id>\n")
            sys.exit(1)
        document_id = sys.argv[2]
        delete_document(pipeline, document_id)
    elif command == "--purge-all":
        purge_all(pipeline)
    else:
        print(f"❌ Unknown command: {command}")
        print("\nAvailable commands: --list, --info, --delete, --purge-all\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
