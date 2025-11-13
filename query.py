"""
Document Query CLI - Query documents in Neo4j GraphRAG
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
from main import GraphRAGPipeline
from langchain_neo4j import Neo4jVector

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


def query_document(pipeline: GraphRAGPipeline, document_id: str, document_title: str):
    """Interactive query session for a specific document"""

    print(f"\n{'='*60}")
    print(f"📚 Querying Document: {document_title}")
    print(f"{'='*60}\n")

    # Get document stats
    stats = get_document_stats(pipeline, document_id)
    print(f"📊 Document Statistics:")
    print(f"  • Entities: {stats['entities']}")
    print(f"  • Relationships: {stats['relationships']}")
    print(f"  • Chunks: {stats['chunks']}\n")

    # Load vector store with document filter
    pipeline.log_section(f"QUERYING DOCUMENT: {document_title}")
    pipeline.log(f"Document ID: {document_id}")

    # Initialize vector store (it uses the existing index)
    vector_store = Neo4jVector.from_existing_index(
        pipeline.embeddings,
        url=pipeline.neo4j_uri,
        username=pipeline.neo4j_username,
        password=pipeline.neo4j_password,
        index_name="vector_index",
        node_label="Chunk",
        embedding_node_property="embedding",
        text_node_property="text",
    )

    pipeline.vector_store = vector_store

    print(f"{'='*60}")
    print(f"GraphRAG is ready! Ask questions about '{document_title}'")
    print(f"Type 'exit' to quit")
    print(f"{'='*60}\n")

    while True:
        try:
            question = input(f"\n❓ Your question: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                pipeline.log("User exited the session")
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            # Query with document filter
            pipeline.log_section(f"GRAPHRAG QUERY: {question}")

            # Vector search with document filter
            relevant_docs = vector_store.similarity_search(
                question, k=10, filter={"document_id": document_id}
            )

            if not relevant_docs:
                print("⚠️ No relevant information found in this document.\n")
                pipeline.log("⚠️ No relevant chunks found")
                continue

            # Add newlines between chunks
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Use LLM to extract entity names from the question
            entity_extraction_prompt = f"""Extract all entity names (people, places, concepts, books, etc.) mentioned in this question.
Return ONLY a JSON list of entity names, nothing else.

Question: {question}

JSON list of entities:"""

            try:
                entity_response = pipeline.llm.invoke(entity_extraction_prompt)
                entity_content = str(entity_response.content).strip()
                # Clean up the response
                if entity_content.startswith("```json"):
                    entity_content = entity_content[7:]
                if entity_content.startswith("```"):
                    entity_content = entity_content[3:]
                if entity_content.endswith("```"):
                    entity_content = entity_content[:-3]
                entity_content = entity_content.strip()

                import json

                extracted_entities = json.loads(entity_content)
                if not isinstance(extracted_entities, list):
                    extracted_entities = []
            except:
                extracted_entities = []

            # If we have entities from LLM, use them; otherwise fall back to a general query
            if extracted_entities:
                # Query for connections involving these entities (case-insensitive matching)
                graph_query = """
                MATCH (e)-[r]->(connected)
                WHERE e.document_id = $document_id 
                  AND connected.document_id = $document_id
                  AND (
                    ANY(entity_name IN $entity_names WHERE toLower(e.name) = toLower(entity_name))
                    OR ANY(entity_name IN $entity_names WHERE toLower(connected.name) = toLower(entity_name))
                  )
                RETURN DISTINCT e.name as entity, 
                       type(r) as relationship, 
                       connected.name as connected_entity, 
                       r.description as description
                LIMIT 50
                """
                graph_result = pipeline.graph.query(
                    graph_query,
                    params={
                        "document_id": document_id,
                        "entity_names": extracted_entities,
                    },
                )
            else:
                # Fallback: get some general connections
                graph_query = """
                MATCH (e)-[r]->(connected)
                WHERE e.document_id = $document_id AND connected.document_id = $document_id
                RETURN DISTINCT e.name as entity, type(r) as relationship, connected.name as connected_entity, r.description as description
                LIMIT 25
                """
                graph_result = pipeline.graph.query(
                    graph_query,
                    params={"document_id": document_id},
                )

            # Build graph context with descriptions
            graph_context = "\nGraph connections:\n"
            for record in graph_result:
                description = record.get("description", "")
                if description:
                    connection = f"- {record['entity']} --{record['relationship']}--> {record['connected_entity']} ({description})"
                else:
                    connection = f"- {record['entity']} --{record['relationship']}--> {record['connected_entity']}"
                graph_context += f"{connection}\n"

            # Combine context
            combined_context = f"Text Content:\n{context_text}\n\n{graph_context}"

            print(
                f"\n📝 Combined Context ({len(relevant_docs)} chunks, {len(graph_result)} connections):"
            )
            print(f"{combined_context}\n")

            pipeline.log_subsection("Combined Context")
            pipeline.log(
                f"Retrieved {len(relevant_docs)} chunks and {len(graph_result)} graph connections"
            )
            pipeline.log_code_block(combined_context, "text")

            # Generate answer
            prompt = f"""Answer the question based on the provided context from the document "{document_title}".

Context:
{combined_context}

Question: {question}

Answer:"""

            pipeline.log_subsection("LLM Answer Generation")
            pipeline.log("Generating answer...")
            response = pipeline.llm.invoke(prompt)

            answer = str(response.content)
            pipeline.log("Answer:")
            pipeline.log_code_block(answer, "text")

            print(f"✅ Answer: {answer}\n")

        except KeyboardInterrupt:
            pipeline.log("User interrupted the session")
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            pipeline.log(f"Error during query: {e}", "ERROR")
            print(f"\n❌ Error: {e}\n")

    pipeline.log_section("SESSION COMPLETE")
    pipeline.log(f"✅ Full log saved to: {pipeline.log_file}")
    print(f"\n📝 Full log saved to: {pipeline.log_file}\n")


def main():
    """Main query CLI"""

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

    # Get all documents
    documents = get_all_documents(pipeline)

    if not documents:
        print("\n❌ No documents found in the database.")
        print("Please ingest documents first using:")
        print('  python ingest.py <file_path> "<document_title>"\n')
        sys.exit(1)

    # Display available documents
    print(f"\n{'='*60}")
    print(f"📚 Available Documents")
    print(f"{'='*60}\n")

    for i, doc in enumerate(documents, 1):
        stats = get_document_stats(pipeline, doc["id"])
        print(f"{i}. {doc['title']}")
        print(f"   ID: {doc['id']}")
        print(f"   Entities: {stats['entities']}, Chunks: {stats['chunks']}\n")

    # Let user select document
    while True:
        try:
            selection = input(
                f"Select document (1-{len(documents)}) or 'q' to quit: "
            ).strip()

            if selection.lower() in ["q", "quit", "exit"]:
                print("\n👋 Goodbye!")
                sys.exit(0)

            idx = int(selection) - 1
            if 0 <= idx < len(documents):
                selected_doc = documents[idx]
                query_document(pipeline, selected_doc["id"], selected_doc["title"])
                break
            else:
                print(f"❌ Invalid selection. Please choose 1-{len(documents)}")
        except ValueError:
            print(f"❌ Invalid input. Please enter a number 1-{len(documents)}")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
