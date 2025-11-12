"""
GraphRAG Implementation - Automatic Entity & Relationship Extraction
This demonstrates how GraphRAG actually works:
1. Takes text and uses LLM to extract entities/relationships automatically
2. Builds a knowledge graph from these extractions
3. Queries using graph traversal + vector search
"""

import os
import json
from typing import List, Dict, cast
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv(".env.local")

# Configuration from environment variables
_neo4j_uri = os.getenv("NEO4J_URI")
_neo4j_username = os.getenv("NEO4J_USERNAME")
_neo4j_password = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate required environment variables
if not all([_neo4j_uri, _neo4j_username, _neo4j_password]):
    raise ValueError(
        "Missing required Neo4j environment variables. "
        "Please check your .env.local file contains: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
    )

# Type-safe variables (validated above)
NEO4J_URI: str = cast(str, _neo4j_uri)
NEO4J_USERNAME: str = cast(str, _neo4j_username)
NEO4J_PASSWORD: str = cast(str, _neo4j_password)

# Validate Gemini API key
if not GEMINI_API_KEY:
    raise ValueError(
        "Missing GEMINI_API_KEY environment variable. "
        "Please add it to your .env.local file"
    )

# Set Google API key for langchain-google-genai (it expects GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Sample text - you can replace this with your own data
SAMPLE_TEXT = """
Elon Musk is the CEO of Tesla and SpaceX. Tesla manufactures electric vehicles and battery technology.
SpaceX develops rockets and spacecraft for space exploration. Tesla was founded in 2003 and is headquartered 
in Austin, Texas. SpaceX was founded in 2002 with the goal of reducing space transportation costs.

Sam Altman is the CEO of OpenAI. OpenAI created ChatGPT, which is a large language model.
ChatGPT was released in November 2022 and quickly gained millions of users. OpenAI also developed
GPT-4, which is more advanced than previous versions.

Anthropic is an AI safety company founded by former OpenAI researchers. Anthropic created Claude,
which is an AI assistant focused on being helpful, harmless, and honest. Claude uses Constitutional AI
to align with human values.

Neo4j is a graph database used for storing connected data. GraphRAG combines graph databases
with retrieval augmented generation to improve AI responses. LangChain is a framework for building
LLM applications and supports integration with Neo4j.
"""


def setup_components():
    """Initialize all components"""
    print("Initializing components...")

    # Neo4j connection
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("✓ Neo4j connected")
    driver.close()

    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2,
    )
    print("✓ LLM initialized")

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    print("✓ Embeddings initialized")

    return graph, llm, embeddings


def extract_entities_and_relationships(text: str, llm) -> Dict:
    """
    Use LLM to automatically extract entities and relationships from text.
    This is the KEY difference from the hard-coded version!
    """
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
- Use UPPERCASE_WITH_UNDERSCORES for relationship types (e.g., "FOUNDED", "CREATED", "CEO_OF")
- Keep descriptions concise
- Only extract what's explicitly mentioned

Text:
{text}

JSON:"""

    response = llm.invoke(prompt)

    # Parse the JSON response
    try:
        # Clean up the response (remove markdown code blocks if present)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {response.content}")
        return {"entities": [], "relationships": []}


def build_graph_from_extractions(graph, extractions: Dict):
    """
    Build the Neo4j graph from extracted entities and relationships.
    This creates the graph AUTOMATICALLY, not hard-coded!
    """
    entities = extractions.get("entities", [])
    relationships = extractions.get("relationships", [])

    print(
        f"\nBuilding graph from {len(entities)} entities and {len(relationships)} relationships..."
    )

    # Create entity nodes
    for entity in entities:
        entity_type = entity.get("type", "Entity")
        name = entity.get("name", "")
        description = entity.get("description", "")

        if not name:
            continue

        # Use MERGE to avoid duplicates
        query = f"""
        MERGE (e:{entity_type} {{name: $name}})
        SET e.description = $description
        """
        graph.query(query, params={"name": name, "description": description})

    print(f"✓ Created {len(entities)} entity nodes")

    # Create relationships
    created_relationships = 0
    for rel in relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")
        rel_type = rel.get("type", "RELATED_TO")
        description = rel.get("description", "")

        if not source or not target:
            continue

        # Create relationship between entities (regardless of type)
        query = f"""
        MATCH (a {{name: $source}})
        MATCH (b {{name: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.description = $description
        """
        try:
            graph.query(
                query,
                params={"source": source, "target": target, "description": description},
            )
            created_relationships += 1
        except Exception as e:
            print(f"Warning: Could not create relationship {source}->{target}: {e}")

    print(f"✓ Created {created_relationships} relationships")


def create_knowledge_graph_from_text(graph, llm, text: str):
    """
    Main function: Takes raw text and builds a knowledge graph automatically
    """
    print("\n" + "=" * 60)
    print("BUILDING KNOWLEDGE GRAPH FROM TEXT")
    print("=" * 60)

    # Clear existing graph
    graph.query("MATCH (n) DETACH DELETE n")
    print("✓ Cleared existing graph")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_text(text)
    print(f"✓ Split text into {len(chunks)} chunks")

    # Process each chunk
    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}...")
        extractions = extract_entities_and_relationships(chunk, llm)

        # Collect all extractions
        all_entities.extend(extractions.get("entities", []))
        all_relationships.extend(extractions.get("relationships", []))

        print(
            f"  Found {len(extractions.get('entities', []))} entities, {len(extractions.get('relationships', []))} relationships"
        )

    # Build the graph
    combined_extractions = {
        "entities": all_entities,
        "relationships": all_relationships,
    }

    build_graph_from_extractions(graph, combined_extractions)

    # Create documents for vector store
    documents = [
        Document(page_content=chunk, metadata={"chunk_id": i, "source": "sample_text"})
        for i, chunk in enumerate(chunks)
    ]

    return documents


def setup_vector_store(graph, embeddings, documents):
    """Create vector store for semantic search"""
    print("\n" + "=" * 60)
    print("SETTING UP VECTOR STORE")
    print("=" * 60)

    # Clean up existing index
    try:
        graph.query("DROP INDEX vector_index IF EXISTS")
    except:
        pass

    vector_store = Neo4jVector.from_documents(
        documents,
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="vector_index",
        node_label="Chunk",
        embedding_node_property="embedding",
        text_node_property="text",
    )

    print("✓ Vector store created")
    return vector_store


def visualize_graph(graph):
    """Show what's in the graph"""
    print("\n" + "=" * 60)
    print("GRAPH CONTENTS")
    print("=" * 60)

    # Show entities by type
    result = graph.query(
        """
        MATCH (n)
        RETURN labels(n)[0] as type, n.name as name, n.description as description
        ORDER BY type, name
        LIMIT 20
    """
    )

    current_type = None
    for record in result:
        entity_type = record["type"]
        if entity_type != current_type:
            print(f"\n{entity_type}s:")
            current_type = entity_type
        print(f"  • {record['name']}: {record['description']}")

    # Show relationships
    print("\nRelationships:")
    rel_result = graph.query(
        """
        MATCH (a)-[r]->(b)
        RETURN a.name as source, type(r) as relationship, b.name as target
        LIMIT 15
    """
    )

    for record in rel_result:
        print(
            f"  • {record['source']} --{record['relationship']}--> {record['target']}"
        )


def graph_query_example(graph):
    """Show how to query the graph directly (not using vector search)"""
    print("\n" + "=" * 60)
    print("GRAPH QUERY EXAMPLE")
    print("=" * 60)

    # Example: Find all companies and what they created
    print("\nQuery: What did companies create?")
    result = graph.query(
        """
        MATCH (company:Company)-[r:CREATED]->(product)
        RETURN company.name as company, product.name as product, r.description as details
    """
    )

    for record in result:
        print(f"  • {record['company']} created {record['product']}")


def graphrag_query(vector_store, graph, llm, question: str):
    """
    The real GraphRAG query: combines vector search + graph traversal
    """
    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    # Step 1: Vector search to find relevant starting points
    print("\n1. Searching for relevant content...")
    relevant_docs = vector_store.similarity_search(question, k=3)

    print("   Found relevant chunks:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"   {i}. {doc.page_content[:100]}...")

    # Step 2: Extract entities mentioned in relevant docs to use as graph entry points
    context_text = "\n".join([doc.page_content for doc in relevant_docs])

    # Step 3: Use graph to find connected information
    print("\n2. Traversing graph for connected information...")

    # Find entities mentioned in context and get their connections
    graph_result = graph.query(
        """
        MATCH (e)-[r]->(connected)
        RETURN e.name as entity, type(r) as relationship, connected.name as connected_entity, 
               connected.description as description
        LIMIT 10
    """
    )

    graph_context = "\nGraph connections:\n"
    for record in graph_result:
        graph_context += f"- {record['entity']} {record['relationship']} {record['connected_entity']}\n"

    print(f"   Found {len(graph_result)} graph connections")

    # Step 4: Combine vector search results + graph traversal for LLM
    combined_context = f"""Text Content:
{context_text}

{graph_context}"""

    prompt = f"""Answer the question based on the provided context.

Context:
{combined_context}

Question: {question}

Answer:"""

    print("\n3. Generating answer with LLM...")
    response = llm.invoke(prompt)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response.content)
    print("=" * 60)

    return response.content


def main():
    """Run the complete GraphRAG demo"""
    print("=" * 60)
    print("REAL GRAPHRAG DEMO - AUTOMATIC EXTRACTION")
    print("=" * 60)

    # Setup
    graph, llm, embeddings = setup_components()

    # Build knowledge graph automatically from text
    documents = create_knowledge_graph_from_text(graph, llm, SAMPLE_TEXT)

    # Setup vector store
    vector_store = setup_vector_store(graph, embeddings, documents)

    # Visualize what we built
    visualize_graph(graph)

    # Show a pure graph query
    graph_query_example(graph)

    # GraphRAG queries
    print("\n" + "=" * 60)
    print("GRAPHRAG QUERIES (Vector + Graph)")
    print("=" * 60)

    questions = [
        "What companies did Elon Musk found?",
        "What AI models were created and by whom?",
        "Tell me about graph databases and how they're used",
    ]

    for question in questions:
        graphrag_query(vector_store, graph, llm, question)

    print("\n✓ Demo complete!")
    print("\nTry changing the SAMPLE_TEXT variable to your own data!")


if __name__ == "__main__":
    main()
