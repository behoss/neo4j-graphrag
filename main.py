"""
GraphRAG Implementation - Automatic Entity & Relationship Extraction with Full Logging
This demonstrates how GraphRAG actually works with complete visibility into the process.
"""

import os
import json
from typing import Dict, cast
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Create log file with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"graphrag_run_{TIMESTAMP}.md"


def log(message: str, level: str = "INFO"):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_message = f"[{timestamp}] {level}: {message}"
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{message}\n\n")


def log_section(title: str):
    """Log a section header"""
    separator = "=" * 60
    log(f"\n{separator}\n{title}\n{separator}")


def log_subsection(title: str):
    """Log a subsection header"""
    log(f"\n### {title}\n")


def log_code_block(content: str, language: str = ""):
    """Log a code block"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"```{language}\n{content}\n```\n\n")


# Load environment variables
load_dotenv(".env.local")

# Configuration
_neo4j_uri = os.getenv("NEO4J_URI")
_neo4j_username = os.getenv("NEO4J_USERNAME")
_neo4j_password = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate
if not all([_neo4j_uri, _neo4j_username, _neo4j_password]):
    raise ValueError("Missing required Neo4j environment variables")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable")

# Type-safe variables
NEO4J_URI: str = cast(str, _neo4j_uri)
NEO4J_USERNAME: str = cast(str, _neo4j_username)
NEO4J_PASSWORD: str = cast(str, _neo4j_password)

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Sample text
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
    """Initialize all components with logging"""
    log_section("INITIALIZING COMPONENTS")

    log_subsection("Neo4j Connection")
    log(f"URI: {NEO4J_URI}")
    log(f"Username: {NEO4J_USERNAME}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    log("✓ Neo4j connected successfully")
    driver.close()

    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    log_subsection("LLM Configuration")
    llm_config = {"model": "gemini-2.5-flash", "temperature": 0, "max_retries": 2}
    log_code_block(json.dumps(llm_config, indent=2), "json")

    llm = ChatGoogleGenerativeAI(**llm_config)
    log("✓ LLM initialized")

    log_subsection("Embeddings Configuration")
    log("Model: models/gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    log("✓ Embeddings initialized")

    return graph, llm, embeddings


def extract_entities_and_relationships(text: str, llm, chunk_num: int) -> Dict:
    """Extract entities and relationships with full logging"""
    log_subsection(f"Chunk #{chunk_num+1} - Entity Extraction")

    log("📄 Input Text:")
    log_code_block(text, "text")

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
{text}

JSON:"""

    log("📤 Sending to LLM...")
    response = llm.invoke(prompt)

    log("📥 LLM Response:")

    # Parse response
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Log the clean JSON
    log_code_block(content, "json")

    try:
        data = json.loads(content)
        log(
            f"✅ Extracted: {len(data.get('entities', []))} entities, {len(data.get('relationships', []))} relationships"
        )
        return data
    except json.JSONDecodeError as e:
        log(f"❌ JSON Parse Error: {e}", "ERROR")
        return {"entities": [], "relationships": []}


def build_graph_from_extractions(graph, extractions: Dict):
    """Build graph with logging"""
    log_subsection("Building Graph in Neo4j")

    entities = extractions.get("entities", [])
    relationships = extractions.get("relationships", [])

    log(
        f"📊 Processing {len(entities)} entities and {len(relationships)} relationships"
    )

    # Create entities
    log("Creating entity nodes...")
    for i, entity in enumerate(entities[:3]):  # Log first 3
        query = f"MERGE (e:{entity.get('type', 'Entity')} {{name: $name}}) SET e.description = $description"
        log(f"Cypher Query #{i+1}:")
        log_code_block(query, "cypher")
        log(
            f"Parameters: {{'name': '{entity.get('name')}', 'description': '{entity.get('description')[:50]}...'}}"
        )

    for entity in entities:
        if not entity.get("name"):
            continue
        query = f"MERGE (e:{entity.get('type', 'Entity')} {{name: $name}}) SET e.description = $description"
        graph.query(
            query,
            params={
                "name": entity.get("name"),
                "description": entity.get("description"),
            },
        )

    log(f"✅ Created {len(entities)} nodes")

    # Create relationships
    log("Creating relationships...")
    created = 0
    for rel in relationships[:3]:  # Log first 3
        query = f"MATCH (a {{name: $source}}) MATCH (b {{name: $target}}) MERGE (a)-[r:{rel.get('type', 'RELATED_TO')}]->(b) SET r.description = $description"
        log(f"Cypher Query:")
        log_code_block(query, "cypher")

    for rel in relationships:
        if not rel.get("source") or not rel.get("target"):
            continue
        query = f"MATCH (a {{name: $source}}) MATCH (b {{name: $target}}) MERGE (a)-[r:{rel.get('type', 'RELATED_TO')}]->(b) SET r.description = $description"
        try:
            graph.query(
                query,
                params={
                    "source": rel.get("source"),
                    "target": rel.get("target"),
                    "description": rel.get("description"),
                },
            )
            created += 1
        except Exception as e:
            log(
                f"⚠️ Could not create {rel.get('source')}->{rel.get('target')}: {e}",
                "WARN",
            )

    log(f"✅ Created {created} relationships")


def create_knowledge_graph_from_text(graph, llm, text: str):
    """Build knowledge graph with full logging"""
    log_section("BUILDING KNOWLEDGE GRAPH FROM TEXT")

    log("📝 Full Input Text:")
    log_code_block(text, "text")

    log("🗑️ Clearing existing graph...")
    graph.query("MATCH (n) DETACH DELETE n")
    log("✓ Graph cleared")

    log("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    log(f"✓ Created {len(chunks)} chunks\n")

    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        log(f"\n🔄 Processing chunk {i+1}/{len(chunks)}...")
        extractions = extract_entities_and_relationships(chunk, llm, i)
        all_entities.extend(extractions.get("entities", []))
        all_relationships.extend(extractions.get("relationships", []))

    log_subsection("Summary of All Extractions")
    log(f"Total entities: {len(all_entities)}")
    log(f"Total relationships: {len(all_relationships)}")

    combined = {"entities": all_entities, "relationships": all_relationships}
    build_graph_from_extractions(graph, combined)

    documents = [
        Document(page_content=chunk, metadata={"chunk_id": i, "source": "sample_text"})
        for i, chunk in enumerate(chunks)
    ]

    return documents


def setup_vector_store(graph, embeddings, documents):
    """Create vector store with logging"""
    log_section("SETTING UP VECTOR STORE")

    log(f"📊 Creating embeddings for {len(documents)} documents...")
    log("Dropping existing index if present...")

    try:
        graph.query("DROP INDEX vector_index IF EXISTS")
        log("✓ Cleaned up old index")
    except:
        pass

    log("Creating new vector index...")
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

    log("✅ Vector store created successfully")
    return vector_store


def visualize_graph(graph):
    """Visualize graph with logging"""
    log_section("GRAPH VISUALIZATION")

    log("Query: Fetching all nodes (excluding Chunk nodes used for vector search)...")
    query = "MATCH (n) WHERE NOT 'Chunk' IN labels(n) RETURN labels(n)[0] as type, n.name as name, n.description as description ORDER BY type, name LIMIT 20"
    log_code_block(query, "cypher")

    result = graph.query(query)

    log(f"📊 Found {len(result)} entity nodes")
    current_type = None
    for record in result:
        entity_type = record["type"]
        if entity_type != current_type:
            log(f"\n{entity_type}s:")
            current_type = entity_type
        log(f"  • {record['name']}: {record['description']}")

    log("\n🔗 Relationships:")
    rel_query = "MATCH (a)-[r]->(b) RETURN a.name as source, type(r) as relationship, b.name as target LIMIT 15"
    log_code_block(rel_query, "cypher")

    rel_result = graph.query(rel_query)
    for record in rel_result:
        log(f"  • {record['source']} --{record['relationship']}--> {record['target']}")


def graph_query_example(graph):
    """Example graph query with logging"""
    log_section("PURE GRAPH QUERY EXAMPLE")

    log("Query: What did companies create?")
    query = "MATCH (company:Company)-[r:CREATED]->(product) RETURN company.name as company, product.name as product"
    log_code_block(query, "cypher")

    result = graph.query(query)
    log(f"Results: {len(result)} items")
    for record in result:
        log(f"  • {record['company']} created {record['product']}")


def graphrag_query(vector_store, graph, llm, question: str):
    """GraphRAG query with full logging"""
    log_section(f"GRAPHRAG QUERY")
    log(f"❓ Question: {question}")

    log_subsection("Step 1: Vector Search")
    log("Searching for relevant chunks (k=3)...")
    relevant_docs = vector_store.similarity_search(question, k=3)

    log(f"Found {len(relevant_docs)} relevant chunks:")
    for i, doc in enumerate(relevant_docs, 1):
        log(f"\nChunk {i}:")
        log_code_block(doc.page_content, "text")

    context_text = "\n".join([doc.page_content for doc in relevant_docs])

    log_subsection("Step 2: Graph Traversal")
    query = "MATCH (e)-[r]->(connected) RETURN e.name as entity, type(r) as relationship, connected.name as connected_entity LIMIT 10"
    log("Cypher Query:")
    log_code_block(query, "cypher")

    graph_result = graph.query(query)
    log(f"Found {len(graph_result)} graph connections:")

    graph_context = "\nGraph connections:\n"
    for record in graph_result:
        connection = f"{record['entity']} --{record['relationship']}--> {record['connected_entity']}"
        log(f"  • {connection}")
        graph_context += f"- {connection}\n"

    log_subsection("Step 3: Combining Context")
    combined_context = f"Text Content:\n{context_text}\n\n{graph_context}"
    log("Combined context:")
    log_code_block(combined_context, "text")

    prompt = f"""Answer the question based on the provided context.

Context:
{combined_context}

Question: {question}

Answer:"""

    log_subsection("Step 4: LLM Answer Generation")
    log("Sending to LLM...")
    response = llm.invoke(prompt)

    log("\n✅ ANSWER:")
    log_code_block(response.content, "text")

    return response.content


def main():
    """Run GraphRAG demo with full logging"""
    log_section("GRAPHRAG DEMO - AUTOMATIC EXTRACTION")
    log(f"Log file: {LOG_FILE}")
    log(f"Timestamp: {TIMESTAMP}")

    graph, llm, embeddings = setup_components()
    documents = create_knowledge_graph_from_text(graph, llm, SAMPLE_TEXT)
    vector_store = setup_vector_store(graph, embeddings, documents)
    visualize_graph(graph)
    graph_query_example(graph)

    log_section("GRAPHRAG QUERIES")

    questions = [
        "What companies did Elon Musk found?",
        "What AI models were created and by whom?",
        "Tell me about graph databases and how they're used",
    ]

    for question in questions:
        graphrag_query(vector_store, graph, llm, question)

    log_section("DEMO COMPLETE")
    log(f"✅ Full log saved to: {LOG_FILE}")
    log("\n📝 Check the logs/ directory for complete execution details!")


if __name__ == "__main__":
    main()
