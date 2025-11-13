"""
GraphRAG Pipeline - Core application logic for knowledge graph construction and querying
"""

import os
import sys
import json
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class GraphRAGPipeline:
    """GraphRAG pipeline for processing text and querying knowledge graphs"""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        gemini_api_key: str,
    ):
        """Initialize GraphRAG pipeline"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = logs_dir / f"graphrag_run_{timestamp}.md"

        # Store credentials
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password

        # Set API key
        os.environ["GOOGLE_API_KEY"] = gemini_api_key

        # Initialize components
        self._initialize_components()

    def log(self, message: str, level: str = "INFO"):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n\n")

    def log_section(self, title: str):
        """Log a section header"""
        separator = "=" * 60
        self.log(f"\n{separator}\n{title}\n{separator}")

    def log_subsection(self, title: str):
        """Log a subsection header"""
        self.log(f"\n### {title}\n")

    def log_code_block(self, content: str, language: str = ""):
        """Log a code block"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"```{language}\n{content}\n```\n\n")

    def _initialize_components(self):
        """Initialize Neo4j, LLM, and embeddings"""
        self.log_section("INITIALIZING COMPONENTS")

        self.log_subsection("Neo4j Connection")
        self.log(f"URI: {self.neo4j_uri}")
        self.log(f"Username: {self.neo4j_username}")

        driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password)
        )
        driver.verify_connectivity()
        self.log("✓ Neo4j connected successfully")
        driver.close()

        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
        )

        self.log_subsection("LLM Configuration")
        llm_config = {"model": "gemini-2.5-flash", "temperature": 0, "max_retries": 2}
        self.log_code_block(json.dumps(llm_config, indent=2), "json")

        self.llm = ChatGoogleGenerativeAI(**llm_config)
        self.log("✓ LLM initialized")

        self.log_subsection("Embeddings Configuration")
        self.log("Model: models/gemini-embedding-001")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self.log("✓ Embeddings initialized")

    def extract_entities_and_relationships(self, text: str, chunk_num: int) -> Dict:
        """Extract entities and relationships with full logging"""
        self.log_subsection(f"Chunk #{chunk_num+1} - Entity Extraction")

        self.log("📄 Input Text:")
        self.log_code_block(text, "text")

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

        self.log("📤 Sending to LLM...")
        response = self.llm.invoke(prompt)

        self.log("📥 LLM Response:")

        # Parse response
        content = str(response.content).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Log the clean JSON
        self.log_code_block(content, "json")

        try:
            data = json.loads(content)
            self.log(
                f"✅ Extracted: {len(data.get('entities', []))} entities, {len(data.get('relationships', []))} relationships"
            )
            return data
        except json.JSONDecodeError as e:
            self.log(f"❌ JSON Parse Error: {e}", "ERROR")
            return {"entities": [], "relationships": []}

    def build_graph_from_extractions(
        self,
        extractions: Dict,
        document_id: Optional[str] = None,
        document_title: Optional[str] = None,
    ):
        """Build graph with logging"""
        self.log_subsection("Building Graph in Neo4j")

        entities = extractions.get("entities", [])
        relationships = extractions.get("relationships", [])

        self.log(
            f"📊 Processing {len(entities)} entities and {len(relationships)} relationships"
        )

        # Create entities in batches
        self.log("Creating entity nodes in batches...")
        BATCH_SIZE = 100
        for i in range(0, len(entities), BATCH_SIZE):
            batch = entities[i : i + BATCH_SIZE]
            for entity in batch:
                if not entity.get("name"):
                    continue
                # Escape entity type with backticks to handle spaces
                entity_type = entity.get("type", "Entity").replace("`", "")
                query = f"MERGE (e:`{entity_type}` {{name: $name, document_id: $document_id}}) SET e.description = $description, e.document_title = $document_title"
                try:
                    self.graph.query(
                        query,
                        params={
                            "name": entity.get("name"),
                            "description": entity.get("description"),
                            "document_id": document_id,
                            "document_title": document_title,
                        },
                    )
                except Exception as e:
                    self.log(
                        f"⚠️ Error creating entity {entity.get('name')}: {e}", "WARN"
                    )
            self.log(
                f"  Batch {(i//BATCH_SIZE)+1}/{(len(entities)+BATCH_SIZE-1)//BATCH_SIZE} completed"
            )

        self.log(f"✅ Created {len(entities)} nodes")

        # Create relationships in batches
        self.log("Creating relationships in batches...")
        print("Creating relationships in batches...")
        REL_BATCH_SIZE = 200
        created = 0

        for i in range(0, len(relationships), REL_BATCH_SIZE):
            batch = relationships[i : i + REL_BATCH_SIZE]
            for rel in batch:
                if not rel.get("source") or not rel.get("target"):
                    continue
                query = f"MATCH (a {{name: $source}}) MATCH (b {{name: $target}}) MERGE (a)-[r:{rel.get('type', 'RELATED_TO')}]->(b) SET r.description = $description"
                try:
                    self.graph.query(
                        query,
                        params={
                            "source": rel.get("source"),
                            "target": rel.get("target"),
                            "description": rel.get("description"),
                        },
                    )
                    created += 1
                except Exception as e:
                    self.log(
                        f"⚠️ Could not create {rel.get('source')}->{rel.get('target')}: {e}",
                        "WARN",
                    )

            batch_num = (i // REL_BATCH_SIZE) + 1
            total_rel_batches = (
                len(relationships) + REL_BATCH_SIZE - 1
            ) // REL_BATCH_SIZE
            self.log(f"  Batch {batch_num}/{total_rel_batches} completed")
            print(f"  Batch {batch_num}/{total_rel_batches} completed")

        self.log(f"✅ Created {created} relationships")

    def process_text(self, text: str, clear_graph: bool = True):
        """Process text and build knowledge graph"""
        self.log_section("BUILDING KNOWLEDGE GRAPH FROM TEXT")

        self.log("📝 Full Input Text:")
        self.log_code_block(text, "text")

        if clear_graph:
            self.log("🗑️ Clearing existing graph...")
            self.graph.query("MATCH (n) DETACH DELETE n")
            self.log("✓ Graph cleared")

        self.log("✂️ Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)
        self.log(f"✓ Created {len(chunks)} chunks\n")

        all_entities = []
        all_relationships = []

        for i, chunk in enumerate(chunks):
            self.log(f"\n🔄 Processing chunk {i+1}/{len(chunks)}...")
            extractions = self.extract_entities_and_relationships(chunk, i)
            all_entities.extend(extractions.get("entities", []))
            all_relationships.extend(extractions.get("relationships", []))

        self.log_subsection("Summary of All Extractions")
        self.log(f"Total entities: {len(all_entities)}")
        self.log(f"Total relationships: {len(all_relationships)}")

        combined = {"entities": all_entities, "relationships": all_relationships}
        self.build_graph_from_extractions(
            combined, document_id=None, document_title=None
        )

        # Create vector store
        self.log_section("SETTING UP VECTOR STORE")
        self.log(f"📊 Creating embeddings for {len(chunks)} documents...")

        documents = [
            Document(
                page_content=chunk, metadata={"chunk_id": i, "source": "input_text"}
            )
            for i, chunk in enumerate(chunks)
        ]

        try:
            self.graph.query("DROP INDEX vector_index IF EXISTS")
            self.log("✓ Cleaned up old index")
        except:
            pass

        self.log("Creating new vector index...")
        self.vector_store = Neo4jVector.from_documents(
            documents,
            self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name="vector_index",
            node_label="Chunk",
            embedding_node_property="embedding",
            text_node_property="text",
        )

        self.log("✅ Vector store created successfully")

    def visualize_graph(self):
        """Visualize graph with logging"""
        self.log_section("GRAPH VISUALIZATION")

        self.log(
            "Query: Fetching all nodes (excluding Chunk nodes used for vector search)..."
        )
        query = "MATCH (n) WHERE NOT 'Chunk' IN labels(n) RETURN labels(n)[0] as type, n.name as name, n.description as description ORDER BY type, name LIMIT 20"
        self.log_code_block(query, "cypher")

        result = self.graph.query(query)

        self.log(f"📊 Found {len(result)} entity nodes")
        current_type = None
        for record in result:
            entity_type = record["type"]
            if entity_type != current_type:
                self.log(f"\n{entity_type}s:")
                current_type = entity_type
            self.log(f"  • {record['name']}: {record['description']}")

        self.log("\n🔗 Relationships:")
        rel_query = "MATCH (a)-[r]->(b) RETURN a.name as source, type(r) as relationship, b.name as target LIMIT 15"
        self.log_code_block(rel_query, "cypher")

        rel_result = self.graph.query(rel_query)
        for record in rel_result:
            self.log(
                f"  • {record['source']} --{record['relationship']}--> {record['target']}"
            )

    def query(self, question: str) -> str:
        """Query the knowledge graph using GraphRAG"""
        self.log_section(f"GRAPHRAG QUERY")
        self.log(f"❓ Question: {question}")

        self.log_subsection("Step 1: Vector Search")
        self.log("Searching for relevant chunks (k=3)...")
        relevant_docs = self.vector_store.similarity_search(question, k=3)

        self.log(f"Found {len(relevant_docs)} relevant chunks:")
        for i, doc in enumerate(relevant_docs, 1):
            self.log(f"\nChunk {i}:")
            self.log_code_block(doc.page_content, "text")

        context_text = "\n".join([doc.page_content for doc in relevant_docs])

        self.log_subsection("Step 2: Graph Traversal")
        query = "MATCH (e)-[r]->(connected) RETURN e.name as entity, type(r) as relationship, connected.name as connected_entity LIMIT 10"
        self.log("Cypher Query:")
        self.log_code_block(query, "cypher")

        graph_result = self.graph.query(query)
        self.log(f"Found {len(graph_result)} graph connections:")

        graph_context = "\nGraph connections:\n"
        for record in graph_result:
            connection = f"{record['entity']} --{record['relationship']}--> {record['connected_entity']}"
            self.log(f"  • {connection}")
            graph_context += f"- {connection}\n"

        self.log_subsection("Step 3: Combining Context")
        combined_context = f"Text Content:\n{context_text}\n\n{graph_context}"
        self.log("Combined context:")
        self.log_code_block(combined_context, "text")

        prompt = f"""Answer the question based on the provided context.

Context:
{combined_context}

Question: {question}

Answer:"""

        self.log_subsection("Step 4: LLM Answer Generation")
        self.log("Sending to LLM...")
        response = self.llm.invoke(prompt)

        self.log("\n✅ ANSWER:")
        answer = str(response.content)
        self.log_code_block(answer, "text")

        return answer


def main():
    """Main CLI application"""
    from dotenv import load_dotenv

    load_dotenv(".env.local")

    # Check for file argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_text_file.txt>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Validate file
    if not file_path.endswith(".txt"):
        print("Error: Only .txt files are supported")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Get credentials
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not all([neo4j_uri, neo4j_username, neo4j_password, gemini_api_key]):
        print("Error: Missing required environment variables")
        sys.exit(1)

    # Initialize pipeline (type-safe after validation)
    pipeline = GraphRAGPipeline(
        neo4j_uri=str(neo4j_uri),
        neo4j_username=str(neo4j_username),
        neo4j_password=str(neo4j_password),
        gemini_api_key=str(gemini_api_key),
    )

    # Process the text file
    print(f"\n🔄 Processing file: {file_path}\n")
    pipeline.process_text(text)
    pipeline.visualize_graph()

    # Interactive query loop
    pipeline.log_section("INTERACTIVE QUERY SESSION")
    print("\n" + "=" * 60)
    print("GraphRAG is ready! Enter your questions (type 'exit' to quit)")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("\n❓ Your question: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                pipeline.log("User exited the session")
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            answer = pipeline.query(question)
            print(f"\n✅ Answer: {answer}\n")

        except KeyboardInterrupt:
            pipeline.log("User interrupted the session")
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            pipeline.log(f"Error during query: {e}", "ERROR")
            print(f"\n❌ Error: {e}\n")

    pipeline.log_section("SESSION COMPLETE")
    pipeline.log(f"✅ Full log saved to: {pipeline.log_file}")
    print(f"\n📝 Full log saved to: {pipeline.log_file}")


if __name__ == "__main__":
    main()
