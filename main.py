"""
Basic GraphRAG Implementation using Neo4j Aura DB and Google Gemini
This script demonstrates how to:
1. Connect to Neo4j Aura DB
2. Use Google Gemini as the LLM
3. Build a simple knowledge graph from text
4. Query the graph using RAG
"""

import os
from typing import cast
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables from .env.local
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

# Set Google API key for langchain-google-genai (it expects GOOGLE_API_KEY)
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


def verify_neo4j_connection():
    """Test the Neo4j connection"""
    print("Testing Neo4j connection...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✓ Successfully connected to Neo4j Aura DB!")
        driver.close()
        return True
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        return False


def setup_graph_db():
    """Initialize Neo4j graph database connection"""
    print("\nInitializing Neo4j Graph...")
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    return graph


def setup_llm():
    """Initialize Google Gemini LLM"""
    print("Initializing Google Gemini...")

    # Check if API key is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n⚠️  WARNING: GOOGLE_API_KEY environment variable not set!")
        print("Please set it before running: export GOOGLE_API_KEY='your-api-key'")
        print("Get your API key from: https://aistudio.google.com/apikey")
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    print("✓ Gemini LLM initialized!")
    return llm


def setup_embeddings():
    """Initialize Google embeddings"""
    print("Initializing Google embeddings...")

    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n⚠️  WARNING: GOOGLE_API_KEY environment variable not set!")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    print("✓ Embeddings initialized!")
    return embeddings


def create_sample_knowledge_graph(graph, llm):
    """Create a simple knowledge graph from sample text"""
    print("\nCreating sample knowledge graph...")

    # Sample text about AI
    sample_text = """
    Artificial Intelligence (AI) is transforming the technology industry. 
    Machine Learning is a subset of AI that enables computers to learn from data.
    Deep Learning is a type of Machine Learning that uses neural networks.
    Natural Language Processing (NLP) is an AI technique used for understanding human language.
    Neo4j is a graph database that stores data in nodes and relationships.
    GraphRAG combines graph databases with Retrieval Augmented Generation for better AI responses.
    Google's Gemini is a powerful large language model developed by Google DeepMind.
    """

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
    )

    chunks = text_splitter.split_text(sample_text)

    # Store chunks as documents with metadata
    documents = [
        Document(page_content=chunk, metadata={"source": "sample_data", "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]

    print(f"✓ Created {len(documents)} document chunks")

    # Create simple graph structure
    # Clear existing data (for demo purposes)
    graph.query("MATCH (n) DETACH DELETE n")

    # Create nodes and relationships
    cypher_queries = [
        # Create concept nodes
        "CREATE (ai:Concept {name: 'Artificial Intelligence', description: 'Technology that enables machines to simulate human intelligence'})",
        "CREATE (ml:Concept {name: 'Machine Learning', description: 'AI subset enabling computers to learn from data'})",
        "CREATE (dl:Concept {name: 'Deep Learning', description: 'ML type using neural networks'})",
        "CREATE (nlp:Concept {name: 'Natural Language Processing', description: 'AI technique for understanding human language'})",
        "CREATE (neo4j:Technology {name: 'Neo4j', description: 'Graph database storing data in nodes and relationships'})",
        "CREATE (graphrag:Concept {name: 'GraphRAG', description: 'Combines graph databases with RAG for better AI responses'})",
        "CREATE (gemini:Technology {name: 'Gemini', description: 'Large language model by Google DeepMind'})",
        # Create relationships
        "MATCH (ml:Concept {name: 'Machine Learning'}), (ai:Concept {name: 'Artificial Intelligence'}) CREATE (ml)-[:SUBSET_OF]->(ai)",
        "MATCH (dl:Concept {name: 'Deep Learning'}), (ml:Concept {name: 'Machine Learning'}) CREATE (dl)-[:TYPE_OF]->(ml)",
        "MATCH (nlp:Concept {name: 'Natural Language Processing'}), (ai:Concept {name: 'Artificial Intelligence'}) CREATE (nlp)-[:TECHNIQUE_OF]->(ai)",
        "MATCH (graphrag:Concept {name: 'GraphRAG'}), (neo4j:Technology {name: 'Neo4j'}) CREATE (graphrag)-[:USES]->(neo4j)",
    ]

    for query in cypher_queries:
        graph.query(query)

    print("✓ Knowledge graph created with concepts and relationships!")
    return documents


def setup_vector_store(graph, embeddings, documents):
    """Create a vector store in Neo4j"""
    print("\nSetting up vector store...")

    if embeddings is None:
        print("⚠️  Skipping vector store setup - embeddings not available")
        return None

    # Drop existing vector index if it exists (to avoid dimension mismatch)
    try:
        graph.query("DROP INDEX document_embeddings IF EXISTS")
        print("✓ Cleaned up existing vector index")
    except Exception as e:
        print(f"Note: {e}")

    # Create vector index in Neo4j
    vector_store = Neo4jVector.from_documents(
        documents,
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="document_embeddings",
        node_label="DocumentChunk",
        embedding_node_property="embedding",
        text_node_property="text",
    )

    print("✓ Vector store created!")
    return vector_store


def query_graph(graph, query_text):
    """Query the knowledge graph directly"""
    print(f"\nQuerying graph: {query_text}")

    # Example: Find all concepts
    result = graph.query(
        """
        MATCH (c:Concept)
        RETURN c.name as name, c.description as description
        """
    )

    print("\nConcepts in the graph:")
    for record in result:
        print(f"  - {record['name']}: {record['description']}")

    return result


def rag_query(vector_store, llm, question):
    """Perform RAG query combining vector search and LLM"""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print("=" * 60)

    if vector_store is None or llm is None:
        print("⚠️  Cannot perform RAG query - vector store or LLM not available")
        return None

    # Search for relevant documents
    relevant_docs = vector_store.similarity_search(question, k=2)

    print("\nRelevant context found:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"{i}. {doc.page_content[:100]}...")

    # Build context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Create prompt
    prompt = f"""Based on the following context, please answer the question.
    
Context:
{context}

Question: {question}

Answer:"""

    # Get response from LLM
    response = llm.invoke(prompt)

    print(f"\nAnswer: {response.content}")
    print("=" * 60)

    return response.content


def main():
    """Main function demonstrating GraphRAG with Neo4j and Gemini"""
    print("=" * 60)
    print("Basic GraphRAG with Neo4j Aura DB and Google Gemini")
    print("=" * 60)

    # Step 1: Verify Neo4j connection
    if not verify_neo4j_connection():
        return

    # Step 2: Setup components
    graph = setup_graph_db()
    llm = setup_llm()
    embeddings = setup_embeddings()

    # Step 3: Create knowledge graph
    documents = create_sample_knowledge_graph(graph, llm)

    # Step 4: Setup vector store
    vector_store = setup_vector_store(graph, embeddings, documents)

    # Step 5: Query the graph directly
    query_graph(graph, "Show all concepts")

    # Step 6: Demonstrate RAG queries
    if vector_store and llm:
        print("\n" + "=" * 60)
        print("Demonstrating GraphRAG Queries")
        print("=" * 60)

        questions = [
            "What is the relationship between Machine Learning and Artificial Intelligence?",
            "What is GraphRAG?",
            "Tell me about Neo4j",
        ]

        for question in questions:
            rag_query(vector_store, llm, question)

    print("\n✓ GraphRAG demonstration complete!")
    print("\nNext steps:")
    print("1. Set your GOOGLE_API_KEY environment variable")
    print("2. Run the script to see RAG in action")
    print("3. Modify the sample text to add your own data")
    print("4. Explore the Neo4j Browser to visualize your graph")


if __name__ == "__main__":
    main()
