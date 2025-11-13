"""
GraphRAG Demo - Automatic Entity & Relationship Extraction Example
This demonstrates how GraphRAG works with a sample text.
"""

import os
from dotenv import load_dotenv
from main import GraphRAGPipeline

# Load environment variables
load_dotenv(".env.local")

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


def main():
    """Run GraphRAG demo with sample text"""
    # Get credentials from environment
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not all([neo4j_uri, neo4j_username, neo4j_password, gemini_api_key]):
        raise ValueError("Missing required environment variables")

    # Initialize pipeline (type-safe after validation)
    pipeline = GraphRAGPipeline(
        neo4j_uri=str(neo4j_uri),
        neo4j_username=str(neo4j_username),
        neo4j_password=str(neo4j_password),
        gemini_api_key=str(gemini_api_key),
    )

    # Process the sample text
    pipeline.process_text(SAMPLE_TEXT)

    # Visualize the graph
    pipeline.visualize_graph()

    # Run example queries
    pipeline.log_section("DEMO QUERIES")

    questions = [
        "What companies did Elon Musk found?",
        "What AI models were created and by whom?",
        "Tell me about graph databases and how they're used",
    ]

    for question in questions:
        answer = pipeline.query(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}\n")

    pipeline.log_section("DEMO COMPLETE")
    pipeline.log(f"✅ Full log saved to: {pipeline.log_file}")
    pipeline.log("\n📝 Check the logs/ directory for complete execution details!")


if __name__ == "__main__":
    main()
