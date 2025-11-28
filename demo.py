"""GraphRAG Demo - Demonstrates the production-ready pipeline."""

import asyncio
from src.config import PipelineConfig
from src.pipeline import GraphRAGPipeline
from src.logging_config import setup_logging

# Sample text for demonstration
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

Google developed the Gemini AI model, which powers many AI applications. Gemini is a multimodal
model that can understand text, images, and code. Google Cloud provides infrastructure for running
AI workloads at scale.
"""


async def main():
    """Run the GraphRAG demo."""
    # Set up logging
    setup_logging()

    print("=" * 60)
    print("GraphRAG Demo - Production Pipeline")
    print("=" * 60)

    # Load configuration
    try:
        config = PipelineConfig.from_env()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease create a .env.local file with:")
        print("  NEO4J_URI=neo4j+s://...")
        print("  NEO4J_USERNAME=neo4j")
        print("  NEO4J_PASSWORD=...")
        print("  GEMINI_API_KEY=...")
        return

    # Create pipeline
    with GraphRAGPipeline(config) as pipeline:
        # Initialize database with constraints and indexes
        print("\n📦 Initializing database...")
        pipeline.db.initialize()
        pipeline.db.setup_vector_index(
            index_name=config.vector_index_name,
            dimensions=config.vector_dimensions,
        )
        print("✓ Database initialized with constraints and indexes")

        # Ingest sample text
        print("\n📝 Ingesting sample document...")
        result = await pipeline.ingest_text(
            text=SAMPLE_TEXT,
            document_id="demo-document",
            document_title="AI Companies Overview",
            metadata={"source": "demo", "type": "sample"},
        )

        print(f"\n✓ Ingestion complete!")
        print(f"  - Document ID: {result.get('document_id')}")
        print(f"  - Chunks: {result.get('chunks')}")
        print(f"  - Entities: {result.get('entities')}")
        print(f"  - Relationships: {result.get('relationships')}")

        # Show database statistics
        print("\n📊 Database Statistics:")
        stats = pipeline.db.get_statistics()
        if stats.get("nodes_by_label"):
            for label, count in stats["nodes_by_label"].items():
                print(f"  - {label}: {count} nodes")

        # Run example queries
        print("\n" + "=" * 60)
        print("Running Example Queries")
        print("=" * 60)

        questions = [
            "Who is the CEO of Tesla?",
            "What AI models were created and by whom?",
            "Tell me about graph databases and how they're used with AI",
            "What companies are working on AI safety?",
        ]

        for question in questions:
            print(f"\n❓ Question: {question}")
            try:
                answer = await pipeline.query(question)
                print(f"✅ Answer: {answer}")
            except Exception as e:
                print(f"❌ Error: {e}")

        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print(f"\n📝 Log file: {pipeline.md_logger.log_file}")
        print("\nNext steps:")
        print("  1. Run 'python -m src.cli list' to see documents")
        print("  2. Run 'python -m src.cli query' for interactive queries")
        print("  3. Run 'python -m src.cli stats' for database statistics")


if __name__ == "__main__":
    asyncio.run(main())
