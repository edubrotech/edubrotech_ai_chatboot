import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import get_logger, setup_logging
from app.vectorstore.repository import VectorRepository
from scripts.ingest import ingest_directory

logger = get_logger("scripts.reindex")


async def reindex_all(data_dir: str):
    """Reset the vector database collection and re-ingest all files."""
    setup_logging()
    logger.info("Starting knowledge base reindex from: %s", data_dir)
    
    repo = VectorRepository()
    logger.info("Purging existing collection...")
    await repo.reset_collection()
    logger.info("Collection reset complete.")

    logger.info("Re-ingesting documents from %s...", data_dir)
    await ingest_directory(data_dir)
    logger.info("Reindexing process finished.")


def main():
    parser = argparse.ArgumentParser(description="Purge and reindex all documents.")
    parser.add_argument(
        "--path",
        type=str,
        default="./data",
        help="Path to documents directory (default: ./data)",
    )
    args = parser.parse_args()
    asyncio.run(reindex_all(args.path))


if __name__ == "__main__":
    main()
