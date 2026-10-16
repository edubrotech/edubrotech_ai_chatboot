import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path so scripts can run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import get_logger, setup_logging
from app.rag.chunker import RecursiveCharacterChunker
from app.rag.loader import DocumentLoader
from app.services.document_service import DocumentService

logger = get_logger("scripts.ingest")


async def ingest_directory(target_dir: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Scan and ingest all supported files from a directory."""
    setup_logging()
    path = Path(target_dir)

    if not path.exists():
        logger.error("Target path does not exist: %s", target_dir)
        return

    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_service = DocumentService(chunker=chunker)

    files_to_process = []
    if path.is_file():
        files_to_process = [path]
    else:
        for ext in DocumentLoader.SUPPORTED_EXTENSIONS:
            files_to_process.extend(path.rglob(f"*{ext}"))

    if not files_to_process:
        logger.warning("No supported document files found in %s", target_dir)
        return

    logger.info("Found %d file(s) to ingest.", len(files_to_process))
    total_chunks = 0
    success_count = 0

    for file_path in files_to_process:
        try:
            logger.info("Ingesting: %s", file_path.name)
            res = await doc_service.ingest_file_path(str(file_path))
            logger.info("-> Indexed %s: %d chunks (%s)", file_path.name, res.chunk_count, res.status)
            total_chunks += res.chunk_count
            success_count += 1
        except Exception as e:
            logger.error("-> Failed to ingest %s: %s", file_path.name, str(e))

    logger.info(
        "Ingestion completed: %d/%d files processed successfully, %d total chunks indexed.",
        success_count,
        len(files_to_process),
        total_chunks,
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest files into AI Chatbot knowledge base.")
    parser.add_argument(
        "--path",
        type=str,
        default="./data",
        help="Path to directory or file to ingest (default: ./data)",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap size in characters")
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
