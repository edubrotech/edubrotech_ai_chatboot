import io
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from pypdf import PdfReader
from app.core.logging import get_logger
from app.core.security import sanitize_text

logger = get_logger(__name__)


class DocumentLoader:
    """Loads and extracts text and metadata from supported file formats (txt, md, json, pdf)."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".pdf"}

    @classmethod
    def load_from_bytes(cls, filename: str, content: bytes) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from raw bytes."""
        ext = Path(filename).suffix.lower()
        metadata: Dict[str, Any] = {
            "filename": filename,
            "extension": ext,
            "size_bytes": len(content),
        }

        if ext in {".txt", ".md", ".csv"}:
            text = content.decode("utf-8", errors="replace")
            return sanitize_text(text), metadata

        elif ext == ".json":
            try:
                data = json.loads(content.decode("utf-8", errors="replace"))
                text = json.dumps(data, indent=2)
                return sanitize_text(text), metadata
            except Exception as e:
                logger.error("Error parsing JSON file %s: %s", filename, str(e))
                raise ValueError(f"Invalid JSON file format: {str(e)}")

        elif ext == ".pdf":
            try:
                pdf_file = io.BytesIO(content)
                reader = PdfReader(pdf_file)
                pages_text = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages_text.append(f"--- Page {i + 1} ---\n{page_text}")
                metadata["page_count"] = len(reader.pages)
                full_text = "\n\n".join(pages_text)
                return sanitize_text(full_text), metadata
            except Exception as e:
                logger.error("Error extracting PDF text from %s: %s", filename, str(e))
                raise ValueError(f"Failed to extract text from PDF: {str(e)}")

        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: {', '.join(cls.SUPPORTED_EXTENSIONS)}")

    @classmethod
    def load_from_path(cls, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata directly from a local file path."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found at: {file_path}")

        content = path.read_bytes()
        return cls.load_from_bytes(path.name, content)
