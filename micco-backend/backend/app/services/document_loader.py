"""
Document Loader Service
Handles loading and extracting text from various document formats.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import logging

logger = logging.getLogger(__name__)


class LoadedDocument(NamedTuple):
    """Represents a loaded document with its content and metadata."""
    content: str
    source: str
    file_type: str
    page_count: int = 1


def load_txt_file(file_path: Path) -> LoadedDocument:
    """Load a plain text file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return LoadedDocument(
            content=content,
            source=str(file_path),
            file_type="txt",
            page_count=1
        )
    except UnicodeDecodeError:
        # Try with different encoding
        content = file_path.read_text(encoding="latin-1")
        return LoadedDocument(
            content=content,
            source=str(file_path),
            file_type="txt",
            page_count=1
        )


def load_pdf_file(file_path: Path) -> LoadedDocument:
    """Load a PDF file and extract text."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        content = "\n\n".join(pages_text)

        return LoadedDocument(
            content=content,
            source=str(file_path),
            file_type="pdf",
            page_count=len(reader.pages)
        )
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        raise ValueError(f"Failed to load PDF: {e}")


def load_markdown_file(file_path: Path) -> LoadedDocument:
    """Load a markdown file."""
    content = file_path.read_text(encoding="utf-8")
    return LoadedDocument(
        content=content,
        source=str(file_path),
        file_type="md",
        page_count=1
    )


def load_doc_file(file_path: Path) -> LoadedDocument:
    """Load a legacy .doc (Word 97-2003 OLE2 binary) file and extract text.

    Uses olefile to read the WordDocument stream and extracts text via
    UTF-16LE decoding (the standard encoding for Word binary format).
    """
    try:
        import olefile
        import re
        import struct

        with olefile.OleFileIO(str(file_path)) as ole:
            if not ole.exists("WordDocument"):
                raise ValueError("WordDocument stream not found — invalid .doc file")

            stream = ole.openstream("WordDocument").read()

            # FIB offsets: fcMin at 0x18 (text start), ccpText at 0x1C (char count)
            fc_min = struct.unpack_from("<I", stream, 0x18)[0]
            ccp_text = struct.unpack_from("<I", stream, 0x1C)[0]

            if ccp_text == 0 or fc_min >= len(stream):
                # Fallback: scan entire stream for unicode text
                raw = stream
            else:
                raw = stream[fc_min: fc_min + ccp_text * 2]

            text = raw.decode("utf-16-le", errors="ignore")

            # Remove binary/control characters but keep newlines and tabs
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
            text = re.sub(r" {2,}", " ", text)
            text = text.strip()

            if len(text) < 20:
                raise ValueError("Extracted text too short — file may be corrupted or encrypted")

            # Estimate page count from FIB cpnBtePap (rough: 1 page per 3000 chars)
            page_count = max(1, len(text) // 3000)

            logger.info(
                f"Loaded .doc {file_path.name}: {len(text)} chars, ~{page_count} pages"
            )
            return LoadedDocument(
                content=text,
                source=str(file_path),
                file_type="doc",
                page_count=page_count,
            )

    except ImportError:
        raise ValueError(
            "olefile is required to read .doc files. Run: pip install olefile"
        )
    except Exception as e:
        logger.error(f"Error loading .doc {file_path}: {e}")
        raise ValueError(f"Failed to load .doc file: {e}")


def load_document(file_path: str | Path) -> LoadedDocument:
    """
    Load a document based on its file type.

    Supported formats: .txt, .pdf, .md, .doc

    Args:
        file_path: Path to the document file

    Returns:
        LoadedDocument with content and metadata

    Raises:
        ValueError: If file type is not supported or file cannot be read
    """
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    loaders = {
        ".txt": load_txt_file,
        ".pdf": load_pdf_file,
        ".md": load_markdown_file,
        ".doc": load_doc_file,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(loaders.keys())}")

    return loader(path)


def get_supported_extensions() -> list[str]:
    """Return list of supported file extensions."""
    return [".txt", ".pdf", ".md", ".doc"]
