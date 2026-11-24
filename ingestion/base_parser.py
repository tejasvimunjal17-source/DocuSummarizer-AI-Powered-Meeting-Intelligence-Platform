"""Common interface for the TXT, DOCX and PDF parsers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from ingestion.errors import InvalidDocumentError


@dataclass
class ParsedDocument:
    """Raw output of a parser (text is NOT normalized yet)."""

    text: str
    page_count: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    encoding: Optional[str] = None


class BaseParser(ABC):
    """A parser turns file bytes into a ParsedDocument or raises IngestionError."""

    @abstractmethod
    def parse_bytes(self, data: bytes) -> ParsedDocument:
        """Parse the full content of a file."""

    def parse_file(self, path: Union[str, Path]) -> ParsedDocument:
        """Read a file from disk and parse it."""
        file_path = Path(path)
        try:
            data = file_path.read_bytes()
        except FileNotFoundError as exc:
            raise InvalidDocumentError(f"File not found: {file_path.name}") from exc
        except OSError as exc:
            raise InvalidDocumentError(
                f"Could not read '{file_path.name}': {exc}"
            ) from exc
        return self.parse_bytes(data)
