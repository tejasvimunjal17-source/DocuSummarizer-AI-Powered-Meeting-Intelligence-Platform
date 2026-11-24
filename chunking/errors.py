"""Exceptions raised by the chunking package."""


class ChunkingError(Exception):
    """Raised when chunking cannot be done safely (for example, text would be lost)."""
