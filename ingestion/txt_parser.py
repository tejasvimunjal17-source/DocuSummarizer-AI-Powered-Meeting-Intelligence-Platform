"""Plain-text transcript parser.

Decoding order:
1. UTF-32 / UTF-16 if the file starts with that byte-order mark.
2. Files containing NUL bytes (and no such BOM) are rejected as binary.
3. UTF-8 (a UTF-8 BOM is removed automatically).
4. Windows-1252 as a fallback for old Windows files (a warning is added).
5. Otherwise the file is rejected. We never use a decoder that accepts
   every byte, because that would silently turn garbage into "text".
"""
from __future__ import annotations

import codecs

from ingestion.base_parser import BaseParser, ParsedDocument
from ingestion.errors import EmptyDocumentError, InvalidDocumentError


class TxtParser(BaseParser):
    def parse_bytes(self, data: bytes) -> ParsedDocument:
        if not data.strip():
            raise EmptyDocumentError("The TXT file is empty.")

        warnings = []
        if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
            text = self._decode(data, "utf-32")
            encoding = "utf-32"
        elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            text = self._decode(data, "utf-16")
            encoding = "utf-16"
        elif b"\x00" in data:
            raise InvalidDocumentError(
                "The file contains binary data and does not look like a text "
                "transcript (unsupported encoding or wrong file type)."
            )
        else:
            try:
                text = data.decode("utf-8-sig")
                encoding = "utf-8-sig" if data.startswith(codecs.BOM_UTF8) else "utf-8"
            except UnicodeDecodeError:
                try:
                    text = data.decode("cp1252")
                except UnicodeDecodeError as exc:
                    raise InvalidDocumentError(
                        "Could not decode the TXT file. Supported encodings: "
                        "UTF-8, UTF-16, UTF-32 and Windows-1252. Please re-save "
                        "the file as UTF-8."
                    ) from exc
                encoding = "cp1252"
                warnings.append(
                    "The file is not valid UTF-8; it was decoded as Windows-1252. "
                    "Some special characters may look different."
                )

        if not text.strip():
            raise EmptyDocumentError("The TXT file contains no text.")
        return ParsedDocument(text=text, warnings=warnings, encoding=encoding)

    @staticmethod
    def _decode(data: bytes, encoding: str) -> str:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError(
                f"The file starts like {encoding.upper()} text but could not be decoded."
            ) from exc
