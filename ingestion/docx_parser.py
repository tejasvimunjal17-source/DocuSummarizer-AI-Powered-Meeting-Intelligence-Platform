"""DOCX transcript parser (python-docx).

Extracted: body paragraphs and table rows, in document order.
NOT extracted: headers, footers, footnotes, text boxes, comments, images.
Table rows are returned as tab-separated cell text and a warning is added.
"""
from __future__ import annotations

import io

from ingestion.base_parser import BaseParser, ParsedDocument
from ingestion.errors import EmptyDocumentError, IngestionError, InvalidDocumentError


def _load_docx():
    try:
        import docx
    except ImportError as exc:
        raise IngestionError(
            "python-docx is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return docx


class DocxParser(BaseParser):
    def parse_bytes(self, data: bytes) -> ParsedDocument:
        if not data:
            raise EmptyDocumentError("The DOCX file is empty (0 bytes).")

        docx = _load_docx()
        try:
            document = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise InvalidDocumentError(
                "Could not open the file as a DOCX document. It may be corrupted "
                "or may not be a real .docx file."
            ) from exc

        try:
            lines, table_count = self._extract_lines(document)
        except Exception as exc:
            raise InvalidDocumentError(
                f"Failed to read the content of the DOCX file: {exc}"
            ) from exc

        text = "\n".join(lines)
        if not text.strip():
            raise EmptyDocumentError("The DOCX document contains no text.")

        warnings = []
        if table_count:
            warnings.append(
                f"The document contains {table_count} table(s). Table text was "
                "extracted row by row with cells separated by tabs."
            )
        return ParsedDocument(text=text, warnings=warnings)

    @staticmethod
    def _extract_lines(document):
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        lines = []
        table_count = 0
        for child in document.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                lines.append(Paragraph(child, document).text)
            elif tag == "tbl":
                table_count += 1
                for row in Table(child, document).rows:
                    cells, seen = [], set()
                    for cell in row.cells:
                        if id(cell._tc) in seen:  # merged cell repeats
                            continue
                        seen.add(id(cell._tc))
                        cells.append(cell.text.strip())
                    lines.append("\t".join(cells))
        return lines, table_count
