"""Unit tests for the TXT parser."""
import codecs

import pytest

from ingestion.errors import EmptyDocumentError, InvalidDocumentError
from ingestion.txt_parser import TxtParser


def parse(data: bytes):
    return TxtParser().parse_bytes(data)


def test_valid_txt_extraction():
    result = parse(b"Rahul: Hello\nPriya: Hi")
    assert result.text == "Rahul: Hello\nPriya: Hi"
    assert result.encoding == "utf-8"
    assert result.warnings == []


def test_utf8_content_preserved():
    text = "Priya: Caf\u00e9, \u0928\u092e\u0938\u094d\u0924\u0947, \u4f60\u597d \u2014 \u201cquotes\u201d"
    assert parse(text.encode("utf-8")).text == text


def test_utf8_bom_is_removed():
    result = parse(codecs.BOM_UTF8 + b"Rahul: Hi")
    assert result.text == "Rahul: Hi"
    assert not result.text.startswith("\ufeff")
    assert result.encoding == "utf-8-sig"


def test_utf16_with_bom():
    text = "Rahul: Hi \u2603"
    result = parse(text.encode("utf-16"))
    assert result.text == text
    assert result.encoding == "utf-16"


def test_windows_1252_fallback_adds_warning():
    text = "Caf\u00e9 \u2018quoted\u2019"
    result = parse(text.encode("cp1252"))
    assert result.text == text
    assert result.encoding == "cp1252"
    assert result.warnings


def test_empty_file_rejected():
    with pytest.raises(EmptyDocumentError):
        parse(b"")


def test_whitespace_only_file_rejected():
    with pytest.raises(EmptyDocumentError):
        parse(b"  \n\t\n  ")


def test_binary_data_rejected():
    with pytest.raises(InvalidDocumentError):
        parse(b"abc\x00\x01\x02def")


def test_undecodable_bytes_rejected():
    # 0x81 0x8d 0x8f 0x90 0x9d are invalid in UTF-8 and undefined in cp1252
    with pytest.raises(InvalidDocumentError, match="decode"):
        parse(b"\x81\x8d\x8f\x90\x9d")


def test_parse_file_from_disk(tmp_path):
    path = tmp_path / "t.txt"
    path.write_bytes(b"Rahul: Hello")
    assert TxtParser().parse_file(path).text == "Rahul: Hello"


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(InvalidDocumentError, match="not found"):
        TxtParser().parse_file(tmp_path / "missing.txt")
