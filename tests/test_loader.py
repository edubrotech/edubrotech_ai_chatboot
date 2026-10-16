import pytest
from app.rag.loader import DocumentLoader


def test_load_text_bytes():
    content = b"This is a test document content."
    text, meta = DocumentLoader.load_from_bytes("sample.txt", content)
    assert "This is a test document content." in text
    assert meta["filename"] == "sample.txt"
    assert meta["extension"] == ".txt"


def test_load_json_bytes():
    content = b'{"name": "RentCrew", "active": true}'
    text, meta = DocumentLoader.load_from_bytes("data.json", content)
    assert '"name": "RentCrew"' in text
    assert meta["filename"] == "data.json"


def test_unsupported_format():
    with pytest.raises(ValueError) as excinfo:
        DocumentLoader.load_from_bytes("test.exe", b"\x00\x01\x02")
    assert "Unsupported file format" in str(excinfo.value)
