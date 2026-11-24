"""Unit tests for text normalization."""
from ingestion.text_normalizer import normalize_text


def test_extra_whitespace_collapsed():
    assert normalize_text("Rahul:   We   should\tlaunch.   ") == "Rahul: We should launch."


def test_multiple_blank_lines_reduced_to_one():
    assert normalize_text("A\n\n\n\n\nB") == "A\n\nB"


def test_whitespace_only_lines_count_as_blank():
    assert normalize_text("A\n   \n \t \n\n\nB") == "A\n\nB"


def test_windows_mac_unix_line_endings():
    assert normalize_text("A\r\nB\rC\nD") == "A\nB\nC\nD"


def test_leading_and_trailing_whitespace_removed():
    assert normalize_text("\n\n  Hello  \n\n") == "Hello"


def test_unicode_preserved():
    text = "Caf\u00e9 \u2014 \u0928\u092e\u0938\u094d\u0924\u0947 \u4f60\u597d \U0001F600"
    assert normalize_text(text) == text


def test_decomposed_accent_is_composed():
    assert normalize_text("Cafe\u0301") == "Caf\u00e9"


def test_punctuation_preserved():
    text = 'Priya: "Wait \u2014 what?!" Costs: $1,000; ok (maybe)... #5 @home 50%'
    assert normalize_text(text) == text


def test_speaker_labels_and_timestamps_preserved():
    text = "[10:30] Rahul: Hi\n\n[10:32] Priya: Yes\n\nSpeaker 1: Ok"
    assert normalize_text(text) == text


def test_separate_speakers_not_merged():
    result = normalize_text("Rahul: Hi\nPriya: Hello")
    assert result.split("\n") == ["Rahul: Hi", "Priya: Hello"]


def test_non_breaking_space_becomes_space():
    assert normalize_text("a\u00a0b") == "a b"


def test_zero_width_space_removed():
    assert normalize_text("a\u200bb") == "ab"


def test_control_characters_removed():
    assert normalize_text("a\x00b\x07c") == "abc"


def test_form_feed_becomes_newline():
    assert normalize_text("a\x0cb") == "a\nb"


def test_empty_input():
    assert normalize_text("") == ""
    assert normalize_text("   \n\n  ") == ""


def test_idempotent():
    messy = "  A  \r\n\r\n\r\n\r\n B\u00a0c \t d  "
    once = normalize_text(messy)
    assert normalize_text(once) == once
