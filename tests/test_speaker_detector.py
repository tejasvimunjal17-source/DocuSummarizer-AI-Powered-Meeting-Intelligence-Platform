"""Unit tests for speaker-turn detection."""
from ingestion.speaker_detector import detect_speaker_turns, get_unique_speakers


def speakers(turns):
    return [t.speaker for t in turns]


def test_simple_speaker_labels():
    turns = detect_speaker_turns(
        "Rahul: We should launch the project next month.\n\n"
        "Priya: I will prepare the presentation."
    )
    assert speakers(turns) == ["Rahul", "Priya"]
    assert turns[0].text == "We should launch the project next month."
    assert turns[1].text == "I will prepare the presentation."
    assert [t.index for t in turns] == [0, 1]
    assert all(t.timestamp is None for t in turns)


def test_timestamped_conversation():
    turns = detect_speaker_turns(
        "[10:30] Rahul: We need to finalize the budget.\n"
        "[10:32] Priya: I agree with that decision."
    )
    assert speakers(turns) == ["Rahul", "Priya"]
    assert [t.timestamp for t in turns] == ["10:30", "10:32"]
    assert turns[0].text == "We need to finalize the budget."


def test_timestamp_variants():
    turns = detect_speaker_turns(
        "(10:30:15) Rahul: a\n10:31 Amit: b\n[10:32 AM] Priya: c"
    )
    assert speakers(turns) == ["Rahul", "Amit", "Priya"]
    assert [t.timestamp for t in turns] == ["10:30:15", "10:31", "10:32 AM"]


def test_generic_speaker_labels():
    turns = detect_speaker_turns(
        "Speaker 1: The first task is complete.\n"
        "Speaker 2: We should begin the next phase."
    )
    assert speakers(turns) == ["Speaker 1", "Speaker 2"]


def test_multiple_speakers_and_unique_list():
    turns = detect_speaker_turns("Rahul: a\nPriya: b\nAmit: c\nRahul: d")
    assert len(turns) == 4
    assert get_unique_speakers(turns) == ["Rahul", "Priya", "Amit"]


def test_single_speaker_repeated_is_accepted():
    turns = detect_speaker_turns("Rahul: a\nRahul: b")
    assert speakers(turns) == ["Rahul", "Rahul"]


def test_multiline_statement():
    turns = detect_speaker_turns(
        "Rahul: First line\ncontinues here\nand more\nPriya: ok"
    )
    assert turns[0].text == "First line\ncontinues here\nand more"
    assert turns[1].speaker == "Priya"


def test_label_alone_on_line_then_statement():
    turns = detect_speaker_turns("Rahul:\nWe should go\nPriya: ok")
    assert turns[0].speaker == "Rahul"
    assert turns[0].text == "We should go"


def test_no_speaker_information_keeps_text_as_unknown():
    turns = detect_speaker_turns(
        "This is just some notes.\nSecond line.\n\nAnother paragraph."
    )
    assert speakers(turns) == [None, None]
    assert turns[0].text == "This is just some notes.\nSecond line."
    assert turns[1].text == "Another paragraph."
    assert get_unique_speakers(turns) == []


def test_ordinary_sentences_are_not_speaker_labels():
    turns = detect_speaker_turns(
        "Note: remember the budget.\nThe plan is: launch next month.\n"
        "Reminder: call Priya at 10:30."
    )
    assert all(t.speaker is None for t in turns)


def test_url_and_time_inside_sentence_are_not_labels():
    turns = detect_speaker_turns(
        "See http://example.com for details.\nMeeting starts at 10:30: be on time."
    )
    assert all(t.speaker is None for t in turns)


def test_single_unconfirmed_label_is_not_trusted():
    turns = detect_speaker_turns("Budget: 5000")
    assert speakers(turns) == [None]
    assert turns[0].text == "Budget: 5000"


def test_lowercase_names_not_recognized_known_limitation():
    turns = detect_speaker_turns("rahul: hi\npriya: yo")
    assert all(t.speaker is None for t in turns)


def test_text_before_first_label_is_unknown_speaker():
    turns = detect_speaker_turns("Meeting notes\n\nRahul: Hi\nPriya: Hello")
    assert speakers(turns) == [None, "Rahul", "Priya"]
    assert turns[0].text == "Meeting notes"


def test_unlabeled_text_after_blank_line_is_not_given_to_previous_speaker():
    turns = detect_speaker_turns("Rahul: Hello\n\nstray text\nPriya: Hi")
    assert speakers(turns) == ["Rahul", None, "Priya"]
    assert turns[1].text == "stray text"


def test_unknown_label_gives_none_speaker():
    turns = detect_speaker_turns("Unknown: Hello\nRahul: Hi")
    assert speakers(turns) == [None, "Rahul"]
    assert turns[0].text == "Hello"


def test_colon_and_time_inside_statement_are_kept():
    turns = detect_speaker_turns(
        "Rahul: Agenda is: budget, hiring at 10:30\nPriya: ok"
    )
    assert turns[0].speaker == "Rahul"
    assert turns[0].timestamp is None
    assert turns[0].text == "Agenda is: budget, hiring at 10:30"


def test_titles_and_multi_word_names():
    turns = detect_speaker_turns(
        "Dr. Rahul Sharma: Welcome\nMs. Priya K.: Thanks"
    )
    assert speakers(turns) == ["Dr. Rahul Sharma", "Ms. Priya K."]


def test_non_latin_speaker_names():
    turns = detect_speaker_turns(
        "\u0930\u093e\u0939\u0941\u0932: \u0928\u092e\u0938\u094d\u0924\u0947\n"
        "\u092a\u094d\u0930\u093f\u092f\u093e: \u0939\u093e\u0901"
    )
    assert speakers(turns) == ["\u0930\u093e\u0939\u0941\u0932", "\u092a\u094d\u0930\u093f\u092f\u093e"]


def test_empty_input_gives_no_turns():
    assert detect_speaker_turns("") == []
    assert detect_speaker_turns("  \n ") == []


def test_labels_without_text_do_not_lose_content():
    turns = detect_speaker_turns("Rahul:\nPriya:")
    assert len(turns) == 1
    assert turns[0].speaker is None
    assert "Rahul:" in turns[0].text


def test_timestamp_in_brackets_after_name():
    turns = detect_speaker_turns(
        "Alice [10:00 AM]: Welcome everyone.\n"
        "Bob [10:01 AM]: I will complete the backend by Friday."
    )
    assert speakers(turns) == ["Alice", "Bob"]
    assert [t.timestamp for t in turns] == ["10:00 AM", "10:01 AM"]
    assert turns[0].text == "Welcome everyone."


def test_timestamp_in_parentheses_after_name():
    turns = detect_speaker_turns("Rahul (10:30:15): a\nPriya (10:31): b")
    assert speakers(turns) == ["Rahul", "Priya"]
    assert [t.timestamp for t in turns] == ["10:30:15", "10:31"]


def test_role_in_parentheses_still_not_recognized():
    turns = detect_speaker_turns("Rahul (PM): hi\nPriya (QA): yo")
    assert all(t.speaker is None for t in turns)
