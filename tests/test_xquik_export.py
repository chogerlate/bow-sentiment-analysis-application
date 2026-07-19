"""Tests for Xquik export parsing."""

import pytest

from sentiment_analysis.libs.xquik_export import parse_xquik_export


def test_parse_xquik_json_export() -> None:
    """Parse standard and Xquik-specific text fields from JSON."""
    tweets = parse_xquik_export(
        '{"tweets": [{"text": "  Great launch  "}, {"tweet_text": "Second launch"}]}'
    )

    assert tweets == ["Great launch", "Second launch"]


def test_parse_xquik_jsonl_export() -> None:
    """Parse each supported JSONL record independently."""
    tweets = parse_xquik_export(
        '{"full_text":"First tweet"}\n'
        '{"tweet":"Second tweet"}\n'
        '{"tweet_text":"Third tweet"}',
        "tweets.jsonl",
    )

    assert tweets == ["First tweet", "Second tweet", "Third tweet"]


def test_parse_xquik_csv_export() -> None:
    """Parse a CSV export through a common content alias."""
    tweets = parse_xquik_export("id,content\n1,Needs faster support\n2,Works well\n", "tweets.csv")

    assert tweets == ["Needs faster support", "Works well"]


def test_parse_xquik_csv_tweet_text_export() -> None:
    """Parse the Xquik tweet_text field from CSV."""
    tweets = parse_xquik_export("id,tweet_text\n1,Works well\n", "tweets.csv")

    assert tweets == ["Works well"]


def test_preserve_newline_in_quoted_csv_text() -> None:
    """Preserve a quoted multiline CSV value as one normalized tweet."""
    tweets = parse_xquik_export(
        'id,content\n1,"First line\nsecond line"\n2,Works well\n',
        "tweets.csv",
    )

    assert tweets == ["First line second line", "Works well"]


def test_parse_whitespace_only_export() -> None:
    """Return no tweets for a whitespace-only export."""
    assert parse_xquik_export(" \n\t ", "tweets.jsonl") == []


def test_reject_malformed_jsonl_line() -> None:
    """Report a malformed JSONL record with a format-specific error."""
    with pytest.raises(
        ValueError,
        match="JSONL export contains an invalid JSON line",
    ):
        parse_xquik_export('{"text":"valid"}\n{"text":', "tweets.jsonl")


def test_ignore_json_scalar_without_tweet_records() -> None:
    """Ignore JSON scalar values that contain no tweet records."""
    assert parse_xquik_export("42", "tweets.json") == []


def test_reject_csv_without_text_column() -> None:
    """Reject CSV exports without a supported tweet text field."""
    with pytest.raises(ValueError, match="CSV export needs"):
        parse_xquik_export("id,url\n1,https://example.com\n", "tweets.csv")


def test_report_invalid_json_file_as_json() -> None:
    """Report malformed JSON files with a JSON-specific error."""
    with pytest.raises(ValueError, match="JSON export"):
        parse_xquik_export('{"tweets": [', "tweets.json")
