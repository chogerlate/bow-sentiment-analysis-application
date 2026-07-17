import pytest

from sentiment_analysis.libs.xquik_export import parse_xquik_export


def test_parse_xquik_json_export() -> None:
    tweets = parse_xquik_export('{"tweets": [{"text": "  Great launch  "}, {"body": ""}]}')

    assert tweets == ["Great launch"]


def test_parse_xquik_jsonl_export() -> None:
    tweets = parse_xquik_export('{"full_text":"First tweet"}\n{"tweet":"Second tweet"}', "tweets.jsonl")

    assert tweets == ["First tweet", "Second tweet"]


def test_parse_xquik_csv_export() -> None:
    tweets = parse_xquik_export("id,content\n1,Needs faster support\n2,Works well\n", "tweets.csv")

    assert tweets == ["Needs faster support", "Works well"]


def test_preserve_newline_in_quoted_csv_text() -> None:
    tweets = parse_xquik_export(
        'id,content\n1,"First line\nsecond line"\n2,Works well\n',
        "tweets.csv",
    )

    assert tweets == ["First line second line", "Works well"]


def test_parse_whitespace_only_export() -> None:
    assert parse_xquik_export(" \n\t ", "tweets.jsonl") == []


def test_reject_malformed_jsonl_line() -> None:
    with pytest.raises(
        ValueError,
        match="JSONL export contains an invalid JSON line",
    ):
        parse_xquik_export('{"text":"valid"}\n{"text":', "tweets.jsonl")


def test_ignore_json_scalar_without_tweet_records() -> None:
    assert parse_xquik_export("42", "tweets.json") == []


def test_reject_csv_without_text_column() -> None:
    with pytest.raises(ValueError, match="CSV export needs"):
        parse_xquik_export("id,url\n1,https://example.com\n", "tweets.csv")


def test_report_invalid_json_file_as_json() -> None:
    with pytest.raises(ValueError, match="JSON export"):
        parse_xquik_export('{"tweets": [', "tweets.json")
