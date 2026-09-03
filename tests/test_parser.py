import json

import pytest

from src.parser import QQChatParser
from src.registry import COL_DATETIME, COL_IS_RECALLED


def make_message(**overrides):
    message = {
        "time": "2024-05-01T00:00:00+08:00",
        "sender": {"uin": 1, "name": "Alice"},
        "content": {"text": "hello", "resources": [], "mentions": []},
    }
    message.update(overrides)
    return message


def parse_messages(messages):
    payload = {"chatInfo": {"name": "test"}, "messages": messages}
    return QQChatParser().parse_json(json.dumps(payload))


def test_offset_iso_time_keeps_business_calendar_year():
    df, _ = parse_messages([make_message(time="2024-01-01T00:30:00+08:00")])

    parsed = df.iloc[0][COL_DATETIME]
    assert parsed.year == 2024
    assert parsed.month == 1
    assert parsed.day == 1
    assert parsed.hour == 0
    assert str(parsed.tzinfo) == "Asia/Shanghai"


def test_offset_time_is_converted_to_business_timezone_before_grouping():
    df, _ = parse_messages([make_message(time="2023-12-31T23:30:00-05:00")])

    parsed = df.iloc[0][COL_DATETIME]
    assert parsed.strftime("%Y-%m-%d %H:%M") == "2024-01-01 12:30"
    assert str(parsed.tzinfo) == "Asia/Shanghai"


@pytest.mark.parametrize(
    "timestamp",
    [
        1714521600,
        1714521600000,
        1714521600000000,
        1714521600000000000,
    ],
)
def test_numeric_time_infers_seconds_milliseconds_microseconds_and_nanoseconds(timestamp):
    df, _ = parse_messages([make_message(time=timestamp)])

    assert df.iloc[0][COL_DATETIME].strftime("%Y-%m-%d %H:%M:%S") == (
        "2024-05-01 08:00:00"
    )


def test_invalid_time_falls_back_to_timestamp():
    df, _ = parse_messages(
        [make_message(time="not-a-date", timestamp=1714521600)]
    )

    assert df.iloc[0][COL_DATETIME].strftime("%Y-%m-%d %H:%M:%S") == (
        "2024-05-01 08:00:00"
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("是", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("否", False),
        ("", False),
    ],
)
def test_recalled_values_are_explicitly_normalized(value, expected):
    df, _ = parse_messages([make_message(recalled=value)])

    assert bool(df.iloc[0][COL_IS_RECALLED]) is expected


def test_unknown_recalled_string_is_false_and_reported():
    df, meta = parse_messages([make_message(recalled="maybe")])

    assert bool(df.iloc[0][COL_IS_RECALLED]) is False
    assert meta["diagnostics"]["unknown_recalled_values"] == ["maybe"]
