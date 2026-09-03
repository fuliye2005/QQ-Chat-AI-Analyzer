# src/parser.py

"""
QQ Chat Parser Module
=====================
负责解析 QQChatExporter 导出的 JSON 文件。
"""

import json
import math
import numbers
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.registry import *


class QQChatParser:
    """将 QQChatExporter 导出的消息转换为结构化 DataFrame。"""

    def __init__(self):
        self.diagnostics = {
            "skipped_messages": 0,
            "unknown_recalled_values": [],
        }

    def parse_json(self, file_content: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """解析 JSON 字符串，并返回消息 DataFrame 与文件元数据。"""
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON format") from exc

        if not isinstance(data, dict):
            raise ValueError("JSON 根节点必须是对象")

        messages = data.get(JSON_FIELD_MESSAGES, [])
        if not isinstance(messages, list):
            raise ValueError("JSON messages 字段必须是数组")

        parsed_data = []
        for msg in messages:
            if not isinstance(msg, dict):
                self.diagnostics["skipped_messages"] += 1
                continue
            parsed_msg = self._parse_single_message(msg)
            if parsed_msg:
                parsed_data.append(parsed_msg)
            else:
                self.diagnostics["skipped_messages"] += 1

        df = pd.DataFrame(parsed_data)
        chat_info = data.get(JSON_FIELD_CHAT_INFO, {})
        statistics = data.get(JSON_FIELD_STATISTICS, {})
        time_range = statistics.get(JSON_FIELD_TIME_RANGE, {})
        if not isinstance(chat_info, dict):
            chat_info = {}
        if not isinstance(statistics, dict):
            statistics = {}
        if not isinstance(time_range, dict):
            time_range = {}

        meta = {
            "chat_name": chat_info.get(JSON_FIELD_CHAT_NAME, UNKNOWN_GROUP_NAME),
            "total_messages": statistics.get(JSON_FIELD_TOTAL_MESSAGES, 0),
            "start_time": time_range.get(JSON_FIELD_START),
            "end_time": time_range.get(JSON_FIELD_END),
            "diagnostics": {
                "skipped_messages": self.diagnostics["skipped_messages"],
                "unknown_recalled_values": list(
                    self.diagnostics["unknown_recalled_values"]
                ),
            },
        }
        return df, meta

    def _parse_single_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析单条消息；无法确定时间时返回 None。"""
        dt = self._parse_message_time(msg)
        if dt is None:
            return None

        sender = msg.get(JSON_FIELD_SENDER, {})
        if not isinstance(sender, dict):
            sender = {}
        user_id = sender.get(JSON_FIELD_SENDER_UIN) or sender.get(
            JSON_FIELD_SENDER_UID, "unknown"
        )
        user_name = sender.get(JSON_FIELD_SENDER_CARD) or sender.get(
            JSON_FIELD_SENDER_NAME, UNKNOWN_USER_NAME
        )

        content_obj = msg.get(JSON_FIELD_CONTENT, {})
        if not isinstance(content_obj, dict):
            content_obj = {JSON_FIELD_TEXT: str(content_obj)}
        text_content = content_obj.get(JSON_FIELD_TEXT, "")
        if text_content is None:
            text_content = ""
        resources = content_obj.get(JSON_FIELD_RESOURCES, [])
        if not isinstance(resources, list):
            resources = []

        resource_types = [
            resource.get("type")
            for resource in resources
            if isinstance(resource, dict)
        ]
        image_count = resource_types.count("image")
        msg_type = MSG_TYPE_TEXT
        if "image" in resource_types:
            msg_type = MSG_TYPE_IMAGE
        elif "video" in resource_types:
            msg_type = MSG_TYPE_VIDEO
        elif "file" in resource_types:
            msg_type = MSG_TYPE_FILE
        if text_content and resource_types:
            msg_type = MSG_TYPE_MIXED

        raw_recalled = msg.get(JSON_FIELD_IS_RECALLED)
        if raw_recalled is None:
            raw_recalled = msg.get(JSON_FIELD_RECALLED, False)
        is_recalled = self._normalize_bool(raw_recalled)
        if is_recalled:
            msg_type = MSG_TYPE_RECALLED

        raw_mentions = content_obj.get(JSON_FIELD_MENTIONS, [])
        if not isinstance(raw_mentions, list):
            raw_mentions = []
        mentions = [
            mention.get("name")
            for mention in raw_mentions
            if isinstance(mention, dict)
        ]

        return {
            COL_DATETIME: dt,
            COL_DATE: dt.date(),
            COL_TIME: dt.time(),
            COL_HOUR: dt.hour,
            COL_USER_ID: str(user_id),
            COL_USER_NAME: user_name,
            COL_CONTENT: text_content,
            COL_TYPE: msg_type,
            COL_IS_RECALLED: is_recalled,
            COL_MENTIONS: mentions,
            COL_IMAGE_COUNT: image_count,
        }

    def _parse_message_time(self, msg: Dict[str, Any]) -> Optional[pd.Timestamp]:
        """解析 time，失败后回退 timestamp。"""
        raw_time = msg.get(JSON_FIELD_TIME)
        if not self._is_missing(raw_time):
            parsed = self._parse_time_value(raw_time)
            if parsed is not None:
                return parsed

        raw_timestamp = msg.get(JSON_FIELD_TIMESTAMP)
        if self._is_missing(raw_timestamp):
            return None
        return self._parse_time_value(raw_timestamp)

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """判断标量是否为空；数组等非标量不强行转换为布尔值。"""
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        try:
            result = pd.isna(value)
        except (TypeError, ValueError):
            return False
        if isinstance(result, bool):
            return result
        if type(result).__name__ == "bool_":
            return bool(result)
        return False

    @staticmethod
    def _infer_timestamp_unit(value: float) -> str:
        """根据 Unix 时间戳数量级推断单位。"""
        magnitude = abs(value)
        if magnitude >= 1e17:
            return "ns"
        if magnitude >= 1e14:
            return "us"
        if magnitude >= 1e11:
            return "ms"
        return "s"

    @classmethod
    def _parse_time_value(cls, value: Any) -> Optional[pd.Timestamp]:
        """解析字符串或数值时间，并统一到业务时区。"""
        if cls._is_missing(value) or isinstance(value, bool):
            return None

        try:
            numeric_timestamp = None
            if isinstance(value, numbers.Real):
                numeric_timestamp = float(value)
            elif isinstance(value, str):
                try:
                    numeric_timestamp = float(value.strip())
                except ValueError:
                    numeric_timestamp = None

            if numeric_timestamp is not None:
                if not math.isfinite(numeric_timestamp):
                    return None
                parsed = pd.to_datetime(
                    numeric_timestamp,
                    unit=cls._infer_timestamp_unit(numeric_timestamp),
                    utc=True,
                )
            else:
                parsed = pd.to_datetime(value)

            if pd.isna(parsed) or not isinstance(parsed, pd.Timestamp):
                return None

            if parsed.tzinfo is None:
                parsed = parsed.tz_localize(BUSINESS_TIMEZONE)
            else:
                parsed = parsed.tz_convert(BUSINESS_TIMEZONE)
            return parsed
        except (TypeError, ValueError, OverflowError):
            return None

    def _normalize_bool(self, value: Any) -> bool:
        """显式规范化导出文件中的撤回布尔值。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "是", "真"}:
                return True
            if normalized in {"false", "0", "no", "n", "否", "假", ""}:
                return False
            if normalized not in self.diagnostics["unknown_recalled_values"]:
                self.diagnostics["unknown_recalled_values"].append(normalized)
            return False

        return False
