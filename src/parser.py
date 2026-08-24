# src/parser.py

"""
QQ Chat Parser Module
=====================
负责解析 QQChatExporter 导出的 JSON 文件。
遵循 Phase 5 编程规范。
"""

import json
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from src.registry import *

class QQChatParser:
    """
    负责解析 QQChatExporter 导出的 JSON 文件，转换为结构化的 DataFrame。
    """
    
    def __init__(self):
        # 意义: 初始化解析器
        # 作用: 目前无特定初始化逻辑，占位
        # 关联: 无
        pass

    def parse_json(self, file_content: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        解析 JSON 字符串内容。
        """
        # 意义: 核心解析方法
        # 作用: 将 JSON 字符串解析为 Pandas DataFrame 和 元数据字典
        # 关联: 被主程序调用，依赖 src.registry 定义的常量
        
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")

        messages = data.get(JSON_FIELD_MESSAGES, [])
        parsed_data = []

        for msg in messages:
            # 意义: 提取单条消息数据
            # 作用: 遍历消息列表，提取时间、发送者、内容等信息
            # 关联: 依赖 _parse_single_message 方法
            
            parsed_msg = self._parse_single_message(msg)
            if parsed_msg:
                parsed_data.append(parsed_msg)

        df = pd.DataFrame(parsed_data)
        
        # 意义: 提取元数据
        # 作用: 获取群名称、总消息数、时间范围等
        # 关联: 返回给调用者用于展示
        meta = {
            "chat_name": data.get(JSON_FIELD_CHAT_INFO, {}).get(JSON_FIELD_CHAT_NAME, UNKNOWN_GROUP_NAME),
            "total_messages": data.get(JSON_FIELD_STATISTICS, {}).get(JSON_FIELD_TOTAL_MESSAGES, 0),
            "start_time": data.get(JSON_FIELD_STATISTICS, {}).get(JSON_FIELD_TIME_RANGE, {}).get(JSON_FIELD_START),
            "end_time": data.get(JSON_FIELD_STATISTICS, {}).get(JSON_FIELD_TIME_RANGE, {}).get(JSON_FIELD_END)
        }
        
        return df, meta

    def _parse_single_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        解析单条消息字典。
        """
        # 意义: 单条消息解析逻辑
        # 作用: 处理时间戳转换、用户ID识别、内容分类等细节
        # 关联: 被 parse_json 循环调用
        
        # 1. 时间处理
        # 新版导出同时提供 ISO 时间字符串和毫秒时间戳，优先使用带时区的 time。
        dt = self._parse_message_time(msg)
        if dt is None:
            # 无法确定时间的消息不能伪造为“当前时间”，否则会污染日期、季度和作息统计。
            return None

        # 2. 发送者识别 (Phase 1 核心: User Identification)
        sender = msg.get(JSON_FIELD_SENDER, {})
        # 优先使用 uin，其次 uid
        user_id = sender.get(JSON_FIELD_SENDER_UIN) or sender.get(JSON_FIELD_SENDER_UID, "unknown")
        # 优先使用 card，其次 name
        user_name = sender.get(JSON_FIELD_SENDER_CARD) or sender.get(JSON_FIELD_SENDER_NAME, UNKNOWN_USER_NAME)

        # 3. 内容与资源处理 (Phase 1 核心: Content Classification)
        content_obj = msg.get(JSON_FIELD_CONTENT, {})
        text_content = content_obj.get(JSON_FIELD_TEXT, "")
        resources = content_obj.get(JSON_FIELD_RESOURCES, [])
        
        msg_type = MSG_TYPE_TEXT
        image_count = 0
        
        if resources:
            res_types = [r.get("type") for r in resources]
            image_count = res_types.count("image")
            
            if "image" in res_types:
                msg_type = MSG_TYPE_IMAGE
            elif "video" in res_types:
                msg_type = MSG_TYPE_VIDEO
            elif "file" in res_types:
                msg_type = MSG_TYPE_FILE
            
            if text_content and resources:
                msg_type = MSG_TYPE_MIXED

        is_recalled = msg.get(
            JSON_FIELD_IS_RECALLED,
            msg.get(JSON_FIELD_RECALLED, False)
        )

        if is_recalled:
            msg_type = MSG_TYPE_RECALLED

        # 4. 提及处理
        mentions = [m.get("name") for m in content_obj.get(JSON_FIELD_MENTIONS, [])]

        return {
            COL_DATETIME: dt,
            COL_DATE: dt.date(),
            COL_TIME: dt.time(),
            COL_HOUR: dt.hour,
            COL_USER_ID: str(user_id), # 确保 ID 为字符串
            COL_USER_NAME: user_name,
            COL_CONTENT: text_content,
            COL_TYPE: msg_type,
            COL_IS_RECALLED: bool(is_recalled),
            COL_MENTIONS: mentions,
            COL_IMAGE_COUNT: image_count
        }

    def _parse_message_time(self, msg: Dict[str, Any]) -> Optional[pd.Timestamp]:
        """解析消息时间，兼容 ISO 字符串以及秒/毫秒/微秒/纳秒时间戳。"""
        raw_time = msg.get(JSON_FIELD_TIME)
        if raw_time not in (None, ""):
            try:
                parsed = pd.to_datetime(raw_time, utc=True)
                if not pd.isna(parsed):
                    return parsed
            except (TypeError, ValueError, OverflowError):
                pass

        raw_timestamp = msg.get(JSON_FIELD_TIMESTAMP)
        if raw_timestamp in (None, "") or isinstance(raw_timestamp, bool):
            return None

        try:
            if isinstance(raw_timestamp, (int, float)):
                numeric_timestamp = raw_timestamp
            elif isinstance(raw_timestamp, str):
                timestamp_text = raw_timestamp.strip()
                try:
                    numeric_timestamp = float(timestamp_text)
                except ValueError:
                    parsed = pd.to_datetime(timestamp_text, utc=True)
                    return None if pd.isna(parsed) else parsed
            else:
                return pd.to_datetime(raw_timestamp, utc=True)

            magnitude = abs(float(numeric_timestamp))
            if magnitude >= 1e17:
                unit = "ns"
            elif magnitude >= 1e14:
                unit = "us"
            elif magnitude >= 1e11:
                unit = "ms"
            else:
                unit = "s"

            parsed = pd.to_datetime(numeric_timestamp, unit=unit, utc=True)
            return None if pd.isna(parsed) else parsed
        except (TypeError, ValueError, OverflowError):
            return None
