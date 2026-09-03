# src/llm_client.py

"""
LLM Client Module
=================
负责与 OpenAI 兼容格式的 API 交互，并提供离线演示 Mock。
"""

import os
import time
import json
from typing import Any, Callable, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.registry import *


class LLMClient:
    """支持真实 API、token 参数兼容回退和默认离线 Mock。"""

    def __init__(
        self,
        mode: str = LLM_MODE_DEFAULT,
        api_key: str = None,
        base_url: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_output_tokens: Optional[int] = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
        token_parameter: str = LLM_TOKEN_PARAMETER_AUTO,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.mode = mode
        self.api_key = api_key
        self.base_url = base_url or DEFAULT_API_BASE
        self.model = model or DEFAULT_MODEL
        self.client = None
        self.timeout = max(1, int(timeout))
        self.max_output_tokens = int(max_output_tokens) if max_output_tokens else None
        self.max_retries = max(1, int(max_retries))
        self.token_parameter = self._normalize_token_parameter(token_parameter)
        self.logger = logger

        if mode == LLM_MODE_DEFAULT and not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "DEMO_KEY")

        if OpenAI and self.api_key and self.api_key != "DEMO_KEY":
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as exc:
                self._log(f"LLM 客户端初始化失败: {type(exc).__name__}: {exc}")

    def _log(self, message: str):
        print(f"[LLM] {message}")
        if self.logger:
            self.logger(message)

    @staticmethod
    def _normalize_token_parameter(value: str) -> str:
        normalized = str(value or LLM_TOKEN_PARAMETER_AUTO).strip().lower()
        valid = {
            LLM_TOKEN_PARAMETER_AUTO,
            LLM_TOKEN_PARAMETER_MAX_TOKENS,
            LLM_TOKEN_PARAMETER_MAX_COMPLETION_TOKENS,
        }
        return normalized if normalized in valid else LLM_TOKEN_PARAMETER_AUTO

    @staticmethod
    def _estimate_input_tokens(system_prompt: str, user_prompt: str) -> int:
        chars = len(system_prompt or "") + len(user_prompt or "")
        return max(1, int(chars / 1.5))

    @staticmethod
    def _alternate_token_parameter(parameter: str) -> str:
        if parameter == LLM_TOKEN_PARAMETER_MAX_COMPLETION_TOKENS:
            return LLM_TOKEN_PARAMETER_MAX_TOKENS
        return LLM_TOKEN_PARAMETER_MAX_COMPLETION_TOKENS

    def _preferred_token_parameter(self, connection_test: bool = False) -> str:
        if self.token_parameter != LLM_TOKEN_PARAMETER_AUTO:
            return self.token_parameter
        return (
            LLM_TOKEN_PARAMETER_MAX_TOKENS
            if connection_test
            else LLM_TOKEN_PARAMETER_MAX_COMPLETION_TOKENS
        )

    @staticmethod
    def _is_token_parameter_error(error: Exception, parameter: str) -> bool:
        """Only classify explicit unsupported-parameter errors as compatible fallbacks."""
        message = str(error or "").lower()
        parameter_name = parameter.lower()
        if parameter_name not in message:
            return False
        markers = (
            "unsupported",
            "not support",
            "not allowed",
            "unknown parameter",
            "unrecognized",
            "unexpected",
            "invalid parameter",
            "extra fields",
            "400",
            "422",
        )
        return any(marker in message for marker in markers)

    def _create_completion(
        self,
        request_kwargs: Dict[str, Any],
        output_tokens: Optional[int],
        preferred_parameter: str,
        request_name: str,
    ):
        """Create one completion, retrying only once for a known token-name mismatch."""
        kwargs = dict(request_kwargs)
        if output_tokens:
            kwargs[preferred_parameter] = output_tokens

        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as first_error:
            if not output_tokens or not self._is_token_parameter_error(
                first_error, preferred_parameter
            ):
                raise

            alternate = self._alternate_token_parameter(preferred_parameter)
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop(preferred_parameter, None)
            fallback_kwargs[alternate] = output_tokens
            self._log(
                f"{request_name} 检测到 {preferred_parameter} 不兼容，"
                f"改用 {alternate} 重试"
            )
            return self.client.chat.completions.create(**fallback_kwargs)

    def generate_summary(self, text_content: str) -> str:
        """生成兼容旧接口的 HTML 总结。"""
        system_prompt = self.build_system_prompt(
            "请生成一份幽默的年度总结报告，包含年度群画像、季度小剧场、年度颁奖典礼、社死时刻和年度总结。"
        )
        return self.chat_completion(
            system_prompt,
            f"以下是部分聊天记录采样：\n{text_content}",
            request_name="Legacy:Summary",
        )

    def analyze_sentiment(self, text_content: str) -> str:
        """生成兼容旧接口的 HTML 情感分析。"""
        system_prompt = (
            "你是一个情感分析师。请分析以下对话的情感基调，并给出积极/消极/中性评价，"
            "以及关键的情绪触发点。请直接返回 HTML 片段。"
        )
        return self.chat_completion(
            system_prompt,
            f"以下是部分聊天记录采样：\n{text_content}",
            request_name="Legacy:Sentiment",
        )

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        request_name: str = "LLM",
    ) -> str:
        """调用 Chat Completions；真实请求失败时绝不静默变成 Mock。"""
        target_model = model or self.model
        input_chars = len(system_prompt or "") + len(user_prompt or "")
        estimated_input_tokens = self._estimate_input_tokens(system_prompt, user_prompt)

        if self.client:
            self._log(
                f"{request_name} 请求开始 | model={target_model} | "
                f"input_chars={input_chars} | approx_input_tokens={estimated_input_tokens} | "
                f"output_limit={self.max_output_tokens or 'provider_default'} | "
                f"timeout={self.timeout}s | attempts={self.max_retries}"
            )
            last_error = None
            for attempt in range(self.max_retries):
                started_at = time.monotonic()
                try:
                    self._log(f"{request_name} 第 {attempt + 1}/{self.max_retries} 次请求")
                    request_kwargs = {
                        "model": target_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "timeout": self.timeout,
                    }
                    response = self._create_completion(
                        request_kwargs,
                        self.max_output_tokens,
                        self._preferred_token_parameter(connection_test=False),
                        request_name,
                    )
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Empty response from LLM")
                    elapsed = time.monotonic() - started_at
                    self._log(
                        f"{request_name} 请求成功 | elapsed={elapsed:.1f}s | "
                        f"output_chars={len(content)}"
                    )
                    return content
                except Exception as exc:
                    last_error = exc
                    elapsed = time.monotonic() - started_at
                    self._log(
                        f"{request_name} 请求失败 | attempt={attempt + 1}/{self.max_retries} | "
                        f"elapsed={elapsed:.1f}s | error_type={type(exc).__name__} | error={exc}"
                    )
                    if attempt < self.max_retries - 1:
                        self._log(f"{request_name} 将在 1 秒后重试")
                        time.sleep(1)

            self._log(
                f"{request_name} 最终失败 | model={target_model} | "
                f"attempts={self.max_retries} | error_type={type(last_error).__name__} | "
                f"error={last_error}"
            )
            raise RuntimeError(f"{request_name} API 请求失败: {last_error}") from last_error

        if self.mode == LLM_MODE_DEFAULT:
            self._log(f"{request_name} 使用内置 Mock | 未初始化真实 API 客户端")
            return self._mock_response(user_prompt, request_name=request_name)

        self._log(f"{request_name} 无法发送 | custom 模式但真实 API 客户端未初始化")
        raise RuntimeError("Custom 模式下 LLM 客户端未初始化")

    def test_connection(self) -> dict:
        """测试 API 连接，默认先使用 max_tokens，必要时回退。"""
        if not self.client:
            if self.mode == LLM_MODE_DEFAULT:
                return {
                    "success": False,
                    "message": "未检测到有效的 API Key。请检查环境变量 OPENAI_API_KEY 是否设置。",
                }
            return {
                "success": False,
                "message": "客户端初始化失败。可能是 API Key 为空或 openai 库未安装。",
            }

        actual_url = str(getattr(self.client, "base_url", self.base_url))
        try:
            response = self._create_completion(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "timeout": self.timeout,
                },
                5,
                self._preferred_token_parameter(connection_test=True),
                "连接测试",
            )
            model_used = getattr(response, "model", self.model)
            return {
                "success": True,
                "message": (
                    f"连接成功！\n\n✅ 目标地址: {actual_url}\n"
                    f"✅ 响应模型: {model_used}\n✅ 状态: 通信正常"
                ),
            }
        except Exception as exc:
            error_msg = str(exc)
            if "401" in error_msg:
                message = f"认证失败 (401)：请检查您的 API Key 是否正确。\n详细信息: {error_msg}"
            elif "404" in error_msg:
                message = (
                    f"请求失败 (404)：可能是 API Base URL 错误或模型名称不正确。\n"
                    f"目标地址: {actual_url}\n详细信息: {error_msg}"
                )
            elif "429" in error_msg:
                message = f"请求过多 (429)：您的账户可能已欠费或达到速率限制。\n详细信息: {error_msg}"
            else:
                message = f"连接测试失败：{error_msg}\n目标地址: {actual_url}"
            return {"success": False, "message": message}

    @staticmethod
    def _mock_map_response() -> Dict[str, Any]:
        return {
            "summary": "Mock 阶段摘要：群聊围绕游戏、日常和临时起意的话题展开，发言密度稳定，偶尔出现集体跑题。",
            "vibe": "轻松、热闹，带有适度的吐槽感。",
            "active_members": ["Mock 用户"],
            "inactive_members": [],
            "events": ["Mock 事件：群友完成了一次高质量跑题。"],
            "memes_born": ["Mock 梗"],
            "memes_died": [],
            "mvp": "Mock 用户以稳定发言和及时吐槽获得本阶段 MVP。",
            "characters": {"Mock 用户": "负责发言、接梗和把话题带向未知方向。"},
            "relations": ["Mock 用户与群聊保持高频互动"],
        }

    @staticmethod
    def _mock_reduce_response() -> Dict[str, Any]:
        return {
            "style_config": {
                "primary_color": "#6D4AFF",
                "secondary_color": "#2EC4B6",
                "background_color": "#F0F2F5",
                "card_bg": "#FFFFFF",
                "text_color": "#1F2937",
                "font_family": "'Microsoft YaHei', sans-serif",
            },
            "keywords": ["日常", "游戏", "吐槽", "跑题", "互动"],
            "portrait": "<h3>Mock 群画像</h3><p>这是一个会在日常话题中稳定跑题的群聊。</p>",
            "timeline": "<h3>关键事件时间线</h3><ul><li>Mock：一次普通聊天顺利发展成集体吐槽。</li></ul>",
            "quarterly_review": "<h3>阶段复盘</h3><p>Mock 数据已完成 Map-Reduce 演示。</p>",
            "roasts": "<h3>群成员锐评</h3><p>Mock 用户：负责把安静的群聊重新变得不安静。</p>",
            "awards": "<h3>荣誉颁奖典礼</h3><p>颁发年度稳定在线奖。</p>",
            "anime_theater": "<h3>动漫IP小剧场</h3><p>旁白：这一集，大家又聊到了完全不同的地方。</p>",
            "moments": "<h3>搞笑时刻</h3><p>没有真实事件，只有稳定的 Mock 混乱。</p>",
            "essay": "<h3>总结小作文</h3><p>即使是演示数据，也能完成一份结构完整的报告。</p>",
        }

    def _mock_response(self, prompt: str, request_name: str = "LLM") -> str:
        """Return schema-compatible JSON for Map/Reduce and HTML for Refine/legacy calls."""
        lowered_name = request_name.lower()
        if lowered_name.startswith("map:"):
            payload = self._mock_map_response()
        elif lowered_name.startswith("reduce:"):
            payload = self._mock_reduce_response()
        elif lowered_name.startswith("refine:"):
            return "<html><body><h1>Mock enhanced report</h1></body></html>"
        else:
            return "<h3>Mock 输出</h3><p>这是离线演示结果。</p>"
        return json.dumps(payload, ensure_ascii=False)

    def build_system_prompt(self, stats_injection: str) -> str:
        """构建兼容旧接口的系统提示词。"""
        base_prompt = (
            "你是一个专业的聊天记录分析师，擅长幽默、犀利的点评。"
            "请根据提供的对话内容进行分析。请直接返回 HTML 片段，不要包含 Markdown 标记。"
        )
        if stats_injection:
            base_prompt += f"\n\n参考统计数据：\n{stats_injection}"
        return base_prompt
