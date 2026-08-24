# src/llm_client.py

"""
LLM Client Module (Phase 2)
===========================
负责与 LLM API 交互。
遵循 Phase 5 编程规范。
"""

import os
import time
from typing import Callable, Dict, Any, List, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.registry import *

class LLMClient:
    """
    LLM 客户端，支持默认配置与自定义配置双模式。
    """

    def __init__(
        self,
        mode: str = LLM_MODE_DEFAULT,
        api_key: str = None,
        base_url: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_output_tokens: Optional[int] = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
        logger: Optional[Callable[[str], None]] = None,
    ):
        # 意义: 初始化客户端
        # 作用: 加载 API Key 和 Base URL
        # 关联: 被主程序调用
        
        self.mode = mode
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        self.timeout = max(1, int(timeout))
        self.max_output_tokens = int(max_output_tokens) if max_output_tokens else None
        self.max_retries = max(1, int(max_retries))
        self.logger = logger
        
        if mode == LLM_MODE_DEFAULT and not self.api_key:
            # 默认模式：尝试从环境变量读取
            self.api_key = os.environ.get("OPENAI_API_KEY", "DEMO_KEY")
        
        # 初始化 OpenAI 客户端 (如果 Key 有效且库已安装)
        if OpenAI and self.api_key and self.api_key != "DEMO_KEY":
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                self._log(f"LLM 客户端初始化失败: {type(e).__name__}: {e}")

    def _log(self, message: str):
        """Send diagnostics to both the console and the task logger when available."""
        print(f"[LLM] {message}")
        if self.logger:
            self.logger(message)

    @staticmethod
    def _estimate_input_tokens(system_prompt: str, user_prompt: str) -> int:
        # This is a conservative display estimate, not a provider tokenizer result.
        chars = len(system_prompt or "") + len(user_prompt or "")
        return max(1, int(chars / 1.5))

    def generate_summary(self, text_content: str) -> str:
        """生成总结报告"""
        system_prompt = self.build_system_prompt("请生成一份幽默的年度总结报告，包含：年度群画像、季度小剧场、年度颁奖典礼、社死时刻、年度总结诗。")
        return self.chat_completion(system_prompt, f"以下是部分聊天记录采样：\n{text_content}")

    def analyze_sentiment(self, text_content: str) -> str:
        """生成情感分析"""
        system_prompt = "你是一个情感分析师。请分析以下对话的情感基调，并给出积极/消极/中性评价，以及关键的情绪触发点。请直接返回 HTML 片段。"
        return self.chat_completion(system_prompt, f"以下是部分聊天记录采样：\n{text_content}")

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        request_name: str = "LLM",
    ) -> str:
        """
        调用 LLM Chat Completion API。
        """
        # 意义: 发送请求
        # 作用: 封装 OpenAI SDK 调用，处理网络异常
        # 关联: 核心 AI 功能入口
        
        target_model = model if model else self.model
        input_chars = len(system_prompt or "") + len(user_prompt or "")
        estimated_input_tokens = self._estimate_input_tokens(system_prompt, user_prompt)

        # 1. 尝试真实调用
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
                            {"role": "user", "content": user_prompt}
                        ],
                        "timeout": self.timeout,
                    }
                    if self.max_output_tokens:
                        request_kwargs["max_completion_tokens"] = self.max_output_tokens
                    response = self.client.chat.completions.create(**request_kwargs)
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Empty response from LLM")
                    elapsed = time.monotonic() - started_at
                    self._log(
                        f"{request_name} 请求成功 | elapsed={elapsed:.1f}s | "
                        f"output_chars={len(content)}"
                    )
                    return content
                    
                except Exception as e:
                    last_error = e
                    elapsed = time.monotonic() - started_at
                    self._log(
                        f"{request_name} 请求失败 | attempt={attempt + 1}/{self.max_retries} | "
                        f"elapsed={elapsed:.1f}s | error_type={type(e).__name__} | error={e}"
                    )
                    if attempt < self.max_retries - 1:
                        self._log(f"{request_name} 将在 1 秒后重试")
                        time.sleep(1)

            self._log(
                f"{request_name} 最终失败 | model={target_model} | "
                f"attempts={self.max_retries} | error_type={type(last_error).__name__} | error={last_error}"
            )
            # A real request must never silently turn into a fake successful result.
            raise RuntimeError(f"{request_name} API 请求失败: {last_error}") from last_error

        # 2. Mock is only used when the application explicitly has no real client.
        if self.mode == LLM_MODE_DEFAULT:
             self._log(f"{request_name} 使用内置 Mock | 未初始化真实 API 客户端")
             return self._mock_response(user_prompt)
        else:
             self._log(f"{request_name} 无法发送 | custom 模式但真实 API 客户端未初始化")
             raise RuntimeError("Custom 模式下 LLM 客户端未初始化")

    def test_connection(self) -> dict:
        """
        测试 API 连接状态 (自检功能)。
        """
        # 意义: 验证配置有效性
        # 作用: 发送极简请求检测连通性，不吞没异常
        # 关联: 前端“测试连接”按钮
        
        if not self.client:
             if self.mode == LLM_MODE_DEFAULT:
                 return {"success": False, "message": "未检测到有效的 API Key。请检查环境变量 OPENAI_API_KEY 是否设置。"}
             else:
                 return {"success": False, "message": "客户端初始化失败。可能是 API Key 为空或 openai 库未安装。"}
        
        # 获取实际使用的 Base URL (OpenAI Client 会自动处理末尾斜杠等)
        actual_url = str(self.client.base_url)
        print(f"[Debug] Testing Connection -> URL: {actual_url}, Key: {self.api_key[:8]}***")

        try:
            # 发送一个极简的测试请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            model_used = response.model
            return {
                "success": True, 
                "message": f"连接成功！\n\n✅ 目标地址: {actual_url}\n✅ 响应模型: {model_used}\n✅ 状态: 通信正常"
            }
        except Exception as e:
            error_msg = str(e)
            print(f"[Debug] Connection Failed: {error_msg}")
            # 尝试提取更友好的错误信息
            if "401" in error_msg:
                return {"success": False, "message": f"认证失败 (401)：请检查您的 API Key 是否正确。\n详细信息: {error_msg}"}
            elif "404" in error_msg:
                return {"success": False, "message": f"请求失败 (404)：可能是 API Base URL 错误或模型名称不正确。\n目标地址: {actual_url}\n详细信息: {error_msg}"}
            elif "429" in error_msg:
                return {"success": False, "message": f"请求过多 (429)：您的账户可能已欠费或达到速率限制。\n详细信息: {error_msg}"}
            else:
                return {"success": False, "message": f"连接测试失败：{error_msg}\n目标地址: {actual_url}"}

    def _mock_response(self, prompt: str) -> str:
        """
        生成模拟数据用于演示。
        """
        print(f"--- [Mock LLM] Mode: {self.mode} ---")
        
        # 简单的关键词匹配以生成稍微相关的 Mock 内容
        if "年度" in prompt or "summary" in prompt.lower():
            return """
            <h3>年度群画像</h3>
            <p><b>🏷️ 标签：赛博精神病院</b></p>
            <p>原因：数据表明，本群夜间活跃度高达 80%，且“哈哈”一词出现频率远超人类正常水平。</p>
            
            <h3>季度小剧场 (Anime Theater)</h3>
            <p><b>Alice (吐槽役):</b> 这一年我们到底聊了些什么？</p>
            <p><b>Bob (复读机):</b> 聊了些什么？+1</p>
            <p><b>Charlie (潜水员):</b> ... (发出抢红包的声音)</p>
            """
        else:
            return f"""
            <h4>季度分析摘要</h4>
            <ul>
            <li><b>核心话题:</b> 摸鱼、游戏、奶茶。</li>
            <li><b>情感倾向:</b> 极度快乐 (Positivity: 0.9)。</li>
            <li><b>高频词:</b> 666, 笑死, 救命。</li>
            </ul>
            <!-- Debug Info: Input length {len(prompt)} -->
            """

    def build_system_prompt(self, stats_injection: str) -> str:
        """
        构建 System Prompt。
        """
        # 意义: Prompt 工程
        # 作用: 注入角色设定和硬性统计数据
        # 关联: Phase 2 Statistical Injection
        
        base_prompt = "你是一个专业的聊天记录分析师，擅长幽默、犀利的点评。请根据提供的对话内容进行分析。请直接返回 HTML 片段，不要包含 Markdown 标记。"
        if stats_injection:
            base_prompt += f"\n\n参考统计数据：\n{stats_injection}"
        return base_prompt
