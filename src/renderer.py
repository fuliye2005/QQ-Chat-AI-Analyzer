# src/renderer.py

"""
Report Renderer Module (Phase 4)
================================
负责将分析结果和 AI 生成的内容渲染为最终的 HTML 报告。
遵循 Phase 5 编程规范。
"""

from jinja2 import Environment, FileSystemLoader
import os
import re
import pandas as pd
from typing import Dict, Any
from src.registry import *


DEFAULT_REPORT_STYLE = {
    "primary_color": "#6D4AFF",
    "secondary_color": "#2EC4B6",
    "background_color": "#F0F2F5",
    "card_bg": "#FFFFFF",
    "text_color": "#1F2937",
    "font_family": "'PingFang SC', 'Microsoft YaHei', sans-serif",
}
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _hex_luminance(value: str) -> float:
    """Return relative luminance for a validated six-digit hex color."""
    rgb = [int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    channels = [
        ((channel + 0.055) / 1.055) ** 2.4
        if channel > 0.03928
        else channel / 12.92
        for channel in rgb
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _normalize_report_style(style_config: Any) -> Dict[str, str]:
    """Validate AI-selected colors and keep text readable on generated surfaces."""
    raw = style_config if isinstance(style_config, dict) else {}
    style = DEFAULT_REPORT_STYLE.copy()

    for key in ("primary_color", "secondary_color", "background_color", "card_bg", "text_color"):
        value = raw.get(key)
        if isinstance(value, str) and HEX_COLOR_RE.fullmatch(value.strip()):
            style[key] = value.strip()

    font_family = raw.get("font_family")
    if isinstance(font_family, str) and 0 < len(font_family) <= 200 and "{" not in font_family and "}" not in font_family:
        style["font_family"] = font_family

    # The model may choose a dark card with dark text or the reverse.
    contrast = abs(_hex_luminance(style["card_bg"]) - _hex_luminance(style["text_color"]))
    if contrast < 0.28:
        style["text_color"] = "#F3F4F6" if _hex_luminance(style["card_bg"]) < 0.45 else "#1F2937"

    return style

class HTMLRenderer:
    """
    HTML 报告渲染器。
    """

    def __init__(self, template_dir: str = "templates"):
        # 意义: 初始化渲染器
        # 作用: 设置 Jinja2 模板环境
        # 关联: 依赖 templates 目录下的 HTML 文件
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template_name = "report.html"

    def render(self, stats: Dict[str, Any], daily_activity: pd.DataFrame, summary: Dict[str, Any], rankings: Dict[str, pd.DataFrame] = None, output_path: str = "output/report.html") -> str:
        """
        渲染并保存报告。
        """
        # 意义: 生成最终文件
        # 作用: 将统计数据 (stats) 和 AI 文案 (ai_content) 注入模板，并保存到磁盘
        # 关联: 被 app.py 调用，输出最终结果
        
        template = self.env.get_template(self.template_name)
        
        # 处理排行榜数据
        processed_rankings = {}
        if rankings:
             for key, df in rankings.items():
                 processed_rankings[key] = df.to_dict('records') if not df.empty else []
        
        # AI controls the theme, but the renderer enforces valid colors and contrast.
        normalized_summary = dict(summary or {})
        normalized_summary["style_config"] = _normalize_report_style(normalized_summary.get("style_config"))

        # 准备渲染上下文
        context = {
            "title": f"{stats.get('chat_name', '群聊')} - 年度总结报告",
            "stats": stats,
            "daily_activity": daily_activity.to_dict('records') if not daily_activity.empty else [],
            "summary": normalized_summary,
            "rankings": processed_rankings,
            "generated_at": pd.Timestamp.now().strftime(DEFAULT_TIME_FORMAT)
        }
        
        html_content = template.render(**context)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return output_path
