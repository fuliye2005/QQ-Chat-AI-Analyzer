# src/prompts.py

"""
Prompt Management Module (Phase 3)
==================================
负责构建和管理与 LLM 交互的 Prompts。
遵循 Phase 5 编程规范。
"""

from src.registry import *
import json

class PromptManager:
    """
    Prompt 管理器，负责组装 Map-Reduce 各阶段的 Prompt。
    """

    def __init__(self):
        # 意义: 初始化
        # 作用: 占位，未来可加载外部模板文件
        pass

    def build_map_prompt(
        self,
        quarter_name: str,
        chat_content: str,
        is_periodic: bool = False,
        coverage: dict = None,
    ) -> str:
        """
        构建 Map 阶段 (季度/阶段分析) 的 Prompt。
        """
        # 意义: 构造单季度分析指令
        # 作用: 填充模板，注入压缩后的聊天记录
        # 关联: 被 Generator 调用，用于获取中间态 JSON
        
        template = PROMPT_MAP_PERIODIC if is_periodic else PROMPT_MAP_QUARTERLY
        prompt = template.format(quarter=quarter_name)
        if coverage:
            prompt += "\n\n" + self._format_coverage_instruction(
                coverage,
                report_scope=coverage.get("report_scope"),
            )
        return prompt + "\n\n聊天记录片段:\n" + chat_content

    def build_reduce_prompt(
        self,
        quarterly_results: list,
        global_stats: dict,
        anime_theme: str = "default",
        custom_theme_prompt: str = "",
        is_periodic: bool = False,
        report_scope: str = None,
    ) -> str:
        """
        构建 Reduce 阶段 (年度/阶段汇总) 的 Prompt。
        
        Args:
            quarterly_results: 季度分析结果列表
            global_stats: 全局统计数据
            anime_theme: 动漫小剧场主题 (default, bandream, gbc, custom)
            custom_theme_prompt: 自定义主题提示词 (仅当 anime_theme='custom' 时有效)
            is_periodic: 是否为非完整年度/阶段性报告
        """
        # 意义: 构造年度汇总指令
        # 作用: 聚合所有季度的分析结果和全局统计数据
        # 关联: 被 Generator 调用，用于生成最终 HTML 内容
        
        # 序列化中间态数据
        q_data_str = json.dumps(quarterly_results, ensure_ascii=False, indent=2)
        
        # 格式化统计数据 (包含硬核榜单)
        hardcore = global_stats.get('hardcore', {})
        stats_str = f"""
        - 消息总数: {global_stats.get('total_messages', 0)}
        - 活跃用户数: {global_stats.get('active_users_count', 0)}
        - 潜水用户数: {global_stats.get('silent_users_count', 0)}
        - 话痨榜 (Top 3): {global_stats.get('top_talkers', [])}
        - 复读机榜 (Top 3): {global_stats.get('top_repeaters', [])}
        - 熬夜冠军 (Top 3): {hardcore.get('night_owls', [])}
        - 早起冠军 (Top 3): {hardcore.get('early_birds', [])}
        - 最长连续发言: {hardcore.get('longest_streak', {})}
        - Map 数据是否完整: {global_stats.get('map_data_complete', True)}
        - Map 失败任务: {global_stats.get('map_failures', [])}
        """
        
        resolved_scope = report_scope or global_stats.get("report_scope")
        selected_years = global_stats.get("selected_years") or []
        is_multi_year_combined = (
            resolved_scope == "combined" and len(selected_years) > 1
        )
        template = (
            PROMPT_REDUCE_PERIODIC
            if is_periodic
            or resolved_scope == "periodic"
            or is_multi_year_combined
            else PROMPT_REDUCE_ANNUAL
        )
        
        # 处理小剧场主题指令
        anime_instruction = self._get_anime_instruction(anime_theme, custom_theme_prompt)
        
        prompt = template.format(
            quarterly_data=q_data_str,
            stats_data=stats_str,
            year=global_stats.get('year', 'Unknown'),
            anime_instruction=anime_instruction
        )
        coverage = global_stats.get("coverage")
        if coverage:
            prompt += "\n\n" + self._format_coverage_instruction(
                coverage,
                report_scope=resolved_scope,
            )
        if resolved_scope == "combined":
            prompt += (
                "\n这是集合报告通道。请保留各选中年份的边界和差异，"
                "不要把不同年份的消息误写成同一年度事件。"
            )
        return prompt

    @staticmethod
    def _format_coverage_instruction(coverage: dict, report_scope: str = None) -> str:
        """Make the observed time range explicit to prevent invented quarters."""
        if report_scope == "combined" or coverage.get("coverage_by_year"):
            entries = coverage.get("coverage_by_year", {})
            scope_label = "多年/集合分析" if len(entries) > 1 else "单年集合分析"
            lines = [
                f"数据覆盖约束：这是一个{scope_label}，以下是各年份实际覆盖范围："
            ]
            for year, item in sorted(entries.items(), key=lambda pair: str(pair[0])):
                lines.append(PromptManager._format_single_coverage(year, item))
            lines.append("只描述上述实际覆盖月份和时间段，不要补写缺失月份或不存在的季度。")
            return "\n".join(lines)

        year = coverage.get("year", "Unknown")
        lines = ["数据覆盖约束：", PromptManager._format_single_coverage(year, coverage)]
        if coverage.get("is_full_year"):
            lines.append("这是完整自然年数据，可以按全年四个季度进行分析。")
        else:
            lines.append(
                "这是不完整年度/阶段数据。只能分析实际覆盖期间，"
                "不得虚构未覆盖月份、季度或事件。"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_single_coverage(year, coverage: dict) -> str:
        covered_months = coverage.get("covered_months", [])
        covered_quarters = coverage.get("covered_quarters", [])
        missing_months = coverage.get("missing_months", [])
        return (
            f"- {year} 年：{coverage.get('start_time', '未知')} 至 "
            f"{coverage.get('end_time', '未知')}；覆盖月份={covered_months}；"
            f"覆盖季度={covered_quarters}；缺失月份={missing_months}；"
            f"完整年度={bool(coverage.get('is_full_year'))}"
        )

    def _get_anime_instruction(self, theme: str, custom_prompt: str) -> str:
        """根据主题生成小剧场指令"""
        if theme == "bandream":
            return """
   - **主题要求**：请使用 **BanG Dream! It's MyGO!!!!!** 或 **Ave Mujica** 的风格。
   - 包含重力场、扭曲的情感、春日影、回旋镖等元素。
   - 剧本风格要“胃痛”且充满戏剧性张力。
   - 角色分配：将群里的 MVP 分配为类似“灯”、“祥子”、“素世”等角色（仅借用性格特质，不要直接改名）。
            """
        elif theme == "gbc":
            return """
   - **主题要求**：请使用 **Girls Band Cry (GBC)** 的风格。
   - 包含竖中指、桃香、仁菜的暴躁互动、路灯、吉野家等元素。
   - 剧本风格要“摇滚”、“暴躁”且充满生命力。
   - 角色分配：将群里的活跃分子分配为类似“仁菜”、“桃香”等角色。
            """
        elif theme == "custom":
            if not custom_prompt or not custom_prompt.strip():
                return """
   - **主题要求**：用户选择了自定义主题但未输入内容。请自由发挥，选择一个充满戏剧性的 ACGN 主题。
                """
            return f"""
   - **主题要求**：请使用用户自定义的主题：**{custom_prompt}**。
   - 请严格按照该主题的风格和设定编写剧本。
            """
        else:
            return """
   - **主题要求**：请根据群聊的实际氛围，自动选择一个最合适的 ACGN 主题（如“赛博朋克”、“异世界转生”、“校园日常”等）。
            """
