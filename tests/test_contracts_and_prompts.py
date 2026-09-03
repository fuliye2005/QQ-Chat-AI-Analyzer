import json

import pytest

from src.generator import ReportGenerator
from src.llm_client import LLMClient
from src.prompts import PromptManager
from src.schema import (
    SchemaValidationError,
    validate_map_result,
    validate_reduce_result,
)


def map_payload():
    return LLMClient._mock_map_response()


def reduce_payload():
    return LLMClient._mock_reduce_response()


def coverage(year=2024, full=False):
    months = list(range(1, 13)) if full else [1]
    return {
        "year": year,
        "start_time": f"{year}-01-01T00:00:00+08:00",
        "end_time": f"{year}-12-31T23:59:59+08:00" if full else f"{year}-01-31T23:59:59+08:00",
        "covered_months": months,
        "covered_quarters": [1] if not full else [1, 2, 3, 4],
        "missing_months": [] if full else list(range(2, 13)),
        "is_full_year": full,
        "report_scope": "annual" if full else "periodic",
    }


def test_map_and_reduce_schema_reject_invalid_shapes_and_types():
    with pytest.raises(SchemaValidationError):
        validate_map_result([])
    with pytest.raises(SchemaValidationError):
        validate_map_result({"summary": "only"})

    invalid_map = map_payload()
    invalid_map["events"] = "not-a-list"
    with pytest.raises(SchemaValidationError):
        validate_map_result(invalid_map)
    invalid_map = map_payload()
    invalid_map["active_members"] = [None]
    with pytest.raises(SchemaValidationError):
        validate_map_result(invalid_map)

    with pytest.raises(SchemaValidationError):
        validate_reduce_result("<html></html>")
    invalid_reduce = reduce_payload()
    invalid_reduce["style_config"] = []
    with pytest.raises(SchemaValidationError):
        validate_reduce_result(invalid_reduce)


def test_default_mock_completes_map_and_reduce_contracts(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generator = ReportGenerator(LLMClient())

    map_result = generator.generate_quarterly_analysis("2024_Q1", "hello")
    reduce_result = generator.generate_annual_report(
        [map_result],
        {
            "year": "2024",
            "report_scope": "annual",
            "coverage": coverage(full=True),
        },
    )

    assert set(map_result) >= {
        "summary",
        "vibe",
        "active_members",
        "inactive_members",
        "events",
        "memes_born",
        "memes_died",
        "mvp",
        "characters",
        "relations",
    }
    assert set(reduce_result) >= {
        "style_config",
        "keywords",
        "portrait",
        "timeline",
        "quarterly_review",
        "roasts",
        "awards",
        "anime_theater",
        "moments",
        "essay",
    }


def test_incomplete_year_prompt_explicitly_forbids_invented_quarters():
    manager = PromptManager()
    prompt = manager.build_reduce_prompt(
        [map_payload()],
        {
            "year": "2024",
            "report_scope": "periodic",
            "coverage": coverage(full=False),
        },
        is_periodic=True,
    )

    assert "不完整年度/阶段数据" in prompt
    assert "不得虚构未覆盖月份、季度或事件" in prompt
    assert "缺失月份" in prompt


def test_combined_prompt_keeps_coverage_separate_by_year():
    manager = PromptManager()
    prompt = manager.build_reduce_prompt(
        [map_payload()],
        {
            "year": "2023、2024",
            "report_scope": "combined",
            "coverage": {
                "report_scope": "combined",
                "coverage_by_year": {
                    "2023": coverage(2023, full=True),
                    "2024": coverage(2024, full=False),
                },
            },
        },
        is_periodic=True,
        report_scope="combined",
    )

    assert "2023" in prompt
    assert "2024" in prompt
    assert "多年/集合分析" in prompt
    assert "不要把不同年份的消息误写成同一年度事件" in prompt


def test_single_year_combined_prompt_is_collection_but_not_multi_year_periodic():
    manager = PromptManager()
    prompt = manager.build_reduce_prompt(
        [map_payload()],
        {
            "year": "2024",
            "selected_years": [2024],
            "report_scope": "combined",
            "coverage": {
                "report_scope": "combined",
                "coverage_by_year": {"2024": coverage(2024, full=True)},
            },
        },
        is_periodic=False,
        report_scope="combined",
    )

    assert "单年集合分析" in prompt
    assert "完整年度的季度分析摘要" in prompt
    assert "这是集合报告通道" in prompt


class StubLLM:
    def __init__(self, response):
        self.response = response

    def chat_completion(self, *args, **kwargs):
        return self.response


def test_generator_validates_decoded_json():
    generator = ReportGenerator(StubLLM(json.dumps({})))

    with pytest.raises(RuntimeError, match="Map 阶段"):
        generator.generate_quarterly_analysis("2024_Q1", "hello")

    with pytest.raises(RuntimeError, match="汇总阶段"):
        generator.generate_annual_report([], {})
