import json
from pathlib import Path

import pytest

import app as app_module
from src.llm_client import LLMClient
from src.parser import QQChatParser


def make_export(years):
    messages = []
    for year in years:
        for month in range(1, 13):
            messages.append(
                {
                    "time": f"{year}-{month:02d}-15T12:00:00+08:00",
                    "sender": {"uin": f"{year}-{month}", "name": f"user-{year}"},
                    "content": {
                        "text": f"message-{year}-{month}",
                        "resources": [],
                        "mentions": [],
                    },
                }
            )
    return {
        "chatInfo": {"name": "workflow-test"},
        "statistics": {"totalMessages": len(messages)},
        "messages": messages,
    }


class FakeHistory:
    def __init__(self):
        self.records = []

    def add_record(self, **kwargs):
        self.records.append(kwargs)


class FakeRenderer:
    def render(self, stats, daily_activity, summary, rankings=None, output_path=None):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html><body>fake report</body></html>", encoding="utf-8")
        return str(path)


def run_workflow(
    tmp_path,
    monkeypatch,
    years,
    report_mode,
    fail_period=None,
    use_fake_generator=True,
    short_data=False,
    use_fake_renderer=True,
):
    output_dir = tmp_path / "output"
    intermediate_dir = output_dir / "intermediate"
    monkeypatch.setattr(app_module, "OUTPUT_FOLDER", str(output_dir))
    monkeypatch.setattr(app_module, "INTERMEDIATE_FOLDER", str(intermediate_dir))
    if use_fake_renderer:
        monkeypatch.setattr(app_module, "ReportRenderer", FakeRenderer)
    if use_fake_generator:
        monkeypatch.setattr(app_module, "LLMClient", lambda **kwargs: object())

    class FakeGenerator:
        instances = []

        def __init__(self, llm_client, logger=None):
            self.map_calls = []
            self.reduce_calls = []
            FakeGenerator.instances.append(self)

        def generate_quarterly_analysis(
            self,
            quarter,
            content,
            model=None,
            is_periodic=False,
            coverage=None,
        ):
            self.map_calls.append(
                {
                    "quarter": quarter,
                    "is_periodic": is_periodic,
                    "coverage": coverage,
                }
            )
            if fail_period and fail_period in quarter:
                raise RuntimeError(f"planned Map failure: {quarter}")
            result = LLMClient._mock_map_response()
            result["summary"] = quarter
            result["characters"] = {quarter: "fake character"}
            return result

        def generate_annual_report(
            self,
            quarterly_results,
            global_stats,
            anime_theme="default",
            custom_theme_prompt="",
            model=None,
            is_periodic=False,
            report_scope=None,
        ):
            self.reduce_calls.append(
                {
                    "results": quarterly_results,
                    "stats": global_stats,
                    "is_periodic": is_periodic,
                    "report_scope": report_scope,
                }
            )
            return LLMClient._mock_reduce_response()

        def refine_report_html(self, html_content, model=None):
            return html_content

    if use_fake_generator:
        monkeypatch.setattr(app_module, "ReportGenerator", FakeGenerator)
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake_history = FakeHistory()
    monkeypatch.setattr(app_module, "history_manager", fake_history)

    original_save = app_module.save_intermediate_artifact
    save_calls = []

    def recording_save(*args, **kwargs):
        save_calls.append(dict(kwargs))
        return original_save(*args, **kwargs)

    monkeypatch.setattr(app_module, "save_intermediate_artifact", recording_save)

    task_id = f"test-{report_mode}-{fail_period or 'none'}"
    input_path = tmp_path / f"{task_id}.json"
    export = make_export(years)
    if short_data:
        export["messages"] = export["messages"][:1]
        export["statistics"]["totalMessages"] = len(export["messages"])
    input_path.write_text(
        json.dumps(export, ensure_ascii=False),
        encoding="utf-8",
    )
    app_module.tasks[task_id] = {
        "state": "queued",
        "progress": 0,
        "status_text": "queued",
        "logs": [],
        "result_url": None,
        "result_urls": [],
        "failed_reports": [],
        "failures": [],
        "intermediate_path": None,
        "error": None,
    }

    config = {
        "mode": "default",
        "selected_years": list(years),
        "report_mode": report_mode,
        "max_tokens": 128000,
        "max_concurrency": 4,
        "request_max_retries": 1,
        "enhance_mode": False,
    }
    app_module.run_analysis_task(task_id, str(input_path), config)
    task = app_module.tasks[task_id]
    artifact_path = Path(task["intermediate_path"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    generator = FakeGenerator.instances[0] if use_fake_generator else None
    return task, artifact, generator, save_calls


def test_year_coverage_marks_short_data_as_periodic():
    export = make_export([2024])
    export["messages"] = export["messages"][:1]
    df, _ = QQChatParser().parse_json(json.dumps(export))

    coverage = app_module.build_year_coverage(df, 2024)

    assert coverage["covered_months"] == [1]
    assert coverage["covered_quarters"] == [1]
    assert coverage["is_full_year"] is False
    assert coverage["report_scope"] == "periodic"


def test_map_partial_failure_keeps_successes_and_artifact_state(tmp_path, monkeypatch):
    task, artifact, generator, save_calls = run_workflow(
        tmp_path,
        monkeypatch,
        [2024],
        "per_year",
        fail_period="First_quarter",
    )

    assert task["state"] == "partial"
    assert len(task["result_urls"]) == 1
    assert len(task["failures"]) == 1
    assert artifact["status"] == "map_partial"
    assert artifact["total_map_jobs"] == 4
    assert artifact["completed_map_jobs"] == 4
    assert artifact["successful_map_jobs"] == 3
    assert len(artifact["map_results"]["2024"]) == 3
    assert len(artifact["failures"]) == 1
    assert len(artifact["failed_map_jobs"]) == 1
    assert artifact["pending_jobs"] == []
    assert artifact["pending_map_jobs"] == []
    assert any(call.get("completed_map_jobs") == 1 for call in save_calls)
    assert len(generator.reduce_calls) == 1
    assert generator.reduce_calls[0]["stats"]["map_data_complete"] is False


@pytest.mark.parametrize(
    "report_mode,expected_reports",
    [("per_year", 2), ("combined", 1), ("both", 3)],
)
def test_report_modes_keep_years_isolated_and_preserve_combined_channel(
    tmp_path, monkeypatch, report_mode, expected_reports
):
    task, artifact, generator, _ = run_workflow(
        tmp_path,
        monkeypatch,
        [2023, 2024],
        report_mode,
    )

    assert task["state"] == "completed"
    assert len(task["result_urls"]) == expected_reports
    assert artifact["report_years"] == [2023, 2024]

    if report_mode == "per_year":
        assert [call["report_scope"] for call in generator.reduce_calls] == [
            "annual",
            "annual",
        ]
        for expected_year, call in zip([2023, 2024], generator.reduce_calls):
            summaries = [item["summary"] for item in call["results"]]
            assert summaries
            assert all(str(expected_year) in summary for summary in summaries)
    elif report_mode == "combined":
        assert len(generator.reduce_calls) == 1
        call = generator.reduce_calls[0]
        assert call["report_scope"] == "combined"
        summaries = [item["summary"] for item in call["results"]]
        assert any("2023" in summary for summary in summaries)
        assert any("2024" in summary for summary in summaries)
    else:
        assert len(generator.reduce_calls) == 3
        assert generator.reduce_calls[-1]["report_scope"] == "combined"
        assert len(task["result_urls"]) == 3
        assert any(item["kind"] == "combined" for item in task["result_urls"])


def test_partial_status_and_download_links_are_exposed_by_status_route(
    tmp_path, monkeypatch
):
    task, _, _, _ = run_workflow(
        tmp_path,
        monkeypatch,
        [2024],
        "both",
        fail_period="First_quarter",
    )

    with app_module.app.test_request_context():
        response = app_module.task_status(
            next(task_id for task_id, value in app_module.tasks.items() if value is task)
        )
    payload = response.get_json()
    assert payload["state"] == "partial"
    assert payload["result_urls"]
    assert payload["failures"]


def test_default_mock_completes_full_task_offline(tmp_path, monkeypatch):
    task, artifact, generator, _ = run_workflow(
        tmp_path,
        monkeypatch,
        [2024],
        "per_year",
        use_fake_generator=False,
        use_fake_renderer=False,
    )

    assert generator is None
    assert task["state"] == "completed"
    assert len(task["result_urls"]) == 1
    assert artifact["successful_map_jobs"] == 4


def test_short_year_uses_periodic_prompt_semantics_in_workflow(tmp_path, monkeypatch):
    task, _, generator, _ = run_workflow(
        tmp_path,
        monkeypatch,
        [2024],
        "per_year",
        short_data=True,
    )

    assert task["state"] == "completed"
    assert generator.map_calls
    assert all(call["is_periodic"] is True for call in generator.map_calls)
    assert generator.reduce_calls[0]["is_periodic"] is True
    assert generator.reduce_calls[0]["report_scope"] == "periodic"


def test_single_year_combined_workflow_remains_annual_when_coverage_is_full(
    tmp_path, monkeypatch
):
    task, _, generator, _ = run_workflow(
        tmp_path,
        monkeypatch,
        [2024],
        "combined",
    )

    assert task["state"] == "completed"
    assert len(generator.reduce_calls) == 1
    assert generator.reduce_calls[0]["is_periodic"] is False
    assert generator.reduce_calls[0]["report_scope"] == "combined"
