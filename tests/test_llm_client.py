from types import SimpleNamespace

from src.llm_client import LLMClient


class FakeCompletions:
    def __init__(self, rejected_parameter):
        self.rejected_parameter = rejected_parameter
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.rejected_parameter in kwargs:
            raise RuntimeError(
                f"400 unsupported parameter: {self.rejected_parameter}"
            )
        return SimpleNamespace(
            model="fake-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}')
                )
            ],
        )


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)
        self.base_url = "https://fake.example/v1"


def make_client(rejected_parameter):
    client = LLMClient(mode="custom", api_key=None, max_retries=1)
    completions = FakeCompletions(rejected_parameter)
    client.client = FakeClient(completions)
    return client, completions


def test_formal_request_prefers_max_completion_tokens_then_falls_back():
    client, completions = make_client("max_completion_tokens")

    result = client.chat_completion("system", "user", request_name="Map:test")

    assert result == '{"ok": true}'
    assert "max_completion_tokens" in completions.calls[0]
    assert "max_tokens" in completions.calls[1]


def test_connection_prefers_max_tokens_then_falls_back():
    client, completions = make_client("max_tokens")

    result = client.test_connection()

    assert result["success"] is True
    assert "max_tokens" in completions.calls[0]
    assert "max_completion_tokens" in completions.calls[1]
