import asyncio

import fastapi.testclient
import pytest

import app as app_module
import llm_generator
import llm_picker
import services
from tests import provider_stub
from tests.conftest import SECRET_KEY

MODELS = ["gemini", "llama", "gpt", "claude", "deepseek"]
OPTIONS = {"재료": ["토마토"]}
MENU_PARTS = ['[{"name": ', '"파스타"}]']
MAX_OUTPUT_TOKENS = 2_048
SAMBANOVA_REQUEST_FIELDS = {"messages", "model", "stream", "max_completion_tokens", "temperature", "top_p", "top_k"}


@pytest.fixture
def provider(monkeypatch, provider_keys):
    stub = provider_stub.ProviderStub()
    for variable, path in provider_stub.BASE_URL_VARIABLES.items():
        monkeypatch.setenv(variable, stub.base_url + path)
    monkeypatch.setattr(llm_generator, "SAMBANOVA_BASE_URL", f"{stub.base_url}/sambanova/v1")
    yield stub
    stub.stop()


@pytest.fixture
def client():
    return fastapi.testclient.TestClient(app_module.app)


def stream(llm_type):
    return list(services.generate_response(llm_type, "메뉴 추천", OPTIONS))


def sync(llm_type):
    return asyncio.run(services.generate_sync_response(llm_type, "메뉴 추천", OPTIONS))


@pytest.mark.parametrize("llm_type", MODELS)
def test_stream_contains_only_the_answer_text(provider, llm_type):
    assert "".join(stream(llm_type)) == "안녕하세요"


@pytest.mark.parametrize("llm_type", MODELS)
def test_sync_parses_the_answer_text_as_json(provider, llm_type):
    provider.parts = MENU_PARTS

    assert sync(llm_type) == [{"name": "파스타"}]


@pytest.mark.parametrize("llm_type", MODELS)
def test_sync_returns_plain_text_when_the_answer_is_not_json(provider, llm_type):
    assert sync(llm_type) == "안녕하세요"


def test_gemini_request_uses_a_current_model_and_keeps_thinking_minimal(provider):
    # 사고 토큰도 출력 상한에 포함된다. low에서는 2,048 중 1,043을 사고에 써서 답변이 잘린 사례가 있다.
    stream("gemini")

    request = provider.last_request
    config = request["body"]["generationConfig"]
    assert "models/gemini-3.6-flash:" in request["path"]
    assert config["maxOutputTokens"] == MAX_OUTPUT_TOKENS
    assert "MINIMAL" in config["thinkingConfig"].values()
    assert "temperature" not in config


def test_gemini_does_not_warn_about_ignored_sampling_parameters(provider, recwarn):
    stream("gemini")

    assert not [warning for warning in recwarn if "sampling" in str(warning.message)]


def test_llama_request_keeps_the_model_and_sampling_settings(provider):
    stream("llama")

    request = provider.last_request
    body = request["body"]
    assert request["path"] == "/sambanova/v1/chat/completions"
    assert body["model"] == "Meta-Llama-3.3-70B-Instruct"
    assert body["max_completion_tokens"] == MAX_OUTPUT_TOKENS
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.7, 0.9, 40)


def test_llama_request_only_sends_fields_sambanova_accepts(provider):
    # SambaNova는 스키마에 없는 필드를 400(unknown_parameter)으로 거부한다.
    stream("llama")

    assert provider.last_request["body"].keys() <= SAMBANOVA_REQUEST_FIELDS


def test_llama_sends_the_sambanova_key_and_not_the_openai_key(provider, monkeypatch):
    monkeypatch.setenv("SAMBANOVA_API_KEY", "sambanova-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    stream("llama")

    assert provider.last_request["authorization"] == "Bearer sambanova-key"


def test_llama_does_not_fall_back_to_the_openai_key_when_its_key_is_missing(provider, monkeypatch):
    # ChatOpenAI는 api_key가 없으면 OPENAI_API_KEY를 읽으므로 OpenAI 키가 SambaNova로 나갈 수 있다.
    monkeypatch.delenv("SAMBANOVA_API_KEY")

    with pytest.raises(KeyError, match="SAMBANOVA_API_KEY"):
        llm_picker.get_llm("llama")


def test_llama_targets_the_sambanova_api():
    # 다른 테스트는 이 주소를 대역으로 바꾸므로 실제 값은 여기서 확인한다.
    assert llm_generator.SAMBANOVA_BASE_URL == "https://api.sambanova.ai/v1"


def test_gpt_request_uses_the_replacement_of_the_retiring_model(provider):
    stream("gpt")

    request = provider.last_request
    body = request["body"]
    assert request["path"] == "/openai/chat/completions"
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == MAX_OUTPUT_TOKENS
    assert "temperature" not in body


def test_claude_request_omits_sampling_parameters_rejected_by_the_api(provider):
    stream("claude")

    body = provider.last_request["body"]
    assert body["model"] == "claude-haiku-5-5"
    assert body["max_tokens"] == MAX_OUTPUT_TOKENS
    assert body["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k"} & body.keys()


def test_deepseek_request_uses_a_current_model_and_turns_thinking_off(provider):
    stream("deepseek")

    body = provider.last_request["body"]
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == MAX_OUTPUT_TOKENS


def test_sse_endpoint_sends_only_the_answer_text(provider, client):
    response = client.post(
        "/streaming/sse",
        json={"llm_type": "gemini", "template": "메뉴 추천", "options": OPTIONS, "secret_key": SECRET_KEY},
    )

    assert response.status_code == 200
    assert [line for line in response.text.splitlines() if line] == ["data: 안녕", "data: 하세요"]


def test_sync_endpoint_returns_the_parsed_menu(provider, client):
    provider.parts = MENU_PARTS

    response = client.post(
        "/sync",
        json={"llm_type": "gemini", "template": "메뉴 추천", "options": OPTIONS, "secret_key": SECRET_KEY},
    )

    assert response.status_code == 200
    assert response.json() == [{"name": "파스타"}]
