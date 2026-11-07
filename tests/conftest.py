import os

import pytest

FRONTEND_URL = "https://frontend.test"
SECRET_KEY = "free-secret"
SECRET_KEY_FOR_PAID_MODEL = "paid-secret"
KV_REST_API_URL = "https://kv.test"
KV_REST_API_TOKEN = "kv-token-must-not-leak"

# app.py는 import 시점에 FRONTEND_URL을 읽고, dotenv는 이미 있는 값을 덮어쓰지 않는다.
os.environ["FRONTEND_URL"] = FRONTEND_URL
os.environ["SECRET_KEY"] = SECRET_KEY
os.environ["SECRET_KEY_FOR_PAID_MODEL"] = SECRET_KEY_FOR_PAID_MODEL
os.environ["KV_REST_API_URL"] = KV_REST_API_URL
os.environ["KV_REST_API_TOKEN"] = KV_REST_API_TOKEN

FAKE_PROVIDER_KEYS = {
    "OPENAI_API_KEY": "fake",
    "NCP_CLOVASTUDIO_API_KEY": "fake",
    "NCP_APIGW_API_KEY": "fake",
    "GOOGLE_API_KEY": "fake",
    "SAMBANOVA_API_KEY": "fake",
    "MISTRAL_API_KEY": "fake",
    "ANTHROPIC_API_KEY": "fake",
    "DEEPSEEK_API_KEY": "fake",
}


@pytest.fixture
def provider_keys(monkeypatch):
    for name, value in FAKE_PROVIDER_KEYS.items():
        monkeypatch.setenv(name, value)
