import pytest

import llm_picker


@pytest.mark.parametrize(
    "llm_type, expected_class",
    [
        ("llama", "ChatOpenAI"),
        ("clovax", "ChatClovaX"),
        ("gemini", "ChatGoogleGenerativeAI"),
        ("gpt", "ChatOpenAI"),
        ("claude", "ChatAnthropic"),
        ("deepseek", "ChatDeepSeek"),
        ("mistral", "ChatMistralAI"),
    ],
)
def test_each_model_is_built_with_locked_packages(provider_keys, llm_type, expected_class):
    assert type(llm_picker.get_llm(llm_type)).__name__ == expected_class


@pytest.mark.parametrize("llm_type", ["default", "", "unknown"])
def test_other_values_fall_back_to_default_model(provider_keys, llm_type):
    assert type(llm_picker.get_llm(llm_type)).__name__ == "ChatMistralAI"
