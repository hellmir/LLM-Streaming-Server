import os

import langchain_anthropic
import langchain_community.chat_models
import langchain_deepseek
import langchain_google_genai
import langchain_mistralai
import langchain_openai

SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"


def generate_samba_nova_cloud():
    return langchain_openai.ChatOpenAI(
        model_name="Meta-Llama-3.3-70B-Instruct",
        base_url=SAMBANOVA_BASE_URL,
        api_key=os.environ["SAMBANOVA_API_KEY"],  # api_key를 넘기지 않으면 ChatOpenAI가 OPENAI_API_KEY를 SambaNova로 보낸다
        max_tokens=2_048,
        temperature=0.7,
        top_p=0.9,
        extra_body={"top_k": 40}
    )


def generate_chat_clova_x():
    return langchain_community.chat_models.ChatClovaX(
        model="HCX-003",
        temperature=0.7,
        max_tokens=2_048,
        disable_streaming=False
    )


def generate_chat_google_generative_ai():
    return langchain_google_genai.ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        max_output_tokens=2_048,
        thinking_level="low"  # 사고 토큰도 max_output_tokens에 포함된다
    )


def generate_chat_open_ai():
    return langchain_openai.ChatOpenAI(
        model_name="gpt-6-luna",
        streaming=True,
        max_tokens=2_048,
        reasoning_effort="none"  # temperature는 추론을 끈 경우에만 받으므로 함께 넘기지 않는다
    )


def generate_chat_anthropic():
    return langchain_anthropic.ChatAnthropic(
        model="claude-haiku-5-5",
        max_tokens=2_048,
        effort="low"  # 사고 토큰도 max_tokens에 포함된다. temperature는 API가 400으로 거부한다
    )


def generate_chat_deep_seek():
    return langchain_deepseek.ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        max_tokens=2_048,
        timeout=None,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}}  # 사고가 기본으로 켜져 있고 사고 토큰이 상한을 잠식한다
    )


def generate_chat_mistral_ai():
    return langchain_mistralai.ChatMistralAI(
        model="ministral-14b-latest",
        temperature=0.7,
        max_tokens=2_048,
        max_retries=2
    )
