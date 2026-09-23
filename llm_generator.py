import langchain_anthropic
import langchain_community.chat_models
import langchain_community.llms.sambanova
import langchain_deepseek
import langchain_google_genai
import langchain_mistralai
import langchain_openai


def generate_samba_nova_cloud():
    return langchain_community.llms.sambanova.SambaNovaCloud(
        model="Meta-Llama-3.3-70B-Instruct",
        max_tokens=2_048,
        temperature=0.7,
        top_p=0.9,
        top_k=40
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
        model="gemini-1.5-pro",
        temperature=0.7,
        max_output_tokens=2_048
    )


def generate_chat_open_ai():
    return langchain_openai.ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        streaming=True,
        max_tokens=2_048
    )


def generate_chat_anthropic():
    return langchain_anthropic.ChatAnthropic(
        model='claude-3-haiku',
        temperature=0,
        max_tokens=2_048
    )


def generate_chat_deep_seek():
    return langchain_deepseek.ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=2_048,
        timeout=None,
        max_retries=2
    )


def generate_chat_mistral_ai():
    return langchain_mistralai.ChatMistralAI(
        model="ministral-14b-latest",
        temperature=0.7,
        max_tokens=2_048,
        max_retries=2
    )
