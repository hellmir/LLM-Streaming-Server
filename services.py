import asyncio
import json
import logging
import os

import dotenv
import httpx
import langchain_core.prompts

import llm_picker

LINE_BREAK = "\n"
TEMPLATE_HEADER = "Given the {options} I want you to create:" + LINE_BREAK
TEMPLATE_FOOTER = LINE_BREAK + "Answer in KOREAN"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("services_logger")


def generate_response(llm_type: str, template_content: str, options: str):
    prompt = langchain_core.prompts.PromptTemplate(
        input_variables=["options"],
        template=TEMPLATE_HEADER + template_content + TEMPLATE_FOOTER
    )
    llm = llm_picker.get_llm(llm_type.lower() if llm_type else "")  # [CHANGED] None 안전 처리
    chain = prompt | llm
    i = 0

    for chunk in chain.stream({"options": options}):
        if hasattr(chunk, "content"):
            i += 1
            log.info("chunk %s: %s", str(i), chunk)
            yield chunk.content
        else:
            yield str(chunk)


async def generate_sync_response(llm_type: str, template_content: str, options: str):
    prompt = langchain_core.prompts.PromptTemplate(
        input_variables=["options"],
        template=TEMPLATE_HEADER + template_content + TEMPLATE_FOOTER
    )
    llm = llm_picker.get_llm(llm_type.lower() if llm_type else "")  # [CHANGED] None 안전 처리
    chain = prompt | llm

    response = chain.invoke({"options": options})
    log.info("LLM API로부터의 수신 데이터: " + (response.content if hasattr(response, "content") else str(response)))

    if hasattr(response, "content"):
        response_content = response.content

        try:
            response_content = json.loads(response_content)
        except json.JSONDecodeError:
            log.info("JSON 파싱 실패: %s", response_content)
            return response_content

        food_names = [item["name"] for item in response_content if "name" in item]
        log.info("추천 메뉴 목록: %s", food_names)

        if isinstance(options, dict) and "ingredients" in options:  # [CHANGED] 옵션 타입/키 방어
            await add_images(food_names, response_content)

        return response_content

    return response


image_search_usage = 0

dotenv.load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_IMAGE_API_KEY")
CX = os.environ.get("CUSTOM_SEARCH_ENGINE_ID")
SEARCH_URL = os.environ.get("SEARCH_URL")
DEFAULT_IMAGE_URL = os.environ.get("DEFAULT_IMAGE_URL")
SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE = "일일 검색 사용량을 모두 소진하여 기본 이미지로 대체되었습니다."


async def add_images(food_names, response_content):
    global image_search_usage
    log.info("일일 이미지 검색 사용량: %s", image_search_usage)

    image_urls = []

    # 이미지 검색 일일 사용 한도를 100회로 제한
    if image_search_usage + len(food_names) >= 100:
        log.warning(SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE)

        # [CHANGED] 기본 이미지 채우기 로직 수정 (list로 일괄 채움)
        image_urls = [[DEFAULT_IMAGE_URL] for _ in food_names]
    else:
        image_urls = await asyncio.gather(*[search_google_images(name) for name in food_names])
        image_search_usage += len(image_urls)

    for i, item in enumerate(response_content):
        if i < len(image_urls):
            item["imageUrls"] = image_urls[i]

    updated_json = json.dumps(response_content, ensure_ascii=False, indent=2)
    log.info("최종 JSON 데이터: %s", updated_json)


async def search_google_images(food_name):
    params = {
        "key": GOOGLE_API_KEY,
        "cx": CX,
        "q": food_name,
        "searchType": "image",
        "num": os.environ.get("NEEDED_IMAGE_COUNT")
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(SEARCH_URL, params=params)

    if response.status_code != 200:
        log.warning(SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE)
        return [DEFAULT_IMAGE_URL]

    data = response.json()
    image_urls = [item["link"] for item in data.get("items", [])]
    for image_url in image_urls:
        log.info("이미지 주소: %s", image_url)

    return image_urls


def reset_image_search_usage():
    global image_search_usage  # [CHANGED] 변수명 오류 수정 (paid_model_usage -> image_search_usage)
    image_search_usage = 0
    log.info("이미지 검색 사용량이 초기화 되었습니다. 일일 사용량: %d", image_search_usage)
