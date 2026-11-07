import asyncio
import json
import logging
import os

import dotenv
import httpx
import langchain_core.prompts

import llm_picker
import usage_store

LINE_BREAK = "\n"
TEMPLATE_HEADER = "Given the {options} I want you to create:" + LINE_BREAK
TEMPLATE_FOOTER = LINE_BREAK + "Answer in KOREAN"

logging.basicConfig(level=logging.INFO)
# httpx는 요청 주소를 INFO로 남기는데 Google 검색 주소에는 API 키가 들어 있다
logging.getLogger("httpx").setLevel(logging.WARNING)
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


dotenv.load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_IMAGE_API_KEY")
CX = os.environ.get("CUSTOM_SEARCH_ENGINE_ID")
SEARCH_URL = os.environ.get("SEARCH_URL")
DEFAULT_IMAGE_URL = os.environ.get("DEFAULT_IMAGE_URL")
SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE = "일일 검색 사용량을 모두 소진하여 기본 이미지로 대체되었습니다."


async def add_images(food_names, response_content):
    image_urls = await get_image_urls(food_names)

    for i, item in enumerate(response_content):
        if i < len(image_urls):
            item["imageUrls"] = image_urls[i]

    updated_json = json.dumps(response_content, ensure_ascii=False, indent=2)
    log.info("최종 JSON 데이터: %s", updated_json)


async def get_image_urls(food_names):
    if not reserve_image_search_usage(len(food_names)):
        return [[DEFAULT_IMAGE_URL] for _ in food_names]

    return await asyncio.gather(*[search_google_images(name) for name in food_names])


def reserve_image_search_usage(count):
    # 동시 요청이 한도를 함께 넘지 않도록 먼저 올리고, 한도에 걸리면 되돌린다
    try:
        # 그 사이 날짜가 바뀌어도 올린 날짜의 카운터를 되돌리도록 날짜를 한 번만 구한다
        usage_day = usage_store.current_usage_day()
        image_search_usage = usage_store.increase_image_search_usage(count, usage_day)
        log.info("일일 이미지 검색 사용량: %s", image_search_usage)

        # 이미지 검색 일일 사용 한도를 100회로 제한
        if image_search_usage < 100:
            return True

        usage_store.decrease_image_search_usage(count, usage_day)
        log.warning(SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE)
        return False
    except usage_store.UsageStoreUnavailableError as e:
        log.warning("이미지 검색 사용량 확인 실패: %s", e)
        return False


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
