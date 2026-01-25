import json  # GET SSE에서 options 파싱
import logging
import os
import time  # 하트비트
import typing

import dotenv
import fastapi
import pydantic
import starlette.responses
from fastapi.responses import StreamingResponse

import global_exception_handler
import services
from stream_formatter import normalize_chunks_to_lines

router = fastapi.APIRouter(prefix="")

paid_model_usage = 0
MODEL_DESCRIPTION = """
    \n- llm_type: LLM 모델 유형 (미입력 시 기본 모델 사용)
    \n  - mistral (현재 기본 모델)
    \n  - clovax
    \n  - gemini
    \n  - llama
    \n  - gpt (유료 모델: 관리자로부터 Secret Key 발급 필요)
    \n  - claude (유료 모델: 관리자로부터 Secret Key 발급 필요)
    \n  - deepseek (유료 모델: 관리자로부터 Secret Key 발급 필요)
    \n- template: 요청 프롬프트(요구사항)
    \n- options: 요청 프롬프트에 적용할 조건들(선택사항)
    \n  - Key: Array Value 형태로 전송
    \n- Secret Key: 사용자 인증을 위한 비밀 키
"""

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gpt_logger")


class LLMRequest(pydantic.BaseModel):
    llm_type: typing.Optional[str] = pydantic.Field(default=None,
                                                    description="LLM 모델 유형(mistral, gemini, llama, clovax, gpt, claude, deepseek): 미입력 시 기본 모델 사용",
                                                    examples=["mistral", "gemini", "llama", "clovax", "gpt", "claude",
                                                              "deepseek"])
    template: str = pydantic.Field(..., description="요청 프롬프트(요구사항)", examples=["레시피 추천해 줘"])
    options: typing.Optional[typing.Dict[str, typing.List[str]]] = pydantic.Field(
        default=None,
        description="요청 프롬프트에 적용할 조건들(선택사항)",
        examples=[
            {
                "재료": ["토마토", "밀가루", "양파"],
                "조리 도구": ["프라이팬", "오븐", "믹서기"],
                "조리 시간": ["30분"],
                "요리 유형": ["이탈리안"],
                "식사 유형": ["아침"]
            }
        ]
    )
    secret_key: str = pydantic.Field(..., description="사용자 인증을 위한 비밀 키", examples=["abcde"])

    class Config:
        title = "LLM 서비스 요청 파라미터"


# GET SSE용 쿼리 파라미터 모델 (EventSource 전용)
class LLMQuery(pydantic.BaseModel):
    llm_type: typing.Optional[str] = None
    template: str
    options: typing.Optional[str] = None  # JSON 문자열로 전달 (URL 인코딩)
    secret_key: str


@router.post(
    "/sync",
    summary="LLM 서비스 이용",
    description="다음 파라미터를 전송해 LLM 서비스를 이용할 수 있습니다. 전체 응답이 한 번에 제공됩니다." + MODEL_DESCRIPTION
)
async def invoke_llm_sync(request: LLMRequest):
    dotenv.load_dotenv()
    authenticate((request.llm_type or "").lower(), request.secret_key)

    llm_type = request.llm_type if request.llm_type else ""
    options = request.options if request.options else {}

    response_content = await services.generate_sync_response(llm_type, request.template, options)

    if isinstance(response_content, set):
        response_content = list(response_content)

    return starlette.responses.JSONResponse(content=response_content)


@router.post(
    "/streaming",
    summary="LLM 스트리밍 서비스 이용",
    description="다음 파라미터를 전송해 LLM 서비스를 이용할 수 있습니다. 응답은 토큰화하여 Stream 형태로 제공됩니다." + MODEL_DESCRIPTION
)
def invoke_llm_streaming(request: LLMRequest):
    dotenv.load_dotenv()
    authenticate((request.llm_type or "").lower(), request.secret_key)

    llm_type = request.llm_type if request.llm_type else ""
    options = request.options if request.options else {}

    def response_generator():
        try:  # 방어 로깅
            lines = normalize_chunks_to_lines(
                services.generate_response(llm_type, request.template, options),
                llm_type=llm_type
            )
            for line in lines:
                yield line + "\n"
        except Exception as e:
            log.exception("streaming error: %s", e)

    return StreamingResponse(response_generator(), media_type="text/plain; charset=utf-8")


# 표준 EventSource 지원: GET SSE 엔드포인트 추가
@router.get(
    "/streaming/sse",
    summary="LLM 스트리밍 SSE (GET, EventSource)",
    description="EventSource로 연결하는 표준 SSE 엔드포인트입니다." + MODEL_DESCRIPTION,
)
def invoke_llm_sse_get(
        llm_type: typing.Optional[str] = None,
        template: str = fastapi.Query(...),
        options: typing.Optional[str] = None,  # JSON string
        secret_key: str = fastapi.Query(...)
):
    dotenv.load_dotenv()
    input_llm_type = (llm_type or "").lower()
    authenticate(input_llm_type, secret_key)

    parsed_options: typing.Dict[str, typing.List[str]] = {}
    if options:
        try:
            parsed_options = json.loads(options)
        except json.JSONDecodeError:
            parsed_options = {}

    llm_type_val = llm_type or ""
    options_val = parsed_options

    def _sse_event_with_newline(line: str) -> str:
        return f"data: {line}\ndata: \n\n"

    def sse_response_generator():
        last_beat = time.time()
        try:
            for line in normalize_chunks_to_lines(
                    services.generate_response(llm_type_val, template, options_val),
                    llm_type=llm_type_val
            ):
                # 데이터 이벤트
                yield _sse_event_with_newline(line)
                # 하트비트(프록시/브라우저 타임아웃 방지)
                if time.time() - last_beat > 15:
                    yield ": keep-alive\n\n"
                    last_beat = time.time()
        except Exception as e:
            # 에러 이벤트로 통지 후 종료
            err = f"event: error\ndata: {str(e)}\n\n"
            yield err

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        sse_response_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers=headers
    )


# 기존 POST SSE도 유지(백워드 호환). fetch 스트리밍용
@router.post(
    "/streaming/sse",
    summary="LLM 스트리밍 SSE (POST, fetch)",
    description="fetch(POST)로 스트리밍 받을 때 사용하는 SSE 호환 스트림입니다." + MODEL_DESCRIPTION
)
def invoke_llm_sse_post(request: LLMRequest):
    dotenv.load_dotenv()
    authenticate((request.llm_type or "").lower(), request.secret_key)

    llm_type = request.llm_type if request.llm_type else ""
    options = request.options if request.options else {}

    def _sse_event_with_newline(line: str) -> str:
        # SSE data lines are joined with "\n"; add an empty data line to preserve line breaks.
        return f"data: {line}\ndata: \n\n"

    def sse_response_generator():
        last_beat = time.time()
        try:
            for line in normalize_chunks_to_lines(
                    services.generate_response(llm_type, request.template, options),
                    llm_type=llm_type
            ):
                yield _sse_event_with_newline(line)
                if time.time() - last_beat > 15:
                    yield ": keep-alive\n\n"
                    last_beat = time.time()
        except Exception as e:
            err = f"event: error\ndata: {str(e)}\n\n"
            yield err

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        sse_response_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers=headers
    )


def authenticate(input_llm_type, input_secret_key):
    if is_paid_model(input_llm_type):
        authorize_paid_model(input_llm_type, input_secret_key)
    elif input_secret_key != os.environ.get("SECRET_KEY"):
        global_exception_handler.throw_unauthorized_exception()


def is_paid_model(llm_type):
    return llm_type in ("gpt", "claude", "deepseek")


def authorize_paid_model(input_llm_type, input_secret_key):
    global paid_model_usage

    if input_secret_key != os.environ.get("SECRET_KEY_FOR_PAID_MODEL"):
        global_exception_handler.throw_paid_model_unauthorized_exception(input_llm_type)
    else:
        paid_model_usage += 1

        # 유료 모델 일일 사용 한도를 10,000회로 제한
        if paid_model_usage > 10_000:
            global_exception_handler.throw_paid_model_usage_limit_exception(input_llm_type)
        log.info("%s API가 호출되었습니다. 일일 사용량: %d", input_llm_type, paid_model_usage)


def reset_gpt_usage():
    global paid_model_usage
    paid_model_usage = 0
    log.info("유료 모델 사용량이 초기화 되었습니다. 일일 사용량: %d", paid_model_usage)
