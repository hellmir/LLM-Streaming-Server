"""제공사 API 대역. 잠금 버전 LangChain 래퍼가 보내는 요청과 응답 처리를 네트워크 없이 확인한다.

응답 모양은 제공사 문서를 따른다.
- Claude Haiku 5.5: thinking 파라미터 없이도 thinking 블록이 텍스트 앞에 올 수 있다
- Gemini 3: 응답의 마지막 파트에 thoughtSignature가 붙을 수 있다
- DeepSeek: 사고를 끄지 않으면 reasoning_content가 먼저 온다
- SambaNova: 스키마상 delta의 필드가 모두 선택이고(content는 null도 허용) 마지막 청크는 choices가 비어 있다
"""
import http.server
import json
import threading

BASE_URL_VARIABLES = {
    "ANTHROPIC_API_URL": "/anthropic",
    "OPENAI_API_BASE": "/openai",
    "DEEPSEEK_API_BASE": "/deepseek",
    "GOOGLE_GEMINI_BASE_URL": "/gemini",
}


def _anthropic_events(parts):
    usage = {"input_tokens": 10, "output_tokens": 1}
    message = {"id": "msg_1", "type": "message", "role": "assistant", "model": "stub", "content": [],
               "stop_reason": None, "stop_sequence": None, "usage": usage}
    yield {"type": "message_start", "message": message}
    yield {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}
    yield {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "c2ln"}}
    yield {"type": "content_block_stop", "index": 0}
    yield {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}
    for part in parts:
        yield {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": part}}
    yield {"type": "content_block_stop", "index": 1}
    yield {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
           "usage": {"output_tokens": 5}}
    yield {"type": "message_stop"}


def _anthropic_message(parts):
    content = [{"type": "thinking", "thinking": "", "signature": "c2ln"}, {"type": "text", "text": "".join(parts)}]
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "stub", "content": content,
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}


def _gemini_response(text, finished):
    if not finished:
        return {"candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "index": 0}]}
    part = {"text": text, "thoughtSignature": "c2ln"}
    return {"candidates": [{"content": {"parts": [part], "role": "model"}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}


def _gemini_events(parts):
    for index, part in enumerate(parts):
        yield _gemini_response(part, finished=index == len(parts) - 1)


def _openai_chunk(delta, finish_reason=None):
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "stub",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def _openai_events(parts, reasoning=False):
    yield _openai_chunk({"role": "assistant", "content": ""})
    if reasoning:
        yield _openai_chunk({"reasoning_content": "생각"})
    for part in parts:
        yield _openai_chunk({"content": part})
    yield _openai_chunk({}, finish_reason="stop")


def _sambanova_events(parts):
    yield from _openai_events(parts)
    yield _openai_chunk({"content": None}, finish_reason="stop")
    yield {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "stub", "choices": [],
           "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _openai_message(parts):
    message = {"role": "assistant", "content": "".join(parts)}
    return {"id": "c1", "object": "chat.completion", "created": 1, "model": "stub",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class ProviderStub:
    def __init__(self):
        self.parts = ["안녕", "하세요"]
        self.requests = []
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                stub.requests.append({"path": self.path, "body": body, "authorization": self.headers.get("authorization")})
                stub.respond(self, body)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    @property
    def last_request(self):
        return self.requests[-1]

    def respond(self, handler, body):
        path = handler.path

        if path.startswith("/anthropic"):
            if body.get("stream"):
                return self._send_events(handler, _anthropic_events(self.parts), named=True)
            return self._send_json(handler, _anthropic_message(self.parts))

        if path.startswith("/gemini"):
            if "streamGenerateContent" in path:
                return self._send_events(handler, _gemini_events(self.parts))
            return self._send_json(handler, _gemini_response("".join(self.parts), finished=True))

        if body.get("stream") and path.startswith("/sambanova"):
            return self._send_events(handler, _sambanova_events(self.parts), done=True)

        if body.get("stream"):
            reasoning = path.startswith("/deepseek") and body.get("thinking", {}).get("type") != "disabled"
            return self._send_events(handler, _openai_events(self.parts, reasoning), done=True)
        return self._send_json(handler, _openai_message(self.parts))

    @staticmethod
    def _send_json(handler, payload):
        data = json.dumps(payload, ensure_ascii=False).encode()
        handler.send_response(200)
        handler.send_header("content-type", "application/json")
        handler.send_header("content-length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @staticmethod
    def _send_events(handler, events, named=False, done=False):
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.send_header("connection", "close")
        handler.end_headers()
        for event in events:
            prefix = f"event: {event['type']}\n" if named else ""
            handler.wfile.write(f"{prefix}data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
        if done:
            handler.wfile.write(b"data: [DONE]\n\n")
        handler.close_connection = True
