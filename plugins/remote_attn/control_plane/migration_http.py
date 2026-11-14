"""Inference and management HTTP endpoints for the DSV migration controller."""

from __future__ import annotations

import json
import socket
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit

from tokenizers.decoders import DecodeStream

from p4_dsv_gateway import DrainConflict


class CompletionFrames:
    def __init__(self, tokenizer, rid, model):
        self.tokenizer, self.rid, self.model = tokenizer, rid, model
        self.decoder = DecodeStream(skip_special_tokens=True)
        self.cursor = -1
        self.output_ids, self.emitted = [], []
        self.created = int(time.time())

    def chunk(self, text, finish_reason=None):
        return json.dumps(
            {
                "id": f"cmpl-{self.rid}",
                "object": "text_completion",
                "created": self.created,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            },
            ensure_ascii=False,
        ).encode()

    def _decode_token(self, event):
        token = event["token_id"]
        self.output_ids.append(token)
        self.cursor = event["seq"]
        text = self.decoder.step(self.tokenizer, token)
        if text:
            self.emitted.append(text)
        return text

    def tokens(self, events):
        for event in events:
            if text := self._decode_token(event):
                yield self.chunk(text)

    def _final_text(self):
        prefix = "".join(self.emitted)
        final = self.tokenizer.decode(self.output_ids, skip_special_tokens=True)
        if not final.startswith(prefix):
            raise RuntimeError("incremental decode changed the output prefix")
        return prefix, final

    def finish(self, reason):
        if reason == "abort":
            yield b'{"error":{"message":"generation aborted","type":"server_error"}}'
            return
        # A final incomplete byte sequence may still be buffered.
        prefix, final = self._final_text()
        if final[len(prefix) :]:
            yield self.chunk(final[len(prefix) :])
        yield self.chunk("", reason)
        yield b"[DONE]"


class GenerateFrames(CompletionFrames):
    def __init__(self, tokenizer, rid, prompt_tokens):
        super().__init__(tokenizer, rid, model=None)
        self.prompt_tokens = prompt_tokens

    def _chunk(self, text, reason=None):
        return json.dumps(
            {
                "text": text,
                "output_ids": self.output_ids,
                "meta_info": {
                    "id": self.rid,
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": len(self.output_ids),
                    "finish_reason": reason,
                },
            },
            ensure_ascii=False,
        ).encode()

    def tokens(self, events):
        if events:
            for event in events:
                self._decode_token(event)
            yield self._chunk("".join(self.emitted))

    def finish(self, reason):
        finish = {"type": reason}
        if reason == "length":
            finish["length"] = len(self.output_ids)
        if reason == "abort":
            finish.update(message="generation aborted", status_code=503)
        _, final = self._final_text()
        yield self._chunk(final, finish)
        yield b"[DONE]"


class MigrationWorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, status, body):
        self.controller.json_response(self, status, body)

    def error(self, status, message):
        kind = "invalid_request_error" if status < 500 else "server_error"
        self.reply(status, {"error": {"message": message, "type": kind}})

    def body(self):
        size = int(self.headers.get("Content-Length", 0))
        if size < 0 or size > 4 * 1024 * 1024:
            raise ValueError("invalid request size")
        body = json.loads(self.rfile.read(size))
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")
        return body

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            return self.reply(200, {"status": "ok"})
        if path == "/ready":
            ready = self.controller.ready()
            return self.reply(200 if ready else 503, {"ready": ready})
        if path in {
            "/model_info",
            "/get_model_info",
            "/server_info",
            "/get_server_info",
        }:
            return self.reply(200, {"served_model_name": self.model})
        if path == "/v1/models":
            return self.reply(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": self.model, "object": "model", "owned_by": "sglang"}
                    ],
                },
            )
        self.error(404, "unknown worker route")

    def do_POST(self):
        # Close on rejected requests too, since an unread body cannot be reused.
        self.close_connection = True
        if urlsplit(self.path).path != "/v1/completions":
            return self.error(404, "unknown worker route")
        try:
            request = self.completion_request(self.body())
            rid = self.controller.accept(request)
        except (ValueError, TypeError) as exc:
            return self.error(400, str(exc))
        except RuntimeError:
            return self.error(503, "worker is draining or unavailable")
        self.stream_completion(rid)

    def completion_request(self, body):
        if body.get("model") != self.model:
            raise ValueError("unknown model")
        if body.get("stream") is not True:
            raise ValueError("this worker requires stream=true")
        supported = {"model", "prompt", "stream", "max_tokens", "temperature", "top_p"}
        if body.keys() - supported:
            raise ValueError("unsupported completion parameters")
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            prompt = self.controller.tokenizer.encode(prompt)
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(token) is not int or token < 0 for token in prompt)
        ):
            raise ValueError("prompt must be text or a single token ID list")
        if any(token >= len(self.controller.tokenizer) for token in prompt):
            raise ValueError("prompt token ID is outside the tokenizer vocabulary")
        max_tokens = body.get("max_tokens", 128)
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p", 1.0)
        if type(temperature) not in (float, int) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if type(top_p) not in (float, int) or not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        return dict(
            prompt_ids=prompt,
            max_new_tokens=max_tokens,
            sampling_params=dict(
                temperature=temperature, top_p=top_p, ignore_eos=False
            ),
        )

    def stream_completion(self, rid):
        frames = CompletionFrames(
            self.controller.tokenizer.backend_tokenizer, rid, self.model
        )
        last_write = time.monotonic()
        completed = False

        def send(data):
            self.wfile.write(b"data: " + data + b"\n\n")
            self.wfile.flush()

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            while not self.controller.stop.is_set():
                try:
                    events, finished, reason = self.controller.stream_events(
                        rid, frames.cursor
                    )
                except ConnectionError:
                    events, finished, reason = [], False, None
                for frame in frames.tokens(events):
                    send(frame)
                    last_write = time.monotonic()
                if finished:
                    for frame in frames.finish(reason):
                        send(frame)
                    completed = True
                    return
                if time.monotonic() - last_write >= 0.5:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_write = time.monotonic()
                self.controller.stop.wait(min(self.controller.poll_s, 0.05))
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except (RuntimeError, ValueError):
            try:
                send(
                    b'{"error":{"message":"generation stream failed","type":"server_error"}}'
                )
            except OSError:
                pass
        finally:
            if not completed:
                self.controller.cancel(rid)


class MigrationAdminHandler(MigrationWorkerHandler):
    def do_GET(self):
        if urlsplit(self.path).path == "/v1/drain/status":
            return self.reply(200, self.controller.drain_status())
        self.error(404, "unknown admin route")

    def do_POST(self):
        self.close_connection = True
        path = urlsplit(self.path).path
        if path not in {"/v1/drain", "/v1/drain/abort"}:
            return self.error(404, "unknown admin route")
        try:
            body = self.body()
            if path == "/v1/drain/abort":
                drain_id = body.get("drain_id")
                if (
                    set(body) != {"drain_id"}
                    or not isinstance(drain_id, str)
                    or not drain_id
                ):
                    raise ValueError("expected a nonempty drain_id")
                result = self.controller.abort_named_drain(drain_id)
            else:
                if set(body) != {"src", "dst"}:
                    raise ValueError("expected src and dst instance names")
                result = self.controller.start_named_drain(body["src"], body["dst"])
        except ValueError as exc:
            return self.error(400, str(exc))
        except DrainConflict as exc:
            return self.error(409, str(exc))
        except (RuntimeError, OSError) as exc:
            return self.error(503, str(exc))
        self.reply(202, result)
