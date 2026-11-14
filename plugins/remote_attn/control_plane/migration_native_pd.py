"""SMG-owned Prefill and Decode ingress for the DSV migration controller."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from migration_controller import MigrationController
from migration_http import CompletionFrames, GenerateFrames
from p4_dsv_gateway import DrainConflict

MAX_REQUEST_BYTES = 4 * 1024 * 1024
COMPLETION_FIELDS = {
    "model",
    "prompt",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "rid",
    "user",
}
NEUTRAL_FIELDS = {
    "n": 1,
    "best_of": 1,
    "echo": False,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "ignore_eos": False,
    "skip_special_tokens": True,
    "min_tokens": 0,
    "top_k": -1,
    "min_p": 0.0,
    "no_stop_trim": False,
    "return_hidden_states": False,
    "return_routed_experts": False,
    "return_cached_tokens_details": False,
    "return_spec_tokens_details": False,
    "return_token_ids": False,
}
GENERATE_FIELDS = {
    "model",
    "input_ids",
    "sampling_params",
    "stream",
    "rid",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
}
GENERATE_NEUTRAL_FIELDS = {
    "return_logprob": False,
    "return_text_in_logprobs": False,
    "log_metrics": True,
    "return_hidden_states": False,
    "background": False,
    "no_logs": False,
    "return_bytes": False,
    "return_entropy": False,
}


def http_base(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
    ):
        raise ValueError("expected a base HTTP URL without credentials or a path")
    return url.rstrip("/")


class NativePdController(MigrationController):
    native_pd = True

    def __init__(
        self,
        *args,
        decode_urls,
        smg_admin_urls,
        smg_admin_token_file=None,
        smg_cordon_timeout_s=15,
        decode_workers,
        autostart=True,
        **kwargs,
    ):
        if type(smg_cordon_timeout_s) is not int or smg_cordon_timeout_s <= 0:
            raise ValueError("smg_cordon_timeout_s must be a positive integer")
        self.smg_cordon_timeout_s = smg_cordon_timeout_s
        self.decode_urls = {name: http_base(url) for name, url in decode_urls.items()}
        if not self.decode_urls.keys() <= decode_workers.keys() or not self.decode_urls:
            raise ValueError("native PD Decode listeners must name registered workers")
        self.listeners = {}
        for name, url in self.decode_urls.items():
            parsed = urlsplit(url)
            if (
                parsed.scheme != "http"
                or not parsed.port
                or parsed.port in self.listeners
            ):
                raise ValueError("Decode listeners require unique explicit HTTP ports")
            self.listeners[parsed.port] = name
        self.smg_admin_urls = tuple(http_base(url) for url in smg_admin_urls)
        if not self.smg_admin_urls:
            raise ValueError(
                "configure the management URL of each SMG serving this pool"
            )
        self.smg_admin_token_file = (
            Path(smg_admin_token_file) if smg_admin_token_file else None
        )
        self.cordon_error = None
        self.cordon_wakeup = threading.Event()
        self.cordon_watchdog = None
        super().__init__(
            *args, decode_workers=decode_workers, autostart=autostart, **kwargs
        )
        if autostart:
            self.cordon_watchdog = threading.Thread(
                target=self._watch_cordon_timeout,
                daemon=True,
                name="smg-cordon-timeout",
            )
            self.cordon_watchdog.start()

    def close(self):
        self.stop.set()
        self.cordon_wakeup.set()
        if self.cordon_watchdog is not None:
            self.cordon_watchdog.join(timeout=5)
            if self.cordon_watchdog.is_alive():
                raise RuntimeError("SMG cordon watchdog is still active")
        super().close()

    def _validate_worker_bindings(self):
        super()._validate_worker_bindings()
        config = dict(
            decode_urls=self.decode_urls, smg_admin_urls=list(self.smg_admin_urls)
        )
        if self.path.exists() and self.journal.get("native_routes") != config:
            raise ValueError("migration journal native PD routes changed")
        self.journal.setdefault("native_routes", config)
        self.journal.setdefault("native_cordon", None)
        self.journal.setdefault("late_routes", {})

    def decode_ready(self, worker):
        journal = self.journal
        return (
            worker not in self.standby
            and worker not in journal["fenced_workers"]
            and worker != (journal["native_cordon"] or {}).get("src")
            and self._decode_ready(worker)
        )

    def _external_admission(self, role, bootstrap):
        if role not in self.decode_urls or role in self.standby:
            raise RuntimeError("Decode listener is unavailable")
        journal = self.journal
        # SMG may already have dispatched P for a request selected before its
        # health probe. Follow the explicit destination(s) with that same room.
        worker = role
        while worker in journal["fenced_workers"]:
            worker = journal["late_routes"].get(worker)
            if worker is None:
                raise RuntimeError("Decode listener is fenced")
        if not self._decode_ready(worker):
            raise RuntimeError("Decode worker is unavailable")
        return dict(
            owner="src",
            initial_owner="src",
            workers={"src": worker},
            instances={"src": [r["instance_id"] for r in self.facts[worker]]},
            prefill_url=None,
            external_prefill=True,
            **bootstrap,
        )

    def accept_native(self, role, body, bootstrap):
        with self.cv:
            host = bootstrap["bootstrap_host"]
            port = bootstrap["bootstrap_port"]
            if not any(
                (host, port) == (prefill.bootstrap_host, prefill.bootstrap_port)
                for prefill in self.prefills.values()
            ):
                raise ValueError("bootstrap address is not a registered Prefill")
            key = json.dumps([host, port, bootstrap["bootstrap_room"]]).encode()
            rid = "pd-" + hashlib.sha256(key).hexdigest()
            if rid in self.journal["entries"]:
                raise DrainConflict("bootstrap room was already admitted")
            return self.accept({**body, "rid": rid}, role=role, external_pd=bootstrap)

    def start_named_drain(self, src, dst):
        self._validate_drain_pair(src, dst)
        if src not in self.decode_urls:
            raise ValueError("source has no SMG Decode listener")
        with self.cv:
            pending = self.journal["native_cordon"]
            if pending:
                if (pending["src"], pending["dst"]) != (src, dst):
                    raise DrainConflict(
                        "an SMG cordon is already active in this model pool"
                    )
                return self.drain_status()
            existing = self._existing_drain(src, dst)
            if existing:
                return existing
            instances = self._drain_instances(src, dst)
            started_at = time.time()
            cordon = dict(
                id=uuid.uuid4().hex,
                active=True,
                src=src,
                dst=dst,
                instances=instances,
                started_at=started_at,
                cordon_deadline_at=started_at + self.smg_cordon_timeout_s,
            )
            self._publish_journal(native_cordon=cordon)
            self.cordon_error = None
            self.cordon_wakeup.set()
            return self.drain_status()

    def _cordon_deadline(self, pending):
        return pending.get(
            "cordon_deadline_at", pending["started_at"] + self.smg_cordon_timeout_s
        )

    def _abort_cordon(self, pending, reason):
        now = time.time()
        error = self.cordon_error
        if error is None and reason == "cordon_timeout":
            error = "SMG confirmations did not arrive before the deadline"
        self._publish_journal(
            native_cordon=None,
            drain={
                **pending,
                "mode": "d2d",
                "active": False,
                "members": [],
                "cordon_deadline_at": self._cordon_deadline(pending),
                "cordon_error": error,
                "abort_requested_at": now,
                "aborted_at": now,
                "abort_reason": reason,
            },
        )
        self.cordon_error = None

    def abort_named_drain(self, drain_id):
        with self.cv:
            pending = self.journal["native_cordon"]
            if pending is not None:
                if pending["id"] != drain_id:
                    raise DrainConflict("drain id does not match the current cordon")
                self._abort_cordon(pending, "operator_abort")
                return self.drain_status()
            return super().abort_named_drain(drain_id)

    def drain_status(self):
        with self.cv:
            pending = self.journal["native_cordon"]
            if pending is not None:
                return {
                    **pending,
                    "state": "CORDONING",
                    "members": [],
                    "cordon_deadline_at": self._cordon_deadline(pending),
                    "cordon_error": self.cordon_error,
                }
            return super().drain_status()

    def _smg_cordoned(self, pending):
        url = self.decode_urls[pending["src"]]
        headers = {}
        if self.smg_admin_token_file is not None:
            headers["Authorization"] = (
                "Bearer " + self.smg_admin_token_file.read_text().strip()
            )
        for gateway in self.smg_admin_urls:
            remaining = self._cordon_deadline(pending) - time.time()
            if self.stop.is_set() or remaining <= 0:
                return False
            try:
                with urlopen(
                    Request(gateway + "/workers", headers=headers),
                    timeout=min(2, remaining),
                ) as response:
                    workers = json.load(response)["workers"]
                matches = [
                    row
                    for row in workers
                    if row["url"].rstrip("/") == url
                    and row["worker_type"].lower() == "decode"
                ]
                if any(row["is_healthy"] is not False for row in matches):
                    raise RuntimeError(f"Decode worker {url} is still healthy")
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                RuntimeError,
            ) as exc:
                raise RuntimeError(f"{gateway}/workers: {exc}") from exc
            if not matches:
                raise RuntimeError(f"{gateway} has no Decode worker {url}")
        return True

    def _expire_cordon(self):
        with self.cv:
            pending = self.journal["native_cordon"]
            if pending is None or time.time() < self._cordon_deadline(pending):
                return
            try:
                self._abort_cordon(pending, "cordon_timeout")
            except OSError as exc:
                self.cordon_error = f"could not persist cordon timeout: {exc}"

    def _watch_cordon_timeout(self):
        # A management HTTP call can outlive its per-socket timeout.
        while not self.stop.is_set():
            self._expire_cordon()
            self.stop.wait(0.25)

    def _reconcile_cordon(self):
        self._expire_cordon()
        pending = self.journal["native_cordon"]
        if not pending:
            return
        error = None
        try:
            if self._smg_cordoned(pending) and not self.stop.is_set():
                self._activate_drain(
                    pending["src"],
                    pending["dst"],
                    cordon={
                        **pending,
                        "cordon_deadline_at": self._cordon_deadline(pending),
                    },
                )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            error = str(exc)
        with self.cv:
            if error is not None and (
                (self.journal["native_cordon"] or {}).get("id") == pending["id"]
            ):
                self.cordon_error = error
        self._expire_cordon()

    def _poll_routing(self):
        while not self.stop.is_set():
            self.cordon_wakeup.clear()
            if self.stop.is_set():
                return
            self._reconcile_cordon()
            self.cordon_wakeup.wait(0.05 if self.journal.get("native_cordon") else 0.25)


def parse_completion(body, model, vocab_size):
    if not isinstance(body, dict) or body.get("model") != model:
        raise ValueError("expected a completion request for the registered model")
    if body.get("stream") is not True:
        raise ValueError("native PD migration requires stream=true")
    # SMG reserializes typed defaults and optional nulls before dispatch.
    for field, value in body.items():
        if field in COMPLETION_FIELDS or value is None:
            continue
        if field not in NEUTRAL_FIELDS or value != NEUTRAL_FIELDS[field]:
            raise ValueError(f"unsupported completion parameter: {field}")
    prompt = body.get("prompt")
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(
            type(token) is not int or not 0 <= token < vocab_size for token in prompt
        )
    ):
        raise ValueError("prompt must be one pretokenized input")
    host, port, room = (
        body.get("bootstrap_" + key) for key in ("host", "port", "room")
    )
    if (
        not isinstance(host, str)
        or not host
        or type(port) is not int
        or not 0 < port < 65536
        or type(room) is not int
        or not 0 < room < 2**63
    ):
        raise ValueError("expected scalar SMG bootstrap host, port and room")
    max_tokens = body.get("max_tokens", 16)
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    temperature, top_p = body.get("temperature", 1.0), body.get("top_p", 1.0)
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    request = dict(
        prompt_ids=prompt,
        max_new_tokens=max_tokens,
        sampling_params=dict(temperature=temperature, top_p=top_p, ignore_eos=False),
    )
    return request, dict(bootstrap_host=host, bootstrap_port=port, bootstrap_room=room)


def parse_generate(body, model, vocab_size):
    if not isinstance(body, dict):
        raise ValueError("expected a generate request object")
    for field, value in body.items():
        if field in GENERATE_FIELDS or value is None:
            continue
        if (
            field not in GENERATE_NEUTRAL_FIELDS
            or value != GENERATE_NEUTRAL_FIELDS[field]
        ):
            raise ValueError(f"unsupported generate parameter: {field}")
    sampling = body.get("sampling_params")
    if not isinstance(sampling, dict) or "max_new_tokens" not in sampling:
        raise ValueError("sampling_params must include max_new_tokens")
    if set(sampling) - {"max_new_tokens", "temperature", "top_p", "ignore_eos"}:
        raise ValueError("unsupported generate sampling parameter")
    ignore_eos = sampling.get("ignore_eos", False)
    if type(ignore_eos) is not bool:
        raise ValueError("ignore_eos must be a boolean")
    request, bootstrap = parse_completion(
        {
            "model": model if body.get("model") is None else body["model"],
            "prompt": body.get("input_ids"),
            "stream": body.get("stream"),
            "max_tokens": sampling["max_new_tokens"],
            "temperature": sampling.get("temperature", 1.0),
            "top_p": sampling.get("top_p", 1.0),
            **{
                field: body.get(field)
                for field in ("bootstrap_host", "bootstrap_port", "bootstrap_room")
            },
        },
        model,
        vocab_size,
    )
    request["sampling_params"]["ignore_eos"] = ignore_eos
    return request, bootstrap


async def _response(send, status, payload):
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": json.dumps(payload).encode()})


async def _error(send, status, message):
    return await _response(
        send,
        status,
        {
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
            }
        },
    )


async def _read_body(receive):
    body = bytearray()
    while True:
        event = await receive()
        if event["type"] == "http.disconnect":
            return None
        body.extend(event["body"])
        if len(body) > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        if not event.get("more_body"):
            return bytes(body)


async def _wait_disconnect(receive):
    while (await receive())["type"] != "http.disconnect":
        pass


class NativeDecodeApp:
    def __init__(self, controller, model):
        self.controller, self.model = controller, model

    async def __call__(self, scope, receive, send):
        worker = self.controller.listeners[scope["server"][1]]
        path, method = scope["path"], scope["method"]
        if method == "GET":
            return await self._get(worker, path, send)
        if method != "POST" or path not in {"/v1/completions", "/generate"}:
            return await _error(send, 404, "unknown Decode route")
        generate = path == "/generate"
        try:
            body = await _read_body(receive)
            if body is None:
                return
            parser = parse_generate if generate else parse_completion
            request, bootstrap = parser(
                json.loads(body), self.model, len(self.controller.tokenizer)
            )
        except (ValueError, UnicodeError) as exc:
            return await _error(send, 400, str(exc))
        stream = asyncio.create_task(
            self._completion(worker, request, bootstrap, send, generate=generate)
        )
        disconnect = asyncio.create_task(_wait_disconnect(receive))
        try:
            done, _ = await asyncio.wait(
                (stream, disconnect), return_when=asyncio.FIRST_COMPLETED
            )
            if stream in done:
                await stream
        finally:
            for task in (stream, disconnect):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stream, disconnect, return_exceptions=True)

    async def _get(self, worker, path, send):
        if path in {"/health", "/ready", "/health_generate"}:
            ready = self.controller.decode_ready(worker)
            return await _response(
                send, 200 if ready else 503, {"ready": ready, "decode": worker}
            )
        if path in {
            "/model_info",
            "/get_model_info",
            "/server_info",
            "/get_server_info",
        }:
            return await _response(
                send,
                200,
                {
                    "served_model_name": self.model,
                    "disaggregation_mode": "decode",
                    "tp_size": len(self.controller.facts.get(worker, ())) or None,
                    "dp_size": 1,
                },
            )
        if path == "/v1/models":
            return await _response(
                send,
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": self.model, "object": "model", "owned_by": "sglang"}
                    ],
                },
            )
        return await _error(send, 404, "unknown Decode route")

    async def _completion(self, worker, request, bootstrap, send, *, generate=False):
        controller = self.controller
        submit = asyncio.create_task(
            asyncio.to_thread(controller.accept_native, worker, request, bootstrap)
        )
        rid = None
        started = completed = False
        try:
            try:
                rid = await asyncio.shield(submit)
            except DrainConflict as exc:
                return await _error(send, 409, str(exc))
            except ValueError as exc:
                return await _error(send, 400, str(exc))
            except (RuntimeError, OSError):
                return await _error(send, 503, "Decode worker is unavailable")
            tokenizer = controller.tokenizer.backend_tokenizer
            frames = (
                GenerateFrames(tokenizer, rid, len(request["prompt_ids"]))
                if generate
                else CompletionFrames(tokenizer, rid, self.model)
            )
            last_write = time.monotonic()
            while not controller.stop.is_set():
                try:
                    events, finished, reason = await asyncio.to_thread(
                        controller.stream_events, rid, frames.cursor
                    )
                except ConnectionError:
                    events, finished, reason = [], False, None
                if not started and not events and not finished:
                    entry = await asyncio.to_thread(
                        controller.entry, rid, copy_payloads=False
                    )
                    if entry["blocked"] and entry["phase"] in {"SUBMIT", "SOURCE"}:
                        return await _error(send, 503, "Decode submission failed")
                output = list(frames.tokens(events))
                if finished:
                    if reason == "abort" and not started:
                        completed = True
                        return await _error(send, 503, "Decode generation aborted")
                    output.extend(frames.finish(reason))
                if output and not started:
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 200,
                            "headers": [
                                (b"content-type", b"text/event-stream"),
                                (b"cache-control", b"no-cache"),
                            ],
                        }
                    )
                    started = True
                for data in output:
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b"data: " + data + b"\n\n",
                            "more_body": True,
                        }
                    )
                    last_write = time.monotonic()
                if finished:
                    await send({"type": "http.response.body", "body": b""})
                    completed = True
                    return
                if started and time.monotonic() - last_write >= 0.5:
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b": keepalive\n\n",
                            "more_body": True,
                        }
                    )
                    last_write = time.monotonic()
                await asyncio.sleep(min(controller.poll_s, 0.05))
            if started:
                await send({"type": "http.response.body", "body": b""})
            else:
                await _error(send, 503, "migration controller is stopping")
        except (RuntimeError, ValueError):
            if started:
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'data: {"error":{"message":"Decode stream failed","type":"server_error"}}\n\n',
                    }
                )
            else:
                await _error(send, 503, "Decode stream failed")
        finally:
            if rid is None and not submit.done():
                # A disconnect can arrive while accept is fsyncing. Recover
                # its result before deciding whether there is a request to cancel.
                try:
                    rid = await submit
                except (RuntimeError, ValueError, OSError):
                    pass
            elif rid is None and not submit.cancelled() and submit.exception() is None:
                rid = submit.result()
            if rid is not None and not completed:
                await asyncio.to_thread(controller.cancel, rid)


def serve_native_pd(controller, model, host):
    import uvicorn

    with ExitStack() as stack:
        sockets = [
            stack.enter_context(socket.create_server((host, port)))
            for port in controller.listeners
        ]
        server = uvicorn.Server(
            uvicorn.Config(
                NativeDecodeApp(controller, model),
                lifespan="off",
                proxy_headers=False,
            )
        )
        server.run(sockets=sockets)
