"""Real P4 CLI process for SIGKILL recovery tests; tokenizer/backend ABI fixtures.

The parent owns live native D2D fixtures. This child executes the unmodified P4
main/Handler/DSV coordinator and forwards scheduler commands over loopback HTTP.
"""

import argparse
import http.client
import importlib.util
import json
import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(
        HANDOFF_STATE_DIR=str(args.state),
        HANDOFF_MODEL="fixture-tokenizer",
        HANDOFF_PD_PREFILL_URL="http://127.0.0.1:30000/generate",
    )
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = NS(
        from_pretrained=lambda *a, **kw: NS(encode=lambda value: [1, 2, 3])
    )
    sys.modules["transformers"] = transformers
    path = args.source_root / "plugins/remote_attn/control_plane/p4_gateway.py"
    spec = importlib.util.spec_from_file_location("p4_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Driver:
        @staticmethod
        def send(role, payload, timeout=30):
            conn = http.client.HTTPConnection(
                "127.0.0.1", args.proxy_port, timeout=timeout
            )
            try:
                conn.request(
                    "POST",
                    "/" + role,
                    json.dumps(payload),
                    {"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                result = json.loads(response.read())
                if response.status != 200:
                    error = (
                        TimeoutError
                        if result.get("type") == "TimeoutError"
                        else RuntimeError
                    )
                    raise error(result["error"])
                return result
            finally:
                conn.close()

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            pending = args.ready.with_suffix(".tmp")
            pending.write_text(json.dumps(dict(pid=os.getpid(), port=self.server_port)))
            os.replace(pending, args.ready)

    module.d = Driver
    module.ThreadingHTTPServer = Server
    sys.argv = [str(path), "--backend", "dsv", "--port", "0", "--poll-ms", "10"]
    module.main()


if __name__ == "__main__":
    main()
