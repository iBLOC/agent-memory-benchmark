"""Tiny OpenAI-compatible proxy that downgrades json_schema requests to json_object
and can split embeddings onto a different upstream.

Some OpenAI-compatible endpoints (DeepSeek) reject `response_format:
{"type": "json_schema"}` with HTTP 400 "This response_format type is
unavailable now". graphrag's community-report step asks for a json_schema, so
its pipeline dies mid-run.

This proxy sits in front of the real endpoint and rewrites such requests:
the schema is appended to the prompt (which also puts the literal token
"json" in the message, required by DeepSeek's json_object mode) and the
request is sent with `response_format: {"type": "json_object"}`. Everything
else — including the request body graphrag hashes for its cache — is passed
through untouched, so cached responses stay valid.

With `--embeddings-upstream`, POST /v1/embeddings is forwarded elsewhere. That
is what an EdgeQuake provider needs: chat on DeepSeek or MiniMax, embeddings
on the local MiniLM server, behind one OpenAI-shaped base URL.

Run:
    python scripts/openai_json_schema_proxy.py --port 8088 \
        --upstream https://api.deepseek.com/v1
    python scripts/openai_json_schema_proxy.py --port 8088 \
        --upstream https://api.minimaxi.com/v1 \
        --embeddings-upstream http://127.0.0.1:8086/v1
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCHEMA_HINT = (
    "\n\nRespond with a single json object that matches this json schema "
    "(no prose, no markdown fence):\n"
)


class _Handler(BaseHTTPRequestHandler):
    upstream = "https://api.deepseek.com/v1"
    embeddings_upstream: str | None = None
    #: Model ids to advertise on GET /v1/models. Clients that discover the
    #: available models from the endpoint (EdgeQuake does) see nothing when
    #: this list is empty.
    advertised_models: list[str] = []
    #: "from=to" model-name rewrites applied on the way out. A client that
    #: only knows one model name (LM Studio's "default") can be pointed at a
    #: different upstream model this way.
    model_rewrites: dict[str, str] = {}
    #: Authorization header to use for the chat upstream. Clients that speak a
    #: local-server shape (LM Studio) send no key at all, so the proxy
    #: supplies the upstream one.
    upstream_key: str | None = None
    model = None

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._reply(200, {
                "object": "list",
                "data": [{"id": m, "object": "model", "owned_by": "proxy"}
                          for m in self.advertised_models],
            })
            return
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        path = self.path
        if path.endswith("/embeddings") and self.embeddings_upstream:
            self._forward(self.embeddings_upstream, path)
            return
        if not path.endswith("/chat/completions"):
            self._reply(404, {"error": "only chat/completions and embeddings are proxied"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n).decode("utf-8") if n else ""
            req = json.loads(body) if body else {}
        except Exception as e:
            self._reply(400, {"error": f"bad json: {e}"})
            return

        fmt = req.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            schema = (fmt.get("json_schema") or {}).get("schema", fmt.get("json_schema"))
            req["response_format"] = {"type": "json_object"}
            prompt = SCHEMA_HINT + json.dumps(schema, ensure_ascii=False)
            messages = req.get("messages") or []
            if messages:
                last = messages[-1]
                if isinstance(last.get("content"), str):
                    last["content"] = last["content"] + prompt
                else:
                    last["content"] = prompt
            else:
                req["messages"] = [{"role": "user", "content": prompt}]

        src = str(req.get("model") or "")
        dst = self.model_rewrites.get(src)
        if dst:
            req["model"] = dst
        self._forward(self.upstream, path, req)

    def _forward(self, upstream: str, path: str, body: dict | None = None) -> None:
        """Forward a request upstream verbatim (or with a rewritten body)."""
        if body is None:
            n = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(n) if n else b"{}"
            payload = raw_body
        else:
            payload = json.dumps(body).encode("utf-8")
        suffix = path[len("/v1"):] if path.startswith("/v1") else path
        url = upstream.rstrip("/") + suffix
        headers = {"Content-Type": "application/json"}
        auth = self.upstream_key or self.headers.get("Authorization")
        if auth:
            if self.upstream_key and not auth.lower().startswith("bearer "):
                auth = f"Bearer {auth}"
            headers["Authorization"] = auth
        try:
            upstream_req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(upstream_req, timeout=600) as resp:
                raw = resp.read()
            self._reply(200, json.loads(raw))
        except urllib.error.HTTPError as e:
            self._reply(e.code, {"error": e.read().decode("utf-8", "replace")[:800]})
        except Exception as e:
            self._reply(502, {"error": f"{type(e).__name__}: {e}"})

    def _reply(self, code, obj):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def main():
    ap = argparse.ArgumentParser(description="json_schema -> json_object proxy for OpenAI-compatible endpoints.")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--upstream", default="https://api.deepseek.com/v1")
    ap.add_argument(
        "--embeddings-upstream",
        default=None,
        help="route POST /v1/embeddings here instead of --upstream",
    )
    ap.add_argument(
        "--upstream-key",
        default=None,
        help="Authorization header to send to the chat upstream",
    )
    ap.add_argument(
        "--rewrite-model",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="rewrite the model name on the way out (repeatable)",
    )
    ap.add_argument(
        "--advertise-model",
        action="append",
        default=[],
        help="model id to list on GET /v1/models (repeatable)",
    )
    args = ap.parse_args()
    _Handler.upstream = args.upstream
    _Handler.embeddings_upstream = args.embeddings_upstream
    _Handler.advertised_models = args.advertise_model
    _Handler.upstream_key = args.upstream_key
    _Handler.model_rewrites = dict(
        pair.split("=", 1) for pair in args.rewrite_model if "=" in pair
    )
    srv = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[json_schema_proxy] {args.host}:{args.port} -> {args.upstream}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
