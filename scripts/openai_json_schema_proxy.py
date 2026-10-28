"""Tiny OpenAI-compatible proxy that downgrades json_schema requests to json_object.

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

Run:
    python scripts/openai_json_schema_proxy.py --port 8088 \
        --upstream https://api.deepseek.com/v1
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
    model = None

    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        path = self.path
        if not path.endswith("/chat/completions"):
            self._reply(404, {"error": "only chat/completions is proxied"})
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

        payload = json.dumps(req).encode("utf-8")
        url = self.upstream.rstrip("/") + path[len("/v1"):] if path.startswith("/v1") else self.upstream.rstrip("/") + path
        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
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
    args = ap.parse_args()
    _Handler.upstream = args.upstream
    srv = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[json_schema_proxy] {args.host}:{args.port} -> {args.upstream}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
