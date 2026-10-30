import json
import os
import re
import time

from .base import LLM, Schema

_MAX_RETRIES = 6
_RETRY_BASE_DELAY = 5

# Find the first balanced {...} block (handles nested braces); used to peel
# the JSON object out of the model's free-form json_object response.
_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}")


class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        # The OpenAI SDK does not read OPENAI_BASE_URL on its own, so an
        # OpenAI-compatible endpoint (DeepSeek, MiMo, a local proxy) is only
        # reachable if the base URL is passed explicitly. Unset -> api.openai.com.
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        # A request timeout is not optional: without one a single hung
        # connection blocks the whole run, because the client waits forever
        # and the harness has no way to move on. A timed-out query is
        # retried by the caller, and if it keeps failing the query is
        # scored as wrong rather than the run hanging.
        timeout = float(os.environ.get("OMB_LLM_TIMEOUT", "180"))
        self._client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=int(os.environ.get("OMB_LLM_RETRIES", "2")),
        )
        self._model = model

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        schema_json = {
            "type": "object",
            "properties": schema.properties,
            "required": schema.required,
            "additionalProperties": False,
        }
        # OpenAI supports strict json_schema; DeepSeek and most local OpenAI
        # clones only support json_object. Try the strict schema first and fall
        # back to json_object (extract the first {...} block) when the backend
        # rejects response_format.
        #
        # MiniMax accepts response_format=json_schema without error but ignores
        # it and answers in prose, so the strict attempt would fail at json.loads
        # instead of falling back. OMB_JSON_MODE=json_object pins the wire
        # format for such endpoints.
        formats = [
            {"type": "json_schema",
             "json_schema": {"name": "response", "schema": schema_json, "strict": True}},
            {"type": "json_object"},
        ]
        if os.environ.get("OMB_JSON_MODE", "").lower() == "json_object":
            formats = [formats[1]]
        delay = _RETRY_BASE_DELAY
        last_exc = None
        for fmt in formats:
            for attempt in range(_MAX_RETRIES):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    # DeepSeek (and others) require the literal token 'json' in
                    # the prompt before they honour response_format=json_object.
                    if fmt["type"] == "json_object":
                        messages = [
                            {"role": "system", "content": "Return a single json object."},
                            {"role": "user", "content": prompt},
                        ]
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        response_format=fmt,
                    )
                    text = (response.choices[0].message.content or "").strip()
                    # DeepSeek-flash occasionally returns empty content on long
                    # prompts; retry with a fresh request rather than reporting
                    # an empty JSON object upstream.
                    if not text:
                        raise RuntimeError("empty content from LLM (silent-fail)")
                    if fmt["type"] == "json_object":
                        match = _JSON_OBJECT_RE.search(text)
                        if match:
                            text = match.group(0)
                        else:
                            raise RuntimeError("no JSON object found in LLM output")
                    return json.loads(text)
                except Exception as e:
                    last_exc = e
                    msg = str(e)
                    if fmt["type"] == "json_schema" and "response_format" in msg.lower():
                        # This backend doesn't support json_schema; try the next format.
                        break
                    if "429" in msg or "rate" in msg.lower():
                        if attempt < _MAX_RETRIES - 1:
                            time.sleep(delay)
                            delay *= 2
                            continue
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    raise
        raise RuntimeError(f"OpenAI request failed after retries: {last_exc}")