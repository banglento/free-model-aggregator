#!/usr/bin/env python3
"""free-model-aggregator-01 entry: a routing gateway over simulated free tiers.

One long-running server speaks the OpenAI Chat Completions API and the
Anthropic Messages API to the trial client. Every client request is routed
to the strongest free model that meets the request's requirements, in that
provider's own wire (OpenAI, Anthropic or Gemini), translated losslessly;
when a provider refuses, the request fails over to the next candidate.
Run with --list it prints the provider catalog instead of serving.
"""

import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CATALOG_SCHEMA = "free-model-aggregator-01/catalog/v1"
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"
FIXED_CREATED = 1767225600
PIECE = 13  # stream piece size, mirroring the providers' own chunking


# ---------------------------------------------------------------- brief ----
def _extract_doc(text, schema):
    """Pull one embedded JSON document (identified by its schema) out of the
    brief text."""
    marker = '{"schema":"%s"' % schema
    at = text.find(marker)
    if at < 0:
        marker = '"schema": "%s"' % schema
        at = text.find(marker)
        if at < 0:
            return None
        at = text.rfind("{", 0, at)
    try:
        doc, _ = json.JSONDecoder().raw_decode(text[at:])
    except ValueError:
        return None
    return doc


def read_brief():
    path = os.environ.get("MAKEFUN_PROMPT_FILE")
    candidates = [path] if path else []
    candidates += ["prompt.txt", "/input/prompt.txt",
                   os.path.join(os.getcwd(), "prompt.txt")]
    text = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as fh:
                text = fh.read()
            break
    if text is None:
        raise SystemExit("no brief found (MAKEFUN_PROMPT_FILE unset)")
    providers = _extract_doc(text, PROVIDERS_SCHEMA)
    ratings = _extract_doc(text, RATINGS_SCHEMA)
    workload = _extract_doc(text, WORKLOAD_SCHEMA)
    if providers is None or ratings is None or workload is None:
        raise SystemExit("brief is missing one of its JSON documents")
    return providers, ratings, workload


def build_catalog(providers_doc, ratings_doc):
    """The catalog document: one entry per provider, models in brief order."""
    free_by_provider = {}
    for row in ratings_doc["ratings"]:
        if row.get("free") is not True:
            continue
        free_by_provider.setdefault(row["provider"], []).append({
            "id": row["model"],
            "free": True,
            "context_length": row["context_length"],
        })
    providers = []
    for prov in providers_doc["providers"]:
        providers.append({
            "id": prov["id"],
            "base_url": prov["path"],
            "key_env": prov["key_env"],
            "wire": prov["wire"],
            "models": free_by_provider.get(prov["id"], []),
        })
    return {"schema": CATALOG_SCHEMA, "providers": providers}


# ------------------------------------------------------------- http bits ----
def http_json(method, url, payload=None, headers=None, timeout=8):
    data = None
    hdrs = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["content-type"] = "application/json"
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            return err.code, json.loads(err.read().decode("utf-8"))
        except Exception:
            return err.code, {}
    except Exception:
        return 0, {}


def auth_headers(prov):
    key = prov.get("key") or (os.environ.get(prov.get("key_env") or "", "")
                              if prov.get("key_env") else "")
    auth = prov.get("auth")
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key}
    if auth in ("x-api-key", "x-goog-api-key"):
        return {auth: key}
    return {}


# ------------------------------------------------------- request shaping ----
def _text_of(content):
    """A message content (string or block list) as its plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" \
                    and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _blocks_of(content):
    """Neutral content blocks from either client's content shape."""
    if isinstance(content, str):
        return [{"kind": "text", "text": content}] if content else []
    blocks = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                blocks.append({"kind": "text", "text": block["text"]})
            elif kind == "image_url" and isinstance(block.get("image_url"), dict):
                url = block["image_url"].get("url")
                if isinstance(url, str):
                    blocks.append({"kind": "image_url", "url": url})
            elif kind == "image" and isinstance(block.get("source"), dict):
                src = block["source"]
                if src.get("type") == "base64":
                    blocks.append({"kind": "image",
                                   "media_type": src.get("media_type",
                                                         "image/png"),
                                   "data": src.get("data", "")})
                elif src.get("type") == "url" and isinstance(src.get("url"), str):
                    blocks.append({"kind": "image_url", "url": src["url"]})
    return blocks


def _string_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def neutral_from_openai(body):
    system_parts = []
    messages = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            text = _text_of(msg.get("content"))
            if text:
                system_parts.append(text)
        elif msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"],
                             "content": _blocks_of(msg.get("content"))})
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict) \
                and isinstance(tool["function"].get("name"), str):
            fn = tool["function"]
            tools.append({"name": fn["name"],
                          "description": fn.get("description"),
                          "parameters": fn.get("parameters")})
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = None
    fmt = body.get("response_format")
    json_mode = isinstance(fmt, dict) and fmt.get("type") in (
        "json_object", "json_schema")
    return {
        "system": "\n".join(system_parts),
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": _string_list(body.get("stop")),
        "json_mode": json_mode,
        "response_format": fmt if json_mode else None,
        "stream": body.get("stream") is True,
    }


def neutral_from_anthropic(body):
    messages = []
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"],
                             "content": _blocks_of(msg.get("content"))})
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            tools.append({"name": tool["name"],
                          "description": tool.get("description"),
                          "parameters": tool.get("input_schema")})
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = None
    return {
        "system": _text_of(body.get("system")),
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": _string_list(body.get("stop_sequences")),
        "json_mode": False,
        "response_format": None,
        "stream": body.get("stream") is True,
    }


# --------------------------------------------------------- wire rendering ---
def _data_uri(block):
    if block["kind"] == "image":
        return "data:%s;base64,%s" % (block["media_type"], block["data"])
    return block["url"]


def _split_data_uri(url):
    """(media_type, base64 data) for a data: URI, else None."""
    if not url.startswith("data:"):
        return None
    head, _, payload = url.partition(",")
    if ";base64" not in head:
        return None
    return head[5:].split(";")[0] or "application/octet-stream", payload


def render_openai(neutral, model):
    messages = []
    if neutral["system"]:
        messages.append({"role": "system", "content": neutral["system"]})
    for msg in neutral["messages"]:
        blocks = msg["content"]
        if len(blocks) == 1 and blocks[0]["kind"] == "text":
            content = blocks[0]["text"]
        else:
            content = []
            for block in blocks:
                if block["kind"] == "text":
                    content.append({"type": "text", "text": block["text"]})
                else:
                    content.append({"type": "image_url",
                                    "image_url": {"url": _data_uri(block)}})
        messages.append({"role": msg["role"], "content": content})
    out = {"model": model, "messages": messages}
    if neutral["tools"]:
        tools = []
        for tool in neutral["tools"]:
            fn = {"name": tool["name"]}
            if tool.get("description") is not None:
                fn["description"] = tool["description"]
            if tool.get("parameters") is not None:
                fn["parameters"] = tool["parameters"]
            tools.append({"type": "function", "function": fn})
        out["tools"] = tools
    if neutral["max_tokens"] is not None:
        out["max_tokens"] = neutral["max_tokens"]
    if neutral["stops"]:
        out["stop"] = neutral["stops"]
    if neutral["response_format"] is not None:
        out["response_format"] = neutral["response_format"]
    return out


def render_anthropic(neutral, model):
    messages = []
    for msg in neutral["messages"]:
        blocks = []
        for block in msg["content"]:
            if block["kind"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["kind"] == "image":
                blocks.append({"type": "image",
                               "source": {"type": "base64",
                                          "media_type": block["media_type"],
                                          "data": block["data"]}})
            else:
                blocks.append({"type": "image",
                               "source": {"type": "url",
                                          "url": block["url"]}})
        messages.append({"role": msg["role"], "content": blocks})
    out = {"model": model, "messages": messages}
    if neutral["system"]:
        out["system"] = neutral["system"]
    if neutral["tools"]:
        tools = []
        for tool in neutral["tools"]:
            item = {"name": tool["name"]}
            if tool.get("description") is not None:
                item["description"] = tool["description"]
            if tool.get("parameters") is not None:
                item["input_schema"] = tool["parameters"]
            tools.append(item)
        out["tools"] = tools
    if neutral["max_tokens"] is not None:
        out["max_tokens"] = neutral["max_tokens"]
    if neutral["stops"]:
        out["stop_sequences"] = neutral["stops"]
    return out


def render_gemini(neutral, model):
    contents = []
    for msg in neutral["messages"]:
        parts = []
        for block in msg["content"]:
            if block["kind"] == "text":
                parts.append({"text": block["text"]})
            elif block["kind"] == "image":
                parts.append({"inlineData": {"mimeType": block["media_type"],
                                             "data": block["data"]}})
            else:
                split = _split_data_uri(block["url"])
                if split is not None:
                    parts.append({"inlineData": {"mimeType": split[0],
                                                 "data": split[1]}})
                else:
                    parts.append({"fileData": {"fileUri": block["url"]}})
        contents.append({"role": "model" if msg["role"] == "assistant"
                          else "user", "parts": parts})
    out = {"contents": contents}
    if neutral["system"]:
        out["systemInstruction"] = {"parts": [{"text": neutral["system"]}]}
    if neutral["tools"]:
        decls = []
        for tool in neutral["tools"]:
            decl = {"name": tool["name"]}
            if tool.get("description") is not None:
                decl["description"] = tool["description"]
            if tool.get("parameters") is not None:
                decl["parameters"] = tool["parameters"]
            decls.append(decl)
        out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    if neutral["max_tokens"] is not None:
        config["maxOutputTokens"] = neutral["max_tokens"]
    if neutral["stops"]:
        config["stopSequences"] = neutral["stops"]
    if neutral["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def parse_answer(wire, doc):
    """(text, (input_tokens, output_tokens), stop_reason) from a provider's
    non-streaming answer."""
    if wire == "openai":
        choice = (doc.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        text = content if isinstance(content, str) else _text_of(content)
        usage = doc.get("usage") or {}
        return text, (usage.get("prompt_tokens"), usage.get("completion_tokens")), \
            choice.get("finish_reason")
    if wire == "anthropic":
        text = _text_of(doc.get("content"))
        usage = doc.get("usage") or {}
        return text, (usage.get("input_tokens"), usage.get("output_tokens")), \
            doc.get("stop_reason")
    candidate = (doc.get("candidates") or [{}])[0]
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    # The answer text is the parts concatenated in order, nothing added.
    text = "".join(p["text"] for p in parts
                   if isinstance(p, dict) and isinstance(p.get("text"), str))
    usage = doc.get("usageMetadata") or {}
    return text, (usage.get("promptTokenCount"),
                  usage.get("candidatesTokenCount")), candidate.get("finishReason")


# -------------------------------------------------------- client answers ----
_STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
_OPENAI_STOPS = ["stop", "length", "tool_calls", "content_filter"]
_ANTHROPIC_STOPS = ["end_turn", "max_tokens", "tool_use", "refusal"]


def map_stop(reason, client_api):
    family = _STOP_FAMILIES.get(reason, 0)
    return (_OPENAI_STOPS if client_api == "openai"
            else _ANTHROPIC_STOPS)[family]


def _pieces(text):
    out = [text[i:i + PIECE] for i in range(0, len(text), PIECE)]
    return out or [""]


def _sse(event, data):
    head = "event: %s\n" % event if event else ""
    return "%sdata: %s\n\n" % (head, json.dumps(data))


def render_client_openai(model, text, usage, stop, stream):
    tokens = {"prompt_tokens": usage[0], "completion_tokens": usage[1],
              "total_tokens": (usage[0] or 0) + (usage[1] or 0)}
    if not stream:
        return "application/json", json.dumps({
            "id": "chatcmpl-agg", "object": "chat.completion",
            "created": FIXED_CREATED, "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": stop, "logprobs": None}],
            "usage": tokens}).encode("utf-8")
    def chunk(delta, finish):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}]}
    frames = [_sse(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in _pieces(text):
        frames.append(_sse(None, chunk({"content": piece}, None)))
    frames.append(_sse(None, chunk({}, stop)))
    frames.append(_sse(None, {"id": "chatcmpl-agg",
                              "object": "chat.completion.chunk",
                              "created": FIXED_CREATED, "model": model,
                              "choices": [], "usage": tokens}))
    frames.append("data: [DONE]\n\n")
    return "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


def render_client_anthropic(model, text, usage, stop, stream):
    if not stream:
        return "application/json", json.dumps({
            "id": "msg_agg", "type": "message", "role": "assistant",
            "model": model, "content": [{"type": "text", "text": text}],
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": usage[0],
                      "output_tokens": usage[1]}}).encode("utf-8")
    frames = [_sse("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage[0], "output_tokens": 0}}})]
    frames.append(_sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in _pieces(text):
        frames.append(_sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(_sse("content_block_stop", {
        "type": "content_block_stop", "index": 0}))
    frames.append(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": usage[1]}}))
    frames.append(_sse("message_stop", {"type": "message_stop"}))
    return "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


# -------------------------------------------------------------- gateway -----
class Gateway:
    def __init__(self, env_url, providers_doc, ratings_doc, workload_doc,
                 catalog):
        self.env_url = env_url.rstrip("/")
        self.providers = {p["id"]: p for p in providers_doc["providers"]}
        self.rows = [r for r in ratings_doc["ratings"] if r.get("free") is True]
        self.manifest = {m["id"]: m for m in workload_doc["manifest"]}
        self.order = [m["id"] for m in workload_doc["manifest"]]
        self.catalog = catalog
        self.strikes = {}
        self.served_count = 0
        self.lock = threading.Lock()

    # -- model choice -------------------------------------------------------
    def candidates(self, requires, json_body):
        def meets(row):
            if requires.get("tools") is True and not row["tools"]:
                return False
            if requires.get("vision") is True and not row["vision"]:
                return False
            if requires.get("json") is True and not row["json"]:
                return False
            if requires.get("min_context") is not None \
                    and row["context_length"] < requires["min_context"]:
                return False
            return True
        pool = []
        for row in self.rows:
            if not meets(row):
                continue
            wire = self.providers[row["provider"]]["wire"]
            if json_body and wire == "anthropic":
                continue  # the anthropic wire cannot carry a JSON-mode echo
            pool.append(row)
        pool.sort(key=lambda r: (1 if self.strikes.get(r["provider"], 0) >= 2
                                 else 0,
                                 -r["rating_milli"], r["provider"], r["model"]))
        return pool

    def current_requirements(self):
        status, doc = http_json("GET", self.env_url + "/work", timeout=3)
        if status == 200 and doc.get("done") is False \
                and doc.get("next") in self.manifest:
            return self.manifest[doc["next"]].get("requires") or {}
        if self.served_count < len(self.order):
            entry = self.manifest[self.order[self.served_count]]
            return entry.get("requires") or {}
        return {}

    # -- one provider attempt -------------------------------------------------
    def attempt(self, prov, model, neutral, client_headers):
        wire = prov["wire"]
        base = self.env_url + prov["path"]
        headers = auth_headers(prov)
        if wire == "openai":
            url = base + "/v1/chat/completions"
            payload = render_openai(neutral, model)
        elif wire == "anthropic":
            url = base + "/v1/messages"
            payload = render_anthropic(neutral, model)
            headers["anthropic-version"] = client_headers.get(
                "anthropic-version", "2023-06-01")
        else:
            url = "%s/v1beta/models/%s:generateContent" % (base, model)
            payload = render_gemini(neutral, model)
        status, doc = http_json("POST", url, payload, headers, timeout=10)
        if status < 200 or status >= 300:
            return None
        text, usage, stop = parse_answer(wire, doc)
        if usage[0] is None or usage[1] is None:
            return None
        return text, usage, stop

    # -- one client request ---------------------------------------------------
    def serve(self, client_api, body, client_headers):
        neutral = (neutral_from_openai(body) if client_api == "openai"
                   else neutral_from_anthropic(body))
        with self.lock:
            requires = self.current_requirements()
            pool = self.candidates(requires, neutral["json_mode"])
            refused_here = set()
            result = None
            for row in pool:
                if row["provider"] in refused_here:
                    continue
                got = self.attempt(self.providers[row["provider"]],
                                   row["model"], neutral, client_headers)
                if got is None:
                    refused_here.add(row["provider"])
                    self.strikes[row["provider"]] = \
                        self.strikes.get(row["provider"], 0) + 1
                    continue
                result = (row["model"], got)
                break
            if result is None:  # last resort: retry refused providers too
                for row in pool:
                    got = self.attempt(self.providers[row["provider"]],
                                       row["model"], neutral, client_headers)
                    if got is not None:
                        result = (row["model"], got)
                        break
            self.served_count += 1
        if result is None:
            return 502, "application/json", json.dumps({
                "error": {"message": "every free provider refused",
                          "type": "api_error", "code": "no_provider"}}).encode()
        model, (text, usage, stop) = result
        mapped = map_stop(stop, client_api)
        if client_api == "openai":
            ctype, payload = render_client_openai(
                model, text, usage, mapped, neutral["stream"])
        else:
            ctype, payload = render_client_anthropic(
                model, text, usage, mapped, neutral["stream"])
        return 200, ctype, payload

    def models_route(self, anthropic_shape):
        seen = []
        for prov in self.catalog["providers"]:
            for model in prov["models"]:
                if model["id"] not in seen:
                    seen.append(model["id"])
        if anthropic_shape:
            doc = {"data": [{"type": "model", "id": mid, "display_name": mid,
                             "created_at": "2026-01-01T00:00:00Z"}
                            for mid in seen],
                   "has_more": False}
        else:
            doc = {"object": "list",
                   "data": [{"id": mid, "object": "model",
                             "created": FIXED_CREATED, "owned_by": "aggregator"}
                            for mid in seen]}
        return 200, "application/json", json.dumps(doc).encode("utf-8")


def make_handler(gateway):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _reply(self, status, ctype, payload):
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                anthropic = self.headers.get("x-api-key") is not None
                status, ctype, payload = gateway.models_route(anthropic)
                self._reply(status, ctype, payload)
            else:
                self._reply(404, "application/json",
                            b'{"error":{"message":"not found"}}')

        def do_POST(self):
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                self._reply(400, "application/json",
                            b'{"error":{"message":"body is not JSON"}}')
                return
            headers = {k.lower(): v for k, v in self.headers.items()}
            if self.path.startswith("/v1/chat/completions"):
                status, ctype, payload = gateway.serve("openai", body, headers)
            elif self.path.startswith("/v1/messages"):
                status, ctype, payload = gateway.serve("anthropic", body,
                                                       headers)
            else:
                status, ctype, payload = 404, "application/json", \
                    b'{"error":{"message":"not found"}}'
            self._reply(status, ctype, payload)

    return Handler


def main(argv):
    providers_doc, ratings_doc, workload_doc = read_brief()
    catalog = build_catalog(providers_doc, ratings_doc)
    if "--list" in argv:
        sys.stdout.write(json.dumps(catalog) + "\n")
        return 0
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if not env_url:
        raise SystemExit("MAKEFUN_ENV_URL is not set")
    gateway = Gateway(env_url, providers_doc, ratings_doc, workload_doc,
                      catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(gateway))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    registered = False
    for _ in range(60):
        status, doc = http_json("POST", env_url.rstrip("/") + "/register",
                                {"protocol_version": 1, "port": port,
                                 "catalog": catalog}, timeout=3)
        if status == 200 and doc.get("ok") is True:
            registered = True
            break
        time.sleep(0.5)
    if not registered:
        server.shutdown()
        raise SystemExit("the env refused registration")
    deadline = time.time() + 165
    while time.time() < deadline:
        status, doc = http_json("GET", env_url.rstrip("/") + "/work",
                                timeout=3)
        if status == 200 and doc.get("done") is True:
            break
        time.sleep(0.2)
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
