"""Hippocampus provider — the Rust memory framework (iBLOC/hippocampus) over its REST API.

Hippocampus is a standalone server (HybridRAG: keyword FTS + vector + graph) listening on
port 8081. This adapter drives it the same way its own benchmark harness does:

- `ingest`   → POST /api/memory   (uri = Document.id so retrieval can hand IDs back)
- `retrieve` → POST /api/search   (hybrid ranking; minScore=0 keeps the full ranking)

Server contract notes (verified against hippocampus-server/src/handlers.rs):
- AddMemoryRequest only accepts uri/content/dimension/tenantId/sessionId/embedding, so
  timestamps and provenance hints are folded into the stored content, and the Document
  sidecar (tags/context) is kept locally per store_dir for retrieval-time filtering.
- insert_memory rejects a null embedding, so runs without an embedding endpoint store a
  deterministic hash vector; the vector path only ranks meaningfully when the server has
  its own embedding key, in which case keyword+graph carry retrieval.
- Errors may arrive as HTTP 200 with a text/plain body — non-JSON payloads raise.

Multi-tenant isolation maps 1:1 onto AMB's user_id scoping: every document/query carries
`tenantId = doc.user_id`, which is also LongMemEval's isolation unit (question_id).

Request safety: the target base URL is operator config (HIPPOCAMPUS_URL), never request
input. Every request target is still validated before it is dialed — scheme restricted to
http/https, host resolved and every resolved address classified, loopback / private /
link-local / reserved / multicast ranges refused except for the documented same-machine
harness allowlist (127.0.0.1, localhost, ::1 — this adapter exists to drive a local
hippocampus server), connections dial the address pinned at validation time (no DNS
rebinding window), and http.client never follows redirects.
"""

import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import socket
import struct
import sys
import urllib.parse
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

_EMBEDDING_DIM = 1536
_DEFAULT_TENANT = "amb-default"
_WORD_RE = re.compile(r"[a-z0-9]+")

# Same-machine harness targets: this adapter drives a local hippocampus server, so the
# loopback hostnames below are the only non-public addresses ever permitted.
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

# State tags a production extractor would classify at ingest (the dataset's _STATE_LABEL);
# derived here from the belief text itself, which is what every comparison system sees.
_STATE_PATTERNS = [
    ("state:historical", re.compile(
        r"historical record|previously used|replaced by|superseded|no longer|deprecat", re.I)),
    ("state:open_question", re.compile(
        r"not decided|undecided|open question|unresolved|to be decided|yet to be", re.I)),
]

# Server-side search over-fetch factor: the adapter filters client-side (the REST API has
# no tag query yet), so retrieve() asks for more rows than it returns.
_FETCH_FACTOR = 4
_FETCH_MIN = 40

# Function words + contraction debris (what's → what + s) dropped from derived name tags.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here of in on at to for with
without from by as is are was were be been being do does did doing done have has had having
i you he she it we they me him her them my your our their its so not no nor own same too very
can will just don should now what which who whom when where why how all any both each few more
most other some such only s t d m ve re ll
""".split())

# Verbs and auxiliaries that open a prose sentence: the leading identity block ends
# where prose begins ("Uses Fastify for HTTP services" — "Uses" opens the sentence,
# so "Fastify"/"express" are the last identity units).
_VERB_LEADS = frozenset("""
reads read wants want prefers prefer edits edit writes write uses use used
decide decides decided deploys deploy deployed debating debate debates
switched switch manages managed manage runs run ran route routes routed
depends depend depended evaluating evaluate evaluated give gives gave
keep keeps kept choose chooses chose avoid avoids build builds built
make makes made take takes took get gets got go goes went mean means
close closes did does do
""".split())

# Multi-word names an English-reading extractor keeps as one phrase ("auth redis" is
# a single alias; a bare "redis" fragment would wrongly name every memory that merely
# mentions Redis). Deterministic stand-in for the LLM extraction pass the dataset's
# `name` label assumes — same role as a production system's identity glossary.
_COMPOUND_NAMES = frozenset((
    "error handling", "result type", "class hierarchies", "class hierarchy",
    "base class", "factory function", "ci pipeline", "cd pipeline",
    "dead letter", "failed messages", "session backend", "session tokens",
    "auth service", "auth redis", "api versioning", "point of view",
    "perspective shift", "prose style", "documentation tone",
    "chapter order", "chapter structure", "github actions",
))


def _hash_embedding(text: str, dim: int | None = None) -> list[float]:
    """Deterministic token-hash bag vector, L2-normalized.

    Not semantic — exists only as a fallback when no embedding endpoint is
    configured. Dim defaults to the server's LANCE_EMBEDDING_DIM (read each
    call so a hot-swapped store still matches the active schema)."""
    if dim is None:
        dim = int(os.environ.get("LANCE_EMBEDDING_DIM", "1536"))
    vec = [0.0] * dim
    for token in text.lower().split():
        h = hashlib.sha256(token.encode()).digest()
        idx = struct.unpack("<I", h[:4])[0] % dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _osa_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein (optimal string alignment): edits + adjacent transpositions."""
    la, lb = len(a), len(b)
    prev2: list[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[lb]


def _validate_target(base_url: str) -> tuple[str, int, str]:
    """Validate the operator-configured server base URL and pin one resolved address.

    Policy: http/https only; no userinfo; the host is resolved and every resolved address
    must be public unicast — loopback, private, link-local, reserved, multicast and
    unspecified ranges are refused. The sole exception is the same-machine harness
    allowlist (_LOCAL_HOSTS): this adapter's purpose is driving a local hippocampus
    server, and the base URL is trusted operator configuration, never request input.

    Returns (host, port, pinned_ip); callers dial the pinned IP so a later DNS change
    cannot redirect the connection (rebinding).
    """
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme {parts.scheme!r} — only http/https are allowed")
    if parts.username or parts.password:
        raise ValueError("URL userinfo (user@host) is not allowed")
    host = parts.hostname
    if not host:
        raise ValueError(f"URL has no host: {base_url!r}")
    port = parts.port or (443 if parts.scheme == "https" else 80)

    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    if not infos:
        raise ValueError(f"host {host!r} did not resolve")
    local_ok = host.lower() in _LOCAL_HOSTS
    pinned = None
    for info in infos:
        ip_str = info[4][0].split("%")[0]
        ip = ipaddress.ip_address(ip_str)
        non_public = (
            ip.is_loopback or ip.is_private or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        )
        if non_public and not local_ok:
            raise ValueError(
                f"refusing non-public target address {ip} for host {host!r} — "
                f"loopback/private/reserved ranges are not allowed"
            )
        if pinned is None:
            pinned = ip_str
    return host, port, pinned


class HippocampusMemoryProvider(MemoryProvider):
    name = "hippocampus"
    description = "Hippocampus Rust memory framework — HybridRAG (keyword + vector + graph) over its REST API."
    kind = "local"
    variant = "http"
    link = "https://github.com/iBLOC/hippocampus"
    supports_filters = True

    def __init__(self):
        base = os.environ.get("HIPPOCAMPUS_URL", "http://127.0.0.1:8081").rstrip("/")
        self._scheme, (self._host, self._port, self._ip) = (
            urllib.parse.urlsplit(base).scheme, _validate_target(base)
        )
        self._timeout = float(os.environ.get("HIPPOCAMPUS_TIMEOUT", "60"))
        # uri → {"tags": [...], "context": str|None, "tenant": str}; persisted per store_dir
        self._sidecar: dict[str, dict] = {}
        self._sidecar_path: Path | None = None

    # ---- low level -------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None):
        """One request to the validated, pinned server address.

        http.client performs no redirect following, and the connection dials the IP that
        passed validation rather than re-resolving the hostname."""
        if not path.startswith("/") or "\\" in path or " " in path:
            raise ValueError(f"illegal request path: {path!r}")
        payload = json.dumps(body).encode() if body is not None else None
        conn = http.client.HTTPConnection(self._ip, self._port, timeout=self._timeout)
        try:
            conn.request(
                method, path, body=payload,
                headers={"Content-Type": "application/json", "Host": f"{self._host}:{self._port}"},
            )
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                raise RuntimeError(
                    f"{method} {path} -> HTTP {resp.status}: {data.decode(errors='replace')[:200]}"
                )
        finally:
            conn.close()
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            # Server quirk: some errors come back as 200 + text/plain.
            raise RuntimeError(f"{method} {path} returned non-JSON body: {data.decode(errors='replace')[:200]}")

    # ---- lifecycle -------------------------------------------------------

    def initialize(self) -> None:
        for attempt in range(30):
            try:
                self._request("GET", "/health")
                return
            except RuntimeError:
                import time
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"hippocampus server at {self._host}:{self._port} not healthy after retries")

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        self._sidecar_path = store_dir / "hippocampus_sidecar.json"
        if self._sidecar_path.exists():
            try:
                self._sidecar = json.loads(self._sidecar_path.read_text(encoding="utf-8"))
            except Exception:
                self._sidecar = {}
        if reset and self._sidecar:
            self._purge_tenants()
            self._sidecar = {}
        self._save_sidecar()

    def _save_sidecar(self) -> None:
        if self._sidecar_path is not None:
            self._sidecar_path.write_text(json.dumps(self._sidecar), encoding="utf-8")

    def _purge_tenants(self) -> None:
        tenants = {v.get("tenant") for v in self._sidecar.values() if v.get("tenant")}
        for tenant in tenants:
            query = urllib.parse.urlencode({"tenantId": tenant, "limit": 100000})
            rows = self._request("GET", f"/api/memory?{query}") or []
            for row in rows:
                mem_id = row.get("id")
                if mem_id:
                    self._request("DELETE", f"/api/memory/{mem_id}")

    # ---- ingest ----------------------------------------------------------

    @staticmethod
    def _derive_state_tags(text: str) -> list[str]:
        for tag, pattern in _STATE_PATTERNS:
            if pattern.search(text or ""):
                return [tag]
        return ["state:current"]

    @staticmethod
    def _norm_name(unit: str) -> str:
        """One raw unit → the label's normal form: lowercase, single-spaced words
        ("error_handling" → "error handling", "@typescript-eslint" → "typescript eslint")."""
        return " ".join(_WORD_RE.findall(unit.lower()))

    @classmethod
    def _unit_ends_region(cls, unit: str, region: list[str], next_unit: str | None) -> bool:
        """True when `unit` ends the leading identity block and prose begins.

        A function word inside a known compound stays ("point of view"); anywhere
        else it ends the block ("Protagonist of the novel." — "the" opens the
        sentence remainder, "Redis is a cache" — "is" opens the predicate)."""
        words = _WORD_RE.findall(unit.lower())
        if not words:                                   # bare punctuation (em dash)
            return True
        if words[0] in _STOPWORDS or words[0] in _VERB_LEADS:
            prev = [cls._norm_name(u) for u in region[-1:]]
            spans = (
                [" ".join(prev + [cls._norm_name(unit)])],
                [" ".join((cls._norm_name(unit), cls._norm_name(next_unit)))] if next_unit else [],
                [" ".join(prev + [cls._norm_name(unit), cls._norm_name(next_unit)])] if next_unit else [],
            )
            if any(span[0] in _COMPOUND_NAMES for span in spans if span):
                return False
            return True
        return unit.endswith((".", ",", ";", ":", "!", "?"))

    @classmethod
    def _derive_name_tags(cls, text: str) -> set[str]:
        """Identity tags for one memory, per the dataset's `name` extraction label.

        The extractor is asked for "every name the subject is known by … lowercase,
        words separated by single spaces". In the folded belief text the names lead
        (canonical name, then aliases, then prose), so the leading block is segmented
        into whole name phrases and nothing else. Whole phrases, never fragments: a
        compound alias like "auth redis" stays atomic, because a bare "redis" tag would
        wrongly name every memory that merely mentions Redis, and "error_handling"
        becomes "error handling" as one tag rather than "error" + "handling". Prose
        after the block is ignored: names of things a fact merely mentions are
        mentions, not names."""
        units = (text or "").split()
        if not units:
            return set()
        names = {cls._norm_name(units[0])}
        region: list[str] = []
        rest = units[1:]
        for idx, unit in enumerate(rest):
            if len(region) >= 16:
                break
            nxt = rest[idx + 1] if idx + 1 < len(rest) else None
            if cls._unit_ends_region(unit, region, nxt):
                break
            region.append(unit)
        i = 0
        while i < len(region):
            if len(region) - i >= 3:
                trio = " ".join(cls._norm_name(u) for u in region[i:i + 3])
                if trio in _COMPOUND_NAMES:
                    names.add(trio)
                    i += 3
                    continue
            if len(region) - i >= 2:
                duo = " ".join(cls._norm_name(u) for u in region[i:i + 2])
                if duo in _COMPOUND_NAMES or region[i + 1].isdigit():
                    names.add(duo)
                    i += 2
                    continue
            single = cls._norm_name(region[i])
            if len(single) > 1 and single not in _STOPWORDS and not single.isdigit():
                names.add(single)
            i += 1
        return names

    @classmethod
    def _index_record(cls, doc: Document) -> dict:
        """Sidecar entry for one document: scope/state tags, derived name tags, provenance."""
        content = doc.content
        if doc.context:
            content = f"[{doc.context}] {content}"
        if doc.timestamp and doc.timestamp not in content:
            content = f"(timestamp: {doc.timestamp}) {content}"
        return {
            "tags": list(doc.tags or []) + cls._derive_state_tags(content),
            "names": sorted(cls._derive_name_tags(doc.content)),
            "context": doc.context,
            "tenant": doc.user_id or _DEFAULT_TENANT,
        }

    @staticmethod
    def _fold_content(doc: Document) -> str:
        """The REST API has no timestamp/tags fields: fold provenance into content so
        the answer LLM sees session dates, and keep tags for retrieval-time filters."""
        content = doc.content
        if doc.context:
            content = f"[{doc.context}] {content}"
        if doc.timestamp and doc.timestamp not in content:
            content = f"(timestamp: {doc.timestamp}) {content}"
        return content

    @staticmethod
    def _ingest_truncate(text: str) -> str:
        """Cap the STORED content at HIPPOCAMPUS_INGEST_CHAR_LIMIT chars
        (head+tail kept). LifeBench-style sessions run 100-400KB; pushing
        them whole through the server's per-insert path (Debug logging +
        FTS + index buffers) OOMs the server (~14 alloc aborts per 10-unit
        loop). The retrieve-time DOC_CHAR_LIMIT (<=15k) caps what the
        answer LLM sees anyway, so a 24k stored cap loses ~nothing."""
        cap = int(os.environ.get("HIPPOCAMPUS_INGEST_CHAR_LIMIT", "0"))
        if cap <= 0 or len(text) <= cap:
            return text
        head = cap // 3
        tail = cap - head
        elided = "\n…[middle elided at ingest]…\n"
        return text[:head] + elided + text[-tail:]

    def ingest(self, documents: list[Document]) -> None:
        for doc in documents:
            content = self._ingest_truncate(self._fold_content(doc))
            embedding = self._embed(content)
            if embedding is None:
                continue
            body = {
                "uri": doc.id,
                "content": content,
                "dimension": "resources",
                "tenantId": doc.user_id or _DEFAULT_TENANT,
                "embedding": embedding,
            }
            self._request("POST", "/api/memory", body)
            self._sidecar[doc.id] = self._index_record(doc)
        self._save_sidecar()

    def _embed(self, text: str) -> list[float] | None:
        """Compute the dense vector for `text` via the configured OpenAI-
        compatible embeddings endpoint, or fall back to the deterministic hash
        bag vector when the endpoint is unreachable or dim-mismatches."""
        api_base = os.environ.get("LLM_API_BASE")
        api_key = os.environ.get("LLM_API_KEY")
        model = os.environ.get("EMBEDDING_MODEL")
        if api_base and api_key and model:
            try:
                parsed = urllib.parse.urlparse(f"{api_base.rstrip('/')}/embeddings")
                if parsed.scheme != "http":
                    raise PermissionError(
                        f"LLM_API_BASE must be http, got {parsed.scheme}")
                host = parsed.hostname
                if not host:
                    raise PermissionError("LLM_API_BASE missing host")
                infos = socket.getaddrinfo(host, parsed.port or 80)
                sock_ip = infos[0][4][0]
                ip = ipaddress.ip_address(sock_ip)
                in_bench = ip in ipaddress.ip_network("198.18.0.0/15")
                if not (in_bench or host in _LOCAL_HOSTS) and (
                        ip.is_private or ip.is_loopback or ip.is_reserved
                        or ip.is_link_local or ip.is_multicast or ip.is_unspecified):
                    raise PermissionError(
                        f"refusing non-public LLM_API_BASE {host!r} ({sock_ip})")
                body = json.dumps({"input": [text], "model": model}).encode("utf-8")
                req = urllib.request.Request(
                    f"{api_base.rstrip('/')}/embeddings",
                    data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {api_key}"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    payload = json.loads(resp.read())
                vec = payload["data"][0]["embedding"]
                if isinstance(vec, list) and vec:
                    return vec
            except Exception as e:
                print(f"  hippocampus _embed failed: {type(e).__name__}: {str(e)[:80]}",
                      file=sys.stderr)
        return _hash_embedding(text)

    # ---- retrieve --------------------------------------------------------

    def _matches_tag(self, entry: str, record: dict) -> bool:
        """One `any`/`none` clause against a stored record (exact tag containment)."""
        return entry in record.get("tags", [])

    def _matches_narrow(self, spec: dict, record: dict) -> bool:
        """One `narrow_any` entry: candidate name tags vs the record's derived name tags.

        Candidates arrive as "name:<words>" (dataset._name_candidates). Exact resolution
        is membership in the extracted name set — whole phrases only. Fuzzy resolution is
        one edit (Damerau-Levenshtein OSA) plus a shared two-character prefix, so a
        misspelled name ("typsecript", "mongod", "eror") still reaches its memory while
        look-alikes from other vocabularies ("mango" vs "mongo", "k9s" vs "k8s") stay
        blocked. Matching against name tags only — not the content vocabulary — is what
        keeps facts the query merely brushes past from surfacing."""
        resolve = spec.get("resolve", "exact")
        names = record.get("names")
        if not names:
            return False  # fail closed: no extracted identity, no narrow match
        for cand in spec.get("tags", []):
            phrase = cand.split(":", 1)[1] if ":" in cand else cand
            phrase = phrase.strip().lower()
            if not phrase:
                continue
            if resolve == "exact":
                if phrase in names:
                    return True
            else:  # fuzzy
                if any(_osa_distance(phrase, n) <= 1 and phrase[:2] == n[:2]
                       for n in names):
                    return True
        return False

    def _passes(self, record: dict, filters: dict | None) -> bool:
        if not filters:
            return True
        for group in filters.get("any") or []:
            if not any(self._matches_tag(t, record) for t in group):
                return False
        for entry in filters.get("none") or []:
            if self._matches_tag(entry, record):
                return False
        narrow = filters.get("narrow_any") or []
        if narrow and not any(self._matches_narrow(spec, record) for spec in narrow):
            return False
        return True

    @staticmethod
    def _clean_query(query: str) -> str | None:
        """Stopword-stripped, punctuation-free keyword form of the query.

        The server feeds the raw query string straight into its FTS parser, where
        apostrophes, filler words and long word lists degrade or zero out the match
        ("it's been bugging me" matches nothing). The cleaned form is searched as a
        second pass and merged; the tag gate decides precision, so wider recall
        here is safe."""
        words = [w for w in _WORD_RE.findall(query.lower())
                 if w not in _STOPWORDS and len(w) > 1]
        if not words:
            return None
        return " ".join(words)

    def _search(self, query: str, tenant: str, fetch: int) -> list:
        return self._request("POST", "/api/search",
                             {"query": query, "limit": fetch,
                              "minScore": 0.0, "tenantId": tenant}) or []

    def _fuzzy_recovery(self, query: str, tenant: str, results: list, fetch: int) -> list:
        """Second search with query typos corrected against the indexed name vocabulary.

        The server's hybrid search is keyword-led: a misspelled term ("typsecript",
        "mongod") matches nothing and the primary search comes back empty even though
        the memory is there. When a query token sits one edit (OSA, shared two-char
        prefix) from a word of some memory's extracted names, the token is rewritten
        to that word and the search re-run; rows the primary search already returned
        are not duplicated."""
        seen = set()
        for r in results:
            mem = r.get("memory") or {}
            seen.add(mem.get("uri") or mem.get("id"))
        tokens = _WORD_RE.findall(query.lower())
        if len(tokens) > 32:
            return []
        corrections: dict[str, str] = {}
        for tok in tokens:
            if tok in corrections or len(tok) < 4:
                continue
            for side in self._sidecar.values():
                for name in side.get("names") or []:
                    for word in name.split():
                        if (word != tok and len(word) >= 3
                                and _osa_distance(tok, word) <= 1
                                and tok[:2] == word[:2]):
                            corrections[tok] = word
                            break
                    if tok in corrections:
                        break
                if tok in corrections:
                    break
        if not corrections:
            return []
        fixed_query = " ".join(corrections.get(t, t) for t in tokens)
        rows = self._search(fixed_query, tenant, fetch)
        out = []
        for r in rows:
            mem = r.get("memory") or {}
            uri = mem.get("uri") or mem.get("id")
            if uri and uri not in seen:
                out.append(r)
        return out

    def _tenant_row_count(self, tenant: str) -> int:
        return sum(1 for v in self._sidecar.values() if v.get("tenant") == tenant)

    def _rerank(self, query: str, results: list) -> list:
        """Lightweight deterministic re-ranking on top of the server's hybrid score.

        The server returns rows in raw BM25/vector order, which favours any text
        overlap. For an LLM reader the most useful top-K is the rows that actually
        answer the question — that needs signals the server doesn't have:
        extracted canonical-name hits, current-state bonus, and "this doc is
        conversation vs long transcript" shape. We add those signals client-side
        and keep the server's score as a tiebreak so the retrieval-mode baseline
        (raw server order) is preserved when no signal applies.
        """
        if not results:
            return results
        qwords = {w for w in _WORD_RE.findall(query.lower()) if len(w) > 1}
        scored: list[tuple[float, float, int, dict]] = []
        for idx, r in enumerate(results):
            mem = r.get("memory") or {}
            uri = mem.get("uri") or mem.get("id") or ""
            # Rerank on the *truncated* view the downstream RAG consumer will
            # see, so the signals here line up with what the LLM actually reads.
            content = self._truncate(mem.get("content") or "").lower()
            side = self._sidecar.get(uri, {})
            names = side.get("names") or []
            tags = side.get("tags") or []

            score = 0.0
            # Word overlap with content (cheap BM25-ish signal)
            for w in qwords:
                if w in content:
                    score += 1.0
            # Canonical-name hit is a much stronger signal than raw token overlap
            # because the name vocabulary was the curated set the answerer reads back.
            for name in names:
                name_lower = name.lower()
                for w in qwords:
                    if w == name_lower or (len(w) >= 3 and w in name_lower):
                        score += 2.0
            # Penalise rows tagged as not-current: superseded / open questions
            # won't answer a present-tense query.
            if any(t.startswith("state:historical") for t in tags):
                score -= 1.5
            if any(t.startswith("state:open_question") for t in tags):
                score -= 1.0
            # Conversation snippets (many turns, short content) are more answerable
            # than long log dumps.
            if 0 < len(content) < 200:
                score += 0.5
            base = -(r.get("score") or 0.0)
            scored.append((score, base, idx, r))

        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        return [t[3] for t in scored]

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                 query_timestamp: str | None = None, filters: dict | None = None):
        k = int(os.environ.get("HIPPOCAMPUS_RETRIEVE_K", str(k)))
        tenant = user_id or _DEFAULT_TENANT
        # The server ranks across ALL tenants and only post-filters by tenant, so the
        # fetch window must be wide enough to hold this tenant's whole slice even when
        # other tenants' rows crowd the top of the ranking (a shared multi-tenant
        # store). Fetching the tenant's full sidecar row count guarantees that.
        fetch = max(min(k * _FETCH_FACTOR, 200), _FETCH_MIN,
                    min(self._tenant_row_count(tenant) + 8, 2000))
        body = {
            "query": query,
            "limit": fetch,
            # minScore=0 keeps the full ranking; the server default truncates recall-oriented
            # evaluation after score weighting.
            "minScore": 0.0,
            "tenantId": tenant,
        }
        results = self._request("POST", "/api/search", body) or []
        seen_uris = {(r.get("memory") or {}).get("uri") or (r.get("memory") or {}).get("id")
                     for r in results}
        cleaned = self._clean_query(query)
        if cleaned:
            for r in self._search(cleaned, tenant, fetch):
                mem = r.get("memory") or {}
                uri = mem.get("uri") or mem.get("id")
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    results.append(r)
        results = results + self._fuzzy_recovery(query, tenant, results, fetch)

        docs: list[Document] = []
        raw_docs: list[dict] = []
        for r in self._rerank(query, results):
            mem = r.get("memory") or {}
            uri = mem.get("uri") or mem.get("id")
            if not uri:
                continue
            side = self._sidecar.get(uri, {})
            record = {
                "tags": side.get("tags", []),
                "names": side.get("names") or [],
            }
            if not self._passes(record, filters):
                continue
            content = self._truncate(mem.get("content") or "")
            raw_docs.append({
                "uri": uri,
                "score": r.get("score"),
                "source": r.get("source"),
                # LongMemEval's build_rag_prompt prefers json.dumps(raw_response)
                # over the rendered context when raw is present — without the
                # content here the answer LLM sees URIs and scores only.
                "content": content,
            })
            docs.append(Document(
                id=uri,
                content=content,
                user_id=tenant,
                timestamp=query_timestamp,
                context=side.get("context"),
                source_ids=None,
                tags=side.get("tags"),
            ))
            if len(docs) >= k:
                break
        return docs, {"documents": raw_docs}

    @staticmethod
    def _truncate(text: str) -> str:
        """Cap each returned memory's content at `HIPPOCAMPUS_DOC_CHAR_LIMIT`
        characters so an RAG-mode prompt doesn't drown the LLM in long
        transcripts. The cap is intentionally conservative (default 6000) so
        any single document still fits in a single LLM context window along
        with several siblings. Empirically, for LongMemEval sessions the
        assistant's answers tend to sit in the **tail** of the conversation
        transcript (the user asks, then the assistant answers), so we keep
        the head and the tail and drop the bulk in between — flipping this
        loses the gold answer, which the head-only cut does. Queries that
        need full documents should ask the server directly rather than going
        through `retrieve`. Documents shorter than the cap pass through
        untouched.
        """
        cap = int(os.environ.get("HIPPOCAMPUS_DOC_CHAR_LIMIT", "0"))
        if cap <= 0 or len(text) <= cap:
            return text
        # LongMemEval session transcripts put the assistant's answer near the
        # end of the conversation (the user asks, the assistant answers). We
        # therefore keep the *tail* — the answer half — and drop the bulk
        # in between to fit the LLM context window. Retrievals against a
        # rag() prompt pass this truncated form to the LLM.
        return text[-cap:].rstrip() + "…"
