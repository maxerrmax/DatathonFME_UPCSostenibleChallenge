"""Guardrails for the Energy Copilot, in four layers.

1. Input   – prompt-injection / secret-exfiltration attempts are answered without
             calling the model; uploaded column names are sanitised before they
             reach a prompt (a CSV header is untrusted text).
2. Tools   – (copilot.clean_args) closed schemas, field whitelist, clamped ranges,
             no code execution, bounded number of tool calls.
3. Output  – every figure must appear in this turn's evidence; sentences with
             unverifiable figures are removed, uncited claims are reported,
             length is capped and markup stripped.
4. Volume  – per-session rate limit, so an exposed demo cannot burn API credit.

Privacy: the model only ever receives aggregated tool outputs and column names,
never raw rows of a dataset.
"""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import defaultdict, deque

MAX_ANSWER_CHARS = 1800


def _plain(text: str) -> str:
    t = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text).casefold())
    return "".join(c for c in t if not unicodedata.combining(c))


# ---------------------------------------------------------------- 1. input
INJECTION_RE = re.compile(
    r"(ignor\w*|oblida\w*|olvida\w*|forget|disregard|salta\w*)\s+(?:\w+\s+){0,4}"
    r"(instruc\w*|normes?|regles?|reglas?|rules?|prompt|indicacions|guardrails?)"
    r"|system\s*prompt|prompt\s+(?:del\s+)?sistema|mode\s+(?:desenvolupador|developer|dan)|\bjailbreak\b"
    r"|(?:act|actua|fes)\s+(?:as|com\s+a?|de)\s+(?:si\s+fossis\s+)?(?:un\s+)?(?:altre|another|dan)\b"
    r"|(?:revela|mostra|ensenya|digues|dime|show|reveal|print)\w*\s+(?:\w+\s+){0,3}(?:clau|key|token|api|secret|contrasenya|password|\.env)")
SECRET_RE = re.compile(r"helmcode_api_key|api[_\s-]?key|\.env\b|bearer\s")

REFUSAL = ("No puc canviar les meves normes, revelar la configuració ni respondre sense evidències. "
           "Sí que puc analitzar les dades energètiques del campus, simular la bateria o comparar estratègies: "
           "per exemple, «Quines hores són les més netes i barates?» o «Quina estratègia de bateria funciona millor?».")


def input_check(message: str) -> str | None:
    """Return a refusal text when the message tries to subvert the assistant."""
    q = _plain(message)
    if INJECTION_RE.search(q) or SECRET_RE.search(q):
        return REFUSAL
    return None


SAFE_FIELD_RE = re.compile(r"^[\w .,%()/°²₂\-]{1,48}$")
FIELD_INJECTION_RE = re.compile(r"ignor|instruc|prompt|system|assistant|respon|answer|http|www\.|<|>|\{|\}")


def safe_fields(fields: list[str]) -> list[str]:
    """Column names that may be shown to the model (uploaded headers are untrusted)."""
    return [f for f in fields if SAFE_FIELD_RE.match(f) and not FIELD_INJECTION_RE.search(_plain(f))]


# ---------------------------------------------------------------- 3. output
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý«\"(])")


def output_guard(text: str, verify_fn, evidence: list[dict]) -> tuple[str, dict]:
    """Remove sentences whose figures are not in the evidence; report uncited lines."""
    text = re.sub(r"<[^>]{0,200}>", "", text or "")  # no markup from the model
    text = re.sub(r"```.*?```", "", text, flags=re.S).strip()
    kept_lines, removed = [], []
    for line in text.split("\n"):
        if not line.strip():
            continue
        sentences = SENTENCE_RE.split(line.strip())
        good = []
        for s in sentences:
            check = verify_fn(s, evidence)
            if check["ok"]:
                good.append(s)
            else:
                removed.append({"sentence": s[:220], "unverified": check["unverified"]})
        if good:
            kept_lines.append(" ".join(good))
    clean = "\n".join(kept_lines)
    if len(clean) > MAX_ANSWER_CHARS:
        clean = clean[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + "…"
    uncited = [l[:160] for l in kept_lines
               if not re.search(r"\[E\d+\]", l) and not re.match(r"^\W*Decisi", l) and re.search(r"\d", l)]
    return clean, {"removed_sentences": removed, "uncited_lines": uncited}


# ---------------------------------------------------------------- 4. volume
class RateLimiter:
    """Sliding window per anonymous session (in memory; enough for a local demo)."""
    def __init__(self, limit: int = 20, window_s: float = 60.0):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str | None) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key or "anonymous"]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


ASSISTANT_LIMIT = RateLimiter()
