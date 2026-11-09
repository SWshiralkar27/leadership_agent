"""
Leadership AI Daily Digest Generator — v3.0
============================================

Workflow:
    discover → dedupe → keyword-screen → contextual analysis
    → deterministic guardrails → conditional drafting
    → export HTML/Markdown/JSON → human review.

Fail-closed by design. No template fallbacks. Human publishes manually.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

try:
    from dotenv import load_dotenv
    load_dotenv()
    try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Streamlit Cloud Secrets → environment
try:
    import streamlit as _st
    for _k, _v in _st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Version & configuration
# ---------------------------------------------------------------------------
VERSION = "3.0"
PROMPT_VERSION = "3.0.0"

AUTHOR_NAME = os.getenv("AUTHOR_NAME", "Shreekant Shiralkar")
BOOK_TITLE = "Games for Gravitas"
BOOK_SUBTITLE = "A Playbook to Build Human Advantage in the Hybrid Intelligence Era"
BOOK_FULL_TITLE = f"{BOOK_TITLE}: {BOOK_SUBTITLE}"
BOOK_URL = os.getenv("BOOK_URL", "https://example.com/games-for-gravitas")

# Used by plain-text platforms (YouTube / Shorts)
BOOK_NAME_YOUTUBE = BOOK_TITLE

# Decision thresholds (spec §4)
PROMOTION_THRESHOLD = int(os.getenv("PROMOTION_THRESHOLD", "80"))
ENGAGEMENT_THRESHOLD = int(os.getenv("ENGAGEMENT_THRESHOLD", "60"))

# Context gate (spec §5)
MIN_CONTEXT_CHARS = int(os.getenv("MIN_CONTEXT_CHARS", "180"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

# History
HISTORY_FILE = Path(os.getenv("HISTORY_FILE", "seen_posts.txt"))
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "90"))
IGNORE_HISTORY = os.getenv("IGNORE_HISTORY", "false").lower() == "true"

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_DELAY_SECONDS = float(os.getenv("GEMINI_DELAY_SECONDS", "3"))

# Retrieval (informational; enabled sources wired in discovery module below)
MAX_RESULTS_PER_SOURCE = int(os.getenv("MAX_RESULTS_PER_SOURCE", "7"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))

PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", Path(__file__).parent / "prompts"))

# Book themes with descriptions — populated into analysis prompt (spec §6).
BOOK_THEMES: list[tuple[str, str]] = [
    ("Centaur leadership",
     "Humans and AI together out-think either one alone."),
    ("Purposeful rehearsal",
     "Structured games let leaders rehearse high-stakes decisions safely."),
    ("Playbook over theory",
     "Modular exercises that leaders actively apply."),
    ("Non-automatable skills",
     "Judgment, intuition, empathy, ethics and creativity."),
    ("Hybrid Intelligence Era",
     "AI as a strategic collaborator; humans and AI co-evolve."),
    ("Structured human-AI teaming",
     "The 7-Phase Model for blended human-AI teams."),
]

PLAIN_TEXT_PLATFORMS = {"YouTube", "YouTube Shorts"}
PLATFORM_ORDER = ["LinkedIn", "X", "Medium", "YouTube", "YouTube Shorts",
                  "Facebook", "Instagram"]
PLATFORM_LABEL = {
    "LinkedIn": "LinkedIn", "X": "X / Twitter", "Medium": "Medium",
    "YouTube": "YouTube", "YouTube Shorts": "YouTube Shorts",
    "Facebook": "Facebook", "Instagram": "Instagram",
}
PLATFORM_ICON = {
    "LinkedIn": "in", "X": "X", "Medium": "M",
    "YouTube": "▶", "YouTube Shorts": "▶", "Facebook": "f",
    "Instagram": "IG",
}

KEYWORDS: tuple[str, ...] = (
    "ai", "artificial", "intelligence", "leadership", "leader", "hybrid",
    "centaur", "judgment", "judgement", "rehearsal", "playbook", "team",
    "teaming", "human", "automat", "empathy", "ethics", "creativity",
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("digest")


# ---------------------------------------------------------------------------
# Enums & data model
# ---------------------------------------------------------------------------
class Decision(str, Enum):
    A_PROMOTE = "A_PROMOTE"
    B_ENGAGE = "B_ENGAGE"
    C_SENSITIVE = "C_SENSITIVE"
    D_SKIP = "D_SKIP"


class Sensitivity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class ContentItem:
    platform: str
    title: str
    url: str
    source: str = ""
    published: str = ""
    snippet: str = ""
    full_content: str = ""
    fetched_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # Discovery score (keyword-only; NOT contextual fit)
    relevance_score: float = 0.0

    # Analysis outputs
    decision: Decision = Decision.D_SKIP
    decision_reason: str = ""
    context_summary: str = ""
    sensitivity: Sensitivity = Sensitivity.LOW
    sensitivity_reason: str = ""
    confidence: Confidence = Confidence.LOW
    book_theme: str = ""
    book_alignment_score: int = 0
    contextual_relevance_score: int = 0
    engagement_score: int = 0
    audience_alignment_score: int = 0
    promotion_naturalness_score: int = 0
    promotion_score: int = 0
    promotion_justification: str = ""

    # Drafts (only for A_PROMOTE / B_ENGAGE)
    comment: str = ""
    alternatives: list[str] = field(default_factory=list)

    # Diagnostics
    rejection_reason: str = ""
    analysis_error: str = ""

    @property
    def content_for_analysis(self) -> str:
        raw = self.full_content or self.snippet
        return clean_text(raw)[:MAX_CONTEXT_CHARS]


@dataclass
class Digest:
    date: str
    total_items: int
    items_by_platform: dict[str, list[ContentItem]]
    summary: str
    generated_at: str
    version: str = VERSION
    prompt_version: str = PROMPT_VERSION


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
_WHITESPACE = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE.sub(" ", text).strip()


def truncate_at_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _parse_date(value: str) -> datetime:
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "ref_src", "src", "si", "s", "share",
}


def normalize_url(url: str) -> str:
    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query) if k not in _TRACKING_PARAMS]
    path = p.path.rstrip("/")
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path,
                       "", urlencode(sorted(q)), ""))


def calculate_relevance_score(item: ContentItem) -> float:
    """Keyword-only discovery score (spec §8). Not contextual fit."""
    text = f"{item.title} {item.snippet}".lower()
    tokens = re.findall(r"[a-z]+", text)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in KEYWORDS)
    return hits / len(tokens)


def _word_range(platform: str) -> tuple[int, int]:
    return {
        "LinkedIn": (40, 80),
        "X": (20, 60),
        "Medium": (60, 120),
        "YouTube": (30, 90),
        "YouTube Shorts": (25, 70),
        "Facebook": (35, 90),
        "Instagram": (20, 60),
    }.get(platform, (40, 90))


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def load_history() -> set[str]:
    if IGNORE_HISTORY or not HISTORY_FILE.exists():
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
    urls: set[str] = set()
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        url, _, ts = line.partition("\t")
        if ts:
            try:
                when = datetime.fromisoformat(ts)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                if when < cutoff:
                    continue
            except ValueError:
                pass
        urls.add(url)
    return urls


def save_history(urls: Iterable[str]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = {}
    if HISTORY_FILE.exists():
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            url, _, ts = line.partition("\t")
            if url:
                existing[url] = ts
    for u in urls:
        existing[normalize_url(u)] = now
    HISTORY_FILE.write_text(
        "\n".join(f"{u}\t{ts}" for u, ts in sorted(existing.items())),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
class GeminiUnavailable(RuntimeError):
    pass


class GeminiClient:
    """Thin wrapper around google-generativeai with retries."""

    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise GeminiUnavailable("GEMINI_API_KEY is not set.")
        try:
            import google.generativeai as genai  # lazy
        except ImportError as exc:
            raise GeminiUnavailable(
                "google-generativeai not installed. pip install google-generativeai"
            ) from exc
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def generate(self, prompt: str, *, max_tokens: int = 1200,
                 retries: int = 2) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = self._model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": 0.3,
                        "response_mime_type": "application/json",
                    },
                )
                return (resp.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("Gemini call failed (attempt %d/%d): %s",
                            attempt + 1, retries + 1, exc)
                time.sleep(2 ** attempt)
        raise GeminiUnavailable(f"Gemini failed after retries: {last_err}")


def _gemini_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        val = json.loads(cleaned)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass
    # balanced-brace scan for first complete object
    depth, start = 0, -1
    in_str, esc = False, False
    for i, ch in enumerate(cleaned):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    val = json.loads(cleaned[start:i + 1])
                    return val if isinstance(val, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _bounded_score(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
_PROMPT_CACHE: dict[str, str] = {}


def _load_prompt(name: str) -> str:
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    _PROMPT_CACHE[name] = path.read_text(encoding="utf-8")
    return _PROMPT_CACHE[name]


def _themes_block() -> str:
    return "\n".join(f"- {name} — {desc}" for name, desc in BOOK_THEMES)


# ---------------------------------------------------------------------------
# Stage 1: Analysis
# ---------------------------------------------------------------------------
# Keys the analysis model MUST return (contract with prompt).
ANALYSIS_KEYS = {
    "decision", "decision_reason", "context_summary",
    "sensitivity", "sensitivity_reason", "confidence",
    "book_theme", "book_alignment_score", "contextual_relevance_score",
    "engagement_score", "audience_alignment_score",
    "promotion_naturalness_score", "promotion_justification",
}

ALLOWED_DECISIONS = {d.value for d in Decision}


def analyse_post(client: GeminiClient, item: ContentItem) -> Optional[dict]:
    """
    Stage 1: classify the post. Returns validated dict or None.
    Fail-closed: any error → None (caller marks D_SKIP).
    """
    context = item.content_for_analysis
    actual_chars = len(context)

    prompt = _load_prompt("v3_analysis.md").format(
        platform=item.platform,
        title=item.title,
        source=item.source or "(unknown)",
        content=context or "(no usable text supplied)",
        book_themes=_themes_block(),
    )
    try:
        raw = client.generate(prompt, max_tokens=1200)
    except GeminiUnavailable as exc:
        item.analysis_error = str(exc)
        return None

    parsed = _gemini_json(raw)
    if not parsed or not ANALYSIS_KEYS.issubset(parsed.keys()):
        missing = ANALYSIS_KEYS - set(parsed.keys()) if parsed else ANALYSIS_KEYS
        item.analysis_error = f"Analysis JSON incomplete. Missing: {sorted(missing)}"
        return None

    # Normalize enums
    decision_raw = str(parsed["decision"]).upper()
    if decision_raw not in ALLOWED_DECISIONS:
        item.analysis_error = f"Invalid decision: {decision_raw}"
        return None
    sensitivity_raw = str(parsed.get("sensitivity", "LOW")).upper()
    confidence_raw = str(parsed.get("confidence", "LOW")).upper()
    sensitivity = (sensitivity_raw if sensitivity_raw in
                   {s.value for s in Sensitivity} else "MEDIUM")
    confidence = (confidence_raw if confidence_raw in
                  {c.value for c in Confidence} else "LOW")

    scores = {k: _bounded_score(parsed.get(k, 0)) for k in (
        "book_alignment_score", "contextual_relevance_score",
        "engagement_score", "audience_alignment_score",
        "promotion_naturalness_score",
    )}

    # Python computes the weighted score (spec §4).
    promotion_score = round(
        scores["book_alignment_score"] * 0.30 +
        scores["contextual_relevance_score"] * 0.25 +
        scores["engagement_score"] * 0.20 +
        scores["audience_alignment_score"] * 0.15 +
        scores["promotion_naturalness_score"] * 0.10
    )

    decision = decision_raw
    rejection = ""

    # Deterministic guardrails (spec §5, §13).
    if sensitivity == "HIGH":
        if decision != "C_SENSITIVE":
            rejection = f"HIGH sensitivity override: {decision} → C_SENSITIVE."
        decision = "C_SENSITIVE"
    elif actual_chars < MIN_CONTEXT_CHARS:
        if decision != "D_SKIP":
            rejection = f"Context too short ({actual_chars}<{MIN_CONTEXT_CHARS}) → D_SKIP."
        decision = "D_SKIP"
        confidence = "LOW"
    elif confidence == "LOW":
        if decision != "D_SKIP":
            rejection = "LOW confidence → D_SKIP."
        decision = "D_SKIP"
    elif decision == "A_PROMOTE" and promotion_score < PROMOTION_THRESHOLD:
        new = "B_ENGAGE" if promotion_score >= ENGAGEMENT_THRESHOLD else "D_SKIP"
        rejection = f"A_PROMOTE below threshold ({promotion_score}) → {new}."
        decision = new
    elif decision == "B_ENGAGE" and promotion_score < ENGAGEMENT_THRESHOLD:
        rejection = f"B_ENGAGE below threshold ({promotion_score}) → D_SKIP."
        decision = "D_SKIP"

    return {
        "decision": decision,
        "decision_reason": clean_text(str(parsed.get("decision_reason", ""))),
        "context_summary": clean_text(str(parsed.get("context_summary", ""))),
        "sensitivity": sensitivity,
        "sensitivity_reason": clean_text(str(parsed.get("sensitivity_reason", ""))),
        "confidence": confidence,
        "book_theme": clean_text(str(parsed.get("book_theme", ""))),
        "promotion_justification": clean_text(
            str(parsed.get("promotion_justification", ""))),
        "promotion_score": promotion_score,
        "guardrail_override": rejection,
        **scores,
    }


# ---------------------------------------------------------------------------
# Stage 2: Drafting (conditional)
# ---------------------------------------------------------------------------
_PROMO_BANNED = re.compile(
    r"games\s+for\s+gravitas|shiralkar|amazon\.|amzn\.|buy\s+now|"
    r"order\s+(the\s+)?book|purchase|https?://",
    re.I,
)

_BOOK_CANON_MD = f'[**"{BOOK_TITLE}"**]({BOOK_URL})'
_BOOK_CANON_PLAIN = f'"{BOOK_NAME_YOUTUBE}"'


def _enforce_book_format(text: str, plain: bool) -> str:
    if plain:
        pat = re.compile(r'["“”]?' + re.escape(BOOK_NAME_YOUTUBE) + r'["“”]?', re.I)
        return pat.sub(lambda _m: _BOOK_CANON_PLAIN, text, count=1)
    link_pat = re.compile(
        r'\*{0,2}\[[^\]]*Games for Gravitas[^\]]*\]\([^)]*\)\*{0,2}', re.I)
    if link_pat.search(text):
        return link_pat.sub(lambda _m: _BOOK_CANON_MD, text, count=1)
    bare = re.compile(r'[*_"“”]{0,3}Games for Gravitas[*_"“”]{0,3}', re.I)
    return bare.sub(lambda _m: _BOOK_CANON_MD, text, count=1)


def _book_mention_count(text: str) -> int:
    return text.lower().count(BOOK_TITLE.lower())


def _validate_promote(text: str, plain: bool) -> tuple[bool, str]:
    n = _book_mention_count(text)
    if n == 0:
        return False, "A_PROMOTE draft omits the book title."
    if n > 1:
        return False, f"A_PROMOTE draft mentions the book {n} times (must be exactly 1)."
    return True, ""


def _validate_engage(text: str) -> tuple[bool, str]:
    if _PROMO_BANNED.search(text):
        return False, "B_ENGAGE draft contains book/author/purchase/URL reference."
    return True, ""


def generate_drafts(client: GeminiClient, item: ContentItem) -> None:
    """Stage 2: populate item.comment / item.alternatives, or leave empty."""
    if item.decision not in (Decision.A_PROMOTE, Decision.B_ENGAGE):
        return

    plain = item.platform in PLAIN_TEXT_PLATFORMS
    lo, hi = _word_range(item.platform)
    context = item.content_for_analysis

    if item.decision == Decision.A_PROMOTE:
        prompt_name = "v3_draft_promote.md"
        validator = lambda t: _validate_promote(t, plain)  # noqa: E731
    else:
        prompt_name = "v3_draft_engage.md"
        validator = _validate_engage  # type: ignore[assignment]

    book_instruction = (
        "Mention the book exactly once. Use markdown link "
        f"{_BOOK_CANON_MD}." if not plain else
        f'Mention the book exactly once as {_BOOK_CANON_PLAIN} in plain text.'
    ) if item.decision == Decision.A_PROMOTE else (
        "Do NOT mention the book, the author, Amazon, any URL, or any "
        "product or service. This must be a standalone contribution.")

    prompt = _load_prompt(prompt_name).format(
        platform=item.platform,
        title=item.title,
        decision=item.decision.value,
        decision_reason=item.decision_reason,
        book_theme=item.book_theme or "(none — do not force one)",
        context=context,
        word_lo=lo,
        word_hi=hi,
        book_instruction=book_instruction,
    )

    try:
        raw = client.generate(prompt, max_tokens=2600)
    except GeminiUnavailable as exc:
        item.rejection_reason = f"Draft generation failed: {exc}"
        return

    parsed = _gemini_json(raw)
    if not parsed:
        item.rejection_reason = "Draft JSON could not be parsed."
        return

    recommended = clean_text(str(parsed.get("recommended", "")))
    alternatives = parsed.get("alternatives", [])
    if not isinstance(alternatives, list):
        alternatives = []
    candidates = [recommended] + [
        clean_text(str(a)) for a in alternatives if str(a).strip()
    ]
    candidates = [c for c in candidates if len(c) >= 30]

    kept: list[str] = []
    rejects: list[str] = []
    for candidate in candidates:
        ok, why = validator(candidate)
        if not ok:
            rejects.append(why)
            continue
        if item.decision == Decision.A_PROMOTE:
            candidate = _enforce_book_format(candidate, plain)
        kept.append(candidate)

    if rejects:
        item.rejection_reason = "Rejected drafts: " + "; ".join(rejects)
    if not kept:
        return
    item.comment = kept[0]
    item.alternatives = kept[1:3]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def dedupe_and_filter(items: list[ContentItem],
                      history: set[str]) -> list[ContentItem]:
    seen: set[str] = set()
    out: list[ContentItem] = []
    repeats = 0
    for it in items:
        key = normalize_url(it.url)
        if key in seen:
            continue
        seen.add(key)
        if key in history:
            repeats += 1
            continue
        out.append(it)
    if repeats:
        log.info("Skipped %d URL(s) already in history", repeats)
    return out


def select_candidates(items: list[ContentItem]) -> list[ContentItem]:
    """Keyword-screen per platform. NO weak-match backfill (spec §8)."""
    selected: list[ContentItem] = []
    for platform in PLATFORM_ORDER:
        pool = [i for i in items if i.platform == platform]
        for i in pool:
            i.relevance_score = calculate_relevance_score(i)
        pool.sort(key=lambda i: i.relevance_score, reverse=True)
        chosen = [i for i in pool
                  if i.relevance_score >= RELEVANCE_THRESHOLD][:MAX_RESULTS_PER_SOURCE]
        if not chosen and pool:
            log.info("%s: 0 candidates passed threshold; no backfill.", platform)
        selected.extend(chosen)
    return selected


def run_digest(raw_items: list[ContentItem],
               *, out_dir: Path = Path(".")) -> Digest:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_history()
    unique = dedupe_and_filter(raw_items, history)
    selected = select_candidates(unique)

    log.info("Analysing %d candidate(s) with Gemini", len(selected))

    try:
        client = GeminiClient(GEMINI_API_KEY, GEMINI_MODEL)
    except GeminiUnavailable as exc:
        log.error("Gemini unavailable: %s — all items will be D_SKIP.", exc)
        client = None  # type: ignore[assignment]

    for n, item in enumerate(selected, 1):
        log.info("[%d/%d] %s — %s", n, len(selected), item.platform,
                 truncate_at_boundary(item.title, 60))

        if client is None:
            item.decision = Decision.D_SKIP
            item.decision_reason = "Gemini unavailable; fail-closed."
            item.analysis_error = "Gemini unavailable."
            continue

        analysis = analyse_post(client, item)
        if analysis is None:
            item.decision = Decision.D_SKIP
            item.decision_reason = item.analysis_error or "Analysis failed."
            item.confidence = Confidence.LOW
            time.sleep(GEMINI_DELAY_SECONDS)
            continue

        item.decision = Decision(analysis["decision"])
        item.decision_reason = analysis["decision_reason"]
        item.context_summary = analysis["context_summary"]
        item.sensitivity = Sensitivity(analysis["sensitivity"])
        item.sensitivity_reason = analysis["sensitivity_reason"]
        item.confidence = Confidence(analysis["confidence"])
        item.book_theme = analysis["book_theme"]
        item.promotion_justification = analysis["promotion_justification"]
        item.book_alignment_score = analysis["book_alignment_score"]
        item.contextual_relevance_score = analysis["contextual_relevance_score"]
        item.engagement_score = analysis["engagement_score"]
        item.audience_alignment_score = analysis["audience_alignment_score"]
        item.promotion_naturalness_score = analysis["promotion_naturalness_score"]
        item.promotion_score = analysis["promotion_score"]

        if analysis.get("guardrail_override"):
            item.rejection_reason = analysis["guardrail_override"]

        if item.decision in (Decision.A_PROMOTE, Decision.B_ENGAGE):
            time.sleep(GEMINI_DELAY_SECONDS)
            generate_drafts(client, item)

        time.sleep(GEMINI_DELAY_SECONDS)

    grouped: dict[str, list[ContentItem]] = {}
    for platform in PLATFORM_ORDER:
        group = [i for i in selected if i.platform == platform]
        if group:
            group.sort(key=lambda i: _parse_date(i.published), reverse=True)
            grouped[platform] = group

    counts = {d.value: sum(1 for i in selected if i.decision == d)
              for d in Decision}
    summary = (
        f"{len(selected)} candidate(s) assessed: "
        f"{counts['A_PROMOTE']} promote · {counts['B_ENGAGE']} engage (no promotion) · "
        f"{counts['C_SENSITIVE']} sensitive/abstain · {counts['D_SKIP']} skip."
        if selected else "No candidates passed discovery filter.")

    digest = Digest(
        date=datetime.now().strftime("%Y-%m-%d"),
        total_items=len(selected),
        items_by_platform=grouped,
        summary=summary,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_html(digest, out_dir / f"digest_{stamp}.html")
    write_markdown(digest, out_dir / f"digest_{stamp}.md")
    write_json(digest, out_dir / f"digest_{stamp}.json")

    save_history(i.url for i in selected)
    log.info("Digest written to %s", out_dir.resolve())
    return digest


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------
def _item_to_dict(it: ContentItem) -> dict[str, Any]:
    d = asdict(it)
    d["decision"] = it.decision.value
    d["sensitivity"] = it.sensitivity.value
    d["confidence"] = it.confidence.value
    return d


def write_json(digest: Digest, path: Path) -> None:
    payload = {
        "version": digest.version,
        "prompt_version": digest.prompt_version,
        "date": digest.date,
        "generated_at": digest.generated_at,
        "summary": digest.summary,
        "total_items": digest.total_items,
        "items": [
            {"platform": p, **_item_to_dict(it)}
            for p, group in digest.items_by_platform.items()
            for it in group
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def write_markdown(digest: Digest, path: Path) -> None:
    lines = [
        f"# Leadership AI Daily Digest — {digest.date}",
        f"**Version:** {digest.version} · **Prompt:** {digest.prompt_version}",
        f"**Generated:** {digest.generated_at}",
        "",
        "## Summary", digest.summary, "",
        "> Drafts are suggestions. Read the source, check the facts, edit before publishing.", "",
    ]
    for platform, items in digest.items_by_platform.items():
        lines.append(f"## {PLATFORM_LABEL.get(platform, platform)}")
        for n, it in enumerate(items, 1):
            lines += [
                f"### {n}. [{it.title}]({it.url})",
                f"- **Source:** {it.source}",
                f"- **Published:** {it.published}",
                f"- **Discovery score:** {it.relevance_score:.2f}",
                f"- **Decision:** `{it.decision.value}`",
                f"- **Reason:** {it.decision_reason}",
                f"- **Context:** {it.context_summary}",
                f"- **Sensitivity:** {it.sensitivity.value} ({it.sensitivity_reason or 'n/a'})",
                f"- **Confidence:** {it.confidence.value}",
                f"- **Promotion score:** {it.promotion_score}/100",
                "- **Components:** "
                f"book {it.book_alignment_score} · context {it.contextual_relevance_score} · "
                f"engagement {it.engagement_score} · audience {it.audience_alignment_score} · "
                f"naturalness {it.promotion_naturalness_score}",
            ]
            if it.book_theme:
                lines.append(f"- **Theme:** {it.book_theme}")
            if it.rejection_reason:
                lines.append(f"- **Notes:** {it.rejection_reason}")
            lines.append("")
            if it.comment:
                lines += ["**Recommended draft:**", "", it.comment, ""]
                for i, alt in enumerate(it.alternatives, 1):
                    lines += [f"**Alternative {i}:**", "", alt, ""]
            else:
                lines += ["_No draft generated — abstain, skip, or review manually._", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


_HTML_CSS = """
body{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
background:#f5f6f8;color:#202733;margin:0}
.wrap{max-width:1000px;margin:auto;padding:24px}
.card{background:white;border:1px solid #dfe3e8;border-radius:10px;padding:18px;margin:16px 0}
.meta,.muted{color:#667085;font-size:13px}
.reason{margin:8px 0}
.pill{display:inline-block;border-radius:20px;padding:3px 10px;font-size:12px;
font-weight:600;background:#e5e7eb}
.pill.A_PROMOTE{background:#dcfce7;color:#166534}
.pill.B_ENGAGE{background:#dbeafe;color:#1e40af}
.pill.C_SENSITIVE{background:#fee2e2;color:#991b1b}
.pill.D_SKIP{background:#e5e7eb;color:#374151}
.draft{border-left:3px solid #d6a23a;padding:10px 14px;margin:12px 0;background:#fafafa;
border-radius:0 6px 6px 0}
button{border:0;border-radius:6px;padding:7px 12px;background:#1d4ed8;color:white;
cursor:pointer;margin:6px 4px 0 0;font-size:13px}
button:hover{background:#1e40af}
a{color:#1d4ed8;text-decoration:none}a:hover{text-decoration:underline}
h1{margin:0 0 4px}h2{margin:24px 0 4px}
.component{display:inline-block;margin-right:12px;font-size:12px;color:#475467}
"""


def write_html(digest: Digest, path: Path) -> None:
    esc = lambda s: html.escape(str(s or ""), quote=True)  # noqa: E731

    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Leadership AI Digest — {esc(digest.date)}</title>",
        f"<style>{_HTML_CSS}</style></head><body><div class='wrap'>",
        "<h1>Leadership AI Daily Digest</h1>",
        f"<div class='meta'>{esc(digest.date)} · {digest.total_items} candidates · "
        f"v{esc(digest.version)} · prompt {esc(digest.prompt_version)}</div>",
        f"<p>{esc(digest.summary)}</p>",
        "<p class='muted'><b>Human review required.</b> A decision is a "
        "recommendation, not an instruction to publish. Read the source post "
        "and edit every draft before posting.</p>",
    ]

    for platform, items in digest.items_by_platform.items():
        parts.append(
            f"<h2>{esc(PLATFORM_ICON.get(platform, ''))} "
            f"{esc(PLATFORM_LABEL.get(platform, platform))}</h2>")

        for n, it in enumerate(items, 1):
            parts.append("<section class='card'>")
            parts.append(
                f"<div class='meta'>Post {n} · discovery {it.relevance_score:.2f}</div>")
            parts.append(
                f"<h3><a href='{esc(it.url)}' target='_blank' rel='noopener'>"
                f"{esc(it.title)}</a></h3>")
            parts.append(
                f"<div class='meta'>{esc(it.source)} · {esc(it.published)}</div>")

            parts.append(
                f"<p><span class='pill {esc(it.decision.value)}'>"
                f"{esc(it.decision.value)}</span> "
                f"<span class='meta'>Sensitivity: {esc(it.sensitivity.value)} · "
                f"Confidence: {esc(it.confidence.value)} · "
                f"Promotion score: {it.promotion_score}/100</span></p>")

            parts.append(f"<p class='reason'><b>Why:</b> {esc(it.decision_reason)}</p>")
            if it.sensitivity_reason:
                parts.append(
                    f"<p class='muted'><b>Sensitivity reason:</b> "
                    f"{esc(it.sensitivity_reason)}</p>")
            if it.context_summary:
                parts.append(
                    f"<p class='muted'><b>Context:</b> "
                    f"{esc(it.context_summary)}</p>")
            if it.book_theme:
                parts.append(
                    f"<p class='muted'><b>Theme:</b> {esc(it.book_theme)}</p>")

            parts.append(
                "<div class='meta'>"
                f"<span class='component'>Book {it.book_alignment_score}</span>"
                f"<span class='component'>Context {it.contextual_relevance_score}</span>"
                f"<span class='component'>Engagement {it.engagement_score}</span>"
                f"<span class='component'>Audience {it.audience_alignment_score}</span>"
                f"<span class='component'>Naturalness "
                f"{it.promotion_naturalness_score}</span>"
                "</div>")

            if it.rejection_reason:
                parts.append(
                    f"<p class='muted'><b>Notes:</b> "
                    f"{esc(it.rejection_reason)}</p>")

            if it.comment:
                drafts = [("Recommended", it.comment)] + [
                    (f"Alternative {i}", d)
                    for i, d in enumerate(it.alternatives, 1)]
                for label, draft in drafts:
                    draft_id = f"draft-{n}-{hash(draft) & 0xffff:x}"
                    parts.append(
                        "<div class='draft'>"
                        f"<b>{esc(label)}</b>"
                        f"<p id='{draft_id}'>{_draft_html(draft, it.platform)}</p>"
                        f"<button type='button' "
                        f"data-copy='{draft_id}'>Copy draft</button>"
                        "</div>")
            else:
                parts.append(
                    "<p class='muted'><b>No draft generated.</b> "
                    "This is intentional for sensitive, weak-fit, "
                    "low-confidence, or failed-analysis items.</p>")

            parts.append("</section>")

    parts.append("""
<script>
document.querySelectorAll("button[data-copy]").forEach(function(btn){
  btn.addEventListener("click", function(){
    var el = document.getElementById(btn.getAttribute("data-copy"));
    if (!el) return;
    var text = el.innerText || el.textContent || "";
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function(){
        btn.textContent = "Copied";
        setTimeout(function(){ btn.textContent = "Copy draft"; }, 1500);
      });
    } else {
      var ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); } catch(e) {}
      document.body.removeChild(ta);
      btn.textContent = "Copied";
      setTimeout(function(){ btn.textContent = "Copy draft"; }, 1500);
    }
  });
});
</script>""")

    parts.append(
        f"<footer><p class='muted'>Book: "
        f"<a href='{esc(BOOK_URL)}' target='_blank' rel='noopener'>"
        f"{esc(BOOK_FULL_TITLE)}</a></p></footer>")
    parts.append("</div></body></html>")

    path.write_text("".join(parts), encoding="utf-8")


def _draft_html(draft: str, platform: str) -> str:
    """Render draft safely. Markdown links stay as-is on non-plain platforms."""
    esc = html.escape(draft, quote=False)
    if platform in PLAIN_TEXT_PLATFORMS:
        return esc.replace("\n", "<br>")
    # Lightweight markdown: [text](url) and **bold** only
    esc = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r"<a href='\2' target='_blank' rel='noopener'>\1</a>",
        esc,
    )
    esc = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", esc)
    return esc.replace("\n", "<br>")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def _streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Leadership AI Digest v3.0", layout="wide")
    st.title("Leadership AI Daily Digest v3.0")
    st.caption(
        "Analysis-first. Drafts only for A_PROMOTE and B_ENGAGE. "
        "You review and publish manually."
    )

    with st.sidebar:
        st.subheader("Config")
        st.write(f"**Model:** `{GEMINI_MODEL}`")
        st.write(f"**Gemini key set:** {'✅' if GEMINI_API_KEY else '❌ MISSING'}")
        st.write(f"**Promote ≥** {PROMOTION_THRESHOLD} · "
                 f"**Engage ≥** {ENGAGEMENT_THRESHOLD}")
        st.write(f"**Min context:** {MIN_CONTEXT_CHARS} chars")

        history_upload = st.file_uploader(
            "History file (optional, from a previous run)",
            type=["txt"],
            help="Upload the seen_posts.txt you downloaded last time.",
        )
        if history_upload is not None:
            try:
                HISTORY_FILE.write_text(
                    history_upload.read().decode("utf-8", errors="replace"),
                    encoding="utf-8",
                )
                st.success("History loaded.")
            except OSError as exc:
                st.warning(f"Could not persist history: {exc}")

        ignore_history = st.checkbox("Ignore history this run", value=False)

    uploaded = st.file_uploader(
        "Posts JSON — array of {platform, title, url, snippet/full_content}",
        type=["json"],
    )
    if uploaded is None:
        st.info(
            "Upload a posts JSON to run the digest. "
            "Each item needs at minimum `platform`, `title`, `url`, and either "
            "`snippet` or `full_content`."
        )
        return

    try:
        rows = json.loads(uploaded.read().decode("utf-8", errors="replace"))
        if not isinstance(rows, list):
            st.error("JSON must be an array of post objects.")
            return
        content_items = [
            ContentItem(
                platform=str(r.get("platform", "LinkedIn")),
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                source=str(r.get("source", "")),
                published=str(r.get("published", "")),
                snippet=str(r.get("snippet", "")),
                full_content=str(r.get("full_content", r.get("snippet", ""))),
            )
            for r in rows
        ]
    except Exception as exc:
        st.error(f"Could not parse JSON: {exc}")
        return

    if not GEMINI_API_KEY:
        st.error(
            "GEMINI_API_KEY is not set. Add it in Streamlit Cloud → "
            "Settings → Secrets as `GEMINI_API_KEY = \"...\"`, then reboot."
        )
        return

    if not st.button("Run digest", type="primary"):
        return

    if ignore_history:
        globals()["IGNORE_HISTORY"] = True

    history = load_history()
    unique = dedupe_and_filter(content_items, history)
    selected = select_candidates(unique)

    if not selected:
        st.warning("No candidates passed the discovery filter. "
                   "Check relevance_threshold or add more substantial posts.")
        return

    try:
        client = GeminiClient(GEMINI_API_KEY, GEMINI_MODEL)
    except GeminiUnavailable as exc:
        st.error(f"Gemini unavailable: {exc}")
        return

    progress = st.progress(0.0, text="Starting…")
    status = st.empty()
    total = len(selected)

    for i, item in enumerate(selected, 1):
        status.write(f"[{i}/{total}] {item.platform} — {item.title[:70]}")
        analysis = analyse_post(client, item)
        if analysis is None:
            item.decision = Decision.D_SKIP
            item.decision_reason = item.analysis_error or "Analysis failed."
            item.confidence = Confidence.LOW
        else:
            item.decision = Decision(analysis["decision"])
            item.decision_reason = analysis["decision_reason"]
            item.context_summary = analysis["context_summary"]
            item.sensitivity = Sensitivity(analysis["sensitivity"])
            item.sensitivity_reason = analysis["sensitivity_reason"]
            item.confidence = Confidence(analysis["confidence"])
            item.book_theme = analysis["book_theme"]
            item.promotion_justification = analysis["promotion_justification"]
            item.book_alignment_score = analysis["book_alignment_score"]
            item.contextual_relevance_score = analysis["contextual_relevance_score"]
            item.engagement_score = analysis["engagement_score"]
            item.audience_alignment_score = analysis["audience_alignment_score"]
            item.promotion_naturalness_score = analysis["promotion_naturalness_score"]
            item.promotion_score = analysis["promotion_score"]
            if analysis.get("guardrail_override"):
                item.rejection_reason = analysis["guardrail_override"]
            if item.decision in (Decision.A_PROMOTE, Decision.B_ENGAGE):
                generate_drafts(client, item)
        progress.progress(i / total, text=f"{i}/{total}")
        time.sleep(min(GEMINI_DELAY_SECONDS, 1.0))

    import tempfile
    out_dir = Path(tempfile.mkdtemp(prefix="digest_"))

    grouped: dict[str, list[ContentItem]] = {}
    for p in PLATFORM_ORDER:
        grp = [x for x in selected if x.platform == p]
        if grp:
            grp.sort(key=lambda x: _parse_date(x.published), reverse=True)
            grouped[p] = grp

    counts = {d.value: sum(1 for x in selected if x.decision == d)
              for d in Decision}
    summary = (
        f"{len(selected)} candidate(s): "
        f"{counts['A_PROMOTE']} promote · {counts['B_ENGAGE']} engage · "
        f"{counts['C_SENSITIVE']} sensitive · {counts['D_SKIP']} skip."
    )
    digest = Digest(
        date=datetime.now().strftime("%Y-%m-%d"),
        total_items=len(selected),
        items_by_platform=grouped,
        summary=summary,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = out_dir / f"digest_{stamp}.html"
    md_path = out_dir / f"digest_{stamp}.md"
    json_path = out_dir / f"digest_{stamp}.json"
    write_html(digest, html_path)
    write_markdown(digest, md_path)
    write_json(digest, json_path)
    save_history(x.url for x in selected)

    status.success(f"Done — {summary}")
    st.subheader("Download")
    c1, c2, c3 = st.columns(3)
    c1.download_button("HTML", html_path.read_bytes(),
                       file_name=html_path.name, mime="text/html")
    c2.download_button("Markdown", md_path.read_bytes(),
                       file_name=md_path.name, mime="text/markdown")
    c3.download_button("JSON", json_path.read_bytes(),
                       file_name=json_path.name, mime="application/json")

    st.subheader("Preview")
    st.components.v1.html(html_path.read_text(encoding="utf-8"),
                          height=900, scrolling=True)

    if HISTORY_FILE.exists():
        st.download_button(
            "Download seen_posts.txt (keep for next run)",
            HISTORY_FILE.read_bytes(),
            file_name="seen_posts.txt", mime="text/plain",
        )


# ---------------------------------------------------------------------------
# Entry point — Streamlit when hosted, CLI otherwise
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    if st.runtime.exists():
        _streamlit_app()
except ImportError:
    pass

if __name__ == "__main__" and "streamlit" not in sys.modules:
    sys.exit(main())
