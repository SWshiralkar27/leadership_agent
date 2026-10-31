#!/usr/bin/env python3
"""
Leadership AI Daily Digest Generator  --  v2.2
(Shorts + X, hybrid/augmented-intelligence filters, per-platform minimum)

Fetch -> score/filter -> Gemini comment -> HTML / Markdown / JSON.
Human-in-the-loop: the tool drafts, the author reviews and posts.
"""
from __future__ import annotations
import streamlit as st
import aiohttp
import pandas as pd
import requests
import plotly.express as px
import google.generativeai as genai
import asyncio
import hashlib
import html
import json
import logging
import os
import re
import time
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional
import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
VERSION = "2.2"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

BOOK_NAME = "Games for Gravitas"
BOOK_FULL_TITLE = ("Games for Gravitas: A Playbook to Build Human Advantage "
                   "in the Hybrid Intelligence Era")
# YouTube / YouTube Shorts: the book is named in plain text ONLY (no hyperlink, no URL).
BOOK_NAME_YOUTUBE = ("Games for Gravitas: Building a Future-Ready Workforce "
                     "Through Game-Based Upskilling")
# One canonical link (v2.1 had two). If your old .env still sets BOOK_URL to the
# a.co link, delete that line or update it.
BOOK_LINK = os.getenv(
    "BOOK_URL",
    "https://www.amazon.in/Games-Gravitas-Playbook-Advantage-Intelligence-ebook/dp/B0FYNLLZ32",
)
AUTHOR_NAME = os.getenv("AUTHOR_NAME", "Shreekant Shiralkar")

DAYS_BACK = int(os.getenv("DAYS_BACK", "7"))
CSE_DAYS_BACK = int(os.getenv("CSE_DAYS_BACK", "30"))  # social posts index slowly
MAX_RESULTS_PER_SOURCE = int(os.getenv("MAX_RESULTS_PER_SOURCE", "7"))
MIN_ITEMS_PER_PLATFORM = int(os.getenv("MIN_ITEMS_PER_PLATFORM", "5"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))
RUN_ONCE = os.getenv("RUN_ONCE", "true").lower() in ("1", "true", "yes")
RUN_INTERVAL_HOURS = float(os.getenv("RUN_INTERVAL_HOURS", "24"))
INCLUDE_INSTAGRAM = os.getenv("INCLUDE_INSTAGRAM", "false").lower() in ("1", "true", "yes")
ENABLE_SCRAPE_FALLBACK = os.getenv("ENABLE_SCRAPE_FALLBACK", "false").lower() in ("1", "true", "yes")
GEMINI_DELAY_SECONDS = float(os.getenv("GEMINI_DELAY_SECONDS", "1.5"))
# Empty = use 70-90 words everywhere. Example: SHORT_FORM_WORDS=40-55
SHORT_FORM_WORDS = os.getenv("SHORT_FORM_WORDS", "").strip()
SHORTS_MAX_SECONDS = int(os.getenv("SHORTS_MAX_SECONDS", "60"))  # Shorts can run up to 3 min

SEARCH_TERMS = [t.strip() for t in os.getenv(
    "SEARCH_TERMS",
    "Leadership AI,AI in Leadership,Human-centered AI leadership,"
    "hybrid intelligence,augmented intelligence,skill development AI",
).split(",") if t.strip()]

MEDIUM_TAGS = [t.strip() for t in os.getenv(
    "MEDIUM_TAGS",
    "leadership-ai,artificial-intelligence,hybrid-intelligence,"
    "augmented-intelligence,skill-development,leadership",
).split(",") if t.strip()]

DESKTOP_PATH = Path(os.path.expanduser("~/Desktop/Leadership_AI_Digest"))
MODEL_CACHE_FILE = Path(".gemini_model_cache.json")
MODEL_CACHE_TTL_HOURS = 24

# (platform key, icon). Order = section order in the outputs.
PLATFORMS = [
    ("Medium", "📝"), ("YouTube", "▶️"), ("YouTube Shorts", "🎬"),
    ("LinkedIn", "💼"), ("Facebook", "📘"), ("X", "🐦"),
]
if INCLUDE_INSTAGRAM:
    PLATFORMS.append(("Instagram", "📸"))
PLATFORM_ORDER = [p for p, _ in PLATFORMS]
PLATFORM_ICON = dict(PLATFORMS)
PLATFORM_LABEL = {"X": "X (Twitter)"}

# Platforms whose comments are plain text (no bold, no links, _underscores_ only).
PLAIN_TEXT_PLATFORMS = {"YouTube", "YouTube Shorts"}
SHORT_FORM_PLATFORMS = {"YouTube Shorts", "X"}

BOOK_THEMES = [
    {"name": "Centaur leadership",
     "detail": "Humans and AI together out-think either one alone."},
    {"name": "Purposeful rehearsal",
     "detail": "Structured games let leaders rehearse high-stakes decisions safely."},
    {"name": "Playbook over theory",
     "detail": "Modular exercises that leaders actively apply, not just read about."},
    {"name": "Non-automatable skills",
     "detail": "Judgment, intuition, empathy, ethics and creativity stay human."},
    {"name": "Hybrid Intelligence Era",
     "detail": "AI is a strategic collaborator; humans and AI co-evolve."},
    {"name": "Structured human-AI teaming",
     "detail": "The 7-Phase Model for building blended human-AI teams."},
]

KEYWORDS = {
    "ai": 1.0, "artificial intelligence": 1.2, "leadership": 1.5, "leader": 1.3,
    "skill": 1.0, "development": 0.8, "management": 0.8, "executive": 0.9,
    "decision making": 1.2, "gravitas": 1.5, "ambiguity": 1.2, "rehearsal": 1.0,
    "coaching": 0.9,
    # added in v2.2
    "hybrid intelligence": 1.5, "augmented intelligence": 1.3, "human-ai": 1.0,
    "human-centered": 1.0, "upskilling": 0.9, "reskilling": 0.9, "learning": 0.7,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------------
@dataclass
class ContentItem:
    platform: str
    title: str
    url: str
    source: str = ""
    published: str = ""
    snippet: str = ""
    comment: str = ""
    relevance_score: float = 0.0
    backfilled: bool = False
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        self.title = clean_text(self.title)
        self.snippet = clean_text(self.snippet)
        self.source = clean_text(self.source)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Digest:
    date: str
    total_items: int
    items_by_platform: Dict[str, List[ContentItem]]
    summary: str
    generated_at: str

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "total_items": self.total_items,
            "items_by_platform": {p: [i.to_dict() for i in its]
                                  for p, its in self.items_by_platform.items()},
            "summary": self.summary,
            "generated_at": self.generated_at,
        }

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def clean_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def strip_html_to_text(raw: Optional[str]) -> str:
    if not raw:
        return ""
    return clean_text(BeautifulSoup(raw, "html.parser").get_text(separator=" "))


def truncate_at_boundary(text: str, limit: int) -> str:
    """Cut at the last space (never mid-word) and append an ellipsis."""
    text = text or ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-\u2013\u2014.!?") + "\u2026"


def is_valid_url(url: Optional[str]) -> bool:
    return bool(url and url.startswith(("http://", "https://")))


def ensure_complete_sentence(text: str) -> str:
    text = (text or "").strip()
    if text and text[-1] not in ".!?":
        text = text.rstrip(",;:-") + "."
    return text


# MUST stay identical to splitIntoSentences() in COPY_BUTTON_SCRIPT (invariant 7).
# v2.2: the lookahead also accepts a curly opening quote and Unicode bold/italic
# letters, because the Copy button splits AFTER converting markdown to Unicode.
_SENT_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z0-9"\'\u201C*_\[\U0001D400-\U0001D7FF\u210E])'
)
_ABBREV = re.compile(r"\b([A-Za-z]\.){1,}[A-Za-z]?$")


def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    for piece in _SENT_SPLIT.split(text):
        if out and _ABBREV.search(out[-1]):
            out[-1] += " " + piece
        else:
            out.append(piece)
    return [s.strip() for s in out if s.strip()]


def format_sentences_on_new_lines(text: str, separator: str = "\n\n") -> str:
    if not text:
        return text
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return text
    return separator.join(s.strip() for s in sentences if s.strip())


def calculate_relevance_score(item: ContentItem) -> float:
    text = f"{item.title} {item.snippet}".lower()
    score = 0.0
    for kw, weight in KEYWORDS.items():
        if kw == "ai":
            # word boundary, otherwise "said"/"again"/"training" all match
            if re.search(r"\bai\b", text):
                score += weight
        elif kw in text:
            score += weight
    score = min(score / 12.0, 1.0)
    if item.platform in ("Medium", "YouTube", "YouTube Shorts"):
        score *= 1.2
    return round(min(score, 1.0), 3)


def _parse_date(s: str) -> datetime:
    floor = datetime.min.replace(tzinfo=timezone.utc)
    if not s:
        return floor
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return floor
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _url_key(url: str) -> str:
    if "youtube.com/watch" in url:
        return url.rstrip("/")
    return url.split("#")[0].split("?")[0].rstrip("/")

# ----------------------------------------------------------------------------
# Fetchers
# ----------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (LeadershipAIDigest)"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def _fetch_feed(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=20)) as resp:
        resp.raise_for_status()
        return await resp.text()


async def fetch_medium_articles(session: aiohttp.ClientSession,
                                limit: int = MAX_RESULTS_PER_SOURCE) -> List[ContentItem]:
    logging.info("Fetching Medium (%d tags)...", len(MEDIUM_TAGS))
    items: List[ContentItem] = []
    seen = set()
    for tag in MEDIUM_TAGS:
        try:
            feed = feedparser.parse(await _fetch_feed(session, f"https://medium.com/feed/tag/{tag}"))
        except Exception as ex:
            logging.error("Medium tag %s error: %s", tag, ex)
            continue
        for e in feed.entries[:limit]:
            link = getattr(e, "link", "")
            if not is_valid_url(link) or _url_key(link) in seen:
                continue
            seen.add(_url_key(link))
            body = ""
            if getattr(e, "content", None):
                body = e.content[0].get("value", "")
            body = strip_html_to_text(body or getattr(e, "summary", ""))
            items.append(ContentItem("Medium", getattr(e, "title", ""), link,
                                     getattr(e, "author", ""), getattr(e, "published", ""),
                                     truncate_at_boundary(body, 300)))
    return items


def _iso_duration_seconds(d: str) -> int:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m:
        return 10 ** 6
    days, h, mi, s = (int(x or 0) for x in m.groups())
    return days * 86400 + h * 3600 + mi * 60 + s


def fetch_youtube_updates(limit: int = MAX_RESULTS_PER_SOURCE) -> List[ContentItem]:
    """Returns YouTube and YouTube Shorts items (Shorts = duration <= SHORTS_MAX_SECONDS, default 60 s)."""
    logging.info("Fetching YouTube + Shorts...")
    if not YOUTUBE_API_KEY:
        logging.warning("YouTube API key not set - skipping YouTube / Shorts.")
        return []
    published_after = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%dT%H:%M:%SZ")
    q = " | ".join(SEARCH_TERMS)
    try:
        yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        snippets: Dict[str, dict] = {}
        # Two searches (200 quota units): any duration, and "short" (<4 min) to
        # make sure Shorts candidates are present.
        for duration in ("any", "short"):
            resp = yt.search().list(
                q=q, part="snippet", type="video", order="date",
                publishedAfter=published_after, videoDuration=duration,
                maxResults=min(limit * 4, 50),
            ).execute()
            for it in resp.get("items", []):
                vid = it.get("id", {}).get("videoId")
                if vid and vid not in snippets:
                    snippets[vid] = it.get("snippet", {})
        durations: Dict[str, int] = {}
        ids = list(snippets)
        for i in range(0, len(ids), 50):
            det = yt.videos().list(part="contentDetails", id=",".join(ids[i:i + 50])).execute()
            for v in det.get("items", []):
                durations[v["id"]] = _iso_duration_seconds(v["contentDetails"].get("duration", ""))
        items: List[ContentItem] = []
        for vid, sn in snippets.items():
            is_short = durations.get(vid, 10 ** 6) <= SHORTS_MAX_SECONDS
            platform = "YouTube Shorts" if is_short else "YouTube"
            url = f"https://www.youtube.com/shorts/{vid}" if is_short else f"https://www.youtube.com/watch?v={vid}"
            items.append(ContentItem(platform, sn.get("title", ""), url, sn.get("channelTitle", ""),
                                     sn.get("publishedAt", ""), truncate_at_boundary(clean_text(sn.get("description", "")), 300)))
        return items
    except HttpError as e:
        logging.error("YouTube API error: %s", e)
        return []
    except Exception as e:  # network etc.
        logging.error("YouTube fetch failed: %s", e)
        return []


CSE_URL = "https://www.googleapis.com/customsearch/v1"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def _cse_request(session: aiohttp.ClientSession, query: str, num: int) -> dict:
    params = {"key": GOOGLE_CSE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query,
              "num": min(num, 10), "dateRestrict": f"d{CSE_DAYS_BACK}"}
    async with session.get(CSE_URL, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"CSE HTTP {resp.status}: {(await resp.text())[:200]}")
        return await resp.json()


def scrape_fallback(query: str) -> List[dict]:
    """Fragile last resort (div.yuRUbf). Off by default; may breach Google's terms."""
    try:
        r = requests.get("https://www.google.com/search", params={"q": query},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        return [{"title": a.get_text(), "link": a["href"], "snippet": ""}
                for a in soup.select("div.yuRUbf a") if a.get("href")]
    except Exception as ex:
        logging.error("scrape_fallback failed: %s", ex)
        return []


async def google_cse_search(session: aiohttp.ClientSession, query: str, platform: str,
                            limit: int = MAX_RESULTS_PER_SOURCE) -> List[ContentItem]:
    raw: List[dict] = []
    if GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
        try:
            data = await _cse_request(session, query, limit)
            raw = data.get("items", []) or []
        except Exception as ex:
            logging.error("%s CSE error: %s", platform, ex)
    else:
        logging.warning("Google CSE not configured - %s will be empty unless fallback is on.", platform)
    if not raw and ENABLE_SCRAPE_FALLBACK:
        logging.info("%s: falling back to scraping", platform)
        raw = await asyncio.to_thread(scrape_fallback, query)
    items = []
    for r in raw:
        link = r.get("link", "")
        if is_valid_url(link):
            items.append(ContentItem(platform, r.get("title", ""), link, "Google CSE",
                                     "", r.get("snippet", "")))
    return items


def _cse_query(domain: str) -> str:
    terms = " OR ".join(f'"{t}"' for t in SEARCH_TERMS)
    return f"({terms}) site:{domain}"


async def fetch_linkedin_updates(session, limit=MAX_RESULTS_PER_SOURCE):
    logging.info("Fetching LinkedIn...")
    return await google_cse_search(session, _cse_query("linkedin.com"), "LinkedIn", limit)


async def fetch_facebook_updates(session, limit=MAX_RESULTS_PER_SOURCE):
    logging.info("Fetching Facebook...")
    return await google_cse_search(session, _cse_query("facebook.com"), "Facebook", limit)


async def fetch_x_updates(session, limit=MAX_RESULTS_PER_SOURCE):
    logging.info("Fetching X (Twitter)...")
    # x.com and twitter.com are both indexed; query both in one request.
    q = _cse_query("x.com").replace("site:x.com", "(site:x.com OR site:twitter.com)")
    return await google_cse_search(session, q, "X", limit)


async def fetch_instagram_updates(session, limit=MAX_RESULTS_PER_SOURCE):
    logging.info("Fetching Instagram...")
    return await google_cse_search(session, _cse_query("instagram.com"), "Instagram", limit)


async def fetch_all_content() -> List[ContentItem]:
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_medium_articles(session),
            asyncio.to_thread(fetch_youtube_updates),  # googleapiclient is synchronous
            fetch_linkedin_updates(session),
            fetch_facebook_updates(session),
            fetch_x_updates(session),
        ]
        if INCLUDE_INSTAGRAM:
            tasks.append(fetch_instagram_updates(session))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    items: List[ContentItem] = []
    for r in results:
        if isinstance(r, Exception):
            logging.error("Fetcher failed: %s", r)
        else:
            items.extend(r)
    logging.info("Total items fetched: %d", len(items))
    return items

# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------
class GeminiModelManager:
    BASE = "https://generativelanguage.googleapis.com/v1beta"
    SKIP = ("embedding", "tts", "image", "vision", "aqa", "live", "audio", "imagen", "veo")

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name: Optional[str] = None

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

    def _get_cached_model(self) -> Optional[str]:
        try:
            data = json.loads(MODEL_CACHE_FILE.read_text())
            age = datetime.now() - datetime.fromisoformat(data["timestamp"])
            if age < timedelta(hours=MODEL_CACHE_TTL_HOURS):
                return data["model"]
        except Exception:
            pass
        return None

    def _save_cached_model(self, name: str) -> None:
        try:
            MODEL_CACHE_FILE.write_text(json.dumps({"model": name, "timestamp": datetime.now().isoformat()}))
        except Exception as ex:
            logging.warning("Could not write model cache: %s", ex)

    def discover_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.BASE}/models", headers=self._headers(), timeout=30)
            r.raise_for_status()
            names = []
            for m in r.json().get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    n = m["name"].split("/")[-1]
                    if not any(s in n for s in self.SKIP):
                        names.append(n)
            # prefer flash, then stable over preview/experimental
            names.sort(key=lambda n: (0 if "flash" in n else 1,
                                      1 if ("preview" in n or "exp" in n) else 0, n))
            return names
        except Exception as ex:
            logging.warning("Model discovery failed: %s", ex)
            return []

    def _get_known_models(self) -> List[str]:
        # Refresh when convenient; discovery normally supersedes this list.
        return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

    @staticmethod
    def _model_uses_thinking_budget(name: str) -> bool:
        return any(k in name for k in ("2.5", "3.0", "gemini-3", "thinking"))

    def _post(self, model: str, prompt: str, max_tokens: int, use_thinking: bool):
        gen = {"temperature": 0.7, "maxOutputTokens": max_tokens, "topP": 0.9, "topK": 40}
        if use_thinking:
            gen["thinkingConfig"] = {"thinkingBudget": 0}  # invariant 1
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
        r = requests.post(f"{self.BASE}/models/{model}:generateContent",
                          headers=self._headers(), json=body, timeout=90)
        try:
            data = r.json()
        except Exception:
            data = {}
        return r.status_code, data

    def test_model(self, name: str) -> bool:
        try:
            status, data = self._post(name, "Say 'test'", 10, self._model_uses_thinking_budget(name))
            if status == 400 and self._model_uses_thinking_budget(name):
                status, data = self._post(name, "Say 'test'", 10, False)
            return status == 200 and bool(data.get("candidates"))
        except Exception:
            return False

    def get_working_model(self) -> Optional[str]:
        if not self.api_key:
            logging.warning("GEMINI_API_KEY not set - fallback comments will be used.")
            return None
        candidates: List[str] = []
        cached = self._get_cached_model()
        if cached:
            candidates.append(cached)
        candidates += [m for m in self.discover_models() if m not in candidates]
        candidates += [m for m in self._get_known_models() if m not in candidates]
        for name in candidates[:12]:
            if self.test_model(name):
                self.model_name = name
                self._save_cached_model(name)
                logging.info("Using Gemini model: %s", name)
                return name
        logging.error("No working Gemini model found")
        return None

    def generate_content(self, prompt: str, max_tokens: int = 800) -> Optional[dict]:
        if not self.model_name:
            return None
        use_thinking = self._model_uses_thinking_budget(self.model_name)
        for attempt in range(4):
            try:
                status, data = self._post(self.model_name, prompt, max_tokens, use_thinking)
            except Exception as ex:
                logging.error("Gemini request error: %s", ex)
                time.sleep(2 ** attempt)
                continue
            if status == 400 and use_thinking:
                logging.warning("thinkingConfig rejected by API; retrying without it")
                use_thinking = False
                continue
            if status in (429, 500, 503):
                time.sleep(2 ** attempt * 2)
                continue
            if status != 200:
                logging.error("Gemini HTTP %s: %s", status, str(data)[:200])
                return None
            cands = data.get("candidates") or []
            if not cands:
                return {"text": "", "truncated": True, "finish_reason": "NO_CANDIDATE"}
            cand = cands[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            finish = cand.get("finishReason", "")
            return {"text": text, "truncated": finish == "MAX_TOKENS" or not text,
                    "finish_reason": finish}
        return None


model_manager: Optional[GeminiModelManager] = None


def _word_range(platform: str) -> tuple:
    if SHORT_FORM_WORDS and platform in SHORT_FORM_PLATFORMS:
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", SHORT_FORM_WORDS)
        if m:
            return int(m.group(1)), int(m.group(2))
    return 70, 90


def _book_ref_instruction(plain: bool) -> str:
    if plain:
        return ("PLAIN TEXT ONLY. No markdown, no bold, no links. Emphasise at most a phrase or two "
                "with _underscores_. There must be NO hyperlink and NO URL anywhere. Name the book only "
                f"as plain text, written exactly like this, without quotation marks: {BOOK_NAME_YOUTUBE}")
    return ("Use markdown: put **bold** around 1-2 key phrases, write the book title in italics and "
            f"as a link exactly like this: [{BOOK_NAME}]({BOOK_LINK}) . Use *single asterisks* for any other italics.")


def build_fallback_comment(item: ContentItem, topic: str, plain: bool) -> str:
    book = BOOK_NAME_YOUTUBE if plain else f"[{BOOK_NAME}]({BOOK_LINK})"
    templates = [
        "This raises a real question about {t}: humans paired with AI out-think either one alone. That centaur idea sits at the heart of {b}, where leaders learn to split the work between machine speed and human judgment. What would change in your team if you practised that split deliberately?",
        "This raises a real question about {t}: do leaders ever get to rehearse the hard calls before they arrive? In {b} I argue that structured games are a safe way to practise high-stakes decisions. Where do you see leaders getting that kind of practice today?",
        "This raises a real question about {t}: knowing a framework is not the same as using it under pressure. {b} is built as a playbook of modular exercises leaders apply, not theory they read. Which part of your own practice would you turn into an exercise first?",
        "This raises a real question about {t}: as AI takes over routine analysis, judgment, empathy and ethics become the scarce skills. {b} treats those non-automatable skills as the core of human advantage. Which of them is hardest to build in your experience?",
        "This raises a real question about {t}: AI is becoming a strategic collaborator, not just a tool. {b} explores how humans and AI co-evolve in the hybrid intelligence era. How are you changing the way you work with it?",
        "This raises a real question about {t}: blended human-AI teams need structure to work. {b} lays out a 7-Phase Model for exactly that kind of teaming. What has been the hardest handoff between people and AI in your teams?",
    ]
    idx = int(hashlib.md5(item.url.encode()).hexdigest(), 16) % len(BOOK_THEMES)  # invariant 5
    return templates[idx].format(t=topic, b=book)


def generate_comment_with_gemini(item: ContentItem) -> str:
    global model_manager
    if model_manager is None:
        model_manager = GeminiModelManager(GEMINI_API_KEY or "")
        model_manager.get_working_model()

    plain = item.platform in PLAIN_TEXT_PLATFORMS  # invariant 9: key off item.platform
    book_label = BOOK_NAME_YOUTUBE if plain else BOOK_NAME
    book_full = BOOK_NAME_YOUTUBE if plain else BOOK_FULL_TITLE
    topic = item.title.split("|")[0].strip() or truncate_at_boundary(item.title, 60)
    topic = truncate_at_boundary(topic, 60)
    lo, hi = _word_range(item.platform)
    themes = "\n".join(f"{i}. {t['name']} - {t['detail']}" for i, t in enumerate(BOOK_THEMES, 1))

    prompt = f"""You are {AUTHOR_NAME}, author of the book "{book_full}". Write ONE genuine comment you would post on the item below, in your own first-person voice.

POST
Platform: {item.platform}
Title: {item.title}
Source: {item.source}
Snippet: {item.snippet or '(none)'}

BOOK THEMES (choose exactly ONE, the single most relevant to THIS post; never stack themes):
{themes}

RULES
- {lo}-{hi} words, one paragraph, warm and peer-to-peer. No hype, no emojis, no hashtags, no "great post".
- Sentence 1 must respond to a specific idea from the post's title or snippet. Show you read it.
- Then connect that idea to your chosen theme with a cause-and-effect bridge (an insight, not a pitch), explained in fresh words.
- Mention the book "{book_label}" exactly once, naturally, as the source of the idea.
- Optional: end with one short question that invites replies. No canned closing line.
- Never attribute claims to the post's author that the title/snippet do not support. If the snippet is thin, stay with the title's idea.
- Every sentence must be complete and end with . ! or ?
- Do not wrap the comment in quotation marks.

FORMAT: {_book_ref_instruction(plain)}

Output ONLY the comment text."""

    draft = ""
    budget = 500
    for _ in range(4):  # invariant 2: escalating budget
        res = model_manager.generate_content(prompt, max_tokens=budget)
        if res and len(res["text"]) > 50 and not res["truncated"]:  # invariant 3
            draft = res["text"]
            break
        if res:
            logging.warning("Comment truncated/empty (finishReason=%s); retrying with a larger budget",
                            res["finish_reason"])
        budget = min(budget * 2, 4000)

    if draft and BOOK_NAME.lower() not in draft.lower():  # invariant 4
        revision = (f"Here is a draft comment:\n\n{draft}\n\nRevise it so it naturally mentions the book "
                    f"\"{book_label}\" exactly once, keeping the same length, voice and format. "
                    f"{_book_ref_instruction(plain)}\nOutput ONLY the revised comment.")
        res = model_manager.generate_content(revision, max_tokens=2000)
        if res and not res["truncated"] and BOOK_NAME.lower() in res["text"].lower():
            draft = res["text"]
        else:
            draft = ""

    if not draft:
        logging.warning("Using fallback comment for %s", item.url)
        return build_fallback_comment(item, topic, plain)

    draft = draft.strip().strip('"\u201c\u201d').strip()
    draft = re.sub(r"\s*\n+\s*", " ", draft)  # canonical single-line form
    if plain:  # invariant 13: no hyperlink on YouTube / Shorts, even if Gemini adds one
        draft = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", draft)
        draft = re.sub(r"https?://\S+", "", draft)
        draft = draft.replace("**", "").replace("*", "")
        draft = re.sub(r"\s{2,}", " ", draft).strip()
    return ensure_complete_sentence(draft)

# ----------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------
def process_content(items: List[ContentItem]) -> Digest:
    seen, unique = set(), []
    for it in items:
        k = _url_key(it.url)
        if k not in seen:
            seen.add(k)
            unique.append(it)
    for it in unique:
        it.relevance_score = calculate_relevance_score(it)

    selected: List[ContentItem] = []
    for platform in PLATFORM_ORDER:
        pool = sorted((i for i in unique if i.platform == platform),
                      key=lambda i: i.relevance_score, reverse=True)
        chosen = [i for i in pool if i.relevance_score >= RELEVANCE_THRESHOLD][:MAX_RESULTS_PER_SOURCE]
        misses = [i for i in pool if i.relevance_score < RELEVANCE_THRESHOLD]
        if misses:
            logging.info("%s near-misses: %s", platform,
                         "; ".join(f"{i.relevance_score:.2f} {truncate_at_boundary(i.title, 50)}" for i in misses[:15]))
        if len(chosen) < MIN_ITEMS_PER_PLATFORM:
            need = MIN_ITEMS_PER_PLATFORM - len(chosen)
            fill = [i for i in misses if i.relevance_score > 0][:need]
            for i in fill:
                i.backfilled = True
                logging.info("Backfilled %s: %.2f %s", platform, i.relevance_score, i.title)
            chosen += fill
        if len(chosen) < MIN_ITEMS_PER_PLATFORM:
            logging.warning("%s has only %d item(s) (target %d).", platform, len(chosen), MIN_ITEMS_PER_PLATFORM)
        selected.extend(chosen)
    logging.info("Filtered to %d relevant items", len(selected))

    for n, it in enumerate(selected, 1):
        if not it.comment or len(it.comment) < 30:
            logging.info("[%d/%d] Commenting on %s: %s", n, len(selected), it.platform, truncate_at_boundary(it.title, 60))
            it.comment = generate_comment_with_gemini(it)
            time.sleep(GEMINI_DELAY_SECONDS)

    grouped: Dict[str, List[ContentItem]] = {}
    for p in PLATFORM_ORDER:
        group = [i for i in selected if i.platform == p]
        if group:
            group.sort(key=lambda i: _parse_date(i.published), reverse=True)
            grouped[p] = group
    counts = ", ".join(f"{PLATFORM_LABEL.get(p, p)} {len(v)}" for p, v in grouped.items())
    summary = f"{len(selected)} items worth a comment today ({counts})." if selected else "No relevant items found today."
    return Digest(datetime.now().strftime("%Y-%m-%d"), len(selected), grouped, summary,
                  datetime.now().isoformat(timespec="seconds"))

# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------
def write_json(digest: Digest, path: Path) -> None:
    data = digest.to_dict()
    data["metadata"] = {"version": VERSION, "exported_at": datetime.now().isoformat(timespec="seconds"),
                        "total_items": digest.total_items, "book_name": BOOK_FULL_TITLE,
                        "book_link": BOOK_LINK}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(digest: Digest, path: Path) -> None:
    lines = ["# Leadership AI Daily Digest", f"**Date:** {digest.date}",
             f"**Total Items:** {digest.total_items}", "---", "## Summary", digest.summary, ""]
    for platform, its in digest.items_by_platform.items():
        lines.append(f"## {PLATFORM_LABEL.get(platform, platform)}")
        for n, it in enumerate(its, 1):
            lines += [f"### {n}. [{it.title}]({it.url})",
                      f"**Source:** {it.source}", f"**Published:** {it.published}",
                      f"**Preview:** {truncate_at_boundary(it.snippet, 150)}",
                      "**Comment:**", format_sentences_on_new_lines(it.comment), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def comment_to_html(comment: str, plain: bool) -> str:
    paras = []
    for s in split_into_sentences(comment):
        t = html.escape(s, quote=False)
        t = re.sub(r'"([^"]+)"', "\u201c\\1\u201d", t)  # before tags are inserted
        if plain:
            t = re.sub(r"(^|[\s(])_([^_]+)_(?=[\s.,!?;:)]|$)", r"\1<em>\2</em>", t)
        else:
            t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
                       r'<a href="\2" target="_blank" rel="noopener">\1</a>', t)
            t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
            t = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", t)
        paras.append(f"<p>{t}</p>")
    return "".join(paras)


# MUST stay identical to the Python splitter (invariant 7).
COPY_BUTTON_SCRIPT = r"""
function splitIntoSentences(text) {
  var pieces = text.split(/(?<=[.!?])\s+(?=[A-Z0-9"'\u201C*_\[\uD835\u210E])/);
  var abbrev = /\b([A-Za-z]\.){1,}[A-Za-z]?$/;
  var out = [];
  for (var i = 0; i < pieces.length; i++) {
    if (out.length && abbrev.test(out[out.length - 1])) out[out.length - 1] += ' ' + pieces[i];
    else out.push(pieces[i]);
  }
  return out.map(function (s) { return s.trim(); }).filter(Boolean);
}
function toUnicodeVariant(str, style) {
  var out = '';
  for (var ch of str) {
    var c = ch.codePointAt(0), cp = null;
    if (c >= 65 && c <= 90) cp = (style === 'bold' ? 0x1D400 : 0x1D434) + (c - 65);
    else if (c >= 97 && c <= 122) {
      if (style === 'italic' && c === 104) cp = 0x210E;
      else cp = (style === 'bold' ? 0x1D41A : 0x1D44E) + (c - 97);
    } else if (c >= 48 && c <= 57 && style === 'bold') cp = 0x1D7CE + (c - 48);
    out += (cp === null) ? ch : String.fromCodePoint(cp);
  }
  return out;
}
function markdownToClipboardText(raw) {
  var t = raw;
  t = t.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '$1 ($2)');
  t = t.replace(/\*\*([^*]+)\*\*/g, function (m, a) { return toUnicodeVariant(a, 'bold'); });
  t = t.replace(/\*([^*]+)\*/g, function (m, a) { return toUnicodeVariant(a, 'italic'); });
  t = t.replace(/(^|[\s(])_([^_]+)_(?=[\s.,!?;:)]|$)/g, function (m, p, a) { return p + toUnicodeVariant(a, 'italic'); });
  t = t.replace(/"([^"]+)"/g, '\u201C$1\u201D');
  return splitIntoSentences(t).join('\n\n');
}
function fallbackCopy(text) {
  var ta = document.createElement('textarea');
  ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.select();
  var ok = false;
  try { ok = document.execCommand('copy'); } catch (e) {}
  document.body.removeChild(ta);
  return ok;
}
function copyComment(btn) {
  var text = markdownToClipboardText(btn.getAttribute('data-comment') || '');
  var done = function (ok) {
    var old = btn.textContent;
    btn.textContent = ok ? 'Copied \u2713' : 'Copy failed';
    setTimeout(function () { btn.textContent = old; }, 1600);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(function () { done(true); },
      function () { done(fallbackCopy(text)); });
  } else { done(fallbackCopy(text)); }
}
"""

HTML_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1f2933;--muted:#667085;--line:#e4e7ec;--accent:#1d4ed8;--sum:#e8f0fe;--amber:#f59e0b;--btn:#1d4ed8;--btntext:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#101418;--card:#1a2027;--text:#e6e9ee;--muted:#9aa4b2;--line:#2b333c;--accent:#7aa2ff;--sum:#1b2a44;--btn:#3b6df0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.55}
.wrap{max-width:900px;margin:0 auto;padding:24px 16px 56px}
header h1{margin:0 0 4px;font-size:1.7rem}
header .meta{color:var(--muted);font-size:.9rem}
.summary{background:var(--sum);border-radius:12px;padding:14px 18px;margin:18px 0 8px}
h2{display:flex;align-items:center;gap:10px;margin:32px 0 12px;font-size:1.25rem}
.badge{background:var(--accent);color:#fff;border-radius:999px;font-size:.75rem;padding:2px 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:0 0 14px;transition:box-shadow .15s}
.card:hover{box-shadow:0 4px 16px rgba(0,0,0,.08)}
.num{color:var(--muted);font-size:.8rem}
.card h3{margin:2px 0 4px;font-size:1.05rem}
.card h3 a{color:var(--accent);text-decoration:none}
.card h3 a:hover{text-decoration:underline}
.src{color:var(--muted);font-size:.85rem}
.snip{color:var(--muted);font-size:.9rem;margin:8px 0}
.near{background:var(--line);border-radius:6px;font-size:.7rem;padding:1px 6px;margin-left:6px}
.comment{border-left:4px solid var(--amber);padding:2px 0 2px 14px;margin:12px 0 10px}
.comment p{margin:0 0 .7em}
button.copy{background:var(--btn);color:var(--btntext);border:0;border-radius:8px;padding:7px 14px;cursor:pointer;font-size:.85rem}
button.copy:hover{opacity:.9}
footer{margin-top:40px;color:var(--muted);font-size:.9rem;text-align:center}
footer a{color:var(--accent)}
@media (max-width:640px){header h1{font-size:1.4rem}.card{padding:14px}}
"""


def write_html(digest: Digest, path: Path) -> None:
    esc = lambda s: html.escape(s or "", quote=True)
    out = ["<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width, initial-scale=1'>",
           f"<title>Leadership AI Digest {esc(digest.date)}</title><style>{HTML_CSS}</style></head><body><div class='wrap'>",
           f"<header><h1>Leadership AI Daily Digest</h1><div class='meta'>{esc(digest.date)} &middot; "
           f"{digest.total_items} items &middot; generated {esc(digest.generated_at)}</div></header>",
           f"<div class='summary'>{esc(digest.summary)}</div>"]
    for platform, its in digest.items_by_platform.items():
        out.append(f"<h2><span>{PLATFORM_ICON.get(platform, '')}</span>{esc(PLATFORM_LABEL.get(platform, platform))}"
                   f"<span class='badge'>{len(its)}</span></h2>")
        plain = platform in PLAIN_TEXT_PLATFORMS
        for n, it in enumerate(its, 1):
            near = "<span class='near'>near match</span>" if it.backfilled else ""
            out.append(
                f"<div class='card'><div class='num'>Post {n}{near}</div>"
                f"<h3><a href='{esc(it.url)}' target='_blank' rel='noopener'>{esc(it.title)}</a></h3>"
                f"<div class='src'>{esc(it.source)} &middot; {esc(it.published)} &middot; score {it.relevance_score:.2f}</div>"
                f"<div class='snip'>{esc(truncate_at_boundary(it.snippet, 220))}</div>"
                f"<div class='comment'>{comment_to_html(it.comment, plain)}</div>"
                f"<button class='copy' onclick='copyComment(this)' data-comment=\"{html.escape(it.comment, quote=True)}\">Copy</button></div>")
    out.append(f"<footer>Comments promote <a href='{esc(BOOK_LINK)}' target='_blank' rel='noopener'>"
               f"{esc(BOOK_FULL_TITLE)}</a></footer></div><script>{COPY_BUTTON_SCRIPT}</script></body></html>")
    path.write_text("".join(out), encoding="utf-8")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
async def run_once() -> None:
    global model_manager
    start = time.time()
    DESKTOP_PATH.mkdir(parents=True, exist_ok=True)
    model_manager = GeminiModelManager(GEMINI_API_KEY or "")
    await asyncio.to_thread(model_manager.get_working_model)

    items = await fetch_all_content()
    digest = await asyncio.to_thread(process_content, items)

    stem = DESKTOP_PATH / f"leadership_digest_{digest.date}"
    write_json(digest, stem.with_suffix(".json"))
    write_markdown(digest, stem.with_suffix(".md"))
    write_html(digest, stem.with_suffix(".html"))
    logging.info("Digest generation complete in %.0f s -> %s", time.time() - start, DESKTOP_PATH)
    try:
        webbrowser.open(stem.with_suffix(".html").resolve().as_uri())
    except Exception:
        pass


async def main() -> None:
    if RUN_ONCE:
        await run_once()
        return
    while True:
        try:
            await run_once()
        except Exception as ex:
            logging.exception("Run failed: %s", ex)
        await asyncio.sleep(RUN_INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    asyncio.run(main())
