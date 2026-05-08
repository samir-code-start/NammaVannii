"""engine.py — Namma Vanni AI pipeline: STT, LLM analysis, TTS, feedback logging, mock mode.

Architecture:
  STT Layer 1 → Sarvam AI (primary, with translate)
  STT Layer 2 → Edge STT fallback (if Sarvam fails)
  LLM Confirmation Layer 1 → LLM-based intent classification (contextual)
  LLM Confirmation Layer 2 → Keyword matching fallback (multilingual)
  LLM Confirmation Layer 3 → Re-record prompt (if both layers fail)
"""

import asyncio
import concurrent.futures
import csv
import json
import logging
import os
import re
import string
import requests
import tempfile
import uuid
from datetime import datetime

import edge_tts
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Secrets: support both .env (local) and st.secrets (Streamlit Cloud)
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str = "") -> str:
    """Fetch secret from environment or Streamlit secrets store."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOCK_MODE: bool = os.getenv("MOCK_MODE", "False").lower() == "true"
SARVAM_API_KEY: str = _get_secret("SARVAM_API_KEY")
GROQ_API_KEY: str = _get_secret("GROQ_API_KEY")

# Sarvam API endpoints
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text-translate"   # STT + translate → English
SARVAM_STT_ORIGINAL_URL = "https://api.sarvam.ai/speech-to-text"    # STT only → native lang
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

# Groq LLM
GROQ_LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "llama-3.3-70b-versatile"

# Timeout values (seconds)
SARVAM_STT_TIMEOUT = 25   # Primary STT timeout before triggering fallback
EDGE_STT_TIMEOUT = 30     # Edge STT fallback timeout

# Kannada common ASR correction map
KANADA_FIXES = {
    "ನಮ್ವ": "ನಮ್ಮ",
    "ವಣಿ": "ವಾಣಿ",
    "ತುಂಬಾ": "ತುಂಬಾ",
    "ಹಾಳಾಗಿದೆ": "ಹಾಳಾಗಿದೆ",
    "ರಸ್ತೆ": "ರಸ್ತೆ",
    "ಪಾಣಿ": "ಪಾಣಿ",
    "ಕೇಂದ್ರ": "ಕೇಂದ್ರ",
    "ಫೋನ್": "ಫೋನ್",
    "ಬೇಕು": "ಬೇಕು",
    "ಸೇವೆ": "ಸೇವೆ",
}

FEEDBACK_FILE = "/tmp/feedback.csv"  # /tmp persists within session on Streamlit Cloud
FEEDBACK_HEADERS = [
    "timestamp", "language", "raw_text", "original_text", "ai_issue",
    "confidence", "sentiment", "citizen_response",
    "agent_correction", "handover", "feedback_weight", "stt_source",
]

# Edge TTS voice mapping (fallback)
TTS_VOICE_MAP: dict[str, str] = {
    "kn": "kn-IN-VarunNeural",
    "hi": "hi-IN-MadhurNeural",
    "en": "en-IN-NeerjaNeural",
}
TTS_FALLBACK_VOICE = "en-IN-NeerjaNeural"
# Removed fixed TTS_OUTPUT_PATH, using dynamic paths in generate_tts

# Sarvam TTS language code mapping
SARVAM_TTS_LANG_MAP = {
    "kn": "kn-IN",
    "hi": "hi-IN",
    "en": "en-IN",
}

# ---------------------------------------------------------------------------
# Keyword dictionaries for multilingual confirmation fallback (Layer 2)
# ---------------------------------------------------------------------------

# YES keywords: Kannada, Hindi, English
_YES_KEYWORDS = {
    # English
    "yes", "yeah", "yep", "correct", "right", "sure", "true", "ok", "okay",
    "affirmative", "absolutely", "definitely", "indeed",
    # Hindi
    "haan", "ha", "sahi", "bilkul", "theek", "thik", "haa", "ji", "ji ha",
    "sahi hai", "haan ji", "bilkul sahi", "correct hai",
    # Kannada
    "houdu", "houdhu", "sari", "sariya", "haan", "aho", "idu sari",
    "sari ide", "houdu sari",
}

# NO keywords: Kannada, Hindi, English
_NO_KEYWORDS = {
    # English
    "no", "nope", "wrong", "incorrect", "false", "not right", "negative",
    "not correct", "that's wrong", "not that",
    # Hindi
    "nahi", "nai", "galat", "nahi hai", "sahi nahi", "yeh nahi",
    "bilkul nahi", "nahi ji", "galat hai",
    # Kannada
    "illa", "illaa", "beda", "thumba beda", "tappagide", "tappu",
    "sari illa", "houdu alla", "beda beda",
}

# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------
_MOCK_TRANSCRIPT = "ನಮ್ಮ ಊರಿ ರಸ್ತೆ ತುಂಬಾ ಕೆಟ್ಟಿದೆ, ಅಧಿಕಾರಿಗಳನ್ನು ಕಳುಹಿಸಿ"

_MOCK_ANALYSIS: dict = {
    "language": "kn",
    "normalized_issue": "The road in the caller's village is severely damaged and needs official inspection.",
    "confidence": 0.92,
    "sentiment": "urgent",
    "verification_prompt": "Your village road is badly damaged and you need officials to visit. Did I understand correctly? Say Yes or No.",
    "handover": False,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Namma Vanni, an expert AI analyst for Karnataka's 1092 Civic Helpline.

🎯 CORE TASKS:
1. DOMAIN TAXONOMY: Map fragmented speech to issues like water leakage, street lights, garbage, etc.
2. SEMANTIC EXTRACTION: Extract PRIMARY issue. Partial understanding > failure.
3. CONFIDENCE RUBRIC:
   - High (0.85-1.0): Clear issue + location
   - Medium (0.60-0.84): Clear issue, missing location
   - Low (0.30-0.59): Ambiguous or dialect-heavy
   - Critical (<0.30): Incoherent
4. DYNAMIC VERIFICATION: Generate a SPECIFIC clarification question tailored to the issue.
5. SENTIMENT DETECTION: Analyze tone, urgency.

OUTPUT STRICT JSON ONLY:
{
  "language": "en|kn|hi",
  "confidence": 0.0-1.0,
  "sentiment": "calm|confused|urgent|distressed|angry|fear",
  "normalized_issue": "Clean 1-line summary",
  "verification_prompt": "Natural question clarifying the specific issue. End with: 'Did I understand correctly? Say Yes or No.' Max 20 words.",
  "handover": true|false
}

DIALECT AWARENESS (Karnataka):
- North Karnataka dialects: "enu" → "ēnu" (what), "barri" → "banni" (come)
- Bangalore Urban: Code-mixed Kannada-English ("current hogide" = power cut)
- Old Mysuru: Formal Kannada with "appa/amma" honorifics
- Hindi-belt migrants: Hinglish mixed with Kannada words
- Common civic expressions:
  "current hogide" = power cut, "neer bandilla" = no water supply
  "gutter overflow" = drainage blockage, "kasa collect aagilla" = garbage not collected
  "road kharab" = road damaged, "light illa" = no street light

ISSUE CATEGORIES (Karnataka 1092):
- ROAD: potholes, damaged roads, flooding, waterlogging
- WATER: supply disruption, contamination, leakage, bore well
- ELECTRICITY: power cuts, street lights, transformer failure
- GARBAGE: collection missed, illegal dumping, burning
- DRAINAGE: overflow, blockage, sewage leak
- SAFETY: crime, harassment, emergency, accidents
- GOVERNMENT: corruption, missing services, documentation issues

GUARDRAILS:
- If confidence < 0.7 -> handover=true
- If sentiment in [distressed, angry, fear] -> handover=true"""


# ---------------------------------------------------------------------------
# Groq LLM helper (OpenAI-compatible API with Bearer token)
# ---------------------------------------------------------------------------

def _sarvam_chat(messages: list, temperature: float = 0.1, max_tokens: int = 500) -> str:
    """Make a chat completion call to Groq LLM endpoint (llama-3.3-70b-versatile)."""
    res = requests.post(
        GROQ_LLM_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_VALID_LANGUAGES = {"kn", "hi", "en"}
_VALID_SENTIMENTS = {"calm", "confused", "urgent", "distressed", "angry", "fear"}


def _strip_fences(raw: str) -> str:
    """Remove markdown/code fences from a raw string."""
    return re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()


def _enforce_guardrails(data: dict) -> dict:
    """Validate schema fields and enforce handover guardrail rules."""
    language = data.get("language", "en")
    if language not in _VALID_LANGUAGES:
        language = "en"

    sentiment = data.get("sentiment", "calm")
    if sentiment not in _VALID_SENTIMENTS:
        sentiment = "calm"

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    normalized_issue = str(data.get("normalized_issue", "Unclear report. Needs agent clarification.")).strip()
    if not normalized_issue or normalized_issue.lower() == "issue could not be determined":
        normalized_issue = "Unclear report. Needs agent clarification."

    verification_prompt = str(
        data.get(
            "verification_prompt",
            "Could you please repeat your issue? Did I understand correctly? Say Yes or No.",
        )
    ).strip()

    handover: bool = bool(data.get("handover", False))
    if confidence < 0.7:
        handover = True
    elif sentiment in {"distressed", "angry", "fear"} or normalized_issue == "Unclear report. Needs agent clarification.":
        handover = True

    return {
        "language": language,
        "normalized_issue": normalized_issue,
        "confidence": confidence,
        "sentiment": sentiment,
        "verification_prompt": verification_prompt,
        "handover": handover,
    }


async def _tts_coroutine(text: str, voice: str, output_path: str) -> None:
    """Async coroutine: synthesise speech with edge-tts and save to disk."""
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(output_path)


def translate_to_english(text: str) -> str:
    """Translates any text to English via LLM. Safe fallback to original."""
    if not text.strip():
        return ""
    try:
        return _sarvam_chat(
            messages=[{"role": "user", "content": f"Translate to English ONLY: '{text}'"}],
            temperature=0.1,
            max_tokens=100,
        )
    except Exception:
        return text  # Fail-safe: return original text unchanged


# ===========================================================================
# STT PIPELINE — Two-Layer Architecture
# ===========================================================================

def _sarvam_stt_with_translate(audio_path: str) -> tuple[str, str]:
    """
    STT Layer 1 — Sarvam AI (Primary).
    
    Calls Sarvam /speech-to-text-translate endpoint which:
      - Transcribes audio
      - Auto-detects language
      - Returns English translation directly
    
    Returns: (english_text, detected_language_code)
    Raises exception on failure — caller handles fallback trigger.
    """
    logger.info("[STT L1] Attempting Sarvam AI STT+Translate for: %s", audio_path)

    with open(audio_path, "rb") as f:
        res = requests.post(
            SARVAM_STT_URL,
            files={"file": ("audio.wav", f, "audio/wav")},
            headers={"api-subscription-key": SARVAM_API_KEY},
            timeout=SARVAM_STT_TIMEOUT,  # Timeout triggers fallback to Edge STT
        )

    res.raise_for_status()
    data = res.json()

    english_text = (data.get("transcript") or data.get("text") or "").strip()

    # Detect original language for TTS voice routing
    raw_lang = (
        data.get("language_code") or data.get("detected_language") or "kn"
    ).split("-")[0][:2]

    if raw_lang not in _VALID_LANGUAGES:
        raw_lang = "kn"

    logger.info("[STT L1 SUCCESS] Sarvam — lang=%s, chars=%d", raw_lang, len(english_text))
    return english_text, raw_lang


def _sarvam_stt_original(audio_path: str) -> str:
    """
    Get original-language transcript via Sarvam /speech-to-text (non-translate).
    Used to preserve the native language text for display on the handover/verify screens.
    Returns empty string if Sarvam STT original fails.
    """
    try:
        with open(audio_path, "rb") as f:
            res = requests.post(
                SARVAM_STT_ORIGINAL_URL,
                files={"file": ("audio.wav", f, "audio/wav")},
                headers={"api-subscription-key": SARVAM_API_KEY},
                data={"language_code": "unknown"},
                timeout=20,
            )
        res.raise_for_status()
        data = res.json()
        original = (data.get("transcript") or data.get("text") or "").strip()
        logger.info("[SARVAM STT ORIGINAL] Got %d chars of native transcript", len(original))
        return original
    except Exception as e:
        logger.warning("[SARVAM STT ORIGINAL] Failed: %s — no native transcript available", e)
        return ""


def _edge_stt_fallback(audio_path: str) -> tuple[str, str]:
    """
    STT Layer 2 — Edge STT Fallback.
    
    Triggered when Sarvam AI STT fails (timeout, API error, etc.)
    Uses faster-whisper or a basic speech recognition library as backup.
    
    Since edge-tts is TTS-only and edge doesn't have a universal free STT,
    we use the Groq Whisper endpoint as the actual fallback STT engine.
    This keeps the "Edge" branding while being practically robust.
    
    Returns: (english_text, detected_language_code)
    """
    logger.info("[STT L2] Fallback triggered — attempting Edge/Whisper STT for: %s", audio_path)

    try:
        # Attempt Groq Whisper as the Edge STT fallback
        # Groq offers Whisper-large-v3 via their API — highly reliable
        whisper_url = "https://api.groq.com/openai/v1/audio/transcriptions"

        with open(audio_path, "rb") as f:
            res = requests.post(
                whisper_url,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json",  # Includes language detection
                    # "language" omitted → Whisper auto-detects; sending None serialises as "None"
                },
                timeout=EDGE_STT_TIMEOUT,
            )
        res.raise_for_status()
        result = res.json()

        raw_text = (result.get("text") or "").strip()
        detected_lang = (result.get("language") or "en").lower()[:2]

        # Map Whisper's language codes to our system's codes
        lang_remap = {
            "ka": "kn",  # Whisper uses 'ka' for Kannada; we use 'kn'
        }
        detected_lang = lang_remap.get(detected_lang, detected_lang)
        if detected_lang not in _VALID_LANGUAGES:
            detected_lang = "kn"

        # If the text came back in a non-English language, translate it via LLM
        english_text = raw_text
        if detected_lang != "en" and raw_text:
            logger.info("[STT L2] Translating Edge STT output (%s) to English…", detected_lang)
            english_text = translate_to_english(raw_text)

        logger.info("[STT L2 SUCCESS] Edge/Whisper — lang=%s, chars=%d", detected_lang, len(english_text))
        return english_text, detected_lang

    except Exception as e:
        # Edge STT also failed — return empty gracefully
        logger.error("[STT L2 FAIL] Edge STT fallback also failed: %s", e)
        return "", "kn"


def transcribe_audio(audio_path: str) -> tuple[str, str, str]:
    """
    Main STT entry point — implements two-layer STT architecture.
    
    Layer 1: Sarvam AI (primary — best for Indian languages)
    Layer 2: Edge/Whisper STT (fallback — activated on Sarvam failure)
    
    Returns: (english_text, detected_language_code, original_native_transcript)
    """
    # Mock mode: return deterministic test data
    if MOCK_MODE or os.getenv("MOCK_MODE", "").lower() == "true":
        logger.info("[STT MOCK] Returning mock transcript.")
        return "The road in our village is very bad, please send officials", "kn", _MOCK_TRANSCRIPT

    logger.info("[STT] Starting transcription pipeline for: %s", audio_path)

    # Basic file validation
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) < 50:
        logger.warning("[STT] Audio file missing or too small — skipping.")
        return "", "en", ""

    stt_source = "sarvam"  # Track which STT layer was used for logging/debugging

    # ── Layer 1: Sarvam AI STT ──────────────────────────────────────────────
    try:
        english_text, detected_lang = _sarvam_stt_with_translate(audio_path)

        # Also fetch the original native-language transcript for UI display
        original_text = _sarvam_stt_original(audio_path)

        logger.info("[STT SOURCE] Sarvam AI used successfully.")
        return english_text, detected_lang, original_text

    except requests.exceptions.Timeout:
        # Timeout is the most common Sarvam failure — log clearly and fall through
        logger.warning(
            "[STT FALLBACK] Sarvam timed out after %ds — switching to Edge STT.", SARVAM_STT_TIMEOUT
        )
        stt_source = "edge_fallback"

    except requests.exceptions.HTTPError as e:
        # HTTP errors (4xx/5xx from Sarvam) — could be API key, quota, service down
        logger.warning("[STT FALLBACK] Sarvam HTTP error (%s) — switching to Edge STT.", e)
        stt_source = "edge_fallback"

    except Exception as e:
        # Any other unexpected error (network, parsing, etc.)
        logger.warning("[STT FALLBACK] Sarvam unexpected error (%s) — switching to Edge STT.", e)
        stt_source = "edge_fallback"

    # ── Layer 2: Edge STT (fallback) ────────────────────────────────────────
    # Reached here only if Sarvam failed above
    logger.info("[STT L2] Activating Edge STT fallback layer…")
    english_text, detected_lang = _edge_stt_fallback(audio_path)

    # For Edge STT we don't have a separate original-language endpoint,
    # so use the same text as original (it may be English or the raw detected language)
    original_text = english_text if detected_lang == "en" else ""

    logger.info("[STT PIPELINE DONE] Source=%s, lang=%s, chars=%d", stt_source, detected_lang, len(english_text))
    return english_text, detected_lang, original_text


# ===========================================================================
# CONFIRMATION PIPELINE — Three-Layer Architecture
# ===========================================================================

def _normalize_for_keyword_match(text: str) -> str:
    """
    Normalize text for keyword matching:
      - Lowercase
      - Remove punctuation
      - Collapse whitespace
    Handles multilingual text safely.
    """
    text = text.lower().strip()
    # Remove common punctuation while preserving non-ASCII (Kannada/Hindi scripts)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_match_confirmation(transcript: str) -> str | None:
    """
    Confirmation Layer 2 — Keyword Matching Fallback.
    
    Activated when the LLM confirmation layer (Layer 1) fails or returns an error.
    Checks the normalized transcript against multilingual YES/NO keyword dictionaries.
    
    Scoring logic:
      - Check if any YES keyword appears as a substring in the normalized text
      - Check if any NO keyword appears as a substring in the normalized text
      - First match wins (order: YES → NO)
      - Returns "YES", "NO", or None (if no keywords matched → triggers Layer 3)
    
    Supports: Kannada, Hindi, English keywords
    """
    normalized = _normalize_for_keyword_match(transcript)
    logger.info("[CONFIRM L2] Keyword matching on: '%s'", normalized[:80])

    # Check YES keywords first
    for kw in _YES_KEYWORDS:
        if kw in normalized:
            logger.info("[CONFIRM L2] YES keyword matched: '%s'", kw)
            return "YES"

    # Then check NO keywords
    for kw in _NO_KEYWORDS:
        if kw in normalized:
            logger.info("[CONFIRM L2] NO keyword matched: '%s'", kw)
            return "NO"

    # No keywords matched
    logger.info("[CONFIRM L2] No keyword matched — returning None (triggers Layer 3).")
    return None


def _llm_confirm_intent(transcript: str) -> dict | None:
    """
    Confirmation Layer 1 — LLM-Based Intent Classification (Primary).
    
    Uses LLaMA via Groq to contextually understand the citizen's response.
    Goes beyond keyword matching — understands:
      - Affirmative phrasing in any language
      - Rejection or correction phrasing
      - Partial confirmations ("yes but ambulance not police")
      - Uncertain or unclear responses
    
    Returns dict with keys: {intent, summary} or None if LLM call fails.
    Intent values: "confirmed" | "denied" | "partial" | "unclear"
    """
    system_prompt = (
        "You are an intent classifier for a multilingual civic helpline (Namma Vanni 1092). "
        "A citizen has just listened to an AI-generated summary of their complaint and replied verbally. "
        "Their reply may be in Kannada, Hindi, English, or a mixture of all three.\n\n"
        "Your job: extract the citizen's intent from their reply and return ONLY a JSON object — "
        "no explanation, no markdown, no extra text.\n\n"
        "Intent rules:\n"
        "  confirmed  → citizen agrees the summary is correct "
        "(e.g. 'yes', 'haan', 'ಹೌದು', 'correct', 'sahi hai', 'bilkul')\n"
        "  denied     → citizen says the summary is wrong or wants to re-explain "
        "(e.g. 'no', 'nahi', 'ಇಲ್ಲ', 'galat', 'wrong', 'that is not right')\n"
        "  partial    → citizen says it is partly right but needs a correction "
        "(e.g. 'yes but...', 'almost', 'haan par...', 'ಆದರೆ', 'thoda alag', 'not fully')\n"
        "  unclear    → reply is inaudible, irrelevant, or impossible to classify\n\n"
        "Examples:\n"
        "  'haan sahi hai' → confirmed\n"
        "  'no ambulance nahi police chahiye' → denied\n"
        "  'samajh nahi aaya' → unclear\n"
        "  'yes but address is different' → partial\n\n"
        "Response format (strict JSON, nothing else):\n"
        '{"intent": "confirmed" | "denied" | "partial" | "unclear", '
        '"summary": "<one concise sentence describing what the citizen said>"}'
    )

    try:
        raw = _sarvam_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=0.0,  # Deterministic — this is a classification task
            max_tokens=100,
        )
        cleaned = _strip_fences(raw)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            intent = parsed.get("intent", "unclear")
            if intent not in ("confirmed", "denied", "partial", "unclear"):
                intent = "unclear"
            logger.info("[CONFIRM L1 SUCCESS] LLM intent: %s", intent)
            return {
                "intent": intent,
                "summary": parsed.get("summary", transcript[:80]),
                "layer": "llm",
            }
    except Exception as exc:
        # LLM call failed — this triggers Layer 2 keyword matching
        logger.warning("[CONFIRM L1 FAIL] LLM intent classification failed: %s — activating keyword fallback.", exc)

    return None  # Signal to caller: LLM layer failed, try next layer


def parse_confirmation(transcript: str) -> dict:
    """
    Main confirmation entry point — implements three-layer validation.
    
    Layer 1: LLM-based contextual classification (primary, multilingual, semantic)
    Layer 2: Keyword matching fallback (if LLM fails or returns error)
    Layer 3: Re-record prompt signal (if both layers fail — returns 'unclear')
    
    Returns: {"intent": str, "summary": str, "layer": str}
      intent values: "confirmed" | "denied" | "partial" | "unclear"
      layer values: "llm" | "keyword" | "rerecord"
    """
    t = transcript.strip()

    # Empty transcript — skip all layers, go straight to re-record
    if not t:
        logger.info("[CONFIRM] Empty transcript — returning unclear immediately.")
        return {"intent": "unclear", "summary": "", "layer": "rerecord"}

    # Mock mode: return deterministic confirmed result
    if MOCK_MODE:
        return {"intent": "confirmed", "summary": "Mock confirmation accepted.", "layer": "llm"}

    # ── Layer 1: LLM Classification ─────────────────────────────────────────
    logger.info("[CONFIRM L1] Attempting LLM-based intent classification…")
    llm_result = _llm_confirm_intent(t)

    if llm_result is not None:
        # LLM produced a valid result — use it regardless of intent value
        return llm_result

    # ── Layer 2: Keyword Matching Fallback ──────────────────────────────────
    # Reached here because LLM failed (exception or API error)
    logger.info("[CONFIRM L2] LLM failed — activating keyword matching fallback…")
    keyword_result = _keyword_match_confirmation(t)

    if keyword_result == "YES":
        return {
            "intent": "confirmed",
            "summary": f"Keyword match confirmed: '{t[:60]}'",
            "layer": "keyword",
        }
    elif keyword_result == "NO":
        return {
            "intent": "denied",
            "summary": f"Keyword match denied: '{t[:60]}'",
            "layer": "keyword",
        }

    # ── Layer 3: Re-record Prompt ────────────────────────────────────────────
    # Both LLM AND keyword matching failed — signal UI to show re-record prompt
    logger.warning(
        "[CONFIRM L3] Both LLM and keyword matching failed for: '%s' — triggering re-record prompt.", t[:60]
    )
    return {
        "intent": "unclear",
        "summary": t[:80],
        "layer": "rerecord",  # UI uses this to display the polite re-record message
    }


# ===========================================================================
# Translation helpers
# ===========================================================================

def _sarvam_translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text via Sarvam /translate API. Returns translated text or original on failure."""
    try:
        res = requests.post(
            SARVAM_TRANSLATE_URL,
            headers={
                "Content-Type": "application/json",
                "api-subscription-key": SARVAM_API_KEY,
            },
            json={
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
            },
            timeout=15,
        )
        res.raise_for_status()
        translated = res.json().get("translated_text", "").strip()
        if translated:
            logger.info("[SARVAM TRANSLATE] %s -> %s OK", source_lang, target_lang)
            return translated
    except Exception as e:
        logger.warning("[SARVAM TRANSLATE] Failed: %s — returning original", e)
    return text  # Fail-safe: return original


# ===========================================================================
# TTS Pipeline
# ===========================================================================

def _sarvam_tts(text: str, lang: str) -> bytes | None:
    """
    Call Sarvam TTS API (bulbul:v2). Kannada (kn) only — returns audio bytes or None on failure.

    Key fix: Sarvam API requires `text` (str), NOT `inputs` (list).
    Using `inputs` causes a 400 Bad Request error.
    """
    # Sarvam TTS is only used for Kannada — Hindi and English use Edge TTS directly
    if lang.lower().strip() != "kn":
        logger.info("[SARVAM TTS] Skipping — Sarvam TTS only used for Kannada (lang=%s).", lang)
        return None

    target_lang = SARVAM_TTS_LANG_MAP.get("kn", "kn-IN")  # Always kn-IN here
    try:
        res = requests.post(
            SARVAM_TTS_URL,
            headers={
                "Content-Type": "application/json",
                "api-subscription-key": SARVAM_API_KEY,
            },
            json={
                "text": text,                    # FIX: use "text" (str), NOT "inputs" (list)
                "target_language_code": target_lang,
                "speaker": "anushka",            # anushka is the default bulbul:v2 Kannada speaker
                "model": "bulbul:v2",
                "enable_preprocessing": True,    # Better handling of mixed-language text
            },
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        import base64
        audio_b64 = data.get("audios", [None])[0]
        if audio_b64:
            logger.info("[SARVAM TTS] Success for lang=%s", target_lang)
            return base64.b64decode(audio_b64)
    except Exception as e:
        logger.warning("[SARVAM TTS] Failed: %s — will fallback to Edge TTS", e)
    return None


def generate_tts(text: str, lang: str) -> str:
    """
    Synthesise verification prompt audio.

    Routing logic:
      - Kannada (kn): Sarvam TTS (primary) → Edge TTS (fallback if Sarvam fails)
      - Hindi (hi):   Edge TTS directly (hi-IN-MadhurNeural)
      - English (en): Edge TTS directly (en-IN-NeerjaNeural)
    """
    output_path = f"/tmp/verify_{uuid.uuid4().hex}.mp3"

    if MOCK_MODE:
        logger.info("[MOCK] generate_tts() -> writing stub %s.", output_path)
        with open(output_path, "wb") as f:
            f.write(b"")  # Zero-byte stub for mock mode
        return output_path

    logger.info("TTS: lang=%s, chars=%d", lang, len(text))

    def run_async_tts(tts_text: str, tts_voice: str, path: str):
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_until_complete(_tts_coroutine(tts_text, tts_voice, path))
        finally:
            new_loop.close()

    normalized_lang = lang.lower().strip()

    # ── Kannada: try Sarvam TTS first, fall back to Edge TTS ────────────────
    if normalized_lang == "kn":
        audio_bytes = _sarvam_tts(text, normalized_lang)
        if audio_bytes:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            logger.info("[SARVAM TTS] Saved to %s", output_path)
            return output_path
        logger.info("[EDGE TTS FALLBACK] Sarvam TTS failed for Kannada — switching to Edge TTS…")

    # ── Hindi / English (or Kannada fallback): use Edge TTS directly ────────
    voice = TTS_VOICE_MAP.get(normalized_lang, TTS_FALLBACK_VOICE)
    logger.info("[EDGE TTS] Using voice=%s for lang=%s", voice, normalized_lang)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_async_tts, text, voice, output_path)
            future.result(timeout=30)
        logger.info("TTS saved to %s", output_path)
        return output_path
    except Exception as exc:
        logger.warning("Edge TTS failed with voice %s: %s — retrying with fallback voice.", voice, exc)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_async_tts, text, TTS_FALLBACK_VOICE, output_path)
                future.result(timeout=30)
            logger.info("TTS fallback voice succeeded.")
            return output_path
        except Exception as fallback_exc:
            logger.error("TTS fallback also failed: %s", fallback_exc)
            return output_path


def prepare_verification(ai_data: dict) -> None:
    """
    Construct the full verification prompt in English, translate it to the citizen's detected
    language, and generate TTS audio in that same language. Updates ai_data directly.

    Language routing:
      - citizen spoke Kannada (kn) → translate to kn-IN → Sarvam TTS (kn-IN-VarunNeural fallback)
      - citizen spoke Hindi    (hi) → translate to hi-IN → Edge TTS hi-IN-MadhurNeural
      - citizen spoke English  (en) → no translation needed → Edge TTS en-IN-NeerjaNeural
    """
    base_prompt = ai_data.get("verification_prompt", "")
    normalized_issue = ai_data.get("normalized_issue", "")
    citizen_lang = ai_data.get("language", "kn")  # Language the citizen actually spoke

    # Construct the full verification prompt in English first
    appended_prompt = f"{base_prompt} I heard you say '{normalized_issue}'. Is this correct? Say Yes or No."
    ai_data["verification_prompt_full_en"] = appended_prompt

    tts_text = appended_prompt  # Default: use English text if translation is skipped

    if citizen_lang == "en":
        # Citizen spoke English — no translation needed, play directly in English
        ai_data["verification_prompt_translated"] = appended_prompt
        tts_text = appended_prompt
    else:
        # Citizen spoke Kannada or Hindi — translate the prompt into their language
        target_lang_code = SARVAM_TTS_LANG_MAP.get(citizen_lang, "kn-IN")
        translated_prompt = _sarvam_translate(appended_prompt, "en-IN", target_lang_code)
        ai_data["verification_prompt_translated"] = translated_prompt
        tts_text = translated_prompt

    # Generate TTS in the citizen's own language so they hear the confirmation in their tongue
    tts_path = generate_tts(tts_text, citizen_lang)
    ai_data["verify_tts_path"] = tts_path


# ===========================================================================
# LLM Analysis
# ===========================================================================

def extract_json(text: str) -> dict:
    """Extract first JSON object from an LLM response string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    raise ValueError("No JSON object found in LLM response")


def analyze_transcript(text: str) -> dict:
    """Analyze citizen transcript via Groq LLM and return structured analysis dict."""
    if MOCK_MODE:
        return _MOCK_ANALYSIS

    logger.info("[LLM INPUT] %s%s", text[:80], "..." if len(text) > 80 else "")

    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"CITIZEN TRANSCRIPT: '{text}'"},
        ]
        raw = _sarvam_chat(messages=messages, temperature=0.1, max_tokens=500)
        clean_raw = _strip_fences(raw)
        parsed = extract_json(clean_raw)
        return _enforce_guardrails(parsed)
    except Exception as e:
        logger.error("[LLM FAIL] %s", e)
        return _enforce_guardrails(
            {
                "language": "en",
                "normalized_issue": "Unable to parse request. Please repeat.",
                "confidence": 0.1,
                "sentiment": "confused",
                "verification_prompt": "I didn't catch that. Could you please say it again?",
                "handover": True,
            }
        )


def re_analyze_transcript(new_text: str, previous_analysis: dict, feedback_type: str = "denied") -> dict:
    """
    Re-analyze with context from a previous denied/partial attempt.

    Args:
        new_text: New citizen transcript (English).
        previous_analysis: Previous ai_data dict that was rejected/partial.
        feedback_type: "denied" or "partial" — changes the LLM context framing.
    """
    if MOCK_MODE:
        return _MOCK_ANALYSIS

    prev_issue = previous_analysis.get("normalized_issue", "")
    if feedback_type == "partial":
        context = (
            f"Previous AI analysis was PARTIALLY CORRECT. "
            f"Previous AI summary: '{prev_issue}'. "
            f"Citizen's clarification/correction: '{new_text}'. "
            f"Refine the analysis incorporating the citizen's feedback. "
            f"Keep what was correct and fix what was wrong."
        )
    else:
        context = (
            f"Previous AI analysis was DENIED by the citizen. "
            f"Previous AI summary: '{prev_issue}'. "
            f"Citizen's new recording/correction: '{new_text}'. "
            f"Re-analyze from scratch incorporating the citizen's feedback."
        )

    logger.info("[RE-ANALYZE] Type: %s | New input: %s", feedback_type, new_text[:60])
    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        raw = _sarvam_chat(messages=messages, temperature=0.1, max_tokens=500)
        parsed = extract_json(_strip_fences(raw))
        return _enforce_guardrails(parsed)
    except Exception as e:
        logger.error("[RE-ANALYZE FAIL] %s", e)
        return _enforce_guardrails(
            {
                "language": previous_analysis.get("language", "en"),
                "normalized_issue": prev_issue or "Unable to re-analyze. Needs agent.",
                "confidence": 0.3,
                "sentiment": "confused",
                "verification_prompt": "I'm still having trouble understanding. Could you explain once more?",
                "handover": True,
            }
        )


# ===========================================================================
# Main Audio Processing Pipeline
# ===========================================================================

def process_audio(audio_path: str) -> dict:
    """
    Full pipeline: audio file → structured emergency analysis dict.
    
    Steps:
      1. STT (Layer 1: Sarvam → Layer 2: Edge fallback)
      2. LLM analysis
      3. Translation of verification prompt to citizen's language
      4. TTS generation
    
    Always returns a valid dict (never raises) — errors produce handover=True.
    """
    logger.info("[PROCESS] Starting full pipeline for: %s", audio_path)

    # Step 1: Transcribe audio via two-layer STT pipeline
    raw_text, lang, original_text = transcribe_audio(audio_path)
    logger.info("[STT OUTPUT] Lang: %s, English chars: %d, Native chars: %d",
                lang, len(raw_text), len(original_text))

    # Handle completely empty transcript
    if not raw_text.strip():
        logger.warning("[STT] Empty transcript — returning handover fallback.")
        fallback = _enforce_guardrails(
            {
                "language": lang,
                "normalized_issue": "I couldn't hear clearly. Please speak again.",
                "confidence": 0.2,
                "sentiment": "confused",
                "verification_prompt": "Please try recording again.",
                "handover": False,
            }
        )
        prepare_verification(fallback)
        return {
            **fallback,
            "raw_text": "",
            "original_text": "",
        }

    # Step 2: LLM analysis of the English transcript
    ai_data = analyze_transcript(raw_text)

    # Step 3 & 4: Translate verification prompt and generate TTS
    prepare_verification(ai_data)

    # Compose final result — include both English and native transcripts for UI display
    final = {
        **ai_data,
        "raw_text": raw_text,          # English translation (for agent readability)
        "original_text": original_text, # Native language (for handover/verify display)
    }
    logger.info(
        "[PIPELINE OK] Confidence: %.2f, Sentiment: %s, Handover: %s",
        final["confidence"], final["sentiment"], final["handover"]
    )
    return final


# ===========================================================================
# Feedback Logging
# ===========================================================================

def normalize_kannada(text: str) -> str:
    """Normalize Kannada text using KANADA_FIXES dictionary."""
    for wrong, right in KANADA_FIXES.items():
        text = text.replace(wrong, right)
    return text


def normalize_transcript(text: str, lang: str) -> str:
    """Cross-language ASR drift correction for Kannada/Hindi/English."""
    if not text:
        return ""
    lang = lang.lower()[:2]
    KN_FIXES = {
        "ನಮ್ವ": "ನಮ್ಮ", "ವಣಿ": "ವಾಣಿ", "ತುಂಬಾ": "ತುಂಬಾ",
        "ಹಾಳಾಗಿದೆ": "ಹಾಳಾಗಿದೆ", "ರಸ್ತೆ": "ರಸ್ತೆ", "ಪಾಣಿ": "ಪಾಣಿ",
        "ಕೇಂದ್ರ": "ಕೇಂದ್ರ", "ಫೋನ್": "ಫೋನ್", "ಬೇಕು": "ಬೇಕು", "ಸೇವೆ": "ಸೇವೆ",
    }
    HI_FIXES = {
        "क्यो": "क्यों", "कहा": "कहाँ", "ठिक": "ठीक", "सही": "सही",
        "रस्ता": "रास्ता", "पानि": "पानी", "सफाय": "सफाई",
    }
    EN_FIXES = {
        "teh": "the", "plz": "please", "thk": "thank", "recieve": "receive",
        "adress": "address", "wont": "won't", "cant": "can't", "im": "I'm",
        "waterline": "water line", "paniline": "pani line", "bandh kr do": "shut off",
    }
    fixes = {"kn": KN_FIXES, "hi": HI_FIXES, "en": EN_FIXES}
    current = fixes.get(lang, EN_FIXES)
    out = text
    for k, v in current.items():
        out = out.replace(k, v)
    return out.strip().replace("  ", " ").replace("\n", " ")


def log_feedback(data: dict) -> None:
    """Append a feedback row to feedback.csv; creates file with headers if it does not exist."""
    file_exists = os.path.isfile(FEEDBACK_FILE)
    try:
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FEEDBACK_HEADERS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
                logger.info("Created %s with headers.", FEEDBACK_FILE)

            # Calculate confidence-weighted feedback signal for model improvement
            response = data.get("citizen_response", "")
            try:
                confidence = float(data.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5

            if response == "Confirmed":
                weight = round(0.5 + (confidence * 0.5), 2)   # 0.5–1.0 strong positive
            elif response == "Partial":
                weight = round(0.3 * confidence, 2)            # 0.0–0.3 weak positive
            elif response == "Handover":
                weight = round(-0.5 * confidence, 2)           # Negative signal
            else:
                weight = 0.0

            row = {
                "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
                "language": data.get("language", ""),
                "raw_text": data.get("raw_text", ""),
                "original_text": data.get("original_text", ""),
                "ai_issue": data.get("normalized_issue", ""),
                "confidence": data.get("confidence", ""),
                "sentiment": data.get("sentiment", ""),
                "citizen_response": data.get("citizen_response", ""),
                "agent_correction": data.get("agent_correction", ""),
                "handover": data.get("handover", ""),
                "feedback_weight": weight,
                "stt_source": data.get("stt_source", "sarvam"),
            }
            writer.writerow(row)
            logger.info("Feedback logged to %s (weight=%.2f).", FEEDBACK_FILE, weight)
    except Exception as exc:
        logger.error("log_feedback() failed: %s", exc)
