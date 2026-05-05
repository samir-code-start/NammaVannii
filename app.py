"""app.py — Namma Vanni Streamlit UI: Record → Verify → Confirm/Handover → Agent Dashboard."""

import os
import pandas as pd
import streamlit as st

from engine import (
    MOCK_MODE,
    FEEDBACK_FILE,
    log_feedback,
    parse_confirmation,
    process_audio,
    transcribe_audio,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Namma Vanni — 1092 AI Helpline", page_icon="📞", layout="wide")

SENTIMENT_BADGES = {
    "calm": "🟢 Calm",
    "confused": "🟡 Confused",
    "urgent": "🟠 Urgent",
    "distressed": "🔴 Distressed",
    "angry": "🟥 Angry",
}

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {"stage": "input_record", "ai_data": None, "attempts": 0}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

def _reset():
    """Reset session state to initial values."""
    for key, val in _DEFAULTS.items():
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center'>📞 Namma Vanni — 1092 AI Helpline</h1>"
    "<p style='text-align:center;opacity:0.6'>Voice-to-voice citizen assistant for Karnataka</p>",
    unsafe_allow_html=True,
)
if MOCK_MODE:
    st.caption("⚙️ Running in **MOCK MODE** — no API calls.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: input_record
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "input_record":
    st.subheader("🎤 Start Call")
    audio_bytes = st.audio_input("Speak your concern or upload audio")

    if audio_bytes is not None:
        wav_path = "input.wav"
        with open(wav_path, "wb") as f:
            f.write(audio_bytes.getvalue())

        with st.spinner("🔄 Processing your voice — transcribing & analysing…"):
            result = process_audio(wav_path)

        st.session_state.ai_data = result
        st.session_state.attempts = 0
        st.session_state.stage = "verify"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: verify
# ═══════════════════════════════════════════════════════════════════════════
import time
if st.session_state.stage == "verify":
    data = st.session_state.ai_data or {}
    lang = data.get("language", "en")
    tts_path = data.get("verify_tts_path", "")
    
    # Initialize phase tracking once
    if "_ver_start" not in st.session_state:
        st.session_state._ver_start = time.time()
        st.session_state._ver_phase = 0
        st.session_state._mic_initialized = False
        
    elapsed_1 = time.time() - st.session_state._ver_start
    
    # Play AI verification summary
    st.markdown(f"**AI Summary:** {data.get('normalized_issue', '—')}")
    st.caption(f"🌐 `{lang.upper()}` | 📊 `{data.get('confidence', 0):.0%}`")
    if os.path.isfile(tts_path): st.audio(tts_path, autoplay=True, key="v_tts_main")
    
    # PHASE 0: Single-click mic initialization (browser security requirement)
    if not st.session_state._mic_initialized:
        st.warning("⚠️ Click ONCE to activate live microphone. Keep device close.")
        if st.button("🎤 Activate Live Microphone", key="btn_init_mic", type="primary"):
            st.session_state._mic_initialized = True
            st.rerun()
    else:
        # PHASE 1: Initial 10-second auto-listening window
        if st.session_state._ver_phase == 0:
            st.info("🔴 LISTENING... Speak Yes/No/Han/Sari/Illa now.")
            conf_audio = st.audio_input("🎙️ Respond", key="mic_live", disabled=False)
            
            if conf_audio:
                with open("confirm.wav", "wb") as f: f.write(conf_audio.getvalue())
                with st.spinner("Processing..."):
                    raw_conf, _ = transcribe_audio("confirm.wav")
                    parsed = parse_confirmation(raw_conf)
                
                st.session_state.attempts += 1
                if parsed["intent"] == "confirmed":
                    st.session_state.stage = "agent_ready"
                    st.success("✅ Verified! Routing to agent...")
                    st.rerun()
                elif parsed["intent"] == "denied":
                    if st.session_state.attempts >= 2 or data.get("handover", False):
                        st.session_state.stage = "handover"
                        st.error("🔄 Handover triggered. Connecting to agent...")
                    else:
                        st.warning("⚠️ Not understood. Try again.")
                        st.session_state.stage = "input_record"
                    st.rerun()
                else:
                    st.info(f"📝 AI heard: '{parsed['summary']}'")
                    st.warning("❓ Did you mean Yes or No?")
                    st.session_state.stage = "input_record"
                    st.rerun()
                    
            if elapsed_1 >= 10:
                st.session_state._ver_phase = 1
                st.rerun()
                
        # PHASE 2: Auto-Replay TTS + Final 3-Second Safety Window
        elif st.session_state._ver_phase == 1:
            if "_rep_start" not in st.session_state:
                st.session_state._rep_start = time.time()
                st.info("🔊 Replay: Do I understand your problem? Reply now.")
                if os.path.isfile(tts_path): st.audio(tts_path, autoplay=True, key="v_tts_replay")
                
            elapsed_2 = time.time() - st.session_state._rep_start
            st.caption(f"⏳ Final response window: {max(0, int(3 - elapsed_2))}s")
            
            conf_audio_2 = st.audio_input("🎙️ Final Response", key="mic_final", disabled=False)
            if conf_audio_2:
                with open("confirm_2.wav", "wb") as f: f.write(conf_audio_2.getvalue())
                with st.spinner("Processing..."):
                    raw_conf, _ = transcribe_audio("confirm_2.wav")
                    parsed = parse_confirmation(raw_conf)
                
                st.session_state.attempts += 1
                if parsed["intent"] == "confirmed":
                    st.session_state.stage = "agent_ready"
                    st.success("✅ Verified!")
                    st.rerun()
                elif parsed["intent"] == "denied":
                    if st.session_state.attempts >= 2 or data.get("handover", False):
                        st.session_state.stage = "handover"
                        st.error("🔄 Handover triggered.")
                    else:
                        st.session_state.stage = "input_record"
                    st.rerun()
                else:
                    st.session_state.stage = "input_record"
                    st.rerun()
                    
            if elapsed_2 >= 3:
                st.warning("⏰ No further response detected. Transferring to human agent with available context...")
                st.session_state.stage = "handover"
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: decision (transient — routes immediately)
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "decision":
    st.session_state.attempts += 1
    data = st.session_state.ai_data or {}

    if st.session_state.attempts >= 2 or data.get("handover", False):
        st.session_state.stage = "handover"
    else:
        st.warning("⚠️ Not understood. Please try again.")
        st.session_state.stage = "input_record"
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: agent_ready
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "agent_ready":
    data = st.session_state.ai_data or {}
    st.subheader("✅ Agent Dashboard — Verified Issue")

    sentiment = data.get("sentiment", "calm")
    st.success(
        f"**Issue:** {data.get('normalized_issue', '—')}  \n"
        f"**Sentiment:** {SENTIMENT_BADGES.get(sentiment, sentiment)} &nbsp;|&nbsp; "
        f"**Confidence:** {data.get('confidence', 0) * 100:.0f}% &nbsp;|&nbsp; "
        f"**Language:** {data.get('language', '—').upper()}"
    )

    with st.expander("📝 Raw Transcript", expanded=False):
        st.code(data.get("raw_text", "—"), language=None)

    agent_note = st.text_area(
        "Agent Notes / Corrections",
        value=data.get("normalized_issue", ""),
        help="Edit if the AI summary needs correction.",
    )

    if st.button("✅ Log & Close Call", use_container_width=True):
        feedback = dict(data)
        feedback["citizen_response"] = "Confirmed"
        feedback["agent_correction"] = agent_note if agent_note != data.get("normalized_issue", "") else ""
        log_feedback(feedback)
        st.toast("📋 Feedback logged!", icon="✅")
        _reset()
        st.rerun()

    st.divider()
    st.subheader("📊 Feedback Log")
    if os.path.isfile(FEEDBACK_FILE):
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.caption("Feedback file is empty or unreadable.")
    else:
        st.caption("No feedback recorded yet.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: handover
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "handover":
    data = st.session_state.ai_data or {}

    st.error("⚠️ **HUMAN AGENT TAKING OVER**")
    st.markdown(
        "The AI could not confidently verify the caller's issue.  \n"
        "A human agent will now handle this call directly.  \n\n"
        f"**Last AI Summary:** {data.get('normalized_issue', '—')}  \n"
        f"**Sentiment:** {SENTIMENT_BADGES.get(data.get('sentiment', ''), '—')}  \n"
        f"**Confidence:** {data.get('confidence', 0) * 100:.0f}%"
    )

    with st.expander("📝 Raw Transcript"):
        st.code(data.get("raw_text", "—"), language=None)

    if st.button("🔄 End Call", use_container_width=True):
        feedback = dict(data)
        feedback["citizen_response"] = "Handover"
        feedback["agent_correction"] = ""
        log_feedback(feedback)
        st.toast("📋 Call logged as handover.", icon="🔄")
        _reset()
        st.rerun()
