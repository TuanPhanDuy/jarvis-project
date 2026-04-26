"""JARVIS Streamlit dashboard — connects to POST /api/chat."""
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="JARVIS", page_icon="🤖", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("JARVIS")
    st.caption("Just A Rather Very Intelligent System")
    st.divider()
    api_base = st.text_input("API Base URL", value="http://localhost:8000")
    researcher_mode = st.checkbox("Researcher Mode", value=False,
                                  help="Use the research-focused agent with web search quota tracking.")
    st.divider()
    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Session: `{st.session_state.get('session_id', '')[:8]}…`")

# ── Session init ───────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history ───────────────────────────────────────────────────────────────

st.header("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("usage"):
            u = msg["usage"]
            st.caption(
                f"↑ {u.get('input_tokens', 0)} / ↓ {u.get('output_tokens', 0)} tokens — "
                f"${u.get('estimated_cost_usd', 0):.4f}"
            )

# ── Input ──────────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask JARVIS anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{api_base}/api/chat",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id,
                        "researcher_mode": researcher_mode,
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
                reply = data["response"]
                usage = data.get("usage", {})
            except requests.exceptions.ConnectionError:
                reply = f"⚠️ Cannot connect to JARVIS API at `{api_base}`. Is the server running?"
                usage = {}
            except Exception as exc:
                reply = f"⚠️ Error: {exc}"
                usage = {}

        st.markdown(reply)
        if usage:
            st.caption(
                f"↑ {usage.get('input_tokens', 0)} / ↓ {usage.get('output_tokens', 0)} tokens — "
                f"${usage.get('estimated_cost_usd', 0):.4f}"
            )

    st.session_state.messages.append({"role": "assistant", "content": reply, "usage": usage})
