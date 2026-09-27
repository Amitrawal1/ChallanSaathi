"""Streamlit chat UI. Run with: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import streamlit as st

from challansaathi.index import IndexNotFoundError
from challansaathi.llm import LLMUnavailableError
from challansaathi.pipeline import ChallanSaathi
from challansaathi.sources import known_states

AUTO = "Auto-detect from question"
EXAMPLES = [
    "What is the penalty for drunk driving?",
    "Haryana mein seating capacity ka rule kya hai?",
    "UP mein helmet na pehnne par kya hoga?",
    "How do I transfer ownership of a vehicle?",
]

st.set_page_config(page_title="ChallanSaathi", page_icon="⚖️", layout="centered")


@st.cache_resource(show_spinner="Loading search index and embedding model…")
def get_assistant() -> ChallanSaathi:
    return ChallanSaathi()


def render_sources(sources) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, result in enumerate(sources, start=1):
            chunk = result.chunk
            st.markdown(
                f"**[{i}] {chunk.citation}**" + (f" — {chunk.heading}" if chunk.heading else "")
            )
            st.caption(result.text[:1200] + ("…" if len(result.text) > 1200 else ""))


with st.sidebar:
    st.title("⚖️ ChallanSaathi")
    st.write("Ask about Indian motor vehicle law in English, Hindi or Hinglish.")
    state_choice = st.selectbox("State", [AUTO, *known_states(), "Central law only"])
    st.divider()
    st.caption(
        "Answers are generated from the Motor Vehicles Act 1988, the Central Motor Vehicles "
        "Rules 1989 and the Haryana and Uttar Pradesh rules. **This is general information, "
        "not legal advice.** Always verify with the official text."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []

try:
    assistant = get_assistant()
except IndexNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("#### Try asking")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("e.g. What is the fine for driving without a licence?")
question = question or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    state = {AUTO: None, "Central law only": "India"}.get(state_choice, state_choice)
    with st.chat_message("assistant"):
        answer = assistant.ask(question, state=state)
        if answer.state and answer.state != "India":
            st.caption(f"Searching {answer.state} rules + central law")
        try:
            text = st.write_stream(answer.stream)
        except LLMUnavailableError as exc:
            text = f"⚠️ {exc}\n\nThe most relevant provisions are listed below."
            st.warning(text)
        if answer.sources:
            render_sources(answer.sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": answer.sources}
    )
