import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


def load_runtime_secrets() -> None:
    # Local development: load .env
    load_dotenv(Path(__file__).resolve().parent / ".env")

    # Streamlit Cloud: load secrets
    for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "GROQ_MODEL"):
        if os.getenv(key):
            continue

        try:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
        except Exception:
            pass


# IMPORTANT: load secrets BEFORE importing graph
load_runtime_secrets()

st.write("GROQ secret exists:", "GROQ_API_KEY" in st.secrets)
st.write("GROQ environment exists:", bool(os.getenv("GROQ_API_KEY")))

from src.graph import run_research_workflow


st.set_page_config(page_title="ResearchSynth", page_icon="📚", layout="centered")

st.title("# ResearchSynth")
st.caption("Research a topic, synthesize evidence, and export it to Obsidian.")

if "report" not in st.session_state:
    st.session_state.report = None

if "questions" not in st.session_state:
    st.session_state.questions = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "markdown" not in st.session_state:
    st.session_state.markdown = ""


def run_research_form(topic: str):
    try:
        output = run_research_workflow(topic, progress_callback=lambda msg: st.write(msg))
        st.session_state.report = output["final_report"]
        st.session_state.questions = output["research_questions"]
        st.session_state.sources = [
            {"title": item["title"], "url": item["url"]}
            for item in output["search_results"]
        ]
        st.session_state.markdown = output["markdown_report"]
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Research workflow failed: {exc}")


with st.form("research_form"):
    topic = st.text_input("Research topic", placeholder="Impact of AI agents on software development")
    submitted = st.form_submit_button("Research")

if submitted:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        run_research_form(topic)

if st.session_state.report:
    st.subheader("Research Report")
    st.text_area(
        "Report text",
        value=st.session_state.report,
        height=500,
        disabled=True,
        label_visibility="collapsed",
    )

    st.subheader("Export")
    safe_topic = "research-note"
    if st.session_state.questions:
        topic_name = "-".join(st.session_state.questions[0].split()[:6]).lower()
        safe_topic = topic_name or safe_topic

    st.download_button(
        label="Download Obsidian Markdown",
        data=st.session_state.markdown,
        file_name=f"{safe_topic}.md",
        mime="text/markdown",
    )
