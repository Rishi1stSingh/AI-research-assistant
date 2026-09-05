import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class ResearchQuestions(BaseModel):
    questions: list[str] = Field(..., min_length=3, max_length=5)


class EvidenceList(BaseModel):
    evidence: list[dict[str, str]] = Field(...)


def get_api_keys() -> tuple[str, str, str]:
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY ")
    tavily_key = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY ")
    model = os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"

    if not groq_key:
        raise ValueError("Missing GROQ_API_KEY environment variable.")
    if not tavily_key:
        raise ValueError("Missing TAVILY_API_KEY environment variable.")
    if not model:
        raise ValueError("Missing GROQ_MODEL environment variable.")

    return groq_key.strip().strip("'\""), tavily_key.strip().strip("'\""), model.strip().strip("'\"")


def get_llm() -> ChatGroq:
    groq_key, _, model = get_api_keys()
    return ChatGroq(
        model=model,
        api_key=groq_key,
        temperature=0.2,
    )


def search_tavily(question: str) -> list[dict[str, Any]]:
    _, tavily_key, _ = get_api_keys()
    search = TavilySearch(api_key=tavily_key, max_results=3)
    response = search.invoke({"query": question})

    if isinstance(response, dict):
        results = response.get("results", [])
    elif isinstance(response, list):
        results = response
    else:
        results = []

    normalized_results: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        normalized_results.append(
            {
                "title": item.get("title") or "Untitled source",
                "url": item.get("url") or "",
                "content": item.get("content") or item.get("snippet") or "No summary available.",
            }
        )

    return normalized_results


def sanitize_topic_for_filename(topic: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9\s-]", "", topic.strip())
    value = re.sub(r"\s+", "-", value).strip("-")
    return value.lower()[:80] or "research-note"


def format_markdown_report(topic: str, report_text: str) -> str:
    date_stamp = datetime.utcnow().strftime("%Y-%m-%d")
    title = topic.strip() or "Research Topic"
    front_matter = (
        "---\n"
        f'title: "{title}"\n'
        f'date: "{date_stamp}"\n'
        "tags:\n"
        "- research\n"
        "- ai\n"
        "---\n\n"
    )
    return front_matter + report_text.strip() + "\n"


def build_question_prompt(topic: str) -> str:
    return (
        f"Generate 3 to 5 focused research questions for this topic: {topic}. "
        "They should be specific, practical, and useful for a research report. "
        "Return only the questions as a JSON list."
    )


def build_evidence_prompt(topic: str, search_results: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"Title: {result.get('title', 'Untitled')}\n"
        f"URL: {result.get('url', '')}\n"
        f"Content: {result.get('content', 'No content available.')}"
        for result in search_results
    )
    return (
        f"Topic: {topic}\n\n"
        "Using only the sources below, identify the strongest evidence-based claims that answer the research questions. "
        "Each evidence item must cite a source title and URL that exactly matches one of the provided sources. "
        "If the information is insufficient, say 'Insufficient evidence available' in the claim and supporting information. "
        "Return JSON with a list of evidence objects: claim, supporting_information, source_title, source_url.\n\n"
        f"Sources:\n{context}"
    )


def build_synthesis_prompt(topic: str, evidence: list[dict[str, str]], research_questions: list[str]) -> str:
    evidence_text = "\n\n".join(
        f"Claim: {item.get('claim', '')}\n"
        f"Supporting information: {item.get('supporting_information', '')}\n"
        f"Source: {item.get('source_title', '')} ({item.get('source_url', '')})"
        for item in evidence
    )
    return (
        f"Topic: {topic}\n\n"
        "Write a concise but evidence-based research report in Markdown. "
        "Use only the evidence provided below. Do not invent facts or sources. "
        "Do not rely on outside knowledge. If the evidence is weak or incomplete, explicitly state that there is insufficient evidence. "
        "The report must have these sections: '# Topic', '## Executive Summary', '## Key Findings', '## Detailed Analysis', '## Limitations', '## Conclusion', and '## Sources'. "
        "For the sources section, include a numbered list of the URLs used.\n\n"
        f"Research Questions:\n{chr(10).join(f'- {q}' for q in research_questions)}\n\n"
        f"Evidence:\n{evidence_text}"
    )


def parse_json_list(response_text: str) -> list[dict[str, Any]]:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def get_structured_questions(llm: ChatGroq, topic: str) -> list[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful research planner. Return only valid JSON matching the requested schema."),
        ("user", build_question_prompt(topic)),
    ])
    response = llm.invoke(prompt.format_messages())
    content = response.content if hasattr(response, "content") else str(response)

    try:
        parsed = parse_json_list(content)
        if isinstance(parsed, list):
            questions = [str(item).strip() for item in parsed if str(item).strip()]
            if len(questions) >= 3:
                return questions[:5]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    return [
        f"How is {topic} currently being used or studied?",
        f"What are the main benefits or opportunities associated with {topic}?",
        f"What limitations, risks, or challenges are associated with {topic}?",
        f"What evidence helps explain the current state of {topic}?",
    ]


def get_structured_evidence(llm: ChatGroq, topic: str, search_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use only the provided sources. Do not invent missing facts."),
        ("user", build_evidence_prompt(topic, search_results)),
    ])
    response = llm.invoke(prompt.format_messages())
    content = response.content if hasattr(response, "content") else str(response)

    try:
        items = parse_json_list(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        items = []

    valid_urls = {entry.get("url", "").strip() for entry in search_results}
    filtered: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source_url = str(item.get("source_url", "")).strip()
        if not source_url or source_url not in valid_urls:
            continue
        filtered.append({
            "claim": str(item.get("claim", "Insufficient evidence available")),
            "supporting_information": str(item.get("supporting_information", "Insufficient evidence available")),
            "source_title": str(item.get("source_title", "Unknown source")),
            "source_url": source_url,
        })

    if not filtered:
        return []

    return filtered


def get_report_markdown(llm: ChatGroq, topic: str, evidence: list[dict[str, str]], research_questions: list[str]) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a clean, evidence-based Markdown research report."),
        ("user", build_synthesis_prompt(topic, evidence, research_questions)),
    ])
    response = llm.invoke(prompt.format_messages())
    report = response.content if hasattr(response, "content") else str(response)
    return report.strip()
