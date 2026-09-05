from __future__ import annotations

from typing import Callable

from langgraph.graph import END, StateGraph

from .state import ResearchState
from .tools import (
    get_llm,
    get_report_markdown,
    get_structured_evidence,
    get_structured_questions,
    search_tavily,
    format_markdown_report,
)


def planner_node(state: ResearchState) -> ResearchState:
    llm = get_llm()
    questions = get_structured_questions(llm, state["topic"])
    state["research_questions"] = questions[:5]
    return state


def researcher_node(state: ResearchState) -> ResearchState:
    search_results: list[dict] = []
    for question in state["research_questions"]:
        results = search_tavily(question)
        for item in results:
            search_results.append({
                "title": item["title"],
                "url": item["url"],
                "content": item["content"],
                "research_question": question,
            })
    state["search_results"] = search_results
    return state


def evidence_analyzer_node(state: ResearchState) -> ResearchState:
    if not state["search_results"]:
        state["evidence"] = []
        return state

    llm = get_llm()
    evidence = get_structured_evidence(llm, state["topic"], state["search_results"])
    state["evidence"] = evidence
    return state


def synthesizer_node(state: ResearchState) -> ResearchState:
    llm = get_llm()
    evidence = state["evidence"]
    if not evidence:
        state["final_report"] = (
            "# " + state["topic"] + "\n\n"
            "## Executive Summary\n"
            "Insufficient evidence is available from the searched sources to make a reliable claim.\n\n"
            "## Key Findings\n"
            "- The available search results do not provide enough evidence to support a confident conclusion.\n\n"
            "## Detailed Analysis\n"
            "The research process gathered sources, but none provided enough directly relevant evidence to support a robust analysis.\n\n"
            "## Limitations\n"
            "This report is limited by insufficient source coverage or weak evidence.\n\n"
            "## Conclusion\n"
            "The available evidence is insufficient to draw a confident conclusion.\n\n"
            "## Sources\n"
            "No verified sources were available for this topic from the current search results."
        )
    else:
        state["final_report"] = get_report_markdown(
            llm,
            state["topic"],
            evidence,
            state["research_questions"],
        )
    return state


def markdown_exporter_node(state: ResearchState) -> ResearchState:
    state["markdown_report"] = format_markdown_report(state["topic"], state["final_report"])
    return state


def build_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("evidence_analyzer", evidence_analyzer_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("markdown_exporter", markdown_exporter_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "evidence_analyzer")
    workflow.add_edge("evidence_analyzer", "synthesizer")
    workflow.add_edge("synthesizer", "markdown_exporter")
    workflow.add_edge("markdown_exporter", END)

    return workflow.compile()


def run_research_workflow(topic: str, progress_callback: Callable[[str], None] | None = None) -> ResearchState:
    if not topic or not topic.strip():
        raise ValueError("A research topic is required.")

    state: ResearchState = {
        "topic": topic.strip(),
        "research_questions": [],
        "search_results": [],
        "evidence": [],
        "final_report": "",
        "markdown_report": "",
    }

    result = graph.invoke(state)

    if progress_callback:
        progress_callback("✓ Research questions generated")
        progress_callback("✓ Web research completed")
        progress_callback("✓ Evidence analyzed")
        progress_callback("✓ Report synthesized")

    return result


graph = build_graph()
