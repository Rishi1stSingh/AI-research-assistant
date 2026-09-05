from typing import TypedDict


class SearchResultEntry(TypedDict):
    title: str
    url: str
    content: str
    research_question: str


class EvidenceEntry(TypedDict):
    claim: str
    supporting_information: str
    source_title: str
    source_url: str


class ResearchState(TypedDict):
    topic: str
    research_questions: list[str]
    search_results: list[SearchResultEntry]
    evidence: list[EvidenceEntry]
    final_report: str
    markdown_report: str
