import uuid
from pydantic import BaseModel, Field
from typing import List


class MinimalSource(BaseModel):
    """The MinimalSource model represents a single source of information"""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """The UnansweredQuestion model represents an unanswered question"""
    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """The AnsweredQuestion model represents an answered question"""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """The RagDataset model represents a dataset of RAG questions"""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The MinimalSearchResults model represents the search results"""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The MinimalAnswer model represents an answer"""
    answer: str


class StudentSearchResults(BaseModel):
    """The StudentSearchResults model represents search results"""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """The StudentSearchResultsAndAnswer model represents search results
    with answers"""
    search_results: List[MinimalAnswer]
    k: int
