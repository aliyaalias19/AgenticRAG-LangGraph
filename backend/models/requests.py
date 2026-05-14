"""API request models with input validation."""

from typing import List, Optional

from pydantic import BaseModel, field_validator


class QueryRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []
    show_reasoning: bool = True

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        if len(v) > 2000:
            raise ValueError("question exceeds 2000 characters")
        return v


class SupervisorRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class EvalRequest(BaseModel):
    custom_test_cases: Optional[List[dict]] = None
