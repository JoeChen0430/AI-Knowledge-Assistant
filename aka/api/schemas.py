"""Request bodies. Responses are plain dicts built by the domain objects'
``as_dict`` methods, so the wire format lives next to the data it describes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=32)


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1, max_length=32)
    helpful: bool
    note: str = Field(default="", max_length=2000)


class MemoryFeedbackRequest(BaseModel):
    keep: bool
    note: str = Field(default="", max_length=2000)


class ResolveRequest(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=2000)


class LoginRequest(BaseModel):
    tenant_id: str = Field(default="default", min_length=1, max_length=64)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=256)


class CreateUserRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=256)
    role: str = "user"


class BootstrapRequest(BaseModel):
    tenant_id: str = Field(default="default", min_length=1, max_length=64)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=256)


class RebuildRequest(BaseModel):
    force_full: bool = False
