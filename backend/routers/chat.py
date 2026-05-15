"""Chat router — RAG-powered streaming chatbot endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services import rag_service

router = APIRouter(tags=["chat"])


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryMessage] = []


@router.post("/chat")
def chat(req: ChatRequest):
    history = [{"role": h.role, "content": h.content} for h in req.history]
    return StreamingResponse(
        rag_service.stream_chat(req.message, history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
