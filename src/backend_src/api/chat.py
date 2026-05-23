import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.backend_src.services.chat import get_answer

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    chat_history: List[Dict[str, Any]]

@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        if not request.chat_history:
            raise HTTPException(status_code=400, detail="chat_history cannot be empty")
        return get_answer(request.chat_history)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
