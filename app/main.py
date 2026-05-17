from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from .agent import ConversationalAgent

app = FastAPI(title="SHL Conversational Recommender API")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# Initialize agent globally. Note: in production, catalog path should be absolute or relative to project root
CATALOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shl_product_catalog.json")
agent = ConversationalAgent(CATALOG_PATH)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert Pydantic models to dicts
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        response = await agent.chat(messages)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
