from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()  # picks up .env locally; no-op in production where env vars are injected directly
from .agent import ConversationalAgent

app = FastAPI(title="SHL Conversational Recommender API")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

CATALOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shl_product_catalog.json")
agent = ConversationalAgent(CATALOG_PATH)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

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
