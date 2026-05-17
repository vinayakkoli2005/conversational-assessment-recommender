# SHL Conversational Agent

This repository contains the solution for the SHL AI Intern Assignment.

## Features
- **FastAPI Backend**: Exposes `/health` and `/chat` endpoints conforming to the exact required schema.
- **Agentic Conversational Logic**: Uses OpenAI's `gpt-4o-mini` with Function Calling and Structured Outputs to strictly adhere to the response schema and conversation guidelines.
- **Fast Local Retrieval**: Uses a TF-IDF vectorizer built on `scikit-learn` to index the SHL catalog on startup. It is lightweight, requires no external database (like Chroma or FAISS), and runs perfectly within free-tier deployment limits.

## Project Structure
- `app/main.py`: FastAPI endpoints.
- `app/agent.py`: Conversational logic handling OpenAI interactions and schema enforcement.
- `app/catalog.py`: The TF-IDF retrieval engine.
- `shl_product_catalog.json`: The scraped product catalog.

## Local Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Set your OpenAI API Key.
   ```bash
   # Windows
   set OPENAI_API_KEY=sk-your-key-here
   # Mac/Linux
   export OPENAI_API_KEY=sk-your-key-here
   ```

3. **Run the API**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Deployment

A `Dockerfile` and `render.yaml` are included for seamless deployment to platforms like Render, Railway, or Fly.io.
Make sure to provide the `OPENAI_API_KEY` in the environment secrets of your deployment platform.

### Deploying to Render
1. Connect this repository to your Render account.
2. Create a new "Web Service".
3. Use the `render.yaml` Blueprint or manually set the start command to:
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add the `OPENAI_API_KEY` under Environment Variables.
