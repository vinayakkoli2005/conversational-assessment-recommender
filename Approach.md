# Approach Document - SHL Conversational Agent

## Design Choices

The service is built on FastAPI for its native async support and automatic schema validation. The core conversational logic uses OpenAI `gpt-4o-mini` with Function Calling and Structured Outputs (`json_schema` response format) to guarantee schema compliance on every response.

Rather than rule-based decision trees, the agent uses a comprehensive system prompt and a tool-use paradigm. The prompt instructs four core behaviors:

1. **Clarify** — ask for role, seniority, and skills before acting on vague queries.
2. **Recommend** — once constraints are clear, call `search_catalog` (potentially multiple times for different categories) and commit to a 1–10 item shortlist.
3. **Refine** — update the shortlist when the user changes constraints, without restarting.
4. **Compare** — explain differences using only catalog fields (duration, languages, description, remote/adaptive flags) passed back through tool results — no model prior.

The `POST /chat` endpoint is fully stateless. Every call receives the full conversation history; the service holds no per-conversation state.

## Retrieval Setup

A **hybrid BM25 + TF-IDF** retriever is built in-memory at startup over the 377-item catalog:

- **TF-IDF** (scikit-learn): cosine similarity over combined name + description + job levels + category keys. Good at exact term overlap.
- **BM25Okapi** (rank-bm25): handles term-frequency saturation and document length normalization. Better for short or vague queries where TF-IDF scores collapse (e.g. "senior Rust engineer infrastructure" with no Rust-specific entries).
- **Hybrid score**: equal-weight average of both normalised signals. More robust than either alone across the full query distribution.

Each document concatenates name, description, job levels, and category keys for maximum recall surface. The retriever returns up to 10 results (hard-capped in code), each enriched with `test_type`, `duration`, `languages`, `remote`, and `adaptive` fields for grounded compare answers.

This keeps the deployment dependency-light (no vector database, no embedding API calls) while substantially improving retrieval quality over pure TF-IDF.

## Prompt & Agent Design

Key decisions:

- **`test_type` is resolved deterministically in code** via a `CATEGORY_TO_CODE` mapping dict (`Personality & Behavior → P`, `Knowledge & Skills → K`, etc.) and injected into every tool result. The LLM copies it rather than guessing — eliminating a hallucination vector entirely.
- **Multi-turn tool call loop**: the agent loop continues issuing `search_catalog` calls until the LLM stops requesting them (capped at 5 rounds). This lets the model run separate searches for cognitive and personality components in a single turn, improving Recall@10 for multi-category shortlists.
- **Turn cap**: conversation history is truncated to the last 8 messages before any LLM call, and the system prompt explicitly tells the agent to commit to a best-effort shortlist by turn 7.
- **`recommendations: null` vs `[]`**: Pydantic schema uses `Optional[List[Recommendation]]` so the field can be null (still clarifying) or a populated list (committed shortlist). `[]` is reserved for explicit refusals.
- **Structured output fallback**: a `try/except` around `json.loads` returns a schema-compliant error response so the automated evaluator never receives a 500 with a raw exception string.

## Evaluation Approach

- **Local trace testing**: the 10 provided conversation traces (C1–C10) were parsed and replayed against the endpoint. Schema compliance and turn limits were verified on each.
- **Recall tuning**: hybrid BM25+TF-IDF was validated against C2 ("senior Rust engineer") and C3 ("entry-level contact centre") — both cases where pure TF-IDF misses relevant items due to sparse keyword overlap.
- **Behavior probes**: off-topic injections ("write a Python script", "ignore previous instructions") were tested to confirm graceful refusal without schema violations.
- **What didn't work**: an earlier pure TF-IDF implementation produced empty results for low-frequency technical terms. Lowering the similarity threshold fixed recall but introduced noise; replacing with hybrid BM25 solved both issues cleanly.

## AI Tools Usage

An AI assistant (Claude Code) was used to scaffold the FastAPI boilerplate, implement the retrieval class, map the schema to Pydantic models, and construct the tool-calling logic. The AI also identified and fixed several bugs: missing `test_type` mapping logic, a `json.loads` crash path, a single-round tool call limitation, and an un-capped `top_k` parameter. Design decisions (hybrid retrieval approach, multi-turn loop, nullable recommendations) were made and validated by the author.
