import json
import os
import re
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from .catalog import CatalogRetriever, resolve_test_type
from pydantic import BaseModel

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class AgentResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]]
    end_of_conversation: bool

MAX_TURNS = 8
MODEL = "google/gemini-2.0-flash-lite-001"

# Signals that the user is satisfied and the conversation should end
EOC_SIGNALS = [
    "perfect", "thanks", "thank you", "that's what", "that works",
    "confirmed", "good", "great", "looks good", "that covers", "done",
]

def _extract_json(text: str) -> str:
    """Strip markdown code fences if present, return the raw JSON string."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text

def _is_eoc_signal(text: str) -> bool:
    lower = text.lower()
    return any(sig in lower for sig in EOC_SIGNALS)


class ConversationalAgent:
    def __init__(self, catalog_path: str):
        self.retriever = CatalogRetriever(catalog_path)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        ) if api_key else None

        self.system_prompt = """You are a conversational AI agent for SHL Labs, helping hiring managers find the right SHL Individual Test Solutions.

DECISION RULE — clarify vs. recommend:
- If you know the ROLE AND at least one of (seniority / skill focus / context) → call search_catalog immediately and recommend.
- Ask a clarifying question ONLY when the request has NO role at all (e.g. "I need an assessment").
- Never ask more than ONE clarifying question before recommending.

HOW TO SEARCH:
- Issue MULTIPLE search_catalog calls to cover different dimensions of the role:
  * One call for technical/skills tests (e.g. "Java programming knowledge test")
  * One call for cognitive/reasoning (e.g. "numerical reasoning ability senior")
  * One call for personality/behaviour (e.g. "OPQ personality leadership")
- Use specific, narrow queries — not one broad query for everything.
- top_k=5 per call is sufficient; the union of results gives you a rich shortlist.

CATALOG GAP HANDLING:
- If no exact test exists for a specific skill (e.g. Rust, Kotlin, Go), say so explicitly.
- Then suggest the closest proxy: Smart Interview Live Coding for any language, plus relevant systems/networking tests.
- Never pretend a test exists when it doesn't.

INSTRUCTIONS:
1. RECOMMEND: Commit to a shortlist of 1–10 items once you have enough context.
2. REFINE: Update the shortlist if the user adds/changes constraints — carry forward unchanged items.
3. COMPARE: Explain differences using ONLY data from tool results (duration, languages, description, remote/adaptive).
4. STAY IN SCOPE: Only discuss SHL assessments. Politely refuse unrelated requests.
5. NO HALLUCINATION: Every name and URL MUST come verbatim from tool results.
6. TURN LIMIT: On turn 7+, commit to a best-effort shortlist immediately.

OUTPUT FORMAT — respond ONLY with a JSON object (markdown fences are fine):
{
  "reply": "<conversational message — always non-empty>",
  "recommendations": null | [{"name": "...", "url": "...", "test_type": "..."}],
  "end_of_conversation": true | false
}

FIELD RULES:
- "reply": always non-empty.
- "recommendations": null while clarifying; an array of 1–10 items when you have a shortlist.
- "test_type": copy EXACTLY from the tool result — never invent.
- "end_of_conversation": set to TRUE when the user expresses satisfaction (says "perfect", "thanks", "confirmed", "that works", "that covers it", "good", etc.) or clearly signals they are done.
"""

    def _format_tool_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        keys = item.get("keys", [])
        langs = item.get("languages", [])
        lang_display = ", ".join(langs[:4])
        if len(langs) > 4:
            lang_display += f" (+{len(langs) - 4} more)"
        return {
            "name": item.get("name"),
            "url": item.get("link"),
            "test_type": resolve_test_type(keys),
            "keys": keys,
            "job_levels": item.get("job_levels", []),
            "description": item.get("description", ""),
            "duration": item.get("duration", ""),
            "languages": lang_display,
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
        }

    async def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if not self.client:
            return {
                "reply": "OPENROUTER_API_KEY is not configured.",
                "recommendations": None,
                "end_of_conversation": False,
            }

        capped_messages = messages[-MAX_TURNS:]
        last_user_content = capped_messages[-1].get("content", "") if capped_messages else ""

        # Detect satisfaction signals early — skip LLM round-trip
        last_is_eoc = _is_eoc_signal(last_user_content)

        oai_messages = [{"role": "system", "content": self.system_prompt}]
        for m in capped_messages:
            oai_messages.append({"role": m["role"], "content": m["content"]})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_catalog",
                    "description": (
                        "Search the SHL catalog for assessments matching a query. "
                        "IMPORTANT: call this multiple times with different narrow queries to cover "
                        "different assessment categories (skills, cognitive, personality, simulation). "
                        "Do NOT use one broad query — issue separate calls per category."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Specific search terms matched to what you need. Use these patterns:\n"
                                    "- Skills/knowledge: 'Java programming', 'financial accounting', 'networking implementation', 'Linux programming'\n"
                                    "- Cognitive/ability: 'numerical reasoning', 'verbal reasoning', 'inductive reasoning', 'deductive reasoning', 'Verify G+'\n"
                                    "- Personality/behaviour: 'OPQ personality', 'OPQ leadership report', 'OPQ32r'\n"
                                    "- Simulations: 'contact center call simulation', 'SVAR spoken English US', 'customer service simulation'\n"
                                    "- Live coding: 'smart interview live coding'\n"
                                    "- Entry level: 'entry level customer service', 'entry level cashier', 'graduate scenarios situational judgement'\n"
                                    "- Leadership: 'enterprise leadership report', 'OPQ leadership executive', 'HiPo high potential'\n"
                                    "- Graduate: 'graduate scenarios', 'graduate financial', 'basic statistics graduate'"
                                ),
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results (default 5, max 10)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        # Force a tool call when we have enough context to recommend
        last_lower = last_user_content.lower()
        needs_tool = len(capped_messages) >= 2 or any(
            kw in last_lower for kw in [
                "recommend", "assess", "test", "evaluat", "hire", "hiring",
                "developer", "engineer", "manager", "analyst", "designer",
                "architect", "sales", "graduate", "entry level", "senior",
                "junior", "leadership", "executive", "contact centre", "contact center",
            ]
        )
        # Don't force tool call if user is just wrapping up
        if last_is_eoc and len(capped_messages) > 2:
            needs_tool = False
        tool_choice = "required" if needs_tool else "auto"

        response = await self.client.chat.completions.create(
            model=MODEL,
            messages=oai_messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.2,
        )
        response_message = response.choices[0].message

        MAX_TOOL_ROUNDS = 5
        tool_rounds = 0

        while response_message.tool_calls and tool_rounds < MAX_TOOL_ROUNDS:
            tool_rounds += 1
            oai_messages.append(response_message)

            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "search_catalog":
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    results = self.retriever.search(
                        args.get("query", ""), top_k=args.get("top_k", 5)
                    )
                    formatted = [self._format_tool_result(r) for r in results]
                    oai_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "search_catalog",
                        "content": json.dumps(formatted),
                    })

            # Determine end_of_conversation for the final-answer injection
            eoc_instruction = (
                ' Set "end_of_conversation": true because the user expressed satisfaction.'
                if last_is_eoc else
                ' Set "end_of_conversation": false unless the user clearly confirmed they are done.'
            )

            oai_messages.append({
                "role": "user",
                "content": (
                    "Now write your final response as a JSON object using ONLY assessments from "
                    "the tool results above — do not invent names or URLs. "
                    'Schema: {"reply": "<summary>", '
                    '"recommendations": [{"name": "...", "url": "...", "test_type": "..."}], '
                    '"end_of_conversation": false}'
                    + eoc_instruction
                )
            })
            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=oai_messages,
                temperature=0.2,
            )
            response_message = response.choices[0].message

        final_content = _extract_json(response_message.content or "")

        try:
            data = json.loads(final_content)
            # Override end_of_conversation if we detected a satisfaction signal
            eoc = data.get("end_of_conversation", False)
            if last_is_eoc and len(capped_messages) > 1:
                eoc = True
            return {
                "reply": data.get("reply", ""),
                "recommendations": data.get("recommendations", None),
                "end_of_conversation": eoc,
            }
        except (json.JSONDecodeError, TypeError):
            return {
                "reply": "I encountered an error processing your request. Please try again.",
                "recommendations": None,
                "end_of_conversation": False,
            }
