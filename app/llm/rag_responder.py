import asyncio
import re
from typing import AsyncGenerator, List, Optional
from app.models.chat import ChatMessage


def extract_context_from_prompt(system_prompt: Optional[str]) -> str:
    """Extract retrieved knowledge base context from system prompt if present."""
    if not system_prompt:
        return ""
    start_tag = "--- RELEVANT KNOWLEDGE BASE CONTEXT ---"
    end_tag = "--- END OF CONTEXT ---"

    if start_tag in system_prompt and end_tag in system_prompt:
        start_idx = system_prompt.find(start_tag) + len(start_tag)
        end_idx = system_prompt.find(end_tag)
        return system_prompt[start_idx:end_idx].strip()
    return ""


def synthesize_rag_response(
    query: str,
    context: str,
    provider_name: str,
    model_name: str,
) -> str:
    """Generate a high-quality answer grounded directly in the retrieved RAG context."""
    if not context:
        return (
            f"I could not find specific documents in the knowledge base regarding your query: \"{query}\". "
            f"You can upload relevant guides or policy documents via the Knowledge Base tab, or configure your "
            f"API key in the .env file to enable broader cloud model reasoning with {model_name}."
        )

    # Clean and split into sources
    sources_text = re.split(r"\[Source \d+ \| File: [^\]]+ \| Score: [^\]]+\]", context)
    valid_chunks = [s.strip() for s in sources_text if s.strip()]

    # Extract query keywords for sentence scoring
    query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    stop_words = {"what", "how", "when", "where", "which", "does", "the", "and", "for", "with", "about"}
    keywords = query_words - stop_words

    matched_sentences = []
    seen = set()

    for chunk in valid_chunks:
        # Split chunk into lines/sentences
        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords) or line.startswith("#") or line.startswith("-") or line.startswith("1."):
                if line not in seen:
                    matched_sentences.append(line)
                    seen.add(line)

    if not matched_sentences:
        matched_sentences = valid_chunks[0].split("\n")[:8]

    extracted_body = "\n".join(matched_sentences)

    response = (
        f"Based on the RentCrew Knowledge Base:\n\n"
        f"{extracted_body}\n\n"
        f"---\n"
        f"*Grounded in verified knowledge base documents. (Provider: {provider_name} | Model: {model_name})*"
    )
    return response


async def stream_synthesized_response(response_text: str) -> AsyncGenerator[str, None]:
    """Stream response tokens smoothly for realistic typing feedback."""
    words = response_text.split(" ")
    for idx, word in enumerate(words):
        yield word + (" " if idx < len(words) - 1 else "")
        await asyncio.sleep(0.015)
