import asyncio
from typing import AsyncGenerator, List, Optional
import google.generativeai as genai
from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.provider import BaseLLMProvider
from app.llm.rag_responder import (
    extract_context_from_prompt,
    stream_synthesized_response,
    synthesize_rag_response,
)
from app.models.chat import ChatMessage, Role

logger = get_logger(__name__)
settings = get_settings()


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API integration for generation and streaming."""

    def __init__(self, model: Optional[str] = None):
        super().__init__(
            model=model or settings.GEMINI_MODEL,
            provider_name="gemini",
        )
        self.api_key = settings.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def _is_valid_key(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your-") and not self.api_key.startswith("sk-dummy"))

    def _convert_messages(self, messages: List[ChatMessage]) -> List[dict]:
        contents = []
        for msg in messages:
            role = "user" if msg.role == Role.USER else "model"
            contents.append({"role": role, "parts": [msg.content]})
        return contents

    async def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        last_user_query = next((m.content for m in reversed(messages) if m.role == Role.USER), "")

        if not self._is_valid_key():
            context = extract_context_from_prompt(system_prompt)
            if context:
                return synthesize_rag_response(
                    query=last_user_query,
                    context=context,
                    provider_name=self.provider_name,
                    model_name=self.model,
                )
            return (
                f"[Gemini Provider] Hello! To receive live responses from {self.model}, "
                "please configure GEMINI_API_KEY in your .env file."
            )

        def _call_gemini():
            generation_config = genai.types.GenerationConfig(
                temperature=temperature if temperature is not None else 0.7,
                max_output_tokens=settings.OPENAI_MAX_TOKENS,
            )
            model_instance = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
                generation_config=generation_config,
            )
            contents = self._convert_messages(messages)
            response = model_instance.generate_content(contents)
            return response.text

        try:
            return await asyncio.to_thread(_call_gemini)
        except Exception as e:
            logger.error("Error generating completion from Gemini: %s", str(e))
            context = extract_context_from_prompt(system_prompt)
            if context:
                return synthesize_rag_response(
                    query=last_user_query,
                    context=context,
                    provider_name=f"{self.provider_name} (offline fallback)",
                    model_name=self.model,
                )
            raise

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        last_user_query = next((m.content for m in reversed(messages) if m.role == Role.USER), "")

        if not self._is_valid_key():
            context = extract_context_from_prompt(system_prompt)
            if context:
                response_text = synthesize_rag_response(
                    query=last_user_query,
                    context=context,
                    provider_name=self.provider_name,
                    model_name=self.model,
                )
            else:
                response_text = (
                    f"[Gemini Provider] Please set GEMINI_API_KEY in your environment "
                    f"to query {self.model}."
                )
            async for token in stream_synthesized_response(response_text):
                yield token
            return

        def _call_stream():
            generation_config = genai.types.GenerationConfig(
                temperature=temperature if temperature is not None else 0.7,
                max_output_tokens=settings.OPENAI_MAX_TOKENS,
            )
            model_instance = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
                generation_config=generation_config,
            )
            contents = self._convert_messages(messages)
            return model_instance.generate_content(contents, stream=True)

        try:
            stream_response = await asyncio.to_thread(_call_stream)
            for chunk in stream_response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error("Error streaming completion from Gemini: %s", str(e))
            context = extract_context_from_prompt(system_prompt)
            if context:
                response_text = synthesize_rag_response(
                    query=last_user_query,
                    context=context,
                    provider_name=f"{self.provider_name} (offline fallback)",
                    model_name=self.model,
                )
                async for token in stream_synthesized_response(response_text):
                    yield token
            else:
                raise
