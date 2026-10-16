import json
from typing import Dict, List, Optional
import redis.asyncio as aioredis
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.chat import ChatMessage, Role

logger = get_logger(__name__)
settings = get_settings()

_redis_client: Optional[aioredis.Redis] = None


def get_redis_client() -> aioredis.Redis:
    """Return singleton async Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=3.0,
            socket_connect_timeout=3.0,
        )
    return _redis_client


class ChatMemory:
    """Redis-backed conversation memory manager with sliding window and memory fallback."""

    def __init__(self, client: Optional[aioredis.Redis] = None):
        self.client = client or get_redis_client()
        self.ttl = settings.MEMORY_TTL_SECONDS
        self.window_size = settings.MEMORY_WINDOW_SIZE * 2  # Turns to messages (user + assistant)
        self._fallback_store: Dict[str, List[ChatMessage]] = {}

    def _key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    async def ping(self) -> bool:
        """Check if Redis connection is alive."""
        try:
            return bool(await self.client.ping())
        except Exception:
            return False

    async def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Append a message to the session's conversational history."""
        key = self._key(session_id)
        msg_json = message.model_dump_json()

        try:
            pipe = self.client.pipeline()
            pipe.rpush(key, msg_json)
            # Trim to max window size (keep latest messages)
            pipe.ltrim(key, -self.window_size, -1)
            # Refresh TTL
            pipe.expire(key, self.ttl)
            await pipe.execute()
        except Exception as e:
            logger.warning("Redis add_message failed (%s), writing to fallback store", str(e))
            if session_id not in self._fallback_store:
                self._fallback_store[session_id] = []
            self._fallback_store[session_id].append(message)
            if len(self._fallback_store[session_id]) > self.window_size:
                self._fallback_store[session_id] = self._fallback_store[session_id][-self.window_size :]

    async def get_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """Retrieve conversation history for a given session."""
        key = self._key(session_id)
        max_items = limit or self.window_size

        try:
            raw_items = await self.client.lrange(key, -max_items, -1)
            messages = []
            for item in raw_items:
                data = json.loads(item)
                messages.append(ChatMessage(**data))
            return messages
        except Exception as e:
            logger.warning("Redis get_history failed (%s), reading from fallback store", str(e))
            items = self._fallback_store.get(session_id, [])
            return items[-max_items:]

    async def clear_history(self, session_id: str) -> bool:
        """Erase conversation history for a session."""
        key = self._key(session_id)
        cleared = False
        try:
            await self.client.delete(key)
            cleared = True
        except Exception as e:
            logger.warning("Redis clear_history error: %s", str(e))

        if session_id in self._fallback_store:
            del self._fallback_store[session_id]
            cleared = True

        return cleared
