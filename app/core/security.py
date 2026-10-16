import re
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import get_settings

settings = get_settings()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Validate the incoming request API key if API_KEY is configured in settings.
    If no API_KEY is set in settings, authentication is bypassed.
    """
    if not settings.API_KEY:
        return None
    
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return api_key


def sanitize_text(text: str) -> str:
    """Sanitize and normalize text input, removing null characters and extra whitespace."""
    if not text:
        return ""
    # Strip null bytes and non-printable control characters except standard whitespace
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned.strip()
