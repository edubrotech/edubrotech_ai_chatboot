import pytest
from app.memory.redis import ChatMemory
from app.models.chat import ChatMessage, Role


@pytest.mark.asyncio
async def test_memory_add_and_get():
    memory = ChatMemory()
    session_id = "test-session-123"

    # Clean existing
    await memory.clear_history(session_id)

    msg1 = ChatMessage(role=Role.USER, content="Hello assistant")
    msg2 = ChatMessage(role=Role.ASSISTANT, content="Hello user! How can I help?")

    await memory.add_message(session_id, msg1)
    await memory.add_message(session_id, msg2)

    history = await memory.get_history(session_id)
    assert len(history) == 2
    assert history[0].role == Role.USER
    assert history[0].content == "Hello assistant"
    assert history[1].role == Role.ASSISTANT
    assert history[1].content == "Hello user! How can I help?"

    # Test clear
    cleared = await memory.clear_history(session_id)
    assert cleared is True
    empty_history = await memory.get_history(session_id)
    assert len(empty_history) == 0
