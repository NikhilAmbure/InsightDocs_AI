import asyncio
import logging
import os
import threading
from django.conf import settings
from anthropic import Anthropic

logger = logging.getLogger(__name__)


def _get_client():
    token = getattr(settings, "ANTHROPIC_AUTH_TOKEN", "") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    base_url = (getattr(settings, "ANTHROPIC_BASE_URL", "") or os.environ.get("ANTHROPIC_BASE_URL", "https://agentrouter.org")).rstrip("/")
    if not token:
        logger.error("ANTHROPIC_AUTH_TOKEN is not configured in settings or environment variables.")
        raise ValueError("ANTHROPIC_AUTH_TOKEN is not configured.")
    return Anthropic(api_key=token, base_url=base_url)


def _get_model_name():
    return getattr(settings, "ANTHROPIC_MODEL", "deepseek-v4-flash")


async def stream_deepseek_chat(system_prompt, messages, max_tokens=3500, usage_callback=None):
    """
    Async generator that streams responses from DeepSeek v4 Flash via AgentRouter Anthropic gateway.
    Emits both Chain-of-Thought (thinking) chunks and final text chunks as structured dictionaries:
      {"type": "thinking", "chunk": "..."}
      {"type": "text", "chunk": "..."}
    Uses Anthropic official client in a thread-safe queue bridge to ensure TLS compatibility with WAF.
    """
    client = _get_client()
    model = _get_model_name()

    # Format messages for Anthropic messages API
    formatted_messages = []
    for msg in messages:
        role = "user" if msg.get("role") in ["user", "human"] else "assistant"
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
            content = "\n".join(filter(None, text_parts))
        if str(content).strip():
            formatted_messages.append({"role": role, "content": str(content)})

    if not formatted_messages:
        formatted_messages = [{"role": "user", "content": "Hello"}]

    stream_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": formatted_messages,
    }
    if system_prompt:
        stream_kwargs["system"] = system_prompt

    q = asyncio.Queue()
    loop = asyncio.get_running_loop()
    received_any_chunk = False

    def worker():
        try:
            with client.messages.stream(**stream_kwargs) as stream:
                for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "thinking_delta":
                            th = getattr(delta, "thinking", "")
                            if th:
                                loop.call_soon_threadsafe(q.put_nowait, ("thinking", th))
                        elif delta_type == "text_delta":
                            tx = getattr(delta, "text", "")
                            if tx:
                                loop.call_soon_threadsafe(q.put_nowait, ("text", tx))

                msg = stream.get_final_message()
                in_tok = getattr(msg.usage, "input_tokens", 0) or 0
                out_tok = getattr(msg.usage, "output_tokens", 0) or 0
                loop.call_soon_threadsafe(q.put_nowait, ("usage", in_tok + out_tok))
        except Exception as e:
            logger.warning(f"DeepSeek v4 Flash worker failed: {e}")
            loop.call_soon_threadsafe(q.put_nowait, ("error", e))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, ("done", None))

    threading.Thread(target=worker, daemon=True).start()

    try:
        while True:
            event_type, val = await q.get()
            if event_type == "thinking":
                received_any_chunk = True
                yield {"type": "thinking", "chunk": val}
            elif event_type == "text":
                received_any_chunk = True
                yield {"type": "text", "chunk": val}
            elif event_type == "usage":
                if usage_callback and val > 0:
                    try:
                        await usage_callback(val)
                    except Exception as ue:
                        logger.warning(f"Usage callback error: {ue}")
            elif event_type == "error":
                if not received_any_chunk:
                    raise val
                break
            elif event_type == "done":
                break
    except Exception as e:
        logger.warning(f"DeepSeek v4 Flash stream failed: {e}")
        if not received_any_chunk:
            raise e


async def call_deepseek_chat(system_prompt, messages, max_tokens=3000):
    """
    Non-streaming call to DeepSeek v4 Flash via AgentRouter Anthropic SDK.
    """
    client = _get_client()
    model = _get_model_name()

    formatted_messages = []
    for msg in messages:
        role = "user" if msg.get("role") in ["user", "human"] else "assistant"
        content = msg.get("content", "")
        if str(content).strip():
            formatted_messages.append({"role": role, "content": str(content)})

    create_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": formatted_messages,
    }
    if system_prompt:
        create_kwargs["system"] = system_prompt

    def _call_sync():
        resp = client.messages.create(**create_kwargs)
        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text" and hasattr(block, "text") and block.text:
                text_parts.append(block.text)
        return "".join(text_parts) if text_parts else ""

    return await asyncio.to_thread(_call_sync)
