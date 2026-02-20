import base64
import hashlib
import json
import mimetypes
import os
from dataclasses import asdict, is_dataclass

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages as _add_messages
from pydantic import ValidationError

#########################################################################
## LLM helpers ##########################################################
#########################################################################

def ensure_human_message_last(messages: list, continuation: str = "Please continue.") -> list:
    """Append a HumanMessage when the last message is an AIMessage.

    GLM-4.6's Jinja2 chat template raises an error when ``enable_thinking``
    is present in the request (even as ``False``) and the last message has the
    assistant role, because it interprets that as an "assistant response
    prefill".  This helper guards every LLM call site by ensuring the message
    list always ends with a user-role message.
    """
    if messages and isinstance(messages[-1], AIMessage):
        return [*messages, HumanMessage(content=continuation)]
    return messages


def invoke_with_validation_retry(llm, messages, *, reasoning=None, max_retries: int = 2):
    """Invoke an LLM and retry when structured output validation fails."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            if reasoning is None:
                result = llm.invoke(messages)
            else:
                result = llm.invoke(messages, reasoning=reasoning)
            # When include_raw=True, parsing failures are stored in result["parsing_error"]
            # and result["parsed"] is set to None instead of raising. Surface the error
            # so the retry logic can handle it.
            if isinstance(result, dict) and result.get("parsed") is None:
                exc = result.get("parsing_error")
                if exc is not None:
                    raise exc
            return result
        except (ValidationError, OutputParserException) as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
    raise last_exc

def inject_optional_prompt_lines(prompt: str, marker: str, lines: str) -> str:
    """Replace a marker line with optional content without leaving blank lines."""
    replacement = lines.strip()
    out_lines: list[str] = []
    inserted = False
    for line in prompt.splitlines():
        if line.strip() == marker:
            if replacement:
                out_lines.extend(replacement.splitlines())
            inserted = True
            continue
        out_lines.append(line.replace(marker, replacement))
    if not inserted:
        return prompt.replace(marker, replacement)
    return "\n".join(out_lines)

def build_json_schema_prompt(model: Type) -> str:
    """Return a prompt snippet telling the model what JSON schema to output.

    Required when using ``method="json_mode"`` with ``with_structured_output``,
    because LangChain does not automatically inject the schema into the prompt
    in that mode — it only sets ``response_format={"type": "json_object"}``.
    """
    schema = json.dumps(model.model_json_schema(), indent=2)
    return (
        "<output_format>\n"
        "Respond ONLY with a valid JSON object. "
        "Do NOT include any prose, markdown, or text outside the JSON object.\n"
        "The JSON object must conform exactly to this JSON schema:\n"
        f"{schema}\n"
        "</output_format>"
    )

def float_env(name: str, default: float) -> float:
    """Parse a float from the environment, gracefully falling back to default."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

#########################################################################
## Content blocks #######################################################
#########################################################################

def pdf_to_content_block(file_path: str, filename: str | None = None) -> dict:
    """Load a PDF from disk and return a file content block payload."""
    if filename is None:
        filename = os.path.basename(file_path) or "document.pdf"

    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "type": "file",
        "base64": b64,
        "mime_type": "application/pdf",
        "filename": filename,
    }

def image_to_content_block(file_path: str) -> dict:
    """Load an image from disk and return an image content block payload."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None or not mime_type.startswith("image/"):
        # fall back
        mime_type = "image/png"

    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "type": "image",
        "base64": b64,
        "mime_type": mime_type
    }

#########################################################################
## Response parsing #####################################################
#########################################################################

def response_to_text(response):
    """
    Extracts visible text from response.content (OpenAI Responses / reasoning API)
    and returns it as a single string.
    """
    # Get the content list whether it's an attribute or a dict key
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content", [])
    if content is None:
        return ""
    
    if isinstance(content, str):
        return content

    texts = []

    for part in content:
        # Handle both dict-like and attribute-like parts
        part_type = getattr(part, "type", None)
        if part_type is None and isinstance(part, dict):
            part_type = part.get("type")

        if part_type == "text":
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                texts.append(text)

    # Join multiple text parts with a newline (adjust if you prefer spaces)
    return "\n".join(texts)

#########################################################################
## Serialization ########################################################
#########################################################################

def save_state_atomic(state, out_path: Path | str):
    """Serialize state to JSON and write atomically to the target path."""
    if isinstance(out_path, str):
        out_path = Path(out_path)

    payload = {
        "saved_at": datetime.now(UTC).isoformat() + "Z",
        "values": _serialize_value(state),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)

def _serialize_value(obj: Any) -> Any:
    """Recursively convert objects into JSON-safe structures."""
    if isinstance(obj, BaseMessage):
        return _serialize_message(obj)
    if isinstance(obj, list):
        return [_serialize_value(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return asdict(obj)
    return obj

def _serialize_message(msg: BaseMessage) -> Dict[str, Any]:
    """Best-effort serialization for BaseMessage with useful metadata."""
    try:
        return msg.to_dict()
    except Exception:
        payload = {
            "type": msg.__class__.__name__,
            "content": getattr(msg, "content", None),
            "name": getattr(msg, "name", None),
            "additional_kwargs": getattr(msg, "additional_kwargs", {}),
        }
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id is not None:
            payload["tool_call_id"] = tool_call_id
        usage_metadata = getattr(msg, "usage_metadata", None)
        if usage_metadata is not None:
            payload["usage_metadata"] = usage_metadata
        response_metadata = getattr(msg, "response_metadata", None)
        if response_metadata is not None:
            payload["response_metadata"] = response_metadata
        return payload

#########################################################################
## Token accounting #####################################################
#########################################################################

def extract_output_token_counts(msg: BaseMessage) -> dict[str, int] | None:
    """Extract output token counts from a message, if available."""
    usage = getattr(msg, "usage_metadata", None) or {}
    output_tokens = usage.get("output_tokens")
    if not isinstance(output_tokens, int) or output_tokens <= 0:
        return None
    details = usage.get("output_token_details") or {}
    reasoning_tokens = details.get("reasoning") if isinstance(details, dict) else None
    reasoning_tokens_valid = isinstance(reasoning_tokens, int) and reasoning_tokens >= 0
    non_reasoning_tokens = (
        max(output_tokens - reasoning_tokens, 0)
        if reasoning_tokens_valid
        else None
    )
    result = {"output_tokens": output_tokens}
    if reasoning_tokens_valid:
        result["reasoning_output_tokens"] = reasoning_tokens
        result["non_reasoning_output_tokens"] = non_reasoning_tokens or 0
    return result

def extract_non_reasoning_output_tokens(msg: BaseMessage) -> int | None:
    """Return non-reasoning output token count when present."""
    counts = extract_output_token_counts(msg)
    if not counts:
        return None
    return counts.get("non_reasoning_output_tokens")

def annotate_output_token_counts(
    msg: BaseMessage,
    *,
    include_reasoning: bool = True,
    include_total: bool = True,
) -> dict | None:
    """Attach token counts to message.additional_kwargs for downstream use."""
    counts = extract_output_token_counts(msg)
    if not counts:
        return None
    additional_kwargs = dict(getattr(msg, "additional_kwargs", None) or {})
    if include_total:
        additional_kwargs["output_tokens"] = counts["output_tokens"]
    non_reasoning = counts.get("non_reasoning_output_tokens")
    if isinstance(non_reasoning, int):
        additional_kwargs["non_reasoning_output_tokens"] = non_reasoning
    if include_reasoning:
        reasoning = counts.get("reasoning_output_tokens")
        if isinstance(reasoning, int):
            additional_kwargs["reasoning_output_tokens"] = reasoning
    msg.additional_kwargs = additional_kwargs
    return additional_kwargs

def make_summary_fingerprint(message_summary: str | None, non_summary_start: int) -> str:
    """Return a stable hash for summary state comparisons."""
    base = f"{message_summary or ''}|{non_summary_start}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _usage_input_tokens(msg: BaseMessage) -> int | None:
    """Infer input tokens from usage metadata when possible."""
    usage = getattr(msg, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens")
    if isinstance(input_tokens, int) and input_tokens > 0:
        return input_tokens
    total_tokens = usage.get("total_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        isinstance(total_tokens, int)
        and isinstance(output_tokens, int)
        and total_tokens > output_tokens
    ):
        return total_tokens - output_tokens
    return None

def _usage_output_tokens(msg: BaseMessage) -> int | None:
    """Return output tokens from usage metadata when available."""
    usage = getattr(msg, "usage_metadata", None) or {}
    output_tokens = usage.get("output_tokens")
    if isinstance(output_tokens, int) and output_tokens > 0:
        return output_tokens
    return None

#########################################################################
## Message merging ######################################################
#########################################################################

def _impute_intermediate_tokens(messages: list[BaseMessage]) -> None:
    """Estimate missing token counts for the middle message in a 3-msg slice."""
    if len(messages) < 3:
        return
    curr = messages[-1]
    mid = messages[-2]
    prev = messages[-3]
    if not isinstance(curr, AIMessage):
        return
    if not isinstance(mid, (HumanMessage, ToolMessage)):
        return
    if not isinstance(prev, AIMessage):
        return

    prev_fp = (getattr(prev, "additional_kwargs", None) or {}).get("summary_fingerprint")
    curr_fp = (getattr(curr, "additional_kwargs", None) or {}).get("summary_fingerprint")
    if not prev_fp or prev_fp != curr_fp:
        return

    input_prev = _usage_input_tokens(prev)
    input_curr = _usage_input_tokens(curr)
    output_prev = _usage_output_tokens(prev)
    if input_prev is None or input_curr is None or output_prev is None:
        return

    imputed = input_curr - input_prev - output_prev
    if imputed <= 0:
        return

    additional_kwargs = dict(getattr(mid, "additional_kwargs", None) or {})
    additional_kwargs.setdefault("imputed_token_count", imputed)
    mid.additional_kwargs = additional_kwargs

def add_messages_with_token_imputation(left, right):
    """Merge messages and impute token counts for intermediate messages."""
    merged = _add_messages(left, right)
    _impute_intermediate_tokens(merged)
    return merged
