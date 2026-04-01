#!/usr/bin/env python3
"""Groq-powered prompt clarity scorer.

Advisory UserPromptSubmit hook that classifies prompts via Groq
llama-3.1-8b-instant. Injects guidance context when a prompt is
underspecified so Claude asks clarifying questions before executing.

Fail-open: any error results in silent pass-through.
"""

import datetime
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

GROQ_PRIMARY_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"
BYPASS_PREFIXES = ("*", "/", "#")
MINIMUM_PROMPT_LENGTH = 0
RATE_LIMIT_WARNING_THRESHOLD_PERCENTAGE = 20
MAXIMUM_CONVERSATION_EXCHANGES = 5
MAXIMUM_RESPONSE_PREVIEW_LENGTH = 500
NOTIFICATION_DISPLAY_SECONDS = 3
SUBPROCESS_NO_WINDOW_FLAG = 0x08000000

AFFIRMATION_PATTERNS = frozenset({
    "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "go ahead",
    "do it", "proceed", "approved", "sounds good", "let's go",
    "agreed", "correct", "right", "exactly", "perfect", "confirmed",
    "go for it", "ship it", "lgtm", "that works", "makes sense",
})

FILE_REFERENCE_PATTERN = re.compile(r"[\w./\\-]+\.\w{1,4}")
LOG_FILE = Path.home() / ".claude" / "cache" / "groq-prompt-scorer.log"
DEBUG_LOG_FILE = Path.home() / ".claude" / "cache" / "groq-prompt-scorer-debug.log"


def log_debug(stage: str, detail: str = "") -> None:
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        DEBUG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as debug_file:
            debug_file.write(f"[{timestamp}] {stage}: {detail}\n")
    except OSError:
        pass

SYSTEM_PROMPT = """You classify prompts for an AI coding assistant. Decide if a prompt is specific enough to execute directly or needs clarification first.

<context>
The AI assistant can read files, run code, and make edits autonomously. Vague prompts waste tokens and produce wrong results. Your classification gates whether the assistant proceeds or asks for clarification.

You will receive the recent conversation history (if available) along with the prompt to classify. Use this context to judge clarity -- a prompt that is vague in isolation may be perfectly clear given recent discussion.
</context>

<always_pass>
These prompt types are ALWAYS clear regardless of how terse they are:
- Affirmative responses to a proposal: "yes", "go ahead", "do it", "option 1", "sounds good"
- Questions and research requests: "how does X work?", "explain Y", "what is Z?"
- Requests for links, docs, or information: "link me to the source", "show me the docs"
- Instructions about the current tool or session: "remove that limit", "keep that", "change the config"
- Follow-ups referencing established conversation context: "do the same for the other one", "fix that too"
- Bug reports describing specific behavior: "every time X happens, Y occurs"
</always_pass>

<clear_criteria>
A prompt is CLEAR when it meets ANY of:
1. It has a specific target (file path, function name, error message) AND a defined action
2. The conversation context establishes enough specifics that the prompt's intent is unambiguous
3. It matches an <always_pass> category above
</clear_criteria>

<guide_criteria>
Flag as GUIDE only when ALL of these are true:
1. The prompt lacks a specific target, action, or scope
2. The conversation context does NOT establish the missing specifics
3. The prompt does NOT match any <always_pass> category

Common GUIDE patterns:
- Greenfield feature request with no scope: "add auth", "build search"
- Multiple unrelated tasks: "add auth AND set up email AND fix the dashboard"
- Scope too broad: "refactor everything", "clean up the whole app"
- Emotional instead of specific: "it's completely broken fix everything"
</guide_criteria>

<output_format>
Respond with JSON only. No other text.

When verdict is "pass":
{"reasoning": "one sentence explaining why the prompt is clear", "verdict": "pass"}

When verdict is "guide":
{"reasoning": "one sentence explaining what is missing", "verdict": "guide", "gaps": ["1-3 short strings naming what is missing"]}
</output_format>

<examples>
<example>
<prompt_to_classify>fix the typo in README.md</prompt_to_classify>
{"reasoning": "Specific file and specific action.", "verdict": "pass"}
</example>

<example>
<prompt_to_classify>why does test_auth.py::test_login_redirect fail with 302 instead of 200?</prompt_to_classify>
{"reasoning": "Specific test, specific observed vs expected behavior.", "verdict": "pass"}
</example>

<example>
<recent_conversation>
[User]: The auth middleware in src/middleware/auth.ts is returning 403 for valid tokens
[Assistant]: I found the issue -- the token expiry check on line 42 uses > instead of <. Should I fix it?
</recent_conversation>
<prompt_to_classify>yes, go ahead</prompt_to_classify>
{"reasoning": "Affirmative response to a specific proposed fix in src/middleware/auth.ts line 42.", "verdict": "pass"}
</example>

<example>
<recent_conversation>
[User]: I'm evaluating Serena as an MCP server for semantic code intelligence
[Assistant]: Serena adds LSP-powered tools. It launches a dashboard in your browser when the MCP server starts.
</recent_conversation>
<prompt_to_classify>every time i load a new session, a chrome tab opens with the serena dashboard. can we disable that?</prompt_to_classify>
{"reasoning": "Specific tool (Serena), specific behavior (dashboard opens on session start), clear desired outcome (disable it).", "verdict": "pass"}
</example>

<example>
<recent_conversation>
[User]: I want to add some features to the app
[Assistant]: What features are you thinking about?
</recent_conversation>
<prompt_to_classify>authentication and maybe a dashboard</prompt_to_classify>
{"reasoning": "Two broad features with no file scope or implementation approach. Context does not narrow the scope.", "verdict": "guide", "gaps": ["no file scope for either feature", "no implementation approach", "two unrelated features in one request"]}
</example>

<example>
<prompt_to_classify>build me a full stack app</prompt_to_classify>
{"reasoning": "No stack, no features, no starting state, no context.", "verdict": "guide", "gaps": ["no stack specified", "no feature scope", "no starting state"]}
</example>

<example>
<prompt_to_classify>improve the code</prompt_to_classify>
{"reasoning": "No target file, no success criteria, no context to narrow scope.", "verdict": "guide", "gaps": ["no target file", "no success criteria", "no specific improvement"]}
</example>
</examples>

<constraints>
- Bias toward PASS. False positives (blocking clear prompts) are far worse than false negatives.
- A prompt with a file path and action is ALWAYS clear, even if terse.
- When conversation context is provided, use it to resolve ambiguity BEFORE considering GUIDE.
- Only flag as GUIDE when a prompt is genuinely unactionable -- if you can imagine a reasonable interpretation given context, PASS it.
- When in doubt, PASS.
</constraints>"""


def should_bypass(prompt: str) -> str:
    stripped = prompt.strip()
    if not stripped:
        return "empty"
    if prompt.startswith(BYPASS_PREFIXES):
        return "prefix"
    if " " not in stripped:
        return "single_word"
    normalized = stripped.lower().rstrip(".,!?")
    if normalized in AFFIRMATION_PATTERNS:
        return "affirmation"
    return ""


def extract_text_from_assistant_content(content_blocks: list) -> str:
    text_parts = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    return " ".join(text_parts)


def extract_recent_conversation(transcript_path: str, current_prompt: str) -> list[dict[str, str]]:
    try:
        with open(transcript_path, "r", encoding="utf-8") as transcript_file:
            lines = transcript_file.readlines()
    except (OSError, FileNotFoundError):
        return []

    messages: list[dict[str, str]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        entry_type = entry.get("type")
        if entry_type not in ("user", "assistant"):
            continue

        message = entry.get("message", {})
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user" and isinstance(content, str) and content.strip():
            messages.append({"role": "user", "content": content})
        elif role == "assistant" and isinstance(content, list):
            text = extract_text_from_assistant_content(content)
            if text.strip():
                truncated = text[:MAXIMUM_RESPONSE_PREVIEW_LENGTH]
                if len(text) > MAXIMUM_RESPONSE_PREVIEW_LENGTH:
                    truncated += "..."
                messages.append({"role": "assistant", "content": truncated})

    if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == current_prompt:
        messages = messages[:-1]

    return messages[-(MAXIMUM_CONVERSATION_EXCHANGES * 2):]


def extract_mentioned_entities(messages: list[dict[str, str]]) -> list[str]:
    entities: set[str] = set()
    for message in messages:
        matches = FILE_REFERENCE_PATTERN.findall(message["content"])
        entities.update(matches)
    return sorted(entities)


def extract_topic_summary(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "assistant":
            first_sentence = message["content"].split(". ")[0]
            return first_sentence[:200]
    return ""


def format_classification_input(conversation_messages: list[dict[str, str]], current_prompt: str) -> str:
    parts: list[str] = []
    if conversation_messages:
        topic = extract_topic_summary(conversation_messages)
        entities = extract_mentioned_entities(conversation_messages)

        parts.append("<conversation_context>")
        if topic:
            parts.append(f"<current_topic>{topic}</current_topic>")
        parts.append("<recent_exchanges>")
        for message in conversation_messages:
            role_label = "User" if message["role"] == "user" else "Assistant"
            parts.append(f"[{role_label}]: {message['content']}")
        parts.append("</recent_exchanges>")
        if entities:
            parts.append(f"<mentioned_entities>{', '.join(entities)}</mentioned_entities>")
        parts.append("</conversation_context>")
        parts.append("")
    parts.append(f"<prompt_to_classify>\n{current_prompt}\n</prompt_to_classify>")
    return "\n".join(parts)


def extract_rate_limit_info(headers: Mapping[str, str]) -> dict[str, int]:
    rate_limit_fields = {
        "remaining_requests": "x-ratelimit-remaining-requests",
        "limit_requests": "x-ratelimit-limit-requests",
        "remaining_tokens": "x-ratelimit-remaining-tokens",
        "limit_tokens": "x-ratelimit-limit-tokens",
    }
    info: dict[str, int] = {}
    for field_name, header_name in rate_limit_fields.items():
        raw_value = headers.get(header_name)
        if raw_value is not None:
            try:
                info[field_name] = int(raw_value)
            except (ValueError, TypeError):
                pass
    return info


def check_rate_limit_warning(rate_limits: dict[str, int]) -> str:
    warnings: list[str] = []
    remaining_tokens = rate_limits.get("remaining_tokens", -1)
    limit_tokens = rate_limits.get("limit_tokens", -1)
    remaining_requests = rate_limits.get("remaining_requests", -1)
    limit_requests = rate_limits.get("limit_requests", -1)

    if limit_tokens > 0 and remaining_tokens >= 0:
        token_percentage_remaining = (remaining_tokens / limit_tokens) * 100
        if token_percentage_remaining < RATE_LIMIT_WARNING_THRESHOLD_PERCENTAGE:
            warnings.append(f"Groq tokens: {remaining_tokens:,}/{limit_tokens:,} remaining ({token_percentage_remaining:.0f}%)")

    if limit_requests > 0 and remaining_requests >= 0:
        request_percentage_remaining = (remaining_requests / limit_requests) * 100
        if request_percentage_remaining < RATE_LIMIT_WARNING_THRESHOLD_PERCENTAGE:
            warnings.append(f"Groq requests: {remaining_requests}/{limit_requests} remaining ({request_percentage_remaining:.0f}%)")

    return "; ".join(warnings)


def show_usage_notification(title: str, message: str) -> None:
    try:
        escaped_title = title.replace("'", "''")
        escaped_message = message.replace("'", "''")
        sleep_seconds = NOTIFICATION_DISPLAY_SECONDS + 1
        powershell_script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$notification = New-Object System.Windows.Forms.NotifyIcon;"
            "$notification.Icon = [System.Drawing.SystemIcons]::Information;"
            "$notification.Visible = $true;"
            f"$notification.ShowBalloonTip({NOTIFICATION_DISPLAY_SECONDS * 1000}, '{escaped_title}', '{escaped_message}', 'Info');"
            f"Start-Sleep {sleep_seconds};"
            "$notification.Dispose()"
        )
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden", "-Command", powershell_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=SUBPROCESS_NO_WINDOW_FLAG,
        )
    except OSError:
        pass


def classify_prompt(prompt: str, api_key: str, conversation_context: str) -> dict | None:
    import time

    from groq import Groq, RateLimitError

    client = Groq(api_key=api_key)

    for model in (GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL):
        try:
            start_time = time.monotonic()
            raw_response = client.chat.completions.with_raw_response.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": conversation_context},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=500,
                temperature=0.1,
            )
            elapsed_milliseconds = int((time.monotonic() - start_time) * 1000)
            response = raw_response.parse()
            rate_limits = extract_rate_limit_info(raw_response.headers)

            usage = response.usage
            token_usage = {}
            if usage:
                token_usage = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }

            content = response.choices[0].message.content
            log_debug("groq_response", f"model={model} latency={elapsed_milliseconds}ms tokens={token_usage} rate_limits={rate_limits} raw={content[:500] if content else 'None'}")

            rate_limit_warning = check_rate_limit_warning(rate_limits)
            if rate_limit_warning:
                log_debug("rate_limit_warning", rate_limit_warning)

            if not content:
                return None
            result = json.loads(content)
            result["_model"] = model
            result["_latency_milliseconds"] = elapsed_milliseconds
            result["_token_usage"] = token_usage
            result["_rate_limits"] = rate_limits
            result["_rate_limit_warning"] = rate_limit_warning
            return result
        except RateLimitError:
            log_debug("groq_rate_limit", f"model={model}")
            continue

    return None


def log_classification(prompt: str, result: dict, context_exchange_count: int, context_entity_count: int, log_path: Path = LOG_FILE) -> None:
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        verdict = result.get("verdict", "unknown")
        reasoning = result.get("reasoning", "")
        gaps = result.get("gaps", [])
        model = result.get("_model", "unknown")
        latency = result.get("_latency_milliseconds", 0)
        prompt_preview = prompt[:80].replace("\n", " ")
        token_usage = result.get("_token_usage", {})
        rate_limit_warning = result.get("_rate_limit_warning", "")
        rate_limits = result.get("_rate_limits", {})
        parts = [
            f"[{timestamp}]",
            f"verdict={verdict}",
            f"model={model}",
            f"latency={latency}ms",
            f"tokens={token_usage.get('total_tokens', 0)}({token_usage.get('prompt_tokens', 0)}+{token_usage.get('completion_tokens', 0)})",
            f"exchanges={context_exchange_count}",
            f"entities={context_entity_count}",
            f"prompt=\"{prompt_preview}\"",
        ]
        if reasoning:
            parts.append(f"reasoning=\"{reasoning}\"")
        if gaps:
            parts.append(f"gaps={gaps}")
        if rate_limits:
            parts.append(f"remaining={rate_limits.get('remaining_tokens', '?')}/{rate_limits.get('limit_tokens', '?')}tok {rate_limits.get('remaining_requests', '?')}/{rate_limits.get('limit_requests', '?')}req")
        if rate_limit_warning:
            parts.append(f"WARNING={rate_limit_warning}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(" ".join(parts) + "\n")
    except OSError:
        pass


def format_guidance(result: dict) -> str:
    reasoning = result.get("reasoning", "")
    gaps = result.get("gaps", [])

    lines = ["Prompt blocked: too vague to execute."]
    if reasoning:
        lines.append(f"Reason: {reasoning}")
    lines.append("")
    if gaps:
        lines.append("Missing:")
        for gap in gaps:
            lines.append(f"  - {gap}")
    return "\n".join(lines)


EDITOR_CURSOR = "cursor"
EDITOR_CLAUDE_CODE = "claude_code"


def detect_editor(payload: dict) -> str:
    if "conversation_id" in payload:
        return EDITOR_CURSOR
    return EDITOR_CLAUDE_CODE


def format_block_response(editor: str, guidance: str) -> dict:
    if editor == EDITOR_CURSOR:
        return {"continue": False, "user_message": guidance}
    return {"decision": "block", "reason": guidance}


def main() -> None:
    log_debug("main_entry", f"argv={sys.argv}")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log_debug("exit_no_api_key")
        sys.exit(0)

    try:
        UNICODE_BOM = "\ufeff"
        CP1252_MANGLED_BOM = "\u00ef\u00bb\u00bf"
        raw_stdin = sys.stdin.read().lstrip(UNICODE_BOM).lstrip(CP1252_MANGLED_BOM)
        input_data = json.loads(raw_stdin)
    except (json.JSONDecodeError, EOFError, ValueError):
        log_debug("exit_stdin_parse_failed")
        sys.exit(0)

    log_debug("stdin_parsed", f"keys={list(input_data.keys())} raw={json.dumps(input_data)[:200]}")

    prompt = input_data.get("prompt", "")
    bypass_reason = should_bypass(prompt)
    if bypass_reason:
        log_debug("bypass", f"rule={bypass_reason} prompt={prompt[:80]}")
        sys.exit(0)

    transcript_path = input_data.get("transcript_path", "")
    conversation_messages = extract_recent_conversation(transcript_path, prompt) if transcript_path else []
    classification_input = format_classification_input(conversation_messages, prompt)
    entities = extract_mentioned_entities(conversation_messages)
    topic = extract_topic_summary(conversation_messages)
    log_debug("context_built", f"exchanges={len(conversation_messages)} entities={entities} topic=\"{topic[:100]}\" transcript={transcript_path}")
    log_debug("classification_input", classification_input[:1000])

    try:
        result = classify_prompt(prompt, api_key, classification_input)
    except Exception as classification_error:
        log_debug("exit_classify_error", str(classification_error))
        sys.exit(0)

    if not result:
        log_debug("exit_no_result")
        sys.exit(0)

    log_classification(prompt, result, len(conversation_messages), len(entities))
    log_debug("classified", f"verdict={result.get('verdict')} model={result.get('_model')} latency={result.get('_latency_milliseconds')}ms")

    rate_limit_warning = result.get("_rate_limit_warning", "")
    token_usage = result.get("_token_usage", {})
    rate_limits = result.get("_rate_limits", {})
    latency = result.get("_latency_milliseconds", 0)
    total_tokens = token_usage.get("total_tokens", 0)
    remaining_tokens = rate_limits.get("remaining_tokens")
    limit_tokens = rate_limits.get("limit_tokens")

    notification_line = f"Pass ({latency}ms, {total_tokens} tok)"
    if remaining_tokens is not None and limit_tokens:
        notification_line += f" | Quota: {remaining_tokens:,}/{limit_tokens:,}"
    if rate_limit_warning:
        notification_line += f" | {rate_limit_warning}"

    if result.get("verdict") != "guide":
        show_usage_notification("Groq Prompt Scorer", notification_line)
        status_line = f"Prompt clarity: pass ({latency}ms, {total_tokens} tokens)"
        if remaining_tokens is not None and limit_tokens:
            status_line += f" -- Groq quota: {remaining_tokens:,}/{limit_tokens:,} tokens remaining"
        if rate_limit_warning:
            status_line += f" -- {rate_limit_warning}"
        print(status_line)
        sys.exit(0)

    blocked_notification = f"BLOCKED ({latency}ms, {total_tokens} tok)"
    if remaining_tokens is not None and limit_tokens:
        blocked_notification += f" | Quota: {remaining_tokens:,}/{limit_tokens:,}"
    show_usage_notification("Groq Prompt Scorer", blocked_notification)

    editor = detect_editor(input_data)
    guidance = format_guidance(result)
    if rate_limit_warning:
        guidance += f"\n\n[groq-prompt-scorer] {rate_limit_warning}"
    block_response = format_block_response(editor, guidance)
    log_debug("blocking", f"editor={editor} response={block_response}")
    json.dump(block_response, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
