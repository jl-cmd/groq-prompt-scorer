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
import sys
from pathlib import Path

GROQ_PRIMARY_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"
BYPASS_PREFIXES = ("*", "/", "#")
MINIMUM_PROMPT_LENGTH = 15
LOG_FILE = Path.home() / ".claude" / "cache" / "groq-prompt-scorer.log"

SYSTEM_PROMPT = """You classify prompts for an AI coding assistant. Decide if a prompt is specific enough to execute directly or needs clarification first.

<context>
The AI assistant can read files, run code, and make edits autonomously. Vague prompts waste tokens and produce wrong results. Your classification gates whether the assistant asks clarifying questions first or acts immediately. Classify the prompt in isolation -- do not assume the assistant has project context that fills in gaps.
</context>

<clear_criteria>
A prompt is CLEAR when it has ALL of:
1. A specific target: file path, function name, component, class, error message, or line number
2. A defined action: what to do (fix, refactor, add, remove, rename, move, extract)
3. Bounded scope: clear boundaries on what to change and what not to touch
</clear_criteria>

<guide_criteria>
Flag as GUIDE when ANY of these apply:

TASK problems:
- Vague verb with no target ("help me with my code", "fix this", "make it work")
- Multiple unrelated tasks in one prompt ("add auth AND set up email")
- No success criteria for an ambiguous outcome ("make it better", "improve performance")
- Emotional description instead of specifics ("it's completely broken fix everything")

SCOPE problems:
- Feature request with no file scope ("add auth", "build search", "implement payments")
- No boundaries on what should and should not be touched
- Scope too broad ("refactor everything", "clean up the whole app", "rewrite the codebase")
- Framework or stack mention without file-level specificity ("build a Next.js app with Supabase")

CONTEXT problems:
- Greenfield request with no starting state ("build me an API", "create a full stack app")
- No mention of what was already tried when describing a bug
- Implicit reference to prior conversation ("now do the other thing we discussed")

AGENTIC problems:
- Open-ended permission with no stop condition ("do whatever it takes")
- Multi-step task with no checkpoint or review triggers
</guide_criteria>

<output_format>
Respond with JSON only. No other text.

When verdict is "pass":
{"verdict": "pass"}

When verdict is "guide":
{"verdict": "guide", "gaps": ["1-3 short strings naming what is missing"], "suggestion": "one sentence showing a more specific version of the prompt"}
</output_format>

<examples>
User: "fix the typo in README.md"
{"verdict": "pass"}

User: "refactor getUserData in src/lib/api.ts to handle null returns with early exit"
{"verdict": "pass"}

User: "delete the unused CSS classes in src/components/Header.module.css"
{"verdict": "pass"}

User: "rename the variable 'x' to 'userCount' in src/utils/stats.py lines 12-30"
{"verdict": "pass"}

User: "why does test_auth.py::test_login_redirect fail with 302 instead of 200?"
{"verdict": "pass"}

User: "add dark mode"
{"verdict": "guide", "gaps": ["no file scope", "no implementation approach", "no toggle mechanism specified"], "suggestion": "Add dark mode toggle to the settings page, applying CSS custom properties to components in src/components/. Store preference in localStorage."}

User: "build me a full stack app"
{"verdict": "guide", "gaps": ["no stack specified", "no feature scope", "no starting state"], "suggestion": "Scaffold a Next.js app with Express backend. Start with project init and basic routing in src/app/."}

User: "it's broken fix everything"
{"verdict": "guide", "gaps": ["no error message", "no file reference", "no reproduction steps"], "suggestion": "Describe the specific error, which file or endpoint fails, and steps to reproduce."}

User: "add auth and also set up the email system and fix the dashboard layout"
{"verdict": "guide", "gaps": ["multiple unrelated tasks", "no file scope for any task"], "suggestion": "Split into three prompts: (1) add auth to src/app/login/, (2) set up email in src/services/email.ts, (3) fix dashboard layout in src/components/Dashboard.tsx."}

User: "improve the code"
{"verdict": "guide", "gaps": ["no target file", "no success criteria", "no specific improvement"], "suggestion": "Specify which file and what aspect to improve, e.g. 'extract repeated fetch logic in src/api.ts into a shared helper function'."}
</examples>

<constraints>
- Be conservative: only flag prompts that are genuinely underspecified. Short but specific is fine.
- A prompt with a clear file path and action is almost always CLEAR, even if terse.
- Questions and research requests ("how does the auth middleware work?", "explain the caching layer") are CLEAR -- they have a specific target.
- When in doubt between pass and guide, choose pass. False positives (blocking clear prompts) are worse than false negatives.
</constraints>"""


def should_bypass(prompt: str) -> bool:
    if not prompt or len(prompt.strip()) < MINIMUM_PROMPT_LENGTH:
        return True
    if prompt.startswith(BYPASS_PREFIXES):
        return True
    return False


def classify_prompt(prompt: str, api_key: str) -> dict | None:
    from groq import Groq, RateLimitError

    client = Groq(api_key=api_key)

    for model in (GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=150,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            if not content:
                return None
            return json.loads(content)
        except RateLimitError:
            continue

    return None


def log_classification(prompt: str, result: dict, log_path: Path = LOG_FILE) -> None:
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        verdict = result.get("verdict", "unknown")
        gaps = result.get("gaps", [])
        suggestion = result.get("suggestion", "")
        prompt_preview = prompt[:80].replace("\n", " ")
        parts = [f"[{timestamp}] verdict={verdict} prompt=\"{prompt_preview}\""]
        if gaps:
            parts.append(f"gaps={gaps}")
        if suggestion:
            parts.append(f"suggestion=\"{suggestion}\"")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(" ".join(parts) + "\n")
    except OSError:
        pass


def format_guidance(result: dict) -> str:
    gaps = result.get("gaps", [])
    suggestion = result.get("suggestion", "")

    lines = ["PROMPT IMPROVEMENT SUGGESTION: This prompt may be underspecified."]
    if gaps:
        lines.append("Missing:")
        for gap in gaps:
            lines.append(f"  - {gap}")
    if suggestion:
        lines.append(f"Consider: {suggestion}")
    lines.append(
        "Ask the user clarifying questions about the gaps above before proceeding."
    )
    return "\n".join(lines)


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit(0)

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    prompt = input_data.get("user_message", "")
    if should_bypass(prompt):
        sys.exit(0)

    try:
        result = classify_prompt(prompt, api_key)
    except Exception:
        sys.exit(0)

    if not result:
        sys.exit(0)

    log_classification(prompt, result)

    if result.get("verdict") != "guide":
        sys.exit(0)

    print(format_guidance(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
