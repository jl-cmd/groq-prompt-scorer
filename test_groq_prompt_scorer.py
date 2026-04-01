"""Tests for groq-prompt-scorer hook."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

HOOK_PATH = Path(__file__).parent / "groq-prompt-scorer.py"
spec = importlib.util.spec_from_file_location("groq_prompt_scorer", HOOK_PATH)
assert spec and spec.loader
groq_prompt_scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(groq_prompt_scorer)

classify_prompt = groq_prompt_scorer.classify_prompt
format_guidance = groq_prompt_scorer.format_guidance
log_classification = groq_prompt_scorer.log_classification
should_bypass = groq_prompt_scorer.should_bypass


class TestShouldBypass:
    def test_empty_prompt_bypasses(self) -> None:
        assert should_bypass("") == "empty"

    def test_single_word_bypasses(self) -> None:
        assert should_bypass("yes") == "single_word"
        assert should_bypass("ok") == "single_word"
        assert should_bypass("thanks") == "single_word"

    def test_two_word_prompt_does_not_bypass(self) -> None:
        assert should_bypass("fix it") == ""
        assert should_bypass("build something") == ""

    def test_asterisk_prefix_bypasses(self) -> None:
        assert should_bypass("* do whatever you want with auth") == "prefix"

    def test_slash_prefix_bypasses(self) -> None:
        assert should_bypass("/commit this change now") == "prefix"

    def test_hash_prefix_bypasses(self) -> None:
        assert should_bypass("# this is a comment") == "prefix"

    def test_specific_prompt_does_not_bypass(self) -> None:
        assert should_bypass("fix the TypeError in src/auth.ts line 42") == ""

    def test_whitespace_only_bypasses(self) -> None:
        assert should_bypass("        ") == "empty"

    def test_affirmation_bypasses(self) -> None:
        assert should_bypass("go ahead") == "affirmation"
        assert should_bypass("sounds good") == "affirmation"

    def test_question_with_command_does_not_bypass(self) -> None:
        assert should_bypass("is our local main updated? if not, update it to be synced") == ""

    def test_pure_question_does_not_bypass(self) -> None:
        assert should_bypass("how does the auth middleware work?") == ""

    def test_question_starter_does_not_bypass(self) -> None:
        assert should_bypass("can you fix the bug in auth.ts?") == ""


class TestFormatGuidance:
    def test_formats_gaps_and_interpretations(self) -> None:
        result = {
            "verdict": "guide",
            "reasoning": "No target file specified.",
            "gaps": ["no file path specified", "no success criteria"],
            "interpretations": [
                {"intent": "Refactor a specific file", "improved_prompt": "Refactor src/auth.ts to extract validation logic into a helper function"},
                {"intent": "Fix a specific bug", "improved_prompt": "Fix the TypeError in src/auth.ts line 42 where user.role is undefined"},
            ],
        }
        output = format_guidance(result)
        assert "Prompt needs refinement" in output
        assert "no file path specified" in output
        assert "no success criteria" in output
        assert "Did you mean one of these?" in output
        assert "Refactor a specific file" in output
        assert "Refactor src/auth.ts" in output
        assert "Fix a specific bug" in output
        assert "resubmit" in output.lower()

    def test_formats_without_gaps(self) -> None:
        result = {
            "verdict": "guide",
            "reasoning": "Ambiguous scope.",
            "gaps": [],
            "interpretations": [
                {"intent": "Possible meaning", "improved_prompt": "Be more specific about the target file."},
            ],
        }
        output = format_guidance(result)
        assert "Missing:" not in output
        assert "Did you mean" in output
        assert "Be more specific" in output

    def test_formats_without_interpretations(self) -> None:
        result = {
            "verdict": "guide",
            "reasoning": "Too vague.",
            "gaps": ["too vague"],
            "interpretations": [],
        }
        output = format_guidance(result)
        assert "too vague" in output
        assert "Did you mean" not in output
        assert "Add the missing details" in output

    def test_formats_interpretations_without_intent(self) -> None:
        result = {
            "verdict": "guide",
            "reasoning": "Unclear.",
            "gaps": ["no target"],
            "interpretations": [
                {"improved_prompt": "Fix the bug in src/auth.ts"},
            ],
        }
        output = format_guidance(result)
        assert "Fix the bug in src/auth.ts" in output


class TestClassifyPrompt:
    def _make_mock_groq(
        self, content: str | None, side_effect: Exception | None = None,
    ) -> MagicMock:
        mock_parsed_response = MagicMock()
        mock_parsed_response.choices = [MagicMock()]
        mock_parsed_response.choices[0].message.content = content
        mock_parsed_response.usage = MagicMock()
        mock_parsed_response.usage.prompt_tokens = 100
        mock_parsed_response.usage.completion_tokens = 50
        mock_parsed_response.usage.total_tokens = 150

        mock_raw_response = MagicMock()
        mock_raw_response.parse.return_value = mock_parsed_response
        mock_raw_response.headers = {}

        mock_client = MagicMock()
        if side_effect:
            mock_client.chat.completions.with_raw_response.create.side_effect = side_effect
        else:
            mock_client.chat.completions.with_raw_response.create.return_value = mock_raw_response
        return mock_client

    def test_returns_parsed_json_from_groq(self) -> None:
        groq_response = {
            "verdict": "guide",
            "gaps": ["no scope"],
            "interpretations": [
                {"intent": "Add dark mode to UI", "improved_prompt": "Add a dark mode toggle to src/components/Settings.tsx"},
            ],
        }
        mock_client = self._make_mock_groq(json.dumps(groq_response))

        with patch("groq.Groq", return_value=mock_client):
            result = classify_prompt("add dark mode", "fake-key", "")

        assert result is not None
        assert result["verdict"] == "guide"
        assert result["gaps"] == ["no scope"]
        assert result["interpretations"] == groq_response["interpretations"]
        assert result["_model"] == groq_prompt_scorer.GROQ_PRIMARY_MODEL
        mock_client.chat.completions.with_raw_response.create.assert_called_once()

    def test_returns_none_on_empty_content(self) -> None:
        mock_client = self._make_mock_groq(None)

        with patch("groq.Groq", return_value=mock_client):
            result = classify_prompt("add dark mode", "fake-key", "")

        assert result is None

    def test_tries_primary_model_first(self) -> None:
        mock_client = self._make_mock_groq('{"verdict": "pass"}')

        with patch("groq.Groq", return_value=mock_client):
            classify_prompt("fix src/auth.ts line 42", "test-key", "")

        call_kwargs = mock_client.chat.completions.with_raw_response.create.call_args[1]
        assert call_kwargs["model"] == groq_prompt_scorer.GROQ_PRIMARY_MODEL
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.1

    def test_falls_back_on_rate_limit(self) -> None:
        from groq import RateLimitError

        mock_parsed_response = MagicMock()
        mock_parsed_response.choices = [MagicMock()]
        mock_parsed_response.choices[0].message.content = '{"verdict": "pass"}'
        mock_parsed_response.usage = MagicMock()
        mock_parsed_response.usage.prompt_tokens = 100
        mock_parsed_response.usage.completion_tokens = 50
        mock_parsed_response.usage.total_tokens = 150

        mock_raw_response = MagicMock()
        mock_raw_response.parse.return_value = mock_parsed_response
        mock_raw_response.headers = {}

        mock_rate_limit_response = MagicMock()
        mock_rate_limit_response.status_code = 429
        mock_rate_limit_response.headers = {}
        mock_rate_limit_response.json.return_value = {
            "error": {"message": "rate limited"},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = [
            RateLimitError(
                message="rate limited",
                response=mock_rate_limit_response,
                body={"error": {"message": "rate limited"}},
            ),
            mock_raw_response,
        ]

        with patch("groq.Groq", return_value=mock_client):
            result = classify_prompt("add dark mode", "fake-key", "")

        assert result is not None
        assert result["verdict"] == "pass"
        calls = mock_client.chat.completions.with_raw_response.create.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["model"] == groq_prompt_scorer.GROQ_PRIMARY_MODEL
        assert calls[1][1]["model"] == groq_prompt_scorer.GROQ_FALLBACK_MODEL

    def test_returns_none_when_both_models_fail(self) -> None:
        from groq import RateLimitError

        mock_rate_limit_response = MagicMock()
        mock_rate_limit_response.status_code = 429
        mock_rate_limit_response.headers = {}
        mock_rate_limit_response.json.return_value = {
            "error": {"message": "rate limited"},
        }

        rate_limit_error = RateLimitError(
            message="rate limited",
            response=mock_rate_limit_response,
            body={"error": {"message": "rate limited"}},
        )

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
        ]

        with patch("groq.Groq", return_value=mock_client):
            result = classify_prompt("add dark mode", "fake-key", "")

        assert result is None


class TestLogClassification:
    def test_writes_log_entry_with_verdict_and_prompt(self, tmp_path: Path) -> None:
        log_file = tmp_path / "groq-prompt-scorer.log"
        result = {"verdict": "guide", "gaps": ["no scope"]}

        log_classification("add dark mode", result, 0, 0, log_file)

        content = log_file.read_text()
        assert "guide" in content
        assert "add dark mode" in content
        assert "no scope" in content

    def test_appends_to_existing_log(self, tmp_path: Path) -> None:
        log_file = tmp_path / "groq-prompt-scorer.log"
        log_file.write_text("existing line\n")

        log_classification("test prompt here", {"verdict": "pass"}, 0, 0, log_file)

        content = log_file.read_text()
        assert "existing line" in content
        assert "pass" in content

    def test_includes_timestamp(self, tmp_path: Path) -> None:
        log_file = tmp_path / "groq-prompt-scorer.log"

        log_classification("test prompt here", {"verdict": "pass"}, 0, 0, log_file)

        content = log_file.read_text()
        assert "2026" in content or "202" in content

    def test_survives_unwritable_path(self) -> None:
        bad_path = Path("/nonexistent/dir/file.log")
        log_classification("test", {"verdict": "pass"}, 0, 0, bad_path)


class TestLogDebug:
    def test_writes_stage_and_detail_to_debug_log(self, tmp_path: Path) -> None:
        debug_log = tmp_path / "debug.log"
        with patch.object(groq_prompt_scorer, "DEBUG_LOG_FILE", debug_log):
            groq_prompt_scorer.log_debug("entry", "stdin received")
        content = debug_log.read_text()
        assert "entry" in content
        assert "stdin received" in content

    def test_appends_to_existing_debug_log(self, tmp_path: Path) -> None:
        debug_log = tmp_path / "debug.log"
        debug_log.write_text("previous\n")
        with patch.object(groq_prompt_scorer, "DEBUG_LOG_FILE", debug_log):
            groq_prompt_scorer.log_debug("second_call", "more data")
        content = debug_log.read_text()
        assert "previous" in content
        assert "second_call" in content

    def test_survives_unwritable_path(self) -> None:
        bad_path = Path("/nonexistent/dir/debug.log")
        with patch.object(groq_prompt_scorer, "DEBUG_LOG_FILE", bad_path):
            groq_prompt_scorer.log_debug("fail_test", "should not crash")


class TestDetectEditor:
    def test_returns_cursor_when_conversation_id_present(self) -> None:
        payload = {"conversation_id": "abc-123", "prompt": "hello"}
        assert groq_prompt_scorer.detect_editor(payload) == "cursor"

    def test_returns_claude_code_when_session_id_present(self) -> None:
        payload = {"session_id": "abc-123", "prompt": "hello"}
        assert groq_prompt_scorer.detect_editor(payload) == "claude_code"

    def test_returns_claude_code_as_default(self) -> None:
        payload = {"prompt": "hello"}
        assert groq_prompt_scorer.detect_editor(payload) == "claude_code"


class TestFormatBlockResponse:
    def test_claude_code_format(self) -> None:
        guidance = "Prompt needs refinement."
        response = groq_prompt_scorer.format_block_response("claude_code", guidance)
        assert response == {"decision": "block", "reason": guidance}

    def test_cursor_format(self) -> None:
        guidance = "Prompt needs refinement."
        response = groq_prompt_scorer.format_block_response("cursor", guidance)
        assert response == {"continue": False, "user_message": guidance}


class TestMainFlow:
    def test_exits_silently_without_api_key(self) -> None:
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            with pytest.raises(SystemExit) as exc_info:
                groq_prompt_scorer.main()
            assert exc_info.value.code == 0

    def test_main_logs_debug_at_entry(self, tmp_path: Path) -> None:
        debug_log = tmp_path / "debug.log"
        with (
            patch.object(groq_prompt_scorer, "DEBUG_LOG_FILE", debug_log),
            patch.dict("os.environ", {"GROQ_API_KEY": ""}),
            pytest.raises(SystemExit),
        ):
            groq_prompt_scorer.main()
        content = debug_log.read_text()
        assert "main_entry" in content

    def test_main_logs_debug_with_stdin_payload(self, tmp_path: Path) -> None:
        import io

        debug_log = tmp_path / "debug.log"
        stdin_payload = json.dumps({"prompt": "fix the bug in auth.py"})
        mock_stdin = io.StringIO(stdin_payload)
        with (
            patch.object(groq_prompt_scorer, "DEBUG_LOG_FILE", debug_log),
            patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}),
            patch("sys.stdin", mock_stdin),
            patch.object(groq_prompt_scorer, "classify_prompt", return_value={"verdict": "pass"}),
            pytest.raises(SystemExit),
        ):
            groq_prompt_scorer.main()
        content = debug_log.read_text()
        assert "stdin_parsed" in content
        assert "fix the bug" in content

    def test_main_logs_debug_on_bypass(self, tmp_path: Path) -> None:
        import io

        debug_log = tmp_path / "debug.log"
        stdin_payload = json.dumps({"prompt": "/commit"})
        mock_stdin = io.StringIO(stdin_payload)
        with (
            patch.object(groq_prompt_scorer, "DEBUG_LOG_FILE", debug_log),
            patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}),
            patch("sys.stdin", mock_stdin),
            pytest.raises(SystemExit),
        ):
            groq_prompt_scorer.main()
        content = debug_log.read_text()
        assert "bypass" in content

    def test_main_parses_stdin_with_utf8_bom(self, tmp_path: Path) -> None:
        import io

        debug_log = tmp_path / "debug.log"
        stdin_payload_with_bom = "\ufeff" + json.dumps({"prompt": "fix the bug in auth.py"})
        mock_stdin = io.StringIO(stdin_payload_with_bom)
        with (
            patch.object(groq_prompt_scorer, "DEBUG_LOG_FILE", debug_log),
            patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}),
            patch("sys.stdin", mock_stdin),
            patch.object(groq_prompt_scorer, "classify_prompt", return_value={"verdict": "pass"}),
            pytest.raises(SystemExit),
        ):
            groq_prompt_scorer.main()
        content = debug_log.read_text()
        assert "stdin_parsed" in content
        assert "fix the bug" in content
        assert "exit_stdin_parse_failed" not in content

    def test_main_parses_stdin_with_cp1252_mangled_bom(self, tmp_path: Path) -> None:
        import io

        debug_log = tmp_path / "debug.log"
        cp1252_bom = "\u00ef\u00bb\u00bf"
        stdin_payload_with_mangled_bom = cp1252_bom + json.dumps({"prompt": "fix the bug in auth.py"})
        mock_stdin = io.StringIO(stdin_payload_with_mangled_bom)
        with (
            patch.object(groq_prompt_scorer, "DEBUG_LOG_FILE", debug_log),
            patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}),
            patch("sys.stdin", mock_stdin),
            patch.object(groq_prompt_scorer, "classify_prompt", return_value={"verdict": "pass"}),
            pytest.raises(SystemExit),
        ):
            groq_prompt_scorer.main()
        content = debug_log.read_text()
        assert "stdin_parsed" in content
        assert "fix the bug" in content
        assert "exit_stdin_parse_failed" not in content
