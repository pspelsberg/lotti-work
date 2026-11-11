"""
Modular System Prompt Loader for Lotti Voice Agent.

Provides dynamic loading, mtime-based hot-reloading, and working-memory
context composition from a local Markdown file (defaulting to
`C:\\integriti\\prompts\\system_prompt.md` on Windows 11).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger("lotti.prompt")

DEFAULT_SYSTEM_PROMPT = """# Role & Identity
You are **Lotti**, a proactive, senior Staff Architecture & Engineering Voice Agent participating live in internal team calls (via Slack / audio bridge).

## Core Objectives
1. **Active Listening**: Follow technical discussions closely and stay quiet during casual banter or routine status updates unless addressed directly.
2. **Constraint & ADR Verification**: When architectural choices, database selections, security patterns, cloud infrastructure, or API contracts are discussed, verify them against our internal Architecture Decision Records (ADRs) and engineering standards using your knowledge base tools.
3. **Concise, Natural Voice Delivery**: Keep spoken responses brief (1 to 3 sentences maximum), conversational, and actionable. Cite the specific ADR ID (e.g., "According to ADR-004...") when flagging a constraint or answering a question.

## Communication Guidelines
- Speak clearly and naturally—avoid reading out markdown formatting, bullet symbols, or raw code blocks aloud.
- If a proposal conflicts with an established ADR, politely highlight the constraint and the original rationale.
- If you are unsure or the knowledge base has no matching record, state that briefly rather than guessing.
"""

ReloadCallback = Callable[[str], Optional[Awaitable[None]]]


class SystemPromptLoader:
    """
    Loads and hot-reloads the agent's system instructions from a Markdown file.

    Automatically detects file modifications via nanosecond modification timestamps
    (`st_mtime_ns`) and supports optional background polling notifications.
    """

    def __init__(
        self,
        prompt_path: Path | str,
        create_if_missing: bool = True,
        fallback_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._prompt_path = Path(prompt_path)
        self._fallback_prompt = fallback_prompt.strip()
        self._cached_content: str = ""
        self._last_mtime_ns: Optional[int] = None
        self._callbacks: List[ReloadCallback] = []

        if create_if_missing:
            self._ensure_prompt_file_exists()

        # Perform initial load
        self.load(force=True)

    @property
    def prompt_path(self) -> Path:
        """Return the configured path to the system prompt Markdown file."""
        return self._prompt_path

    def _ensure_prompt_file_exists(self) -> None:
        """Create the prompt directory and default Markdown file if they do not exist."""
        try:
            if not self._prompt_path.exists():
                self._prompt_path.parent.mkdir(parents=True, exist_ok=True)
                self._prompt_path.write_text(
                    self._fallback_prompt + "\n", encoding="utf-8"
                )
                logger.info(
                    "Created default system prompt file at: %s", self._prompt_path
                )
        except OSError as exc:
            logger.warning(
                "Could not bootstrap default prompt at '%s' (%s). "
                "Using in-memory fallback prompt until file becomes accessible.",
                self._prompt_path,
                exc,
            )

    def has_changed(self) -> bool:
        """Check whether the Markdown file on disk has been modified since last read."""
        try:
            if not self._prompt_path.exists():
                return False
            current_mtime_ns = self._prompt_path.stat().st_mtime_ns
            return (
                self._last_mtime_ns is None
                or current_mtime_ns != self._last_mtime_ns
            )
        except OSError as exc:
            logger.debug("Unable to stat prompt file '%s': %s", self._prompt_path, exc)
            return False

    def load(self, force: bool = False) -> str:
        """
        Read the system prompt from disk if `force=True` or if the file changed.

        Returns the current system prompt string.
        """
        if not force and not self.has_changed() and self._cached_content:
            return self._cached_content

        try:
            if self._prompt_path.exists():
                stat_info = self._prompt_path.stat()
                raw_text = self._prompt_path.read_text(encoding="utf-8").strip()
                if raw_text:
                    self._cached_content = raw_text
                    self._last_mtime_ns = stat_info.st_mtime_ns
                    logger.info(
                        "Loaded system prompt from '%s' (%d chars).",
                        self._prompt_path,
                        len(self._cached_content),
                    )
                    return self._cached_content
                logger.warning(
                    "System prompt file '%s' is empty. Retaining previous/fallback prompt.",
                    self._prompt_path,
                )
            else:
                logger.warning(
                    "System prompt file '%s' not found. Using fallback prompt.",
                    self._prompt_path,
                )
        except OSError as exc:
            logger.error(
                "Failed to read system prompt from '%s': %s", self._prompt_path, exc
            )

        if not self._cached_content:
            self._cached_content = self._fallback_prompt

        return self._cached_content

    def build_instructions(self, rag_context: Optional[str] = None) -> str:
        """
        Return the latest system prompt, optionally enriched with retrieved
        LanceDB RAG context for the active turn/working memory.
        """
        base_prompt = self.load(force=False)
        if not rag_context or not rag_context.strip():
            return base_prompt

        return (
            f"{base_prompt}\n\n"
            "## Active Working Memory (Retrieved Internal Knowledge / ADRs)\n"
            "Use the following verified internal context if relevant to the current turn:\n"
            f"{rag_context.strip()}\n"
        )

    def on_reload(self, callback: ReloadCallback) -> None:
        """Register a callback invoked whenever the prompt file changes on disk."""
        self._callbacks.append(callback)

    async def watch_loop(
        self, poll_interval_sec: float = 5.0, stop_event: Optional[asyncio.Event] = None
    ) -> None:
        """
        Background async task that watches `system_prompt.md` for modifications
        and triggers registered reload callbacks when updated.
        """
        logger.debug(
            "Starting system prompt watcher on '%s' (interval=%.1fs)",
            self._prompt_path,
            poll_interval_sec,
        )
        while stop_event is None or not stop_event.is_set():
            try:
                await asyncio.sleep(poll_interval_sec)
                if self.has_changed():
                    old_content = self._cached_content
                    new_content = self.load(force=True)
                    if new_content != old_content:
                        logger.info(
                            "Detected update to '%s'; notifying %d listener(s).",
                            self._prompt_path,
                            len(self._callbacks),
                        )
                        for callback in self._callbacks:
                            res = callback(new_content)
                            if asyncio.iscoroutine(res):
                                await res
            except asyncio.CancelledError:
                logger.debug("System prompt watcher cancelled.")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error in system prompt watcher loop: %s", exc)
