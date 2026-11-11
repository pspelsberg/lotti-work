"""
Main Entry Point for "Lotti" - Live Voice Agent with Local LanceDB RAG (Windows 11).

Integrates:
1. Dynamic Markdown System Prompt Loader (`C:\\integriti\\prompts\\system_prompt.md`)
2. Local LanceDB RAG Pipeline (`C:\\integriti\\lancedb_data`)
3. Mistral Realtime Voice Agents SDK (`client.audio.sessions.init` & `DefaultAudioInterface`)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections import deque
from typing import Any, Callable, Deque, List, Optional

try:
    from mistralai.client import Mistral
except ImportError:
    Mistral = None  # type: ignore[assignment,misc]

from audio_utils import create_audio_interface
from config import LottiConfig
from prompt_loader import SystemPromptLoader
from rag_manager import RAGManager

logger = logging.getLogger("lotti.main")

# Import `function_tool` from mistralai.extra.voice_agents with a graceful
# fallback decorator if the optional extra is not yet installed.
try:
    from mistralai.extra.voice_agents import function_tool
except ImportError:

    def function_tool(  # type: ignore[misc]
        *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Fallback decorator when `mistralai[voice-agents]` extra is absent."""

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func._is_fallback_function_tool = True  # type: ignore[attr-defined]
            func._tool_metadata = kwargs  # type: ignore[attr-defined]
            return func

        if args and callable(args[0]):
            return _decorator(args[0])
        return _decorator


class LottiVoiceAgent:
    """
    Production Voice Agent orchestrating Mistral Realtime Voice sessions,
    live transcript monitoring, hot-reloadable Markdown prompts, and LanceDB RAG.
    """

    def __init__(
        self,
        config: LottiConfig,
        prompt_loader: Optional[SystemPromptLoader] = None,
        rag_manager: Optional[RAGManager] = None,
    ) -> None:
        self.config = config
        self.prompt_loader = prompt_loader or SystemPromptLoader(
            prompt_path=config.system_prompt_path,
            create_if_missing=True,
        )
        self.rag_manager = rag_manager or RAGManager(
            db_uri=config.lancedb_uri,
            table_name=config.lancedb_table_name,
            embedding_model=config.mistral_embedding_model,
            embedding_dim=config.embedding_dimension,
            trigger_keywords=config.trigger_keywords,
        )

        # Rolling working memory for recent transcript snippets and RAG context
        self.transcript_history: Deque[str] = deque(maxlen=25)
        self.working_memory_rag_context: str = ""
        self.current_instructions: str = ""
        self._session: Optional[Any] = None
        self._stop_event = asyncio.Event()
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Register hot-reload callback on the prompt loader
        self.prompt_loader.on_reload(self._on_prompt_reloaded)

    async def _on_prompt_reloaded(self, new_prompt: str) -> None:
        """Handle live updates to `system_prompt.md`."""
        self.current_instructions = self.prompt_loader.build_instructions(
            self.working_memory_rag_context
        )
        logger.info(
            "System prompt hot-reloaded (%d chars). Active working memory synced.",
            len(new_prompt),
        )
        await self._sync_session_instructions()

    async def _sync_session_instructions(self) -> None:
        """
        Push updated instructions/context to the live VoiceAgentSession if the
        installed SDK exposes a runtime update method.
        """
        if self._session is None:
            return

        for method_name in ("update_instructions", "set_instructions", "update_session"):
            updater = getattr(self._session, method_name, None)
            if callable(updater):
                try:
                    res = updater(self.current_instructions)
                    if asyncio.iscoroutine(res):
                        await res
                    logger.debug(
                        "Synced updated instructions to live session via %s().",
                        method_name,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Could not dynamically update session via %s: %s",
                        method_name,
                        exc,
                    )
                break

    def build_voice_tools(self) -> List[Callable[..., Any]]:
        """
        Build client-side `@function_tool` handlers registered with the Mistral
        Voice Agent session so the model can retrieve LanceDB ADR context mid-turn.
        """
        rag = self.rag_manager
        prompt_loader = self.prompt_loader
        top_k = self.config.rag_top_k
        score_threshold = self.config.rag_score_threshold
        agent_ref = self

        @function_tool(
            filler={"text": "Let me check our architecture records."},
            timeout_ms=10000,
        )
        async def search_internal_adrs(query: str) -> str:
            """
            Search internal company documentation and Architecture Decision Records (ADRs)
            in local LanceDB when technical topics, databases, security constraints,
            or architecture questions arise in the call.

            Parameters
            ----------
            query : str
                The technical topic, question, or meeting transcript snippet to search for.
            """
            logger.info("Voice tool `search_internal_adrs` invoked with query: '%s'", query)
            context = await rag.retrieve_formatted_context(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            agent_ref.working_memory_rag_context = context
            agent_ref.current_instructions = prompt_loader.build_instructions(context)
            return context

        @function_tool(timeout_ms=5000)
        async def get_active_working_memory() -> str:
            """
            Retrieve the latest pre-fetched RAG context and hot-reloaded system
            guidelines from Lotti's working memory for the current meeting turn.
            """
            latest_prompt = prompt_loader.load(force=False)
            if agent_ref.working_memory_rag_context:
                return (
                    f"Active System Guidelines:\n{latest_prompt}\n\n"
                    f"Pre-fetched LanceDB Context for Current Turn:\n"
                    f"{agent_ref.working_memory_rag_context}"
                )
            return f"Active System Guidelines:\n{latest_prompt}"

        return [search_internal_adrs, get_active_working_memory]

    async def handle_transcript_event(self, role: str, content: str) -> Optional[str]:
        """
        Process a live transcript snippet from the voice session.

        When a user or participant mentions technical keywords or asks a question:
        1. Checks if `system_prompt.md` was modified on disk.
        2. Queries local LanceDB asynchronously for matching ADRs.
        3. Appends the retrieved context to the agent's working memory for that turn.
        """
        cleaned_content = content.strip()
        if not cleaned_content:
            return None

        logger.info("[Transcript] %s: %s", role.upper(), cleaned_content)
        self.transcript_history.append(f"{role}: {cleaned_content}")

        # Only trigger proactive RAG lookup on user/participant speech
        if role.lower() in {"assistant", "agent", "lotti"}:
            return None

        if not self.rag_manager.should_trigger_rag(cleaned_content):
            return None

        logger.info(
            "Technical keyword/question detected in transcript; querying local LanceDB..."
        )
        retrieved_context = await self.rag_manager.retrieve_formatted_context(
            query=cleaned_content,
            top_k=self.config.rag_top_k,
            score_threshold=self.config.rag_score_threshold,
        )

        self.working_memory_rag_context = retrieved_context
        self.current_instructions = self.prompt_loader.build_instructions(
            retrieved_context
        )
        logger.info(
            "Updated turn working memory with LanceDB RAG context:\n%s",
            retrieved_context,
        )
        await self._sync_session_instructions()
        return retrieved_context

    def _register_session_handlers(self, session: Any) -> None:
        """Bind event listeners (`connected`, `transcript`, `error`, tool events) to the session."""

        def on_connected(event: Any) -> None:
            session_id = getattr(event, "session_id", "unknown")
            logger.info("Lotti Voice Session connected! (session_id=%s)", session_id)

        def on_transcript(event: Any) -> None:
            role = str(getattr(event, "role", "user"))
            content = str(getattr(event, "content", ""))
            # Schedule non-blocking RAG enrichment on the running asyncio loop
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(self.handle_transcript_event(role, content))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except RuntimeError:
                pass

        def on_tool_started(event: Any) -> None:
            tool_name = getattr(event, "name", getattr(event, "tool_name", "unknown"))
            logger.info("Executing voice agent tool: %s", tool_name)

        def on_error(event: Any) -> None:
            if hasattr(event, "model_dump_json"):
                err_detail = event.model_dump_json()
            else:
                err_detail = str(event)
            logger.error("Mistral Voice Session error: %s", err_detail)

        session.on("connected", on_connected)
        session.on("transcript", on_transcript)
        session.on("tool_call_started", on_tool_started)
        session.on("error", on_error)

    async def run(self, allow_audio_fallback: bool = False) -> None:
        """
        Initialize LanceDB, load the Markdown system prompt, open the Mistral
        Realtime Voice session, and stream bidirectional audio until stopped.
        """
        self.config.validate()

        # 1. Connect to local LanceDB instance (C:\integriti\lancedb_data)
        self.rag_manager.connect(seed_if_empty=self.config.seed_sample_data_if_empty)

        # 2. Load initial system instructions from Markdown (C:\integriti\prompts\system_prompt.md)
        self.current_instructions = self.prompt_loader.build_instructions()

        # 3. Initialize Windows 11 audio interface (DefaultAudioInterface)
        audio_interface = create_audio_interface(
            self.config, allow_fallback=allow_audio_fallback
        )

        # 4. Start optional background watcher for system_prompt.md hot-reloads
        watcher_task: Optional[asyncio.Task[None]] = None
        if self.config.prompt_reload_on_change:
            watcher_task = asyncio.create_task(
                self.prompt_loader.watch_loop(
                    poll_interval_sec=self.config.prompt_poll_interval_sec,
                    stop_event=self._stop_event,
                )
            )

        if Mistral is None:
            raise RuntimeError(
                'Mistral SDK is not installed. Run: pip install "mistralai[voice-agents]"'
            )

        logger.info("Starting Lotti Realtime Voice Agent with Mistral SDK...")
        try:
            async with Mistral(api_key=self.config.mistral_api_key) as client:
                # Provide Mistral client to RAGManager for optional remote embeddings
                self.rag_manager._mistral_client = client

                session = await client.audio.sessions.init(
                    audio_interface=audio_interface,
                    instructions=self.current_instructions,
                    tools=self.build_voice_tools(),
                )
                self._session = session
                self._register_session_handlers(session)

                async with session:
                    logger.info(
                        "Lotti is now live in the call. Press Ctrl+C to disconnect."
                    )
                    wait_task = asyncio.create_task(session.wait())
                    stop_wait_task = asyncio.create_task(self._stop_event.wait())

                    done, pending = await asyncio.wait(
                        {wait_task, stop_wait_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        if not task.cancelled() and task.exception() is not None:
                            raise task.exception()  # type: ignore[misc]
        finally:
            self._stop_event.set()
            if watcher_task is not None:
                watcher_task.cancel()
                await asyncio.gather(watcher_task, return_exceptions=True)
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._session = None
            logger.info("Lotti Voice Session closed cleanly.")

    def request_shutdown(self) -> None:
        """Signal the running voice session to shut down gracefully."""
        logger.info("Shutdown requested; stopping Lotti voice session...")
        self._stop_event.set()


def _install_signal_handlers(agent: LottiVoiceAgent) -> None:
    """Install cross-platform Ctrl+C / termination signal handlers (Windows 11 + POSIX)."""
    def _handle_signal(sig: int, _frame: Any) -> None:
        logger.info("Received signal %s.", sig)
        agent.request_shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Lotti Voice Agent."""
    parser = argparse.ArgumentParser(
        description="Lotti - Live Voice Agent with Local LanceDB RAG for Windows 11"
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=None,
        help="Override path to system_prompt.md (default: C:\\integriti\\prompts\\system_prompt.md)",
    )
    parser.add_argument(
        "--lancedb-path",
        type=str,
        default=None,
        help="Override path to local LanceDB directory (default: C:\\integriti\\lancedb_data)",
    )
    parser.add_argument(
        "--test-rag",
        type=str,
        default=None,
        help="Run a local LanceDB RAG test query and exit without starting a live voice call.",
    )
    parser.add_argument(
        "--headless-audio-fallback",
        action="store_true",
        help="Allow fallback audio interface if no physical microphone/speaker is available.",
    )
    return parser.parse_args(argv)


async def async_main(argv: Optional[List[str]] = None) -> int:
    """Async entrypoint for CLI execution."""
    args = parse_args(argv)
    config = LottiConfig()

    if args.prompt_path:
        from pathlib import Path

        config.system_prompt_path = Path(args.prompt_path)
    if args.lancedb_path:
        from pathlib import Path

        config.lancedb_uri = Path(args.lancedb_path)

    config.configure_logging()

    agent = LottiVoiceAgent(config=config)
    _install_signal_handlers(agent)

    # Diagnostic mode: test prompt loading and local LanceDB retrieval without a live call
    if args.test_rag:
        agent.rag_manager.connect(seed_if_empty=True)
        context = await agent.rag_manager.retrieve_formatted_context(
            query=args.test_rag,
            top_k=config.rag_top_k,
            score_threshold=config.rag_score_threshold,
        )
        composed = agent.prompt_loader.build_instructions(context)
        print("\n=== Composed System Prompt + Working Memory Context ===")
        print(composed)
        return 0

    try:
        await agent.run(allow_audio_fallback=args.headless_audio_fallback)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Fatal error running Lotti Voice Agent: %s", exc, exc_info=True)
        return 1


def main() -> None:
    """Synchronous CLI wrapper compatible with Windows 11 Proactor/Selector event loops."""
    if sys.platform == "win32":
        # Ensure clean asyncio subprocess/socket transport behavior on Windows 11
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    exit_code = asyncio.run(async_main())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
