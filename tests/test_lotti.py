"""
Automated unit tests for Lotti Voice Agent components:
1. SystemPromptLoader dynamic loading & hot-reloading
2. RAGManager LanceDB seeding, trigger detection, and async retrieval
3. LottiVoiceAgent live transcript RAG pipeline & function tools
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_utils import HeadlessFallbackAudioInterface, create_audio_interface
from config import LottiConfig
from main import LottiVoiceAgent
from prompt_loader import SystemPromptLoader
from rag_manager import RAGManager


class TestLottiVoiceAgent(unittest.IsolatedAsyncioTestCase):
    """End-to-end unit tests for Lotti's modular components."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.prompt_path = self.base_path / "prompts" / "system_prompt.md"
        self.lancedb_path = self.base_path / "lancedb_data"

        self.config = LottiConfig(
            mistral_api_key="test-mistral-api-key",
            system_prompt_path=self.prompt_path,
            lancedb_uri=self.lancedb_path,
            rag_top_k=2,
            prompt_poll_interval_sec=0.05,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prompt_loader_bootstrap_and_hot_reload(self) -> None:
        """Verify that SystemPromptLoader creates the markdown file and detects edits."""
        loader = SystemPromptLoader(prompt_path=self.prompt_path, create_if_missing=True)
        self.assertTrue(self.prompt_path.exists())
        initial_text = loader.load()
        self.assertIn("Lotti", initial_text)

        # Modify file on disk and ensure hot-reload picks up the change
        time.sleep(0.01)
        updated_markdown = "# Custom Prompt\nYou are Lotti v2 with strict ADR checks."
        self.prompt_path.write_text(updated_markdown, encoding="utf-8")

        self.assertTrue(loader.has_changed())
        reloaded_text = loader.load()
        self.assertEqual(reloaded_text, updated_markdown)

    async def test_rag_manager_lancedb_retrieval(self) -> None:
        """Verify local LanceDB connection, ADR seeding, and vector retrieval."""
        rag = RAGManager(
            db_uri=self.lancedb_path,
            table_name="test_knowledge",
            trigger_keywords=self.config.trigger_keywords,
        )
        rag.connect(seed_if_empty=True)

        # Should trigger on technical keywords or questions
        self.assertTrue(
            rag.should_trigger_rag("Can we use MongoDB instead of Postgres for this service?")
        )
        self.assertFalse(rag.should_trigger_rag("Good morning everyone"))

        docs = await rag.retrieve_async("PostgreSQL relational database standard", top_k=2)
        self.assertGreaterEqual(len(docs), 1)
        top_ids = [d.doc_id for d in docs]
        self.assertIn("ADR-001", top_ids)

    async def test_transcript_pipeline_updates_working_memory(self) -> None:
        """Verify that live user transcripts with technical questions populate working memory."""
        agent = LottiVoiceAgent(config=self.config)
        agent.rag_manager.connect(seed_if_empty=True)

        # Casual greeting should be ignored by RAG
        res_casual = await agent.handle_transcript_event("user", "Hey team, happy Monday.")
        self.assertIsNone(res_casual)
        self.assertEqual(agent.working_memory_rag_context, "")

        # Technical question should trigger LanceDB lookup and update instructions
        res_tech = await agent.handle_transcript_event(
            "user",
            "Should we publish durable domain events to Kafka or Redis?",
        )
        self.assertIsNotNone(res_tech)
        self.assertIn("ADR-", agent.working_memory_rag_context)
        self.assertIn(
            "Active Working Memory (Retrieved Internal Knowledge / ADRs)",
            agent.current_instructions,
        )

    def test_audio_fallback_when_hardware_unavailable(self) -> None:
        """Verify that create_audio_interface falls back cleanly when requested."""
        with patch.dict("sys.modules", {"mistralai.extra.voice_agents": None}):
            iface = create_audio_interface(self.config, allow_fallback=True)
            self.assertIsInstance(iface, HeadlessFallbackAudioInterface)

    async def test_voice_tools_and_session_lifecycle(self) -> None:
        """Verify client-side voice tools and Mistral Realtime Voice session lifecycle."""
        agent = LottiVoiceAgent(config=self.config)
        agent.rag_manager.connect(seed_if_empty=True)

        tools = agent.build_voice_tools()
        self.assertEqual(len(tools), 2)
        search_tool, memory_tool = tools[0], tools[1]

        # Invoke search_internal_adrs directly as the Mistral voice session would
        tool_result = await search_tool(query="What is our authentication policy for internal APIs?")
        self.assertIn("ADR-003", tool_result)
        self.assertIn("mTLS", tool_result)

        # Verify working memory tool returns both system prompt and retrieved ADR context
        memory_result = await memory_tool()
        self.assertIn("ADR-003", memory_result)
        self.assertIn("Lotti", memory_result)


if __name__ == "__main__":
    unittest.main()
