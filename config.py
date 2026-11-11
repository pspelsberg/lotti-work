"""
Configuration module for the Lotti Live Voice Agent.

Loads environment variables from a `.env` file (if present) and provides
strongly-typed configuration settings tailored for Windows 11 deployments
with sensible cross-platform fallbacks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        """Fallback .env loader when python-dotenv is not installed."""
        env_path = Path(".env")
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key, _, val = stripped.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))
            return True
        return False

# Load environment variables from .env file if available
load_dotenv()


def _parse_bool(value: str | None, default: bool = True) -> bool:
    """Parse a boolean environment variable string."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_keywords(value: str | None) -> List[str]:
    """Parse a comma-separated list of technical trigger keywords."""
    if not value:
        return [
            "architecture",
            "adr",
            "database",
            "postgres",
            "lancedb",
            "latency",
            "security",
            "compliance",
            "deployment",
            "kubernetes",
            "docker",
            "aws",
            "azure",
            "microservice",
            "api",
            "authentication",
            "oauth",
            "jwt",
            "kafka",
            "redis",
            "constraint",
            "standard",
            "policy",
            "lotti",
        ]
    return [kw.strip().lower() for kw in value.split(",") if kw.strip()]


@dataclass(slots=True)
class LottiConfig:
    """Centralized configuration for Lotti Voice Agent."""

    # Mistral API Credentials & Models
    mistral_api_key: str = field(
        default_factory=lambda: os.getenv("MISTRAL_API_KEY", "")
    )
    mistral_embedding_model: str = field(
        default_factory=lambda: os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("LOTTI_EMBEDDING_DIM", "1024"))
    )

    # Windows 11 Default Paths (configurable via environment variables)
    system_prompt_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "LOTTI_SYSTEM_PROMPT_PATH",
                r"C:\integriti\prompts\system_prompt.md",
            )
        )
    )
    lancedb_uri: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "LOTTI_LANCEDB_PATH",
                r"C:\integriti\lancedb_data",
            )
        )
    )
    lancedb_table_name: str = field(
        default_factory=lambda: os.getenv("LOTTI_LANCEDB_TABLE", "internal_knowledge")
    )

    # RAG & Context Injection Settings
    rag_top_k: int = field(
        default_factory=lambda: int(os.getenv("LOTTI_RAG_TOP_K", "3"))
    )
    rag_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("LOTTI_RAG_SCORE_THRESHOLD", "1.95"))
    )
    seed_sample_data_if_empty: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("LOTTI_SEED_SAMPLE_DATA"), default=True
        )
    )
    trigger_keywords: List[str] = field(
        default_factory=lambda: _parse_keywords(os.getenv("LOTTI_TRIGGER_KEYWORDS"))
    )

    # Audio Interface Settings (DefaultAudioInterface tuning for Windows 11 / Slack)
    echo_cancellation: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("LOTTI_ECHO_CANCELLATION"), default=True
        )
    )
    noise_suppression: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("LOTTI_NOISE_SUPPRESSION"), default=True
        )
    )
    high_pass_filter: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("LOTTI_HIGH_PASS_FILTER"), default=True
        )
    )
    auto_gain_control: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("LOTTI_AUTO_GAIN_CONTROL"), default=True
        )
    )
    output_buffer_size_ms: int = field(
        default_factory=lambda: int(os.getenv("LOTTI_OUTPUT_BUFFER_MS", "300"))
    )

    # Prompt Hot-Reload Settings
    prompt_reload_on_change: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("LOTTI_PROMPT_HOT_RELOAD"), default=True
        )
    )
    prompt_poll_interval_sec: float = field(
        default_factory=lambda: float(os.getenv("LOTTI_PROMPT_POLL_SEC", "5.0"))
    )

    # Logging configuration
    log_level: str = field(
        default_factory=lambda: os.getenv("LOTTI_LOG_LEVEL", "INFO").upper()
    )

    def validate(self) -> None:
        """Validate critical configuration settings before starting a live session."""
        if not self.mistral_api_key:
            raise ValueError(
                "MISTRAL_API_KEY is missing. Set it in your environment or .env file."
            )

    def configure_logging(self) -> None:
        """Configure structured application logging."""
        numeric_level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
