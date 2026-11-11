"""
Windows 11 Audio Configuration & PyAudio / PortAudio Fallback Utilities.

Configures Mistral's `DefaultAudioInterface` with echo cancellation, noise
suppression, high-pass filtering, and auto-gain control for Slack / team calls
on Windows 11, while providing clear diagnostics and fallback handling if
PortAudio or audio devices are missing.
"""

from __future__ import annotations

import logging
import platform
from typing import Any, Dict, List, Optional

from config import LottiConfig

logger = logging.getLogger("lotti.audio")


class HeadlessFallbackAudioInterface:
    """
    No-op fallback audio interface used only when PyAudio / PortAudio hardware
    is unavailable (e.g., during automated tests or headless server runs).
    """

    def __init__(self) -> None:
        self.is_fallback = True
        self._running = False

    async def start(self, *args: Any, **kwargs: Any) -> None:
        """Start the headless fallback interface."""
        self._running = True
        logger.warning(
            "Running with HeadlessFallbackAudioInterface (no physical mic/speaker bound)."
        )

    async def stop(self, *args: Any, **kwargs: Any) -> None:
        """Stop the headless fallback interface."""
        self._running = False


def inspect_windows_audio_devices() -> List[Dict[str, Any]]:
    """
    Inspect available input/output audio devices via PyAudio (PortAudio).

    Logs helpful diagnostics on Windows 11 (e.g. detecting Virtual Audio Cables
    such as VB-Cable used to bridge Lotti into Slack huddles/calls).
    """
    devices: List[Dict[str, Any]] = []
    try:
        import pyaudio  # type: ignore[import-untyped]

        pa = pyaudio.PyAudio()
        try:
            device_count = pa.get_device_count()
            for idx in range(device_count):
                info = pa.get_device_info_by_index(idx)
                dev_summary = {
                    "index": idx,
                    "name": info.get("name", f"Device {idx}"),
                    "max_input_channels": int(info.get("maxInputChannels", 0)),
                    "max_output_channels": int(info.get("maxOutputChannels", 0)),
                    "default_sample_rate": int(info.get("defaultSampleRate", 48000)),
                }
                devices.append(dev_summary)

            input_devs = [d for d in devices if d["max_input_channels"] > 0]
            output_devs = [d for d in devices if d["max_output_channels"] > 0]
            logger.info(
                "Detected %d audio input device(s) and %d output device(s) on %s.",
                len(input_devs),
                len(output_devs),
                platform.system(),
            )

            # Check for virtual audio cables commonly used for Slack integration on Windows 11
            virtual_cables = [
                d["name"]
                for d in devices
                if any(
                    tag in str(d["name"]).lower()
                    for tag in ("cable", "virtual", "voicemeeter", "blackhole")
                )
            ]
            if virtual_cables:
                logger.info(
                    "Detected virtual audio routing device(s) for Slack calls: %s",
                    ", ".join(virtual_cables),
                )
        finally:
            pa.terminate()
    except ImportError:
        logger.warning(
            "PyAudio is not installed. On Windows 11, run: "
            '`pip install "mistralai[voice-agents]" pyaudio`'
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "PortAudio / PyAudio device enumeration warning: %s. "
            "Ensure a microphone and speaker (or VB-Cable for Slack) are enabled in "
            "Windows 11 Settings -> System -> Sound.",
            exc,
        )

    return devices


def create_audio_interface(
    config: LottiConfig,
    allow_fallback: bool = False,
) -> Any:
    """
    Instantiate Mistral's `DefaultAudioInterface` tuned for live calls on Windows 11.

    Parameters
    ----------
    config : LottiConfig
        Application configuration containing echo cancellation and buffer settings.
    allow_fallback : bool
        If True, returns `HeadlessFallbackAudioInterface` when `DefaultAudioInterface`
        or `PyAudio` cannot be initialized.

    Returns
    -------
    DefaultAudioInterface or HeadlessFallbackAudioInterface
    """
    inspect_windows_audio_devices()

    try:
        from mistralai.extra.voice_agents import DefaultAudioInterface

        # Try initializing with full voice-call DSP parameters supported by recent SDKs,
        # falling back to zero-arg DefaultAudioInterface() if an older SDK version is installed.
        try:
            audio_interface = DefaultAudioInterface(
                echo_cancellation=config.echo_cancellation,
                noise_suppression=config.noise_suppression,
                high_pass_filter=config.high_pass_filter,
                auto_gain_control=config.auto_gain_control,
                output_buffer_size_ms=config.output_buffer_size_ms,
            )
        except TypeError:
            audio_interface = DefaultAudioInterface()

        logger.info(
            "Initialized Mistral DefaultAudioInterface "
            "(echo_cancellation=%s, noise_suppression=%s, buffer=%dms).",
            config.echo_cancellation,
            config.noise_suppression,
            config.output_buffer_size_ms,
        )
        return audio_interface

    except Exception as exc:  # noqa: BLE001
        if allow_fallback:
            logger.warning(
                "Could not initialize DefaultAudioInterface (%s). "
                "Using HeadlessFallbackAudioInterface.",
                exc,
            )
            return HeadlessFallbackAudioInterface()

        raise RuntimeError(
            "Failed to initialize Mistral DefaultAudioInterface on Windows 11. "
            "Verify that `pip install \"mistralai[voice-agents]\"` succeeded and that "
            "Windows 11 Microphone Privacy permissions allow desktop apps to access audio. "
            f"Original error: {exc}"
        ) from exc
