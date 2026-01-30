"""Porcupine wake word support."""

import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import pvporcupine
except ImportError:
    pvporcupine = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


def get_system_type() -> str:
    """Get system type (linux or raspberry-pi)."""
    # Simple heuristic: check if running on RPi
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            if "raspberry pi" in model:
                return "raspberry-pi"
    except FileNotFoundError:
        pass
    return "linux"


@dataclass
class PorcupineWakeWord:
    """Wrapper for Porcupine wake word model."""

    id: str
    wake_word: str
    threshold: float
    porcupine: Optional["pvporcupine.Porcupine"]
    access_key: Optional[str] = None
    frame_buffer: bytearray = None  # type: ignore
    frame_size_bytes: int = 0

    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.porcupine is None:
            raise ValueError("Porcupine instance not initialized")
        
        # Get frame size in samples and convert to bytes (16-bit = 2 bytes per sample)
        self.frame_size_bytes = self.porcupine.frame_length * 2
        self.frame_buffer = bytearray()

    @classmethod
    def from_config(cls, config_path: Path, base_path: Path) -> "PorcupineWakeWord":
        """Load Porcupine model from config file.
        
        Args:
            config_path: Path to JSON config file
            base_path: Base directory for resolving relative model paths
            
        Returns:
            PorcupineWakeWord instance
        """
        if pvporcupine is None:
            raise ImportError("pvporcupine is not installed")

        model_id = config_path.stem
        
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        wake_word_text = config["wake_word"]
        threshold = config.get("threshold", 0.5)
        model_file = config["model"]
        language = config.get("language", "en")
        system = config.get("system", get_system_type())
        access_key = config.get("access_key")

        # Resolve model path
        model_path = base_path / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Resolve parameter file path
        access_key_path = base_path / "data" / "lib" / "common" / f"porcupine_params_{language}.pv"
        if not access_key_path.exists():
            _LOGGER.warning(
                "Parameter file not found: %s, trying to use default",
                access_key_path,
            )
            access_key_path = None

        _LOGGER.debug(
            "Loading Porcupine model: %s (language=%s, system=%s, access_key=%s)",
            model_id,
            language,
            system,
            "***" if access_key else "None",
        )

        # Initialize Porcupine
        try:
            create_kwargs = {
                "keyword_paths": [str(model_path)],
                "sensitivities": [threshold],
            }
            
            # Add access_key if provided
            if access_key:
                create_kwargs["access_key"] = access_key
            
            # Add model_path for language parameter file
            if access_key_path and access_key_path.exists():
                create_kwargs["model_path"] = str(access_key_path)
            
            porcupine = pvporcupine.create(**create_kwargs)
        except Exception as e:
            _LOGGER.error("Failed to initialize Porcupine: %s", e)
            raise

        return cls(
            id=model_id,
            wake_word=wake_word_text,
            threshold=threshold,
            porcupine=porcupine,
            access_key=access_key,
        )

    def process_streaming(self, audio_frame: bytes) -> bool:
        """Process audio frame and detect wake word.
        
        Args:
            audio_frame: Audio frame as bytes (16-bit PCM, little-endian)
            
        Returns:
            True if wake word detected, False otherwise
        """
        if self.porcupine is None:
            return False

        try:
            import struct
            import numpy as np

            # Buffer audio until we have enough for a complete frame
            self.frame_buffer.extend(audio_frame)

            # Process all complete frames in the buffer
            while len(self.frame_buffer) >= self.frame_size_bytes:
                # Extract one frame
                frame_bytes = bytes(self.frame_buffer[:self.frame_size_bytes])
                self.frame_buffer = self.frame_buffer[self.frame_size_bytes:]

                # Convert bytes to int16 array (little-endian)
                audio_data = struct.unpack(
                    f"<{len(frame_bytes)//2}h", frame_bytes
                )
                audio_array = np.array(audio_data, dtype=np.int16)

                # Process frame - pvporcupine expects int16 array
                keyword_index = self.porcupine.process(audio_array)

                # keyword_index >= 0 means a keyword was detected
                if keyword_index >= 0:
                    _LOGGER.debug(
                        "Wake word detected: %s (keyword index: %d)",
                        self.wake_word,
                        keyword_index,
                    )
                    return True

            return False

        except Exception as e:
            _LOGGER.error("Error processing audio frame: %s", e)
            return False

    def __del__(self):
        """Cleanup Porcupine resources."""
        if self.porcupine is not None:
            try:
                self.porcupine.delete()
            except Exception:
                pass
        
        # Clear buffer
        if self.frame_buffer is not None:
            self.frame_buffer.clear()
