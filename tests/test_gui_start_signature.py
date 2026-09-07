# coding: utf-8
import os
import sys
from pathlib import Path
from typing import Optional, Callable
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.source import AudioSource
from app.audio.recorder import AudioRecorder
from app.pipeline.session_manager import SessionManager


class MockAudioSource(AudioSource):
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._is_active = False
        self._callback = None

    def start(self, callback: Callable[[np.ndarray], None]):
        self._is_active = True
        self._callback = callback

    def stop(self):
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active


def test_session_manager_signature_and_device_index():
    print("=== Testing SessionManager start_session signature ===")
    
    mock_source = MockAudioSource()
    manager = SessionManager()
    
    # 1. Test start_session with course_id and device_index
    session_id = manager.start_session(
        course_id="political_economy",
        device_index=3,
        source=mock_source
    )
    
    assert session_id is not None
    assert manager.recorder.device_index == 3
    assert manager.recorder.is_recording is True
    assert mock_source.is_active is True
    
    # End session
    manager.end_session(timeout=5.0)
    assert manager.recorder.is_recording is False
    assert mock_source.is_active is False
    print("  [PASS] Custom device_index and source passed correctly.")

    # 2. Test start_session with default device_index (None)
    mock_source2 = MockAudioSource()
    session_id2 = manager.start_session(
        course_id="political_economy",
        device_index=None,
        source=mock_source2
    )
    assert session_id2 is not None
    assert manager.recorder.device_index is None
    assert manager.recorder.is_recording is True
    
    manager.end_session(timeout=5.0)
    assert manager.recorder.is_recording is False
    print("  [PASS] Default device_index=None handled correctly.")


if __name__ == "__main__":
    test_session_manager_signature_and_device_index()
    print("All GUI start signature tests passed successfully!")
