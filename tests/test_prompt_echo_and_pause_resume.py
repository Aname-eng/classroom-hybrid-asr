# coding: utf-8
import sys
import time
import shutil
import tempfile
from pathlib import Path
import json
from typing import Callable, Optional, List
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.courses.course_manager import CourseManager, CourseInfo
from app.pipeline.session_manager import SessionManager
from app.offline.qwen_worker import QwenWorker, SegmentTask, SegmentResult
from app.audio.source import AudioSource


class MockControlledAudioSource(AudioSource):
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        self._is_active = False

    def start(self, callback: Callable[[np.ndarray], None]):
        self._callback = callback
        self._is_active = True

    def stop(self):
        self._is_active = False
        self._callback = None

    @property
    def is_active(self) -> bool:
        return self._is_active

    def emit_audio(self, duration_sec: float, freq: float = 440.0):
        if not self._is_active or not self._callback:
            return
        samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, samples, endpoint=False, dtype=np.float32)
        chunk = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        # emit in 100ms blocks
        blocksize = 1600
        for i in range(0, len(chunk), blocksize):
            self._callback(chunk[i:i+blocksize])


def test_prompt_echo_hallucination_filter():
    print("[Test] Starting test_prompt_echo_hallucination_filter...")
    
    # 1. Exact prompt echo starting with '热词：'
    hallucinated_output = "热词：理性，产权，科斯，诺斯，威廉姆森，阿尔钦，德姆塞茨，巴泽尔，奥斯特罗姆，张五常，交易费用，交易成本，产权。"
    context = "专业术语参考: 理性、产权、科斯、诺斯、威廉姆森、阿尔钦"
    
    is_echo = QwenWorker._is_hallucinated_prompt_echo(hallucinated_output, context, online_text="")
    assert is_echo is True, "Should detect prefix '热词：' as hallucination"

    # 2. Hotword list repetition without prefix on empty speech
    list_output = "科斯，诺斯，威廉姆森，阿尔钦，德姆塞茨，交易费用"
    is_echo2 = QwenWorker._is_hallucinated_prompt_echo(list_output, context, online_text="")
    assert is_echo2 is True, "Should detect keyword list dump as hallucination"

    # 3. Genuine recognition containing one or two terms
    genuine_output = "科斯定理指出在交易费用为零时，产权明晰将带来有效配置。"
    is_echo3 = QwenWorker._is_hallucinated_prompt_echo(genuine_output, context, online_text="科斯定理指出")
    assert is_echo3 is False, "Should NOT falsely flag genuine recognition sentences"

    print("  [PASS] Prompt echo hallucination filter passed!")


def test_pause_and_resume_session():
    print("[Test] Starting test_pause_and_resume_session...")
    temp_courses_dir = Path(tempfile.mkdtemp(prefix="test_courses_"))
    try:
        cm = CourseManager(courses_dir=temp_courses_dir)
        c = cm.create_course(name="制度经济学测试", hotwords=["科斯", "交易费用", "产权界定"])

        source = MockControlledAudioSource()
        manager = SessionManager(audio_source=source, course_manager=cm)

        # 1. Start Session
        session_id = manager.start_session(course_id=c.id, source=source)
        session_dir = manager.session_dir
        assert manager.is_active is True
        assert manager.recorder.is_paused is False

        # Emit 0.5s audio before pause
        source.emit_audio(0.5, freq=300.0)
        time.sleep(0.1)

        # 2. Pause Session
        manager.pause_session()
        assert manager.recorder.is_paused is True

        # Emit audio while paused (should be discarded by recorder)
        source.emit_audio(0.5, freq=500.0)
        time.sleep(0.1)

        # 3. Resume Session
        manager.resume_session()
        assert manager.recorder.is_paused is False

        # Emit audio after resume
        source.emit_audio(0.5, freq=400.0)
        time.sleep(0.1)

        # 4. End Session
        manager.end_session()
        assert manager.is_active is False

        # Verify artifacts
        assert (session_dir / "audio.wav").exists()
        assert (session_dir / "transcript_final.md").exists()
        assert (session_dir / "events.jsonl").exists()
        assert (session_dir / "meta.json").exists()

        # Check events for session_pause and session_resume
        with open(session_dir / "events.jsonl", "r", encoding="utf-8") as f:
            event_types = [json.loads(line).get("type") for line in f if line.strip()]
            assert "session_start" in event_types
            assert "session_pause" in event_types
            assert "session_resume" in event_types
            assert "session_end" in event_types

        print("  [PASS] Pause and resume single-session recording passed!")
    finally:
        shutil.rmtree(temp_courses_dir, ignore_errors=True)


if __name__ == "__main__":
    test_prompt_echo_hallucination_filter()
    test_pause_and_resume_session()
    print("\n[ALL PASS] All prompt echo & pause/resume tests PASSED!")
