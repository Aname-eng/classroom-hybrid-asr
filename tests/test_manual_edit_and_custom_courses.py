# coding: utf-8
import sys
import shutil
import tempfile
from pathlib import Path
import json
from typing import Callable, Optional, List
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.courses.course_manager import CourseManager, CourseInfo
from app.pipeline.session_manager import SessionManager, SegmentRecord
from app.offline.qwen_worker import SegmentResult
from app.audio.source import AudioSource


class MockAudioSource(AudioSource):
    def __init__(self):
        self._is_active = False

    def start(self, callback: Callable[[np.ndarray], None]):
        self._is_active = True

    def stop(self):
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active


def test_course_creation_and_hotwords():
    print("[Test] Starting test_course_creation_and_hotwords...")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_courses_"))
    try:
        cm = CourseManager(courses_dir=temp_dir)
        assert len(cm.list_courses()) == 0

        # 1. Create a new custom course
        c = cm.create_course(
            name="深度学习与大语言模型",
            hotwords=["Transformer", "Self-Attention", "FlashAttention", "RoPE"],
            description="前沿大模型与分布式训练课程"
        )
        assert c.name == "深度学习与大语言模型"
        assert len(c.hotwords) == 4
        assert "Transformer" in c.hotwords
        
        # Verify persistence on disk
        course_folder = temp_dir / c.id
        assert (course_folder / "course.yaml").exists()
        assert (course_folder / "hotwords.txt").exists()

        # 2. Reload all courses in a fresh manager instance
        cm2 = CourseManager(courses_dir=temp_dir)
        loaded = cm2.get_course(c.id)
        assert loaded is not None
        assert loaded.name == "深度学习与大语言模型"
        assert loaded.hotwords == ["Transformer", "Self-Attention", "FlashAttention", "RoPE"]
        assert loaded.description == "前沿大模型与分布式训练课程"

        # 3. Update hotwords
        updated_hw = ["Transformer", "Self-Attention", "DeepSeek", "LoRA"]
        success = cm2.update_course_hotwords(c.id, updated_hw)
        assert success is True
        assert cm2.get_course(c.id).hotwords == updated_hw

        # Verify disk update
        with open(course_folder / "hotwords.txt", "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            assert lines == updated_hw

        print("  [PASS] Course creation and hotword management passed!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_manual_edit_in_session():
    print("[Test] Starting test_manual_edit_in_session...")
    temp_courses_dir = Path(tempfile.mkdtemp(prefix="test_courses_"))
    try:
        cm = CourseManager(courses_dir=temp_courses_dir)
        cm.create_course(name="微观计量测试", hotwords=["Hausman"])

        source = MockAudioSource()
        manager = SessionManager(audio_source=source)
        manager.course_manager = cm

        # Start session
        session_id = manager.start_session(course_id=cm.list_courses()[0].id, source=source)
        session_dir = manager.session_dir

        # Simulate incoming partial and initial final results for segment 1
        manager._on_streaming_partial(1, "我们使用豪斯曼检验", 1.0)
        
        # Initial Qwen final result
        res1 = SegmentResult(
            segment_id=1,
            start_sec=0.0,
            end_sec=2.0,
            final_text="我们使用豪斯曼检验固定效应模型和随机效应模型。",
            final_model="Qwen3-ASR-1.7B-q4_k",
            online_text="我们使用豪斯曼检验",
            latency_sec=0.45,
            success=True
        )
        manager._on_qwen_result(res1)
        
        assert manager.segments[1].final_text == "我们使用豪斯曼检验固定效应模型和随机效应模型。"
        assert manager.segments[1].is_manual_edited is False

        # User performs in-place manual edit ("随听随改")
        user_edited_text = "我们使用 Hausman 检验比较固定效应模型 (Fixed Effects) 与随机效应模型 (Random Effects)。"
        manager.update_segment_text(1, user_edited_text)

        # Check in-memory status
        assert manager.segments[1].final_text == user_edited_text
        assert manager.segments[1].is_manual_edited is True

        # Check incremental disk sync
        md_file = session_dir / "transcript_final.md"
        raw_file = session_dir / "transcript_raw.jsonl"
        events_file = session_dir / "events.jsonl"

        assert md_file.exists()
        assert raw_file.exists()
        assert events_file.exists()

        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
            assert user_edited_text in md_content
            assert "[人工已修改]" in md_content

        with open(raw_file, "r", encoding="utf-8") as f:
            raw_lines = [json.loads(line) for line in f if line.strip()]
            assert len(raw_lines) == 1
            assert raw_lines[0]["final_text"] == user_edited_text
            assert raw_lines[0]["is_manual_edited"] is True

        with open(events_file, "r", encoding="utf-8") as f:
            event_types = [json.loads(line).get("type") for line in f if line.strip()]
            assert "manual_edit" in event_types

        # Verify that late Qwen results do NOT overwrite manual edits
        late_res = SegmentResult(
            segment_id=1,
            start_sec=0.0,
            end_sec=2.0,
            final_text="迟到的模型输出，不应覆盖人工改错",
            final_model="Qwen3-ASR-1.7B-q4_k",
            online_text="我们使用豪斯曼检验",
            latency_sec=1.2,
            success=True
        )
        manager._on_qwen_result(late_res)
        assert manager.segments[1].final_text == user_edited_text
        assert manager.segments[1].is_manual_edited is True

        manager.end_session()
        print("  [PASS] In-place manual editing in session passed!")
    finally:
        shutil.rmtree(temp_courses_dir, ignore_errors=True)


if __name__ == "__main__":
    test_course_creation_and_hotwords()
    test_manual_edit_in_session()
    print("\n[ALL PASS] All manual edit & custom courses tests PASSED!")
