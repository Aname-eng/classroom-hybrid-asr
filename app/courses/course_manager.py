# coding: utf-8
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from app.config import COURSES_DIR

@dataclass
class CourseInfo:
    id: str
    name: str
    language: str = "zh"
    online_model: str = "paraformer_streaming"
    offline_model: str = "qwen3_asr_1.7b_q4k"
    description: str = ""
    hotwords: List[str] = field(default_factory=list)

class CourseManager:
    def __init__(self, courses_dir: Path = COURSES_DIR):
        self.courses_dir = courses_dir
        self.courses: Dict[str, CourseInfo] = {}
        self.load_all_courses()

    def load_all_courses(self) -> Dict[str, CourseInfo]:
        self.courses.clear()
        if not self.courses_dir.exists():
            self.courses_dir.mkdir(parents=True, exist_ok=True)
            return self.courses

        for course_folder in self.courses_dir.iterdir():
            if course_folder.is_dir():
                yaml_file = course_folder / "course.yaml"
                hotwords_file = course_folder / "hotwords.txt"
                
                info_data = {}
                if yaml_file.exists():
                    try:
                        with open(yaml_file, "r", encoding="utf-8") as f:
                            info_data = yaml.safe_load(f) or {}
                    except Exception as e:
                        print(f"Error loading {yaml_file}: {e}")
                
                hotwords = []
                if hotwords_file.exists():
                    try:
                        with open(hotwords_file, "r", encoding="utf-8") as f:
                            hotwords = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                    except Exception as e:
                        print(f"Error loading {hotwords_file}: {e}")
                
                course_id = course_folder.name
                course_name = info_data.get("name", course_id)
                
                self.courses[course_id] = CourseInfo(
                    id=course_id,
                    name=course_name,
                    language=info_data.get("language", "zh"),
                    online_model=info_data.get("online_model", "paraformer_streaming"),
                    offline_model=info_data.get("offline_model", "qwen3_asr_1.7b_q4k"),
                    description=info_data.get("description", ""),
                    hotwords=hotwords
                )
        return self.courses

    def get_course(self, course_id: str) -> Optional[CourseInfo]:
        return self.courses.get(course_id)

    def list_courses(self) -> List[CourseInfo]:
        return list(self.courses.values())
