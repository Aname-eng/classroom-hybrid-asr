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

    @staticmethod
    def _generate_course_id(name: str) -> str:
        """从课程名称生成安全的文件系统目录名 ID"""
        import re
        import hashlib
        # 保留英文字母和数字
        ascii_slug = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()
        if ascii_slug:
            return ascii_slug
        # 若为纯中文或其他字符，使用短哈希或拼音 slug
        short_hash = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        return f"course_{short_hash}"

    def create_course(
        self,
        name: str,
        hotwords: Optional[List[str]] = None,
        description: str = "",
        course_id: Optional[str] = None,
        language: str = "zh"
    ) -> CourseInfo:
        """
        新建并持久化一门新课程及其专属专业词库
        """
        if not name or not name.strip():
            raise ValueError("Course name cannot be empty.")
        
        name = name.strip()
        cid = course_id.strip() if course_id and course_id.strip() else self._generate_course_id(name)
        
        # 防止 ID 冲突，若冲突则追加序号
        target_dir = self.courses_dir / cid
        base_cid = cid
        counter = 1
        while target_dir.exists() and cid not in self.courses:
            cid = f"{base_cid}_{counter}"
            target_dir = self.courses_dir / cid
            counter += 1

        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 写入 course.yaml
        yaml_path = target_dir / "course.yaml"
        yaml_content = {
            "name": name,
            "language": language,
            "online_model": "paraformer_streaming",
            "offline_model": "qwen3_asr_1.7b_q4k",
            "description": description.strip() if description else f"{name} 课堂转写笔记"
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        # 2. 写入 hotwords.txt
        hotwords_path = target_dir / "hotwords.txt"
        clean_hotwords = [w.strip() for w in (hotwords or []) if w.strip()]
        with open(hotwords_path, "w", encoding="utf-8") as f:
            f.write(f"# {name} 专业术语与热词库 (一行一个)\n")
            for hw in clean_hotwords:
                f.write(f"{hw}\n")

        info = CourseInfo(
            id=cid,
            name=name,
            language=language,
            online_model="paraformer_streaming",
            offline_model="qwen3_asr_1.7b_q4k",
            description=description,
            hotwords=clean_hotwords
        )
        self.courses[cid] = info
        return info

    def update_course_hotwords(self, course_id: str, hotwords: List[str]) -> bool:
        """更新已有课程的热词库文件"""
        course = self.get_course(course_id)
        if not course:
            return False
        
        target_dir = self.courses_dir / course_id
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            
        hotwords_path = target_dir / "hotwords.txt"
        clean_hotwords = [w.strip() for w in hotwords if w.strip()]
        with open(hotwords_path, "w", encoding="utf-8") as f:
            f.write(f"# {course.name} 专业术语与热词库 (一行一个)\n")
            for hw in clean_hotwords:
                f.write(f"{hw}\n")
                
        course.hotwords = clean_hotwords
        return True
