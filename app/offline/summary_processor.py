# coding: utf-8
"""录音结束后的本地小模型整理与热词生成。

该模块支持直接加载已下载的 Qwen Transformers 模型，也兼容 Ollama 和
OpenAI-compatible 本地接口。没有本地 LLM 服务时不会让课堂失败，而是退化为
保守的口头禅清理，并保留原始稿件。
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover - 允许极简离线安装继续运行
    requests = None


@dataclass
class CleanupResult:
    cleaned_segments: Dict[int, str]
    model: str
    used_model: bool
    error: str = ""


class LocalLLMAdapter:
    """极薄的本地聊天接口适配器，不绑定某一个推理框架。"""

    def __init__(
        self,
        provider: str,
        endpoint: str,
        model: str,
        timeout_sec: float = 12.0,
        model_path: str = "",
    ):
        self.provider = (provider or "none").strip().lower()
        self.endpoint = (endpoint or "").strip().rstrip("/")
        self.model = (model or "").strip()
        self.model_path = (model_path or "").strip()
        self.timeout_sec = max(1.0, float(timeout_sec or 12.0))
        self.last_error = ""
        self._tokenizer = None
        self._local_model = None
        self._torch = None
        self._local_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        if self.provider in {"local_transformers", "transformers"}:
            return bool(self.model_path and Path(self.model_path).expanduser().is_dir())
        return bool(
            requests is not None
            and self.provider not in {"", "none", "disabled", "offline"}
            and self.endpoint
            and self.model
        )

    def _chat_local(self, system: str, user: str) -> Optional[str]:
        """使用已下载的 Qwen 小模型，不要求 Ollama 常驻。"""
        try:
            # 模型首次加载和 generate 都可能很慢；锁只保护模型对象初始化/推理，
            # 外层 SummaryProcessor 会在 daemon 线程中等待有界时间。
            with self._local_lock:
                return self._chat_local_locked(system, user)
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def _chat_local_locked(self, system: str, user: str) -> Optional[str]:
        if self._local_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            path = str(Path(self.model_path).expanduser())
            self._tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, local_files_only=True
            )
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._local_model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._local_model.to(device)
            self._local_model.eval()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self._tokenizer([prompt], return_tensors="pt")
        device = next(self._local_model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._local_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def chat(self, system: str, user: str) -> Optional[str]:
        if not self.enabled:
            return None
        if self.provider in {"local_transformers", "transformers"}:
            return self._chat_local(system, user)

        try:
            if self.provider in {"ollama", "ollama_api"}:
                url = self.endpoint
                if not url.endswith("/api/chat"):
                    url += "/api/chat"
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                }
                response = requests.post(
                    url,
                    json=payload,
                    # 本地服务未启动时快速退化，不让录音结束被连接重试拖住。
                    timeout=(min(0.75, self.timeout_sec), self.timeout_sec),
                )
                response.raise_for_status()
                data = response.json()
                content = (data.get("message") or {}).get("content", "")
            else:
                # openai_compatible: endpoint 可以是 base URL，也可以已经包含路径。
                url = self.endpoint
                if not url.endswith("/chat/completions"):
                    url += "/v1/chat/completions"
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                    "stream": False,
                }
                response = requests.post(
                    url,
                    json=payload,
                    timeout=(min(0.75, self.timeout_sec), self.timeout_sec),
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                content = ((choices[0].get("message") or {}).get("content", "")) if choices else ""

            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            return str(content).strip() if content is not None else ""
        except Exception as exc:
            self.last_error = str(exc)
            return None


class SummaryProcessor:
    """用课程主题约束小模型，只做删除/整理，不覆盖原始识别证据。"""

    _FILLER_WORDS = ("嗯", "呃", "额", "啊", "哦", "唉", "然后")
    _BOUNDARY = r"，。！？；：、,!?;:\s"

    def __init__(
        self,
        provider: str = "local_transformers",
        endpoint: str = "http://127.0.0.1:11434",
        model: str = "Qwen3-0.6B",
        model_path: str = "",
        timeout_sec: float = 12.0,
        batch_chars: int = 6000,
        enabled: bool = True,
        remove_off_topic: bool = True,
    ):
        self.enabled = bool(enabled)
        self.remove_off_topic = bool(remove_off_topic)
        self.batch_chars = max(1000, int(batch_chars or 6000))
        self.adapter = LocalLLMAdapter(provider, endpoint, model, timeout_sec, model_path=model_path)
        self.model_name = model or "local-llm"

    @classmethod
    def remove_fillers(cls, text: str) -> str:
        """保守删除独立口头禅，不删除词语内部的同音字。"""
        value = (text or "").strip()
        if not value:
            return ""

        boundary = cls._BOUNDARY
        for word in cls._FILLER_WORDS:
            # 两端有标点/空白才视为口头禅，例如不动“啊Q”或术语内部字符。
            pattern = rf"(?:(?<=^)|(?<=[{boundary}])){re.escape(word)}(?=$|[{boundary}])"
            value = re.sub(pattern, "", value)
        value = re.sub(r"([，、；：])\s*\1+", r"\1", value)
        value = re.sub(r"\s{2,}", " ", value)
        value = re.sub(r"^[，、；：\s]+|[，、；：\s]+$", "", value)
        return value.strip()

    @staticmethod
    def _extract_json(text: str) -> Any:
        if not text:
            return None
        value = text.strip()
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE | re.DOTALL).strip()
        for left, right in (("[", "]"), ("{", "}")):
            start = value.find(left)
            end = value.rfind(right)
            if start >= 0 and end > start:
                try:
                    return json.loads(value[start : end + 1])
                except json.JSONDecodeError:
                    continue
        return None

    def _chat_with_timeout(self, system: str, user: str) -> Optional[str]:
        """让本地 Transformers 推理有界，不阻塞录音启动或会话收尾。"""
        if self.adapter.provider not in {"local_transformers", "transformers"}:
            return self.adapter.chat(system, user)

        result: List[Optional[str]] = [None]

        def run() -> None:
            result[0] = self.adapter.chat(system, user)

        thread = threading.Thread(target=run, daemon=True, name="LocalSummaryInference")
        thread.start()
        thread.join(self.adapter.timeout_sec)
        if thread.is_alive():
            self.adapter.last_error = f"local model timed out after {self.adapter.timeout_sec:.1f}s"
            return None
        return result[0]

    def _batches(self, items: List[Dict[str, Any]]) -> Iterable[List[Dict[str, Any]]]:
        batch: List[Dict[str, Any]] = []
        size = 0
        for item in items:
            item_size = len(json.dumps(item, ensure_ascii=False))
            if batch and size + item_size > self.batch_chars:
                yield batch
                batch = []
                size = 0
            batch.append(item)
            size += item_size
        if batch:
            yield batch

    def clean_segments(
        self,
        segments: Iterable[Tuple[int, str]],
        course_name: str,
        hotwords: Optional[List[str]] = None,
    ) -> CleanupResult:
        base: Dict[int, str] = {
            int(segment_id): self.remove_fillers(text)
            for segment_id, text in segments
        }
        if not base:
            return CleanupResult({}, self.model_name, False)

        if not self.enabled or not self.remove_off_topic or not self.adapter.enabled:
            return CleanupResult(base, "deterministic-filler-cleaner", False)

        items = [
            {"segment_id": segment_id, "text": text}
            for segment_id, text in sorted(base.items())
            if text
        ]
        if not items:
            return CleanupResult(base, "deterministic-filler-cleaner", False)

        terms = "、".join((hotwords or [])[:30]) or "无"
        system = (
            "你是课堂录音稿整理器。只删除与课程无关的闲聊、寒暄和口头禅，"
            "保留讲课事实、例子、数字、公式、专业名词和原有顺序。不要补写、"
            "不要臆测、不要把内容改写成不存在的事实。必须只输出 JSON 数组，"
            "每项格式为 {\"segment_id\":整数,\"keep\":true/false,\"text\":字符串}。"
        )
        used_model = False
        error = ""
        for batch in self._batches(items):
            user = (
                f"课程主题：{course_name}\n"
                f"专业热词（仅用于判断语境，不要原样复述）：{terms}\n"
                "整理规则：独立的“嗯、呃、额、啊、然后”等口头禅可以删除；"
                "与主题无关的闲聊整句删除；有课程信息的句子不要删除。\n"
                f"待整理稿件：{json.dumps(batch, ensure_ascii=False)}"
            )
            response = self._chat_with_timeout(system, user)
            parsed = self._extract_json(response or "")
            if parsed is None:
                error = self.adapter.last_error or "local summary returned invalid JSON"
                continue
            rows = parsed.get("segments") if isinstance(parsed, dict) else parsed
            if not isinstance(rows, list):
                error = "local summary JSON is not an array"
                continue

            known_ids = {int(item["segment_id"]) for item in batch}
            for row in rows:
                if not isinstance(row, dict) or "segment_id" not in row:
                    continue
                try:
                    segment_id = int(row["segment_id"])
                except (TypeError, ValueError):
                    continue
                if segment_id not in known_ids:
                    continue
                if row.get("keep") is False:
                    base[segment_id] = ""
                elif isinstance(row.get("text"), str):
                    base[segment_id] = self.remove_fillers(row["text"])
            used_model = True

        model_name = self.model_name if used_model else "deterministic-filler-cleaner"
        return CleanupResult(base, model_name, used_model, error)

    @staticmethod
    def _fallback_hotwords(course_name: str, description: str, existing: List[str], max_count: int) -> List[str]:
        source = f"{course_name} {description}"
        candidates: List[str] = []
        candidates.extend(existing or [])
        # 保留完整课程名，再提取较长的中文短语和英文术语。
        candidates.append(course_name)
        candidates.extend(re.findall(r"[\u4e00-\u9fff]{2,12}", source))
        candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9_+.#-]{1,31}", source))

        result: List[str] = []
        for word in candidates:
            word = str(word).strip(" ，,。；;：:\t\n")
            if len(word) < 2 or word in result:
                continue
            result.append(word)
            if len(result) >= max_count:
                break
        return result

    def generate_hotwords(
        self,
        course_name: str,
        description: str = "",
        existing: Optional[List[str]] = None,
        max_count: int = 30,
    ) -> Tuple[List[str], bool]:
        """生成本次会话使用的热词；失败时返回可重复的规则兜底词表。"""
        seed = [str(x).strip() for x in (existing or []) if str(x).strip()]
        fallback = self._fallback_hotwords(course_name, description, seed, max_count)
        if not self.enabled or not self.adapter.enabled:
            return fallback, False

        system = (
            "你是中文课程术语抽取器。根据课程名称和简介给出最可能出现在课堂中的"
            "专业术语、人物、方法、英文缩写或公式名。只输出 JSON 字符串数组，"
            "不要解释，不要输出课程闲聊词。"
        )
        user = (
            f"课程名称：{course_name}\n"
            f"课程简介：{description or '无'}\n"
            f"已有热词：{json.dumps(seed, ensure_ascii=False)}\n"
            f"最多输出 {max_count} 个词。"
        )
        response = self._chat_with_timeout(system, user)
        parsed = self._extract_json(response or "")
        values: List[str] = []
        if isinstance(parsed, list):
            values = [str(item).strip() for item in parsed if isinstance(item, (str, int, float))]
        elif isinstance(parsed, dict) and isinstance(parsed.get("hotwords"), list):
            values = [str(item).strip() for item in parsed["hotwords"]]

        if not values:
            return fallback, False

        merged: List[str] = []
        for word in seed + values + [course_name]:
            word = word.strip(" ，,。；;：:\t\n")
            if len(word) >= 2 and word not in merged:
                merged.append(word)
            if len(merged) >= max_count:
                break
        return merged or fallback, True
