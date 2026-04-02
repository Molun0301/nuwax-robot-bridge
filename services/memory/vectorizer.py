from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
import hashlib
import json
import re
from typing import Iterable, List, Optional

import numpy as np


class VectorCodecMixin:
    """向量编解码与相似度公共逻辑。"""

    model_name = "unknown"
    dimension = 0

    def cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0 or right.size == 0:
            return 0.0
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    def encode_vector(self, vector: np.ndarray) -> str:
        return json.dumps([round(float(item), 8) for item in vector.tolist()], ensure_ascii=False)

    def decode_vector(self, payload: str) -> np.ndarray:
        values = json.loads(payload)
        return np.asarray(values, dtype=np.float32)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        result = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(result))
        if norm > 0.0:
            result = result / norm
        return result


class TextEmbedder(VectorCodecMixin, ABC):
    """文本向量化接口。"""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """把文本编码成归一化向量。"""


class VisionLanguageEmbedder(VectorCodecMixin, ABC):
    """图像与文本共享空间的多模态向量化接口。"""

    @abstractmethod
    def embed_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """把图像字节编码成归一化向量。"""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """把文本编码到与图像同构的空间。"""


class HashingTextEmbedder(TextEmbedder):
    """无外部依赖的文本向量化器。"""

    def __init__(self, *, model_name: str = "hashing-v1", dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("向量维度必须为正整数。")
        self.model_name = model_name
        self.dimension = dimension

    def embed_text(self, text: str) -> np.ndarray:
        features = self._build_features(text)
        vector = np.zeros(self.dimension, dtype=np.float32)
        if not features:
            return vector

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            feature_hash = int.from_bytes(digest, byteorder="big", signed=False)
            bucket = feature_hash % self.dimension
            sign = 1.0 if ((feature_hash >> 1) & 1) == 0 else -1.0
            vector[bucket] += sign

        return self._normalize_vector(vector)

    def _build_features(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        base_tokens = normalized.split()
        condensed = normalized.replace(" ", "")
        features = list(base_tokens)
        if condensed:
            features.append(condensed)
            features.extend(self._iter_char_ngrams(condensed, 2))
            features.extend(self._iter_char_ngrams(condensed, 3))
        return list(dict.fromkeys(item for item in features if item))

    def _normalize_text(self, text: str) -> str:
        raw = str(text or "").strip().lower()
        if not raw:
            return ""
        normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", raw)
        return " ".join(part for part in normalized.split() if part)

    def _iter_char_ngrams(self, text: str, size: int) -> Iterable[str]:
        if size <= 1:
            return []
        if len(text) <= size:
            return [text]
        return [text[index : index + size] for index in range(0, len(text) - size + 1)]


class SentenceTransformerTextEmbedder(TextEmbedder):
    """基于 sentence-transformers 的文本向量化器，可直接加载 BGE。"""

    def __init__(
        self,
        *,
        model_name: str,
        dimension: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "当前环境未安装 sentence-transformers，无法启用 BGE 文本向量化。"
            ) from exc

        resolved_model_name = self._resolve_model_name(model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(resolved_model_name, device=device)
        probe_vector = self._model.encode(
            ["向量维度探针"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        self.dimension = int(np.asarray(probe_vector).shape[0])
        if dimension is not None and int(dimension) != self.dimension:
            raise ValueError(
                "配置的文本向量维度与模型实际维度不一致："
                f"{dimension} != {self.dimension}"
            )

    def embed_text(self, text: str) -> np.ndarray:
        encoded = self._model.encode(
            [str(text or "")],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return self._normalize_vector(np.asarray(encoded, dtype=np.float32))

    def _resolve_model_name(self, model_name: str) -> str:
        normalized = str(model_name or "").strip()
        if normalized.startswith("sentence-transformers:"):
            return normalized.split(":", 1)[1].strip()
        return normalized


class CLIPVisionLanguageEmbedder(VisionLanguageEmbedder):
    """基于 transformers CLIP 的图文共享空间向量化器。"""

    def __init__(
        self,
        *,
        model_name: str,
        dimension: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "当前环境未安装 transformers，无法启用 CLIP 多模态向量化。"
            ) from exc
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("当前环境未安装 torch，无法启用 CLIP 多模态向量化。") from exc

        self._torch = torch
        resolved_model_name = self._resolve_model_name(model_name)
        self.model_name = model_name
        self._processor = CLIPProcessor.from_pretrained(resolved_model_name)
        self._model = CLIPModel.from_pretrained(resolved_model_name)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self.dimension = int(self._model.config.projection_dim)
        if dimension is not None and int(dimension) != self.dimension:
            raise ValueError(
                "配置的图像向量维度与模型实际维度不一致："
                f"{dimension} != {self.dimension}"
            )

    def embed_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        from PIL import Image

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            vector = self._model.get_image_features(**inputs)[0].detach().cpu().numpy()
        return self._normalize_vector(np.asarray(vector, dtype=np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self._processor(text=[str(text or "")], return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            vector = self._model.get_text_features(**inputs)[0].detach().cpu().numpy()
        return self._normalize_vector(np.asarray(vector, dtype=np.float32))

    def _resolve_model_name(self, model_name: str) -> str:
        normalized = str(model_name or "").strip()
        if normalized.startswith("transformers-clip:"):
            return normalized.split(":", 1)[1].strip()
        return normalized


def build_text_embedder(model_name: str, dimension: int) -> TextEmbedder:
    """按配置构造文本向量化器。"""

    normalized = str(model_name or "").strip()
    lowered = normalized.lower()
    if not normalized or lowered.startswith("hashing"):
        return HashingTextEmbedder(model_name=normalized or "hashing-v1", dimension=dimension)
    if lowered.startswith("sentence-transformers:") or "bge" in lowered:
        return SentenceTransformerTextEmbedder(model_name=normalized, dimension=dimension)
    raise ValueError(f"不支持的文本向量化模型：{model_name}")


def build_vision_language_embedder(
    model_name: str,
    dimension: int,
) -> Optional[VisionLanguageEmbedder]:
    """按配置构造图文共享空间向量化器。"""

    normalized = str(model_name or "").strip()
    lowered = normalized.lower()
    if not normalized or lowered in {"disabled", "none", "off"}:
        return None
    if lowered.startswith("transformers-clip:") or "clip" in lowered:
        return CLIPVisionLanguageEmbedder(model_name=normalized, dimension=dimension)
    raise ValueError(f"不支持的图文向量化模型：{model_name}")

