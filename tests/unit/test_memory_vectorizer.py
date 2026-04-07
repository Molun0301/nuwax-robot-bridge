from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import httpx

from services.memory.vectorizer import (
    build_text_embedder,
    build_vision_language_embedder,
)


def test_build_text_embedder_accepts_local_directory_and_disables_auto_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """本地文本模型目录应可直接加载，并强制仅使用本地文件。"""

    model_dir = tmp_path / "BAAI__bge-m3"
    model_dir.mkdir()
    calls = {}

    class _FakeSentenceTransformer:
        def __init__(self, model_name, device=None, local_files_only=False):
            calls["model_name"] = model_name
            calls["device"] = device
            calls["local_files_only"] = local_files_only

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            del texts, normalize_embeddings, convert_to_numpy
            return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )

    embedder = build_text_embedder(str(model_dir), dimension=4)

    assert embedder.dimension == 4
    assert calls["model_name"] == str(model_dir)
    assert calls["local_files_only"] is True


def test_build_vision_language_embedder_accepts_local_directory_and_disables_auto_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """本地 CLIP 模型目录应可直接加载，并强制仅使用本地文件。"""

    model_dir = tmp_path / "openai__clip-vit-base-patch32"
    model_dir.mkdir()
    calls = {}

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name, local_files_only=False):
            calls["processor_model_name"] = model_name
            calls["processor_local_files_only"] = local_files_only
            return cls()

    class _FakeModel:
        config = types.SimpleNamespace(projection_dim=8)

        @classmethod
        def from_pretrained(cls, model_name, local_files_only=False):
            calls["model_name"] = model_name
            calls["model_local_files_only"] = local_files_only
            return cls()

        def to(self, device):
            calls["device"] = device

        def eval(self):
            calls["eval_called"] = True

    class _FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(CLIPModel=_FakeModel, CLIPProcessor=_FakeProcessor),
    )
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)

    embedder = build_vision_language_embedder(str(model_dir), dimension=8)

    assert embedder is not None
    assert calls["processor_model_name"] == str(model_dir)
    assert calls["processor_local_files_only"] is True
    assert calls["model_name"] == str(model_dir)
    assert calls["model_local_files_only"] is True
    assert calls["device"] == "cpu"
    assert calls["eval_called"] is True


def test_build_dashscope_text_embedder_supports_fused_text_image(monkeypatch) -> None:
    """DashScope 文本向量器应能按官方融合格式发送 text+image。"""

    calls = []

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "output": {
                    "embeddings": [
                        {"embedding": [1.0] + ([0.0] * 767)},
                    ]
                }
            }

    class _FakeClient:
        def __init__(self, *, timeout=None, headers=None):
            calls.append({"timeout": timeout, "headers": headers})

        def post(self, url, json):
            calls.append({"url": url, "json": json})
            return _FakeResponse()

    monkeypatch.setattr("services.memory.vectorizer.httpx.Client", _FakeClient)

    embedder = build_text_embedder(
        "tongyi-embedding-vision-flash-2026-03-06",
        dimension=768,
        api_key="sk-test",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        timeout_sec=12.0,
    )

    text_vector = embedder.embed_text("机器人看到充电桩")
    fused_vector = embedder.embed_fused_text_image(
        text="机器人看到充电桩",
        image_bytes=b"\xff\xd8\xff\xdbdemo-jpeg",
    )

    assert text_vector.shape == (768,)
    assert fused_vector.shape == (768,)
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[1]["url"].endswith("/services/embeddings/multimodal-embedding/multimodal-embedding")
    assert calls[1]["json"]["input"]["contents"] == [{"text": "机器人看到充电桩"}]
    assert calls[2]["json"]["input"]["contents"][0]["text"] == "机器人看到充电桩"
    assert calls[2]["json"]["input"]["contents"][0]["image"].startswith("data:image/jpeg;base64,")
    assert calls[2]["json"]["parameters"]["dimension"] == 768


def test_build_dashscope_text_embedder_requires_api_key() -> None:
    """在线多模态向量默认实现缺少 API Key 时应明确报错。"""

    with pytest.raises(RuntimeError, match="API Key"):
        build_text_embedder(
            "tongyi-embedding-vision-flash-2026-03-06",
            dimension=768,
        )


def test_build_dashscope_text_embedder_clamps_unsupported_dimension(monkeypatch) -> None:
    """Tongyi 在线向量模型配置非法维度时应自动回退到合法值。"""

    calls = []

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "output": {
                    "embeddings": [
                        {"embedding": [1.0] + ([0.0] * 767)},
                    ]
                }
            }

    class _FakeClient:
        def __init__(self, *, timeout=None, headers=None):
            del timeout, headers

        def post(self, url, json):
            calls.append({"url": url, "json": json})
            return _FakeResponse()

    monkeypatch.setattr("services.memory.vectorizer.httpx.Client", _FakeClient)

    embedder = build_text_embedder(
        "tongyi-embedding-vision-flash-2026-03-06",
        dimension=1024,
        api_key="sk-test",
    )
    vector = embedder.embed_text("测试记忆查询")

    assert vector.shape == (768,)
    assert embedder.dimension == 768
    assert calls[0]["json"]["parameters"]["dimension"] == 768


def test_build_dashscope_text_embedder_surfaces_response_body_message(monkeypatch) -> None:
    """DashScope HTTP 错误应把服务端返回原因带到异常里。"""

    class _FakeResponse:
        status_code = 400
        text = '{"code":"BadRequest","message":"dimension must be one of [64, 128, 256, 512, 768], got 1024"}'

        def json(self):
            return {
                "code": "BadRequest",
                "message": "dimension must be one of [64, 128, 256, 512, 768], got 1024",
            }

    class _FakeClient:
        def __init__(self, *, timeout=None, headers=None):
            del timeout, headers

        def post(self, url, json):
            request = httpx.Request("POST", url, json=json)
            response = httpx.Response(status_code=400, request=request, json=_FakeResponse().json())
            raise httpx.HTTPStatusError("bad request", request=request, response=response)

    monkeypatch.setattr("services.memory.vectorizer.httpx.Client", _FakeClient)

    embedder = build_text_embedder(
        "tongyi-embedding-vision-flash-2026-03-06",
        dimension=768,
        api_key="sk-test",
    )

    with pytest.raises(RuntimeError, match="dimension must be one of \\[64, 128, 256, 512, 768\\], got 1024"):
        embedder.embed_text("测试记忆查询")
