from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

from contracts.artifacts import (
    ArtifactCleanupResult,
    ArtifactKind,
    ArtifactRef,
    ArtifactRetentionPolicy,
    ArtifactStorageSummary,
)
from contracts.base import utc_now
from contracts.naming import build_artifact_id
from gateways.errors import ArtifactNotFoundError


class LocalArtifactStore:
    """宿主机本地制品存储。"""

    def __init__(self, base_dir: str, public_base_url: str) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._public_base_url = public_base_url.rstrip("/")
        self._refs: Dict[str, ArtifactRef] = {}
        self._paths: Dict[str, Path] = {}

    @property
    def base_dir(self) -> Path:
        """返回制品根目录。"""

        return self._base_dir

    def save_bytes(
        self,
        *,
        kind: ArtifactKind,
        mime_type: str,
        data: bytes,
        metadata: Optional[Dict[str, object]] = None,
        extension: Optional[str] = None,
    ) -> ArtifactRef:
        """保存二进制制品并返回引用。"""

        artifact_id = build_artifact_id(kind.value)
        suffix = extension or self._guess_extension(mime_type)
        directory = self._base_dir / kind.value
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / f"{artifact_id}{suffix}"
        file_path.write_bytes(data)

        ref = ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            mime_type=mime_type,
            size_bytes=len(data),
            uri=f"{self._public_base_url}/artifacts/{artifact_id}",
            sha256=hashlib.sha256(data).hexdigest(),
            metadata=dict(metadata or {}),
        )
        meta_path = directory / f"{artifact_id}.meta.json"
        meta_path.write_text(ref.model_dump_json(indent=2), encoding="utf-8")
        self._refs[artifact_id] = ref
        self._paths[artifact_id] = file_path
        return ref

    def get_ref(self, artifact_id: str) -> ArtifactRef:
        """读取制品元数据。"""

        if artifact_id in self._refs:
            return self._refs[artifact_id]

        meta_path = self._find_meta_path(artifact_id)
        if meta_path is None:
            raise ArtifactNotFoundError(artifact_id)

        ref = ArtifactRef.model_validate_json(meta_path.read_text(encoding="utf-8"))
        self._refs[artifact_id] = ref
        return ref

    def get_file_path(self, artifact_id: str) -> Path:
        """解析制品文件路径。"""

        if artifact_id in self._paths:
            return self._paths[artifact_id]

        artifact_path = self._find_file_path(artifact_id)
        if artifact_path is None:
            raise ArtifactNotFoundError(artifact_id)
        self._paths[artifact_id] = artifact_path
        return artifact_path

    def list_refs(self, *, kind: Optional[ArtifactKind] = None) -> Tuple[ArtifactRef, ...]:
        """列出当前全部制品引用。"""

        refs: List[ArtifactRef] = []
        for meta_path in sorted(self._base_dir.glob("**/*.meta.json")):
            ref = ArtifactRef.model_validate_json(meta_path.read_text(encoding="utf-8"))
            if kind is not None and ref.kind != kind:
                continue
            self._refs[ref.artifact_id] = ref
            refs.append(ref)
        refs.sort(key=lambda item: item.timestamp)
        return tuple(refs)

    def build_summary(self, refs: Optional[Iterable[ArtifactRef]] = None) -> ArtifactStorageSummary:
        """构造当前制品存储摘要。"""

        items = list(refs if refs is not None else self.list_refs())
        by_kind: Dict[str, int] = {}
        total_size_bytes = 0
        for ref in items:
            by_kind[ref.kind.value] = by_kind.get(ref.kind.value, 0) + 1
            total_size_bytes += int(ref.size_bytes or 0)
        return ArtifactStorageSummary(
            artifact_count=len(items),
            total_size_bytes=total_size_bytes,
            by_kind=by_kind,
            oldest_artifact_id=items[0].artifact_id if items else None,
            newest_artifact_id=items[-1].artifact_id if items else None,
        )

    def delete_artifact(self, artifact_id: str) -> ArtifactRef:
        """删除单个制品。"""

        ref = self.get_ref(artifact_id)
        file_path = self.get_file_path(artifact_id)
        meta_path = self._find_meta_path(artifact_id)
        if file_path.exists():
            file_path.unlink()
        if meta_path is not None and meta_path.exists():
            meta_path.unlink()
        self._refs.pop(artifact_id, None)
        self._paths.pop(artifact_id, None)
        return ref

    def cleanup(self, policy: ArtifactRetentionPolicy) -> ArtifactCleanupResult:
        """按策略清理制品。"""

        refs = list(self.list_refs())
        removed_refs: List[ArtifactRef] = []
        retained_refs = list(refs)

        if policy.retention_days is not None:
            expire_before = utc_now() - timedelta(days=policy.retention_days)
            expired = [ref for ref in retained_refs if ref.timestamp < expire_before]
            for ref in expired[: policy.cleanup_batch_size]:
                removed_refs.append(self.delete_artifact(ref.artifact_id))
            retained_ids = {ref.artifact_id for ref in retained_refs} - {ref.artifact_id for ref in removed_refs}
            retained_refs = [ref for ref in retained_refs if ref.artifact_id in retained_ids]

        total_size_bytes = sum(int(ref.size_bytes or 0) for ref in retained_refs)
        while retained_refs and len(removed_refs) < policy.cleanup_batch_size:
            should_trim_count = policy.max_count is not None and len(retained_refs) > policy.max_count
            should_trim_size = policy.max_total_bytes is not None and total_size_bytes > policy.max_total_bytes
            if not should_trim_count and not should_trim_size:
                break
            victim = retained_refs.pop(0)
            removed_refs.append(self.delete_artifact(victim.artifact_id))
            total_size_bytes -= int(victim.size_bytes or 0)

        summary = self.build_summary()
        return ArtifactCleanupResult(
            removed_count=len(removed_refs),
            freed_bytes=sum(int(ref.size_bytes or 0) for ref in removed_refs),
            removed_artifact_ids=[ref.artifact_id for ref in removed_refs],
            summary=summary,
        )

    def _find_meta_path(self, artifact_id: str) -> Optional[Path]:
        matches = list(self._base_dir.glob(f"**/{artifact_id}.meta.json"))
        return matches[0] if matches else None

    def _find_file_path(self, artifact_id: str) -> Optional[Path]:
        for candidate in self._base_dir.glob(f"**/{artifact_id}*"):
            if candidate.name.endswith(".meta.json"):
                continue
            if candidate.is_file():
                return candidate
        return None

    def _guess_extension(self, mime_type: str) -> str:
        normalized = mime_type.lower().strip()
        if normalized == "image/jpeg":
            return ".jpg"
        if normalized == "image/png":
            return ".png"
        if normalized == "application/json":
            return ".json"
        if normalized == "audio/wav":
            return ".wav"
        return ".bin"
