from __future__ import annotations

import json
from pathlib import Path
import shutil
from urllib.parse import quote

from contracts.map_workspace import MapAsset
from typing import List, Optional


class MapCatalogRepository:
    """地图资产目录仓储。"""

    def __init__(self, root_dir: str) -> None:
        self._root_dir = Path(root_dir).expanduser()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._active_map_path = self._root_dir / "active_map.json"

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def get_map_asset(self, map_name: str) -> Optional[MapAsset]:
        metadata_path = self._metadata_path(map_name)
        if not metadata_path.exists():
            return None
        return MapAsset.model_validate_json(metadata_path.read_text(encoding="utf-8"))

    def list_map_assets(self) -> List[MapAsset]:
        assets: List[MapAsset] = []
        for map_dir in sorted(self._root_dir.iterdir(), key=lambda item: item.name):
            if not map_dir.is_dir():
                continue
            metadata_path = map_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            assets.append(MapAsset.model_validate_json(metadata_path.read_text(encoding="utf-8")))
        return assets

    def save_map_asset(self, asset: MapAsset) -> MapAsset:
        map_dir = self._map_dir(asset.map_name)
        map_dir.mkdir(parents=True, exist_ok=True)
        (map_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        (map_dir / "versions").mkdir(parents=True, exist_ok=True)
        (map_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        self._metadata_path(asset.map_name).write_text(
            asset.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return asset

    def delete_map_asset(self, map_name: str) -> bool:
        map_dir = self._map_dir(map_name)
        if not map_dir.exists():
            return False
        shutil.rmtree(map_dir)
        if self.get_active_map_name() == str(map_name or "").strip():
            self.clear_active_map_name()
        return True

    def get_active_map_name(self) -> Optional[str]:
        if not self._active_map_path.exists():
            return None
        try:
            payload = json.loads(self._active_map_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        map_name = str(payload.get("map_name", "")).strip()
        return map_name or None

    def set_active_map_name(self, map_name: str) -> None:
        resolved_name = str(map_name or "").strip()
        if not resolved_name:
            self.clear_active_map_name()
            return
        self._active_map_path.write_text(
            json.dumps({"map_name": resolved_name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_active_map_name(self) -> None:
        if self._active_map_path.exists():
            self._active_map_path.unlink()

    def _metadata_path(self, map_name: str) -> Path:
        return self._map_dir(map_name) / "metadata.json"

    def _map_dir(self, map_name: str) -> Path:
        safe_name = quote(str(map_name or "").strip(), safe="")
        return self._root_dir / safe_name
