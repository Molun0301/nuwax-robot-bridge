from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from scripts.nuwaclaw_mcp_config_sync import (
    apply_json_config,
    export_json_config,
    read_json_setting,
    read_setting,
)


def _build_db(path: Path, key: str, value: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        with conn:
            conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO settings(key, value) VALUES (?, ?)", (key, value))
    finally:
        conn.close()


def test_read_and_export_json_setting(tmp_path: Path) -> None:
    db_path = tmp_path / "nuwaxbot.db"
    output_path = tmp_path / "current.json"
    payload = {"mcpServers": {"demo": {"url": "http://127.0.0.1:8766/mcp"}}}
    _build_db(db_path, "mcp_proxy_config", json.dumps(payload))

    assert read_setting(db_path) is not None
    assert read_json_setting(db_path) == payload

    export_json_config(db_path, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_apply_json_config_updates_setting_and_creates_backup(tmp_path: Path) -> None:
    db_path = tmp_path / "nuwaxbot.db"
    config_path = tmp_path / "desired.json"
    backup_path = tmp_path / "nuwaxbot.backup.db"

    current = {"mcpServers": {"old": {"command": "npx"}}}
    desired = {"mcpServers": {"nuwax_robot_bridge": {"url": "http://host.docker.internal:8766/mcp"}}}

    _build_db(db_path, "mcp_proxy_config", json.dumps(current))
    config_path.write_text(json.dumps(desired), encoding="utf-8")

    changed = apply_json_config(
        db_path,
        config_path,
        backup_path=backup_path,
    )

    assert changed is True
    assert backup_path.exists()
    assert read_json_setting(db_path) == {
        "mcpServers": {
            "old": {"command": "npx"},
            "nuwax_robot_bridge": {"url": "http://host.docker.internal:8766/mcp"},
        }
    }


def test_apply_json_config_skips_when_unchanged(tmp_path: Path) -> None:
    db_path = tmp_path / "nuwaxbot.db"
    config_path = tmp_path / "desired.json"
    payload = {"mcpServers": {"nuwax_robot_bridge": {"url": "http://host.docker.internal:8766/mcp"}}}

    normalized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    _build_db(db_path, "mcp_proxy_config", normalized)
    config_path.write_text(normalized, encoding="utf-8")

    changed = apply_json_config(db_path, config_path, create_backup=False)

    assert changed is False


def test_apply_json_config_supports_replace_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "nuwaxbot.db"
    config_path = tmp_path / "desired.json"

    current = {"mcpServers": {"old": {"command": "npx"}}}
    desired = {"mcpServers": {"nuwax_robot_bridge": {"url": "http://host.docker.internal:8766/mcp"}}}

    _build_db(db_path, "mcp_proxy_config", json.dumps(current))
    config_path.write_text(json.dumps(desired), encoding="utf-8")

    changed = apply_json_config(db_path, config_path, create_backup=False, merge=False)

    assert changed is True
    assert read_json_setting(db_path) == desired
