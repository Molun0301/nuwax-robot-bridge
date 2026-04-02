#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""把声明式 JSON 配置同步到 NuwaClaw 的 SQLite 设置表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any, Dict, Optional


DEFAULT_DB_PATH = Path("/var/lib/docker/volumes/nuwax_home/_data/.nuwaxbot/nuwaxbot.db")
DEFAULT_KEY = "mcp_proxy_config"


class ConfigSyncError(RuntimeError):
    """配置同步错误。"""


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigSyncError("配置文件不存在: %s" % path) from exc
    except json.JSONDecodeError as exc:
        raise ConfigSyncError("配置文件不是合法 JSON: %s" % exc) from exc

    if not isinstance(payload, dict):
        raise ConfigSyncError("配置文件根节点必须是对象。")
    if "mcpServers" not in payload:
        raise ConfigSyncError("配置文件缺少 mcpServers 字段。")
    if not isinstance(payload["mcpServers"], dict):
        raise ConfigSyncError("mcpServers 必须是对象。")
    return payload


def _normalize_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _merge_payload(current_payload: Optional[Dict[str, Any]], incoming_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not current_payload:
        return dict(incoming_payload)

    merged = dict(current_payload)
    for key, value in incoming_payload.items():
        if key == "mcpServers" and isinstance(value, dict):
            current_servers = merged.get("mcpServers", {})
            if not isinstance(current_servers, dict):
                current_servers = {}
            merged_servers = dict(current_servers)
            merged_servers.update(value)
            merged["mcpServers"] = merged_servers
        else:
            merged[key] = value
    return merged


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise ConfigSyncError("数据库文件不存在: %s" % db_path)
    return sqlite3.connect(str(db_path))


def read_setting(db_path: Path, key: str = DEFAULT_KEY) -> Optional[str]:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


def read_json_setting(db_path: Path, key: str = DEFAULT_KEY) -> Dict[str, Any]:
    raw_value = read_setting(db_path, key)
    if raw_value is None:
        raise ConfigSyncError("数据库中不存在 key=%s 的设置项。" % key)
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ConfigSyncError("数据库中的 %s 不是合法 JSON: %s" % (key, exc)) from exc
    if not isinstance(payload, dict):
        raise ConfigSyncError("数据库中的 %s 不是对象结构。" % key)
    return payload


def export_json_config(db_path: Path, output_path: Path, key: str = DEFAULT_KEY) -> Path:
    payload = read_json_setting(db_path, key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_normalize_json(payload) + "\n", encoding="utf-8")
    return output_path


def _build_backup_path(db_path: Path) -> Path:
    return db_path.with_name("%s.%s.bak" % (db_path.name, db_path.stat().st_mtime_ns))


def apply_json_config(
    db_path: Path,
    config_path: Path,
    *,
    key: str = DEFAULT_KEY,
    backup_path: Optional[Path] = None,
    create_backup: bool = True,
    merge: bool = True,
) -> bool:
    incoming_payload = _load_json_file(config_path)
    current_value = read_setting(db_path, key)
    current_payload = None
    if current_value is not None:
        current_payload = read_json_setting(db_path, key)
    target_payload = _merge_payload(current_payload, incoming_payload) if merge else incoming_payload
    normalized_value = _normalize_json(target_payload)
    if current_value == normalized_value:
        return False

    if create_backup:
        backup_target = backup_path or _build_backup_path(db_path)
        backup_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(db_path), str(backup_target))

    conn = _connect(db_path)
    try:
        with conn:
            existing = conn.execute("SELECT 1 FROM settings WHERE key = ?", (key,)).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES (?, ?)",
                    (key, normalized_value),
                )
            else:
                conn.execute(
                    "UPDATE settings SET value = ? WHERE key = ?",
                    (normalized_value, key),
                )
    finally:
        conn.close()
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="同步 NuwaClaw 的 mcp_proxy_config 配置。")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="NuwaClaw SQLite 数据库路径。",
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help="settings 表中的配置键名，默认是 mcp_proxy_config。",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show", help="读取数据库中的当前配置并打印到标准输出。")
    show_parser.add_argument("--raw", action="store_true", help="直接输出原始字符串，不做 JSON 格式化。")

    export_parser = subparsers.add_parser("export", help="把数据库中的当前配置导出成 JSON 文件。")
    export_parser.add_argument("--output", required=True, help="导出目标文件路径。")

    apply_parser = subparsers.add_parser("apply", help="把 JSON 文件写回数据库。")
    apply_parser.add_argument("--config-file", required=True, help="声明式 JSON 配置文件路径。")
    apply_parser.add_argument("--backup-path", help="备份数据库文件输出路径。")
    apply_parser.add_argument("--no-backup", action="store_true", help="不在写入前备份数据库。")
    apply_parser.add_argument(
        "--replace",
        action="store_true",
        help="默认会合并已有 mcpServers；显式传入该参数时才会全量替换数据库中的现有配置。",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    db_path = Path(args.db_path).expanduser().resolve()

    try:
        if args.command == "show":
            value = read_setting(db_path, args.key)
            if value is None:
                raise ConfigSyncError("数据库中不存在 key=%s 的设置项。" % args.key)
            if args.raw:
                print(value)
            else:
                print(_normalize_json(read_json_setting(db_path, args.key)))
            return 0

        if args.command == "export":
            output_path = Path(args.output).expanduser().resolve()
            export_json_config(db_path, output_path, args.key)
            print("已导出到 %s" % output_path)
            return 0

        if args.command == "apply":
            config_path = Path(args.config_file).expanduser().resolve()
            backup_path = None
            if args.backup_path:
                backup_path = Path(args.backup_path).expanduser().resolve()
            changed = apply_json_config(
                db_path,
                config_path,
                key=args.key,
                backup_path=backup_path,
                create_backup=not args.no_backup,
                merge=not args.replace,
            )
            if changed:
                print("已写入 %s:%s" % (db_path, args.key))
            else:
                print("配置未变化，跳过写入。")
            return 0

        parser.error("未知命令。")
        return 2
    except ConfigSyncError as exc:
        print("错误: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
