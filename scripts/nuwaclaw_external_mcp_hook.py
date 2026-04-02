#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""为 NuwaClaw 安装外部 MCP 配置注入包装层。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Optional


DEFAULT_NUWAXBOT_HOME = Path("/var/lib/docker/volumes/nuwax_home/_data/.nuwaxbot")
DEFAULT_EXTERNAL_CONFIG = Path("/var/lib/docker/volumes/nuwax_home/_data/.nuwaxbot/mcp/external_servers.json")
DEFAULT_PROXY_SCRIPT = Path(
    "/var/lib/docker/volumes/nuwax_home/_data/.nuwaxbot/node_modules/nuwax-mcp-stdio-proxy/dist/index.js"
)
DEFAULT_RUNTIME_EXTERNAL_CONFIG = Path("/home/user/.nuwaxbot/mcp/external_servers.json")
BACKUP_SCRIPT_NAME = "index.codex-real.js"
WRAPPER_MARKER = "NUWAX_EXTERNAL_MCP_WRAPPER"


class HookInstallError(RuntimeError):
    """安装外部 MCP hook 时的错误。"""


def _load_json_file(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HookInstallError("外部 MCP 配置文件不存在: %s" % path) from exc
    except json.JSONDecodeError as exc:
        raise HookInstallError("外部 MCP 配置文件不是合法 JSON: %s" % exc) from exc

    if not isinstance(payload, dict):
        raise HookInstallError("外部 MCP 配置文件根节点必须是对象。")
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        raise HookInstallError("外部 MCP 配置文件必须包含对象类型的 mcpServers。")
    return payload


def _normalize_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _build_wrapper_script(runtime_external_config_path: Path, backup_script_name: str) -> str:
    external_config_literal = json.dumps(str(runtime_external_config_path))
    backup_script_literal = json.dumps(backup_script_name)
    return """#!/usr/bin/env node
// %s
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const WRAPPER_MARKER = %s;
const REAL_SCRIPT_NAME = %s;
const DEFAULT_EXTERNAL_CONFIG = %s;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REAL_SCRIPT_PATH = path.join(__dirname, REAL_SCRIPT_NAME);

function readJsonFile(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

function writeMergedConfig(baseConfig, externalConfigPath) {
  if (!fs.existsSync(externalConfigPath)) {
    return null;
  }

  const externalConfig = readJsonFile(externalConfigPath);
  if (!externalConfig || typeof externalConfig !== "object" || typeof externalConfig.mcpServers !== "object") {
    throw new Error("外部 MCP 配置文件缺少 mcpServers。");
  }

  const mergedConfig = {
    ...baseConfig,
    mcpServers: {
      ...(baseConfig.mcpServers || {}),
      ...(externalConfig.mcpServers || {}),
    },
  };

  const configDir = path.join(os.tmpdir(), "nuwax-mcp-configs");
  fs.mkdirSync(configDir, { recursive: true });
  const mergedPath = path.join(configDir, "mcp-config-external-merged.json");
  fs.writeFileSync(mergedPath, JSON.stringify(mergedConfig, null, 2), "utf-8");
  return mergedPath;
}

function buildChildArgs(originalArgs) {
  const args = originalArgs.slice();
  const externalConfigPath = process.env.NUWAX_EXTERNAL_MCP_CONFIG || DEFAULT_EXTERNAL_CONFIG;

  let configIndex = -1;
  let configFileIndex = -1;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--config" && i + 1 < args.length) {
      configIndex = i;
    }
    if (args[i] === "--config-file" && i + 1 < args.length) {
      configFileIndex = i;
    }
  }

  if (configIndex === -1 && configFileIndex === -1) {
    return args;
  }

  let baseConfig;
  if (configFileIndex !== -1) {
    baseConfig = readJsonFile(args[configFileIndex + 1]);
  } else {
    baseConfig = JSON.parse(args[configIndex + 1]);
  }

  const mergedPath = writeMergedConfig(baseConfig, externalConfigPath);
  if (!mergedPath) {
    return args;
  }

  if (configFileIndex !== -1) {
    args[configFileIndex + 1] = mergedPath;
    return args;
  }

  args.splice(configIndex, 2, "--config-file", mergedPath);
  return args;
}

function main() {
  let childArgs = process.argv.slice(2);
  try {
    childArgs = buildChildArgs(childArgs);
  } catch (error) {
    console.error("[nuwax-external-mcp] 合并外部 MCP 配置失败，已回退到原始参数: %%s", error && error.stack ? error.stack : String(error));
  }
  const child = spawn(process.execPath, [REAL_SCRIPT_PATH, ...childArgs], {
    stdio: "inherit",
    env: process.env,
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code === null ? 1 : code);
  });
}

main();
""" % (WRAPPER_MARKER, json.dumps(WRAPPER_MARKER), backup_script_literal, external_config_literal)


def install_hook(
    proxy_script_path: Path,
    external_config_path: Path,
    runtime_external_config_path: Path,
    *,
    force: bool = False,
) -> None:
    if not proxy_script_path.exists():
        raise HookInstallError("nuwax-mcp-stdio-proxy 入口文件不存在: %s" % proxy_script_path)

    _load_json_file(external_config_path)

    backup_path = proxy_script_path.with_name(BACKUP_SCRIPT_NAME)
    wrapper_content = _build_wrapper_script(runtime_external_config_path, BACKUP_SCRIPT_NAME)

    current_content = proxy_script_path.read_text(encoding="utf-8")
    if WRAPPER_MARKER in current_content and not force:
        return

    if not backup_path.exists():
        backup_path.write_text(current_content, encoding="utf-8")

    proxy_script_path.write_text(wrapper_content, encoding="utf-8")
    current_mode = proxy_script_path.stat().st_mode
    proxy_script_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def restore_hook(proxy_script_path: Path) -> None:
    backup_path = proxy_script_path.with_name(BACKUP_SCRIPT_NAME)
    if not backup_path.exists():
        raise HookInstallError("找不到备份文件，无法恢复: %s" % backup_path)
    proxy_script_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
    current_mode = backup_path.stat().st_mode
    proxy_script_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_external_config(config_path: Path, payload: dict, *, overwrite: bool = False) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and not overwrite:
        return
    config_path.write_text(_normalize_json(payload), encoding="utf-8")


def _build_default_payload() -> dict:
    return {
        "mcpServers": {
            "nuwax_robot_bridge": {
                "url": "http://host.docker.internal:8766/mcp",
                "headers": {
                    "Authorization": "Bearer agent-robot-bridge-token",
                },
            }
        }
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="安装 NuwaClaw 外部 MCP 配置注入 hook。")
    parser.add_argument(
        "--nuwaxbot-home",
        default=str(DEFAULT_NUWAXBOT_HOME),
        help="NuwaClaw 数据目录，一般是 ~/.nuwaxbot 对应的宿主机路径。",
    )
    parser.add_argument(
        "--proxy-script",
        default=str(DEFAULT_PROXY_SCRIPT),
        help="nuwax-mcp-stdio-proxy 的入口脚本路径。",
    )
    parser.add_argument(
        "--external-config",
        default=str(DEFAULT_EXTERNAL_CONFIG),
        help="宿主机上持久化外部 MCP JSON 文件路径。",
    )
    parser.add_argument(
        "--runtime-external-config",
        default=str(DEFAULT_RUNTIME_EXTERNAL_CONFIG),
        help="容器内运行时读取的外部 MCP JSON 文件路径。",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install", help="安装配置注入 hook。")
    install_parser.add_argument("--force", action="store_true", help="即使已经安装过也覆盖写入包装层。")
    install_parser.add_argument(
        "--init-config",
        action="store_true",
        help="如果外部配置文件不存在，则写入默认示例配置。",
    )
    install_parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="与 --init-config 一起使用时，覆盖已有外部配置文件。",
    )

    subparsers.add_parser("status", help="检查 hook 是否已安装。")
    subparsers.add_parser("restore", help="恢复原始 nuwax-mcp-stdio-proxy 入口脚本。")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    nuwaxbot_home = Path(args.nuwaxbot_home).expanduser().resolve()
    proxy_script_path = Path(args.proxy_script).expanduser().resolve()
    external_config_path = Path(args.external_config).expanduser().resolve()
    runtime_external_config_path = Path(args.runtime_external_config).expanduser()

    try:
        if args.command == "install":
            if args.init_config:
                write_external_config(
                    external_config_path,
                    _build_default_payload(),
                    overwrite=args.overwrite_config,
                )
            install_hook(
                proxy_script_path,
                external_config_path,
                runtime_external_config_path,
                force=args.force,
            )
            print("已安装 hook。")
            print("proxy=%s" % proxy_script_path)
            print("external_config=%s" % external_config_path)
            print("runtime_external_config=%s" % runtime_external_config_path)
            return 0

        if args.command == "status":
            if not proxy_script_path.exists():
                raise HookInstallError("proxy 入口脚本不存在: %s" % proxy_script_path)
            content = proxy_script_path.read_text(encoding="utf-8")
            installed = WRAPPER_MARKER in content
            print("installed=%s" % ("true" if installed else "false"))
            print("proxy=%s" % proxy_script_path)
            print("external_config=%s" % external_config_path)
            print("runtime_external_config=%s" % runtime_external_config_path)
            if external_config_path.exists():
                print("external_config_exists=true")
            else:
                print("external_config_exists=false")
            return 0 if installed else 1

        if args.command == "restore":
            restore_hook(proxy_script_path)
            print("已恢复原始 proxy 入口脚本。")
            return 0

        parser.error("未知命令。")
        return 2
    except HookInstallError as exc:
        print("错误: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
