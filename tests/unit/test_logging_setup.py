from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient

from gateways.relay.server import create_relay_app
from settings import LoggingConfig, RelayServerConfig, configure_logging, load_config


def test_load_config_prefers_nuwax_log_variables(monkeypatch, tmp_path: Path) -> None:
    """新日志变量名应优先覆盖旧变量名。"""

    new_log_dir = tmp_path / "new_logs"
    old_log_dir = tmp_path / "old_logs"
    new_log_file = new_log_dir / "nuwax_robot_bridge.log"
    old_log_file = old_log_dir / "go2_proxy.log"

    monkeypatch.setenv("NUWAX_LOG_DIR", str(new_log_dir))
    monkeypatch.setenv("NUWAX_LOG_FILE", str(new_log_file))
    monkeypatch.setenv("NUWAX_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("NUWAX_LOG_MAX_LINES", "123")
    monkeypatch.setenv("NUWAX_LOG_TRIM_CHECK_INTERVAL", "9")
    monkeypatch.setenv("GO2_PROXY_LOG_DIR", str(old_log_dir))
    monkeypatch.setenv("GO2_PROXY_LOG_FILE", str(old_log_file))
    monkeypatch.setenv("GO2_PROXY_LOG_LEVEL", "INFO")
    monkeypatch.setenv("GO2_PROXY_LOG_MAX_LINES", "100000")
    monkeypatch.setenv("GO2_PROXY_LOG_TRIM_CHECK_INTERVAL", "200")

    config = load_config()

    assert config.logging.log_dir == str(new_log_dir)
    assert config.logging.log_file == str(new_log_file)
    assert config.logging.level == "WARNING"
    assert config.logging.max_lines == 123
    assert config.logging.trim_check_interval == 9


def test_configure_logging_writes_file_and_request_log(tmp_path: Path) -> None:
    """日志初始化后应能写入文件，并记录 HTTP 请求摘要。"""

    log_dir = tmp_path / "logs"
    log_file = log_dir / "runtime.log"
    configure_logging(
        LoggingConfig(
            log_dir=str(log_dir),
            log_file=str(log_file),
            level="INFO",
            max_lines=1000,
            trim_check_interval=10,
        )
    )

    app = create_relay_app(
        RelayServerConfig(
            enabled=True,
            host="127.0.0.1",
            port=9766,
            upstream_base_url="http://example.invalid",
            incoming_tokens=("relay-token",),
            upstream_token="agent-token",
        )
    )

    with TestClient(app) as client:
        response = client.get("/relay/health")
        assert response.status_code == 200

    for handler in logging.getLogger().handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            flush()

    content = log_file.read_text(encoding="utf-8")
    assert "Relay 请求完成" in content
    assert "path=/relay/health" in content
