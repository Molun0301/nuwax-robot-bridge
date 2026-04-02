from pathlib import Path

from scripts.nuwaclaw_external_mcp_hook import (
    BACKUP_SCRIPT_NAME,
    WRAPPER_MARKER,
    _build_default_payload,
    install_hook,
    restore_hook,
    write_external_config,
)


def test_install_hook_writes_wrapper_and_backup(tmp_path: Path) -> None:
    proxy_script = tmp_path / "index.js"
    external_config = tmp_path / "mcp" / "external_servers.json"
    runtime_external_config = Path("/home/user/.nuwaxbot/mcp/external_servers.json")

    proxy_script.write_text("#!/usr/bin/env node\nconsole.log('real');\n", encoding="utf-8")
    write_external_config(external_config, _build_default_payload(), overwrite=True)

    install_hook(proxy_script, external_config, runtime_external_config)

    wrapper_content = proxy_script.read_text(encoding="utf-8")
    backup_path = proxy_script.with_name(BACKUP_SCRIPT_NAME)

    assert WRAPPER_MARKER in wrapper_content
    assert str(runtime_external_config) in wrapper_content
    assert backup_path.exists()
    assert "console.log('real');" in backup_path.read_text(encoding="utf-8")


def test_restore_hook_recovers_original_script(tmp_path: Path) -> None:
    proxy_script = tmp_path / "index.js"
    external_config = tmp_path / "mcp" / "external_servers.json"
    runtime_external_config = Path("/home/user/.nuwaxbot/mcp/external_servers.json")
    original_content = "#!/usr/bin/env node\nconsole.log('real');\n"

    proxy_script.write_text(original_content, encoding="utf-8")
    write_external_config(external_config, _build_default_payload(), overwrite=True)
    install_hook(proxy_script, external_config, runtime_external_config)

    restore_hook(proxy_script)

    assert proxy_script.read_text(encoding="utf-8") == original_content

