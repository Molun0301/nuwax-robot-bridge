from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, List, Optional
import urllib.request


DEFAULT_RELEASE = "v8.4.0"
DEFAULT_REPO = "ultralytics/assets"


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="准备 YOLO 权重文件。")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="模型名或本地权重路径，例如 yolo26n.pt。",
    )
    parser.add_argument(
        "--output",
        default="runtime_data/models/yolo26n.pt",
        help="输出权重路径。默认写入项目 runtime_data/models/。",
    )
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help="Ultralytics 官方 release 版本，默认使用 v8.4.0。",
    )
    parser.add_argument(
        "--download-url",
        default="",
        help="显式指定下载地址。设置后优先使用该地址。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出文件已存在，则覆盖。",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="单次下载请求超时时间，单位秒。",
    )
    return parser


def _normalize_output_path(output: str) -> Path:
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    return output_path


def _iter_candidate_urls(model_name: str, release: str, explicit_url: str) -> Iterable[str]:
    name = Path(model_name).name
    if explicit_url.strip():
        yield explicit_url.strip()
    yield f"https://github.com/{DEFAULT_REPO}/releases/download/{release}/{name}"


def _run_external_downloader(command: List[str]) -> bool:
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError:
        return False
    return completed.returncode == 0


def _download_with_wget(url: str, target: Path, timeout_sec: float) -> bool:
    return _run_external_downloader(
        [
            "wget",
            "-4",
            "-O",
            str(target),
            "--timeout=%d" % max(1, int(timeout_sec)),
            "--tries=2",
            url,
        ]
    )


def _download_with_curl(url: str, target: Path, timeout_sec: float) -> bool:
    return _run_external_downloader(
        [
            "curl",
            "-L",
            "-4",
            "--fail",
            "--connect-timeout",
            str(max(1, int(timeout_sec))),
            "-o",
            str(target),
            url,
        ]
    )


def _download_with_urllib(url: str, target: Path, timeout_sec: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            with target.open("wb") as file:
                shutil.copyfileobj(response, file)
    except Exception:
        return False
    return True


def _download_to_temp(urls: Iterable[str], timeout_sec: float) -> Optional[Path]:
    with tempfile.TemporaryDirectory(prefix="nuwax_yolo_download_") as temp_dir:
        temp_path = Path(temp_dir) / "weights.pt"
        for url in urls:
            if _download_with_wget(url, temp_path, timeout_sec):
                if temp_path.exists() and temp_path.stat().st_size > 0:
                    return _materialize_temp_file(temp_path)
            if _download_with_curl(url, temp_path, timeout_sec):
                if temp_path.exists() and temp_path.stat().st_size > 0:
                    return _materialize_temp_file(temp_path)
            if _download_with_urllib(url, temp_path, timeout_sec):
                if temp_path.exists() and temp_path.stat().st_size > 0:
                    return _materialize_temp_file(temp_path)
    return None


def _materialize_temp_file(source: Path) -> Path:
    temp_file = tempfile.NamedTemporaryFile(prefix="nuwax_yolo_weights_", suffix=".pt", delete=False)
    temp_file.close()
    target = Path(temp_file.name)
    shutil.copy2(str(source), str(target))
    return target


def _resolve_local_source(model_name: str) -> Optional[Path]:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return None


def _ensure_min_bytes(path: Path, min_bytes: int = 100_000) -> None:
    if not path.exists():
        raise SystemExit("权重文件不存在：%s" % path)
    size = path.stat().st_size
    if size < min_bytes:
        raise SystemExit("权重文件大小异常，可能下载未完成：%s (%d bytes)" % (path, size))


def main() -> int:
    """执行权重准备流程。"""

    parser = build_parser()
    args = parser.parse_args()
    output_path = _normalize_output_path(args.output)

    if output_path.exists() and not args.overwrite:
        _ensure_min_bytes(output_path)
        print("目标权重已存在：%s" % output_path)
        return 0

    source_path = _resolve_local_source(args.model)
    temp_download: Optional[Path] = None

    if source_path is None:
        temp_download = _download_to_temp(
            _iter_candidate_urls(args.model, args.release, args.download_url),
            timeout_sec=max(1.0, float(args.timeout_sec)),
        )
        if temp_download is None:
            raise SystemExit(
                "未能自动下载 YOLO 权重。请手工下载后放到目标路径，"
                "或使用 --download-url 指向一个当前网络可访问的地址。"
            )
        source_path = temp_download

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_path), str(output_path))
    _ensure_min_bytes(output_path)
    print("YOLO 权重已准备完成：%s" % output_path)

    if temp_download is not None and temp_download.exists():
        temp_download.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
