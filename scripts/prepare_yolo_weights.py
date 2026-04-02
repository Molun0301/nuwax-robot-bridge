from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Iterable, Optional


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="预下载并固化 YOLO 权重文件。")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Ultralytics 模型名或本地权重路径，例如 yolo26n.pt。",
    )
    parser.add_argument(
        "--output",
        default="runtime_data/models/yolo26n.pt",
        help="输出权重路径。默认写入项目 runtime_data/models/。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出文件已存在，则覆盖。",
    )
    return parser


def _iter_candidate_paths(model: object, model_name: str) -> Iterable[Path]:
    """尽量从 Ultralytics 实例和常见缓存目录中定位已下载的权重文件。"""

    requested = Path(model_name).expanduser()
    if requested.exists():
        yield requested.resolve()

    for attr_name in ("ckpt_path", "pt_path", "checkpoint_path"):
        value = getattr(model, attr_name, None)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if path.exists():
                yield path.resolve()

    ckpt_value = getattr(model, "ckpt", None)
    if isinstance(ckpt_value, str) and ckpt_value.strip():
        path = Path(ckpt_value).expanduser()
        if path.exists():
            yield path.resolve()

    filename = requested.name
    search_roots = (
        Path.cwd(),
        Path.home() / ".cache" / "ultralytics",
        Path.home() / ".config" / "Ultralytics",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
    )
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob(filename):
                if path.is_file():
                    yield path.resolve()
        except OSError:
            continue


def _resolve_downloaded_weights(model: object, model_name: str) -> Optional[Path]:
    """返回最可能的权重文件路径。"""

    seen = set()
    candidates = []
    for path in _iter_candidate_paths(model, model_name):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> int:
    """执行权重准备流程。"""

    parser = build_parser()
    args = parser.parse_args()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    if output_path.exists() and not args.overwrite:
        print("目标权重已存在：%s" % output_path)
        return 0

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("当前环境未安装 ultralytics，无法准备 YOLO 权重。") from exc

    # 触发 Ultralytics 的本地加载或官方权重下载逻辑。
    model = YOLO(args.model)
    source_path = _resolve_downloaded_weights(model, args.model)
    if source_path is None:
        raise SystemExit(
            "未能定位已下载的权重文件。请确认当前机器能联网访问 Ultralytics 官方模型资源，"
            "或手工把权重放到目标路径。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_path), str(output_path))
    print("YOLO 权重已准备完成：%s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
