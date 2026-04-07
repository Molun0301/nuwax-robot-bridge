from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence, Tuple


DEFAULT_TEXT_MODEL_ID = "BAAI/bge-m3"
DEFAULT_IMAGE_MODEL_ID = "openai/clip-vit-base-patch32"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runtime_data" / "models"


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="手工下载记忆向量模型到本地目录。")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="模型输出根目录，默认写入 runtime_data/models/。",
    )
    parser.add_argument(
        "--text-model-id",
        default=DEFAULT_TEXT_MODEL_ID,
        help="文本向量模型仓库 ID，默认使用 BAAI/bge-m3。",
    )
    parser.add_argument(
        "--image-model-id",
        default=DEFAULT_IMAGE_MODEL_ID,
        help="图像向量模型仓库 ID，默认使用 openai/clip-vit-base-patch32。",
    )
    parser.add_argument(
        "--text-output-dir",
        default="",
        help="文本向量模型输出目录。未设置时自动写入 output-root 下的固定目录。",
    )
    parser.add_argument(
        "--image-output-dir",
        default="",
        help="图像向量模型输出目录。未设置时自动写入 output-root 下的固定目录。",
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="跳过文本向量模型下载。",
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="跳过图像向量模型下载。",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face 访问令牌。未设置时复用环境变量 HF_TOKEN。",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="下载超时时间，单位秒。",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="强制重新下载并覆盖目标目录内容。",
    )
    return parser


def _normalize_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _repo_id_to_dir_name(repo_id: str) -> str:
    return repo_id.strip().replace("/", "__")


def _resolve_output_dir(output_root: Path, explicit_dir: str, repo_id: str) -> Path:
    if explicit_dir.strip():
        return _normalize_path(explicit_dir)
    return output_root / _repo_id_to_dir_name(repo_id)


def _looks_prepared(target_dir: Path, required_files: Sequence[str]) -> bool:
    if not target_dir.is_dir():
        return False
    return all((target_dir / file_name).exists() for file_name in required_files)


def _iter_download_jobs(args: argparse.Namespace) -> Iterable[Tuple[str, str, Path, Tuple[str, ...]]]:
    output_root = _normalize_path(args.output_root)
    if not args.skip_text:
        yield (
            "文本向量模型",
            args.text_model_id.strip(),
            _resolve_output_dir(output_root, args.text_output_dir, args.text_model_id),
            ("modules.json",),
        )
    if not args.skip_image:
        yield (
            "图像向量模型",
            args.image_model_id.strip(),
            _resolve_output_dir(output_root, args.image_output_dir, args.image_model_id),
            ("config.json", "preprocessor_config.json"),
        )


def _download_snapshot(
    *,
    repo_id: str,
    target_dir: Path,
    token: str,
    timeout_sec: float,
    force_download: bool,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "当前环境未安装 huggingface_hub。请先执行 "
            "`python3 -m pip install -r requirements-memory-models.txt`。"
        ) from exc

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(max(10, int(timeout_sec)))
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(max(10, int(timeout_sec)))
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            token=token or None,
            force_download=force_download,
        )
    except TypeError:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            token=token or None,
        )
    return target_dir


def _prepare_one_model(
    *,
    label: str,
    repo_id: str,
    target_dir: Path,
    required_files: Sequence[str],
    token: str,
    timeout_sec: float,
    force_download: bool,
) -> Path:
    if not repo_id:
        raise SystemExit(f"{label}缺少仓库 ID。")

    if _looks_prepared(target_dir, required_files) and not force_download:
        print(f"{label}已存在，跳过下载：{target_dir}")
        return target_dir

    print(f"开始下载{label}：repo_id={repo_id} target={target_dir}")
    _download_snapshot(
        repo_id=repo_id,
        target_dir=target_dir,
        token=token,
        timeout_sec=timeout_sec,
        force_download=force_download,
    )

    missing_files = [name for name in required_files if not (target_dir / name).exists()]
    if missing_files:
        raise SystemExit(
            f"{label}下载后缺少必要文件：{', '.join(missing_files)}。"
            f"目标目录：{target_dir}"
        )

    print(f"{label}已准备完成：{target_dir}")
    return target_dir


def main() -> int:
    """执行模型下载流程。"""

    parser = build_parser()
    args = parser.parse_args()
    jobs = list(_iter_download_jobs(args))
    if not jobs:
        raise SystemExit("至少保留一个模型下载任务。")

    prepared_paths = {}
    for label, repo_id, target_dir, required_files in jobs:
        prepared_paths[label] = _prepare_one_model(
            label=label,
            repo_id=repo_id,
            target_dir=target_dir,
            required_files=required_files,
            token=args.hf_token.strip(),
            timeout_sec=max(10.0, float(args.timeout_sec)),
            force_download=bool(args.force_download),
        )

    print("")
    print("建议把以下配置写入 .env：")
    if "文本向量模型" in prepared_paths:
        print(f"NUWAX_MEMORY_TEXT_EMBEDDING_MODEL={prepared_paths['文本向量模型']}")
    if "图像向量模型" in prepared_paths:
        print(f"NUWAX_MEMORY_IMAGE_EMBEDDING_MODEL={prepared_paths['图像向量模型']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
