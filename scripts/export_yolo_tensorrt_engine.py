from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="导出 YOLO TensorRT 引擎文件。")
    parser.add_argument("--weights", default="yolo26n.pt", help="输入权重路径，通常为 .pt 文件。")
    parser.add_argument("--output", default="", help="输出 engine 文件路径。留空时使用 Ultralytics 默认导出位置。")
    parser.add_argument("--imgsz", type=int, default=640, help="导出时的图像尺寸。")
    parser.add_argument("--device", default="", help="导出设备，例如 0、cuda:0、dla:0。")
    parser.add_argument("--half", action="store_true", default=True, help="启用 FP16 导出。")
    parser.add_argument("--no-half", action="store_false", dest="half", help="关闭 FP16 导出。")
    parser.add_argument("--int8", action="store_true", help="启用 INT8 导出。")
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace 大小，单位 GiB。")
    return parser


def main() -> int:
    """导出 TensorRT engine。"""

    parser = build_parser()
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("当前环境未安装 ultralytics，无法导出 TensorRT 引擎。") from exc

    model = YOLO(args.weights)
    export_kwargs = {
        "format": "engine",
        "imgsz": max(32, int(args.imgsz)),
        "half": bool(args.half),
        "int8": bool(args.int8),
    }
    if args.device:
        export_kwargs["device"] = args.device
    if args.workspace is not None:
        export_kwargs["workspace"] = float(args.workspace)

    exported_path = Path(model.export(**export_kwargs))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if exported_path.resolve() != output_path.resolve():
            shutil.move(str(exported_path), str(output_path))
        exported_path = output_path

    print("TensorRT 引擎导出完成：%s" % exported_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

