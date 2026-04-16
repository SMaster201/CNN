"""依序訓練並評估多種 CNN（dataset 模式：dataset/nmos + dataset/dienumbers）。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

ARCHS = (
    "lenet",
    "alexnet",
    "vgg11",
    "nin",
    "googlenet",
    "resnet18",
    "densenet121",
)

# 僅 train_classifier 使用；eval 不認識，需剝除以免報錯
_TRAIN_ONLY_FLAGS = frozenset({"--epochs", "--lr", "--val-ratio"})


def _strip_train_only_for_eval(argv: list[str]) -> list[str]:
    """保留可傳給 eval_classifier 的參數（例如 --device、--batch-size）。"""
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in _TRAIN_ONLY_FLAGS:
            i += 1
            if i < len(argv) and not str(argv[i]).startswith("-"):
                i += 1
            continue
        if tok.startswith("--") and "=" not in tok:
            out.append(tok)
            i += 1
            if i < len(argv) and not str(argv[i]).startswith("-"):
                out.append(argv[i])
                i += 1
        else:
            out.append(tok)
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="對 ARCHS 依序 train→eval。資料集固定為 --dataset dataset，請用 --dataset-root 指定根目錄。"
    )
    ap.add_argument(
        "--dataset-root",
        default=str(ROOT / "dataset"),
        help="資料集根目錄（需含 nmos/ 與 dienumbers/）",
    )
    args, extra = ap.parse_known_args()
    dataset_root = Path(args.dataset_root).expanduser()
    dataset_s = str(dataset_root)

    # 必須放在命令尾端，否則 extra 裡的 --dataset 會覆寫成 ImageFolder 分支
    train_tail = ["--dataset", "dataset", "--dataset-root", dataset_s]
    eval_tail = ["--dataset", "dataset", "--dataset-root", dataset_s, "--max-test-samples", "0"]
    eval_extra = _strip_train_only_for_eval(extra)

    for arch in ARCHS:
        train_cmd = [
            PY,
            str(ROOT / "src" / "train_classifier.py"),
            "--arch",
            arch,
        ] + extra + train_tail
        eval_cmd = [
            PY,
            str(ROOT / "src" / "eval_classifier.py"),
            "--arch",
            arch,
        ] + eval_extra + eval_tail
        print("+", " ".join(train_cmd), flush=True)
        subprocess.run(train_cmd, cwd=str(ROOT), check=True)
        print("+", " ".join(eval_cmd), flush=True)
        subprocess.run(eval_cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
