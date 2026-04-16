"""訓練經典 CNN 分類模型（LeNet / AlexNet / VGG11 / NiN）。預設 CIFAR-10；可改用 ImageFolder。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from src.classic_cnn.data import get_cifar10_loaders, get_dataset_loaders, get_imagefolder_loaders, get_nmos_loaders
from src.classic_cnn.models import ARCH_CHOICES, build_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, required=True, choices=ARCH_CHOICES)
    p.add_argument(
        "--dataset",
        type=str,
        default="nmos",
        help="cifar10 | nmos | ImageFolder 根目錄（內含 train/、val/）",
    )
    p.add_argument("--data-root", type=str, default="data", help="CIFAR-10 下載目錄（dataset=cifar10 時）")
    p.add_argument("--dataset-root", type=str, default="dataset", help="dataset 根目錄（內含 nmos/ 與 dienumbers/）")
    p.add_argument("--nmos-dir", type=str, default="nmos", help="nmos 扁平影像資料夾（相對專案根或絕對路徑）")
    p.add_argument("--nmos-val-ratio", type=float, default=0.3, help="驗證集比例（預設 0.3；train≈70%）")
    p.add_argument("--nmos-test-ratio", type=float, default=0.0, help="測試集比例（nmos 模式可用；預設 0）")
    p.add_argument("--split-seed", type=int, default=42, help="nmos 切分隨機種子（評估時須相同）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ds = args.dataset.lower()
    class_names: list[str] | None = None
    nmos_meta: dict | None = None

    if ds == "cifar10":
        num_classes = 10
        data_root = Path(args.data_root)
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            data_root, args.arch, args.batch_size, args.val_ratio, args.workers
        )
    elif ds == "nmos":
        nmos_root = Path(args.nmos_dir)
        if not nmos_root.is_absolute():
            nmos_root = _ROOT / nmos_root
        train_loader, val_loader, test_loader, num_classes, class_names = get_nmos_loaders(
            nmos_root,
            args.arch,
            args.batch_size,
            args.nmos_val_ratio,
            args.nmos_test_ratio,
            args.workers,
            seed=args.split_seed,
        )
        nmos_meta = {
            "nmos_dir": str(nmos_root.resolve()),
            "nmos_val_ratio": float(args.nmos_val_ratio),
            "nmos_test_ratio": float(args.nmos_test_ratio),
            "split_seed": int(args.split_seed),
            "class_names": class_names,
        }
    elif ds == "dataset":
        dataset_root = Path(args.dataset_root)
        if not dataset_root.is_absolute():
            dataset_root = _ROOT / dataset_root
        train_loader, val_loader, num_classes, class_names = get_dataset_loaders(
            dataset_root,
            args.arch,
            args.batch_size,
            args.nmos_val_ratio,
            args.workers,
            seed=args.split_seed,
        )
        test_loader = val_loader
        nmos_meta = {
            "dataset_root": str(dataset_root.resolve()),
            "nmos_val_ratio": float(args.nmos_val_ratio),
            "split_seed": int(args.split_seed),
            "class_names": class_names,
        }
    else:
        root = Path(args.dataset)
        train_loader, val_loader, test_loader, num_classes = get_imagefolder_loaders(
            root, args.arch, args.batch_size, args.workers
        )

    model = build_model(args.arch, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = _ROOT / "runs_classic" / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        model.eval()
        correct = 0
        tot = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
        acc = correct / max(tot, 1)
        if acc >= best_val:
            best_val = acc
            ckpt: dict = {
                "model": model.state_dict(),
                "arch": args.arch,
                "num_classes": num_classes,
                "dataset": args.dataset,
            }
            if class_names is not None:
                ckpt["class_names"] = class_names
            if nmos_meta is not None:
                ckpt.update(nmos_meta)
            torch.save(ckpt, out_dir / "best.pt")
        print(f"epoch {epoch+1}/{args.epochs}  loss={total_loss/n:.4f}  val_acc={acc:.4f}")

    elapsed = time.perf_counter() - t0
    meta = {
        "arch": args.arch,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "training_time_sec": elapsed,
        "best_val_acc": best_val,
        "weights": str(out_dir / "best.pt"),
        "num_classes": num_classes,
    }
    if nmos_meta is not None:
        meta.update(nmos_meta)
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完成。權重：{out_dir / 'best.pt'}，耗時 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
