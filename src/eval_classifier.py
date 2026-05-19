"""評估經典 CNN 分類：準確率、macro P/R/F1、推論時間、GFLOPS、可選 VRAM。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from thop import profile
from torch.utils.data import DataLoader, Subset

from src.classic_cnn.data import (
    get_cifar10_test_ids_and_loader,
    get_dataset_eval_paths_and_loader,
    get_dienumbers_eval_paths_and_loader,
    get_imagefolder_test_paths_and_loader,
    get_nmos_test_paths_and_loader,
    resolve_image_size,
)
from src.classic_cnn.models import arch_uses_imagenet_224, build_model

CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _accuracy_and_macro_prf1(ys: list[int], ps: list[int], num_classes: int) -> tuple[float, float, float, float]:
    y = np.asarray(ys, dtype=np.int64)
    p = np.asarray(ps, dtype=np.int64)
    acc = float((y == p).mean()) if len(y) else 0.0
    precs, recs, f1s = [], [], []
    for c in range(num_classes):
        tp = int(np.sum((y == c) & (p == c)))
        fp = int(np.sum((y != c) & (p == c)))
        fn = int(np.sum((y == c) & (p != c)))
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
        precs.append(pr)
        recs.append(rc)
        f1s.append(f1)
    return acc, float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def _cuda_idx() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.current_device()
    except Exception:
        return None


def _reset_cuda_peak_safe(idx: int | None) -> None:
    if idx is None:
        return
    try:
        torch.cuda.reset_peak_memory_stats(idx)
    except RuntimeError:
        pass


def _peak_mb_safe(idx: int | None) -> float | None:
    if idx is None:
        return None
    try:
        return torch.cuda.max_memory_allocated(idx) / (1024**2)
    except RuntimeError:
        return None


def _cap_test_samples(
    image_ids: list[str],
    loader: DataLoader,
    max_samples: int,
) -> tuple[list[str], DataLoader]:
    """僅評估前 max_samples 張（與 image_ids 順序一致）；max_samples<=0 表示不限制。"""
    if max_samples <= 0 or len(image_ids) <= max_samples:
        return image_ids, loader
    n = max_samples
    ds = loader.dataset
    sub = Subset(ds, list(range(n)))
    kw = dict(
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        shuffle=False,
    )
    return image_ids[:n], DataLoader(sub, **kw)


def _gflops(model: torch.nn.Module, arch: str, image_size: tuple[int, int] | None = None) -> float | None:
    try:
        m = model
        m.eval()
        dev = next(m.parameters()).device
        h, w = resolve_image_size(arch, image_size)
        dummy = torch.zeros(1, 3, h, w, device=dev)
        flops, _ = profile(m, inputs=(dummy,), verbose=False)
        return float(flops) / 1e9
    except Exception:
        return None


def _char_level_metrics(rows: list[dict]) -> dict:
    digit_total = 0
    digit_correct = 0
    alpha_total = 0
    alpha_correct = 0
    wrong_digit_gt: dict[str, int] = {}
    wrong_alpha_gt: dict[str, int] = {}

    for r in rows:
        gt = str(r["gt_label"])
        pr = str(r["pred_label"])
        for idx, ch in enumerate(gt):
            pch = pr[idx] if idx < len(pr) else None
            if ch.isdigit():
                digit_total += 1
                if pch == ch:
                    digit_correct += 1
                else:
                    wrong_digit_gt[ch] = wrong_digit_gt.get(ch, 0) + 1
            elif ch.isalpha():
                alpha_total += 1
                if pch is not None and pch.upper() == ch.upper():
                    alpha_correct += 1
                else:
                    key = ch.upper()
                    wrong_alpha_gt[key] = wrong_alpha_gt.get(key, 0) + 1

    def _top1(d: dict[str, int]) -> dict | None:
        if not d:
            return None
        k = max(d, key=d.get)
        return {"char": k, "count": int(d[k])}

    return {
        "digit_position_accuracy": (digit_correct / digit_total) if digit_total else None,
        "alpha_position_accuracy": (alpha_correct / alpha_total) if alpha_total else None,
        "digit_position_total": int(digit_total),
        "alpha_position_total": int(alpha_total),
        "most_wrong_digit_in_label": _top1(wrong_digit_gt),
        "most_wrong_alpha_in_label": _top1(wrong_alpha_gt),
    }


def _per_class_accuracy(ys: list[int], ps: list[int], class_names: list[str]) -> dict[str, dict]:
    """每個類別標籤（如 A3）的樣本數、正確數、準確率。"""
    y = np.asarray(ys, dtype=np.int64)
    p = np.asarray(ps, dtype=np.int64)
    out: dict[str, dict] = {}
    for c, name in enumerate(class_names):
        mask = y == c
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int((p[mask] == c).sum())
        out[str(name)] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
        }
    return out


def _per_alnum_symbol_accuracy(rows: list[dict]) -> dict:
    """
    dienumbers 等 2 字元標籤：統計「有出現過」的英文字母（第 1 碼）與數字（第 2 碼）各自準確率。
    """
    letters: dict[str, dict[str, int]] = {}
    digits: dict[str, dict[str, int]] = {}
    for r in rows:
        gt = str(r["gt_label"])
        pr = str(r["pred_label"])
        if len(gt) >= 1 and gt[0].isalpha():
            ch = gt[0].upper()
            letters.setdefault(ch, {"total": 0, "correct": 0})
            letters[ch]["total"] += 1
            if len(pr) >= 1 and pr[0].upper() == ch:
                letters[ch]["correct"] += 1
        if len(gt) >= 2 and gt[1].isdigit():
            ch = gt[1]
            digits.setdefault(ch, {"total": 0, "correct": 0})
            digits[ch]["total"] += 1
            if len(pr) >= 2 and pr[1] == ch:
                digits[ch]["correct"] += 1

    def _pack(d: dict[str, dict[str, int]]) -> dict[str, dict]:
        return {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
            }
            for k, v in sorted(d.items())
        }

    return {"per_letter": _pack(letters), "per_digit": _pack(digits)}


def _dataset_split_summary(rows: list[dict]) -> dict:
    nmos_total = 0
    nmos_correct = 0
    dienumbers_total = 0
    dienumbers_correct = 0
    for r in rows:
        image_id = str(r["image_id"]).replace("\\", "/").lower()
        ok = int(r["correct"])
        if "/test/nmos/" in image_id or "/nmos/" in image_id:
            nmos_total += 1
            nmos_correct += ok
        elif "/test/dienumbers/" in image_id or "/dienumbers/" in image_id:
            dienumbers_total += 1
            dienumbers_correct += ok
    return {
        "nmos_test_total": int(nmos_total),
        "nmos_test_correct": int(nmos_correct),
        "dienumbers_test_total": int(dienumbers_total),
        "dienumbers_test_correct": int(dienumbers_correct),
    }


def _evaluate_loader_and_write(
    *,
    model: torch.nn.Module,
    device: torch.device,
    arch: str,
    num_classes: int,
    class_names: list[str],
    image_ids: list[str],
    loader: DataLoader,
    preprocess: str,
    wpath: Path,
    metrics_basename_suffix: str,
    per_image_csv_arg: str,
    no_per_image_csv: bool,
    max_test_samples: int,
    dataset_test_split: str | None = None,
    image_size: tuple[int, int] | None = None,
    report_alnum_accuracy: bool = False,
) -> dict:
    """跑單一 DataLoader 的推論並寫入 metrics（與可選 CSV）。"""
    out_dir = _ROOT / "results_classic" / arch
    out_dir.mkdir(parents=True, exist_ok=True)

    cuda_i = _cuda_idx() if device.type == "cuda" else None
    _reset_cuda_peak_safe(cuda_i)

    ys: list[int] = []
    ps: list[int] = []
    per_rows: list[dict] = []
    t_infer = 0.0
    n_img = 0
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_infer += time.perf_counter() - t0
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            bs = x.size(0)
            y_np = y.cpu().numpy()
            pred_np = pred.cpu().numpy()
            conf_np = conf.cpu().numpy()
            for i in range(bs):
                gt_i = int(y_np[i])
                pr_i = int(pred_np[i])
                stem = Path(image_ids[offset + i]).name
                gt_label = class_names[gt_i] if gt_i < len(class_names) else str(gt_i)
                pred_label = class_names[pr_i] if pr_i < len(class_names) else str(pr_i)
                digit_total = sum(ch.isdigit() for ch in gt_label)
                alpha_total = sum(ch.isalpha() for ch in gt_label)
                digit_correct = 0
                alpha_correct = 0
                for j, ch in enumerate(gt_label):
                    pch = pred_label[j] if j < len(pred_label) else None
                    if ch.isdigit() and pch == ch:
                        digit_correct += 1
                    elif ch.isalpha() and pch is not None and pch.upper() == ch.upper():
                        alpha_correct += 1
                row = {
                    "image_id": image_ids[offset + i],
                    "filename": f"{arch}_{stem}",
                    "gt_class_index": gt_i,
                    "gt_label": gt_label,
                    "pred_class_index": pr_i,
                    "pred_label": pred_label,
                    "correct": int(gt_i == pr_i),
                    "confidence": float(conf_np[i]),
                    "digit_correct_count": digit_correct,
                    "digit_total_count": digit_total,
                    "alpha_correct_count": alpha_correct,
                    "alpha_total_count": alpha_total,
                }
                per_rows.append(row)
            ys.extend(y_np.tolist())
            ps.extend(pred_np.tolist())
            offset += bs
            n_img += x.size(0)

    acc, prec, rec, f1 = _accuracy_and_macro_prf1(ys, ps, num_classes)
    ms_per_img = (t_infer / max(n_img, 1)) * 1000.0
    vram_mb = _peak_mb_safe(cuda_i)
    gfl = _gflops(model, arch, image_size)
    resolved_size = list(resolve_image_size(arch, image_size))

    metrics_name = f"{arch}_metrics{metrics_basename_suffix}.json"
    payload: dict = {
        "arch": arch,
        "preprocess": preprocess,
        "image_size": resolved_size,
        "accuracy": acc,
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "inference_time_ms_per_image": ms_per_img,
        "gflops": gfl,
        "vram_peak_mb": vram_mb,
        "num_test_samples": n_img,
        "max_test_samples": max_test_samples,
        "weights": str(wpath.resolve()),
    }
    if dataset_test_split is not None:
        payload["dataset_test_split"] = dataset_test_split
    payload.update(_char_level_metrics(per_rows))
    payload.update(_dataset_split_summary(per_rows))
    payload["per_class_accuracy"] = _per_class_accuracy(ys, ps, class_names)
    if report_alnum_accuracy:
        payload.update(_per_alnum_symbol_accuracy(per_rows))

    if not no_per_image_csv and per_rows:
        if per_image_csv_arg:
            pcsv = Path(per_image_csv_arg)
            if not pcsv.is_absolute():
                pcsv = _ROOT / pcsv
            csv_path = pcsv.with_name(f"{pcsv.stem}{metrics_basename_suffix}{pcsv.suffix}")
        else:
            csv_path = out_dir / f"{arch}_per_image_predictions{metrics_basename_suffix}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "image_id",
            "filename",
            "gt_class_index",
            "gt_label",
            "pred_class_index",
            "pred_label",
            "correct",
            "confidence",
            "digit_correct_count",
            "digit_total_count",
            "alpha_correct_count",
            "alpha_total_count",
        ]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
            w = csv.DictWriter(fp, fieldnames=fields)
            w.writeheader()
            w.writerows(per_rows)
        payload["per_image_csv"] = str(csv_path.resolve())

    metrics_path = out_dir / metrics_name
    payload["metrics_json"] = str(metrics_path.resolve())
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if report_alnum_accuracy and payload.get("per_class_accuracy"):
        print("\n--- 各類別標籤準確率 ---", flush=True)
        for label, st in sorted(payload["per_class_accuracy"].items()):
            print(
                f"  {label}: {st['correct']}/{st['total']} = {st['accuracy']:.4f}",
                flush=True,
            )
        pl = payload.get("per_letter") or {}
        pd = payload.get("per_digit") or {}
        if pl:
            print("\n--- 英文字母（標籤第 1 碼）---", flush=True)
            for ch, st in pl.items():
                print(f"  {ch}: {st['correct']}/{st['total']} = {st['accuracy']:.4f}", flush=True)
        if pd:
            print("\n--- 數字（標籤第 2 碼）---", flush=True)
            for ch, st in pd.items():
                print(f"  {ch}: {st['correct']}/{st['total']} = {st['accuracy']:.4f}", flush=True)

    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="", help="train_classifier 產生的 best.pt")
    p.add_argument(
        "--arch",
        type=str,
        default="",
        help="若未指定 --weights，則使用 runs_classic/<arch>/best.pt（見 models.ARCH_CHOICES）",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="nmos",
        help="cifar10 | nmos | dataset | dienumbers | ImageFolder 根目錄",
    )
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--dataset-root", type=str, default="dataset", help="dataset 根目錄（內含 nmos/ 與 dienumbers/）")
    p.add_argument("--nmos-dir", type=str, default="nmos", help="nmos 資料夾（評估時若權重內有 nmos_dir 則優先使用）")
    p.add_argument("--nmos-val-ratio", type=float, default=0.3)
    p.add_argument("--nmos-test-ratio", type=float, default=0.0)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--preprocess", type=str, default="", help="留空則沿用權重內設定；可指定 none|p10")
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="評估時最多使用的測試張數（預設 0；<=0 表示使用全部測試集）",
    )
    p.add_argument(
        "--per-image-csv",
        type=str,
        default="",
        help="逐張 GT/預測輸出 CSV；留空則 results_classic/<arch>/<arch>_per_image_predictions.csv",
    )
    p.add_argument("--no-per-image-csv", action="store_true", help="不寫逐張預測 CSV")
    p.add_argument(
        "--dataset-test-split",
        type=str,
        default="combined",
        choices=("combined", "nmos", "dienumbers", "separate"),
        help="dataset 模式：測試集來源；separate 會各跑一次並輸出兩份 metrics/csv",
    )
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Resize 高度與寬度；未指定則用權重內 image_size 或依架構預設",
    )
    args = p.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if args.weights:
        wpath = Path(args.weights)
    elif args.arch:
        wpath = _ROOT / "runs_classic" / args.arch / "best.pt"
    else:
        raise SystemExit("請提供 --weights，或同時用 --arch（例如 lenet）以自動找 runs_classic/<arch>/best.pt")

    if not wpath.is_file():
        raise FileNotFoundError(f"找不到權重: {wpath}")

    ckpt = torch.load(wpath, map_location=device)
    arch = ckpt["arch"]
    num_classes = int(ckpt["num_classes"])
    model = build_model(arch, num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = args.dataset.lower()
    ckpt_preprocess = str(ckpt.get("preprocess", "none")).lower()
    preprocess = (args.preprocess or ckpt_preprocess).lower()
    if preprocess not in {"none", "p10"}:
        raise ValueError("preprocess 只支援 none 或 p10")

    image_size: tuple[int, int] | None = None
    if args.image_size is not None:
        image_size = (int(args.image_size[0]), int(args.image_size[1]))
    elif isinstance(ckpt.get("image_size"), (list, tuple)) and len(ckpt["image_size"]) == 2:
        image_size = (int(ckpt["image_size"][0]), int(ckpt["image_size"][1]))

    class_names: list[str] = []
    eval_jobs: list[tuple[str, list[str], DataLoader, str | None]] = []
    report_alnum = ds == "dienumbers"

    if ds == "cifar10":
        image_ids, loader = get_cifar10_test_ids_and_loader(
            Path(args.data_root), arch, args.batch_size, args.workers
        )
        class_names = list(CIFAR10_CLASS_NAMES)
        eval_jobs = [("", image_ids, loader, None)]
    elif ds == "nmos":
        nmos_dir = ckpt.get("nmos_dir") or args.nmos_dir
        nmos_root = Path(nmos_dir)
        if not nmos_root.is_absolute():
            nmos_root = _ROOT / nmos_root
        vr = float(ckpt.get("nmos_val_ratio", args.nmos_val_ratio))
        tr = float(ckpt.get("nmos_test_ratio", args.nmos_test_ratio))
        sd = int(ckpt.get("split_seed", args.split_seed))
        image_ids, loader = get_nmos_test_paths_and_loader(
            nmos_root, arch, args.batch_size, args.workers, vr, tr, preprocess, sd, image_size
        )
        raw_names = ckpt.get("class_names")
        if isinstance(raw_names, list) and len(raw_names) == num_classes:
            class_names = [str(x) for x in raw_names]
        else:
            class_names = [str(i) for i in range(num_classes)]
        eval_jobs = [("", image_ids, loader, None)]
    elif ds == "dienumbers":
        dataset_root = ckpt.get("dataset_root") or args.dataset_root
        root = Path(dataset_root)
        if not root.is_absolute():
            root = _ROOT / root
        die_train = ckpt.get("dienumbers_dir")
        die_train_root = Path(die_train) if die_train else (root / "dienumbers")
        if not die_train_root.is_absolute():
            die_train_root = _ROOT / die_train_root
        args.max_test_samples = 0
        raw_names = ckpt.get("class_names")
        iids, loader, eval_class_names = get_dienumbers_eval_paths_and_loader(
            root,
            arch,
            args.batch_size,
            args.workers,
            preprocess,
            image_size,
            die_train_root,
        )
        if isinstance(raw_names, list) and len(raw_names) == num_classes:
            class_names = [str(x) for x in raw_names]
        else:
            class_names = eval_class_names
        eval_jobs = [("", iids, loader, "dienumbers")]
    elif ds == "dataset":
        dataset_root = ckpt.get("dataset_root") or args.dataset_root
        root = Path(dataset_root)
        if not root.is_absolute():
            root = _ROOT / root
        vr = float(ckpt.get("nmos_val_ratio", args.nmos_val_ratio))
        sd = int(ckpt.get("split_seed", args.split_seed))
        raw_names = ckpt.get("class_names")
        if isinstance(raw_names, list) and len(raw_names) == num_classes:
            class_names = [str(x) for x in raw_names]
        else:
            class_names = [str(i) for i in range(num_classes)]
        # dataset 模式直接讀 dataset/test 下完整測試集；不再全域二次截斷。
        args.max_test_samples = 0
        split_arg = args.dataset_test_split.lower()
        if split_arg == "separate":
            for tag in ("nmos", "dienumbers"):
                iids, ld = get_dataset_eval_paths_and_loader(
                    root,
                    arch,
                    args.batch_size,
                    args.workers,
                    vr,
                    preprocess,
                    sd,
                    test_split=tag,
                    image_size=image_size,
                )
                eval_jobs.append((f"_{tag}", iids, ld, tag))
        elif split_arg in ("nmos", "dienumbers"):
            iids, ld = get_dataset_eval_paths_and_loader(
                root,
                arch,
                args.batch_size,
                args.workers,
                vr,
                preprocess,
                sd,
                test_split=split_arg,
                image_size=image_size,
            )
            eval_jobs.append((f"_{split_arg}", iids, ld, split_arg))
            if split_arg == "dienumbers":
                report_alnum = True
        else:
            iids, ld = get_dataset_eval_paths_and_loader(
                root,
                arch,
                args.batch_size,
                args.workers,
                vr,
                preprocess,
                sd,
                test_split="combined",
                image_size=image_size,
            )
            eval_jobs.append(("", iids, ld, "combined"))
    else:
        image_ids, loader, if_classes = get_imagefolder_test_paths_and_loader(
            Path(args.dataset), arch, args.batch_size, args.workers
        )
        class_names = [str(c) for c in if_classes]
        eval_jobs = [("", image_ids, loader, None)]

    outputs: list[dict] = []
    for metrics_suffix, iids, ld, dts in eval_jobs:
        iids2, ld2 = _cap_test_samples(iids, ld, args.max_test_samples)
        pl = _evaluate_loader_and_write(
            model=model,
            device=device,
            arch=arch,
            num_classes=num_classes,
            class_names=class_names,
            image_ids=iids2,
            loader=ld2,
            preprocess=preprocess,
            wpath=wpath,
            metrics_basename_suffix=metrics_suffix,
            per_image_csv_arg=args.per_image_csv,
            no_per_image_csv=args.no_per_image_csv,
            max_test_samples=args.max_test_samples,
            dataset_test_split=dts,
            image_size=image_size,
            report_alnum_accuracy=report_alnum or dts == "dienumbers",
        )
        outputs.append(pl)

    if len(outputs) == 1:
        print(json.dumps(outputs[0], indent=2, ensure_ascii=False))
    else:
        keys = [j[3] for j in eval_jobs]
        print(json.dumps(dict(zip(keys, outputs)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
