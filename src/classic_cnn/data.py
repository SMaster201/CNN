"""CIFAR-10 / ImageFolder / nmos 載入與影像增強（依架構區分 32×32 與 224×224）。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.classic_cnn.models import arch_uses_imagenet_224


def transforms_nmos(arch: str, train: bool) -> transforms.Compose:
    """nmos 工業影像：ImageNet 正規化（eval 不加 flip，避免 Lambda 無法於 Windows worker pickle）。"""
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if arch_uses_imagenet_224(arch):
        ops: list = [transforms.Resize((224, 224))]
        if train:
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend([transforms.ToTensor(), norm])
        return transforms.Compose(ops)
    ops = [transforms.Resize((32, 32))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), norm])
    return transforms.Compose(ops)


def transforms_for(arch: str, train: bool) -> transforms.Compose:
    cifar_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if arch_uses_imagenet_224(arch):
        ops: list = [transforms.Resize((224, 224))]
        if train:
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend([transforms.ToTensor(), cifar_norm])
        return transforms.Compose(ops)
    ops: list = []
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), cifar_norm])
    return transforms.Compose(ops)


def get_cifar10_loaders(
    root: Path,
    arch: str,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    t_train = transforms_for(arch, True)
    t_eval = transforms_for(arch, False)
    ds_tr = datasets.CIFAR10(root=str(root), train=True, download=True, transform=t_train)
    ds_ev = datasets.CIFAR10(root=str(root), train=True, download=True, transform=t_eval)
    n = len(ds_tr)
    n_val = max(1, int(n * val_ratio))
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_set = Subset(ds_tr, train_idx)
    val_set = Subset(ds_ev, val_idx)
    test_set = datasets.CIFAR10(root=str(root), train=False, download=True, transform=t_eval)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_set, shuffle=True, **kw),
        DataLoader(val_set, shuffle=False, **kw),
        DataLoader(test_set, shuffle=False, **kw),
    )


def get_imagefolder_loaders(
    root: Path,
    arch: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    t_tr = transforms_for(arch, True)
    t_ev = transforms_for(arch, False)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"需要 {root}/train 與 {root}/val，各類別一子資料夾。")
    train_set = datasets.ImageFolder(str(train_dir), transform=t_tr)
    val_set = datasets.ImageFolder(str(val_dir), transform=t_ev)
    num_classes = len(train_set.classes)
    if num_classes != len(val_set.classes):
        raise ValueError("train 與 val 類別數不一致")
    test_dir = root / "test"
    test_set = datasets.ImageFolder(str(test_dir), transform=t_ev) if test_dir.is_dir() else val_set
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_set, shuffle=True, **kw),
        DataLoader(val_set, shuffle=False, **kw),
        DataLoader(test_set, shuffle=False, **kw),
        num_classes,
    )


def get_test_loader_cifar10(root: Path, arch: str, batch_size: int, num_workers: int) -> DataLoader:
    _, loader = get_cifar10_test_ids_and_loader(root, arch, batch_size, num_workers)
    return loader


def get_cifar10_test_ids_and_loader(
    root: Path, arch: str, batch_size: int, num_workers: int
) -> tuple[list[str], DataLoader]:
    """回傳與 DataLoader 順序一致的測試集識別字串（供逐張輸出）。"""
    t_eval = transforms_for(arch, False)
    test_set = datasets.CIFAR10(root=str(root), train=False, download=True, transform=t_eval)
    ids = [f"cifar10_test_{i:05d}" for i in range(len(test_set))]
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    loader = DataLoader(test_set, shuffle=False, **kw)
    return ids, loader


def get_test_loader_imagefolder(root: Path, arch: str, batch_size: int, num_workers: int) -> DataLoader:
    _, loader, _ = get_imagefolder_test_paths_and_loader(root, arch, batch_size, num_workers)
    return loader


def get_imagefolder_test_paths_and_loader(
    root: Path, arch: str, batch_size: int, num_workers: int
) -> tuple[list[str], DataLoader, list[str]]:
    t_ev = transforms_for(arch, False)
    test_dir = root / "test"
    val_dir = root / "val"
    folder = test_dir if test_dir.is_dir() else val_dir
    if not folder.is_dir():
        raise FileNotFoundError(f"需要 {root}/test 或 {root}/val")
    test_set = datasets.ImageFolder(str(folder), transform=t_ev)
    paths = [s[0] for s in test_set.samples]
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    loader = DataLoader(test_set, shuffle=False, **kw)
    return paths, loader, test_set.classes


_NMOS_STEM_RE = re.compile(r"^(\d{6})")
_DIENUM_STEM_RE = re.compile(r"^([A-Za-z]\d)")


def collect_nmos_samples(nmos_root: Path) -> tuple[list[Path], list[int], list[str]]:
    """
    掃描扁平資料夾 nmos_root；檔名（不含副檔名）開頭須為連續 6 位數字，以此為 GT 類別碼。
    回傳 (paths, label_indices, class_names)，class_names[i] 為第 i 類的 6 位字串。
    """
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".webp"}
    code_to_paths: dict[str, list[Path]] = {}
    for p in sorted(nmos_root.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        m = _NMOS_STEM_RE.match(p.stem)
        if not m:
            continue
        code = m.group(1)
        code_to_paths.setdefault(code, []).append(p)
    if not code_to_paths:
        raise ValueError(
            f"{nmos_root} 內找不到符合規則的影像：檔名須以連續 6 位數字開頭（例如 101001 (1).bmp）"
        )
    codes = sorted(code_to_paths.keys())
    str_to_idx = {c: i for i, c in enumerate(codes)}
    paths: list[Path] = []
    labels: list[int] = []
    for code in codes:
        for path in sorted(code_to_paths[code]):
            paths.append(path)
            labels.append(str_to_idx[code])
    return paths, labels, codes


def collect_dienumbers_samples(dienumbers_root: Path) -> tuple[list[Path], list[int], list[str]]:
    """
    掃描扁平資料夾 dienumbers_root；檔名開頭須為 1 英文 + 1 數字（例如 A3_xxx）。
    回傳 (paths, label_indices, class_names)，class_names[i] 為如 A3 的字串。
    """
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".webp"}
    code_to_paths: dict[str, list[Path]] = {}
    for p in sorted(dienumbers_root.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        m = _DIENUM_STEM_RE.match(p.stem)
        if not m:
            continue
        code = m.group(1).upper()
        code_to_paths.setdefault(code, []).append(p)
    if not code_to_paths:
        raise ValueError(
            f"{dienumbers_root} 內找不到符合規則的影像：檔名須以 1 英文 + 1 數字開頭（例如 A3_xxx.bmp）"
        )
    codes = sorted(code_to_paths.keys())
    str_to_idx = {c: i for i, c in enumerate(codes)}
    paths: list[Path] = []
    labels: list[int] = []
    for code in codes:
        for path in sorted(code_to_paths[code]):
            paths.append(path)
            labels.append(str_to_idx[code])
    return paths, labels, codes


def collect_dataset_samples(dataset_root: Path) -> tuple[list[Path], list[int], list[str]]:
    """讀取 dataset/nmos + dataset/dienumbers，合併成同一分類空間。"""
    nmos_root = dataset_root / "nmos"
    die_root = dataset_root / "dienumbers"
    if not nmos_root.is_dir() or not die_root.is_dir():
        raise FileNotFoundError(f"需要 {dataset_root}/nmos 與 {dataset_root}/dienumbers")

    n_paths, _, n_codes = collect_nmos_samples(nmos_root)
    d_paths, _, d_codes = collect_dienumbers_samples(die_root)
    class_names = sorted(set(n_codes + d_codes))
    str_to_idx = {c: i for i, c in enumerate(class_names)}

    def _code_from_path(p: Path) -> str | None:
        m6 = _NMOS_STEM_RE.match(p.stem)
        if m6:
            return m6.group(1)
        m2 = _DIENUM_STEM_RE.match(p.stem)
        if m2:
            return m2.group(1).upper()
        return None

    paths = sorted(n_paths + d_paths)
    labels: list[int] = []
    kept_paths: list[Path] = []
    for p in paths:
        code = _code_from_path(p)
        if code is None:
            continue
        kept_paths.append(p)
        labels.append(str_to_idx[code])
    if not kept_paths:
        raise ValueError(f"{dataset_root} 內沒有可用樣本")
    return kept_paths, labels, class_names


class NmosDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], transform) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.transform(img)
        return x, self.labels[i]


def _nmos_counts_per_split(n: int, v: float, t: float) -> tuple[int, int, int]:
    """
    單一類別 n 張時，回傳 (n_train, n_val, n_test)。
    小 n 時 round(n*比例) 常為 0，會導致全域測試集為空；此處保證 n>=2 至少 1 筆測試、n>=3 盡量保留驗證。
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1
    # n >= 3：至少各留 1 筆訓練，測試/驗證用比例取整並設下限
    n_test = max(1, int(round(n * t)))
    n_test = min(n_test, n - 2)
    n_val = max(1, int(round(n * v)))
    n_val = min(n_val, n - n_test - 1)
    n_train = n - n_val - n_test
    while n_train < 1 and (n_val > 1 or n_test > 1):
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
        n_train = n - n_val - n_test
    return n_train, n_val, n_test


def _nmos_train_val_test_indices(
    labels: list[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """依類別分層切分（僅 NumPy）；索引對應 paths/labels 同一順序。"""
    y = np.asarray(labels, dtype=np.int64)
    v = float(val_ratio)
    t = float(test_ratio)
    if v + t >= 1.0:
        raise ValueError("nmos-val-ratio + nmos-test-ratio 須小於 1")
    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = int(len(idx))
        n_train, n_val, n_test = _nmos_counts_per_split(n, v, t)
        i1 = n_train
        i2 = i1 + n_val
        train_ids.extend(idx[:i1].tolist())
        val_ids.extend(idx[i1:i2].tolist())
        test_ids.extend(idx[i2:].tolist())

    # 若每類都只有 1 張，測試仍為空；自訓練集隨機挪出與整體比例相近的筆數
    N = len(train_ids) + len(val_ids) + len(test_ids)
    if len(test_ids) == 0 and N >= 2 and t > 0:
        k = int(round(N * t))
        want_test = min(N - 1, max(1, k if k > 0 else 1))
        rng2 = np.random.default_rng(seed + 1000)
        tri = np.asarray(train_ids, dtype=np.int64)
        while len(test_ids) < want_test and len(tri) > 1:
            j = int(rng2.integers(0, len(tri)))
            test_ids.append(int(tri[j]))
            tri = np.delete(tri, j)
        train_ids = tri.tolist()

    return np.asarray(train_ids, dtype=np.int64), np.asarray(val_ids, dtype=np.int64), np.asarray(test_ids, dtype=np.int64)


def _train_val_indices(labels: list[int], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """依類別分層切分 train/val（無 test）。"""
    y = np.asarray(labels, dtype=np.int64)
    v = float(val_ratio)
    if not (0.0 < v < 1.0):
        raise ValueError("val_ratio 須介於 (0,1)")
    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = int(len(idx))
        n_val = max(1, int(round(n * v)))
        n_val = min(n_val, n - 1) if n > 1 else 1
        val_ids.extend(idx[:n_val].tolist())
        train_ids.extend(idx[n_val:].tolist())
    return np.asarray(train_ids, dtype=np.int64), np.asarray(val_ids, dtype=np.int64)


def get_nmos_loaders(
    nmos_root: Path,
    arch: str,
    batch_size: int,
    val_ratio: float,
    test_ratio: float,
    num_workers: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int, list[str]]:
    paths, labels, class_names = collect_nmos_samples(nmos_root)
    train_idx, val_idx, test_idx = _nmos_train_val_test_indices(labels, val_ratio, test_ratio, seed)
    tr_paths = [paths[i] for i in train_idx]
    tr_labels = [labels[i] for i in train_idx]
    va_paths = [paths[i] for i in val_idx]
    va_labels = [labels[i] for i in val_idx]
    te_paths = [paths[i] for i in test_idx]
    te_labels = [labels[i] for i in test_idx]
    num_classes = len(class_names)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_ds = NmosDataset(tr_paths, tr_labels, transforms_nmos(arch, True))
    val_ds = NmosDataset(va_paths, va_labels, transforms_nmos(arch, False))
    test_ds = NmosDataset(te_paths, te_labels, transforms_nmos(arch, False))
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        num_classes,
        class_names,
    )


def get_nmos_test_paths_and_loader(
    nmos_root: Path,
    arch: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[list[str], DataLoader]:
    """與 get_nmos_loaders 相同測試切分；回傳絕對路徑字串列表（與 batch 順序一致）。"""
    paths, labels, _ = collect_nmos_samples(nmos_root)
    _, _, test_idx = _nmos_train_val_test_indices(labels, val_ratio, test_ratio, seed)
    te_paths = [paths[i] for i in test_idx]
    te_labels = [labels[i] for i in test_idx]
    test_ds = NmosDataset(te_paths, te_labels, transforms_nmos(arch, False))
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    loader = DataLoader(test_ds, shuffle=False, **kw)
    return [str(p.resolve()) for p in te_paths], loader


def get_nmos_test_loader(
    nmos_root: Path,
    arch: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> DataLoader:
    """與 get_nmos_loaders 相同切分，僅回傳測試集 DataLoader。"""
    _, loader = get_nmos_test_paths_and_loader(
        nmos_root, arch, batch_size, num_workers, val_ratio, test_ratio, seed
    )
    return loader


def get_dataset_loaders(
    dataset_root: Path,
    arch: str,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    paths, labels, class_names = collect_dataset_samples(dataset_root)
    train_idx, val_idx = _train_val_indices(labels, val_ratio, seed)
    tr_paths = [paths[i] for i in train_idx]
    tr_labels = [labels[i] for i in train_idx]
    va_paths = [paths[i] for i in val_idx]
    va_labels = [labels[i] for i in val_idx]
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_ds = NmosDataset(tr_paths, tr_labels, transforms_nmos(arch, True))
    val_ds = NmosDataset(va_paths, va_labels, transforms_nmos(arch, False))
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw), len(class_names), class_names


def get_dataset_eval_paths_and_loader(
    dataset_root: Path,
    arch: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list[str], DataLoader]:
    """
    從 dataset/test/nmos 與 dataset/test/dienumbers 讀取完整測試集：
    - nmos: 檔名前 6 碼
    - dienumbers: 檔名前 2 碼（1 英文 + 1 數字）
    """
    del val_ratio, seed
    test_root = dataset_root / "test"
    nmos_root = test_root / "nmos"
    die_root = test_root / "dienumbers"
    if not nmos_root.is_dir() or not die_root.is_dir():
        raise FileNotFoundError(f"需要 {dataset_root}/test/nmos 與 {dataset_root}/test/dienumbers")

    _, _, class_names = collect_dataset_samples(dataset_root)
    str_to_idx = {c: i for i, c in enumerate(class_names)}

    nmos_paths, _, _ = collect_nmos_samples(nmos_root)
    die_paths, _, _ = collect_dienumbers_samples(die_root)

    te_paths = sorted(nmos_paths + die_paths)
    te_labels: list[int] = []
    for p in te_paths:
        m6 = _NMOS_STEM_RE.match(p.stem)
        if m6:
            te_labels.append(str_to_idx[m6.group(1)])
            continue
        m2 = _DIENUM_STEM_RE.match(p.stem)
        if m2:
            te_labels.append(str_to_idx[m2.group(1).upper()])
            continue
        raise ValueError(f"測試檔名不符合規則: {p.name}")

    ds = NmosDataset(te_paths, te_labels, transforms_nmos(arch, False))
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    loader = DataLoader(ds, shuffle=False, **kw)
    return [str(p.resolve()) for p in te_paths], loader
