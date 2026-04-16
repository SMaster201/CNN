"""對 runs_classic 內所有已存在的 best.pt 依序執行 eval_classifier（不訓練）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
RUNS = ROOT / "runs_classic"
EVAL = ROOT / "src" / "eval_classifier.py"


def main() -> None:
    extra = sys.argv[1:]
    if not RUNS.is_dir():
        print(f"找不到目錄: {RUNS}", file=sys.stderr)
        raise SystemExit(1)

    weights = sorted(RUNS.glob("*/best.pt"))
    if not weights:
        print(f"{RUNS} 下沒有任何 */best.pt，請先訓練模型。", file=sys.stderr)
        raise SystemExit(1)

    for w in weights:
        arch = w.parent.name
        # eval_classifier 預設 --dataset nmos；可於命令列附加參數覆寫（例如 --dataset cifar10）
        cmd = [PY, str(EVAL), "--weights", str(w)] + extra
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)

    print(f"完成：已評估 {len(weights)} 個模型（{', '.join(p.parent.name for p in weights)}）。")


if __name__ == "__main__":
    main()
