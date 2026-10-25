"""
Tabulate every training run under training.runs_root.

    python scripts/list_runs.py                      # all runs
    python scripts/list_runs.py --variant baseline   # one variant
    python scripts/list_runs.py --stage stage2_dit

Reads only each run's run_info.json, so it is instant and needs no GPU or
pyearthtools. `*` marks the run each stage's `latest` link points at -- the one
`--init-from <stage>` would pick up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_root() -> Path:
    with open(REPO_ROOT / "configs" / "train_base.yaml") as f:
        return Path(yaml.safe_load(f)["training"]["runs_root"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=None, help="Default: runs_root in train_base.yaml")
    p.add_argument("--variant", default=None)
    p.add_argument("--stage", default=None)
    args = p.parse_args()

    root = args.root or default_root()
    rows = []
    for info_path in sorted(root.glob("*/*/*/run_info.json")):
        run_dir = info_path.parent
        if run_dir.is_symlink():          # the `latest` link itself
            continue
        info = json.loads(info_path.read_text())
        variant, stage = run_dir.parent.parent.name, run_dir.parent.name
        if (args.variant and variant != args.variant) or (args.stage and stage != args.stage):
            continue
        latest = run_dir.parent / "latest"
        is_latest = latest.is_symlink() and latest.resolve() == run_dir.resolve()
        best = info.get("best_metric") or {}
        metric = ", ".join(f"{k}={v:.5g}" for k, v in best.items()) or "-"
        if info.get("best_epoch"):
            metric += f" @{info['best_epoch']}"
        scale = info.get("latent_scale")
        rows.append([
            variant, stage, ("* " if is_latest else "  ") + run_dir.name,
            info.get("status", "?"), str(info.get("epochs_completed", 0)), metric,
            "-" if scale is None else f"{scale:.4g}",
            str(len(info.get("channels", []))),
        ])

    if not rows:
        print(f"No runs under {root}")
        return
    header = ["variant", "stage", "run", "status", "epochs", "best", "latent_scale", "channels"]
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    for r in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
    print(f"\n{root}/<variant>/<stage>/<run>/  (* = latest)")


if __name__ == "__main__":
    main()
