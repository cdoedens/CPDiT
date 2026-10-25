"""
One directory per training run, so model variants can be told apart.

    <runs_root>/<variant>/<stage>/<run_id>/
        best_model.pt, checkpoint_epoch_*.pt
        config.yaml      the fully resolved config the run trained with
        run_info.json    provenance and progress: inputs, git commit, command,
                         job id, what it was initialised from, best metric so far
        train.log        this run's log (rank 0)
    <runs_root>/<variant>/<stage>/latest -> <run_id>

`variant` is the inputs file's stem (configs/inputs/bands.yaml -> "bands") and
`stage` is `training.stage_name`. A run never reuses another run's directory,
so retraining a variant cannot overwrite the weights of an earlier attempt;
`--resume` is the one exception, and continues in the directory of the
checkpoint it resumes.

`scripts/list_runs.py` tabulates every run_info.json under the root.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_INFO = "run_info.json"
BEST_CHECKPOINT_NAME = "best_model.pt"


def variant_name(config: dict) -> str:
    """The inputs file's stem, or the config's own stem when inputs are inline."""
    run = config.get("run", {})
    return run.get("variant") or Path(run.get("config", "default")).stem


def stage_name(config: dict) -> str:
    return str(config["training"].get("stage_name") or "run")


def _new_run_id() -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job = os.environ.get("PBS_JOBID", "").split(".")[0]
    return f"{stamp}-{job}" if job else stamp


def prepare_run_dir(config: dict, resume: Optional[str] = None) -> Path:
    """
    The directory this run writes to. Rank 0 only; other ranks receive it.

    Precedence: a resumed checkpoint's own directory, then an explicit
    `training.checkpoint_dir` (a fixed location, e.g. the smoke configs), then
    a fresh directory under `training.runs_root`.
    """
    training = config["training"]
    if resume is not None:
        return Path(resume).resolve().parent
    if training.get("checkpoint_dir"):
        return Path(training["checkpoint_dir"])

    stage_root = Path(training["runs_root"]) / variant_name(config) / stage_name(config)
    run_dir = stage_root / _new_run_id()
    suffix = 1
    while run_dir.exists():                 # two launches within one second
        run_dir = stage_root / f"{_new_run_id()}-{suffix}"
        suffix += 1
    return run_dir


def resolve_init_from(arg: str, config: dict) -> Path:
    """
    A checkpoint path, or a stage name meaning "the latest best model of that
    stage for this variant" -- e.g. ``--init-from stage1_vae``.

    The shortcut is what keeps stage chaining variant-safe: stage 2 of
    `bands` can only pick up stage 1 of `bands`, never another variant's VAE
    whose input layer has a different number of channels.
    """
    path = Path(arg)
    if path.exists():
        return path.resolve()

    runs_root = config["training"].get("runs_root")
    if runs_root and os.sep not in arg and not arg.endswith(".pt"):
        latest = Path(runs_root) / variant_name(config) / arg / "latest"
        candidate = latest / BEST_CHECKPOINT_NAME
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"--init-from {arg!r}: no {candidate}. Has stage {arg!r} been "
            f"trained for variant {variant_name(config)!r}?"
        )
    raise FileNotFoundError(f"Checkpoint not found: {arg}")


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def start_run(
    run_dir:   Path,
    config:    dict,
    init_from: Optional[Path] = None,
    resume:    Optional[str] = None,
    link_latest: bool = True,
) -> None:
    """
    Create the run directory and record what is about to train in it. Rank 0.

    A resumed run keeps its original record and appends to `resumes`; the
    config it was started with is kept as config.yaml.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    launch = {
        "time":    now,
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "job_id":  os.environ.get("PBS_JOBID"),
        "host":    socket.gethostname(),
        "git":     _git_state(),
    }

    info_path = run_dir / RUN_INFO
    if resume is not None and info_path.exists():
        info = json.loads(info_path.read_text())
        info.setdefault("resumes", []).append({**launch, "from": str(resume)})
        info["status"] = "running"
    else:
        run = config.get("run", {})
        info = {
            "run_id":    run_dir.name,
            "variant":   variant_name(config),
            "stage":     stage_name(config),
            "channels":  _channels(config),
            "inputs":    run.get("inputs"),
            "config":    run.get("config"),
            "init_from": str(init_from) if init_from else None,
            "status":    "running",
            **launch,
        }
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        if link_latest:
            _point_latest(run_dir)
    _write_json(info_path, info)

    # This run's log beside its weights; the PBS -o log is shared across jobs.
    handler = logging.FileHandler(run_dir / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logger.info("Run directory: %s", run_dir)


def update_run_info(run_dir: Optional[Path], **fields: Any) -> None:
    """Merge fields into run_info.json. A no-op without a run directory."""
    if run_dir is None:
        return
    path = Path(run_dir) / RUN_INFO
    try:
        info = json.loads(path.read_text()) if path.exists() else {}
        info.update(fields)
        _write_json(path, info)
    except OSError as exc:
        # Bookkeeping must never take a training run down.
        logger.warning("Could not update %s: %s", path, exc)


def _channels(config: dict) -> list[str]:
    from .config import channel_names
    return channel_names(config)


def _point_latest(run_dir: Path) -> None:
    link = run_dir.parent / "latest"
    tmp = run_dir.parent / f".latest.{os.getpid()}"
    try:
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(run_dir.name)
        os.replace(tmp, link)               # atomic swap of the link
    except OSError as exc:
        logger.warning("Could not update %s: %s", link, exc)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)
