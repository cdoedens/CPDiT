#!/usr/bin/env python3
"""Run CPDiT inference on a small one-day subset of the dataset.

Example:
  python scripts/inference_single_day.py \
    --checkpoint /scratch/er8/cd3022/CPDiT/checkpoints/checkpoint_epoch_005.pt \
    --split val \
    --date 2021-01-05 \
    --output-dir outputs/inference_single_day
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import HIMAWARIDataset
from src.inference import load_model_from_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(split: str, config: dict) -> HIMAWARIDataset:
    data_cfg = config["data"]
    return HIMAWARIDataset(
        index_path=data_cfg["valid_timestamps"][split],
        heliosat_data_dir=data_cfg["heliosat"][split],
        barra_data_dir=data_cfg["barra"][split],
        heliosat_stats_path=data_cfg["normalisation_stats"]["heliosat"],
        barra_stats_path=data_cfg["normalisation_stats"]["barra"],
        regrid_weights_path=data_cfg["regrid_weights"]["barra_to_heliosat"],
        heliosat_vars=data_cfg["heliosat_vars"],
        barra_vars=data_cfg["barra_vars"],
        context_length=data_cfg["context_length"],
        forecast_length=data_cfg["forecast_length"],
        satellite_timestep=f"{data_cfg['satellite_timestep_min']}min",
    )


def select_day(dataset: HIMAWARIDataset, date: str | None, max_samples: int | None) -> pd.DatetimeIndex:
    times = dataset.start_times
    if times.empty:
        raise ValueError("Dataset index is empty")

    if date is None:
        selected_day = times[0].normalize()
    else:
        selected_day = pd.to_datetime(date).normalize()

    day_mask = times.normalize() == selected_day
    selected = times[day_mask]
    if len(selected) == 0:
        available_days = sorted({t.normalize().strftime("%Y-%m-%d") for t in times})
        raise ValueError(
            f"No valid start times found for date {selected_day.strftime('%Y-%m-%d')}\n"
            f"Available days: {available_days[:10]}"
        )

    if max_samples is not None:
        selected = selected[:max_samples]

    dataset.start_times = selected
    return selected


def run_inference(
    checkpoint_path: Path,
    config_path: Path,
    split: str,
    date: str | None,
    max_samples: int | None,
    forecast_steps: int,
    batch_size: int,
    output_dir: Path,
    device: str,
):
    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)

    logger.info("Building dataset for split=%s", split)
    dataset = build_dataset(split, config)
    selected_times = select_day(dataset, date, max_samples)

    logger.info(
        "Selected %d sample(s) from %s for split=%s",
        len(dataset),
        selected_times[0].strftime("%Y-%m-%d"),
        split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logger.info("Loading model checkpoint from %s", checkpoint_path)
    _, forecaster = load_model_from_checkpoint(str(checkpoint_path), device=device)

    outputs = []
    contexts = []
    targets = []

    with torch.no_grad():
        for batch_idx, (context, target) in enumerate(dataloader, start=1):
            logger.info("Running batch %d/%d", batch_idx, len(dataloader))
            predictions = forecaster.forecast_deterministic(context, num_steps=forecast_steps)
            outputs.append(predictions.cpu().numpy())
            contexts.append(context.cpu().numpy())
            targets.append(target.cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    contexts = np.concatenate(contexts, axis=0)
    targets = np.concatenate(targets, axis=0)

    # ── NaN diagnostics ──────────────────────────────────────────────────
    logger.info("contexts   NaN fraction : %.4f", np.isnan(contexts).mean())
    logger.info("targets    NaN fraction : %.4f", np.isnan(targets).mean())
    logger.info("predictions NaN fraction: %.4f", np.isnan(outputs).mean())
    logger.info("contexts   range: [%.4f, %.4f]", np.nanmin(contexts),  np.nanmax(contexts))
    logger.info("targets    range: [%.4f, %.4f]", np.nanmin(targets),   np.nanmax(targets))
    logger.info("predictions range:[%.4f, %.4f]", np.nanmin(outputs),   np.nanmax(outputs))
    # ─────────────────────────────────────────────────────────────────────


    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "selected_start_times.npy", selected_times.astype(str).to_numpy())
    np.save(output_dir / "contexts.npy", contexts)
    np.save(output_dir / "targets.npy", targets)
    np.save(output_dir / "predictions.npy", outputs)

    logger.info("Saved inference results to %s", output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPDiT inference on one day of data")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/scratch/er8/cd3022/CPDiT/checkpoints/checkpoint_epoch_005.pt"),
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to the training config YAML",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Data split to use for the small inference run",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to run inference on (YYYY-MM-DD). Defaults to the first available day.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12,
        help="Maximum number of samples to run for the selected day",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=6,
        help="Number of forecast frames to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference_single_day"),
        help="Directory to save inference outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        date=args.date,
        max_samples=args.max_samples,
        forecast_steps=args.forecast_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
