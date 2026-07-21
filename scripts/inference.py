"""Inference script for satellite image forecasting"""

import torch
import argparse
import logging
from pathlib import Path
from typing import Optional
import yaml

from src.models import LatentDiffusionTransformer
from src.inference import Forecaster, load_model_from_checkpoint
from src.data import HIMAWARIDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_inference_config(config_path: str) -> dict:
    """Load inference configuration from YAML"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_inference(
    checkpoint_path: str,
    data_paths: list,
    num_forecast_steps: int = 6,
    output_dir: str = "outputs/predictions",
    device: str = "cuda",
    context_length: int = 12,
    batch_size: int = 32,
):
    """
    Run inference on satellite data.

    Args:
        checkpoint_path: Path to trained model checkpoint
        data_paths: List of data file paths
        num_forecast_steps: Number of steps to forecast
        output_dir: Directory to save predictions
        device: Device to run on
        context_length: Number of context frames
        batch_size: Batch size for inference
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {checkpoint_path}")

    # Load model
    model, forecaster = load_model_from_checkpoint(checkpoint_path, device=device)

    # Load data
    logger.info("Loading inference data...")
    dataset = HIMAWARIDataset(
        data_paths,
        context_length=context_length,
        forecast_length=num_forecast_steps,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Run inference
    logger.info(f"Running inference on {len(dataset)} samples...")
    all_predictions = []
    all_contexts = []

    with torch.no_grad():
        for batch_idx, (context, target) in enumerate(dataloader):
            logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            # Generate predictions
            predictions = forecaster.forecast_deterministic(
                context, num_steps=num_forecast_steps
            )

            all_contexts.append(context.numpy())
            all_predictions.append(predictions.numpy())

    logger.info(f"Completed inference. Saving predictions to {output_dir}")

    # Save results (you can implement custom saving based on your needs)
    import numpy as np

    contexts = np.concatenate(all_contexts, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)

    np.save(f"{output_dir}/contexts.npy", contexts)
    np.save(f"{output_dir}/predictions.npy", predictions)

    logger.info("Inference complete!")
    return contexts, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with latent diffusion transformer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Path to inference config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint (overrides config)",
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        help="Data paths (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=6,
        help="Number of forecast steps",
    )
    args = parser.parse_args()

    # Load config
    config = load_inference_config(args.config)

    # Override with CLI args if provided
    checkpoint_path = args.checkpoint or config["model"]["checkpoint_path"]
    data_paths = args.data_paths or config.get("data_paths", [])
    device = args.device or config["model"]["device"]
    forecast_steps = args.forecast_steps or config["inference"]["num_forecast_steps"]

    if not data_paths:
        raise ValueError("No data paths provided")

    # Run inference
    run_inference(
        checkpoint_path=checkpoint_path,
        data_paths=data_paths,
        num_forecast_steps=forecast_steps,
        output_dir=args.output_dir,
        device=device,
        context_length=config["data"]["context_length"],
        batch_size=config["inference"]["batch_size"],
    )


if __name__ == "__main__":
    main()
