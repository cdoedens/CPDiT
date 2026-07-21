"""Evaluation script for model performance"""

import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from src.inference import load_model_from_checkpoint
from src.data import HIMAWARIDataset
from src.utils import compute_metrics
from torch.utils.data import DataLoader


def evaluate(
    checkpoint_path: str,
    data_paths: list,
    output_dir: str = "outputs/evaluation",
    context_length: int = 12,
    forecast_length: int = 6,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Evaluate model on test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_paths: List of test data paths
        output_dir: Directory to save evaluation results
        context_length: Number of context frames
        forecast_length: Number of forecast frames
        batch_size: Batch size
        device: Device to use
    """
    # Create output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, forecaster = load_model_from_checkpoint(checkpoint_path, device=device)
    
    # Load data
    print("Loading test data...")
    dataset = HIMAWARIDataset(
        data_paths,
        context_length=context_length,
        forecast_length=forecast_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Evaluate
    print(f"Evaluating on {len(dataset)} samples...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for context, target in tqdm(dataloader):
            predictions = forecaster.forecast_deterministic(
                context, 
                num_steps=forecast_length
            )
            
            all_predictions.append(predictions.numpy())
            all_targets.append(target.numpy())
    
    # Compute metrics
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_metrics(predictions, targets)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print("\nPer-timestep RMSE:")
    for t, rmse in enumerate(metrics['per_step_rmse']):
        print(f"  Step {t+1}: {rmse:.6f}")
    
    # Save results
    np.save(f"{output_dir}/predictions.npy", predictions)
    np.save(f"{output_dir}/targets.npy", targets)
    
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write("Evaluation Metrics\n")
        f.write("="*50 + "\n")
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
        f.write("\nPer-timestep RMSE:\n")
        for t, rmse in enumerate(metrics['per_step_rmse']):
            f.write(f"  Step {t+1}: {rmse:.6f}\n")
    
    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-paths", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--context-length", type=int, default=12)
    parser.add_argument("--forecast-length", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        data_paths=args.data_paths,
        output_dir=args.output_dir,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
