"""Visualization utilities for satellite image predictions"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Optional, Tuple


def plot_sequence(
    context: np.ndarray,
    forecast: np.ndarray = None,
    channel: int = 0,
    title: str = "Satellite Image Sequence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot sequence of satellite images.
    
    Args:
        context: (seq_len, channels, height, width)
        forecast: Optional forecast sequence
        channel: Which channel to plot
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    # Handle batch dimension if present
    if context.ndim == 5:
        context = context[0]  # Take first sample
    if forecast is not None and forecast.ndim == 5:
        forecast = forecast[0]
    
    n_context = context.shape[0]
    n_forecast = forecast.shape[0] if forecast is not None else 0
    n_total = n_context + n_forecast
    
    # Create subplots
    fig, axes = plt.subplots(
        2, max(n_context // 2, 3),
        figsize=(15, 6),
        tight_layout=True
    )
    axes = axes.flatten()
    
    # Plot context
    for i in range(n_context):
        img = context[i, channel]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Context {i+1}')
        axes[i].axis('off')
    
    # Plot forecast
    if forecast is not None:
        for i in range(n_forecast):
            idx = n_context + i
            if idx < len(axes):
                img = forecast[i, channel]
                axes[idx].imshow(img, cmap='gray', alpha=0.7)
                axes[idx].set_title(f'Forecast {i+1}', color='red')
                axes[idx].axis('off')
    
    # Hide unused axes
    for i in range(n_total, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_comparison(
    predictions: np.ndarray,
    targets: np.ndarray,
    indices: list = None,
    channel: int = 0,
    title: str = "Prediction vs Target",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot side-by-side comparison of predictions vs targets.
    
    Args:
        predictions: (batch, seq_len, channels, height, width)
        targets: Same shape as predictions
        indices: List of sample indices to plot (default first 3)
        channel: Channel to visualize
        title: Plot title
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    if indices is None:
        indices = [0, 1, 2]
    
    n_samples = len(indices)
    n_steps = predictions.shape[1]
    
    fig, axes = plt.subplots(
        n_samples * 2, n_steps,
        figsize=(3 * n_steps, 6 * n_samples),
        tight_layout=True
    )
    
    for row, idx in enumerate(indices):
        # Plot predictions
        for col in range(n_steps):
            ax = axes[row * 2, col]
            img = predictions[idx, col, channel]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            if col == 0:
                ax.set_ylabel('Prediction', fontsize=10)
            ax.set_title(f'Step {col+1}')
            ax.axis('off')
        
        # Plot targets
        for col in range(n_steps):
            ax = axes[row * 2 + 1, col]
            img = targets[idx, col, channel]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            if col == 0:
                ax.set_ylabel('Target', fontsize=10)
            ax.axis('off')
    
    fig.suptitle(title, fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
    
    return fig


def plot_error_map(
    predictions: np.ndarray,
    targets: np.ndarray,
    sample_idx: int = 0,
    step_idx: int = 0,
    channel: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot prediction error map.
    
    Args:
        predictions: (batch, seq_len, channels, height, width)
        targets: Same shape
        sample_idx: Sample index to plot
        step_idx: Time step index
        channel: Channel to visualize
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    pred = predictions[sample_idx, step_idx, channel]
    targ = targets[sample_idx, step_idx, channel]
    error = np.abs(pred - targ)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Prediction
    im1 = axes[0].imshow(pred, cmap='gray')
    axes[0].set_title('Prediction')
    plt.colorbar(im1, ax=axes[0])
    axes[0].axis('off')
    
    # Target
    im2 = axes[1].imshow(targ, cmap='gray')
    axes[1].set_title('Target')
    plt.colorbar(im2, ax=axes[1])
    axes[1].axis('off')
    
    # Error
    im3 = axes[2].imshow(error, cmap='hot')
    axes[2].set_title('Absolute Error')
    plt.colorbar(im3, ax=axes[2])
    axes[2].axis('off')
    
    fig.suptitle(f'Sample {sample_idx}, Step {step_idx}', fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error map saved to {save_path}")
    
    return fig


def plot_metric_evolution(
    metrics_per_step: list,
    metric_name: str = "RMSE",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how metrics evolve over forecast steps.
    
    Args:
        metrics_per_step: List of metric values per step
        metric_name: Name of metric being plotted
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = range(1, len(metrics_per_step) + 1)
    ax.plot(steps, metrics_per_step, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Forecast Step', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Evolution Over Forecast Horizon')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metric plot saved to {save_path}")
    
    return fig


def create_animation(
    frames: np.ndarray,
    channel: int = 0,
    title: str = "Satellite Image Forecast",
    save_path: Optional[str] = None,
    fps: int = 2,
) -> None:
    """
    Create animation from sequence of images.
    
    Args:
        frames: (seq_len, channels, height, width) or (seq_len, height, width)
        channel: Channel to visualize (if multi-channel)
        title: Animation title
        save_path: Path to save animation as MP4
        fps: Frames per second
    """
    if frames.ndim == 4:
        frames = frames[:, channel]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(frames[0], cmap='gray', vmin=frames.min(), vmax=frames.max())
    ax.set_title(title)
    ax.axis('off')
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'{title} - Step {frame_idx + 1}')
        return [im]
    
    anim = FuncAnimation(
        fig, update, frames=len(frames),
        blit=True, repeat=True, interval=1000 / fps
    )
    
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {save_path}")
    
    return anim


def visualize_predictions(
    predictions_path: str = "outputs/predictions/predictions.npy",
    targets_path: str = "outputs/predictions/contexts.npy",
    output_dir: str = "outputs/visualizations",
):
    """
    Generate comprehensive visualizations from prediction results.
    
    Args:
        predictions_path: Path to predictions .npy file
        targets_path: Path to targets/contexts .npy file
        output_dir: Directory to save visualizations
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading predictions...")
    predictions = np.load(predictions_path)
    targets = np.load(targets_path)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Plot comparisons for first few samples
    for i in range(min(3, len(predictions))):
        fig = plot_sequence(
            targets[i],
            predictions[i],
            title=f"Sample {i+1}: Context and Forecast",
            save_path=f"{output_dir}/sequence_{i}.png"
        )
        plt.close(fig)
    
    # Plot comparison
    fig = plot_comparison(
        predictions, targets,
        indices=[0, 1, 2],
        title="Prediction vs Target Comparison",
        save_path=f"{output_dir}/comparison.png"
    )
    plt.close(fig)
    
    # Plot error maps
    for i in range(min(2, len(predictions))):
        for t in range(predictions.shape[1]):
            fig = plot_error_map(
                predictions, targets,
                sample_idx=i, step_idx=t,
                save_path=f"{output_dir}/error_map_s{i}_t{t}.png"
            )
            plt.close(fig)
    
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize predictions")
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/predictions/predictions.npy",
        help="Path to predictions file"
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="outputs/predictions/contexts.npy",
        help="Path to targets file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Output directory"
    )
    
    args = parser.parse_args()
    visualize_predictions(args.predictions, args.targets, args.output_dir)
