"""Quick start script demonstrating the CPDiT model"""

import torch
import numpy as np
from pathlib import Path

from src.models import LatentDiffusionTransformer
from src.training import TrainingConfig, Trainer
from src.inference import Forecaster


def create_dummy_data(
    batch_size=2, seq_len=18, channels=3, height=64, width=64
) -> tuple:
    """Create dummy satellite image data for testing"""
    context = torch.randn(batch_size, 12, channels, height, width)
    forecast = torch.randn(batch_size, 6, channels, height, width)
    return context, forecast


def example_model_creation():
    """Example 1: Create and inspect the model"""
    print("="*60)
    print("Example 1: Model Creation")
    print("="*60)
    
    model = LatentDiffusionTransformer(
        image_channels=3,
        latent_dim=256,
        num_transformer_layers=4,
        num_heads=8,
        feedforward_dim=1024,
        num_diffusion_steps=1000,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model structure:\n{model}\n")


def example_forward_pass():
    """Example 2: Forward pass through the model"""
    print("="*60)
    print("Example 2: Forward Pass")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentDiffusionTransformer().to(device)
    
    # Create dummy data
    context, forecast = create_dummy_data()
    context = context.to(device)
    forecast = forecast.to(device)
    
    print(f"\nInput shapes:")
    print(f"  Context: {context.shape}")
    print(f"  Target forecast: {forecast.shape}")
    
    # Forward pass
    with torch.no_grad():
        predictions, encoded = model(context, forecast)
    
    print(f"\nOutput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Encoded context: {encoded.shape}\n")


def example_encoding_decoding():
    """Example 3: VAE encoding and decoding"""
    print("="*60)
    print("Example 3: VAE Encoding/Decoding")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentDiffusionTransformer().to(device)
    
    # Create dummy images
    images = torch.randn(4, 12, 3, 64, 64).to(device)
    
    print(f"\nOriginal images shape: {images.shape}")
    
    # Encode
    latent_codes = model.encode_images(images)
    print(f"Latent codes shape: {latent_codes.shape}")
    
    # Decode
    reconstructed = model.decode_latents(latent_codes)
    print(f"Reconstructed images shape: {reconstructed.shape}")
    
    # Reconstruction error
    mse = torch.nn.functional.mse_loss(images, reconstructed)
    print(f"Reconstruction MSE: {mse:.6f}\n")


def example_sampling():
    """Example 4: Generating forecasts with diffusion sampling"""
    print("="*60)
    print("Example 4: Diffusion-based Sampling")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentDiffusionTransformer().to(device)
    forecaster = Forecaster(model, device=device)
    
    # Create context
    context = torch.randn(2, 12, 3, 64, 64).to(device)
    
    print(f"\nContext shape: {context.shape}")
    
    # Generate samples
    with torch.no_grad():
        samples = forecaster.forecast(
            context, num_steps=6, num_samples=3
        )
    
    print(f"Generated samples shape: {samples.shape}")
    print(f"Number of forecast steps: {samples.shape[1]}")
    print(f"Number of samples per input: {1}\n")


def example_training_config():
    """Example 5: Training configuration"""
    print("="*60)
    print("Example 5: Training Configuration")
    print("="*60)
    
    config = TrainingConfig(
        image_channels=3,
        latent_dim=256,
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-4,
        context_length=12,
        forecast_length=6,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Context length: {config.context_length}")
    print(f"  Forecast length: {config.forecast_length}")
    print(f"  Device: {config.device}")
    print(f"  Mixed precision: {config.mixed_precision}\n")
    
    # Save config
    config_path = "outputs/example_config.yaml"
    Path(config_path).parent.mkdir(exist_ok=True)
    config.to_yaml(config_path)
    print(f"Config saved to {config_path}")
    
    # Load config back
    loaded_config = TrainingConfig.from_yaml(config_path)
    print(f"Config loaded successfully\n")


def example_batch_operations():
    """Example 6: Batch operations"""
    print("="*60)
    print("Example 6: Batch Operations")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentDiffusionTransformer().to(device)
    
    # Different batch sizes
    for batch_size in [1, 8, 16]:
        context = torch.randn(batch_size, 12, 3, 64, 64).to(device)
        
        with torch.no_grad():
            predictions, _ = model(context)
        
        print(f"Batch size {batch_size:2d}: Input {context.shape} → "
              f"Output {predictions.shape}")
    
    print()


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("CPDiT Quick Start Examples")
    print("="*60 + "\n")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}\n")
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Run examples
    try:
        example_model_creation()
        example_forward_pass()
        example_encoding_decoding()
        example_sampling()
        example_training_config()
        example_batch_operations()
        
        print("="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Prepare your HIMAWARI satellite data")
        print("2. Update configs/train_config.yaml with your data paths")
        print("3. Run: python -m src.training.train --config configs/train_config.yaml")
        print("4. Monitor training with: mlflow ui")
        print()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
