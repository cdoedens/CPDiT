# Getting Started with CPDiT

Welcome to the Latent Diffusion Transformer for satellite image forecasting! This guide will help you get started quickly.

## 📋 Quick Checklist

- [x] Project structure created
- [x] Model architecture implemented
- [ ] Dependencies installed
- [ ] Data prepared
- [ ] Training started

## 🚀 First Steps

### 1. Install Dependencies

```bash
cd /home/548/cd3022/repos/solar-nowcast/CPDiT
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python quickstart.py
```

This will run several examples to verify everything is working:
- Model creation and parameter counting
- Forward pass through the model
- VAE encoding/decoding
- Diffusion sampling
- Training configuration
- Batch operations

### 3. Prepare Your Data

**Format**: netCDF files with structure:
```
dataset.nc
├── time (unlimited dimension)
├── lat (latitude)
├── lon (longitude)
└── TBD (brightness temperature)
    └── shape: (time, lat, lon)
```

**Location**: Update `configs/train_config.yaml`:
```yaml
data:
  train_data_paths:
    - /path/to/himawari/train/
  val_data_paths:
    - /path/to/himawari/val/
  test_data_paths:
    - /path/to/himawari/test/
```

### 4. Start Training

**Local GPU**:
```bash
python -m src.training.train --config configs/train_config.yaml
```

**With MLflow monitoring** (in another terminal):
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db
# Then visit http://localhost:5000
```

**HPC cluster** (PBS):
```bash
qsub scripts/train.pbs
```

### 5. Run Inference

```bash
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --data-paths /path/to/himawari/test/ \
    --forecast-steps 6
```

### 6. Evaluate & Visualize

```bash
# Evaluate
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --data-paths /path/to/himawari/test/

# Visualize results
python scripts/visualize.py \
    --predictions outputs/predictions/predictions.npy \
    --targets outputs/predictions/contexts.npy
```

## 📁 Project Structure

```
CPDiT/
├── src/              # Source code
│   ├── models/       # Model architectures (VAE, Transformer, Diffusion)
│   ├── data/         # Data loaders for satellite imagery
│   ├── training/     # Training pipeline with MLflow
│   └── inference/    # Inference utilities
├── configs/          # YAML configuration files
├── scripts/          # Training/inference scripts and PBS job files
├── quickstart.py     # Quick start examples
└── README.md         # Full documentation
```

## 🏗️ Model Architecture

```
Input Images (B, T, C, H, W)
    ↓
VAE Encoder → Latent Space (B, T, D)
    ↓
Transformer → Temporal Encoding (B, T, D)
    ↓
Diffusion Model
    ↓
Predictions → VAE Decoder
    ↓
Forecast Images (B, T, C, H, W)
```

## 📊 Key Files by Purpose

| Task | Files |
|------|-------|
| **Training** | `src/training/train.py`, `configs/train_config.yaml` |
| **Model** | `src/models/latent_diffusion.py` |
| **Data Loading** | `src/data/__init__.py` |
| **Inference** | `src/inference/__init__.py`, `scripts/inference.py` |
| **Evaluation** | `scripts/evaluate.py` |
| **Visualization** | `scripts/visualize.py` |
| **Configuration** | `configs/train_config.yaml`, `configs/inference_config.yaml` |

## 💻 Python API Examples

### Create Model

```python
from src.models import LatentDiffusionTransformer

model = LatentDiffusionTransformer(
    image_channels=3,
    latent_dim=256,
    num_transformer_layers=4,
    num_diffusion_steps=1000
)
```

### Training

```python
from src.training import Trainer, TrainingConfig

config = TrainingConfig.from_yaml('configs/train_config.yaml')
trainer = Trainer(config)
trainer.train(num_epochs=100)
```

### Inference

```python
from src.inference import load_model_from_checkpoint

model, forecaster = load_model_from_checkpoint('path/to/checkpoint.pt')

# Generate forecast
forecasts = forecaster.forecast_deterministic(context_images, num_steps=6)

# Generate probabilistic samples
samples = forecaster.forecast(context_images, num_steps=6, num_samples=10)
```

## ⚙️ Configuration

Key parameters in `configs/train_config.yaml`:

```yaml
# Model
latent_dim: 256              # Latent space dimension
num_transformer_layers: 4    # Depth of transformer
num_heads: 8                 # Attention heads
num_diffusion_steps: 1000    # Diffusion process steps

# Training
batch_size: 32               # Batch size
learning_rate: 1.0e-4        # Learning rate
num_epochs: 100              # Training epochs

# Data
context_length: 12           # Input sequence length
forecast_length: 6           # Output forecast length
```

## 🔍 Monitoring Training

Use MLflow to track experiments:

```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db

# View at http://localhost:5000
# Monitor: loss curves, hyperparameters, model artifacts
```

## 📈 Next Steps

1. **Customize Data Loading**:
   - Modify `src/data/__init__.py` for your HIMAWARI data format
   - Add preprocessing/augmentation

2. **Tune Hyperparameters**:
   - Experiment with different `latent_dim`, `num_heads`
   - Adjust learning rate schedule in `train.py`

3. **Improve Model**:
   - Try different transformer depths
   - Add attention mechanisms
   - Implement progressive training

4. **Deploy**:
   - Use inference scripts for batch prediction
   - Create REST API wrapper if needed
   - Package with Docker

## ❓ Troubleshooting

**CUDA Out of Memory**:
```python
# Reduce batch size in config
batch_size: 16  # instead of 32
```

**Data not loading**:
```python
# Verify data format
import xarray as xr
ds = xr.open_dataset('your_file.nc')
print(ds)  # Check dimensions and variables
```

**Training too slow**:
- Use `num_workers` in dataloader
- Enable mixed precision: `mixed_precision: true`
- Distribute across multiple GPUs

## 📚 Resources

- **Main Docs**: See [README.md](README.md)
- **Project Structure**: See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## 🤝 Support

- Check [README.md](README.md) for detailed documentation
- See examples in [quickstart.py](quickstart.py)
- Review training logs in `outputs/logs/`

---

**Ready to get started?** Run `python quickstart.py` to verify installation, then prepare your data and start training!
