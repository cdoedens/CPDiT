# Latent Diffusion Transformer for Satellite Image Forecasting

A deep learning framework for forecasting future satellite images using latent diffusion transformers. This architecture combines:

- **Variational Autoencoder (VAE)**: Compresses high-dimensional satellite images into a compact latent space
- **Transformer**: Models temporal dependencies in latent space sequences  
- **Diffusion Model**: Generates probabilistic forecasts of future satellite observations

## Architecture Overview

```
Input Images → VAE Encoder → Latent Codes → Transformer → Diffusion Model → Future Images
```

### Components

1. **VAE** (`src/models/vae.py`): Encodes/decodes between image and latent spaces
2. **Transformer** (`src/models/transformer_backbone.py`): Temporal sequence modeling
3. **Latent Diffusion Transformer** (`src/models/latent_diffusion.py`): Complete forecasting model
4. **Data Loaders** (`src/data/__init__.py`): Multi-dataset support (HIMAWARI, GOES, etc.)

## Installation

1. **Clone and setup**:
```bash
cd /path/to/solar-nowcast/CPDiT
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Install MLflow (optional, for experiment tracking)**:
```bash
pip install mlflow
mlflow server --backend-store-uri sqlite:///mlflow.db
```

## Data Preparation

### Supported Datasets

- **HIMAWARI-8/9**: Japanese geostationary satellite imagery
- Custom satellite data (must be in netCDF format)

### Expected Data Format

Data should be stored as netCDF files with structure:
```
dataset.nc
├── time (unlimited dimension)
├── lat (latitude coordinates)
├── lon (longitude coordinates)
└── TBD (brightness temperature data)
    └── shape: (time, lat, lon)
```

## Configuration

### Training Configuration (`configs/train_config.yaml`)

```yaml
model:
  image_channels: 3
  latent_dim: 256
  num_transformer_layers: 4
  
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 1.0e-4
  
data:
  context_length: 12    # Input sequence length
  forecast_length: 6    # Output sequence length
  train_data_paths: [...]
  val_data_paths: [...]
```

### Inference Configuration (`configs/inference_config.yaml`)

```yaml
model:
  checkpoint_path: outputs/checkpoints/best_model.pt
  
inference:
  num_forecast_steps: 6
  num_samples: 1
  batch_size: 32
```

## Usage

### Training

**Local GPU:**
```bash
bash scripts/train.sh configs/train_config.yaml
```

**Python script:**
```python
from src.training import Trainer, TrainingConfig

config = TrainingConfig.from_yaml('configs/train_config.yaml')
trainer = Trainer(config)
trainer.train()
```

**HPC (PBS/SLURM):**
```bash
qsub scripts/train.pbs
```

### Inference

**Generate forecasts:**
```bash
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --data-paths /path/to/himawari/test/ \
    --forecast-steps 6
```

**Python API:**
```python
from src.inference import load_model_from_checkpoint

model, forecaster = load_model_from_checkpoint('path/to/checkpoint.pt')

# Generate deterministic forecast
forecast = forecaster.forecast_deterministic(context_images, num_steps=6)

# Generate probabilistic samples (with diffusion)
samples = forecaster.forecast(context_images, num_steps=6, num_samples=10)

# Extended autoregressive sequence
full_sequence = forecaster.forecast_sequence(context_images, num_steps=24, autoregressive=True)
```

## Model Components

### Variational Autoencoder

```python
from src.models import VariationalAutoencoder

vae = VariationalAutoencoder(
    image_channels=3,
    latent_dim=256,
    hidden_dim=256
)

# Encode images to latent space
mu, logvar = vae.encode(images)  # (batch, latent_dim)

# Reparameterize
z = vae.reparameterize(mu, logvar)

# Decode back to images
reconstructed = vae.decode(z)
```

### Transformer Backbone

```python
from src.models import TransformerBackbone

transformer = TransformerBackbone(
    latent_dim=256,
    num_layers=4,
    num_heads=8,
    feedforward_dim=1024
)

# Process latent sequences
encoded = transformer(latent_sequence)  # (batch, seq_len, latent_dim)
```

### Complete Model

```python
from src.models import LatentDiffusionTransformer

model = LatentDiffusionTransformer(
    image_channels=3,
    latent_dim=256,
    num_transformer_layers=4,
    num_diffusion_steps=1000
)

# Training forward pass
predictions, encoded_context = model(context_images, target_images)

# Inference sampling
samples = model.sample(context_images, num_forecast_steps=6, num_samples=1)
```

## Experiment Tracking

Monitor training with MLflow:

```bash
# Start MLflow UI
mlflow ui

# View experiments at http://localhost:5000
```

The trainer automatically logs:
- Model hyperparameters
- Training/validation loss
- Model checkpoints

## Performance Monitoring

Key metrics tracked during training:
- **Train Loss**: MSE between predictions and targets
- **Validation Loss**: Generalization performance
- **Diffusion Loss**: KL divergence for VAE component

## Output Structure

```
outputs/
├── checkpoints/
│   ├── checkpoint_epoch_010.pt
│   ├── checkpoint_epoch_020.pt
│   └── best_model.pt
├── logs/
│   ├── training.log
│   └── inference.log
└── predictions/
    ├── contexts.npy        # Input sequences
    ├── predictions.npy     # Generated forecasts
    └── metrics.json        # Evaluation metrics
```

## Advanced Usage

### Custom Datasets

```python
from src.data import SatelliteDataset

custom_dataset = SatelliteDataset(
    data_paths=['data1.nc', 'data2.nc'],
    context_length=12,
    forecast_length=6,
    stride=2,  # Temporal stride
    normalize=True
)
```

### Fine-tuning

```python
from src.inference import load_model_from_checkpoint
from src.training import Trainer, TrainingConfig

# Load pre-trained model
model, _ = load_model_from_checkpoint('pretrained.pt')

# Fine-tune on new data
config = TrainingConfig()
config.learning_rate = 1e-5  # Lower LR for fine-tuning
trainer = Trainer(config)
trainer.model = model  # Use loaded weights
trainer.train()
```

### Ensemble Predictions

```python
# Generate multiple samples
num_samples = 10
samples_list = []
for _ in range(num_samples):
    samples = forecaster.forecast(context_images, num_steps=6, num_samples=1)
    samples_list.append(samples)

# Compute ensemble statistics
ensemble = np.concatenate(samples_list, axis=0)
mean_forecast = ensemble.mean(axis=0)
std_forecast = ensemble.std(axis=0)
```

## Troubleshooting

### CUDA Out of Memory

Reduce `batch_size` in config or use gradient accumulation:

```python
# In training loop
accumulated_loss = 0
for i, (context, target) in enumerate(dataloader):
    loss = compute_loss(model(context), target)
    (loss / accumulation_steps).backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Data Loading Issues

Verify data format:
```python
import xarray as xr
ds = xr.open_dataset('your_data.nc')
print(ds)  # Check dimensions and variables
```

## Citation

If you use this code, please cite:

```bibtex
@software{cpdft2024,
  author = {Your Name},
  title = {Latent Diffusion Transformer for Satellite Image Forecasting},
  year = {2024},
  url = {https://github.com/your-repo/CPDiT}
}
```

## License

MIT License - see LICENSE file for details

## References

- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Contact

For questions or issues, please open an issue on GitHub.
