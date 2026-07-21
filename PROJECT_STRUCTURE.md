# Project Structure

```
CPDiT/
├── README.md                          # Main documentation
├── CONTRIBUTING.md                    # Contribution guidelines
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── utils.py                       # Utility functions
│   │
│   ├── models/                        # Model architectures
│   │   ├── __init__.py
│   │   ├── vae.py                     # Variational Autoencoder
│   │   ├── transformer_backbone.py    # Transformer for temporal modeling
│   │   └── latent_diffusion.py        # Main CPDiT model
│   │
│   ├── data/                          # Data loading and preprocessing
│   │   └── __init__.py
│   │       - SatelliteDataset         # Base dataset class
│   │       - HIMAWARIDataset          # HIMAWARI-specific loader
│   │       - MultiDatasetLoader       # Multi-source data handling
│   │
│   ├── training/                      # Training pipeline
│   │   ├── __init__.py
│   │   ├── config.py                  # TrainingConfig dataclass
│   │   └── train.py                   # Training loop and Trainer class
│   │
│   └── inference/                     # Inference pipeline
│       └── __init__.py
│           - Forecaster               # Inference wrapper
│           - BatchPredictor           # Batch prediction handler
│           - load_model_from_checkpoint
│
├── configs/                           # Configuration files
│   ├── train_config.yaml              # Training hyperparameters
│   └── inference_config.yaml          # Inference settings
│
├── scripts/                           # Executable scripts
│   ├── train.sh                       # Training launcher
│   ├── inference.sh                   # Inference launcher
│   ├── inference.py                   # Python inference script
│   ├── evaluate.py                    # Evaluation script
│   ├── visualize.py                   # Visualization utilities
│   ├── train.pbs                      # PBS job script for training
│   └── inference.pbs                  # PBS job script for inference
│
├── notebooks/                         # Jupyter notebooks
│   └── (exploratory notebooks)
│
├── outputs/                           # Generated during runtime
│   ├── checkpoints/                   # Model checkpoints
│   ├── logs/                          # Training logs
│   ├── predictions/                   # Inference results
│   ├── evaluation/                    # Evaluation metrics
│   └── visualizations/                # Generated plots
│
└── quickstart.py                      # Quick start examples
```

## Component Descriptions

### Core Models (`src/models/`)

1. **VariationalAutoencoder** (`vae.py`)
   - Encodes satellite images to latent space
   - Decodes latent codes back to images
   - Computes VAE loss (reconstruction + KL divergence)

2. **TransformerBackbone** (`transformer_backbone.py`)
   - Multi-head self-attention for temporal sequences
   - Positional encoding for temporal information
   - Configurable depth and width

3. **LatentDiffusionTransformer** (`latent_diffusion.py`)
   - Combines VAE + Transformer + Diffusion
   - Handles image encoding/decoding
   - Implements diffusion process for forecasting

### Data Loading (`src/data/`)

- **SatelliteDataset**: Generic satellite data loader
- **HIMAWARIDataset**: HIMAWARI-specific with channel selection
- **MultiDatasetLoader**: Combines multiple datasets

### Training (`src/training/`)

- **TrainingConfig**: Dataclass for all hyperparameters
- **Trainer**: Main training loop with validation
- MLflow integration for experiment tracking

### Inference (`src/inference/`)

- **Forecaster**: Wrapper for generating predictions
- **BatchPredictor**: Batch processing utility
- Model loading from checkpoints

### Scripts (`scripts/`)

- `train.sh` / `train.pbs`: Launch training
- `inference.sh` / `inference.py` / `inference.pbs`: Run inference
- `evaluate.py`: Compute metrics on test set
- `visualize.py`: Generate visualizations

## Data Flow

```
Raw Satellite Images
         ↓
Data Loader (SatelliteDataset)
         ↓
Normalization & Preprocessing
         ↓
VAE Encoder → Latent Space
         ↓
Transformer (Temporal Context)
         ↓
Diffusion Model
         ↓
Predictions → VAE Decoder
         ↓
Forecasted Satellite Images
```

## File Purpose Reference

| File | Purpose |
|------|---------|
| `src/models/vae.py` | Image compression via VAE |
| `src/models/transformer_backbone.py` | Temporal sequence modeling |
| `src/models/latent_diffusion.py` | Main forecasting model |
| `src/data/__init__.py` | Dataset loaders |
| `src/training/train.py` | Training loop |
| `src/training/config.py` | Configuration management |
| `src/inference/__init__.py` | Inference utilities |
| `scripts/train.py` | Training entry point |
| `scripts/inference.py` | Inference entry point |
| `scripts/evaluate.py` | Model evaluation |
| `scripts/visualize.py` | Result visualization |
| `configs/train_config.yaml` | Training hyperparameters |
| `configs/inference_config.yaml` | Inference settings |

## Running the Project

### 1. Quick Start
```bash
python quickstart.py
```

### 2. Training
```bash
python -m src.training.train --config configs/train_config.yaml
```

### 3. Inference
```bash
python scripts/inference.py --checkpoint outputs/checkpoints/best_model.pt --data-paths /path/to/data/
```

### 4. Evaluation
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt --data-paths /path/to/test/data/
```

### 5. Visualization
```bash
python scripts/visualize.py --predictions outputs/predictions/predictions.npy
```

## Key Design Decisions

1. **Modular Architecture**: Each component (VAE, Transformer, Diffusion) can be used independently
2. **Configuration-Based**: All hyperparameters in YAML files
3. **Multi-Dataset Support**: Extensible to different satellite sources
4. **Experiment Tracking**: MLflow integration for reproducibility
5. **HPC Ready**: PBS job scripts for cluster computing
