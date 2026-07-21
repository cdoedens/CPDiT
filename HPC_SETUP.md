# HPC Environment Setup for CPDiT

This guide explains how to set up and use the CPDiT project with the HPC conda environment.

## Quick Start (HPC)

```bash
# Load the HPC environment
source hpc_setup.sh

# Test the setup
python quickstart.py
```

## Detailed Setup Instructions

### 1. Initial HPC Environment Setup

The HPC system has a pre-configured conda environment available via module loading. To use it:

```bash
cd /home/548/cd3022/repos/solar-nowcast/CPDiT

# Load the environment (one-time per session)
source hpc_setup.sh
```

This script:
- Loads the module from `/g/data/dk92/apps/Modules/modulefiles/`
- Activates the `pet/0.4.0` conda environment
- Verifies Python and conda are available
- Displays environment information for verification

### 2. Verify Installation

```bash
# Confirm Python is from the conda environment
python --version
which python

# Test the quickstart
python quickstart.py
```

### 3. For Interactive Development (VS Code / IPython)

Each time you open a new terminal or start a session:

```bash
cd /home/548/cd3022/repos/solar-nowcast/CPDiT
source hpc_setup.sh
```

The environment will remain active for the duration of your terminal session.

### 4. For Batch Jobs (PBS)

In your job script (`.pbs` file), add at the beginning:

```bash
#!/bin/bash
#PBS -N cpdit_training
#PBS -l ncpus=4
#PBS -l mem=32GB
#PBS -l walltime=24:00:00

# Load the HPC environment
module use /g/data/dk92/apps/Modules/modulefiles/
module load pet/0.4.0

# Navigate to project directory
cd /home/548/cd3022/repos/solar-nowcast/CPDiT

# Run your training or analysis script
python scripts/train.py --config configs/train_config.yaml
```

### 5. Installing Additional Packages (if needed)

If the conda environment doesn't have all dependencies:

```bash
# Load the environment first
source hpc_setup.sh

# Install missing packages
pip install package_name
```

**Note:** Prefer modifying the conda environment through the system administrators if additional system dependencies are needed.

### 6. Jupyter Notebooks on HPC

For interactive notebook work:

```bash
# Load the environment
source hpc_setup.sh

# Start Jupyter (adjust settings as needed for HPC)
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

Then connect via SSH tunneling from your local machine:
```bash
ssh -L 8888:localhost:8888 user@hpc-host
# Visit http://localhost:8888 in your browser
```

## Troubleshooting

### Python still not found after module load
```bash
module list  # Check what's loaded
module avail  # See available modules
```

### pip: command not found
Make sure you've sourced `hpc_setup.sh` in your current shell:
```bash
source hpc_setup.sh
pip --version
```

### ImportError for project modules
Ensure you're in the correct directory:
```bash
cd /home/548/cd3022/repos/solar-nowcast/CPDiT
source hpc_setup.sh
python -c "import sys; print(sys.path)"
```

### Permission denied for hpc_setup.sh
Make it executable:
```bash
chmod +x hpc_setup.sh
source hpc_setup.sh  # Use source, not bash
```

## Environment Details

- **Module Path**: `/g/data/dk92/apps/Modules/modulefiles/`
- **Environment Name**: `pet/0.4.0`
- **Project Path**: `/home/548/cd3022/repos/solar-nowcast/CPDiT`
- **Local venv**: Not needed when using HPC environment (but can coexist)

## Related Files

- `hpc_setup.sh` - Automated setup script
- `requirements.txt` - Python package dependencies
- `configs/train_config.yaml` - Training configuration
- `scripts/train.py` - Main training script
