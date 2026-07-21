#!/bin/bash
# HPC Environment Setup Script
# Loads the pet/0.4.0 conda environment for CPDiT development
# 
# Usage:
#   source hpc_setup.sh
#   # or
#   bash hpc_setup.sh

set -e

echo "Setting up HPC environment for CPDiT..."

# Load the module
module use /g/data/dk92/apps/Modules/modulefiles/
module load pet/0.4.0

echo "✓ Loaded pet/0.4.0 module"

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "✗ Error: conda not found after module load"
    exit 1
fi

echo "✓ Conda available: $(conda --version)"

# Show active environment
echo ""
echo "Active conda environment:"
conda info | grep "active environment"
echo ""
echo "Python executable:"
which python
echo ""
echo "Python version:"
python --version
echo ""

# Optional: Install dependencies if needed
# Uncomment the following lines if you want automatic dependency installation
#
# if [ ! -f ".hpc_installed" ]; then
#     echo "Installing dependencies from requirements.txt..."
#     pip install -r requirements.txt
#     touch .hpc_installed
#     echo "✓ Dependencies installed"
# fi

echo "✓ HPC environment ready!"
