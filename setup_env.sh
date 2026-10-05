#!/bin/bash
set -e

ENV_NAME="brent_wti_project"

echo "Creating Conda environment..."
conda env create -f environment.yml || conda env update -f environment.yml --prune

echo "Installing Jupyter kernel..."
conda run -n $ENV_NAME python -m ipykernel install \
    --user \
    --name $ENV_NAME \
    --display-name "Python (Brent-WTI)"

echo ""
echo "Installation completed successfully!"
echo "You can now open Jupyter or VS Code and select the kernel:"
echo "    Python (Brent-WTI)"