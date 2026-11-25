#!/bin/bash
# Usage: bash run_notebook.sh 00_data_acquisition_and_merging
# Converts notebooks_src/<name>.py (jupytext percent format) to <name>.ipynb and executes it in place.
set -e
cd /Users/home/workspace/School/DataMining/nfip_claims
NAME=$1
nfip_env/bin/jupytext --to ipynb --output "${NAME}.ipynb" "notebooks_src/${NAME}.py" >/dev/null
nfip_env/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=nfip_env "${NAME}.ipynb"
echo "EXECUTED ${NAME}"
