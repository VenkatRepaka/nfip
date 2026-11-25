#!/bin/bash
# Run several notebooks in order; stop at the first failure.
# Example (whole project): bash run_chain.sh 00_data_acquisition_and_merging 01_EDA_and_cleaning ... 99_summary
cd /Users/home/workspace/School/DataMining/nfip_claims
for NAME in "$@"; do
  bash run_notebook.sh "$NAME" > "logs_${NAME:0:2}.txt" 2>&1 || { echo "FAILED $NAME"; exit 1; }
  echo "OK $NAME $(date '+%H:%M:%S')"
done
