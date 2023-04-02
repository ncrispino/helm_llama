#!/bin/bash
#BSUB -N
#BSUB -o logs/llama_test.%J
#BSUB -n 1
#BSUB -m chenguang02
#BSUB -gpu "num=1:gmodel=NVIDIARTXA6000"

module add fix-sing
cd /home/research/cnicholas
# singularity pull helm.sif docker://geraldzeng/helm:latest
singularity exec --bind /home,/scratch/cnicholas/helm_llama --nv /scratch/helm.sif /bin/bash

# From eval.sh
source ~/.bashrc
cd /scratch/cnicholas/helm_llama

python3 helm_process.py

