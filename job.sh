#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J quadruped_rl_sac
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Set the span of the job to 1 node --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request system-memory
#BSUB -R "rusage[mem=8GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o job_logs/quadruped_rl_%J.out
#BSUB -e quadruped_rl_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source /zhome/d4/a/214319/miniconda3/bin/activate

conda activate quadruped

python run_sb3_hpc.py --project-name quadruped_rl  --use-gpu --save-path /work3/s243600 --learning-alg SAC