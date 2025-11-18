#!/bin/sh
### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J quadruped_rl_ppo_30_cores_100_envs
### -- ask for number of cores (default: 1) --
#BSUB -n 30
### -- Set the span of the job to 1 node --
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 4:00
# request system-memory
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==XeonGold6226R]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o job_logs/quadruped_rl_ppo_%J.out
#BSUB -e quadruped_rl_%J.err
# -- end of LSF options --

source /zhome/d4/a/214319/miniconda3/bin/activate

conda activate quadruped

python run_sb3_wandb.py --project-name quadruped_rl --save-path /work3/s243600 --learning-alg PPO --num-envs 4096