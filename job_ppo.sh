#!/bin/sh
### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J quadruped_rl_ppo_fixed_vel_scratch
### -- ask for number of cores (default: 1) --
#BSUB -n 30
### -- Set the span of the job to 1 node --
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 5:00
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

# python run_sb3_wandb.py --project-name quadruped_rl --run-name obstacles_curr_finetune --save-path /work3/s243600 --learning-alg PPO --num-envs 40 --task_env FWD_CUSTOM --observation_space_mode LR_COURSE_OBS_EXTENDED --total_timesteps 5000000 --learning_rate 3e-4 --max_episode_length 20 --learning_rate_adaptive --load-nn --num_stairs 5 --terrain_difficulty 3 --terrain STAIRS --load_model_path /work3/s243600/logs/intermediate_models/quadruped_rl/obstacles_curr_scratch/112825001845

# python run_sb3_wandb.py --project-name quadruped_rl --run-name adaptive_lr_adapt_vel_finetune_2 --save-path /work3/s243600 --learning-alg PPO --num-envs 40 --task_env FWD_CUSTOM --observation_space_mode LR_COURSE_OBS_EXTENDED --total_timesteps 5000000 --learning_rate 3e-4 --max_episode_length 20 --randomize_velocity_command --learning_rate_adaptive --load-nn --load_model_path /work3/s243600/logs/intermediate_models/quadruped_rl/adaptive_lr_adapt_vel_finetune/112825001511

# python run_sb3_wandb.py --project-name quadruped_rl --run-name obstacles_curr_scratch --save-path /work3/s243600 --learning-alg PPO --num-envs 40 --task_env FWD_CUSTOM --observation_space_mode LR_COURSE_OBS_EXTENDED --total_timesteps 3000000 --learning_rate 3e-4 --max_episode_length 20 --learning_rate_adaptive --num_stairs 5 --terrain_difficulty 3 --terrain STAIRS

python run_sb3_wandb.py --project-name quadruped_rl --run-name fixed_vel_scratch --save-path /work3/s243600 --learning-alg PPO --num-envs 40 --task_env FWD_CUSTOM --observation_space_mode LR_COURSE_OBS_EXTENDED --total_timesteps 5000000 --learning_rate 3e-4 --max_episode_length 20 --learning_rate_adaptive --randomize_cpg_params False --des_h 0.3 --des_g_c 0.15 --des_x_vel 0.8