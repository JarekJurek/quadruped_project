#!/bin/sh
### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J quadruped_rl_ppo_cpg_rl_slopes_curr
### -- ask for number of cores (default: 1) --
#BSUB -n 30
### -- Set the span of the job to 1 node --
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 2:00
# request system-memory
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==XeonGold6226R]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o job_logs/quadruped_rl_ppo_%J.out
#BSUB -e quadruped_rl_%J.err
# -- end of LSF options --

source /zhome/d4/a/214319/miniconda3/bin/activate

conda activate quadruped

# python run_sb3_wandb.py --project-name quadruped_rl --run-name obstacles_fixed_vel_finetune --save-path /work3/s243600 --learning-alg PPO --num-envs 40 --task_env FWD_CUSTOM --observation_space_mode LR_COURSE_OBS_EXTENDED --total_timesteps 5000000 --learning_rate 3e-4 --max_episode_length 20 --learning_rate_adaptive --num_stairs 3 --terrain_difficulty 3 --terrain STAIRS --randomize_cpg_params False --des_h 0.3 --des_g_c 0.15 --des_x_vel 0.8 --load-nn --load_model_path /work3/s243600/logs/intermediate_models/quadruped_rl/fixed_vel_scratch/113025133517

# python run_sb3_wandb.py --project-name quadruped_rl --run-name adaptive_lr_adapt_vel_finetune_2 --save-path /work3/s243600 --learning-alg PPO --num-envs 40 --task_env FWD_CUSTOM --observation_space_mode LR_COURSE_OBS_EXTENDED --total_timesteps 5000000 --learning_rate 3e-4 --max_episode_length 20 --randomize_velocity_command --learning_rate_adaptive --load-nn --load_model_path /work3/s243600/logs/intermediate_models/quadruped_rl/adaptive_lr_adapt_vel_finetune/112825001511

# python run_sb3_wandb.py --project-name quadruped_rl --run-name obstacles_curr_scratch_wo_weights --save-path /work3/s243600 --learning-alg PPO --num-envs 40 --task_env FWD_CUSTOM --observation_space_mode LR_COURSE_OBS_EXTENDED --total_timesteps 3000000 --learning_rate 3e-4 --max_episode_length 20 --learning_rate_adaptive --num_stairs 5 --terrain_difficulty 3 --terrain STAIRS --difficulty_objective height --drift_weight 0.0 --yaw_weight 0.0 --orientation_weight 0.0 --height_weight 0.0 --survival_weight 0.0

python run_sb3_wandb.py --project-name quadruped_rl --run-name simple_slopes_curr_scratch --save-path /work3/s243600 --learning-alg PPO --num-envs 40 --task_env FWD_CUSTOM --observation_space_mode LR_COURSE_OBS_EXTENDED --total_timesteps 3000000 --learning_rate 3e-4 --max_episode_length 20 --learning_rate_adaptive --des_x_vel 0.8 --drift_weight 0.0 --yaw_weight 0.0 --orientation_weight 0.0 --height_weight 0.0 --survival_weight 0.0 --terrain SLOPES --terrain_difficulty 3  #--load-nn --load_model_path /work3/s243600/logs/intermediate_models/quadruped_rl/cpg_vel_fixed/121325194216

# python run_sb3_wandb.py --project-name quadruped_rl --run-name cpg_vel_rand --save-path /work3/s243600 --learning-alg PPO --num-envs 40 --task_env FWD_CUSTOM --observation_space_mode LR_COURSE_OBS_EXTENDED --total_timesteps 3000000 --learning_rate 3e-4 --max_episode_length 20 --learning_rate_adaptive --des_x_vel 0.8  #--disable_energy True --disable_orientation True --disable_yaw True --disable_drift True
