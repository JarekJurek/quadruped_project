# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import argparse
import os
import sys
import time
from sys import platform

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC

# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import (
    make_vec_env,  # fix for newer versions of stable-baselines3
)

# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')
# stable-baselines3
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import VecNormalize

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.file_utils import get_latest_model, load_all_results
from utils.utils import plot_results


def load_sb3(args):

    # LEARNING_ALG = "PPO" #"SAC"
    # interm_dir = "./logs/intermediate_models/"
    interm_dir = f"{args.save_path}/logs/intermediate_models/{args.project_name}"
    # path to saved models, i.e. interm_dir + '102824115106'
    # log_dir = interm_dir + args.model_id
    log_dir = args.full_path

    # initialize env configs (render at test time)
    # check ideal conditions, as well as robustness to UNSEEN noise during training
    env_config = {}
    env_config['render'] = True
    env_config['record_video'] = args.record_video
    env_config['add_noise'] = args.add_noise 
    env_config["motor_control_mode"]=args.motor_control_mode
    env_config["task_env"]=args.task_env
    env_config["observation_space_mode"]=args.observation_space_mode

    # get latest model and normalization stats, and plot 
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    model_name = get_latest_model(log_dir)
    monitor_results = load_results(log_dir)
    print(monitor_results)
    plot_results([log_dir] , 10e10, 'timesteps', args.learning_alg + ' ')
    plt.show() 

    # reconstruct env 
    env = lambda: QuadrupedGymEnv(**env_config)
    env = make_vec_env(env, n_envs=1)
    env = VecNormalize.load(stats_path, env)
    env.training = False    # do not update stats at test time
    env.norm_reward = False # reward normalization is not needed at test time

    # load model
    if args.learning_alg == "PPO":
        model = PPO.load(model_name, env)
    elif args.learning_alg == "SAC":
        model = SAC.load(model_name, env)
    print("\nLoaded model", model_name, "\n")

    obs = env.reset()
    episode_reward = 0

    # [TODO] initialize arrays to save data from simulation 

    for i in range(args.sim_time):
        action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test if the outputs make sense)
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards
        
        if dones:
            print('episode_reward', episode_reward)
            print('Final base position', info[0]['base_pos'])
            episode_reward = 0

        # [TODO] save data from current robot states for plots 
        # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
        
    # [TODO] make plots


def parse_arguments():
    parser = argparse.ArgumentParser(description="Quadruped RL training with Stable Baselines 3")
    parser.add_argument("--project-name", type=str, default="quadruped_rl", help="Name of the project")

    parser.add_argument("--record_video", type=bool, default=False, help="Record video flag")
    parser.add_argument("--add_noise", type=bool, default=False, help="Add noise flag")

    parser.add_argument("--sim_time", type=int, default=5000, help="Duration of the simulation in miliseconds (has to be integer)")
    
    parser.add_argument("--learning-alg", type=str, default="PPO", choices=["PPO", "SAC"], help="Learning algorithm to use (default: PPO)")
    parser.add_argument("--motor_control_mode", type=str, default="CPG", choices=["CPG", "PD","TORQUE", "CARTESIAN_PD"], help="Motor control mode")
    parser.add_argument("--observation_space_mode", type=str, default="LR_COURSE_OBS", choices=["DEFAULT", "LR_COURSE_OBS"], help="Observation space mode")
    parser.add_argument("--task_env", type=str, default="LR_COURSE_TASK", choices=["LR_COURSE_TASK", "FLAGRUN","FWD_LOCOMOTION"], help="Task to be executed")
    parser.add_argument("--save-path", type=str, help="Path for storing intermediate models", default=".")
    parser.add_argument("--full_path", type=str, help="Full path to the model location", required=True)


    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    load_sb3(args)


if __name__ == "__main__":
    main()