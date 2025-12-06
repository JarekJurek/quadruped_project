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
    env_config["des_vel_x"]=args.des_x_vel
    env_config["des_h"]=args.des_h
    env_config["des_g_c"]=args.des_g_c
    env_config["terrain"]=args.terrain
    env_config["num_stairs"]=args.num_stairs
    env_config["stair_height"]=args.stair_height
    env_config["stair_width"]=args.stair_width
    env_config["enable_vmc"] = args.enable_vmc
    env_config["k_vmc"] = args.k_vmc
    env_config["orientation_weight"] = args.orientation_weight
    env_config["max_episode_length"] = args.sim_time / 100

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

    base_pos = np.zeros((3, args.sim_time))
    base_vel = np.zeros((3, args.sim_time))
    t = np.arange(args.sim_time) * 0.001

    for i in range(args.sim_time):
        # print(f"sim time: {i}")
        action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test if the outputs make sense)
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards

        print(f"load sb3 steps: {i}")
        
        if dones:
            print('episode_reward', episode_reward)
            print('Final base position', info[0]['base_pos'])
            episode_reward = 0

        robot = env.envs[0].unwrapped.robot
        base_pos[:, i] = robot.GetBasePosition()
        base_vel[:, i] = robot.GetBaseLinearVelocity()

        # [TODO] save data from current robot states for plots 
        # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
        
    # [TODO] make plots

    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig1.suptitle('Base Position', fontsize=16)
    labels_pos = ['X', 'Y', 'Z']
    
    for i in range(3):
        axes1[i].plot(t, base_pos[i, :], linewidth=2)
        axes1[i].set_ylabel(f'{labels_pos[i]} [m]')
        axes1[i].grid(True, alpha=0.3)
    axes1[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()

    # Plot Base Velocity
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig2.suptitle('Base Velocity', fontsize=16)
    labels_vel = ['Vx', 'Vy', 'Vz']
    
    for i in range(3):
        axes2[i].plot(t, base_vel[i, :], linewidth=2)
        axes2[i].set_ylabel(f'{labels_vel[i]} [m/s]')
        axes2[i].grid(True, alpha=0.3)
    axes2[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Quadruped RL training with Stable Baselines 3")
    parser.add_argument("--project-name", type=str, default="quadruped_rl", help="Name of the project")

    parser.add_argument("--record_video", type=bool, default=False, help="Record video flag")
    parser.add_argument("--add_noise", action="store_true", help="Add noise flag")

    parser.add_argument("--sim_time", type=int, default=500, help="Duration of the simulation in miliseconds (has to be integer)")

    parser.add_argument("--des_x_vel", type=float, default=0.4, help="desired linear velocity x axis")

    parser.add_argument("--des_h", type=float, default=0.3, help="desired h - z of the body")
    parser.add_argument("--des_g_c", type=float, default=0.07, help="desired g_c - max z distance of a feet in swing phase")

    parser.add_argument("--enable_vmc", action="store_true", help="Enable Virtual Model Control (VMC)")
    parser.add_argument("--k_vmc", type=float, default=250.0, help="VMC gain parameter")

    parser.add_argument("--orientation_weight", type=float, default=1.0, help="Weight for orientation penalty in the reward function")


    parser.add_argument("--terrain", type=str, default="NONE", choices=["STAIRS", "SLOPES", "GAPS", "RANDOM", "NONE"], help="Terrain, obstacles")
    parser.add_argument("--num_stairs", type=int, default=12, help="desired h - z of the body")
    parser.add_argument("--stair_height", type=float, default=0.05, help="desired h - z of the body")
    parser.add_argument("--stair_width", type=float, default=0.25, help="desired h - z of the body")

    parser.add_argument("--learning-alg", type=str, default="PPO", choices=["PPO", "SAC"], help="Learning algorithm to use (default: PPO)")
    parser.add_argument("--motor_control_mode", type=str, default="CPG", choices=["CPG", "PD","TORQUE", "CARTESIAN_PD"], help="Motor control mode")
    parser.add_argument("--observation_space_mode", type=str, default="LR_COURSE_OBS_EXTENDED", choices=["DEFAULT", "LR_COURSE_OBS", "LR_COURSE_OBS_EXTENDED"], help="Observation space mode")
    parser.add_argument("--task_env", type=str, default="FWD_CUSTOM", choices=["LR_COURSE_TASK", "FLAGRUN","FWD_LOCOMOTION", "FWD_CUSTOM", "FWD_BASIC"], help="Task to be executed")
    parser.add_argument("--save-path", type=str, help="Path for storing intermediate models", default=".")
    parser.add_argument("--full_path", type=str, help="Full path to the model location", required=True)


    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    load_sb3(args)


if __name__ == "__main__":
    main()