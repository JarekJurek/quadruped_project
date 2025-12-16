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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import VecNormalize

from env.quadruped_gym_env import QuadrupedGymEnv
from env.hopf_network import MU_LOW, MU_UPP
from utils.file_utils import get_latest_model
from utils.utils import plot_results


def load_sb3(args):
    log_dir = args.full_path

    # Initialize environment configuration
    env_config = {
        "render": True,
        "record_video": args.record_video,
        "add_noise": args.add_noise,
        "motor_control_mode": args.motor_control_mode,
        "task_env": args.task_env,
        "observation_space_mode": args.observation_space_mode,
        "des_vel_x": args.des_x_vel,
        "des_h": args.des_h,
        "des_g_c": args.des_g_c,
        "terrain": args.terrain,
        "num_stairs": args.num_stairs,
        "stair_height": args.stair_height,
        "stair_width": args.stair_width,
        "enable_vmc": args.enable_vmc,
        "k_vmc": args.k_vmc,
        "orientation_weight": args.orientation_weight,
        "max_episode_length": 10000000,
        "slope_pitch": args.slope_pitch,
        "randomize_velocity_command": args.randomize_velocity_command,
    }

    # Load model and normalization stats
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    model_name = get_latest_model(log_dir)
    monitor_results = load_results(log_dir)
    print(monitor_results)
    plot_results([log_dir], 10e10, "timesteps", args.learning_alg + " ")
    plt.show()

    # Create environment
    def create_env():
        return QuadrupedGymEnv(**env_config)

    env = make_vec_env(create_env, n_envs=1)

    # Get the actual time step from the environment
    quad_env = env.envs[0].unwrapped
    env_dt = quad_env._time_step * quad_env._action_repeat

    # Calculate number of steps based on sim_time (ms)
    sim_duration_seconds = args.sim_time / 1000.0
    num_steps = int(sim_duration_seconds / env_dt)

    print(f"Simulation duration: {sim_duration_seconds}s")
    print(f"Environment dt: {env_dt}s")
    print(f"Total steps: {num_steps}")

    env = VecNormalize.load(stats_path, env)
    env.training = False  # do not update stats at test time
    env.norm_reward = False  # reward normalization is not needed at test time

    # load model
    if args.learning_alg == "PPO":
        model = PPO.load(model_name, env)
    elif args.learning_alg == "SAC":
        model = SAC.load(model_name, env)
    print("\nLoaded model", model_name, "\n")

    obs = env.reset()
    episode_reward = 0

    # [TODO] initialize arrays to save data from simulation

    base_pos = np.zeros((3, num_steps))
    base_vel = np.zeros((3, num_steps))
    des_vel_x_hist = np.zeros(num_steps)
    t = np.arange(num_steps) * env_dt

    # CPG data
    cpg_r_history = np.zeros((4, num_steps))
    cpg_theta_history = np.zeros((4, num_steps))
    foot_pos_history = np.zeros((4, 3, num_steps))

    cot_history = []

    velocity_list = [0.75, 0.35]

    for i in range(num_steps):
        if args.randomize_velocity_command:
            bin_size = num_steps // len(velocity_list)
            n = min(i // bin_size, len(velocity_list) - 1)
            env.envs[0].unwrapped._des_vel_x = velocity_list[n]
            print(f"des_vel: {env.envs[0].unwrapped._des_vel_x}")
            print(f"bin_size: {bin_size}")

        if args.randomize_cpg_params:
            if args.cpg_rand_param == "body_height":
                param_list = [0.25, 0.3, 0.25]
                bin_size = num_steps // len(param_list)
                n = min(i // bin_size, len(param_list) - 1)
                env.envs[0].unwrapped._cpg._robot_height = param_list[n]
                print(f"des_h: {env.envs[0].unwrapped._cpg._robot_height}")
                print(f"bin_size: {bin_size}")
            else:
                param_list = [0.04, 0.12, 0.04]
                bin_size = num_steps // len(param_list)
                n = min(i // bin_size, len(param_list) - 1)
                env.envs[0].unwrapped._cpg._ground_clearance = param_list[n]
                print(f"des_h: {env.envs[0].unwrapped._cpg._ground_clearance}")
                print(f"bin_size: {bin_size}")

        if hasattr(env.envs[0].unwrapped, "_des_vel_x"):
            des_vel_x_hist[i] = env.envs[0].unwrapped._des_vel_x
        else:
            des_vel_x_hist[i] = args.des_x_vel

        action, _states = model.predict(obs, deterministic=False)  # sample at test time? ([TODO]: test if the outputs make sense)
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards

        print(f"load sb3 steps: {i}")

        if dones:
            print("episode_reward", episode_reward)
            print("Final base position", info[0]["base_pos"])
            episode_reward = 0

        robot = env.envs[0].unwrapped.robot
        base_pos[:, i] = robot.GetBasePosition()
        base_vel[:, i] = robot.GetBaseLinearVelocity()

        # Save CPG data
        quad_env = env.envs[0].unwrapped
        if hasattr(quad_env, "_cpg"):
            cpg_r_history[:, i] = quad_env._cpg.X[0, :]
            cpg_theta_history[:, i] = quad_env._cpg.X[1, :]

        # Save foot positions
        for leg_i in range(4):
            _, pos = quad_env.robot.ComputeJacobianAndPosition(leg_i)
            foot_pos_history[leg_i, :, i] = pos

        # [TODO] save data from current robot states for plots
        # To get base position, for example: env.envs[0].env.robot.GetBasePosition()
        quad_env = env.envs[0].unwrapped
        if hasattr(quad_env, "_dt_motor_torques") and hasattr(quad_env, "_dt_motor_velocities"):
            step_power = 0
            for tau, vel in zip(quad_env._dt_motor_torques, quad_env._dt_motor_velocities):
                step_power += np.sum(np.abs(np.array(tau) * np.array(vel)))

            avg_power = step_power / len(quad_env._dt_motor_torques)

            total_mass = sum(quad_env.robot._total_mass_urdf)
            v_x = base_vel[0, i]

            if v_x > 0.05:  # Avoid division by zero or very small velocities
                cot = avg_power / (total_mass * 9.81 * v_x)
                cot_history.append(cot)
            else:
                cot_history.append(0.0)

    # Generate plots
    _plot_cpg_states(args, t, cpg_r_history, cpg_theta_history, env_dt, num_steps)
    _plot_base_position(t, base_pos)
    _plot_base_velocity(t, base_vel, des_vel_x_hist)
    _plot_foot_positions(foot_pos_history, quad_env)

    print(f"mean COT: {np.mean(cot_history)}")

    data = {
        "time": t,
        "pos_x": base_pos[0, :],
        "pos_y": base_pos[1, :],
        "pos_z": base_pos[2, :],
        "vel_x": base_vel[0, :],
        "vel_y": base_vel[1, :],
        "vel_z": base_vel[2, :],
        "des_vel_x": des_vel_x_hist,
        "cot": cot_history,
        "cpg_r_FR": cpg_r_history[0, :],
        "cpg_r_FL": cpg_r_history[1, :],
        "cpg_r_RR": cpg_r_history[2, :],
        "cpg_r_RL": cpg_r_history[3, :],
        "cpg_theta_FR": cpg_theta_history[0, :],
        "cpg_theta_FL": cpg_theta_history[1, :],
        "cpg_theta_RR": cpg_theta_history[2, :],
        "cpg_theta_RL": cpg_theta_history[3, :],
    }

    model_name = log_dir.split("/")[-1]
    if model_name == "":
        model_name = log_dir.split("/")[-2]
    print(f"model name: {model_name}")
    print(f"log_dir: {log_dir}")

    df = pd.DataFrame(data)
    # csv_filename = os.path.join(log_dir, f"simulation_data_{args.project_name}_{model_name}.csv")
    csv_filename = f"simulation_data_{args.project_name}_{model_name}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")


def _plot_cpg_states(args, t, cpg_r_history, cpg_theta_history, env_dt, num_steps):
    """Plot CPG states (r and theta) for all legs within specified time range."""
    start_time = args.plot_start_time
    end_time = args.plot_finish_time
    start_idx = int(start_time / env_dt)
    end_idx = int(end_time / env_dt)

    # Check bounds
    start_idx = max(0, start_idx)
    end_idx = min(num_steps, end_idx)

    if start_idx < end_idx:
        time_range = t[start_idx:end_idx]
        legs = ["FR", "FL", "RR", "RL"]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        
        for i, leg in enumerate(legs):
            ax1.plot(time_range, cpg_r_history[i, start_idx:end_idx], label=leg, linewidth=2)
            ax2.plot(time_range, cpg_theta_history[i, start_idx:end_idx], label=leg, linewidth=2)

        ax1.set_ylabel("Amplitude (r)")
        ax1.set_title(f"CPG States ({start_time:.3f}s - {end_time:.3f}s)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel("Phase (theta)")
        ax2.set_xlabel("Time [s]")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        sim_duration = num_steps * env_dt
        print(f"Warning: Time range {start_time}s-{end_time}s is outside simulation duration {sim_duration:.3f}s.")


def _plot_base_position(t, base_pos):
    """Plot base position in X, Y, Z coordinates."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Base Position", fontsize=16)
    
    labels = ["X", "Y", "Z"]
    colors = ["blue", "orange", "green"]
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        axes[i].plot(t, base_pos[i, :], linewidth=2, color=color, alpha=0.8)
        axes[i].set_ylabel(f"{label} Position [m]")
        axes[i].grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def _plot_base_velocity(t, base_vel, des_vel_x_hist):
    """Plot base velocity in X, Y, Z coordinates with desired velocity overlay."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Base Velocity", fontsize=16)
    
    labels = ["Vx", "Vy", "Vz"]
    colors = ["blue", "orange", "green"]
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        axes[i].plot(t, base_vel[i, :], linewidth=2, label="Actual", color=color, alpha=0.8)
        
        if i == 0:  # Add desired velocity for x-axis only
            axes[i].plot(t, des_vel_x_hist, linewidth=2, linestyle="--", 
                        label="Desired", color="red", alpha=0.7)
            
        axes[i].set_ylabel(f"{label} [m/s]")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def _plot_foot_positions(foot_pos_history, quad_env):
    """Plot foot positions in XZ plane with theoretical reference trajectory."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Foot Positions in XZ Plane (Leg Frame)", fontsize=16)
    
    legs = ["Front Right", "Front Left", "Rear Right", "Rear Left"]
    leg_abbrev = ["FR", "FL", "RR", "RL"]
    colors = ["red", "blue", "orange", "green"]
    
    # Generate theoretical reference trajectory
    cpg = quad_env._cpg
    theta_cycle = np.linspace(0, 2 * np.pi, 200)
    x_ref = -cpg._max_step_len_rl * (MU_UPP - MU_LOW) * np.cos(theta_cycle)
    z_ref = np.zeros_like(theta_cycle)
    
    for t_idx, th in enumerate(theta_cycle):
        theta_sin = np.sin(th)
        if theta_sin > 0:
            z_ref[t_idx] = -cpg._robot_height + cpg._ground_clearance * theta_sin
        else:
            z_ref[t_idx] = -cpg._robot_height + cpg._ground_penetration * theta_sin
    
    for i, (leg_name, leg_abbr, color) in enumerate(zip(legs, leg_abbrev, colors)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Plot actual foot trajectory
        ax.plot(foot_pos_history[i, 0, :], foot_pos_history[i, 2, :], 
               label="Actual", linewidth=2, color=color, alpha=0.8)
        
        # Plot reference trajectory
        ax.plot(x_ref, z_ref, label="Reference", linestyle="--", 
               color="black", alpha=0.6, linewidth=1.5)
        
        ax.set_title(f"{leg_name} ({leg_abbr})")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")
    
    plt.tight_layout()
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Quadruped RL Model Evaluation")

    # Model and project configuration
    parser.add_argument("--project-name", type=str, default="quadruped_rl", help="Name of the project")
    parser.add_argument("--full_path", type=str, required=True, help="Full path to the model location")
    parser.add_argument("--learning-alg", type=str, default="PPO", choices=["PPO", "SAC"], help="Learning algorithm used for training")

    # Simulation configuration
    parser.add_argument("--sim_time", type=int, default=500, help="Duration of simulation in milliseconds")
    parser.add_argument("--record_video", action="store_true", help="Record video during evaluation")
    parser.add_argument("--add_noise", action="store_true", help="Add noise during evaluation")

    # Environment configuration
    parser.add_argument(
        "--motor_control_mode", type=str, default="CPG", choices=["CPG", "PD", "TORQUE", "CARTESIAN_PD"], help="Motor control mode"
    )
    parser.add_argument(
        "--task_env",
        type=str,
        default="FWD_CUSTOM",
        choices=["LR_COURSE_TASK", "FLAGRUN", "FWD_LOCOMOTION", "FWD_CUSTOM", "FWD_BASIC"],
        help="Task environment to evaluate",
    )
    parser.add_argument(
        "--observation_space_mode",
        type=str,
        default="LR_COURSE_OBS_EXTENDED",
        choices=["DEFAULT", "LR_COURSE_OBS", "LR_COURSE_OBS_EXTENDED"],
        help="Observation space configuration",
    )

    # Robot control parameters
    parser.add_argument("--des_x_vel", type=float, default=0.6, help="Desired forward velocity (m/s)")
    parser.add_argument("--des_h", type=float, default=0.25, help="Desired body height (m)")
    parser.add_argument("--des_g_c", type=float, default=0.08, help="Desired ground clearance during swing phase (m)")
    parser.add_argument("--enable_vmc", action="store_true", help="Enable Virtual Model Control")
    parser.add_argument("--k_vmc", type=float, default=250.0, help="VMC stiffness parameter")
    parser.add_argument("--orientation_weight", type=float, default=1.0, help="Weight for orientation penalty in reward function")

    # Terrain configuration
    parser.add_argument(
        "--terrain", type=str, default="NONE", choices=["STAIRS", "SLOPES", "GAPS", "RANDOM", "NONE"], help="Terrain type for evaluation"
    )
    parser.add_argument("--num_stairs", type=int, default=12, help="Number of stairs in stair terrain")
    parser.add_argument("--stair_height", type=float, default=0.05, help="Height of each stair (m)")
    parser.add_argument("--stair_width", type=float, default=0.25, help="Width of each stair (m)")
    parser.add_argument("--slope_pitch", type=float, default=0.2, help="Slope angle for slope terrain (radians)")

    # Curriculum and randomization
    parser.add_argument("--randomize_velocity_command", action="store_true", help="Randomize velocity commands during evaluation")
    parser.add_argument("--randomize_cpg_params", action="store_true", help="Randomize CPG parameters during evaluation")
    parser.add_argument(
        "--cpg_rand_param", type=str, default="body_height", choices=["foot_height", "body_height"], help="Which CPG parameter to randomize"
    )

    # Plotting configuration
    parser.add_argument("--plot_start_time", type=float, default=0.0, help="Start time for CPG state plots (seconds)")
    parser.add_argument("--plot_finish_time", type=float, default=0.0001, help="End time for CPG state plots (seconds)")

    return parser.parse_args()


def main():
    args = parse_arguments()
    load_sb3(args)


if __name__ == "__main__":
    main()
