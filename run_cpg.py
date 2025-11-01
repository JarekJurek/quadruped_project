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

""" Run CPG """

import numpy as np

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = False
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

PLOTTING = True
# PLOTTING = False

# Plot category flags
# PLOT_CPG_STATES = True
PLOT_CPG_STATES = False
# PLOT_FOOT_POSITIONS = True
PLOT_FOOT_POSITIONS = False
# PLOT_JOINT_ANGLES = True
PLOT_JOINT_ANGLES = False

GAIT = "TROT"
# GAIT = "PACE"
# GAIT = "BOUND"
# GAIT = "WALK"

SIM_DURATION = 5.

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP, 
                  mu=0.8**2,                 # intrinsic amplitude, converges to sqrt(mu)
                  omega_swing=6*2*np.pi,   # frequency in swing phase (can edit)
                  omega_stance=4*2*np.pi,  # frequency in stance phase (can edit)
                  gait=GAIT,             # Gait, can be TROT, WALK, PACE, BOUND, etc.
                  couple=True,             # whether oscillators should be coupled
                  )
# cpg = HopfNetwork(time_step=TIME_STEP, gait="PACE")
# cpg = HopfNetwork(time_step=TIME_STEP, gait="BOUND")
# cpg = HopfNetwork(time_step=TIME_STEP, gait="WALK")

TEST_STEPS = int(SIM_DURATION / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
cpg_r = np.zeros((4, TEST_STEPS))      # amplitudes
cpg_theta = np.zeros((4, TEST_STEPS))  # phases
cpg_r_dot = np.zeros((4, TEST_STEPS))  # amplitude derivatives
cpg_theta_dot = np.zeros((4, TEST_STEPS))  # phase derivatives

# Add data structures for foot positions
desired_foot_pos = np.zeros((4, 3, TEST_STEPS))  # [leg, xyz, time]
actual_foot_pos = np.zeros((4, 3, TEST_STEPS))   # [leg, xyz, time]

# Add data structures for joint angles
desired_joint_angles = np.zeros((12, TEST_STEPS))  # [joint, time] - 3 joints per leg * 4 legs
actual_joint_angles = np.zeros((12, TEST_STEPS))   # [joint, time]

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.array([500, 400, 400])
kdCartesian = np.array([30, 50, 40])

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 

  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # Store actual joint angles
  actual_joint_angles[:, j] = q

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

    # Store desired foot position
    desired_foot_pos[i, :, j] = leg_xyz

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # [TODO] 

    # Store desired joint angles
    desired_joint_angles[3*i:3*i+3, j] = leg_q

    # Add joint PD contribution to tau for leg i (Equation 4)
    tau += kp * (leg_q - q[3*i:3*i+3]) + kd * (0 - dq[3*i:3*i+3]) # [TODO] 

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      # [TODO] 

      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J, pos_leg_frame = env.robot.ComputeJacobianAndPosition(i)

      # Get current foot velocity in leg frame (Equation 2)
      foot_lin_vel_leg_frame = J @ dq[3*i:3*i+3]

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += J.T @ (kpCartesian @ (leg_xyz - pos_leg_frame) + kdCartesian @ (0 - foot_lin_vel_leg_frame)) # [TODO]

    # Get actual foot position and store it
    _, actual_pos_leg_frame = env.robot.ComputeJacobianAndPosition(i)
    actual_foot_pos[i, :, j] = actual_pos_leg_frame

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states
  cpg_r[:, j] = cpg.get_r()
  cpg_theta[:, j] = cpg.get_theta()
  cpg_r_dot[:, j] = cpg.get_dr()
  cpg_theta_dot[:, j] = cpg.get_dtheta()

##################################################### 
# PLOTS
#####################################################
# [TODO] Create your plots

if PLOTTING:

  # CPG States plotting
  if PLOT_CPG_STATES:
    leg_names = ['FR (Front Right)', 'FL (Front Left)', 'RR (Rear Right)', 'RL (Rear Left)']
    colors = ['red', 'blue', 'green', 'orange']
    joint_names = ['Hip', 'Thigh', 'Calf']

    # Create subplots for CPG states
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle('CPG States for Each Leg', fontsize=16)

    for i in range(4):
        # Plot amplitude r
        axes[0, i].plot(t, cpg_r[i, :], color=colors[i], linewidth=2)
        axes[0, i].set_title(f'{leg_names[i]} - Amplitude (r)')
        axes[0, i].set_ylabel('r')
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot phase θ
        axes[1, i].plot(t, cpg_theta[i, :], color=colors[i], linewidth=2)
        axes[1, i].set_title(f'{leg_names[i]} - Phase (θ)')
        axes[1, i].set_ylabel('θ [rad]')
        axes[1, i].grid(True, alpha=0.3)
        
        # Plot amplitude derivative ṙ
        axes[2, i].plot(t, cpg_r_dot[i, :], color=colors[i], linewidth=2)
        axes[2, i].set_title(f'{leg_names[i]} - Amplitude Rate (ṙ)')
        axes[2, i].set_ylabel('ṙ')
        axes[2, i].grid(True, alpha=0.3)
        
        # Plot phase derivative θ̇
        axes[3, i].plot(t, cpg_theta_dot[i, :], color=colors[i], linewidth=2)
        axes[3, i].set_title(f'{leg_names[i]} - Phase Rate (θ̇)')
        axes[3, i].set_ylabel('θ̇ [rad/s]')
        axes[3, i].set_xlabel('Time [s]')
        axes[3, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Optional: Create a phase portrait plot showing r vs θ for each leg
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('CPG Phase Portraits (r vs θ)', fontsize=16)

    for i in range(4):
        row = i // 2
        col = i % 2
        axes2[row, col].plot(cpg_theta[i, :], cpg_r[i, :], color=colors[i], linewidth=2)
        axes2[row, col].set_title(f'{leg_names[i]} Phase Portrait')
        axes2[row, col].set_xlabel('θ [rad]')
        axes2[row, col].set_ylabel('r')
        axes2[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

  # Foot position plots
  if PLOT_FOOT_POSITIONS:
    if not PLOT_CPG_STATES:  # Only define these if not already defined
      leg_names = ['FR (Front Right)', 'FL (Front Left)', 'RR (Rear Right)', 'RL (Rear Left)']
      colors = ['red', 'blue', 'green', 'orange']
    coord_names = ['X', 'Y', 'Z']
    
    # Create subplots for foot position comparison
    fig3, axes3 = plt.subplots(3, 4, figsize=(16, 12))
    fig3.suptitle('Desired vs Actual Foot Positions', fontsize=16)

    for i in range(4):  # for each leg
        for coord in range(3):  # for each coordinate (x, y, z)
            axes3[coord, i].plot(t, desired_foot_pos[i, coord, :], 
                                color=colors[i], linewidth=2, linestyle='--', 
                                label='Desired', alpha=0.8)
            axes3[coord, i].plot(t, actual_foot_pos[i, coord, :], 
                                color=colors[i], linewidth=2, linestyle='-', 
                                label='Actual')
            
            axes3[coord, i].set_title(f'{leg_names[i]} - {coord_names[coord]} Position')
            axes3[coord, i].set_ylabel(f'{coord_names[coord]} [m]')
            axes3[coord, i].grid(True, alpha=0.3)
            axes3[coord, i].legend()
            
            if coord == 2:  # only add x-label to bottom row
                axes3[coord, i].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.show()

    # 3D trajectory plots for each leg
    fig4 = plt.figure(figsize=(16, 12))
    fig4.suptitle('3D Foot Trajectories: Desired vs Actual', fontsize=16)

    for i in range(4):
        ax = fig4.add_subplot(2, 2, i+1, projection='3d')
        
        # Plot desired trajectory
        ax.plot(desired_foot_pos[i, 0, :], desired_foot_pos[i, 1, :], desired_foot_pos[i, 2, :],
                color=colors[i], linewidth=3, linestyle='--', label='Desired', alpha=0.8)
        
        # Plot actual trajectory
        ax.plot(actual_foot_pos[i, 0, :], actual_foot_pos[i, 1, :], actual_foot_pos[i, 2, :],
                color=colors[i], linewidth=2, linestyle='-', label='Actual')
        
        # Mark start and end points
        ax.scatter(desired_foot_pos[i, 0, 0], desired_foot_pos[i, 1, 0], desired_foot_pos[i, 2, 0],
                  color='green', s=100, marker='o', label='Start')
        ax.scatter(desired_foot_pos[i, 0, -1], desired_foot_pos[i, 1, -1], desired_foot_pos[i, 2, -1],
                  color='red', s=100, marker='x', label='End')
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f'{leg_names[i]} 3D Trajectory')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([desired_foot_pos[i, :, :].max() - desired_foot_pos[i, :, :].min(),
                             actual_foot_pos[i, :, :].max() - actual_foot_pos[i, :, :].min()]).max() / 2.0
        mid_x = (desired_foot_pos[i, 0, :].max() + desired_foot_pos[i, 0, :].min()) * 0.5
        mid_y = (desired_foot_pos[i, 1, :].max() + desired_foot_pos[i, 1, :].min()) * 0.5
        mid_z = (desired_foot_pos[i, 2, :].max() + desired_foot_pos[i, 2, :].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

    # Position tracking error plot
    fig5, axes5 = plt.subplots(2, 2, figsize=(12, 10))
    fig5.suptitle('Position Tracking Error (Desired - Actual)', fontsize=16)

    for i in range(4):
        row = i // 2
        col = i % 2
        
        # Calculate tracking errors for each coordinate
        error_x = desired_foot_pos[i, 0, :] - actual_foot_pos[i, 0, :]
        error_y = desired_foot_pos[i, 1, :] - actual_foot_pos[i, 1, :]
        error_z = desired_foot_pos[i, 2, :] - actual_foot_pos[i, 2, :]
        
        axes5[row, col].plot(t, error_x, 'r-', linewidth=2, label='X Error')
        axes5[row, col].plot(t, error_y, 'g-', linewidth=2, label='Y Error')
        axes5[row, col].plot(t, error_z, 'b-', linewidth=2, label='Z Error')
        
        axes5[row, col].set_title(f'{leg_names[i]} Position Error')
        axes5[row, col].set_xlabel('Time [s]')
        axes5[row, col].set_ylabel('Error [m]')
        axes5[row, col].grid(True, alpha=0.3)
        axes5[row, col].legend()
        
        # Add RMS error in title
        rms_error = np.sqrt(np.mean(error_x**2 + error_y**2 + error_z**2))
        axes5[row, col].text(0.02, 0.98, f'RMS Error: {rms_error:.4f}m', 
                            transform=axes5[row, col].transAxes, 
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

  # Joint angle plots  
  if PLOT_JOINT_ANGLES:
    if not PLOT_CPG_STATES and not PLOT_FOOT_POSITIONS:  # Only define these if not already defined
      leg_names = ['FR (Front Right)', 'FL (Front Left)', 'RR (Rear Right)', 'RL (Rear Left)']
      colors = ['red', 'blue', 'green', 'orange']
    if not PLOT_CPG_STATES:  # Only define joint_names if not already defined
      joint_names = ['Hip', 'Thigh', 'Calf']

    # Joint angle comparison plot
    fig6, axes6 = plt.subplots(3, 4, figsize=(16, 12))
    fig6.suptitle('Desired vs Actual Joint Angles', fontsize=16)

    for i in range(4):  # for each leg
        for joint in range(3):  # for each joint (hip, thigh, calf)
            joint_idx = 3*i + joint
            
            axes6[joint, i].plot(t, desired_joint_angles[joint_idx, :], 
                                color=colors[i], linewidth=2, linestyle='--', 
                                label='Desired', alpha=0.8)
            axes6[joint, i].plot(t, actual_joint_angles[joint_idx, :], 
                                color=colors[i], linewidth=2, linestyle='-', 
                                label='Actual')
            
            axes6[joint, i].set_title(f'{leg_names[i]} - {joint_names[joint]} Joint')
            axes6[joint, i].set_ylabel(f'{joint_names[joint]} Angle [rad]')
            axes6[joint, i].grid(True, alpha=0.3)
            axes6[joint, i].legend()
            
            if joint == 2:  # only add x-label to bottom row
                axes6[joint, i].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.show()

    # Joint angle tracking error plot
    fig7, axes7 = plt.subplots(2, 2, figsize=(12, 10))
    fig7.suptitle('Joint Angle Tracking Error (Desired - Actual)', fontsize=16)

    for i in range(4):
        row = i // 2
        col = i % 2
        
        # Calculate tracking errors for each joint
        error_hip = desired_joint_angles[3*i, :] - actual_joint_angles[3*i, :]
        error_thigh = desired_joint_angles[3*i+1, :] - actual_joint_angles[3*i+1, :]
        error_calf = desired_joint_angles[3*i+2, :] - actual_joint_angles[3*i+2, :]
        
        axes7[row, col].plot(t, error_hip, 'r-', linewidth=2, label='Hip Error')
        axes7[row, col].plot(t, error_thigh, 'g-', linewidth=2, label='Thigh Error')
        axes7[row, col].plot(t, error_calf, 'b-', linewidth=2, label='Calf Error')
        
        axes7[row, col].set_title(f'{leg_names[i]} Joint Angle Error')
        axes7[row, col].set_xlabel('Time [s]')
        axes7[row, col].set_ylabel('Error [rad]')
        axes7[row, col].grid(True, alpha=0.3)
        axes7[row, col].legend()
        
        # Add RMS error statistics
        rms_error_hip = np.sqrt(np.mean(error_hip**2))
        rms_error_thigh = np.sqrt(np.mean(error_thigh**2))
        rms_error_calf = np.sqrt(np.mean(error_calf**2))
        
        axes7[row, col].text(0.02, 0.98, 
                            f'RMS Errors:\nHip: {rms_error_hip:.4f} rad\nThigh: {rms_error_thigh:.4f} rad\nCalf: {rms_error_calf:.4f} rad', 
                            transform=axes7[row, col].transAxes, 
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Overall joint angle comparison - all joints in one plot
    fig8, axes8 = plt.subplots(4, 1, figsize=(14, 12))
    fig8.suptitle('All Joint Angles by Leg', fontsize=16)

    for i in range(4):
        # Plot all three joints for this leg
        for joint in range(3):
            joint_idx = 3*i + joint
            axes8[i].plot(t, desired_joint_angles[joint_idx, :], 
                         linewidth=2, linestyle='--', alpha=0.8,
                         label=f'{joint_names[joint]} Desired')
            axes8[i].plot(t, actual_joint_angles[joint_idx, :], 
                         linewidth=2, linestyle='-',
                         label=f'{joint_names[joint]} Actual')
        
        axes8[i].set_title(f'{leg_names[i]} - All Joints')
        axes8[i].set_ylabel('Joint Angle [rad]')
        axes8[i].grid(True, alpha=0.3)
        axes8[i].legend(ncol=3, loc='upper right')
        
        if i == 3:  # only add x-label to bottom plot
            axes8[i].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.show()