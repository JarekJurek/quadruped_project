import argparse
import warnings
from functools import partial
from typing import List

import numpy as np
import optuna
from optuna.trial import Trial

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv
from run_cpg import CPGSimulator


N_LEGS = 4
N_JOINTS = 3
TIME_STEP = 0.001

LOG_TO_TERMINAL = True


def quadruped_cpg_optimization(args):
    # Initialize simulation
    # Feel free to change these options! (except for control_mode and timestep)
    env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

    n_trials = args.n_trials


    gait_type = get_gait_type(args)

    # Create a maximization problem
    objective = partial(evaluate_run, env=env, gait_type=gait_type)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name="Quadruped Running Optimization",
        sampler=sampler,
        direction="maximize",
    )

    # Run the optimization
    study.optimize(objective, n_trials=n_trials)

    # Log the results
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    # OPTIONAL: add additional functions here (e.g., plotting, recording to file)
    # E.g., cycling through all the evaluated parameters and values:
    for trial in study.get_trials():
        trial.number  # Number of the trial
        trial.params  # Used parameters
        trial.value  # Resulting objective function value


def evaluate_run(trial: Trial, env: QuadrupedGymEnv, gait_type: str) -> float:
    # Create parameters dictionary with optimization variables
    params = {
        "mu": trial.suggest_float(name="mu", low=0.5**2, high=3**2, step=0.25),                 # intrinsic amplitude, converges to sqrt(mu)
        "omega_swing": trial.suggest_float(name="omega_swing", low=0.25, high=4, step=0.25),   # frequency in swing phase (can edit)
        "omega_stance": trial.suggest_float(name="omega_stance", low=0.25, high=4, step=0.25),
        "alpha": 50,                # amplitude convergence factor
        "des_step_len": 0.05,       # desired step length
        "sim_duration": 5,
        "timestep": 1e-3,
        "make_plots": False,
        "save_to_csv": False,
        "gait_type": gait_type,
    }

    env.reset()

    cpg_sim = CPGSimulator(env, params)

    cpg_sim.run()

    total_objective, objective_values = get_objective(env)

    return total_objective


def get_gait_type(args):
    gait_types = ["TROT", "PACE", "BOUND", "WALK"]

    if args.gait_type in gait_types:
        gait_type = args.gait_type
    else:
        warnings.warn(
            f"Gait type '{args.gait_type}' not in the list of available gaits: {gait_types}. Using default 'TROT' instead."
        )
        gait_type = "TROT"
    return gait_type


def get_objective(env: QuadrupedGymEnv):
    base_pos_x, base_pos_y, base_pos_z = env.robot.GetBasePosition()
    base_vel_x, base_vel_y, base_vel_z = env.robot.GetBaseLinearVelocity()
    base_roll, base_pitch, base_yaw = env.robot.GetBaseOrientationRollPitchYaw()

    vel_xy_plane = np.sqrt(base_vel_x ** 2 + base_vel_y ** 2)

    # Define objective components with weights in a structured way
    objective_config = {
        "vel_xy_plane": {"value": vel_xy_plane, "weight": 10.0, "description": "Forward velocity"},
        # "pos_y_objective": {
        #     "value": -np.abs(base_pos_y),
        #     "weight": 5.0,
        #     "description": "Penalty for lateral deviation",
        # },
        "height_bonus": {
            "value": max(0, base_pos_z - 0.10),
            "weight": 2.0,
            "description": "Bonus for jumping height above 10cm",
        },
        "roll_penalty": {"value": -np.abs(base_roll), "weight": 5.0, "description": "Penalty for excessive roll"},
        "pitch_penalty": {"value": -np.abs(base_pitch), "weight": 5.0, "description": "Penalty for excessive pitch"},
        "z_extreme_penalty": {
            "value": -10000.0 if base_pos_z > 1.0 else 0.0,
            "weight": 1.0,
            "description": "Huge penalty for z height above 3m",
        },
        # "y_extreme_penalty": {
        #     "value": -10000.0 if np.abs(base_pos_y) > 1.0 else 0.0,
        #     "weight": 1.0,
        #     "description": "Huge penalty for y deviation above 3m",
        # },
    }

    # Calculate weighted objectives and total
    total_objective = 0.0
    objective_values = {}

    if LOG_TO_TERMINAL:
        print("Objective breakdown:")
        print("-" * 50)

    for component_name, config in objective_config.items():
        weighted_value = config["weight"] * config["value"]
        total_objective += weighted_value
        objective_values[component_name] = weighted_value

        if LOG_TO_TERMINAL:
            print(
                f"{config['description']:<30}: "
                f"raw={config['value']:.3f}, "
                f"weighted={weighted_value:.3f} "
                f"(w={config['weight']})"
            )
    if LOG_TO_TERMINAL:
        print("-" * 50)
        print(f"{'Total objective':<30}: {total_objective:.3f}")

    return total_objective, objective_values


def get_thresholded_penalty(value_container: List[np.array], threshold, penalty):
    if not value_container:
        return 0.0
    
    # Find the maximum absolute value across all arrays in the container
    max_value = 0.0
    for values in value_container:
        current_max = np.max(np.abs(values))
        max_value = max(max_value, current_max)

    # print("=================")
    # print(max_value)
    # print("=================")
    
    return penalty if max_value > threshold else 0.0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Quadruped CPG optimization with Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials to run (default: 50)")
    parser.add_argument("--gait-type", type=str, default="TROT", help="Gait type to be optimized")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    quadruped_cpg_optimization(args)


if __name__ == "__main__":
    main()
