import argparse
import os
from functools import partial

import optuna
from optuna.trial import Trial

import wandb
from env.quadruped_gym_env import QuadrupedGymEnv
from run_cpg import CPGSimulator
from quadruped_jump_opt import get_gait_type, get_objective

N_LEGS = 4
N_JOINTS = 3

TIME_STEP = 0.001

class WeightsAndBiasesCallback:
    def __init__(self, metric_name: str = "objective_value"):
        self._metric_name = metric_name
        self._best_value = float('-inf')  # Track best value seen so far
        self._best_params = None  # Track best parameters

    def __call__(self, study, trial):
        # Always log current trial metrics
        wandb.log({
            self._metric_name: trial.value,
            "trial_number": trial.number,
            "study_best_value": study.best_value,
        })
        
        # Check if this trial achieved a new best value
        if trial.value is not None and trial.value > self._best_value:
            self._best_value = trial.value
            self._best_params = trial.params.copy()
            
            # Log the new best parameters
            best_params_log = {f"best_{param_name}": param_value 
                              for param_name, param_value in self._best_params.items()}
            
            wandb.log({
                "new_best_found": True,
                "best_objective_value": self._best_value,
                **best_params_log
            })


def quadruped_cpg_optimization(args):
    # Get worker ID from LSF environment (or default for local testing)
    worker_id = int(os.getenv('LSB_JOBINDEX', '1'))

    n_trials = args.n_trials

    gait_type = get_gait_type(args)
    
    # Initialize wandb with unique run name for each worker
    wandb.init(
        project=args.project_name,
        name=f"worker-{worker_id}",
        config={
            "worker_id": worker_id,
            "optimization_method": "optuna-distributed",
            "sampler": "TPE",
            "n_trials": n_trials,
            "gait_type": gait_type
        }
    )
    
    print(f"Starting worker {worker_id}")
    
    env = QuadrupedGymEnv(render=True,              # visualize
                on_rack=False,              # useful for debugging! 
                isRLGymInterface=False,     # not using RL
                time_step=TIME_STEP,
                action_repeat=1,
                motor_control_mode="TORQUE",
                add_noise=False,    # start in ideal conditions
                # record_video=True
                )

    # Use shared database storage for distributed optimization
    storage_path = os.path.expanduser(f"~/{args.project_name}.db")
    storage = f"sqlite:///{storage_path}"
    
    # Create wandb callback for optuna
    wandbc = WeightsAndBiasesCallback(metric_name="objective_value")
    
    # Create a maximization problem with distributed storage
    objective = partial(evaluate_run, env=env, gait_type=gait_type)
    sampler = optuna.samplers.TPESampler(seed=42 + worker_id)  # Different seed per worker
    study = optuna.create_study(
        study_name="Quadruped_Jumping_Distributed",
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,  # CRITICAL: Continue existing study
    )
    
    print(f"Worker {worker_id} starting {n_trials} trials...")

    # Run the optimization with wandb callback
    study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])

    # Close the simulation
    # simulator.close()

    # Log final results to wandb
    final_best_params = study.best_params
    best_params_final_log = {f"final_best_{param_name}": param_value 
                            for param_name, param_value in final_best_params.items()}
    
    wandb.log({
        "worker_final_best_value": study.best_value,
        "worker_total_trials": len(study.trials),
        "worker_completed": True,
        **best_params_final_log  # Include all best parameters with 'final_best_' prefix
    })

    # Log the results
    print(f"Worker {worker_id} completed.")
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    print(f"Total trials in study: {len(study.trials)}")

    # OPTIONAL: Log optimization history plot to wandb
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        wandb.log({"optimization_history": fig})
    except Exception as e:
        print(f"Could not create optimization plot: {e}")

    # OPTIONAL: Log parameter importance plot
    try:
        fig = optuna.visualization.plot_param_importances(study)
        wandb.log({"parameter_importance": fig})
    except Exception as e:
        print(f"Could not create parameter importance plot: {e}")

    wandb.finish()


def evaluate_run(trial: Trial, env: QuadrupedGymEnv, gait_type: str) -> float:
    try:
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

        # Log parameters to wandb
        wandb.log({
            "trial_number": trial.number,
            "mu": params["mu"],
            "omega_swing": params["omega_swing"],
            "omega_stance": params["omega_stance"],
            "alpha": params["alpha"],
            "des_step_len": params["des_step_len"],
            "sim_duration": params["sim_duration"],
            "timestep": params["timestep"],
            "gait_type": params["gait_type"],
        })

        # Reset the simulation
        env.reset()

        cpg_sim = CPGSimulator(env, params)

        cpg_sim.run()

        total_objective, objective_values = get_objective(env)

        # Log results to wandb (replace print statements for HPC)
        wandb.log({
            # "com_pos_x": stats_manager.com_positions[-1][0],
            # "com_pos_y": stats_manager.com_positions[-1][1],
            # "com_pos_z": stats_manager.com_positions[-1][2],
            "vel_x_objective": objective_values["vel_x_objective"],
            "pos_y_objective": objective_values["pos_y_objective"],
            "total_objective": total_objective,
        })

        return total_objective

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        wandb.log({
            "trial_failed": True,
            "error_message": str(e),
            "objective": -1000.0
        })
        return -1000.0  # Return very negative value for failed trials

def parse_arguments():
    parser = argparse.ArgumentParser(description="Quadruped jump optimization with Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials to run (default: 50)")
    parser.add_argument("--gait-type", type=str, default="TROT", help="Gait type to be optimized")
    parser.add_argument("--project_name", type=str, default="quadruped_cpg", help="Name of the project")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    quadruped_cpg_optimization(args)


if __name__ == "__main__":
    main()