import argparse
import os
from functools import partial

import optuna
from optuna.trial import Trial

import wandb
from env.simulation import QuadSimulator, SimulationOptions
from quadruped_jump import quadruped_jump
from quadruped_jump_opt import get_motion_mode, get_objective

N_LEGS = 4
N_JOINTS = 3


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


def quadruped_jump_optimization(args):
    # Get worker ID from LSF environment (or default for local testing)
    worker_id = int(os.getenv('LSB_JOBINDEX', '1'))

    n_trials = args.n_trials

    motion_mode = get_motion_mode(args)
    
    # Initialize wandb with unique run name for each worker
    wandb.init(
        project=args.project_name,
        name=f"worker-{worker_id}",
        config={
            "worker_id": worker_id,
            "optimization_method": "optuna-distributed",
            "sampler": "TPE",
            "n_trials": n_trials,
            "motion_mode": motion_mode
        }
    )
    
    print(f"Starting worker {worker_id}")
    
    # Initialize simulation - NO RENDERING for HPC
    sim_options = SimulationOptions(
        on_rack=False,  # Whether to suspend the robot in the air (helpful for debugging)
        render=False,  # CRITICAL: Whether to use the GUI visualizer (must be False for HPC)
        record_video=False,  # Whether to record a video to file (needs render=True)
        tracking_camera=False,  # Whether the camera follows the robot (False for HPC)
    )
    simulator = QuadSimulator(sim_options)

    # Use shared database storage for distributed optimization
    storage_path = os.path.expanduser(f"~/{args.project_name}.db")
    storage = f"sqlite:///{storage_path}"
    
    # Create wandb callback for optuna
    wandbc = WeightsAndBiasesCallback(metric_name="objective_value")
    
    # Create a maximization problem with distributed storage
    objective = partial(evaluate_jumping, simulator=simulator, motion_mode=motion_mode)
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
    simulator.close()

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


def evaluate_jumping(trial: Trial, simulator: QuadSimulator, motion_mode: str) -> float:
    try:
        # params = {
        #     "kpCartesian_1": trial.suggest_float(name="kpCartesian_1", low=100.0, high=1000.0),
        #     "kpCartesian_2": trial.suggest_float(name="kpCartesian_2", low=100.0, high=1000.0),
        #     "kpCartesian_3": trial.suggest_float(name="kpCartesian_3", low=100.0, high=1000.0),
        #     "kdCartesian_1": trial.suggest_float(name="kdCartesian_1", low=10.0, high=300.0),
        #     "kdCartesian_2": trial.suggest_float(name="kdCartesian_2", low=10.0, high=300.0),
        #     "kdCartesian_3": trial.suggest_float(name="kdCartesian_3", low=10.0, high=300.0),
        #     "k_vmc": trial.suggest_float(name="k_vmc", low=100.0, high=400.0),
        #     "f0": trial.suggest_float(name="f0", low=0.5, high=4.),
        #     "f1": trial.suggest_float(name="f1", low=0.5, high=4.),
        #     "Fx": trial.suggest_float(name="Fx", low=30.0, high=200.0),
        #     "Fy": trial.suggest_float(name="Fy", low=10.0, high=50.0),
        #     "Fz": trial.suggest_float(name="Fz", low=30.0, high=300.0),
        #     "nom_x": trial.suggest_float(name="nom_x", low=-0.15, high=0.15),
        #     "nom_y": trial.suggest_float(name="nom_y", low=simulator.config["com_hip_offset_y"] - 0.15, high=simulator.config["com_hip_offset_y"] + 0.15),
        #     "nom_z": trial.suggest_float(name="nom_z", low=-0.25-0.1, high=-0.25+0.15),
        #     "enable_force_profile": True,
        #     "enable_gravity_compensation": True,
        #     "enable_virtual_model": True,
        #     "make_plots": False,
        #     "n_jumps": 3,
        #     "jump_duration": 2.0,
        #     "timestep": 1e-3,
        # }
        params = {
            "kpCartesian_1": 500.,
            "kpCartesian_2": 400.0,
            "kpCartesian_3": 400.0,
            "kdCartesian_1": 30.0,
            "kdCartesian_2": 50.0,
            "kdCartesian_3": 40.0,
            "k_vmc": 250.,
            "f0": trial.suggest_float(name="f0", low=1.5, high=2.5),  # around 2.0
            "f1": trial.suggest_float(name="f1", low=0.3, high=0.7),  # around 0.5
            "Fx": trial.suggest_float(name="Fx", low=90.0, high=130.0),  # around 110
            "Fy": 0.0,
            "Fz": trial.suggest_float(name="Fz", low=120.0, high=180.0),  # around 150
            "nom_x": 0.0,
            "nom_y": simulator.config["com_hip_offset_y"],
            "nom_z": -0.25,
            "enable_force_profile": True,
            "enable_gravity_compensation": True,
            "enable_virtual_model": True,
            "n_jumps": 10,
            "jump_duration": 2.0,
            "timestep": 1e-3,
            "make_plots": False,
            "save_to_csv": False,
        }

        # Log parameters to wandb
        wandb.log({
            "trial_number": trial.number,
            "kpCartesian_1": params["kpCartesian_1"],
            "kpCartesian_3": params["kpCartesian_3"],
            "kdCartesian_1": params["kdCartesian_1"],
            "kdCartesian_3": params["kdCartesian_3"],
            "k_vmc": params["k_vmc"],
            "f0": params["f0"],
            "f1": params["f1"],
            "Fx": params["Fx"],
            "Fz": params["Fz"],
        })

        # Reset the simulation
        simulator.reset()

        stats_manager = quadruped_jump(simulator, params, motion_mode)

        total_objective, objective_values = get_objective(simulator, stats_manager)

        # Log results to wandb (replace print statements for HPC)
        wandb.log({
            "com_pos_x": stats_manager.com_positions[-1][0],
            "com_pos_y": stats_manager.com_positions[-1][1],
            "com_pos_z": stats_manager.com_positions[-1][2],
            "pos_x_objective": objective_values["pos_x_objective"],
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
    parser.add_argument("--motion-type", type=str, default="fwd_jump", help="Motion type to be optimized")
    parser.add_argument("--project_name", type=str, default="quadruped-jump", help="Name of the project")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    quadruped_jump_optimization(args)


if __name__ == "__main__":
    main()