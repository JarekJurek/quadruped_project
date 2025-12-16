# misc
import argparse
import os
import traceback
from datetime import datetime

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# stable baselines 3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

import wandb

# gym environment
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.file_utils import get_latest_model

# utils
from utils.utils import CheckpointCallback


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        verbose: int = 0,
        learning_rate_adaptive=False,
        target_kl=0.01,
        total_timesteps=1000000,
        terrain_difficulty=1,
        max_num_stairs=10,
        difficulty_objective="number",
        max_step_height=0.05,
        terrain=None,
        slope_pitch=0.2,
    ):
        super().__init__(verbose)
        self.learning_rate_adaptive = learning_rate_adaptive
        self.target_kl = target_kl
        self.current_lr = None

        self.total_timesteps_train = total_timesteps
        self.terrain_difficulty_levels = terrain_difficulty
        self.max_num_stairs = max_num_stairs
        self.last_set_value = -1
        self.max_step_height = max_step_height
        self.difficulty_objective = difficulty_objective

        self.slope_pitch = slope_pitch

        self.terrain = terrain

    def _get_current_level(self):
        """Calculates the current difficulty level based on timesteps."""
        if self.terrain_difficulty_levels <= 0:
            return 1

        steps_per_level = self.total_timesteps_train / self.terrain_difficulty_levels

        current_level = int(self.num_timesteps / steps_per_level) + 1

        return min(current_level, self.terrain_difficulty_levels)

    def _update_environment_difficulty(self, current_level, max_val, min_val, param_name, aux_params=None):
        """Updates the environment parameter based on the current level."""
        ratio = current_level / self.terrain_difficulty_levels

        if isinstance(max_val, int):
            new_val = int(ratio * max_val)
        else:
            new_val = ratio * max_val

        new_val = max(min_val, new_val)

        if new_val != self.last_set_value:
            if self.verbose > 0:
                print(
                    f"Curriculum Update at step {self.num_timesteps}: Difficulty Level {current_level}/{self.terrain_difficulty_levels}, {param_name} set to {new_val}"
                )

            self.training_env.set_attr(param_name, new_val)

            if aux_params:
                for k, v in aux_params.items():
                    self.training_env.set_attr(k, v)

            self.last_set_value = new_val

        self.logger.record("train/curriculum_level", current_level)
        self.logger.record(f"train/current_{param_name}", new_val)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.current_lr = self.model.policy.optimizer.param_groups[0]["lr"]

    def _on_rollout_start(self) -> None:
        """
        This is triggered before a new rollout starts, which implies
        the previous training phase just finished.
        """
        if "train/approx_kl" in self.logger.name_to_value and self.learning_rate_adaptive:
            current_kl = self.logger.name_to_value["train/approx_kl"]

            if current_kl > self.target_kl * 2.0:
                self.current_lr = max(1e-5, self.current_lr / 1.5)
                if self.verbose > 0:
                    print(f"KL ({current_kl:.4f}) too high. Reducing LR to {self.current_lr:.6f}")

            elif current_kl < self.target_kl * 0.5:
                self.current_lr = min(1e-2, self.current_lr * 1.5)
                if self.verbose > 0:
                    print(f"KL ({current_kl:.4f}) too low. Increasing LR to {self.current_lr:.6f}")

            self._update_learning_rate(self.current_lr)
            self.logger.record("train/learning_rate_adaptive", self.current_lr)

        if self.terrain_difficulty_levels > 0:
            current_level = self._get_current_level()

            if self.terrain == "STAIRS":
                if self.difficulty_objective == "number" and self.max_num_stairs > 0:
                    self._update_environment_difficulty(
                        current_level=current_level, max_val=self.max_num_stairs, min_val=1, param_name="num_stairs"
                    )
                elif self.difficulty_objective == "height" and self.max_step_height > 0:
                    self._update_environment_difficulty(
                        current_level=current_level,
                        max_val=self.max_step_height,
                        min_val=0.01,
                        param_name="stair_height",
                        aux_params={"num_stairs": 1},
                    )
            elif self.terrain == "SLOPES":
                self._update_environment_difficulty(
                    current_level=current_level,
                    max_val=self.slope_pitch,
                    min_val=0.01,
                    param_name="slope_pitch",
                    aux_params={"num_stairs": 1},
                )

    def _update_learning_rate(self, new_lr):
        self.model.learning_rate = new_lr  # Update SB3 internal tracker
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = new_lr

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        cpg_h_container = self.training_env.get_attr("cpg_h_container")
        cpg_g_c_container = self.training_env.get_attr("cpg_g_c_container")
        des_vel_x_container = self.training_env.get_attr("des_vel_x_container")
        wandb.log({"cpg_h_container": cpg_h_container, "cpg_g_c_container": cpg_g_c_container, "des_vel_x_container": des_vel_x_container})

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def _setup_wandb_and_directories(args):
    """Setup wandb logging and create necessary directories."""
    worker_id = int(os.getenv("LSB_JOBINDEX", "1"))
    timestamp = datetime.now().strftime("%m%d%y%H%M%S")

    # Setup wandb directory
    wandb_dir = os.path.join(args.save_path, "wandb_runs", f"{args.project_name}-worker-{worker_id}-{timestamp}")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir

    # Setup run name
    run_name = f"{args.run_name or args.learning_alg}-worker-{worker_id}-{timestamp}"

    # Setup model save directory
    if args.run_name:
        save_path = f"{args.save_path}/logs/intermediate_models/{args.project_name}/{args.run_name}/{timestamp}/"
    else:
        save_path = f"{args.save_path}/logs/intermediate_models/{args.project_name}/{timestamp}/"
    os.makedirs(save_path, exist_ok=True)

    return worker_id, timestamp, wandb_dir, run_name, save_path


def _initialize_wandb(args, worker_id, run_name, wandb_dir, save_path):
    """Initialize wandb with configuration."""
    wandb.init(
        project=args.project_name,
        name=run_name,
        dir=wandb_dir,
        sync_tensorboard=True,
        config={
            "worker_id": worker_id,
            "learning_algorithm": args.learning_alg,
            "num_envs": args.num_envs,
            "use_gpu": args.use_gpu,
            "load_existing_model": args.load_nn,
            "total_timesteps": args.total_timesteps,
            "control_frequency": args.control_frequency,
            "learning_rate_adaptive": args.learning_rate_adaptive,
            "model_save_path": save_path,
        },
    )


def _build_env_config(args):
    """Build environment configuration dictionary."""
    action_repeat = calculate_action_repeat(args)

    env_config = {
        "motor_control_mode": args.motor_control_mode,
        "task_env": args.task_env,
        "observation_space_mode": args.observation_space_mode,
        "time_step": args.time_step,
        "max_episode_length": args.max_episode_length,
        "randomize_cpg_params": args.randomize_cpg_params,
        "action_repeat": action_repeat,
        "des_vel_x": args.des_x_vel,
        "des_vel_x_min": args.des_vel_x_min,
        "des_vel_x_max": args.des_vel_x_max,
        "terrain": args.terrain,
        "randomize_velocity_command": args.randomize_velocity_command,
        "num_stairs": args.num_stairs,
        "stair_height": args.stair_height,
        "stair_width": args.stair_width,
        "des_h": args.des_h,
        "des_g_c": args.des_g_c,
        "vel_tracking_weight": args.vel_tracking_weight,
        "drift_weight": args.drift_weight,
        "yaw_weight": args.yaw_weight,
        "orientation_weight": args.orientation_weight,
        "survival_weight": args.survival_weight,
        "height_weight": args.height_weight,
        "enable_vmc": args.enable_vmc,
        "k_vmc": args.k_vmc,
        "kp": args.kp,
        "kd": args.kd,
        "disable_drift": args.disable_drift,
        "disable_yaw": args.disable_yaw,
        "disable_orientation": args.disable_orientation,
        "disable_energy": args.disable_energy,
        "add_noise": args.add_noise,
    }

    wandb.config.update({"env_configs": env_config})
    return env_config


def _create_environment(env_config, args, save_path):
    """Create and configure the training environment."""

    def env_fn():
        return QuadrupedGymEnv(**env_config)

    env = make_vec_env(env_fn, monitor_dir=save_path, n_envs=args.num_envs, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.0)
    return env


def _get_model_configs(args):
    """Get model configuration dictionaries for PPO and SAC."""
    gpu_arg = "auto" if (args.use_gpu and args.learning_alg == "SAC") else "cpu"

    policy_kwargs = dict(net_arch=[512, 256, 128], activation_fn=torch.nn.modules.activation.ELU)

    minibatch_size = args.batch_size // args.n_mini_batch

    ppo_config = {
        "gamma": args.discount,
        "n_steps": int(args.batch_size / args.num_envs),
        "ent_coef": args.ent_coef,
        "learning_rate": args.learning_rate,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gae_lambda": args.gae_discount,
        "batch_size": int(minibatch_size),
        "n_epochs": args.n_epochs,
        "clip_range": args.clip_range,
        "clip_range_vf": 1,
        "verbose": 1,
        "tensorboard_log": args.save_path,
        "_init_setup_model": True,
        "policy_kwargs": policy_kwargs,
        "target_kl": args.des_kl_divergence,
        "device": gpu_arg,
        "use_sde": args.use_sde,
    }

    sac_config = {
        "learning_rate": 1e-4,
        "buffer_size": 300000,
        "batch_size": 256,
        "ent_coef": "auto",
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 10000,
        "verbose": 1,
        "tensorboard_log": None,
        "policy_kwargs": policy_kwargs,
        "seed": None,
        "device": gpu_arg,
    }

    return ppo_config, sac_config, gpu_arg


def _load_or_create_model(args, env, ppo_config, sac_config, gpu_arg):
    """Load existing model or create new one based on arguments."""
    if args.load_nn:
        return _load_existing_model(args, env, gpu_arg)
    else:
        return _create_new_model(args, env, ppo_config, sac_config)


def _load_existing_model(args, env, gpu_arg):
    """Load an existing model from disk."""
    try:
        log_dir = args.load_model_path
        stats_path = os.path.join(log_dir, "vec_normalize.pkl")
        model_name = get_latest_model(log_dir)

        wandb.config.update({"loaded_model_path": model_name, "loaded_stats_path": stats_path})

        # Load VecNormalize statistics
        env = VecNormalize.load(stats_path, env)
        env.training = True
        env.norm_reward = False

        # Load the model
        if args.learning_alg == "PPO":
            model = PPO.load(model_name, env, device=gpu_arg)
        elif args.learning_alg == "SAC":
            model = SAC.load(model_name, env, device=gpu_arg)
        else:
            raise ValueError(f"{args.learning_alg} not implemented")

        print(f"\nLoaded model: {model_name}\n")
        wandb.log({"model_loaded": True, "loaded_model_name": model_name})
        return model, env

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Failed to load pretrained model. Training from scratch. Error: {e}\n{tb_str}")
        wandb.log({"model_loading_failed": True, "error_message": str(e), "traceback": tb_str})
        raise e


def _create_new_model(args, env, ppo_config, sac_config):
    """Create a new model from scratch."""
    if args.learning_alg == "PPO":
        wandb.config.update({"ppo_config": ppo_config})
        model = PPO("MlpPolicy", env, **ppo_config)
    elif args.learning_alg == "SAC":
        wandb.config.update({"sac_config": sac_config})
        model = SAC("MlpPolicy", env, **sac_config)
    else:
        raise ValueError(f"{args.learning_alg} not implemented")

    return model, env


def _save_final_model(model, env, save_path, args):
    """Save the final trained model and normalization stats."""
    final_model_path = os.path.join(save_path, "rl_model")
    final_normalize_path = os.path.join(save_path, "vec_normalize.pkl")

    model.save(final_model_path)
    env.save(final_normalize_path)

    wandb.log({"final_model_saved": True, "final_model_path": final_model_path, "final_normalize_path": final_normalize_path})

    # Save replay buffer for SAC
    if args.learning_alg == "SAC":
        replay_buffer_path = os.path.join(save_path, "off_policy_replay_buffer")
        model.save_replay_buffer(replay_buffer_path)
        wandb.log({"replay_buffer_saved": True, "replay_buffer_path": replay_buffer_path})


def _log_training_completion(args, env, worker_id):
    """Log final training completion metrics."""
    wandb.log(
        {
            "worker_completed": True,
            "total_training_steps": args.total_timesteps,
            "algorithm_used": args.learning_alg,
            "cpg_h": env.get_attr("cpg_h_container"),
            "cpg_g_c": env.get_attr("cpg_g_c_container"),
            "des_vel_x": env.get_attr("des_vel_x_container"),
        }
    )

    wandb.finish()
    print(f"Worker {worker_id} completed successfully.")


def run_sb3(args):
    """Main training function for quadruped RL with stable-baselines3."""
    # Setup directories and wandb
    worker_id, timestamp, wandb_dir, run_name, save_path = _setup_wandb_and_directories(args)
    _initialize_wandb(args, worker_id, run_name, wandb_dir, save_path)

    # Build environment configuration
    env_config = _build_env_config(args)

    # Create training environment
    env = _create_environment(env_config, args, save_path)

    # Setup model configurations
    ppo_config, sac_config, gpu_arg = _get_model_configs(args)

    # Load or create model
    model, env = _load_or_create_model(args, env, ppo_config, sac_config, gpu_arg)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(save_freq=30000, save_path=save_path, name_prefix="rl_model", verbose=2)

    custom_callback = CustomCallback(
        verbose=2,
        learning_rate_adaptive=args.learning_rate_adaptive,
        target_kl=args.des_kl_divergence,
        total_timesteps=args.total_timesteps,
        terrain_difficulty=args.terrain_difficulty,
        max_num_stairs=args.num_stairs,
        difficulty_objective=args.difficulty_objective,
        max_step_height=args.stair_height,
        terrain=args.terrain,
        slope_pitch=args.slope_pitch,
    )

    # Train the model
    try:
        model.learn(total_timesteps=args.total_timesteps, log_interval=1, callback=[checkpoint_callback, WandbCallback(), custom_callback])

        wandb.log({"training_completed": True, "final_timesteps": args.total_timesteps})

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Training failed with error: {e}\n{tb_str}")
        wandb.log({"training_failed": True, "error_message": str(e), "traceback": tb_str})
        wandb.finish()
        raise e

    # Save final model and log completion
    _save_final_model(model, env, save_path, args)
    _log_training_completion(args, env, worker_id)


def calculate_action_repeat(args):
    """
    Mimics the delay in the control of the policy on an actual robot.
    """
    control_frequency = args.control_frequency
    sim_frequency = 1.0 / args.time_step
    action_repeat = int(sim_frequency // control_frequency)

    # Validate that frequencies are compatible
    if sim_frequency % control_frequency != 0:
        print(f"sim_freq ({sim_frequency}) must be divisible by control_freq ({control_frequency}). Setting action repeat to 10.")
        action_repeat = 10
    return action_repeat


def parse_arguments():
    parser = argparse.ArgumentParser(description="Quadruped RL Training with Stable Baselines 3")

    # Project and Run Configuration
    parser.add_argument("--project-name", type=str, default="quadruped_rl", help="Name of the project for logging and organization")
    parser.add_argument("--run-name", type=str, default=None, help="Custom name for this training run")
    parser.add_argument("--save-path", type=str, default=".", help="Root path for storing models and logs")

    # Learning Algorithm Configuration
    parser.add_argument("--learning-alg", type=str, default="PPO", choices=["PPO", "SAC"], help="Learning algorithm to use")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training (requires CUDA)")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments for data collection")

    # Model Loading Configuration
    parser.add_argument("--load-nn", action="store_true", help="Load and continue training from existing model")
    parser.add_argument("--load_model_path", type=str, default=".", help="Path to pretrained model directory")

    # Environment Configuration
    parser.add_argument(
        "--motor_control_mode", type=str, default="CPG", choices=["CPG", "PD", "TORQUE", "CARTESIAN_PD"], help="Motor control strategy"
    )
    parser.add_argument(
        "--task_env",
        type=str,
        default="LR_COURSE_TASK",
        choices=["LR_COURSE_TASK", "FLAGRUN", "FWD_LOCOMOTION", "FWD_CUSTOM", "FWD_BASIC", "ETH", "CPG_RL", "FWD_CUSTOM_OLD"],
        help="Task environment type",
    )
    parser.add_argument(
        "--observation_space_mode",
        type=str,
        default="LR_COURSE_OBS_EXTENDED",
        choices=["DEFAULT", "LR_COURSE_OBS", "LR_COURSE_OBS_EXTENDED"],
        help="Observation space configuration",
    )

    # Simulation Parameters
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--time_step", type=float, default=0.001, help="Physics simulation time step (seconds)")
    parser.add_argument("--max_episode_length", type=float, default=10.0, help="Maximum episode length (seconds)")
    parser.add_argument("--control_frequency", type=int, default=100, help="Policy control frequency (Hz)")

    # Robot Control Parameters
    parser.add_argument("--des_x_vel", type=float, default=0.6, help="Desired forward velocity (m/s)")
    parser.add_argument("--des_vel_x_min", type=float, default=0.3, help="Minimum desired velocity for randomization (m/s)")
    parser.add_argument("--des_vel_x_max", type=float, default=0.8, help="Maximum desired velocity for randomization (m/s)")
    parser.add_argument("--des_h", type=float, default=0.25, help="Desired body height (m)")
    parser.add_argument("--des_g_c", type=float, default=0.08, help="Desired ground clearance during swing phase (m)")

    # CPG and Motor Control
    parser.add_argument("--randomize_cpg_params", type=bool, default=True, help="Randomize CPG parameters during training")
    parser.add_argument("--randomize_velocity_command", action="store_true", help="Randomize velocity commands during training")
    parser.add_argument("--enable_vmc", action="store_true", help="Enable Virtual Model Control")
    parser.add_argument("--k_vmc", type=float, default=250.0, help="VMC stiffness parameter")
    parser.add_argument("--kp", type=float, default=None, help="PD controller proportional gain")
    parser.add_argument("--kd", type=float, default=None, help="PD controller derivative gain")

    # Reward Function Weights
    parser.add_argument("--vel_tracking_weight", type=float, default=1.0, help="Weight for velocity tracking reward")
    parser.add_argument("--drift_weight", type=float, default=0.5, help="Weight for lateral drift penalty")
    parser.add_argument("--yaw_weight", type=float, default=0.5, help="Weight for yaw orientation penalty")
    parser.add_argument("--orientation_weight", type=float, default=1.0, help="Weight for body orientation penalty")
    parser.add_argument("--survival_weight", type=float, default=1.0, help="Weight for survival reward")
    parser.add_argument("--height_weight", type=float, default=1.0, help="Weight for height maintenance reward")

    # Reward Component Toggles
    parser.add_argument("--disable_drift", action="store_true", help="Disable drift penalty in reward function")
    parser.add_argument("--disable_yaw", action="store_true", help="Disable yaw penalty in reward function")
    parser.add_argument("--disable_orientation", action="store_true", help="Disable orientation penalty in reward function")
    parser.add_argument("--disable_energy", action="store_true", help="Disable energy penalty in reward function")

    # Environment Variations
    parser.add_argument("--add_noise", action="store_true", help="Add noise to observations and dynamics")

    # Terrain Configuration
    parser.add_argument(
        "--terrain", type=str, default="NONE", choices=["STAIRS", "SLOPES", "GAPS", "RANDOM", "NONE"], help="Terrain type for training"
    )
    parser.add_argument("--terrain_difficulty", type=int, default=1, help="Number of curriculum difficulty levels")
    parser.add_argument(
        "--difficulty_objective",
        type=str,
        default="number",
        choices=["number", "height"],
        help="Curriculum objective (number of obstacles or height)",
    )
    parser.add_argument("--num_stairs", type=int, default=12, help="Number of stairs in stair terrain")
    parser.add_argument("--stair_height", type=float, default=0.05, help="Height of each stair (m)")
    parser.add_argument("--stair_width", type=float, default=0.25, help="Width of each stair (m)")
    parser.add_argument("--slope_pitch", type=float, default=0.2, help="Slope angle for slope terrain (radians)")

    # PPO Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8192, help="Rollout buffer size for PPO")
    parser.add_argument("--n_mini_batch", type=int, default=4, help="Number of mini-batches per update")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="Entropy coefficient for exploration")
    parser.add_argument("--gae_discount", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of optimization epochs per update")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--use_sde", type=bool, default=False, help="Use State Dependent Exploration")

    # Learning Rate Configuration
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--learning_rate_adaptive", action="store_true", help="Enable adaptive learning rate based on KL divergence")
    parser.add_argument("--des_kl_divergence", type=float, default=0.01, help="Target KL divergence for adaptive learning rate")

    return parser.parse_args()


def main():
    args = parse_arguments()
    run_sb3(args)


if __name__ == "__main__":
    main()
