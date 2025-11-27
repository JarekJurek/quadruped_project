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
    def __init__(self, verbose: int = 0, learning_rate_adaptive=False):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.learning_rate_adaptive = learning_rate_adaptive

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        This is triggered before a new rollout starts, which implies
        the previous training phase just finished.
        """
        # 1. Access the logged KL divergence from the previous update
        # We use 'train/approx_kl' which SB3 logs automatically
        if "train/approx_kl" in self.logger.name_to_value and self.learning_rate_adaptive:
            current_kl = self.logger.name_to_value["train/approx_kl"]
            
            # 2. Update Learning Rate based on Target KL logic
            if current_kl > self.target_kl * 2.0:
                self.current_lr = max(1e-5, self.current_lr / 1.5)
                if self.verbose > 0:
                    print(f"KL ({current_kl:.4f}) too high. Reducing LR to {self.current_lr:.6f}")
            
            elif current_kl < self.target_kl * 0.5:
                self.current_lr = min(1e-2, self.current_lr * 1.5)
                if self.verbose > 0:
                    print(f"KL ({current_kl:.4f}) too low. Increasing LR to {self.current_lr:.6f}")

            # 3. Apply the new learning rate to the optimizer
            self._update_learning_rate(self.current_lr)
            self.logger.record("train/learning_rate_adaptive", self.current_lr)

    def _update_learning_rate(self, new_lr):
        self.model.learning_rate = new_lr # Update SB3 internal tracker
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
        wandb.log({
            "cpg_h_container": cpg_h_container,
            "cpg_g_c_container": cpg_g_c_container,
            "des_vel_x_container": des_vel_x_container
        })

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def run_sb3(args):
    # Get worker ID from LSF environment (or default for local testing)
    worker_id = int(os.getenv('LSB_JOBINDEX', '1'))

    timestamp = datetime.now().strftime('%m%d%y%H%M%S')
    wandb_dir = os.path.join(args.save_path, "wandb_runs", f"{args.project_name}-worker-{worker_id}-{timestamp}")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir

    if args.run_name is not None:
        run_name = f"{args.run_name}-worker-{worker_id}-{timestamp}"
    else:
        run_name = f"{args.learning_alg}-worker-{worker_id}-{timestamp}"
    
    # Initialize wandb
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
        }
    )

    action_repeat = calculate_action_repeat(args)
    
    env_configs = {"motor_control_mode":args.motor_control_mode,
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
                   "randomize_velocity_command": args.randomize_velocity_command}
    
    # Log environment configuration to wandb
    wandb.config.update({"env_configs": env_configs})

    if args.use_gpu and args.learning_alg=="SAC":
        gpu_arg = "auto" 
    else:
        gpu_arg = "cpu"

    # directory to save policies and normalization parameters
    if args.run_name is not None:
        save_path = f'{args.save_path}/logs/intermediate_models/{args.project_name}/{args.run_name}/'+ timestamp + '/'
    else:
        save_path = f'{args.save_path}/logs/intermediate_models/{args.project_name}/'+ timestamp + '/'
    os.makedirs(save_path, exist_ok=True)
    
    # Log save path to wandb
    wandb.config.update({"model_save_path": save_path})

    # checkpoint to save policy network periodically
    checkpoint_callback = CheckpointCallback(save_freq=30000, save_path=save_path,name_prefix='rl_model', verbose=2)
    
    # Create wandb callback
    # wandb_callback = WandbCallback(log_freq=1000, verbose=1)
    # wandb_callback = WandbCallback(
    #     gradient_save_freq=100,
    #     # verbose=1,
    # )
    custom_callback = CustomCallback(verbose=2, learning_rate_adaptive=args.learning_rate_adaptive)

    # create Vectorized gym environment
    env = lambda: QuadrupedGymEnv(**env_configs)  
    env = make_vec_env(env, monitor_dir=save_path,n_envs=args.num_envs, vec_env_cls=SubprocVecEnv)

    # normalize observations to stabilize learning (why?)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)



    # Multi-layer perceptron (MLP) policy of two layers of size _,_ each with tanh activation function
    # policy_kwargs = dict(net_arch=[256,256]) # act_fun=tf.nn.tanh
    policy_kwargs = dict(net_arch=[512, 256, 128], activation_fn=torch.nn.modules.activation.ELU)

    # What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    # learning_rate = lambda f: 1e-4

    minibatch_size = args.batch_size // args.n_mini_batch

    ppo_config = {
        "gamma": args.discount,
        "n_steps": int(args.batch_size / args.num_envs), # steps per env
        "ent_coef": args.ent_coef,
        "learning_rate": args.learning_rate,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gae_lambda": args.gae_discount,
        "batch_size": int(minibatch_size), # SB3 minibatch size
        "n_epochs": args.n_epochs,
        "clip_range": args.clip_range,
        "clip_range_vf": 1,
        "verbose": 1,
        "tensorboard_log": args.save_path,
        "_init_setup_model": True,
        "policy_kwargs": policy_kwargs,
        "target_kl": args.des_kl_divergence,
        "device": gpu_arg,
        "use_sde": args.use_sde
    }

    # What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
    sac_config={"learning_rate":1e-4,
                "buffer_size":300000,
                "batch_size":256,
                "ent_coef":'auto', 
                "gamma":0.99, 
                "tau":0.005,
                "train_freq":1, 
                "gradient_steps":1,
                "learning_starts": 10000,
                "verbose":1, 
                "tensorboard_log":None,
                "policy_kwargs": policy_kwargs,
                "seed":None, 
                "device": gpu_arg}

    # Load model if specified
    if args.load_nn:
        try:
            log_dir = args.load_model_path
            stats_path = os.path.join(log_dir, "vec_normalize.pkl")
            model_name = get_latest_model(log_dir)
            
            wandb.config.update({
                "loaded_model_path": model_name,
                "loaded_stats_path": stats_path
            })

            # Load VecNormalize statistics
            env = VecNormalize.load(stats_path, env)
            env.training = True
            env.norm_reward = False

            # Load the model based on the specified algorithm
            if args.learning_alg == "PPO":
                model = PPO.load(model_name, env, device=gpu_arg)
            elif args.learning_alg == "SAC":
                model = SAC.load(model_name, env, device=gpu_arg)
            else:
                raise ValueError(f"{args.learning_alg} not implemented")
            
            print(f"\nLoaded model: {model_name}\n")
            wandb.log({"model_loaded": True, "loaded_model_name": model_name})
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Failed to load pretrained model. Training from scratch. Error: {e}\n{tb_str}")
            wandb.log({"model_loading_failed": True, "error_message": str(e), "traceback": tb_str})
            # Create new model if loading failed
            if args.learning_alg == "PPO":
                wandb.config.update({"ppo_config": ppo_config})
                model = PPO('MlpPolicy', env, **ppo_config)
            elif args.learning_alg == "SAC":
                wandb.config.update({"sac_config": sac_config})
                model = SAC('MlpPolicy', env, **sac_config)
            else:
                raise ValueError(args.learning_alg + ' not implemented')
    #Create new model
    else:
        if args.learning_alg == "PPO":
            wandb.config.update({"ppo_config": ppo_config})
            model = PPO('MlpPolicy', env, **ppo_config)
        elif args.learning_alg == "SAC":
            wandb.config.update({"sac_config": sac_config})
            model = SAC('MlpPolicy', env, **sac_config)
        else:
            raise ValueError(args.learning_alg + ' not implemented')

    # Learn and save (may need to train for longer)
    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            log_interval=1,
            # callback=[checkpoint_callback, wandb_callback]
            callback=[checkpoint_callback, WandbCallback(), custom_callback]
        )
        
        # Log successful completion
        wandb.log({"training_completed": True, "final_timesteps": args.total_timesteps})
        
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Training failed with error: {e}\n{tb_str}")
        wandb.log({"training_failed": True, "error_message": str(e), "traceback": tb_str})
        wandb.finish()
        raise e

    # Don't forget to save the VecNormalize statistics when saving the agent
    final_model_path = os.path.join(save_path, "rl_model")
    final_normalize_path = os.path.join(save_path, "vec_normalize.pkl")
    
    model.save(final_model_path) 
    env.save(final_normalize_path)
    
    # Log final model paths
    wandb.log({
        "final_model_saved": True,
        "final_model_path": final_model_path,
        "final_normalize_path": final_normalize_path
    })

    if args.learning_alg == "SAC": # save replay buffer 
        replay_buffer_path = os.path.join(save_path,"off_policy_replay_buffer")
        model.save_replay_buffer(replay_buffer_path)
        wandb.log({"replay_buffer_saved": True, "replay_buffer_path": replay_buffer_path})

    # Log final summary
    wandb.log({
        "worker_completed": True,
        "total_training_steps": args.total_timesteps,
        "algorithm_used": args.learning_alg,
        "cpg_h": env.get_attr("cpg_h_container"),
        "cpg_g_c": env.get_attr("cpg_g_c_container"),
        "des_vel_x": env.get_attr("des_vel_x_container"),
    })

    # Finish wandb run
    wandb.finish()
    
    print(f"Worker {worker_id} completed successfully.")

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
    parser = argparse.ArgumentParser(description="Quadruped RL training with Stable Baselines 3")
    parser.add_argument("--project-name", type=str, default="quadruped_rl", help="Name of the project")
    parser.add_argument("--run-name", type=str, default=None, help="Name of the run")
    
    parser.add_argument("--learning-alg", type=str, default="PPO", choices=["PPO", "SAC"], help="Learning algorithm to use (default: PPO)")
    parser.add_argument("--motor_control_mode", type=str, default="CPG", choices=["CPG", "PD","TORQUE", "CARTESIAN_PD"], help="Motor control mode")
    parser.add_argument("--observation_space_mode", type=str, default="LR_COURSE_OBS_EXTENDED", choices=["DEFAULT", "LR_COURSE_OBS", "LR_COURSE_OBS_EXTENDED"], help="Observation space mode")
    parser.add_argument("--task_env", type=str, default="LR_COURSE_TASK", choices=["LR_COURSE_TASK", "FLAGRUN","FWD_LOCOMOTION", "FWD_CUSTOM", "FWD_BASIC"], help="Task to be executed")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of pybullet environments to create for data collection (default: 1)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training (make sure to install all necessary drivers)")
    parser.add_argument("--save-path", type=str, help="Path for storing intermediate models", default=".")

    parser.add_argument("--load-nn", action="store_true", help="Initialize training with a previous model")
    parser.add_argument("--load_model_path", type=str, help="Path for loading pretrained model", default=".")

    parser.add_argument("--des_x_vel", type=float, default=0.4, help="desired linear velocity x axis")
    parser.add_argument("--des_vel_x_min", type=float, default=0.3, help="desired linear velocity x axis")
    parser.add_argument("--des_vel_x_max", type=float, default=0.8, help="desired linear velocity x axis")

    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps")
    parser.add_argument("--time_step", type=float, default=0.001, help="time step, for CPG_RL 0.01 s")
    parser.add_argument("--max_episode_length", type=float, default=10., help="max episode lenght in seconds in CPG_RL 20.0 s")
    parser.add_argument("--randomize_cpg_params", type=bool, default=True, help="Whether to randomize cpg params")
    parser.add_argument("--control_frequency", type=int, default=100, help="The control frequency of the policy [Hz]")

    parser.add_argument("--randomize_velocity_command", action="store_true", help="Whether to randomize velocity commands")

    parser.add_argument("--terrain", type=str, default="NONE", choices=["STAIRS", "SLOPES", "GAPS", "RANDOM", "NONE"], help="Terrain, obstacles")
    parser.add_argument("--terrain_difficulty", type=int, default=5, help="Levels of difficulty of obstacles")

    # PPO Hyperparams
    parser.add_argument("--batch_size", type=int, default=8192, help="Size of rollout / batch size")
    parser.add_argument("--n_mini_batch", type=int, default=4, help="Number of minibatch")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor in PPO (gamma)")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="Entropy coefficient in PPO in CPG-RL 0.01")
    parser.add_argument("--gae_discount", type=float, default=0.95, help="GAE discount factor in PPO")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs in PPO in CPG-RL 5")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range in PPO")
    parser.add_argument("--use_sde", type=bool, default=False, help="Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--learning_rate_adaptive", action="store_true", help="Whether to ude adaptive learning rate")
    parser.add_argument("--des_kl_divergence", type=float, default=0.01, help="Desired KL divergence in PPOin CPG-RL 0.01")    

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    run_sb3(args)


if __name__ == "__main__":
    main()