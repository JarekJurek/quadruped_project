# misc
import argparse
import os
from datetime import datetime

import wandb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

# stable baselines 3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# gym environment
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.file_utils import get_latest_model

# utils
from utils.utils import CheckpointCallback


class WandbCallback(BaseCallback):
    """Custom callback for logging to Weights & Biases during training"""
    
    def __init__(self, log_freq=1000, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        # Log metrics every log_freq steps
        if self.n_calls % self.log_freq == 0:
            # Get training metrics from the logger
            if len(self.logger.name_to_value) > 0:
                metrics = {}
                for key, value in self.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value
                
                # Add timestep information
                metrics["timesteps"] = self.num_timesteps
                metrics["n_calls"] = self.n_calls
                
                # Log to wandb
                wandb.log(metrics)
        
        return True


def run_sb3(args):
    # Get worker ID from LSF environment (or default for local testing)
    worker_id = int(os.getenv('LSB_JOBINDEX', '1'))
    
    # Initialize wandb
    wandb.init(
        project=args.project_name,
        name=f"{args.learning_alg}-worker-{worker_id}-{datetime.now().strftime('%m%d%y%H%M%S')}",
        config={
            "worker_id": worker_id,
            "learning_algorithm": args.learning_alg,
            "num_envs": args.num_envs,
            "use_gpu": args.use_gpu,
            "load_existing_model": args.load_nn,
            "total_timesteps": 1000000,  # You might want to make this configurable
        }
    )

    # after implementing, you will want to test how well the agent learns with your MDP: 
    env_configs = {}

    env_configs = {"motor_control_mode":"CPG",
                #    "task_env": "FWD_LOCOMOTION",
                   "task_env": "LR_COURSE_TASK",
                   "observation_space_mode": "LR_COURSE_OBS",
                   "timestep": args.time_step,
                   "max_episode_length": args.max_episode_length,
                   "randomize_cpg_params": args.randomize_cpg_params,}
    
    # Log environment configuration to wandb
    wandb.config.update({"env_configs": env_configs})

    if args.use_gpu and args.learning_alg=="SAC":
        gpu_arg = "auto" 
    else:
        gpu_arg = "cpu"

    if args.load_nn:
        interm_dir = f"{args.save_path}/logs/intermediate_models/{args.project_name}"
        log_dir = interm_dir + '' # add path
        stats_path = os.path.join(log_dir, "vec_normalize.pkl")
        model_name = get_latest_model(log_dir)
        
        wandb.config.update({
            "loaded_model_path": model_name,
            "loaded_stats_path": stats_path
        })

    # directory to save policies and normalization parameters
    save_path = f'{args.save_path}/logs/intermediate_models/{args.project_name}/'+ datetime.now().strftime("%m%d%y%H%M%S") + '/'
    os.makedirs(save_path, exist_ok=True)
    
    # Log save path to wandb
    wandb.config.update({"model_save_path": save_path})

    # checkpoint to save policy network periodically
    checkpoint_callback = CheckpointCallback(save_freq=30000, save_path=save_path,name_prefix='rl_model', verbose=2)
    
    # Create wandb callback
    wandb_callback = WandbCallback(log_freq=1000, verbose=1)

    # create Vectorized gym environment
    env = lambda: QuadrupedGymEnv(**env_configs)  
    env = make_vec_env(env, monitor_dir=save_path,n_envs=args.num_envs)

    # normalize observations to stabilize learning (why?)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

    if args.load_nn:
        env = lambda: QuadrupedGymEnv(**env_configs)
        env = make_vec_env(env, monitor_dir=save_path, n_envs=args.num_envs)
        env = VecNormalize.load(stats_path, env)

    # Multi-layer perceptron (MLP) policy of two layers of size _,_ each with tanh activation function
    policy_kwargs = dict(net_arch=[256,256]) # act_fun=tf.nn.tanh

    # What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    n_steps = 4096 
    learning_rate = lambda f: 1e-4 
    ppo_config = {  "gamma":0.99, 
                    "n_steps": int(n_steps/args.num_envs), 
                    "ent_coef":0.0, 
                    "learning_rate":learning_rate, 
                    "vf_coef":0.5,
                    "max_grad_norm":0.5, 
                    "gae_lambda":0.95, 
                    "batch_size":128,
                    "n_epochs":10, 
                    "clip_range":0.2, 
                    "clip_range_vf":1,
                    "verbose":1, 
                    "tensorboard_log":None, 
                    "_init_setup_model":True, 
                    "policy_kwargs":policy_kwargs,
                    "device": gpu_arg}

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

    # Log hyperparameters to wandb
    if args.learning_alg == "PPO":
        wandb.config.update({"ppo_config": ppo_config})
        model = PPO('MlpPolicy', env, **ppo_config)
    elif args.learning_alg == "SAC":
        wandb.config.update({"sac_config": sac_config})
        model = SAC('MlpPolicy', env, **sac_config)
    else:
        raise ValueError(args.learning_alg + ' not implemented')

    if args.load_nn:
        if args.learning_alg == "PPO":
            model = PPO.load(model_name, env)
        elif args.learning_alg == "SAC":
            model = SAC.load(model_name, env)
        print("\nLoaded model", model_name, "\n")
        wandb.log({"model_loaded": True, "loaded_model_name": model_name})

    # Learn and save (may need to train for longer)
    try:
        model.learn(
            total_timesteps=1000000, 
            log_interval=1,
            callback=[checkpoint_callback, wandb_callback]
        )
        
        # Log successful completion
        wandb.log({"training_completed": True, "final_timesteps": 1000000})
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        wandb.log({"training_failed": True, "error_message": str(e)})
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
        "total_training_steps": 1000000,
        "algorithm_used": args.learning_alg,
        "cpg_h": env.cpg_h_container,
        "cpg_g_c": env.cpg_g_c_container,
        "des_vel_x": env.des_vel_x_container,
    })

    # Finish wandb run
    wandb.finish()
    
    print(f"Worker {worker_id} completed successfully.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Quadruped RL training with Stable Baselines 3")
    parser.add_argument("--gait-type", type=str, default="TROT", help="Gait type to be optimized")
    parser.add_argument("--project-name", type=str, default="quadruped_rl", help="Name of the project")
    
    parser.add_argument("--learning-alg", type=str, default="PPO", choices=["PPO", "SAC"], help="Learning algorithm to use (default: PPO)")
    parser.add_argument("--load-nn", action="store_true", help="Initialize training with a previous model")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of pybullet environments to create for data collection (default: 1)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training (make sure to install all necessary drivers)")
    parser.add_argument("--save-path", type=str, help="Path for storing intermediate models", default=".")
    parser.add_argument("--time_step", type=float, default=0.001, help="time step")
    parser.add_argument("--max_episode_length", type=float, default=20., help="max episode lenght")
    parser.add_argument("--randomize_cpg_params", type=bool, default=True, help="Whether to randomize cpg params")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    run_sb3(args)


if __name__ == "__main__":
    main()