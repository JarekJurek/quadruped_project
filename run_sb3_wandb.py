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


def run_sb3(args):
    # Get worker ID from LSF environment (or default for local testing)
    worker_id = int(os.getenv('LSB_JOBINDEX', '1'))

    timestamp = datetime.now().strftime('%m%d%y%H%M%S')
    wandb_dir = os.path.join(args.save_path, "wandb_runs", f"{args.project_name}-worker-{worker_id}-{timestamp}")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir
    
    # Initialize wandb
    wandb.init(
        project=args.project_name,
        name=f"{args.learning_alg}-worker-{worker_id}-{timestamp}",
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
                   "des_vel_x": args.des_x_vel}
    
    # Log environment configuration to wandb
    wandb.config.update({"env_configs": env_configs})

    if args.use_gpu and args.learning_alg=="SAC":
        gpu_arg = "auto" 
    else:
        gpu_arg = "cpu"

    # directory to save policies and normalization parameters
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

    # create Vectorized gym environment
    env = lambda: QuadrupedGymEnv(**env_configs)  
    env = make_vec_env(env, monitor_dir=save_path,n_envs=args.num_envs, vec_env_cls=SubprocVecEnv)

    # normalize observations to stabilize learning (why?)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

    # Multi-layer perceptron (MLP) policy of two layers of size _,_ each with tanh activation function
    # policy_kwargs = dict(net_arch=[256,256]) # act_fun=tf.nn.tanh
    policy_kwargs = dict(net_arch=[512, 256, 128], activation_fn=torch.nn.modules.activation.ELU)

    # What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    learning_rate = lambda f: 1e-4

    minibatch_size = args.batch_size // args.n_mini_batch

    ppo_config = {
        "gamma": args.discount,
        "n_steps": int(args.batch_size / args.num_envs), # steps per env
        "ent_coef": args.ent_coef,
        "learning_rate": learning_rate,
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
        # "target_kl": args.des_kl_divergence,
        "device": gpu_arg,
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

    #Load model
    if args.load_nn:
        interm_dir = f"{args.save_path}/logs/intermediate_models/{args.project_name}"
        log_dir = interm_dir + '' # add path
        stats_path = os.path.join(log_dir, "vec_normalize.pkl")
        model_name = get_latest_model(log_dir)
        
        wandb.config.update({
            "loaded_model_path": model_name,
            "loaded_stats_path": stats_path
        })

        env = VecNormalize.load(stats_path, env)

        if args.learning_alg == "PPO":
            model = PPO.load(model_name, env)
        elif args.learning_alg == "SAC":
            model = SAC.load(model_name, env)
        else:
            raise ValueError(args.learning_alg + ' not implemented')
        print("\nLoaded model", model_name, "\n")
        wandb.log({"model_loaded": True, "loaded_model_name": model_name})
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
            callback=[checkpoint_callback, WandbCallback()]
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
        "cpg_h": env.cpg_h_container,
        "cpg_g_c": env.cpg_g_c_container,
        "des_vel_x": env.des_vel_x_container,
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
    
    parser.add_argument("--learning-alg", type=str, default="PPO", choices=["PPO", "SAC"], help="Learning algorithm to use (default: PPO)")
    parser.add_argument("--motor_control_mode", type=str, default="CPG", choices=["CPG", "PD","TORQUE", "CARTESIAN_PD"], help="Motor control mode")
    parser.add_argument("--observation_space_mode", type=str, default="LR_COURSE_OBS", choices=["DEFAULT", "LR_COURSE_OBS"], help="Observation space mode")
    parser.add_argument("--task_env", type=str, default="LR_COURSE_TASK", choices=["LR_COURSE_TASK", "FLAGRUN","FWD_LOCOMOTION"], help="Task to be executed")
    parser.add_argument("--load-nn", action="store_true", help="Initialize training with a previous model")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of pybullet environments to create for data collection (default: 1)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training (make sure to install all necessary drivers)")
    parser.add_argument("--save-path", type=str, help="Path for storing intermediate models", default=".")


    parser.add_argument("--des_x_vel", type=float, default=0.4, help="desired linear velocity x axis")


    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps")
    parser.add_argument("--time_step", type=float, default=0.001, help="time step, for CPG_RL 0.01 s")
    parser.add_argument("--max_episode_length", type=float, default=10., help="max episode lenght in seconds in CPG_RL 20.0 s")
    parser.add_argument("--randomize_cpg_params", type=bool, default=True, help="Whether to randomize cpg params")
    parser.add_argument("--control_frequency", type=int, default=100, help="The control frequency of the policy [Hz]")

    # PPO Hyperparams
    parser.add_argument("--batch_size", type=int, default=8192, help="Size of rollout / batch size")
    parser.add_argument("--n_mini_batch", type=int, default=4, help="Number of minibatch")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor in PPO (gamma)")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="Entropy coefficient in PPO in CPG-RL 0.01")
    parser.add_argument("--gae_discount", type=float, default=0.95, help="GAE discount factor in PPO")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs in PPO in CPG-RL 5")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range in PPO")
    # parser.add_argument("--des_kl_divergence", type=float, default=None, help="Desired KL divergence in PPOin CPG-RL 0.01")    

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    run_sb3(args)


if __name__ == "__main__":
    main()