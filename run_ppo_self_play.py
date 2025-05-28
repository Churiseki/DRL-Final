#!/usr/bin/env python3

import os
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from env_self_play import MeleeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import copy # For deep copying models

# 设置随机种子以确保可重复性
set_random_seed(42)

# 创建自定义回调函数来记录奖励
class SelfPlayRewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq=1000, log_dir=None, verbose=1):
        super(SelfPlayRewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []
        self.current_episode_frames = []
        self.win_loss_draw = {'win': 0, 'loss': 0, 'tie': 0}

    def _init_callback(self):
        if self.log_dir is not None:
            os.makedirs(os.path.join(self.log_dir, "rewards"), exist_ok=True)
            self.reward_file = open(os.path.join(self.log_dir, "rewards", "rewards.csv"), "w", newline='')
            self.reward_writer = csv.writer(self.reward_file)
            self.reward_writer.writerow(["Steps", "Mean Reward", "Min Reward", "Max Reward", "Win Rate", "Loss Rate", "Tie Rate"])
            self.episode_summary_file = open(os.path.join(self.log_dir, "rewards", "episode_summary.csv"), "w", newline='')
            self.episode_summary_writer = csv.writer(self.episode_summary_file)
            self.episode_summary_writer.writerow(["Episode", "Total Timesteps", "Reward", "Length", "Result"])

    def _on_step(self):
        # For a VecEnv, info is a list of dicts, rewards is a list of floats
        rewards = self.locals['rewards']
        infos = self.locals['infos']

        for i, info in enumerate(infos):
            self.current_rewards.append(rewards[i])
            self.current_episode_frames.append(1) # Assuming 1 frame per step

            if info.get('terminated') or info.get('truncated'):
                episode_total_reward = sum(self.current_rewards)
                episode_length = sum(self.current_episode_frames)
                self.episode_rewards.append(episode_total_reward)
                self.episode_lengths.append(episode_length)

                game_over_flag = info.get('game_over_flag')
                if game_over_flag:
                    self.win_loss_draw[game_over_flag] += 1

                # Log individual episode summary
                self.episode_summary_writer.writerow([
                    len(self.episode_rewards),
                    self.num_timesteps,
                    episode_total_reward,
                    episode_length,
                    game_over_flag
                ])

                self.current_rewards = []
                self.current_episode_frames = []

        if self.num_timesteps % self.check_freq == 0:
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-self.check_freq//self.model.n_envs:]) # Avg over recent episodes
                min_reward = np.min(self.episode_rewards[-self.check_freq//self.model.n_envs:])
                max_reward = np.max(self.episode_rewards[-self.check_freq//self.model.n_envs:])

                total_games = sum(self.win_loss_draw.values())
                win_rate = self.win_loss_draw['win'] / total_games if total_games > 0 else 0
                loss_rate = self.win_loss_draw['loss'] / total_games if total_games > 0 else 0
                tie_rate = self.win_loss_draw['tie'] / total_games if total_games > 0 else 0

                self.reward_writer.writerow([self.num_timesteps, mean_reward, min_reward, max_reward, win_rate, loss_rate, tie_rate])
                if self.verbose > 0:
                    print(f"Steps: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}")

        return True

    def _on_training_end(self):
        if self.log_dir is not None:
            self.reward_file.close()
            self.episode_summary_file.close()
            np.save(os.path.join(self.log_dir, "rewards", "all_episode_rewards.npy"), np.array(self.episode_rewards))
            np.save(os.path.join(self.log_dir, "rewards", "all_episode_lengths.npy"), np.array(self.episode_lengths))

# Self-Play Specific Configuration
TOTAL_TIMESTEPS = 10000000 # Total timesteps for the entire self-play process
TRAIN_INTERVAL_TIMESTEPS = 50000 # How many timesteps to train one agent before updating opponent
SAVE_FREQ_INTERVAL = 100000 # Save full models every X timesteps

# Create models directory
base_models_dir = "self_play_models"
experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S_self_play")
models_dir = os.path.join(base_models_dir, experiment_name)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(models_dir, "p1"), exist_ok=True)
os.makedirs(os.path.join(models_dir, "p2"), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize two PPO models
# Player 1 is the primary agent we're training
# Player 2 acts as the opponent
# We will deep copy P2 for the actual opponent during training
# Initializing with a dummy env, as the env will be set in the loop
model_p1 = PPO(
    policy="MlpPolicy",
    env=None,
    verbose=0, # Set to 1 for more verbose output during training
    device=device,
    tensorboard_log=os.path.join("./tensorboard_logs", experiment_name, "p1_logs"),
    n_steps=2048, # Number of steps to run for each environment per update
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    learning_rate=3e-4,
    n_epochs=10,
)

model_p2 = PPO(
    policy="MlpPolicy",
    env=None,
    verbose=0,
    device=device,
    tensorboard_log=os.path.join("./tensorboard_logs", experiment_name, "p2_logs"),
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    learning_rate=3e-4,
    n_epochs=10,
)

print("Starting self-play training...")

current_timesteps = 0
while current_timesteps < TOTAL_TIMESTEPS:
    print(f"\n--- Self-Play Training Iteration: {current_timesteps} / {TOTAL_TIMESTEPS} ---")

    # --- Phase 1: Train P1 against frozen P2 ---
    print("Training Player 1 against Player 2 (frozen)...")
    # Create a deep copy of P2's policy to act as a fixed opponent for P1
    # This prevents P2 from learning during P1's training phase
    frozen_opponent_for_p1 = PPO(
        policy="MlpPolicy",
        env=None, # Dummy env
        device=device,
    )
    # Load P2's state dict into the frozen opponent
    frozen_opponent_for_p1.policy.load_state_dict(model_p2.policy.state_dict())
    frozen_opponent_for_p1.set_parameters(model_p2.get_parameters()) # Ensure all parameters are set

    # Create environment for P1 where P2 is the opponent
    # P1 is agent_port=1, P2 is opponent_port=2
    env_p1 = DummyVecEnv([lambda: MeleeEnv(opponent_model=frozen_opponent_for_p1, agent_port=1, opponent_port=2)])
    model_p1.set_env(env_p1)

    # Callbacks for P1
    checkpoint_callback_p1 = CheckpointCallback(
        save_freq=TRAIN_INTERVAL_TIMESTEPS // 2, # Save checkpoints within the interval
        save_path=os.path.join(models_dir, "p1", "checkpoints"),
        name_prefix=f"p1_checkpoint_{current_timesteps}",
        save_replay_buffer=False,
        verbose=0
    )
    reward_callback_p1 = SelfPlayRewardLoggerCallback(
        check_freq=1000,
        log_dir=os.path.join(models_dir, "p1_logs"),
        verbose=1
    )

    model_p1.learn(
        total_timesteps=TRAIN_INTERVAL_TIMESTEPS,
        callback=[checkpoint_callback_p1, reward_callback_p1],
        reset_num_timesteps=False, # Continue training
        progress_bar=True,
    )
    env_p1.close() # Close environment after training
    print("Player 1 training complete for this interval.")

    # After P1 trains, update P2's current model with P1's learned policy
    # This is the "self-play" step where the opponent improves
    print("Updating Player 2 with Player 1's policy...")
    model_p2.policy.load_state_dict(model_p1.policy.state_dict())
    model_p2.set_parameters(model_p1.get_parameters())
    print("Player 2 policy updated.")

    # --- Phase 2: Train P2 against frozen P1 (optional, for symmetric training) ---
    # You can choose to symmetrically train P2 against P1 here if desired.
    # For simplicity in initial setup, we will skip this and just update P2 with P1's weights.
    # The self-play effect still comes from P1 continuously training against an improving P2.
    # If you want truly symmetric training, uncomment and adapt the following:

    # print("Training Player 2 against Player 1 (frozen)...")
    # frozen_opponent_for_p2 = PPO(policy="MlpPolicy", env=None, device=device)
    # frozen_opponent_for_p2.policy.load_state_dict(model_p1.policy.state_dict())
    # frozen_opponent_for_p2.set_parameters(model_p1.get_parameters())

    # env_p2 = DummyVecEnv([lambda: MeleeEnv(opponent_model=frozen_opponent_for_p2, agent_port=2, opponent_port=1)])
    # model_p2.set_env(env_p2)

    # checkpoint_callback_p2 = CheckpointCallback(
    #     save_freq=TRAIN_INTERVAL_TIMESTEPS // 2,
    #     save_path=os.path.join(models_dir, "p2", "checkpoints"),
    #     name_prefix=f"p2_checkpoint_{current_timesteps}",
    #     save_replay_buffer=False,
    #     verbose=0
    # )
    # reward_callback_p2 = SelfPlayRewardLoggerCallback(
    #     check_freq=1000,
    #     log_dir=os.path.join(models_dir, "p2_logs"),
    #     verbose=1
    # )

    # model_p2.learn(
    #     total_timesteps=TRAIN_INTERVAL_TIMESTEPS,
    #     callback=[checkpoint_callback_p2, reward_callback_p2],
    #     reset_num_timesteps=False,
    #     progress_bar=True,
    # )
    # env_p2.close()
    # print("Player 2 training complete for this interval.")

    # At regular intervals, save the current P1 and P2 models
    if (current_timesteps % SAVE_FREQ_INTERVAL == 0 and current_timesteps > 0) or \
       (current_timesteps + TRAIN_INTERVAL_TIMESTEPS >= TOTAL_TIMESTEPS):
        print(f"Saving models at {current_timesteps} timesteps...")
        model_p1.save(os.path.join(models_dir, f"ppo_melee_p1_final_t{current_timesteps}"))
        model_p2.save(os.path.join(models_dir, f"ppo_melee_p2_final_t{current_timesteps}")) # P2 is now the updated policy
        print("Models saved.")

    current_timesteps += TRAIN_INTERVAL_TIMESTEPS

print("Self-play training completed.")
# Final save of the models
model_p1.save(os.path.join(models_dir, "ppo_melee_p1_final_total"))
model_p2.save(os.path.join(models_dir, "ppo_melee_p2_final_total"))

print(f"Final models saved to: {models_dir}")