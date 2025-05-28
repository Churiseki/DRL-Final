#!/usr/bin/env python3

import melee
from melee import enums, Menu, MenuHelper
import numpy as np
import random
import os
import sys
import time
from config import CONFIG # Assuming config.py exists and contains CONFIG
from state import form_state, calculate_reward, get_distance # Assuming state.py exists and contains these functions
debug = False

import gymnasium as gym
from gymnasium import spaces
gym_module = gym
env_class = gym.Env

# Original action space definitions
cardinal_sticks = [
    (0, 0.5),
    (1, 0.5),
    (0.5, 0),
    (0.5, 1),
    (0.5, 0.5)
]
tilt_sticks = [
    (0.4, 0.5),
    (0.6, 0.5),
    (0.5, 0.4),
    (0.5, 0.6)
]
diagonal_sticks = [
    (0, 0), (0, 0.5), (0, 1),
    (0.5, 0), (0.5, 0.5), (0.5, 1),
    (1, 0), (1, 0.5), (1, 1)
]
neutral_stick = [(0.5, 0.5)]

SimpleButton = enums.Button

ACTION_SPACE = []
for btn in [SimpleButton.BUTTON_A, SimpleButton.BUTTON_B]:
    for stick in cardinal_sticks:
        ACTION_SPACE.append((btn, stick))
for stick in tilt_sticks:
    ACTION_SPACE.append((SimpleButton.BUTTON_A, stick))
for btn in [SimpleButton.BUTTON_Z, SimpleButton.BUTTON_L, SimpleButton.BUTTON_R, SimpleButton.BUTTON_X, SimpleButton.BUTTON_Y]:
    ACTION_SPACE.append((btn, (0.5, 0.5)))
for stick in diagonal_sticks:
    ACTION_SPACE.append((None, stick)) # No button, just stick
ACTION_SPACE.append((None, neutral_stick[0])) # Neutral stick

class MeleeEnv(env_class):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config=None, opponent_model=None, agent_port=1, opponent_port=2):
        super().__init__()
        self.total_reward = 0
        self.config = config if config is not None else CONFIG

        self.action_space = spaces.Discrete(len(ACTION_SPACE))
        # This observation space might need adjustment based on your form_state output
        # Assuming form_state returns a fixed-size numpy array of floats
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(46,), # This should match the output shape of form_state
            dtype=np.float32
        )

        self.opponent_model = opponent_model # The SB3 model for the opponent
        self.agent_port = agent_port         # Port for the agent being trained (e.g., P1)
        self.opponent_port = opponent_port   # Port for the opponent (e.g., P2)
        self.ports = [self.agent_port, self.opponent_port] # Ordered for character selection consistency

        self.console = melee.Console(
            slippi_path=self.config["SLIPPI_PATH"],
            debug=debug,
            blocking_on_step=False,
            display=False, # Set to True if you want to see the Dolphin window
        )

        self.ctrls = {
            self.agent_port: melee.Controller(console=self.console, port=self.agent_port, type=melee.ControllerType.STANDARD),
            self.opponent_port: melee.Controller(console=self.console, port=self.opponent_port, type=melee.ControllerType.STANDARD)
        }

        # Connect controllers
        for port, ctrl in self.ctrls.items():
            if not ctrl.connect():
                raise RuntimeError(f"Failed to connect controller on port {port}.")

        self.console.connect()

        self.stages = self.config["STAGES"]
        self.characters = self.config["CHARACTERS"]
        self.agent_levels = self.config["CPU_LEVELS"] # Can be used if opponent is CPU, or simply ignored
        self.costumes = self.config["COSTUMES"]

        self.stage = random.choice(self.stages)
        self.cur_char = [None, None]
        self.cur_level = [None, None]

        self.current_action = None
        self.frame_count = 0
        self.game_over_flag = None
        self.prev_gamestate = None
        self.menu_helper = MenuHelper()
        self.in_game_frame_counter = 0

    def reset(self, seed=None, options=None):
        self.total_reward = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Character selection for both agent and opponent based on their roles
        # Ensure that if opponent_model is present, the opponent character is consistent or selected appropriately
        for i, port in enumerate(self.ports):
            if port == self.agent_port:
                self.cur_char[i] = random.choice(self.characters[0])
                self.cur_level[i] = random.choice(self.agent_levels[0])
            elif port == self.opponent_port:
                self.cur_char[i] = random.choice(self.characters[1])
                # If opponent_model is provided, the level is not relevant as it's not a CPU
                self.cur_level[i] = random.choice(self.agent_levels[1])

        while True:
            gs = self.console.step()
            if gs is None:
                continue

            state = gs.menu_state

            if state == Menu.INITIALIZING:
                print("Dolphin is initializing. Waiting...")
                continue
            elif state == Menu.CHARACTER_SELECT:
                # Agent's character selection
                self.menu_helper.choose_character(
                    gamestate=gs,
                    controller=self.ctrls[self.agent_port],
                    character=self.cur_char[self.ports.index(self.agent_port)],
                    cpu_level=1, # Agent is not CPU
                    costume=self.costumes[self.ports.index(self.agent_port)],
                    swag=False,
                    start=(self.agent_port == self.ports[0])
                )
                # Opponent's character selection
                self.menu_helper.choose_character(
                    gamestate=gs,
                    controller=self.ctrls[self.opponent_port],
                    character=self.cur_char[self.ports.index(self.opponent_port)],
                    cpu_level=1 if self.opponent_model else self.cur_level[self.ports.index(self.opponent_port)], # CPU level if no opponent_model
                    costume=self.costumes[self.ports.index(self.opponent_port)],
                    swag=False,
                    start=(self.opponent_port == self.ports[0])
                )
            elif state == Menu.STAGE_SELECT:
                self.menu_helper.choose_stage(self.stage, gs, self.ctrls[self.agent_port]) # Agent chooses stage
            elif state in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
                self.in_game_frame_counter = 0
                self.prev_gamestate = gs
                observation = form_state(gs, self.agent_port, self.opponent_port, stage_mode=1)
                info = {
                    "stage": gs.stage.name if hasattr(gs.stage, "name") else str(gs.stage),
                    f"agent_{self.agent_port}_character": gs.players[self.agent_port].character.name,
                    f"opponent_{self.opponent_port}_character": gs.players[self.opponent_port].character.name,
                    f"opponent_{self.opponent_port}_level": gs.players[self.opponent_port].cpu_level if self.opponent_model is None else "Model",
                    "frame": 0
                }
                return observation, info
            else:
                self.menu_helper.press_start(gs, self.ctrls[self.agent_port]) # Agent presses start

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            # Action for the agent being trained
            agent_action_decoded = ACTION_SPACE[action]
        else:
            agent_action_decoded = action # Already decoded if directly provided

        agent_button, agent_direction = agent_action_decoded
        if agent_button is not None:
            self.ctrls[self.agent_port].press_button(agent_button)
        self.ctrls[self.agent_port].tilt_analog(enums.Button.BUTTON_MAIN, agent_direction[0], agent_direction[1])

        # Get and apply opponent's action if an opponent_model is provided
        opponent_action_decoded = (None, (0.5, 0.5)) # Default to neutral if no model
        if self.opponent_model and self.prev_gamestate:
            # The opponent model needs its own observation from its perspective
            opponent_obs = form_state(self.prev_gamestate, self.opponent_port, self.agent_port, stage_mode=1)
            # stable_baselines3 predict expects a batch, so we need to add a batch dimension
            opponent_action_idx, _states = self.opponent_model.predict(np.expand_dims(opponent_obs, axis=0), deterministic=True)
            opponent_action_decoded = ACTION_SPACE[opponent_action_idx[0]] # Remove batch dimension

        opponent_button, opponent_direction = opponent_action_decoded
        if opponent_button is not None:
            self.ctrls[self.opponent_port].press_button(opponent_button)
        self.ctrls[self.opponent_port].tilt_analog(enums.Button.BUTTON_MAIN, opponent_direction[0], opponent_direction[1])

        # Flush both controllers
        for ctrl in self.ctrls.values():
            ctrl.flush()

        prev_gs = self.prev_gamestate
        next_gs = self.console.step()
        if next_gs is None:
            # Handle cases where gamestate might be None (e.g., emulator crashed or not ready)
            info = {"warning": "Received None gamestate"}
            return self.observation_space.sample(), 0.0, False, False, info

        # Check for game over conditions
        terminated = False
        truncated = False
        self.game_over_flag = None

        if next_gs.menu_state == Menu.END_OF_GAME:
            if next_gs.players[self.agent_port].stock > next_gs.players[self.opponent_port].stock:
                self.game_over_flag = "win"
            elif next_gs.players[self.agent_port].stock < next_gs.players[self.opponent_port].stock:
                self.game_over_flag = "lose"
            else:
                # If stocks are equal, check percent for sudden death
                if next_gs.players[self.agent_port].percent < next_gs.players[self.opponent_port].percent:
                    self.game_over_flag = "win"
                elif next_gs.players[self.agent_port].percent > next_gs.players[self.opponent_port].percent:
                    self.game_over_flag = "lose"
                else:
                    self.game_over_flag = "tie" # Or handle as a loss or small negative reward

            terminated = True
        elif self.in_game_frame_counter >= self.config["MAX_GAME_FRAMES"]:
            # Timeout
            terminated = True
            truncated = True # Indicate truncation due to time limit
            # You might want to assign a reward based on current state at timeout
            if next_gs.players[self.agent_port].stock > next_gs.players[self.opponent_port].stock:
                self.game_over_flag = "win"
            elif next_gs.players[self.agent_port].stock < next_gs.players[self.opponent_port].stock:
                self.game_over_flag = "lose"
            else:
                if next_gs.players[self.agent_port].percent < next_gs.players[self.opponent_port].percent:
                    self.game_over_flag = "win"
                elif next_gs.players[self.agent_port].percent > next_gs.players[self.opponent_port].percent:
                    self.game_over_flag = "lose"
                else:
                    self.game_over_flag = "tie"


        # Calculate reward for the current agent
        reward = calculate_reward(prev_gs, next_gs, self.agent_port, self.opponent_port)

        self.prev_gamestate = next_gs
        self.in_game_frame_counter += 1

        # Release buttons for both controllers
        if agent_button is not None:
            self.ctrls[self.agent_port].release_button(agent_button)
        self.ctrls[self.agent_port].tilt_analog(enums.Button.BUTTON_MAIN, 0.5, 0.5)

        if opponent_button is not None:
            self.ctrls[self.opponent_port].release_button(opponent_button)
        self.ctrls[self.opponent_port].tilt_analog(enums.Button.BUTTON_MAIN, 0.5, 0.5)

        for ctrl in self.ctrls.values():
            ctrl.flush()

        info = {
            "frame": self.in_game_frame_counter,
            "game_state": str(next_gs.menu_state),
            f"agent_{self.agent_port}_stocks": next_gs.players[self.agent_port].stock,
            f"opponent_{self.opponent_port}_stocks": next_gs.players[self.opponent_port].stock,
            f"agent_{self.agent_port}_percent": next_gs.players[self.agent_port].percent,
            f"opponent_{self.opponent_port}_percent": next_gs.players[self.opponent_port].percent,
            "game_over_flag": self.game_over_flag
        }
        self.total_reward += reward

        if terminated:
            # Add final bonus/penalty at episode end based on game_over_flag
            if self.game_over_flag == "win":
                reward += 1.0 # Bonus for winning
            elif self.game_over_flag == "lose":
                reward += -1.0 # Penalty for losing
            print(f"DEBUG: Game Over - {self.game_over_flag}")
            print(f"DEBUG: Final Reward for Agent {self.agent_port}: {self.total_reward}")

        observation = form_state(next_gs, self.agent_port, self.opponent_port, stage_mode=1)
        return observation, reward, terminated, truncated, info

    def render(self):
        # Implement rendering logic if needed, otherwise leave as pass
        pass

    def close(self):
        if self.console:
            self.console.stop()
            self.console = None
        for ctrl in self.ctrls.values():
            ctrl.disconnect()
        print("MeleeEnv closed.")