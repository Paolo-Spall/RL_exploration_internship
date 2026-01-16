#/usr/bin/python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env 

class Simple2DGridMultiObs(gym.Env):
    metadata = {"render_modes": ["human"]}
    target_cell = 127  # Light gray for unknown cells
    agent_cell = 255  # Black for agent position
    free_cell = 0  # White for free cells

    def __init__(self, size: int = 5,
                 render_mode: bool = None,
                 max_steps: int = 50,
                 policy_type: str = 'MlpPolicy'):
        self.size = size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.policy_type = policy_type

        

        self._agent_position = np.array([-1,-1], dtype = np.int32)
        self._target_position = np.array([-1,-1], dtype = np.int32)

        self.action_space = spaces.Discrete(4)  # R,D,L,U
        if self.policy_type == 'CnnPolicy':
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(size, size, 1), dtype=np.uint8
            )
        elif self.policy_type == 'MlpPolicy':
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(size * size,), dtype=np.uint8
            )
        elif self.policy_type == 'MultiInputPolicy':
            self.observation_space = spaces.Dict({
                'grid': spaces.Box(low=0, high=255, shape=(size, size, 1), dtype=np.uint8),
                'position': spaces.Box(low=0, high=max(size, size), shape=(2,), dtype=np.int32)
            })

        self.action_to_move = {0 : [0 , +1],
                               1 : [+1 , 0],
                               2 : [ 0, -1],
                               3 : [-1 , 0]}
        if self.render_mode == "human":
            self.init_simulation_render()

    def init_simulation_render(self):
        self.fig , self.ax2 = plt.subplots(1,1, figsize=(10,5))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.ones((self.size, self.size), dtype=np.uint8) * self.free_cell
        self.steps = 0

        self._agent_position = self.np_random.integers(0, self.size, size = 2, dtype = int)
        self.initial_agent_position = self._agent_position.copy()
        self.grid[tuple(self._agent_position)] = self.agent_cell  # Mark new position as agent
        
        self._target_position = self._agent_position


        while np.array_equal(self._target_position, self._agent_position):
            self._target_position = self.np_random.integers(0, self.size, size = 2, dtype = int)

        self.grid[tuple(self._target_position)] = self.target_cell  # Mark target position

        if self.render_mode == "human":
            print("Environment reset.")
            print("Initial agent position:", self._agent_position)
            print("Target position:", self._target_position)
            self.render()

        obs = self._get_obs()
        return obs, {}


    def step(self, action):
        reward = 0.0
        truncated = False
        terminated = False
        move = self.action_to_move[int(action)]
        new_pos = np.clip(self._agent_position +move, 0, self.size-1)
        self.grid[tuple(self._agent_position)] = self.free_cell  # Mark previous position as free
        self.grid[tuple(new_pos)] = self.agent_cell  # Mark new position as agent
    
        self._agent_position = new_pos
        self.steps += 1
        if np.array_equal(self._agent_position, self._target_position):
            reward = 1.0
            terminated = True
            if self.render_mode == "human":
                print("Target reached!")
        
        if self.steps >= self.max_steps:
            reward = -1.0
            truncated = True
            if self.render_mode == "human":
                print("Max steps reached.")

        obs = self._get_obs()

        if self.render_mode == "human":
            print("Action:", action, "Move:", move)

            print("New agent position:", self._agent_position)
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        #grid = self.grid.copy()
        print("Grid:\n", self.grid)

        self.ax2.clear()
        # self.ax1.set_xlim(0,self.size)
        # self.ax1.set_ylim(0,self.size)
        # self.ax1.set_xticks(range(self.size+1))
        # self.ax1.set_yticks(range(self.size+1))

        # # Customize the grid
        # self.ax1.grid(
        #     visible=True,       # show grid
        #     which='major',       # 'major', 'minor', or 'both'
        #     axis='both',        # 'x', 'y', or 'both'
        #     linestyle='--',     # e.g. '-', '--', ':', '-.'
        #     linewidth=0.7,
        #     alpha=0.7            # transparency
        # )
        # Customize the grid
        self.ax2.grid(
            visible=True,       # show grid
            which='major',       # 'major', 'minor', or 'both'
            axis='both',        # 'x', 'y', or 'both'
            linestyle='--',     # e.g. '-', '--', ':', '-.'
            linewidth=0.7,
            alpha=0.7            # transparency
        )

        #self.ax1.scatter(self._agent_position[0]+0.5, self._agent_position[1]+0.5, marker='o')
        # put grid lines on cell borders
        ticks = np.arange(-0.5, self.size, 1)
        self.ax2.set_xticks(ticks)
        self.ax2.set_yticks(ticks)
        #self.ax2.grid(True, which="both", color="k", linewidth=0.7, alpha=0.7)
        centers = np.arange(0, self.size+1, 1)
        # self.ax2.set_xticks(centers + 0.0, minor=True)
        # self.ax2.set_yticks(centers + 0.0, minor=True)
        self.ax2.set_xticklabels(centers)
        self.ax2.set_yticklabels(centers)
        #self.ax2.scatter(self._target_position[1], self._target_position[0], marker='*', s=200, color='gold', edgecolors='black')
        self.ax2.imshow(self.grid, cmap='Greys', origin='lower')
            #extent=[-0.5, self.size - 0.5, -0.5, self.size - 0.5],
            #interpolation="none",)
        plt.pause(0.1)
        # plt.show()
        # input()
        

    def _get_obs(self):
        if self.policy_type == 'CnnPolicy':
            return self.grid[:, :, np.newaxis]
        elif self.policy_type == 'MlpPolicy':
            return self.grid.flatten()
    
if __name__ == "__main__":
    grid_size = 36
    env = Simple2DGridMultiObs(size=grid_size, 
                               render_mode='human', 
                               policy_type='CnnPolicy')
    check_env(env)
    

    env.reset()
    
    for _ in range(10):
        action = random.randint(0,3)
        env.step(action)