#/usr/bin/python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env 

class Simple2DGrid(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size: int = 5,
                 render_mode: bool = False):
        self.size = size
        self.render_mode = render_mode

        self.grid = np.zeros((size, size), dtype=np.int32)

        self._agent_position = np.array([-1,-1], dtype = np.int32)
        self._target_position = np.array([-1,-1], dtype = np.int32)

        self.action_space = spaces.Discrete(4)  # R,D,L,U
        self.observation_space = spaces.Dict({
            "agent_position": spaces.Box(low=0, high=size, shape=(2,), dtype=int),
            "taret_position": spaces.Box(low=0, high=size, shape=(2,), dtype=int),
        })

        self.action_to_move = {0 : [0 , +1],
                               1 : [+1 , 0],
                               2 : [ 0, -1],
                               3 : [-1 , 0]}
        if self.render_mode:
            self.init_simulation_render()

    def init_simulation_render(self):
        self.fig , (self.ax1, self.ax2) = plt.subplots(1,2, figsize=(10,5))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_position = self.np_random.integers(0, self.size, size = 2, dtype = int)
        print("Reset agent position:", self._agent_position)
        self.grid[tuple(self._agent_position)] = 1  # Mark new position as agent
        
        self._target_position = self._agent_position


        while np.array_equal(self._target_position, self._agent_position):
            self._target_position = self.np_random.integers(0, self.size, size = 2, dtype = int)

        if self.render_mode:
            self.render()


    def step(self, action):
        move = self.action_to_move[action]
        print("Action:", action, "Move:", move)
        new_pos = np.clip(self._agent_position +move, 0, self.size-1)
        self.grid[tuple(self._agent_position)] = 0  # Mark previous position as free
        self.grid[tuple(new_pos)] = 1  # Mark new position as agent
    
        self._agent_position = new_pos
        print("New agent position:", self._agent_position)

        if self.render_mode:
            self.init_simulation_render()
            self.render()

    def render(self):
        #grid = self.grid.copy()
        print("Grid:\n", self.grid)

        self.ax1.clear()
        self.ax1.set_xlim(0,self.size)
        self.ax1.set_ylim(0,self.size)
        self.ax1.set_xticks(range(self.size+1))
        self.ax1.set_yticks(range(self.size+1))

        # Customize the grid
        self.ax1.grid(
            visible=True,       # show grid
            which='major',       # 'major', 'minor', or 'both'
            axis='both',        # 'x', 'y', or 'both'
            linestyle='--',     # e.g. '-', '--', ':', '-.'
            linewidth=0.7,
            alpha=0.7            # transparency
        )
        # Customize the grid
        self.ax2.grid(
            visible=True,       # show grid
            which='major',       # 'major', 'minor', or 'both'
            axis='both',        # 'x', 'y', or 'both'
            linestyle='--',     # e.g. '-', '--', ':', '-.'
            linewidth=0.7,
            alpha=0.7            # transparency
        )

        self.ax1.scatter(self._agent_position[0]+0.5, self._agent_position[1]+0.5, marker='o')
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
        self.ax2.imshow(self.grid, cmap='Greys', origin='lower')
            #extent=[-0.5, self.size - 0.5, -0.5, self.size - 0.5],
            #interpolation="none",)
        plt.show()
        input()

    def _get_obs(self):
        return {
            "agent_position": self._agent_position,
            "target_position": self._target_position,
        }
    
if __name__ == "__main__":
    grid_size = 6
    env = Simple2DGrid(size=grid_size, render_mode=True)
    

    env.reset()
    
    for _ in range(10):
        action = random.randint(0,3)
        env.step(action)