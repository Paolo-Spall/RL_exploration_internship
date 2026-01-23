#/usr/bin/python3

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    sys.path.append(".")

from lib.grid_env.obst_grid_env import ObstGridEnv

class ObstGridAgentEnv(ObstGridEnv):
    """ Environment that add agent to a 2d grid env with obstacles.
        Methods to be overridden in child classes: 
         - update_obs_grid(), render()
        Reset method can be extended if needed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    ## ENVIRONMENT DYNAMICS AND INTERACTION METHODS

    def reset(self, seed=None, options=None, init_agent_pos = None):
        super().reset(seed=seed)

        if init_agent_pos is not None:
            self.set_agent_position(*init_agent_pos)
        else:
            self.init_agent_position()
        
        if self.render_mode == "human":
            print("Agent initialized at position: ", self.agent_pos)



    def init_agent_position(self):
        """initialize agent position randomly in an acceptable cell"""
        x = random.randint(0, self.width-1) 
        y = random.randint(0, self.height-1)
        while not self.acceptable_move(x, y):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
        self.set_agent_position(x, y)

    def set_agent_position(self, x, y):
        self.agent_pos = np.array((x, y))

    ## OBSERVATIONS

    def update_obs_grid(self):
        self.obs_grid = np.copy(self.grid)
        x_agent, y_agent = self.agent_pos
        self.obs_grid[y_agent, x_agent] = self.agent_color

    ## RENDERING FUNCTIONS

    def render(self):
        self.ax_env.clear()
        
        self.ax_env.imshow(self.obs_grid, cmap='Greys', vmin=self.min_color, vmax=self.max_color)#, origin='upper', vmin=0, vmax=255)



if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    print("Creating environment...")
    env = ObstGridAgentEnv(width=width, 
                       height=height, 
                       render_mode="human", 
                       obstacle_prob=obstacle_prob
                    )

    print("Resetting environment...")
    env.reset()
    env.update_obs_grid()
    print("Rendering the environment...")
    env.render()
    plt.show()