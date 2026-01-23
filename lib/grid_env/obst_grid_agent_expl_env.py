#/usr/bin/python3

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    sys.path.append(".")

from lib.grid_env.obst_grid_agent_env import ObstGridAgentEnv
from lib.grid_env.stepper_wrapper import StepperWrapper



class ObstGridAgentExplEnv(ObstGridAgentEnv):
    """ Environment that implement EXPLORATION on a 2d grid env with obstacles."""

    def __init__(self, perception_range=3,  *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.perception_range = perception_range


    def reset(self, seed=None, *args, **kwargs):
        super().reset(seed=seed, *args, **kwargs)
        self._generate_obs_grid()
        

    ## OBSERVATIONS
    def _generate_obs_grid(self):
        self.obs_grid = np.ones_like(self.grid) * self.unknown_color


    def update_obs_grid(self):
        discovered_cells = 0
        x = self.agent_pos[0]
        y = self.agent_pos[1]
        r = self.perception_range
        ymin = max(0, y - r)
        ymax = min(self.height - 1, y + r)
        xmin = max(0, x - r)
        xmax = min(self.width - 1, x + r)
        obs_area = self.grid[ymin:ymax+1, xmin:xmax+1]

        discovered_cells = np.sum(self.obs_grid[ymin:ymax+1, xmin:xmax+1] == self.unknown_color)
        self.discovered_cells += discovered_cells

        self.obs_grid[ymin:ymax+1, xmin:xmax+1] = obs_area

        self.obs_grid[self.agent_pos[1]][self.agent_pos[0]] = self.agent_color  # mark agent position
        return discovered_cells

    ## RENDERING FUNCTIONS

    def init_simulation_render(self):
        self.fig , (self.ax_env, self.ax_obs) = plt.subplots(1,2, figsize=(10,5))

    def render(self):
        self.ax_env.clear()
        self.ax_obs.clear()
        
        x_agent, y_agent = self.agent_pos

        old_agent_cell = self.grid[y_agent][x_agent]
        self.grid[y_agent][x_agent] = self.agent_color
        
        self.ax_env.imshow(self.grid, cmap='Greys', 
                           vmin=self.min_color, vmax=self.max_color)
                            #, origin='upper', vmin=0, vmax=255)
        self.ax_obs.imshow(self.obs_grid, cmap='Greys', 
                           vmin=self.min_color, vmax=self.max_color)
                            #, origin='upper', vmin=0, vmax=255)

        self.grid[y_agent][x_agent] = old_agent_cell


if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    perception_range = 3
    print("Creating environment...")
    env = ObstGridAgentExplEnv( width=width, 
                                height=height, 
                                obstacle_prob=obstacle_prob,
                                perception_range=perception_range,
                                render_mode="human"
                                )

    env = StepperWrapper(env)  

    print("Resetting environment...")
    env.reset()

    plt.show()

    for action in range(4):
        print("Taking action: ", action)
        env._env.init_simulation_render()
        env.step(action)
        plt.show()