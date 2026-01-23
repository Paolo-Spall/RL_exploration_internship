#/usr/bin/python3

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    sys.path.append(".")

from lib.grid_env.obst_grid_agent_env import ObstGridAgentEnv

class ObstGridAgentStepEnv(ObstGridAgentEnv):
    """ Environment that add agent to a 2d grid env with obstacles.
        Methods to be overridden in child classes: 
         - step(), update_obs_grid(), render()
        Reset method can be extended if needed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }
        self._action_meaning = {
            0: "RIGHT",
            1: "UP",
            2: "LEFT",
            3: "DOWN",
        }
    
    ## ENVIRONMENT DYNAMICS AND INTERACTION METHODS

    def step(self, action):
        move = self._action_to_direction[int(action)]
        newx, newy = self.agent_pos + move

        if self.render_mode == "human":
            print("Action taken: ", self._action_meaning[int(action)] )

        # move the agent to new position only if inside bounds and not an obstacle
        if self.acceptable_move(newx, newy):
            self.set_agent_position(newx, newy)
            if self.render_mode == "human":
                print("Agent moved to ({},{})".format(newx, newy))
        else:
            if self.render_mode == "human":
                print("Invalid move attempted to ({},{})".format(newx, newy))
        self.update_obs_grid()

        if self.render_mode == "human":
            self.render()

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.update_obs_grid()
        if self.render_mode == "human":
            self.render()



if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    print("Creating environment...")
    env = ObstGridAgentStepEnv(width=width, 
                            height=height, 
                            render_mode="human", 
                            obstacle_prob=obstacle_prob
                        )

    print("Resetting environment...")
    env.reset(init_agent_pos=(1,5))

    plt.show()

    for action in range(4):
        print("Taking action: ", action)
        env.init_simulation_render()
        env.step(action)
        plt.show()