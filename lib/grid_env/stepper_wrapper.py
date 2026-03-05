#/usr/bin/python3

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    sys.path.append(".")

from lib.grid_env.obst_grid_agent_env import ObstGridAgentEnv

class StepperWrapper:
    """ A wrapper for grid environments that allows movement in four cardinal directions:
        0: Move right (positive x)
        1: Move up (positive y) 
        etc.
    """

    def __init__(self, grid_env):
        self._env = grid_env
    
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
        newx, newy = self._env.agent_pos + move

        if self._env.render_mode == "human":
            print("Action taken: ", self._action_meaning[int(action)] )

        # move the agent to new position only if inside bounds and not an obstacle
        if self._env.acceptable_move(newx, newy):
            self._env.set_agent_position(newx, newy)
            if self._env.render_mode == "human":
                print("Agent moved to ({},{})".format(newx, newy))
        else:
            if self._env.render_mode == "human":
                print("Invalid move attempted to ({},{})".format(newx, newy))
        self._env.update_obs_grid()

        if self._env.render_mode == "human":
            self.render()

    def reset(self, *args, **kwargs):
        self._env.reset(*args, **kwargs)
        self._env.update_obs_grid()
        if self._env.render_mode == "human":
            self.render()
    
    def render(self):
        self._env.render()
        if self._env.render_mode == "human":
            plt.pause(0.5)



if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    print("Creating environment...")
    not_wrapped_env = ObstGridAgentEnv(width=width, 
                            height=height, 
                            render_mode="human", 
                            obstacle_prob=obstacle_prob
                        )
    env = StepperWrapper(not_wrapped_env)    

    print("Resetting environment...")
    env.reset(init_agent_pos=(1,5))

    plt.show()

    for action in range(4):
        print("Taking action: ", action)
        env._env.init_simulation_render()
        env.step(action)
        plt.show()