#/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env 

if __name__ == "__main__":
    import sys
    sys.path.append(".")

from lib.grid_env.obst_grid_agent_env import ObstGridAgentEnv
from lib.rendering_utils import fig_to_rgb

class SimpleTargetAgentEnv(ObstGridAgentEnv):
    def __init__(self, max_steps=500, *args, **kwargs):
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

        self.max_steps = max_steps
        self.steps = 0
        self.max_absolute = max(self.width, self.height)

        self.observation_space = spaces.Dict({
                'agent_position': spaces.Box(low=0, 
                                            high=self.max_absolute, 
                                            shape=(2,), 
                                            dtype=np.int64),

                'target_position': spaces.Box(low=0, 
                                            high=self.max_absolute, 
                                            shape=(2,), 
                                            dtype=np.int64)
            })
        
        self.action_space = spaces.Discrete(4)  # R,D,L,U
    
    def reset(self, seed=None, options=None, init_agent_pos = None):
        super().reset(seed=seed, init_agent_pos = init_agent_pos)

        self.init_target_position()

        if self.render_mode == "human":
            print("Target initialized at position: ", self.target_pos)

        self.update_obs_grid()
        if self.render_mode is not None:
            self.render()
        
        obs = self.get_obs()
        return obs,  {}
    
    def step(self, action):
        self.steps += 1
        try:
            move = self._action_to_direction[action]
        except TypeError:
            print("Action: ", action, "type: ", type(action))
            raise
        new_x = self.agent_pos[0] + move[0]
        new_y = self.agent_pos[1] + move[1]
        if self.acceptable_move(new_x, new_y):
            # self.grid[tuple(self.agent_pos)] = self.free_color  # Mark previous position as free
            # self.grid[tuple((new_x, new_y))] = self.agent_color  # Mark new position as agent
            self.set_agent_position(new_x, new_y)
            self.update_obs_grid()
        
        
        reward = 0
        done = False
        if np.array_equal(self.agent_pos, self.target_pos):
            reward = 1
            done = True
            if self.render_mode == "human":
                print("Target reached in {} steps!".format(self.steps))

        if self.steps >= self.max_steps:
            done = True
            reward = -1
            if self.render_mode == "human":
                print("Max steps reached. Target not reached.")
        
        if self.render_mode is not None:
            if self.render_mode == "human":
                print("Action taken: ", self._action_meaning[action])
                print("New agent position: ", self.agent_pos)
            self.render()

        obs = self.get_obs()
        return obs, reward, done, False, {}

    def get_obs(self):
        return {'agent_position':self.agent_pos, 
                'target_position': self.target_pos}

    def render(self):
        super().render()
        self.ax_env.scatter(self.target_pos[0], 
                         self.target_pos[1], 
                         marker='*', 
                         s=200, 
                         color='gold', 
                         edgecolors='black')
        if self.render_mode == "human":
            plt.pause(0.1)
        elif self.render_mode == "rgb_array":
            return fig_to_rgb(self.fig)
    

    def random_position(self):
        """initialize agent position randomly in an acceptable cell"""
        x = self.np_random.integers(0, self.width) 
        y = self.np_random.integers(0, self.height)
        while not self.acceptable_move(x, y):
            x = self.np_random.integers(0, self.width)
            y = self.np_random.integers(0, self.height)
        return x, y
    
    
    def init_target_position(self):
        x, y = self.random_position()
        self.target_pos = np.array((x, y))

if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.0
    
    env = SimpleTargetAgentEnv( width=width, 
                                height=height, 
                                obstacle_prob=obstacle_prob, 
                                render_mode="human")
    plt.ion()
    obs, _ = env.reset()
    trunc = False
    done = False
    while not done and not trunc:
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        input()
        