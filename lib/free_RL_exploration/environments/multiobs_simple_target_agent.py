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
from lib.utils import move_toward

class MultiobsSimpleTargetAgentEnv(ObstGridAgentEnv):
    target_color = 2  # Dark gray for obstacles
    agent_color = 1  # Black for agent position
    obstacle_color = -1  # Dark gray for obstacles
    free_color = 0
    min_color = -1
    max_color = 2

    def __init__(self, 
                 max_steps=500, 
                 obs_type = "pos_dict",
                 no_obst_color=False,
                 *args, **kwargs):
        if no_obst_color:
            self.target_color = -1
            self.max_color = 1
        super().__init__(*args, **kwargs)
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }
        self.direction_to_action = {tuple(v): k for k, v in self._action_to_direction.items()}
        self._action_meaning = {
            0: "RIGHT",
            1: "UP",
            2: "LEFT",
            3: "DOWN",
        }

        self.max_steps = max_steps
        self.obs_type = obs_type

        #self.agent_color = 1


        
        self.max_absolute = max(self.width, self.height)

        if obs_type == "pos_dict":
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
        
        elif obs_type == "flat":
            self.observation_space = spaces.Box(low=0, 
                                                high=self.max_absolute, 
                                                shape=(4,), 
                                                dtype=np.int64)
        
        elif obs_type == "grid":
            self.observation_space = spaces.Box(low=self.min_color, 
                                                high=self.max_color, 
                                                shape=(self.width * self.height,), 
                                                dtype=np.int64)
        
        self.action_space = spaces.Discrete(4)  # R,D,L,U
    
    def reset(self, seed=None, options=None, init_agent_pos = None):
        super().reset(seed=seed, init_agent_pos = init_agent_pos)
        self.steps = 0

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

        move = self._action_to_direction[action]

        new_x = self.agent_pos[0] + move[0]
        new_y = self.agent_pos[1] + move[1]
        if self.acceptable_move(new_x, new_y):
            self.set_agent_position(new_x, new_y)
            self.update_obs_grid()
        else:
            reward = -0.1
            if self.render_mode == "human":
                print("Invalid move attempted: ", self._action_meaning[action])
                print("Agent position remains: ", self.agent_pos)
        
        
        reward = 0
        term = False
        trunc = False
        if np.array_equal(self.agent_pos, self.target_pos):
            reward = 1
            term = True
            if self.render_mode == "human":
                print("Target reached in {} steps!".format(self.steps))

        if self.steps >= self.max_steps:
            trunc = True
            reward = -1
            if self.render_mode == "human":
                print("Max steps reached. Target not reached.")
        
        if self.render_mode is not None:
            if self.render_mode == "human":
                print("Action taken: ", self._action_meaning[action])
                print("New agent position: ", self.agent_pos)
            self.render()

        obs = self.get_obs()
        return obs, reward, term, trunc, {}

    def get_obs(self):
        if self.obs_type == "pos_dict":
            return {'agent_position':self.agent_pos, 
                    'target_position': self.target_pos}
        elif self.obs_type == "flat":
            return np.concatenate([self.agent_pos, self.target_pos])
        elif self.obs_type == "grid":
            return self.obs_grid.flatten()
        
    def update_obs_grid(self):
        super().update_obs_grid()
        x_target, y_target = self.target_pos
        self.obs_grid[y_target, x_target] = self.target_color

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
    width, height = 5, 5
    obstacle_prob = 0.1

    plt.ion()

    for obs_type in ["grid", "pos_dict", "flat"]:
        print(f"Testing observation type: {obs_type}")
    
        env = MultiobsSimpleTargetAgentEnv( width=width, 
                                    height=height, 
                                    obstacle_prob=obstacle_prob, 
                                    obs_type=obs_type,
                                    render_mode="human",
                                    static_obstacles=True,
                                    static_obstacles_seed=40)
        print(env.agent_color, env.target_color, env.free_color, env.obstacle_color)
        
        
        #check_env(env)
        
        obs, _ = env.reset()
        
        # obs, _ = env.reset()
        # print()
        # print("Grid:")
        # print(env.grid)
        # print()
        # print()
        # print("Obs grid:")
        # print(env.obs_grid)
        
        trunc = False
        done = False
        count = 0
        while True:
            count += 1
            if count > 10:
                print("Stopping after 10 steps to avoid infinite loop.")
                break
            action = env.action_space.sample()
            # move = move_toward(obs['agent_position'], obs['target_position'])
            # action = env.direction_to_action[tuple(move)]
            obs, reward, done, trunc, info = env.step(action)
            #input()
            if done:
                print("Episode completed succesfully with reward: ", reward)
                break
            if trunc:
                print("Episode truncated with reward: ", reward)
                break
    
        