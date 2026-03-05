#/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env 

if __name__ == "__main__":
    import sys
    sys.path.append(".")

from lib.grid_env.obst_grid_agent_expl_env import ObstGridAgentExplEnv
from lib.rendering_utils import fig_to_rgb
from lib.utils import move_toward

class MultiobsSimpleAgentExplorationEnv(ObstGridAgentExplEnv):
    unknown_color = 2  # Dark gray for obstacles
    agent_color = 1  # Black for agent position
    obstacle_color = -1  # Dark gray for obstacles
    free_color = 0
    min_color = -1
    max_color = 2

    def __init__(self, 
                 max_steps=500, 
                 obs_type = "pos_dict",
                 perception_range=1,
                 target_discovery_percent=0.7,
                 *args, **kwargs):
        
        super().__init__(perception_range=perception_range, *args, **kwargs)

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
        self.perception_range = perception_range
        self.target_discovery_percent = target_discovery_percent
        #self.agent_color = 1

        
        self.steps = 0
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
        self.discovered_cells = 0
        self.update_obs_grid()
        if self.render_mode is not None:
            self.render()
        
        obs = self.get_obs()

        return obs,  {}
    
    def step(self, action):
        self.steps += 1
        reward = 0

        move = self._action_to_direction[action]

        new_x = self.agent_pos[0] + move[0]
        new_y = self.agent_pos[1] + move[1]

        if self.acceptable_move(new_x, new_y):

            self.set_agent_position(new_x, new_y)
            discovered_cells = self.update_obs_grid()
            #self.discovered_cells += discovered_cells

            if self.render_mode == "human":
                print("Action taken: ", self._action_meaning[action])
                print("New agent position: ", self.agent_pos)

            if discovered_cells == 0:
                # small penalty for no new discovery
                # reward -= (2.* self.perception_range +1) / self.total_cells
                reward = -0.1
                if self.render_mode == "human":
                    print(f"No new cells discovered.")
            else:
                # reward proportional to new discovered cells
                # reward += discovered_cells / self.total_cells 
                reward = 0.1
        else:
            reward = -0.1
            if self.render_mode == "human":
                print("Invalid move attempted: ", self._action_meaning[action])
                print("Agent position remains: ", self.agent_pos)
        
        
        
        term = False
        trunc = False
        if (self.discovered_cells / self.total_cells) > self.target_discovery_percent:
            reward += 1
            term = True
            if self.render_mode == "human":
                print(f"Exploration completed in {self.steps} steps!")

        if self.steps >= self.max_steps:
            trunc = True
            reward -= 1
            if self.render_mode == "human":
                print("Max steps reached. Exploration failed.")
        
        if self.render_mode is not None:
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
        return super().update_obs_grid()

    def render(self):
        super().render()
        if self.render_mode == "human":
            plt.pause(0.1)
        elif self.render_mode == "rgb_array":
            return fig_to_rgb(self.fig)
    

    

if __name__ == "__main__":
    width, height = 5, 5
    obstacle_prob = 0.0
    perc_range = 0

    plt.ion()

    for obs_type in ["grid","grid"]:#, "pos_dict", "flat"]:
        print(f"Testing observation type: {obs_type}")
    
        env = MultiobsSimpleAgentExplorationEnv( width=width, 
                                    height=height, 
                                    obstacle_prob=obstacle_prob,
                                    perception_range=perc_range,
                                    target_discovery_percent=0.9,
                                    obs_type=obs_type,
                                    render_mode="human",)
                                    # static_obstacles=True,
                                    # static_obstacles_seed=40)
        print(env.agent_color, env.unknown_color, env.free_color, env.obstacle_color)
        
        
        # check_env(env)
        # print("Environment check passed")
        # exit()
        
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
            if count > 30:
                print("Stopping after 30.")
                break
            action = env.action_space.sample()
            # move = move_toward(obs['agent_position'], obs['target_position'])
            # action = env.direction_to_action[tuple(move)]
            obs, reward, done, trunc, info = env.step(action)
            print("0bs:")
            print(obs)
            input("Press Enter to continue...")
            print(f"Reward: {reward}")
            #input()
            if done:
                print("Episode completed succesfully with reward: ", reward)
                break
            if trunc:
                print("Episode truncated with reward: ", reward)
                break
    
        