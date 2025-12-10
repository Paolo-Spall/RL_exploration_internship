#/usr/bin/python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env 


class ExpGrid2D(gym.Env):
    obstacle_color = 170  # Dark gray for obstacles
    unknown_color = 85  # Light gray for unknown cells
    agent_color = 255  # Black for agent position
    free_color = 0  # White for free cells

    def __init__(self, 
                 width, 
                 height, 
                 obstacle_prob=0.2, 
                 perc_range=1, 
                 render=False,
                 cnn=True):
        super().__init__()
        self.width = width
        self.height = height
        self.obstacle_prob = obstacle_prob
        self.perc_range = perc_range
        self.render = render
        self.cnn = cnn

        self.total_cells = width * height
        self.discovered_cells = 0

        self.action_space = spaces.Discrete(4)  # R,D,L,U
        if cnn:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(height, width, 1), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(height * width,), dtype=np.uint8
            )


        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }
        if self.render:
            self.init_simulation_render()

    ## SETTING THE ENVIRONMENT GRID
    def _generate_grid(self):
        self.grid = np.ones((self.height, self.width), dtype=np.uint8) * self.free_color  # start with all free cells
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.obstacle_prob:
                    self.make_obstacle(x,y)  # 1 represents an obstacle
    
    def _generate_obs_grid(self):
        self.obs_grid = np.ones_like(self.grid) * self.unknown_color

    def init_agent_position(self):
        x = random.randint(0, self.width-1) 
        y = random.randint(0, self.height-1)
        while not self.acceptable_move(x, y):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
        self.set_agent_position(x, y)

    def make_obstacle(self, x, y):
        self.grid[y][x] = 1
        max_obstacle_size = min(self.width, self.height) // 5
        obstacle_size = random.randint(max_obstacle_size//2, max_obstacle_size)
        for i in range(obstacle_size):
            xnew,ynew = self.next_obst_cell(x,y)

            n_iter = 0
            while not self.acceptable_move(xnew, ynew) and n_iter < 20:
                xnew,ynew = self.next_obst_cell(x,y)
                n_iter += 1
            
            while not self.is_in_grid(xnew, ynew):
                xnew,ynew = self.next_obst_cell(x,y)
        
                
            self.grid[ynew][xnew] = self.obstacle_color  # mark as obstacle: dark gray
            x,y = xnew,ynew
    
    def next_obst_cell(self, x,y):
        move = random.randint(0,3)
        if move == 0:
            x += 1
        elif move == 1:
            x -= 1 
        elif move == 2:
            y += 1
        elif move == 3:
            y -= 1
        return x,y

    ## ENVIRONMENT DYNAMICS AND INTERACTION METHODS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_grid()
        self._generate_obs_grid()
        self.init_agent_position()
        self.update_obs_grid()

        if self.render:
            self.display()

        obs = self._get_obs()

        return obs,  {}

    def step(self, action):
        reward = 0
        terminated = False

        move = self._action_to_direction[int(action)]
        newx, newy = self.agent_pos + move

        # move the agent to new position only if inside bounds and not an obstacle
        if self.acceptable_move(newx, newy):
            self.set_agent_position(newx, newy)
            discovered_cells = self.update_obs_grid()
            reward += discovered_cells * 0.1  # reward for discovering new cells
        else:
            reward = -1  # penalty for invalid move
        
        if (self.discovered_cells / self.total_cells) > 0.7:
            terminated = True
            reward += 10  # big reward for completing exploration

        if self.render:
            self.display()
        
        obs = self._get_obs()

        return  obs, reward, bool(terminated) , False, {}

    def _get_obs(self):
        if self.cnn:
            return self.obs_grid[:, :, np.newaxis]
        else:
            return self.obs_grid.flatten()
        

    def update_obs_grid(self):
        discovered_cells = 0
        x = self.agent_pos[0]
        y = self.agent_pos[1]
        r = self.perc_range
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
    
    def set_agent_position(self, x, y):
        self.agent_pos = np.array((x, y))

    ## VALIDATION METHODS
    
    def acceptable_move(self, new_x, new_y):
        if not self.is_in_grid(new_x, new_y):
            return False
        if self.is_obstacle(new_x, new_y):
            return False
        return True
    
    def is_in_grid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, x, y):
        return self.grid[y][x] == self.obstacle_color

    ## RENDERING FUNCTIONS

    def init_simulation_render(self):
        self.fig , (self.ax_env, self.ax_obs) = plt.subplots(1,2, figsize=(10,5))

    def display(self):
        self.ax_env.clear()
        self.ax_obs.clear()
        
        self.grid[self.agent_pos[1]][self.agent_pos[0]] = self.agent_color
        #obs_map[self.agent_pos[1]][self.agent_pos[0]] = self.agent_color
        
        self.ax_env.imshow(self.grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)
        self.ax_obs.imshow(self.obs_grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)

        self.grid[self.agent_pos[1]][self.agent_pos[0]] = self.free_color
        plt.pause(0.1)


if __name__ == "__main__":
    width, height = 36, 36
    obstacle_prob = 0.025
    perc_range = 2
    print("Creating environment...")
    env = ExpGrid2D(width, 
                       height, 
                       obstacle_prob, 
                       perc_range=perc_range, 
                       render=True, 
                       cnn=False)

    check_env(env, warn=True)
    # print("Resetting environment...")
    # env.reset()
    # print("Stepping through the environment...")

    # for _ in range(100):
    #     action = random.randint(0,3)
    #     env.step(action)
