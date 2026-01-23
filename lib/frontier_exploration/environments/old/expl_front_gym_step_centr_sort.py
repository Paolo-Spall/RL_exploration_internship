#/usr/bin/python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env 
from lib.utils import greedy_index, step_toward, sort_array_by_distance


class ExplFrontGymStepCentrSort(gym.Env):
    metadata = {"render_modes": ["human"]}
    obstacle_color = 170  # Dark gray for obstacles
    unknown_color = 85  # Light gray for unknown cells
    agent_color = 255  # Black for agent position
    free_color = 0  # White for free cells

    def __init__(self, 
                 width, 
                 height, 
                 obstacle_prob=0.2, 
                 target_discovery_percent=0.7, 
                 perc_range=1, 
                 max_steps = 250,      # or any number you want
                 render_mode=None,
                 policy_type='MlpPolicy',
                 centroids_obs_len=10):
        
        self.width = width
        self.height = height
        self.obstacle_prob = obstacle_prob
        self.target_discovery_percent = target_discovery_percent
        self.perc_range = perc_range
        self.render_mode = render_mode
        self.policy_type = policy_type
        self.max_steps = max_steps
        self.current_step = 0
        self.centroids_obs_len = centroids_obs_len

        self.total_cells = width * height
        self.discovered_cells = 0

        self.action_space = spaces.Discrete(1)  # R,D,L,U
        self.observation_space = spaces.Dict({
            'agent_position': spaces.Box(low=0, high=max(height, width), shape=(2,), dtype=np.int64),
            'frontier_centroids': spaces.Box(low=0, high=max(height, width), shape=(10 * 2,), dtype=np.uint8)
        })


        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }
        if self.render_mode == "human":
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

        self.find_frontiers()
        self.cluster_frontiers()

        self.current_step = 0

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()

        return obs,  {}

    def step(self, action):
        """action: 
                integer, representing the target centroid index to move toward
        """
        reward = 0
        terminated = False
        truncated = False
        self.current_step += 1
        # extracting the target centroid coordinates
        target_centroid = self.obs_centroids[action].copy()
        # computing the next position toward the target centroid
        newx, newy = step_toward(self.agent_pos, target_centroid, manhattan=True)
        # move the agent to new position only if inside bounds
        if self.acceptable_move(newx, newy):
            self.set_agent_position(newx, newy)
            discovered_cells = self.update_obs_grid()
            if discovered_cells == 0:
                # small penalty for no new discovery
                reward -= 1. / self.total_cells 
            else:
                self.find_frontiers()
                self.cluster_frontiers()
                # reward proportional to new discovered cells
                reward += discovered_cells / self.total_cells  
        else:
            reward = -1  # penalty for invalid move
        
        if (self.discovered_cells / self.total_cells) > self.target_discovery_percent:
            terminated = True
            reward += 1  # big reward for completing exploration
        
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 1  # penalty for running out of time

        if self.render_mode == "human":
            print("Action: ", action, "Target centroid: ", target_centroid)
            self.render()
        
        obs = self._get_obs()

        return  obs, reward, bool(terminated) , bool(truncated), {}

    def _get_obs(self):
        
        return {'agent_position': self.agent_pos,
                'frontier_centroids':self.obs_centroids.flatten()}
        

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

    def find_frontiers(self):
        self.frontiers = []
        for y in range(self.height):
            for x in range(self.width):
                if self.obs_grid[y, x] == self.free_color:
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x + dx, y + dy
                        if self.is_in_grid(ny, nx) and self.obs_grid[ny, nx] == self.unknown_color:
                            self.frontiers.append((x, y))
                            break
    

    def cluster_frontiers(self):
        self.clusters = []
        visited = set()

        frontiers = [tuple(f) for f in self.frontiers]

        for f in frontiers:
            if f in visited:
                continue

            cluster = []
            queue = [f]
            visited.add(f)

            while queue:
                cx, cy = queue.pop(0)
                cluster.append((cx, cy))

                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),
                            (-1,-1),(-1,1),(1,-1),(1,1)]:
                    n = (cx + dx, cy + dy)
                    if n in frontiers and n not in visited:
                        visited.add(n)
                        queue.append(n)

            self.clusters.append(np.array(cluster))
        centroids = np.array([c.mean(axis=0) for c in self.clusters])
        
        sorted_centroids = sort_array_by_distance(self, centroids, self.agent_pos)
        

        self.obs_centroids = np.zeros((self.centroids_obs_len , 2), dtype=np.uint8)
        num_centroids = min(len(sorted_centroids), self.centroids_obs_len)
        if num_centroids >0:
            self.obs_centroids[:num_centroids] = sorted_centroids[:num_centroids].copy()

        self.centroids = centroids



    def set_agent_position(self, x, y):
        self.agent_pos = np.array((x, y))

    ## VALIDATION METHODS
    
    def acceptable_move(self, new_x, new_y):
        if not self.is_in_grid(new_x, new_y):
            return False
        # if self.is_obstacle(new_x, new_y):
        #     return False
        return True
    
    def is_in_grid(self, x, y):
        in_grid= (  np.all(0 <= x),
                    np.all(x < self.width),
                    np.all(0 <= y ),
                    np.all(y < self.height) 
                )
        return np.all(in_grid)

    def is_obstacle(self, x, y):
        return self.grid[y][x] == self.obstacle_color

    ## RENDERING FUNCTIONS

    def init_simulation_render(self):
        self.fig , (self.ax_env, self.ax_obs) = plt.subplots(1,2, figsize=(10,5))

    def render(self):
        self.ax_env.clear()
        self.ax_obs.clear()
        
        self.grid[self.agent_pos[1]][self.agent_pos[0]] = self.agent_color
        print("Agent position:", self.agent_pos)
        #obs_map[self.agent_pos[1]][self.agent_pos[0]] = self.agent_color
        
        self.ax_env.imshow(self.grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)
        self.ax_obs.imshow(self.obs_grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)

        self.grid[self.agent_pos[1]][self.agent_pos[0]] = self.free_color

        for c in self.clusters:
            self.ax_obs.scatter(c[:,0], c[:,1], s=10)

        self.ax_obs.scatter(self.centroids[:,0], self.centroids[:,1], c="green", s=80, marker="x")

        plt.pause(0.1)


if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    target_discovery_percent = 0.9
    perc_range = 3
    print("Creating environment...")
    env = ExplFrontGymStepCentrSort(width=width, 
                       height=height, 
                       obstacle_prob=obstacle_prob, 
                       target_discovery_percent=target_discovery_percent,
                       perc_range=perc_range, 
                       render_mode="human", 
                    #    render_mode=False, 
                       policy_type='mlp')

    # check_env(env, warn=True)
    # print("Env checked.")
    # exit()
    print("Resetting environment...")
    obs, _ = env.reset()
    print("Stepping through the environment...")

    term = False

    while not term:
        centroids = obs['frontier_centroids'].reshape(-1,2)
        agent_pos = obs['agent_position']
        action = greedy_index(centroids, agent_pos)
        obs, reward, term,  trunc, _ = env.step(action)
