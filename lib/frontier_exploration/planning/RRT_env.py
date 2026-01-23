#/usr/bin/python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from RRT import rrt, render_rrt_path


#class ExpGrid2D(gym.Env):
class ExpGrid2D:
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
                 max_steps = 100,      # or any number you want
                 render_mode=None,
                 policy_type='MlpPolicy'):
        
        self.width = width
        self.height = height
        self.obstacle_prob = obstacle_prob
        self.target_discovery_percent = target_discovery_percent
        self.perc_range = perc_range
        self.render_mode = render_mode
        self.policy_type = policy_type
        self.max_steps = max_steps
        self.current_step = 0

        self.total_cells = width * height
        self.discovered_cells = 0

        #self.action_space = spaces.Discrete(4)  # R,D,L,U
        
        # if self.policy_type == 'CnnPolicy':
        #     self.observation_space = spaces.Box(
        #         low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        #     )
        # elif self.policy_type == 'MlpPolicy':
        #     self.observation_space = spaces.Box(
        #         low=0, high=255, shape=(height * width,), dtype=np.uint8
        #     )
        # elif self.policy_type == 'MultiInputPolicy':
        #     self.observation_space = spaces.Dict({
        #         'grid': spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8),
        #         'position': spaces.Box(low=0, high=max(width, height), shape=(2,), dtype=np.int32)
        #     })


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
        # super().reset(seed=seed)
        self._generate_grid()
        self._generate_obs_grid()
        self.init_agent_position()
        self.update_obs_grid()


        self.current_step = 0

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()

        return obs,  {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        self.current_step += 1

        #move = self._action_to_direction[int(action)]
        move = action
        newx, newy = self.agent_pos + move

        # move the agent to new position only if inside bounds and not an obstacle
        if self.acceptable_move(newx, newy):
            self.set_agent_position(newx, newy)
            discovered_cells = self.update_obs_grid()
            if discovered_cells == 0:
                reward -= 0.1  # small penalty for no new discovery
            else:
                self.find_frontiers()
                self.cluster_frontiers()
                reward += discovered_cells/self.perc_range * 0.1  # reward for discovering new cells
        else:
            reward = -1  # penalty for invalid move
        
        if (self.discovered_cells / self.total_cells) > self.target_discovery_percent:
            terminated = True
            reward += 10  # big reward for completing exploration
        
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 5  # penalty for running out of time

        if self.render_mode == "human":
            print("Action: ", action, "Move: ", move)
            self.render()
        
        obs = self._get_obs()

        return  obs, reward, bool(terminated) , bool(truncated), {}

    def _get_obs(self):
        return self.grid, self.agent_pos
        # if self.policy_type == 'CnnPolicy':
        #     return self.obs_grid[:, :, np.newaxis]
        # elif self.policy_type == 'MlpPolicy':
        #     return self.obs_grid.flatten()
        # elif self.policy_type == 'MultiInputPolicy':
        #     return {
        #         'grid': self.obs_grid[:, :, np.newaxis],
        #         'position': self.agent_pos.copy()
        #     }
        

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

    def render(self):
        self.ax_env.clear()
        self.ax_obs.clear()
        
        
        print("Agent position:", self.agent_pos)
        #obs_map[self.agent_pos[1]][self.agent_pos[0]] = self.agent_color
        
        self.ax_env.imshow(self.grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)
        self.ax_obs.imshow(self.obs_grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)

        self.grid[self.agent_pos[1]][self.agent_pos[0]] = self.free_color


        #plt.pause(0.1)
    
    def rrt(self, start, goal, connect_dist = 1, ed_length = 1.5):
        
        tree1 = []
        tree1.append(start.copy())
        parents1 = [-1]
        tree2 = []
        tree2.append(goal.copy())
        parents2 = [-1]
        
        

        while True:
            
            aux_point = self.random_position()
            child1, parent1 = generate_vertex(tree1[:], aux_point, ed_length)
            if self.check_edge(tree1[parent1].copy(), child1, self.grid, self.obstacle_color):
                tree1.append(child1.copy())
                parents1.append(parent1)

                closer =connected(child1, tree2, connect_dist)
                
                if closer is not False:
                    closests = [len(tree1)-1, closer]
                    break
            child2, parent2 = generate_vertex(tree2[:], aux_point, ed_length)
            if self.check_edge(tree2[parent2].copy(), child2, self.grid, self.obstacle_color):
                tree2.append(child2.copy())
                parents2.append(parent2)
                closer =connected(child2, tree1, connect_dist)
                if closer is not False:
                    closests = [closer, len(tree2)-1]
                    break
        path1 = recur_path(closests[0], parents1, tree1)
        path2 = recur_path(closests[1], parents2, tree2)
        
        path1.reverse()
        path = path1 + path2
        #path.insert(0, start)
        #path.append(goal)
        return path, tree1, tree2
    
    def check_edge(self, start, goal, grid, obst_code):
        while not np.array_equal(start, goal):
            direction = goal - start
            move = manatthan_move(direction)
            start += move
            if self.acceptable_move(start[0], start[1]) is False:
                return False
        return True           

    def random_position(self):
        x = random.randint(0, self.width-1) 
        y = random.randint(0, self.height-1)
        while not self.acceptable_move(x, y):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
        return np.array([x,y])

    def render_rrt_path(self, path, tree1, tree2, start, goal):
        self.fig , (self.ax_env, self.ax_obs) = plt.subplots(1,2, figsize=(10,5))
        
        
        print("Agent position:", self.agent_pos)
        #obs_map[self.agent_pos[1]][self.agent_pos[0]] = self.agent_color
        
        self.ax_env.imshow(self.grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)
        self.ax_obs.imshow(self.obs_grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)
        tree1_arr = np.array(tree1)
        tree2_arr = np.array(tree2)
        path_arr = np.array(path)

        self.ax_env.scatter(tree1_arr[:, 0], tree1_arr[:, 1], c='blue', s=20)
        self.ax_env.scatter(tree2_arr[:, 0], tree2_arr[:, 1], c='orange', s=20)
        self.ax_env.scatter(start[0], start[1], c='green', s=50, marker='o')
        self.ax_env.scatter(goal[0], goal[1], c='red', s=50, marker='x')
        self.ax_env.plot(path_arr[:, 0], path_arr[:, 1], c='red', linewidth=2, marker='>')
        #self.grid[self.agent_pos[1]][self.agent_pos[0]] = self.free_color
        
        self.ax_obs.scatter(tree1_arr[:, 0], tree1_arr[:, 1], c='blue', s=20)
        self.ax_obs.scatter(tree2_arr[:, 0], tree2_arr[:, 1], c='orange', s=20)

        plt.show()

def recur_path(index, parents, tree):
    path = []

    while index != -1 :
        path.append(tree[index])
        index = parents[index]
    return path

def generate_vertex(tree, auxpt, ed_length):
    distances = np.linalg.norm(tree - auxpt, axis=1)
    parent_index = np.argmin(distances)
    direction = (auxpt - tree[parent_index]) / np.linalg.norm(auxpt - tree[parent_index]+1e-6)
    child_exact = direction * ed_length
    child = np.round(tree[parent_index] + child_exact).astype(int)
    return child[:], parent_index

def connected(node, tree, connect_dist):    
    dists = np.linalg.norm(tree - node, axis=1)
    if np.any(dists <= connect_dist):
        return np.argmin(dists)
    return False


        


def target_move(obs):
    agent_position = obs['agent_position']
    centroids = obs['frontier_centroids']
    distances = np.linalg.norm(centroids - agent_position, axis=1)
    target_i = np.argmin(distances)
    target = centroids[target_i]
    delta = (target-agent_position) / np.linalg.norm(target-agent_position)
    return manatthan_move(delta)

def manatthan_move(delta):
        dx, dy = delta
        move = np.array([np.sign(dx) * (abs(dx) >= abs(dy)), 
                         np.sign(dy) * (abs(dy) > abs(dx))
                         ], dtype=int)
        return move

if __name__ == "__main__":
    width, height = 60, 40
    obstacle_prob = 0.001
    
    print("Creating environment...")
    env = ExpGrid2D(width, 
                       height, 
                       obstacle_prob, 
                       render_mode="human", 
                       policy_type='mlp')

    # check_env(env, warn=True)
    print("Resetting environment...")
    obs, _ = env.reset()
    print("Stepping through the environment...")

    start = env.random_position() 
    goal = env.random_position()   

    path, tree1, tree2 = rrt(env.grid, start, goal, obstacle_code=env.obstacle_color)
    
    render_rrt_path(env.grid, path, tree1, tree2, start, goal)

    

    
