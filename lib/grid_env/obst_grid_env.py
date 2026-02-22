#/usr/bin/python3
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class ObstGridEnv(gym.Env):
    """Environment representing a 2D grid with obstacles."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    obstacle_color = 170  # Dark gray for obstacles
    unknown_color = 85  # Light gray for unknown cells
    agent_color = 255  # Black for agent position
    free_color = 0  # White for free cells
    min_color = 0
    max_color = 255

    def __init__(self, 
                 width, 
                 height, 
                 obstacle_prob=0.2,
                 render_mode=None):
        
        self.width = width
        self.height = height
        self.obstacle_prob = obstacle_prob
        self.render_mode = render_mode

        self.total_cells = width * height
        self.discovered_cells = 0


        if self.render_mode is not None:#== "human":
            
            self.init_simulation_render()

    ## SETTING THE ENVIRONMENT GRID
    def _generate_grid(self):
        self.grid = np.ones((self.height, self.width), dtype=np.uint8) * self.free_color  # start with all free cells
        for y in range(self.height):
            for x in range(self.width):
                if self.np_random.random() < self.obstacle_prob:
                    self.make_obstacle(x,y)  # 1 represents an obstacle
    
    def _generate_obs_grid(self):
        self.obs_grid = np.ones_like(self.grid) * self.unknown_color


    def make_obstacle(self, x, y):
        self.grid[y][x] = 1
        max_obstacle_size = min(self.width, self.height) // 5
        obstacle_size = self.np_random.integers(max_obstacle_size//2, max_obstacle_size+1)
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
        move = self.np_random.integers(0,4)
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

    
    def acceptable_move(self, new_x, new_y):
        if not self.is_in_grid(new_x, new_y):
            return False
        if self.is_obstacle(new_x, new_y):
            return False
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
        self.fig , self.ax_env = plt.subplots(1,1, figsize=(10, 5), dpi=48)

    def render(self):
        self.ax_env.clear()
        
        self.ax_env.imshow(self.grid, cmap='Greys', vmin=self.min_color, vmax=self.max_color)#, origin='upper', vmin=0, vmax=255)


if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    print("Creating environment...")
    env = ObstGridEnv(width=width, 
                       height=height, 
                       render_mode="human", 
                       obstacle_prob=obstacle_prob
                    )

    print("Resetting environment...")
    env.reset(seed=42)
    print("Rendering the environment...")
    env.render()
    plt.show()
    