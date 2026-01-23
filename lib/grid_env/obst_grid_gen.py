#/usr/bin/python3
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_grid(width, 
                height, 
                obstacle_prob=0.2, 
                obstacle_color = 255,
                free_color = 0):
    grid = np.ones((height, width), dtype=np.uint8) * free_color  # start with all free cells
    for y in range(height):
        for x in range(width):
            if random.random() < obstacle_prob:
                make_obstacle(grid, x, y, obstacle_color)  # 1 represents an obstacle
    return grid

def make_obstacle(grid, x, y, obstacle_color):
    grid[y][x] = obstacle_color
    max_obstacle_size = min(grid.shape[1], grid.shape[0]) // 5
    obstacle_size = random.randint(max_obstacle_size//2, max_obstacle_size)
    for i in range(obstacle_size):
        xnew,ynew = next_obst_cell(x,y)

        n_iter = 0
        while not acceptable_move(grid, xnew, ynew, obstacle_color) and n_iter < 20:
            xnew,ynew = next_obst_cell(x,y)
            n_iter += 1
        
        while not is_in_grid(grid, xnew, ynew):
            xnew,ynew = next_obst_cell(x,y)
    
            
        grid[ynew][xnew] = obstacle_color  # mark as obstacle: dark gray
        x,y = xnew,ynew

def next_obst_cell(x,y):
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


def acceptable_move(grid, new_x, new_y, obstacle_color):
    if not is_in_grid(grid, new_x, new_y):
        return False
    if is_obstacle(grid, new_x, new_y, obstacle_color):
        return False
    return True

def is_in_grid(grid, x, y):
    height, width = grid.shape
    in_grid= (  np.all(0 <= x),
                np.all(x < width),
                np.all(0 <= y ),
                np.all(y < height) 
            )
    return np.all(in_grid)

def is_obstacle(grid, x, y, obstacle_color):
    return grid[y][x] == obstacle_color

    ## RENDERING FUNCTIONS

def init_simulation_render():
    fig , ax_env= plt.subplots(1,1, figsize=(10,5))
    return fig, ax_env

def render_grid(grid, ax_env, min_color=0, max_color=255):
    
    ax_env.imshow(grid, cmap='Greys', vmin=min_color, vmax=max_color)#, origin='upper', vmin=0, vmax=255)
    ax_env.set_title("Environment Grid")

if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    print("Creating Grid...")
    grid = generate_grid(width=width, 
                        height=height, 
                        obstacle_prob=obstacle_prob, 
                        obstacle_color = 255,
                        free_color = 0
                    )


    print("Rendering the Grid...")
    fig , ax_env= plt.subplots(1,1, figsize=(10,5))
    render_grid(grid, ax_env)
    plt.show()
    