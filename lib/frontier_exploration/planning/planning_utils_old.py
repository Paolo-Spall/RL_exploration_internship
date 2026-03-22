#!/usr/bin/python3
import random
import numpy as np
import matplotlib.pyplot as plt

from lib.utils import manatthan_move

def check_edge(grid, start, goal, free_code, reverse=False, manhattan=False):
    while not np.array_equal(start, goal):
        direction = goal - start
        if manhattan:
            move = manatthan_move(direction)
        else:
            move = np.sign(direction)
        start += move
        if acceptable_edge_move(grid, start[0], start[1], free_code, reverse=reverse) is False:
            return False
    return True           

def straight_path(start, goal, manhattan=False):
    path = []
    while not np.array_equal(start, goal):
        direction = goal - start
        if manhattan:
            move = manatthan_move(direction)
        else:
            move = np.sign(direction)
        start += move
        path.append(start.copy())

    return path 

def random_point(grid):
    height, width = grid.shape
    x = random.randint(0, width-1) 
    y = random.randint(0, height-1)
    while not is_in_grid(grid, x, y):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
    return np.array([x,y])

def acceptable_edge_move(grid, new_x, new_y, free_code, reverse=False):
        if not is_in_grid(grid, new_x, new_y):
            return False
        if not is_free(grid, new_x, new_y, free_code, reverse):
            return False
        return True
    
def is_in_grid(grid, x, y):
    height, width = grid.shape
    return 0 <= x < width and 0 <= y < height

def is_obstacle(grid, x, y, obstacle_code, reverse=False):
    if reverse:
        return grid[x][y] == obstacle_code
    return grid[y][x] == obstacle_code


def is_free(grid, x, y, free_code, reverse=False):
    if reverse:
        return grid[x][y] == free_code
    return grid[y][x] == free_code

def render_path(grid, path, start, goal):
    #fig , (ax_env, ax_obs) = plt.subplots(1,2, figsize=(10,5))
    fig , ax_env = plt.subplots(1,1, figsize=(10,5))
    
    
    print("Start pos:", start)
    print("Goal pos:", goal)
    #obs_map[self.agent_pos[1]][self.agent_pos[0]] = self.agent_color
    
    ax_env.imshow(grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)
    #self.ax_obs.imshow(self.obs_grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)

    path_arr = np.concatenate ( (np.array([start]),np.array(path)), axis=0)

    ax_env.scatter(start[0], start[1], c='black', s=100, marker='o')
    ax_env.plot(path_arr[:, 0], path_arr[:, 1], c='blue', linewidth=2, )#marker='>')
    ax_env.scatter(goal[0], goal[1], c='red', s=200, marker='*')
    #self.grid[self.agent_pos[1]][self.agent_pos[0]] = self.free_color

