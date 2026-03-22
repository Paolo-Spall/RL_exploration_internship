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
    """
    Render a path on a grid with professional thesis-quality visualization.
    
    Parameters:
    -----------
    grid : np.ndarray
        2D grid where occupied cells = 1 (black) and free cells = 0 (white)
    path : list
        List of waypoints representing the planned path
    start : np.ndarray
        Starting position [x, y]
    goal : np.ndarray
        Goal/target position [x, y]
    """
    # Create figure with high DPI for thesis quality
    fig, ax_env = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
    
    print("Start pos:", start)
    print("Goal pos:", goal)
    
    # Display grid with meaningful colors
    # Create a custom visualization of the grid
    ax_env.imshow(grid, cmap='Greys', origin='upper', vmin=0, vmax=1)
    
    # Add grid lines for better visibility
    height, width = grid.shape
    for i in np.arange(-0.5, height, 1):
        ax_env.axhline(y=i, color='gray', linewidth=0.5, alpha=0.3)
    for j in np.arange(-0.5, width, 1):
        ax_env.axvline(x=j, color='gray', linewidth=0.5, alpha=0.3)
    
    # Prepare path array with start point
    path_arr = np.concatenate((np.array([start]), np.array(path)), axis=0)
    
    # Plot the path
    ax_env.plot(path_arr[:, 0], path_arr[:, 1], c='#2E86AB', linewidth=3, 
                label='Path', zorder=3, marker='o', markersize=4, markevery=max(1, len(path_arr)//10))
    
    # Plot start position (agent)
    ax_env.scatter(start[0], start[1], c='#06A77D', s=300, marker='o', 
                   edgecolors='black', linewidth=2, label='Agent', zorder=4)
    
    # Plot goal position (target)
    ax_env.scatter(goal[0], goal[1], c='#D62828', s=400, marker='*', 
                   edgecolors='black', linewidth=1.5, label='Target position', zorder=4)
    
    # Create custom legend for grid cells
    from matplotlib.patches import Patch
    legend_elements = [
        ax_env.get_legend_handles_labels()[0][0],  # Path
        ax_env.get_legend_handles_labels()[0][1],  # Agent
        ax_env.get_legend_handles_labels()[0][2],  # Target position
        Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='Free cells'),
        Patch(facecolor='black', edgecolor='gray', linewidth=0.5, label='Occupied cells')
    ]
    
    ax_env.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                  framealpha=0.95, edgecolor='black')
    
    # Set title and labels
    ax_env.set_title("A* obstacle avoidance path planner", fontsize=16, fontweight='bold', pad=20)
    ax_env.set_xlabel('X coordinate (cells)', fontsize=12, fontweight='bold')
    ax_env.set_ylabel('Y coordinate (cells)', fontsize=12, fontweight='bold')
    
    # Set axis limits and ticks for better appearance
    ax_env.set_xlim(-0.5, width - 0.5)
    ax_env.set_ylim(height - 0.5, -0.5)
    ax_env.set_xticks(np.arange(0, width, max(1, width // 10)))
    ax_env.set_yticks(np.arange(0, height, max(1, height // 10)))
    ax_env.tick_params(labelsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax_env

