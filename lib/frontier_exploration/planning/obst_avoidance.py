#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    sys.path.append(".")
from lib.frontier_exploration.planning.planning_utils import is_in_grid, is_obstacle, render_path
from lib.utils import find_agent

class ObstAvoidance:
    def __init__(self, obst_code, moves):
        self.moves = moves
        self.obst_code = obst_code
        self.path = None
        self.target = None
        
    
    def compute_path(self, grid, start, goal):
        path = compute_path(grid, start, goal, self.obst_code, self.moves)
        self.path = path
        self.target = goal
        
        return path

    def active(self, target_centroid):
        if self.path:
            if np.array_equal(self.target, target_centroid):
                return True
            else:
                self.path = None
                self.target = None
                return False
        return False
        
    
    def get_next(self):
        if self.path:
            next_pos = self.path.pop(0)
            return next_pos
        return None

def compute_path(grid, start, goal, obst_code, moves):
    path = [start.copy()]
    current = start.copy()
    count1 = 0

    # Main loop to reach the goal point by point
    while not arrived(current, goal):
        
        count1 += 1
        if count1 == 1000:
            print("Warning: external loop exceeded 1000 iterations.")
            print("len path:", len(path))

        possibles = possible_moves(current, moves, grid, obst_code)# set of possible nodes
        if not possibles:
            return None  # No path found
        
        # Choose the move that minimizes the distance to the goal
        count2 = 0
        while True:
            count2 += 1
            if count2 == 100:
                print("Warning: internal loop exceeded 100 iterations.")

            if not possibles:
                return path
            try:
                min_index = heuristic(possibles, goal)
            except ValueError:
                print("Possibles:", possibles)
                print("Current:", current)
                print("Goal:", goal)
                print("Path so far:", path)
                raise
            next_pos = possibles.pop(min_index)
            if not already_in_path(next_pos, path):
                break

        path.append(next_pos)
        current = next_pos
    path.pop(0)  # remove the starting position
    return path

def already_in_path(position, path):
    if path:
        if np.any(np.all(path == position, axis=1)):
            return True
    return False

def arrived(position, goal):
    return np.all(np.abs(position - goal) < 0.5)



def heuristic(moves, goal):
    moves = np.array(moves)
    dists = np.linalg.norm( moves - goal, axis=1 )
    min_index = np.argmin(dists)
    return min_index

def possible_moves( position, moves , grid, obst_code):
    possible = []
    for move in moves:
        new_pos = position + move
        if is_in_grid(grid, new_pos[0], new_pos[1]) and not is_obstacle(grid, new_pos[0], new_pos[1], obst_code ):
            possible.append(new_pos)
    return possible

if __name__ == "__main__":
    moves = [np.array([1, 0]),   # Move right (positive x)
             np.array([0, 1]),   # Move up (positive y)
             np.array([-1, 0]),  # Move left (negative x)
             np.array([0, -1])]  # Move down (negative y)
    import joblib


    file ='frontiers_grid.joblib'
    print("Rendering from file:", file)
    obs_grid, clusters, centroids = joblib.load(file)
    agent = find_agent(obs_grid, agent_color=255)
    goal = np.array([9, 17])
    path = compute_path(obs_grid, agent, goal, 170, moves)
    render_path(obs_grid, path, agent, goal)
    plt.show()