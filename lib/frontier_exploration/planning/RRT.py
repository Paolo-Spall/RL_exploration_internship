#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn import tree

def rrt(grid, start, goal, free_code, connect_dist = 1, ed_length = 1.5, reverse=False, manhattan=False):
    
    tree1 = []
    tree1.append(start.copy())
    parents1 = [-1]
    tree2 = []
    tree2.append(goal.copy())
    parents2 = [-1]
    height, width = grid.shape
    
    step = 0
    while True:
        step += 1
        if step > 1000:
            print("RRT: Max steps reached, no path found")
            input()
            dists = np.linalg.norm(tree1- goal, axis=1)
            closest = np.argmin(dists)
            path1 = recur_path(closest, parents1, tree1)
            return path1.reverse(), tree1, tree2
        aux_point = random_point(grid)
        child1, parent1 = generate_vertex(tree1[:], aux_point, ed_length)
        if check_edge(grid, tree1[parent1].copy(), child1, free_code, reverse, manhattan):
            tree1.append(child1.copy())
            parents1.append(parent1)

            closer =connected(child1, tree2, connect_dist)
            
            if closer is not False:
                closests = [len(tree1)-1, closer]
                break
        child2, parent2 = generate_vertex(tree2[:], aux_point, ed_length)
        if check_edge(grid, tree2[parent2].copy(), child2, free_code, reverse, manhattan):
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

def render_rrt_path(grid, path, tree1, tree2, start, goal):
    #fig , (ax_env, ax_obs) = plt.subplots(1,2, figsize=(10,5))
    fig , ax_env = plt.subplots(1,1, figsize=(10,5))
    
    
    print("Start pos:", start)
    print("Goal pos:", goal)
    #obs_map[self.agent_pos[1]][self.agent_pos[0]] = self.agent_color
    
    ax_env.imshow(grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)
    #self.ax_obs.imshow(self.obs_grid, cmap='Greys')#, origin='upper', vmin=0, vmax=255)
    tree1_arr = np.array(tree1)
    tree2_arr = np.array(tree2)
    path_arr = np.array(path)

    ax_env.scatter(tree1_arr[:, 0], tree1_arr[:, 1], c='blue', s=20)
    ax_env.scatter(tree2_arr[:, 0], tree2_arr[:, 1], c='orange', s=20)
    ax_env.scatter(start[0], start[1], c='green', s=50, marker='o')
    ax_env.scatter(goal[0], goal[1], c='red', s=50, marker='x')
    ax_env.plot(path_arr[:, 0], path_arr[:, 1], c='red', linewidth=2, marker='>')
    #self.grid[self.agent_pos[1]][self.agent_pos[0]] = self.free_color
    
    # self.ax_obs.scatter(tree1_arr[:, 0], tree1_arr[:, 1], c='blue', s=20)
    # self.ax_obs.scatter(tree2_arr[:, 0], tree2_arr[:, 1], c='orange', s=20)

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

def manatthan_move(delta):
        dx, dy = delta
        move = np.array([np.sign(dx) * (abs(dx) >= abs(dy)), 
                         np.sign(dy) * (abs(dy) > abs(dx))
                         ], dtype=int)
        return move


if __name__ == "__main__":
    # Example usage
    grid = np.zeros((20, 20), dtype=int)
    grid[7:19, 10] = 1  # Add an obstacle
    start = np.array([2, 2])
    goal = np.array([17, 17])
    path, tree1, tree2 = rrt(grid, start, goal, free_code=0)
    render_rrt_path(grid, path, tree1, tree2, start, goal)