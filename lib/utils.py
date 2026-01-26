import numpy as np

def closest(points, position):
    dists = np.linalg.norm(points - position, axis=1)
    exact_target = points[np.argmin(dists)]
    target = np.round(exact_target).astype(int)
    return target

def greedy_move(centroids, agent_position):
    distances = np.linalg.norm(centroids - agent_position, axis=1)
    target_i = np.argmin(distances)
    target = centroids[target_i]
    delta = (target-agent_position) / (np.linalg.norm(target-agent_position) + 1e-8)
    return manatthan_move(delta)

def greedy_index(points_list, agent_position):
    distances = np.linalg.norm(points_list - agent_position, axis=1)
    target_i = np.argmin(distances)
    return target_i

def manatthan_move(delta):
        dx, dy = delta
        try:
            move = np.array([np.sign(dx) * (abs(dx) >= abs(dy)), 
                            np.sign(dy) * (abs(dy) > abs(dx))
                            ], dtype=int)
        except ValueError as e:
            print("Error in delta:", delta)
            raise e
        return move

def step_toward(start, target, manhattan=True):
        """Computes the next position from start toward target by one step."""
        delta = target - start
        if manhattan:
            move = manatthan_move(delta)
        else:
            move = np.sign(delta)
        next_position  = start + move
        return next_position

def move_toward(start, target, manhattan=True):
        """Computes the next position from start toward target by one step."""
        delta = target - start
        if manhattan:
            move = manatthan_move(delta)
        else:
            move = np.sign(delta)
        return move

def sort_array_by_distance(points_array, point):
    if len(points_array) == 0:
        return points_array
    distances = np.linalg.norm(points_array - point, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_points = points_array[sorted_indices]
    return sorted_points

def find_agent(grid, agent_color=255):
    positions = np.argwhere(grid == agent_color)
    if positions.size == 0:
        return None
    return positions[0][::-1]  # return (x,y)