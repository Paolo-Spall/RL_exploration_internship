import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# -----------------------
# Environment definition
# -----------------------
H, W = 30, 30

# Ground-truth map (unknown to robot)
gt_map = np.zeros((H, W), dtype=int)

# Add obstacles
gt_map[10:15, 5:20] = 1
gt_map[20, 10:25] = 1

# Robot belief map
belief = -np.ones((H, W), dtype=int)

robot_pos = np.array([2, 2])
sensor_range = 3

# -----------------------
# Functions
# -----------------------
def in_bounds(p):
    return 0 <= p[0] < H and 0 <= p[1] < W

def sense(robot_pos):
    """Reveal cells around robot"""
    for dx in range(-sensor_range, sensor_range + 1):
        for dy in range(-sensor_range, sensor_range + 1):
            p = robot_pos + np.array([dx, dy])
            if in_bounds(p):
                belief[p[0], p[1]] = gt_map[p[0], p[1]]

def find_frontiers():
    frontiers = []
    for x in range(H):
        for y in range(W):
            if belief[x, y] == 0:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x + dx, y + dy
                    if in_bounds((nx, ny)) and belief[nx, ny] == -1:
                        frontiers.append((x, y))
                        break
    return np.array(frontiers)

def cluster_frontiers(frontiers):
    clusters = []
    visited = set()

    frontiers = [tuple(f) for f in frontiers]

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

        clusters.append(np.array(cluster))

    return clusters


def step_toward(target):
    global robot_pos
    delta = target - robot_pos
    move = np.sign(delta)
    robot_pos += move

# -----------------------
# Exploration loop
# -----------------------
plt.ion()

for step in range(200):
    sense(robot_pos)
    frontiers = find_frontiers()

    if len(frontiers) == 0:
        print("Exploration complete")
        break

    # # Choose nearest frontier
    # dists = np.linalg.norm(frontiers - robot_pos, axis=1)
    # target = frontiers[np.argmin(dists)]

    clusters = cluster_frontiers(frontiers)

    # Compute centroids
    centroids = np.array([c.mean(axis=0) for c in clusters])

    # Choose nearest cluster centroid
    dists = np.linalg.norm(centroids - robot_pos, axis=1)
    exact_target = centroids[np.argmin(dists)]
    target = np.round(exact_target).astype(int)



    if np.array_equal(target, robot_pos):
        # fallback: nearest frontier in that cluster
        cluster = clusters[np.argmin(dists)]
        d = np.linalg.norm(cluster - robot_pos, axis=1)
        target = cluster[np.argmin(d)]




    step_toward(target)

    # Visualization
    plt.clf()
    plt.title(f"Step {step}")
    plt.imshow(belief, cmap="gray_r", vmin=-1, vmax=1)
    plt.scatter(robot_pos[1], robot_pos[0], c="red", s=50)
    for c in clusters:
        plt.scatter(c[:,1], c[:,0], s=10)

    plt.scatter(centroids[:,1], centroids[:,0], c="green", s=80, marker="x")


    # if len(frontiers) > 0:
    #     plt.scatter(frontiers[:,1], frontiers[:,0], c="blue", s=10)
    plt.pause(0.1)

plt.ioff()
plt.show()
