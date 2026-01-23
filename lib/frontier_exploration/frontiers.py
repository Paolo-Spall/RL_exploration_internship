#!/usr/bin/python3

import numpy as np

if __name__ == "__main__":
    import sys
    sys.path.append(".")

from lib.grid_env.obst_grid_gen import is_in_grid
from lib.utils import sort_array_by_distance

class FrontierDetector:
    def __init__(self,
                 height,
                 width,
                 free_color=0, 
                 unknown_color=255,
                 centroids_obs_len=10,
                 max_cluster_size=10,
                 sorting=True,
                 reverse=False ):
        
        self.free_color = free_color
        self.unknown_color = unknown_color
        self.centroids_obs_len = centroids_obs_len
        self.max_cluster_size = max_cluster_size
        self.sorting = sorting
        self.reverse = reverse

        self.max_relative = max(height, width)
        self.max_distance = int(np.sqrt(height**2 + width**2))

    def detect(self, obs_grid, agent_pos=np.array([0,0])):
        frontiers = find_frontiers(obs_grid, self.free_color, self.unknown_color)
        centroids, clusters = cluster_frontiers(frontiers, 
                                                max_cluster_size=self.max_cluster_size)
        if self.sorting:
            centroids = sort_array_by_distance(centroids, agent_pos)
            if self.reverse:
                centroids = centroids[::-1]
        if centroids.size > 0:
            relative_centroids = centroids - agent_pos
            relative_distances = np.linalg.norm(relative_centroids, axis=1)
        else:
            relative_centroids = centroids.copy()
            relative_distances = np.array([])
        self.centroids = centroids
        self.pad_centroids = pad_obs_array(centroids, 
                                       target_shape=(self.centroids_obs_len, 2))
        self.relative_centroids = pad_obs_array(relative_centroids, 
                                                target_shape=(self.centroids_obs_len, 2),
                                                value=self.max_relative)
        self.relative_distances = pad_obs_array(relative_distances, 
                                                target_shape=(self.centroids_obs_len,),
                                                value=self.max_distance)
        self.clusters = clusters


def find_frontiers(obs_grid, free_color, unknown_color):
    height, width = obs_grid.shape
    frontiers = []
    for y in range(height):
        for x in range(width):
            if obs_grid[y, x] == free_color:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x + dx, y + dy
                    if is_in_grid(obs_grid, ny, nx) and obs_grid[ny, nx] == unknown_color:
                        frontiers.append((x, y))
                        break
    return np.array(frontiers)


def cluster_frontiers(frontiers, max_cluster_size=5):
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
        
        subclusters = split_cluster(np.array(cluster), 
                                    max_cluster_size=max_cluster_size)
        clusters.extend(subclusters)

    centroids = np.array([c.mean(axis=0) for c in clusters])

    return centroids, clusters

def split_clusters_list(clusters, max_cluster_size=5):
    splitted = []
    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            splitted.append(cluster)
        else:
            splitted.extend( split_cluster(cluster, max_cluster_size) )
    return splitted

def split_cluster(cluster, max_cluster_size=5):
    if len(cluster) <= max_cluster_size:
        return [cluster]
    
    pmean = cluster.mean(axis=0)
    xmax, ymax = cluster.max(axis=0)
    xmin, ymin = cluster.min(axis=0)
    iaxis =  np.argmax( [xmax - xmin, ymax - ymin] )
    group1 = cluster[ cluster[:,iaxis] <= pmean[iaxis] ]
    group2 = cluster[ cluster[:,iaxis] > pmean[iaxis] ]

    return clusterify(group1, max_cluster_size) + clusterify(group2, max_cluster_size)

def clusterify(group, max_cluster_size):
    #remaining = set(map(tuple, group))
    remaining = group.tolist()
    subclusters = []
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]
    #neighbors = list(map(np.array,neighbors))
    
    while remaining:
        f = remaining.pop()
        cluster = []
        queue = [f]

        while queue:
            point = queue.pop(0)
            cluster.append(point)
            for x,y in neighbors:
                neigh = [point[0]+x, point[1]+y]
                if neigh in remaining:
                    remaining.remove(neigh)
                    queue.append(neigh)
        if len(cluster) > max_cluster_size:
            subclusters.extend( split_cluster( np.array(cluster), max_cluster_size) )
        else:
            subclusters.append(np.array(cluster))
    return subclusters

def pad_obs_array(array, target_shape=(10,2), value=0):
    padded = np.full(target_shape, value, dtype=np.int64)
    if array.size == 0:
        return padded
    
    num_centroids = min( array.shape[0], target_shape[0])
    padded[:num_centroids] = array[:num_centroids]
    return padded

def render(ax, grid, centroids, clusters):
    ax.imshow(grid, cmap='gray')
    for c in clusters:
        ax.scatter(c[:,0], c[:,1], s=10)
    ax.scatter(centroids[:,0], centroids[:,1], c="green", s=80, marker="x")

def find_agent(grid, agent_color=255):
    positions = np.argwhere(grid == agent_color)
    if positions.size == 0:
        return None
    return positions[0][::-1]  # return (x,y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import joblib


    for file in ['frontiers_grid.joblib',
                 'frontiers_grid_1.joblib',
                 'frontiers_grid_2.joblib']:
        print("Rendering from file:", file)
        obs_grid, clusters, centroids = joblib.load(file)

        fig, (ax_orig, ax_split, ax_fromscratch) = plt.subplots(1,3)

        #titoling the plots
        ax_orig.set_title("Not splitted")
        render(ax_orig, obs_grid, centroids, clusters)
        

        ax_split.set_title("Original clusters splitted")
        split_clusters = split_clusters_list(clusters, max_cluster_size=15)
        split_centroids = np.array([c.mean(axis=0) for c in split_clusters])
        render(ax_split, obs_grid, split_centroids, split_clusters)


        ax_fromscratch.set_title("Computed with splitting")
        agent_pos = find_agent(obs_grid, agent_color=255)
        
        detector = FrontierDetector(obs_grid.shape[0],
                        obs_grid.shape[1],
                        free_color=0, 
                        unknown_color=85,
                        max_cluster_size=15)
        
        detector.detect(obs_grid, agent_pos=agent_pos)
        render(ax_fromscratch, obs_grid, detector.centroids, detector.clusters)

        plt.show()
    