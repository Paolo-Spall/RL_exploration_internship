#/usr/bin/python3
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env 
if __name__ == "__main__":
    import sys
    sys.path.append(".")
from lib.grid_env.obst_grid_agent_expl_env import ObstGridAgentExplEnv
from lib.utils import greedy_index, step_toward, sort_array_by_distance
from lib.frontier_exploration.frontiers import find_frontiers, cluster_frontiers, pad_obs_array


class ExplFrontStepDistancesEnv(ObstGridAgentExplEnv):
    """Gym environment for frontier-based exploration.
       - Returns centroids of frontier clusters as observation.(Can be sorted and reversed))
       - Action is the index of the target centroid to move toward.
       - Step update every sigle-cell move toward the target centroid.
       """
    def __init__(self, 
                 width, 
                 height, 
                 obstacle_prob=0.2, 
                 perc_range=3, 
                 render_mode=None,
                 target_discovery_percent=0.7, 
                 max_steps = 250,
                 centroids_obs_len=10,
                 sorting=True,
                 reverse=False):
        
        super().__init__(perception_range=perc_range, 
                         width=width, 
                         height=height, 
                         obstacle_prob=obstacle_prob)
        
        self.height = height
        self.width = width
        self.target_discovery_percent = target_discovery_percent
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.centroids_obs_len = centroids_obs_len
        self.sorting = sorting
        self.reverse = reverse

        self.total_cells = width * height
        self.discovered_cells = 0

        self.action_space = spaces.Discrete(centroids_obs_len)

        # Observations: relative distances between agent and frontier centroids
        self.observation_space = spaces.Box(low=0, high=height+width, shape=(centroids_obs_len,), dtype=np.uint8)
    

        if self.render_mode == "human":
            self.init_simulation_render()

    ## ENVIRONMENT DYNAMICS AND INTERACTION METHODS

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        
        self.update_obs_grid()

        frontiers = find_frontiers(self.obs_grid, self.free_color, self.unknown_color)
        centroids, self.clusters = cluster_frontiers(frontiers)
        self.centroids = centroids
        if self.sorting:
            centroids = sort_array_by_distance(self, centroids, self.agent_pos)
            if self.reverse:
                centroids = centroids[::-1]
        if centroids.shape[0] == 0:
            distances = np.array([])
        else:
            distances = np.linalg.norm(centroids - self.agent_pos, axis=1)
        self.obs_centroids = pad_obs_array(centroids, 
                                           target_shape=(self.centroids_obs_len, 2))
        self.obs_distances = pad_obs_array(distances, 
                                           target_shape=(self.centroids_obs_len,), 
                                           value=self.height+self.width)

        self.current_step = 0

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        return obs,  {}


    def step(self, action):
        """action: 
                integer, representing the target centroid index to move toward
        """
        reward = 0
        terminated = False
        truncated = False
        self.current_step += 1


        # extracting the target centroid coordinates
        target_centroid = self.obs_centroids[action].copy()

        # computing the next position toward the target centroid
        newx, newy = step_toward(self.agent_pos, target_centroid, manhattan=True)
        # move the agent to new position only if inside bounds
        if self.is_in_grid(newx, newy):
            self.set_agent_position(newx, newy)

            discovered_cells = self.update_obs_grid()

            if discovered_cells == 0:
                # small penalty for no new discovery
                reward -= 1. / self.total_cells 
            else:
                frontiers = find_frontiers(self.obs_grid, self.free_color, self.unknown_color)
                centroids, self.clusters = cluster_frontiers(frontiers)
                self.centroids = centroids
                if self.sorting:
                    centroids = sort_array_by_distance(self, centroids, self.agent_pos)
                    if self.reverse:
                        centroids = centroids[::-1]
                if centroids.shape[0] == 0:
                    distances = np.array([])
                else:
                    distances = np.linalg.norm(centroids - self.agent_pos, axis=1)
                self.obs_centroids = pad_obs_array(centroids, target_shape=(self.centroids_obs_len, 2))
                self.obs_distances = pad_obs_array(distances, 
                                                   target_shape=(self.centroids_obs_len,), 
                                                   value=self.height+self.width)

                # reward proportional to new discovered cells
                reward += discovered_cells / self.total_cells  
        else:
            reward = -1  # penalty for invalid move
        
        if (self.discovered_cells / self.total_cells) > self.target_discovery_percent:
            terminated = True
            reward += 1  # big reward for completing exploration
        
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 1  # penalty for running out of time

        if self.render_mode == "human":
            print("Action: ", action, "Target centroid: ", target_centroid)
            self.render()
        
        obs = self._get_obs()

        return  obs, reward, bool(terminated) , bool(truncated), {}


    def _get_obs(self):  
        return self.obs_distances

    ## RENDERING FUNCTIONS
    
    def render(self):
        super().render()

        for c in self.clusters:
            self.ax_obs.scatter(c[:,0], c[:,1], s=10)
        self.ax_obs.scatter(self.centroids[:,0], self.centroids[:,1], c="green", s=80, marker="x")

        plt.pause(0.001)


if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    target_discovery_percent = 0.9
    perc_range = 3
    print("Creating environment...")
    env = ExplFrontStepDistancesEnv(width=width, 
                       height=height, 
                       obstacle_prob=obstacle_prob, 
                       target_discovery_percent=target_discovery_percent,
                       perc_range=perc_range, 
                       render_mode="human")

    # check_env(env, warn=True)
    # print("Env checked.")
    # exit()

    print("Resetting environment...")
    obs, _ = env.reset()
    print("Stepping through the environment...")

    term = False
    trunc = False

    while not term and not trunc:
        action = np.argmin(obs)  # greedy action: move toward closest centroid
        obs, reward, term,  trunc, _ = env.step(action)
