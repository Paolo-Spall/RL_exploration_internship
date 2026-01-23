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


class ExplFrontStepEnv(ObstGridAgentExplEnv):
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
                 relative=False,
                 sorting=True,
                 reverse=False):
        
        super().__init__(perception_range=perc_range, 
                         width=width, 
                         height=height, 
                         obstacle_prob=obstacle_prob)
        
        self.target_discovery_percent = target_discovery_percent
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.centroids_obs_len = centroids_obs_len
        self.sorting = sorting
        self.reverse = reverse
        self.relative = relative

        self.total_cells = width * height
        self.discovered_cells = 0

        self.action_space = spaces.Discrete(centroids_obs_len)
        
        self.observation_space = spaces.Dict({
            'agent_position': spaces.Box(low=0, high=max(height, width), shape=(2,), dtype=np.int64),
            'frontier_centroids': spaces.Box(low=0, high=max(height, width), shape=(centroids_obs_len * 2,), dtype=np.uint8)
        })

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
        if self.relative:
            if centroids.shape[0] > 0:
                centroids -= self.agent_pos  # make centroids relative to agent position
            # self.rel_centroids = pad_obs_array(rel_centroids, self.centroids_obs_len)
        
        self.obs_centroids = pad_obs_array(centroids, target_shape=(self.centroids_obs_len, 2))

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
        target_centroid = np.array(self.obs_centroids[action].copy(), dtype=np.int64)
        if self.relative:
            target_centroid += self.agent_pos  # convert back to absolute coordinates

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
                if self.relative:
                    if centroids.shape[0] > 0:
                        centroids -= self.agent_pos  # make centroids relative to agent position
                    # self.rel_centroids = pad_obs_array(rel_centroids, self.centroids_obs_len)
                
                self.obs_centroids = pad_obs_array(centroids, target_shape=(self.centroids_obs_len, 2))

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
        return {'agent_position': self.agent_pos,
                'frontier_centroids':self.obs_centroids.flatten()}

    ## RENDERING FUNCTIONS
    
    def render(self):
        super().render()

        for c in self.clusters:
            self.ax_obs.scatter(c[:,0], c[:,1], s=10)
        
        mask = np.any(self.centroids != 0, axis=1)
        filtered_centroids = self.centroids[mask]
        self.ax_obs.scatter(filtered_centroids[:,0], filtered_centroids[:,1], c="green", s=80, marker="x")

        plt.pause(0.01)


if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    target_discovery_percent = 0.9
    perc_range = 3
    print("Creating environment...")
    env = ExplFrontStepEnv(width=width, 
                       height=height, 
                       obstacle_prob=obstacle_prob, 
                       target_discovery_percent=target_discovery_percent,
                       perc_range=perc_range, 
                       render_mode="human",
                       sorting=True,
                       reverse=True,
                       relative=False,)

    # check_env(env, warn=True)
    # print("Env checked.")
    # exit()

    print("Resetting environment...")
    obs, _ = env.reset()
    print("Stepping through the environment...")

    term = False
    trunc = False

    while not term and not trunc:
        centroids = obs['frontier_centroids'].reshape(-1,2)
        agent_pos = obs['agent_position']
        action = greedy_index(centroids, agent_pos)
        obs, reward, term,  trunc, _ = env.step(action)

    if trunc:
        print("Episode truncated.")
    else:
        print("Episode terminated.")