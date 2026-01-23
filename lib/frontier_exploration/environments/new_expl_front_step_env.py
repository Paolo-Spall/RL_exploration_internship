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
from lib.frontier_exploration.frontiers import FrontierDetector


class NewExplFrontStepEnv(ObstGridAgentExplEnv):
    """Gym environment for frontier-based exploration.
       - Returns centroids of frontier clusters as observation.(Can be sorted and reversed))
       - Action is the index of the target centroid to move toward.
       - Step update every sigle-cell move toward the target centroid.
        obstacle_color = 170  # Dark gray for obstacles
        unknown_color = 85  # Light gray for unknown cells
        agent_color = 255  # Black for agent position
        free_color = 0  # White for free cells
        min_color = 0
        max_color = 255
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
                 distance=False,
                 sorting=True,
                 reverse=False):
        
        super().__init__(perception_range=perc_range, 
                         width=width, 
                         height=height, 
                         obstacle_prob=obstacle_prob)
        
        self.front_detector = FrontierDetector( height=height,
                                                width=width,
                                                free_color=self.free_color, 
                                                unknown_color=self.unknown_color,
                                                centroids_obs_len=centroids_obs_len,
                                                max_cluster_size=perc_range * 5,
                                                sorting=sorting,
                                                reverse=reverse)
        
        self.target_discovery_percent = target_discovery_percent
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.centroids_obs_len = centroids_obs_len
        self.relative = relative
        self.distance = distance
        self.sorting = sorting
        self.reverse = reverse

        self.total_cells = width * height
        self.discovered_cells = 0

        self.action_space = spaces.Discrete(centroids_obs_len)
        
        highbound = max(height, width)

        if relative:
            lowbound = -highbound
        elif distance:
            highbound = int(np.sqrt(width**2 + height**2))
            lowbound = -highbound
        else:
            lowbound = 0

        if distance:
            self.observation_space = spaces.Box(low=lowbound, 
                                                high=highbound,
                                                shape=(centroids_obs_len,),
                                                dtype=np.int64)
        else:
            self.observation_space = spaces.Dict({
                'agent_position': spaces.Box(low=0, 
                                            high=highbound, 
                                            shape=(2,), 
                                            dtype=np.int64),

                'frontier_centroids': spaces.Box(low=lowbound, 
                                                high=highbound, 
                                                shape=(centroids_obs_len * 2,), 
                                                dtype=np.int64)
            })

        if self.render_mode == "human":
            self.init_simulation_render()

    ## ENVIRONMENT DYNAMICS AND INTERACTION METHODS

    def reset(self, seed=None, *args, **kwargs):
        super().reset(seed=seed, *args, **kwargs)
        
        self.update_obs_grid()
        self.update_frontier_obs()
        
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
                self.update_frontier_obs()

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

    def update_frontier_obs(self):
        self.front_detector.detect(self.obs_grid, 
                                    agent_pos=self.agent_pos)
        
        self.centroids = self.front_detector.centroids
        self.obs_centroids = self.front_detector.pad_centroids
        self.clusters = self.front_detector.clusters
        self.relative_centroids = self.front_detector.relative_centroids
        self.relative_distances = self.front_detector.relative_distances


    def _get_obs(self): 
        if self.relative:
            return {'agent_position': self.agent_pos,
                    'frontier_centroids':self.relative_centroids.flatten()}
        elif self.distance:
            return self.relative_distances
        else:
            return {'agent_position': self.agent_pos,
                    'frontier_centroids':self.obs_centroids.flatten()}

    ## RENDERING FUNCTIONS
    
    def render(self):
        super().render()

        for c in self.clusters:
            self.ax_obs.scatter(c[:,0], c[:,1], s=10)
        
        mask = np.any(self.obs_centroids != 0, axis=1)
        filtered_centroids = self.obs_centroids[mask]
        self.ax_obs.scatter(filtered_centroids[:,0], filtered_centroids[:,1], c="green", s=80, marker="x")

        plt.pause(0.01)


if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.05
    target_discovery_percent = 0.9
    perc_range = 3
    print("Creating environment...")
    env = NewExplFrontStepEnv(  width=width, 
                                height=height, 
                                obstacle_prob=obstacle_prob, 
                                target_discovery_percent=target_discovery_percent,
                                perc_range=perc_range, 
                                render_mode="human",
                                sorting=True,
                                reverse=False,
                                relative=False,
                                distance=False)

    check_env(env, warn=True)
    print("Env checked.")
    exit()

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
        
        # import joblib 
        # env.init_simulation_render()
        # save = input('Save?')
        # if save == 'y':
        #     joblib.dump([env.obs_grid,
        #                 env.clusters,
        #                 env.centroids],
        #              'frontiers_grid.joblib')

    if trunc:
        print("Episode truncated.")
    else:
        print("Episode terminated.")