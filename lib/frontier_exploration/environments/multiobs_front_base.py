#/usr/bin/python3
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env 

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    
from lib.grid_env.obst_grid_agent_expl_env import ObstGridAgentExplEnv
from lib.frontier_exploration.frontiers import FrontierMixin
from lib.utils import greedy_index, step_toward


class MultiObsFrontBase(FrontierMixin, ObstGridAgentExplEnv):
    """Base Class implementing a Gym environment for frontier-based exploration.
    Multiple inheritance:
      - ObstGridAgentExplEnv: base environment with agent and obstacle grid
      - FrontierMixin: methods for frontier detection and processing
         - Returns centroids of frontier clusters as observation.(Can be sorted and reversed))
    
    obs_type: Dict with keys:
        - 'type': 'absolute', 'relative', 'distance'
        - 'agent_position': True or False
        - 'Information_gain': True or False

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
                 obs_spec={'type':'absolute',
                           'ag_pos':True,
                           'i_gain':False},
                 sorting=True,
                 reverse=False,
                 padding_value=0. ,
                 static_obstacles=False,
                 static_obstacles_seed=None
                 ):
        
        super().__init__(perception_range=perc_range, 
                         width=width, 
                         height=height, 
                         obstacle_prob=obstacle_prob,
                         render_mode=render_mode,
                         static_obstacles=static_obstacles,
                         static_obstacles_seed=static_obstacles_seed)
        
        
        self.target_discovery_percent = target_discovery_percent
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.centroids_obs_len = centroids_obs_len
        self.sorting = sorting
        self.reverse = reverse
        self.obs_spec = obs_spec
        self.padding_value = padding_value
        
        self.total_cells = width * height

        

        ## FRONTIER ENGINE
        max_front_cluster_size = perc_range * 5
        self.frontier_init(max_cluster_size=max_front_cluster_size)#sort_by='distance', 

        self.max_relative = max(height, width, self.padding_value)
        self.max_distance = max(int(np.sqrt(height**2 + width**2)), self.padding_value) 
        if self.obs_spec['i_gain']:
            self.max_relative = max(self.max_relative, max_front_cluster_size)
            self.max_distance = max(self.max_distance, max_front_cluster_size)

        ## ACTION AND OBSERVATION SPACES
        lowbound = min(0, self.padding_value)
        
        self.action_space = spaces.Discrete(centroids_obs_len)        
        if obs_spec['type'] in ['absolute', 'relative']:
            if obs_spec['type'] == 'relative':
                lowbound = min(-self.max_relative, self.padding_value)
            n = 3 if obs_spec['i_gain'] else 2
            self.observation_space = spaces.Box(low=lowbound, 
                                                high=self.max_relative, 
                                                shape=(centroids_obs_len * n,), 
                                                dtype=np.float64)

        elif obs_spec['type'] == 'distance':
            n = 2 if obs_spec['i_gain'] else 1
            self.observation_space = spaces.Box(low=lowbound, 
                                                high=self.max_distance,
                                                shape=(centroids_obs_len * n,),
                                                dtype=np.float64)
        
        if obs_spec['ag_pos']:
            self.observation_space = spaces.Dict({
                'agent_position': spaces.Box(low=0, 
                                            high=self.max_relative, 
                                            shape=(2,), 
                                            dtype=np.int64),

                'frontier_centroids': self.observation_space
            })

        # if self.render_mode == "human":
        #     self.init_simulation_render()

    ## ENVIRONMENT DYNAMICS AND INTERACTION METHODS

    def reset(self, seed=None, *args, **kwargs):
        super().reset(seed=seed, *args, **kwargs)


    def stack_obs(self):
        if self.obs_spec['type'] in ['absolute', 'relative']:
            if self.obs_spec['type'] == 'absolute':
                obs = self.obs_centroids
            elif self.obs_spec['type'] == 'relative':
                obs = self.relative_centroids

            igain = self.info_gain.reshape(-1,1)
            if self.obs_spec['i_gain']:
                obs = np.hstack( (obs, igain) )

        elif self.obs_spec['type'] == 'distance':
            obs = self.relative_distances
            
            igain = self.info_gain
            if self.obs_spec['i_gain']:
                obs = np.stack( (obs, igain) ).transpose()
        
        return obs.flatten()

            
    def _get_obs(self):
        obs = self.stack_obs()

        if self.obs_spec['ag_pos']:
            return {'agent_position': self.agent_pos,
                    'frontier_centroids':obs}
        else:
            return obs
    


