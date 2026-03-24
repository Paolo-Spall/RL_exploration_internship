#/usr/bin/python3
import numpy as np

if __name__ == "__main__":
    import sys
    from stable_baselines3.common.env_checker import check_env
    
    sys.path.append(".")

from lib.frontier_exploration.environments.dynamics import StepMixin
from lib.frontier_exploration.environments.multiobs_front_base import MultiObsFrontBase
from lib.utils import step_toward

class MultiObsFrontierEnv(StepMixin, MultiObsFrontBase):
    """MultiObsFrontBase ( FrontierMixin + ObstGridAgentExplEnv ( 
                                                ObstGridAgentEnv (
                                                    ObstGridEnv ( gym.Env) ) ) )
       + StepMixin (step and reset methods for frontier-based exploration)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def reset(self, seed=None, *args, **kwargs):
        super().reset(seed=seed, *args, **kwargs)
        
        return self.mixin_reset()
    
    def step(self, action):
        """action: index of the target centroid
        """

        # extracting the target centroid coordinates
        target_centroid = np.array(self.obs_centroids[action].copy(), dtype=np.int64)

        # case of SELECTED PADDING ELEMENT 
        if np.all(target_centroid == self.padding_value):
            # running the StepMixin step(<current position>) anyway ??
            obs, reward, terminated, truncated, info = super().step(self.agent_pos)
            if self.padding_penalty:
                reward -= 1
            else:
                # if not flagged only truncation, no penalty
                pass
            truncated = True

            # render
            if self.render_mode == "human":
                print(f"Invalid action = {action}: Selected padding element. Episode truncated.")
            
            return obs, reward, terminated, truncated, info
        
        # computing the next position toward the target centroid
        next_pos = step_toward(self.agent_pos, target_centroid, manhattan=True)

        if self.render_mode == "human":
            print("Action: ", action, "Target centroid: ", target_centroid)

        return super().step(next_pos)

if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.025
    target_discovery_percent = 0.9
    perc_range = 3
    
    truncations = 0
    terminations = 0
    for obs_type in ['relative', 'absolute', 'distance']:#['relative', 'absolute', 'distance']:
        for info_gain in [True, False]:# [True, False]
            for ag_pos in [True, False ]:# [True, False]
                # input("Press Enter to create the new environment...")
                #print(f"Obs type: {obs_type}, Info gain: {info_gain}, Agent pos: {ag_pos}")
                print("Creating environment...")
                env = MultiObsFrontierEnv(  width=width, 
                                            height=height, 
                                            obstacle_prob=obstacle_prob, 
                                            target_discovery_percent=target_discovery_percent,
                                            perc_range=perc_range, 
                                            render_mode=  "human", # "human", None,
                                            sorting=True,
                                            reverse=True,
                                            obs_spec={'type':obs_type,
                                                    'ag_pos':ag_pos,
                                                    'i_gain':info_gain},
                                            static_obstacles=True,
                                            static_obstacles_seed=42
                                            )
                #for i in range(50):

                # check_env(env, warn=True)
                # print("Env checked.")
                # exit()

                print("obs_spec: ")
                print(env.obs_spec)

                print("Resetting environment...")
                obs, _ = env.reset()
                print("Stepping through the environment...")

                

                term = False
                trunc = False

                while not term and not trunc:
                    if ag_pos:
                        centroids = obs['frontier_centroids']
                    else:
                        centroids = obs

                    if info_gain:
                        if obs_type in ['absolute', 'relative']:
                            centroids = np.stack((centroids[::3],centroids[1::3])).transpose()
                        else:
                            centroids = centroids[::2]
                    else:
                        if obs_type in ['absolute', 'relative']:
                            centroids = centroids.reshape((-1,2))


                    
                    if obs_type == 'distance':
                        mask = centroids == env.padding_value
                        centroids[mask] = np.inf
                        action = np.argmin( centroids )
                    else:
                        padding_element = np.array([env.padding_value, env.padding_value])
                        mask = np.all(centroids == padding_element, axis=1)

                        if obs_type == 'absolute':
                            centroids = centroids - env.agent_pos

                        centroids_dist = np.linalg.norm(centroids, axis=1)

                            #action = np.argmin( np.linalg.norm(  , axis=1) )
                        # elif obs_type == 'relative':
                        centroids_dist[mask] = np.inf
                        
                        action = np.argmin(  centroids_dist  )
                    

                    #action = np.random.randint(0, len(centroids))
                    obs, reward, term,  trunc, _ = env.step(action)

                if trunc:
                    truncations += 1
                    print("Episode truncated.")
                elif term:
                    terminations += 1
                    print("Exploration completed.")
    print(f"Total terminations: {terminations}, Total truncations: {truncations}")