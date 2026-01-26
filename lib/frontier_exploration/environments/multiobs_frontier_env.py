#/usr/bin/python3

if __name__ == "__main__":
    import sys
    import numpy as np
    from stable_baselines3.common.env_checker import check_env 
    
    sys.path.append(".")
    
from lib.frontier_exploration.environments.step_mixin import StepStraightMixin
from lib.frontier_exploration.environments.multiobs_front_env_base import MultiObsFrontEnvBase


class MultiObsFrontierEnv(StepStraightMixin, MultiObsFrontEnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def reset(self, seed=None, *args, **kwargs):
        super().reset(seed=seed, *args, **kwargs)
        
        return self.mixin_reset()

if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.025
    target_discovery_percent = 0.9
    perc_range = 3
    
    truncations = 0
    terminations = 0
    for obs_type in ['distance']:#'relative', 'absolute', 'distance']:
        for info_gain in [True]:#, True]:
            for ag_pos in [False]:#, False]:
                #print(f"Obs type: {obs_type}, Info gain: {info_gain}, Agent pos: {ag_pos}")
                print("Creating environment...")
                env = MultiObsFrontierEnv(  width=width, 
                                            height=height, 
                                            obstacle_prob=obstacle_prob, 
                                            target_discovery_percent=target_discovery_percent,
                                            perc_range=perc_range, 
                                            render_mode= "human",#None,
                                            sorting=True,
                                            reverse=False,
                                            obs_spec={'type':obs_type,
                                                    'ag_pos':ag_pos,
                                                    'i_gain':info_gain})
                #for i in range(50):

                # check_env(env, warn=True)
                # print("Env checked.")
                # exit()

                print("Resetting environment...")
                obs, _ = env.reset()
                print("Stepping through the environment...")

                

                term = False
                trunc = False

                while not term and not trunc:
                    # centroids = obs['frontier_centroids'].reshape(-1,2)
                    # agent_pos = obs['agent_position']
                    # centroids = obs.reshape(-1,2)
                    # action = greedy_index(centroids, env.agent_pos)#np.array([0,0]))
                    c = np.stack((obs[::3],obs[1::3])).transpose()
                    # action = np.argmin( np.linalg.norm(c , axis=1) )
                    c = obs[::2]
                    action = np.argmin( c )
                    obs, reward, term,  trunc, _ = env.step(action)

                if trunc:
                    truncations += 1
                    print("Episode truncated.")
                elif term:
                    terminations += 1
                    print("Exploration completed.")
    print(f"Total terminations: {terminations}, Total truncations: {truncations}")