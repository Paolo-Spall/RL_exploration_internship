#/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    import sys
    sys.path.append(".")
from lib.utils import  move_toward, step_toward
from lib.rendering_utils import fig_to_rgb
from lib.frontier_exploration.planning.obst_avoidance import compute_path


class StepMixin:

    ## ENVIRONMENT DYNAMICS AND INTERACTION METHODS

    def mixin_reset(self, seed=None, *args, **kwargs):
        #super().reset(seed=seed, *args, **kwargs)
        self.discovered_cells = 0
        self.update_obs_grid()
        self.detect_frontiers()
        
        self.current_step = 0

        #self.stuck_steps = 0
        self.prev_pos = None
        self.prev_prev_pos = None
        self.no_progress_steps = 0

        if self.render_mode is not None:#== "human":
            self.render()

        obs = self._get_obs()
        return obs,  {}


    def step(self, next_pos):
        """move: np.array([dx, dy])
        """

        reward = 0
        terminated = False
        truncated = False
        self.current_step += 1

        old_pos = tuple(self.agent_pos.copy())
        
        newx, newy = next_pos

        # move the agent to new position only if inside bounds
        if self.is_in_grid(newx, newy):
            self.set_agent_position(newx, newy)
            discovered_cells = self.update_obs_grid()

            if discovered_cells == 0:
                # small penalty for no new discovery
                #reward -= 2.* self.perception_range / self.total_cells
                #reward -= 1. / self.total_cells
                self.detect_frontiers(changed = False)
                self.no_progress_steps += 1

                reward -= 0.05 
            else:
                self.detect_frontiers(changed = True)
                self.no_progress_steps = 0

                # reward proportional to new discovered cells
                # reward += discovered_cells / self.total_cells  
                reward += discovered_cells *0.05  
                
        else:
            reward = -1  # penalty for invalid move
        
        new_pos = tuple(self.agent_pos.copy())

        # A<->B oscillation penalty
        if (
            self.prev_prev_pos is not None
            and new_pos == self.prev_prev_pos
            and old_pos == self.prev_pos
        ):
            reward -= 0.05  # penalty for oscillation

        self.prev_prev_pos = self.prev_pos
        self.prev_pos = new_pos
        
        # no progress for too many steps
        if self.no_progress_steps >= 20:
            truncated = True
            reward -= 0.5
            if self.render_mode == "human":
                print("No progress for 20 steps. Exploration truncated.")


        if (self.discovered_cells / self.total_cells) > self.target_discovery_percent:
            terminated = True
            reward += 1  # big reward for completing exploration
            if self.render_mode == "human":
                print("Target discovery percent reached. Exploration complete!")
        
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 1  # penalty for running out of time
            if self.render_mode == "human":
                print("Max steps reached. Exploration truncated.")

        if self.render_mode is not None: #== "human":
            self.render()        
        
        obs = self._get_obs()

        return  obs, reward, bool(terminated) , bool(truncated), {}


    ## RENDERING FUNCTIONS
    
    def render(self):
        super().render()

        for c in self.clusters:
            self.ax_obs.scatter(c[:,0], c[:,1], s=10)
        
        mask = np.any(self.obs_centroids != self.padding_value, axis=1)
        filtered_centroids = self.obs_centroids[mask]
        self.ax_obs.scatter(filtered_centroids[:,0], filtered_centroids[:,1], c="green", s=80, marker="x")

        if self.render_mode == "human":
            plt.pause(0.001)
        elif self.render_mode == "rgb_array":
            return fig_to_rgb(self.fig)
            


class StepStraightMixin(StepMixin):
    def step(self, action):
        """action: index of the target centroid
        """

        # extracting the target centroid coordinates
        target_centroid = np.array(self.obs_centroids[action].copy(), dtype=np.int64)
        # computing the next position toward the target centroid
        next_pos = step_toward(self.agent_pos, target_centroid, manhattan=True)

        if self.render_mode == "human":
            print("Action: ", action, "Target centroid: ", target_centroid)

        return super().step(next_pos)

