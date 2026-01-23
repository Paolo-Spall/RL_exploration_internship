# Environments Package of RL-frontier based exploration

env_2D_grid_frontiers_gym:
    -> class: FrontierExplorationEnv(gym)
        gym env class for implement frontier based exploration on 2d grid
        with obstacles suitable for RL training.

env_planning_frontiers:
    -> class: FrontierExplPlannEnv
        environment python class (no gym inheritance) for implementing frontier-based
        exploration and path planning (rrt) a 2d grid with obstacles

env_2D_grid_frontiers_plan

environment naming convention
Plann: if any path planning is implemented
Front: frontier exploration is implemented
Gym: Gymnasium class inherited
Centr: Observation are centroids
Sort: Centroids are sorted

ExplFrontGymStepCentr
