# Environments Package of RL-frontier based exploration

environments/

_init__.py
README.md
dynamics.py                 # Environment dynamics
 	class StepMixin
 	class StepStraightMixin(StepMixin)

multiobs_front_base.py      # Base multi-observation frontier env
	class MultiObsFrontBase(FrontierMixin, ObstGridAgentExplEnv)

multiobs_frontier_env.py    # Main frontier exploration env
	class MultiObsFrontierEnv(StepMixin, MultiObsFrontBase):
	
multiobs_frontier_avoidance_env.py  # Frontier env with obstacle avoidance
	class MultiObsFrontAvoidanceEnv(StepMixin, MultiObsFrontBase):
	
environment classes naming convention:
