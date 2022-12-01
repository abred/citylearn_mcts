from copy import deepcopy
import logging

from utils import (
    CityLearnLocalDiscreteActions,
    equi_dist_discrete_action_space
    )

logger = logging.getLogger(__name__)

def make_iter(val):
    try:
        return iter(val)
    except TypeError:
        return iter([val])


class MCTSController():
    def __init__(self, env, agents, budget):
        self.env = env
        self.agents = agents
        self.budget = budget

    def step(self, observations):
        action = self.get_joint_action(observations)
        return self.perform_joint_action(action)

    def get_joint_action(self, observations):
        """Returns the joint action

        Args:
            observations: the current observations

        Returns:
            dict: the actions for the agents
        """
        joint_action = []
        for agent_id in range(len(self.agents)):
            logger.info("computing action for agent %s", agent_id)
            agent = self.agents[agent_id]
            obs = observations[agent_id]
            env = self.make_local_temp_env(agent_id)
            obs = {
                "obs": obs,
                "time_step": env.time_step
            }
            joint_action.append(make_iter(agent.get_action_time(obs, env, self.budget)))

        return joint_action

    def perform_joint_action(self, joint_action):
        logger.info("applying action %s", [list(a) for a in joint_action])
        return self.env.step(joint_action)

    def make_local_temp_env(self, agent_id):
        return CityLearnLocalDiscreteActions(
            deepcopy(self.env),
            agent_id,
            equi_dist_discrete_action_space(
                self.env.action_space[agent_id], 3)
        )
