########################################################################
##                                                                    ##
##                         Utility Functions                          ##
##                                                                    ##
########################################################################
import gym
from gym.spaces import Box, Discrete
import numpy as np

class CityLearnLocalDiscreteActions(gym.ActionWrapper):
    def __init__(self, env, agent_id, disc_to_cont, action_space=None):
        """
        Args:
            env: gym.Env
                original env
            agent_id: int
                restrict env to this agent
            disc_to_cont: Callable
                callable that can be used to convert discrete action id
                to continouus action.
            action_space: optional, gym.Space
                 action space to be used by agent, if not supplied,
                 disc_to_cont has to have an action_space attribute.
        """
        super().__init__(env)
        self.agent_id = agent_id
        self.disc_to_cont = disc_to_cont
        self.action_space = (action_space
                             if action_space is not None
                             else disc_to_cont.action_space)

    def step(self, action):
        """Runs the environment :meth:`env.step` using the modified ``action``
        from :meth:`self.action`."""
        observations, rewards, done, info = self.env.step(self.action(action))
        return (
            observations[self.agent_id],
            rewards[self.agent_id],
            done,
            info
        )

    def action(self, local_action):
        """Returns a modified action before :meth:`env.step` is called."""
        num_agents = len(self.env.action_space)
        # assumes 0 as "neutral" action
        actions = [np.reshape(np.array(0), (1,))] * num_agents
        actions[self.agent_id] = self.disc_to_cont(local_action)

        return actions


class equi_dist_discrete_action_space():
    def __init__(self, env_action_space, num):
        assert isinstance(env_action_space, gym.spaces.Box), \
            "Should only be used for Box action spaces."
        self.env_action_space = env_action_space
        self.action_space = Discrete(num)
        self.discr_actions = np.linspace(
            self.env_action_space.low, self.env_action_space.high, num)

    def __call__(self, action_id):
        return self.discr_actions[action_id]

def reset_citylearn_env(env, state):
    ts = state['time_step']
    env.time_step = ts
    env.rewards = env.rewards[:ts]
    env.net_electricity_consumption = env.net_electricity_consumption[:ts]
    env.net_electricity_consumption_price = \
        env.net_electricity_consumption_price[:ts]
    env.net_electricity_consumption_emission = \
        env.net_electricity_consumption_emission[:ts]
    for b in env.buildings:
        setattr(b, "__cooling_electricity_consumption",
                b.cooling_electricity_consumption[:ts])
        setattr(b, "__heating_electricity_consumption",
                b.heating_electricity_consumption[:ts])
        setattr(b, "__dhw_electricity_consumption",
                b.dhw_electricity_consumption[:ts])
        setattr(b, "__solar_generation",
                b.pv.get_generation(b.energy_simulation.solar_generation)*-1)
        setattr(b, "__net_electricity_consumption",
                b.net_electricity_consumption[:ts])
        setattr(b, "__net_electricity_consumption_emission",
                b.net_electricity_consumption_emission[:ts])
        setattr(b, "__net_electricity_consumption_price",
                b.net_electricity_consumption_price[:ts])

        b.cooling_storage.__soc = b.cooling_storage.soc[:ts]
        b.cooling_storage.__energy_balance = \
            b.cooling_storage.energy_balance[:ts]
        b.heating_storage.__soc = b.heating_storage.soc[:ts]
        b.heating_storage.__energy_balance = \
            b.heating_storage.energy_balance[:ts]
        b.dhw_storage.__soc = b.dhw_storage.soc[:ts]
        b.dhw_storage.__energy_balance = b.dhw_storage.energy_balance[:ts]
        b.electrical_storage.__efficiency_history = \
            b.electrical_storage.efficiency_history[:ts]
        b.electrical_storage.__capacity_history = \
            b.electrical_storage.capacity_history[:ts]
        b.electrical_storage.__soc = b.electrical_storage.soc[:ts]
        b.electrical_storage.__energy_balance = \
            b.electrical_storage.energy_balance[:ts]
        b.electrical_storage.__electricity_consumption = \
            b.electrical_storage.electricity_consumption[:ts]
        b.cooling_device.__electricity_consumption = \
            b.cooling_device.electricity_consumption[:ts]
        b.heating_device.__electricity_consumption = \
            b.heating_device.electricity_consumption[:ts]
        b.dhw_device.__electricity_consumption = \
            b.dhw_device.electricity_consumption[:ts]
        b.pv.__electricity_consumption = \
            b.pv.electricity_consumption[:ts]
