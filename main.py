import logging
import types

from citylearn.citylearn import CityLearnEnv

from ai_dm.Search.mcts import mcts_factory
import ai_dm.Search.mcts.mcts_policies as mp

from controller import MCTSController
from evaluate import evaluate
from utils import reset_citylearn_env

logger = logging.getLogger(__name__)


def main():
    config = types.SimpleNamespace()
    config.schema_path = 'data/citylearn_challenge_2022_phase_1/schema.json'
    config.evaluation_days = 1
    config.verbose = True

    # Create a new environment
    env = CityLearnEnv(schema=config.schema_path)
    num_buildings = len(env.buildings)

    budget = 10
    exploration_constant = 0.1
    lower_bound_reward = -250
    upper_bound_reward = 20
    max_depth = 100
    agent_fact = mcts_factory(
        set_env_to_state=reset_citylearn_env,
        select_and_expand=mp.standard_tree_policy(
            mp.uct_action_selection(
                exploration_constant=exploration_constant,
                lower_bound_reward=lower_bound_reward,
                upper_bound_reward=upper_bound_reward)
        ),
        rollout=mp.standard_default_policy(max_depth=max_depth),
        backprop=mp.standard_backprop,
        best_action=mp.standard_best_action_selection,
        prune_tree=mp.standard_prune_tree_function
        )
    agents = []
    for _ in range(num_buildings):
        agents.append(agent_fact())

    # Add coordinator
    controller = MCTSController(env, agents, budget)

    evaluate(controller, config)


if __name__ == "__main__":
    logging.basicConfig(level=20)

    main()
