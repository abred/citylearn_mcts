import time

import numpy as np


def evaluate(controller, config):
    print("Starting local evaluation")
    start_time = time.process_time()

    observation = controller.env.reset()

    # Init local variables
    episodes_completed = 0
    num_steps = 0

    # Define the agents' training time
    evaluation_steps = 24 * config.evaluation_days

    # Start the evaluation process
    while True:
        num_steps += 1

        observation, rewards, done, info = controller.step(observation)

        # # evaluate the last episode and reset the environment
        if done:
            episodes_completed += 1

            # Reset the environment
            done = False
            observation = controller.env.reset()

        # terminate evaluation
        if num_steps == evaluation_steps:
            print(f"Evaluation process is terminated after {num_steps} steps.")
            break

    print(f"Evaluation took {time.process_time() - start_time}s")
