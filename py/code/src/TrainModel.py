from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from JavaDungeon import LevelLoader
from DungeonTFEnvironment import DungeonTFEnvironment

import argparse
import json
from os.path import join, abspath

def train(config: dict):
    """Trains a RL model.

    Args:
        config (dict): The training configuration.
    """
    dungeon = DungeonTFEnvironment(
        dungeon=LevelLoader().loadLevel(config["environment"]["dungeon"])
    )

    environment = Environment.create(
        environment=dungeon,
        max_episode_timesteps=config["environment"]["max_timesteps"]
    )

    agent = Agent.create(
        agent=config["agent"],
        environment=environment
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        max_episode_timesteps=config["runner"]["max_timesteps"]
    )

    runner.run(num_episodes=config["runner"]["episodes"])

    agent.save(directory=config["output"]+'/saved-model', format='saved-model')
    agent.save(directory=config["output"]+'/numpy-model', format='numpy')

    runner.close()
    agent.close()
    environment.close()

def checkPositive(value):
    """Checks if input values are positive integers

    Args:
        value (Any): Input value

    Raises:
        argparse.ArgumentTypeError: Raised if input is not a positive integer

    Returns:
        int: A positive integer
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def setupArgumentParser():
    """Configure a command line argument parser.

    Returns:
        ArgumentParser: A command line argument parser.
    """
    parser = argparse.ArgumentParser()

    # TODO: Add checks for paths/files
    parser.add_argument("-d", "--dungeon", help="")
    parser.add_argument("-a", "--agent", help="")
    parser.add_argument("-o", "--out", default="out", help="")
    parser.add_argument("-m", "--max_timesteps", type=checkPositive, default=100, help="")
    parser.add_argument("-e", "--episodes", type=checkPositive, default=100, help="")
    parser.add_argument("-s", "--summarize", action='store_true', help="")

    return parser

def saveConfiguration(args):
    """Assembles a training configuration from command line arguments and saves it as a JSON file.

    Args:
        args (Namespace): Namespace with the parsed command line arguments

    Returns:
        dict: A dictionary representing the configuration
    """
    with open(args.agent) as agentFile, open(join(args.out,"config.json"), 'w') as configFile:
        agent = json.load(agentFile)
        if args.summarize:
            agent["summarizer"] = {
                "directory": join(args.out,"summary"),
                "summaries": ['entropy', 'loss', 'reward', 'update-norm']
            }

        config = {
            "environment": {
                "dungeon": abspath(args.dungeon),
                "max_timesteps": args.max_timesteps
            },
            "agent": agent,
            "runner": {
                "episodes": args.episodes,
                "max_timesteps": args.max_timesteps
            },
            "output": abspath(args.out)
        }

        json.dump(config, configFile, indent=2)
        return config

if __name__ == '__main__':
    parser = setupArgumentParser()
    config = saveConfiguration(parser.parse_args())
    train(config)
