from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from JavaDungeon import LevelLoader
from DungeonTFEnvironment import DungeonTFEnvironment

import argparse

def train(config: dict):
    dungeon = DungeonTFEnvironment(
        dungeon=LevelLoader().loadLevel(config["dungeon"])
    )

    environment = Environment.create(
        environment=dungeon,
        max_episode_timesteps=config["timesteps"]
    )

    agent = Agent.create(
        agent=config["agent"],
        environment=environment
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        max_episode_timesteps=config["timesteps"]
    )

    runner.run(num_episodes=config["episodes"])

    agent.save(directory=config["out"],          format='saved-model')
    agent.save(directory=config["out"]+'/numpy', format='numpy')

    runner.close()
    agent.close()
    environment.close()

def checkPositive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def setupArgumentParser():

    parser = argparse.ArgumentParser()

    # TODO: Add checker for paths/files
    parser.add_argument("-d", "--dungeon", help="")
    parser.add_argument("-a", "--agent", help="")
    parser.add_argument("-o", "--out", default="saved-model", help="")
    parser.add_argument("-m", "--maxtimesteps", type=checkPositive, default=100, help="")
    parser.add_argument("-e", "--episodes", type=checkPositive, default=100, help="")

    return parser

if __name__ == '__main__':

    parser = setupArgumentParser()
    args = parser.parse_args()

    config = {
        "dungeon": args.dungeon,
        "agent": args.agent,
        "episodes": args.episodes,
        "timesteps": args.maxtimesteps,
        "out": args.out
    }

    train(config)
