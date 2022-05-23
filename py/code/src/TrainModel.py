from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from JavaDungeon import LevelLoader
from DungeonTFEnvironment import DungeonTFEnvironment

import argparse
import json
from os.path import join, abspath

def train(config: dict):
    dungeon = DungeonTFEnvironment(
        dungeon=LevelLoader().loadLevel(config["environment"]["dungeon"])
    )

    environment = Environment.create(
        environment=dungeon,
        max_episode_timesteps=config["environment"]["maxTimesteps"]
    )

    agent = Agent.create(
        agent=config["agent"],
        environment=environment
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        max_episode_timesteps=config["runner"]["maxTimesteps"]
    )

    runner.run(num_episodes=config["runner"]["episodes"])

    agent.save(directory=config["output"]+'/saved-model', format='saved-model')
    agent.save(directory=config["output"]+'/numpy-model', format='numpy')

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

def saveConfiguration(dungeon, agent, epsiodes, maxTimesteps, out):
    with open(agent) as agentFile, open(join(out,"config.json"), 'w') as configFile:
        agent = json.load(agentFile)

        config = {
            "environment": {
                "dungeon": abspath(dungeon),
                "maxTimesteps": maxTimesteps
            },
            "agent": agent,
            "runner": {
                "episodes": epsiodes,
                "maxTimesteps": maxTimesteps
            },
            "output": abspath(out)
        }

        json.dump(config, configFile, indent=2)
        return config

if __name__ == '__main__':

    parser = setupArgumentParser()
    args = parser.parse_args()

    config = saveConfiguration(args.dungeon, args.agent, args.episodes, args.maxtimesteps, args.out)
    train(config)
