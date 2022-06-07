import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple
from os.path import join

from tensorforce.environments import Environment
from tensorforce.agents import Agent
from JavaDungeon import LevelLoader, Point
from DungeonTFEnvironment import DungeonTFEnvironment

def setupArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration", help="")
    parser.add_argument("-o", "--out", default="out", help="")
    return parser

def loadConfigurationFromFile(path):
    with open(path) as configFile:
        return json.load(configFile)

def CreateEnvironment(config):
    dungeon = DungeonTFEnvironment(
        dungeon=LevelLoader().loadLevel(config["environment"]["dungeon"])
    )

    environment = Environment.create(
        environment=dungeon,
        max_episode_timesteps=config["environment"]["max_timesteps"]
    )
    
    return dungeon, environment

def CreateAgent(config, environment):
    return Agent.load(
        directory=join(config["output"], "numpy-model"),
        format='numpy',
        environment=environment,
        agent=config["agent"]
    )

def EvaluateModel(dungeon, environment, agent):
    StateAction = namedtuple("StateAction", ["xPos", "yPos", "action"])
    state_action_list = []

    for coordinate in dungeon.start_coords:
        states = environment.setState(Point(coordinate.x, coordinate.y))
        internals = agent.initial_internals()

        actions, internals = agent.act(
            states=states, internals=internals,
            independent=True, deterministic=True
        )

        s = StateAction(xPos=coordinate.x, yPos=coordinate.y, action=actions)
        state_action_list.append(s)
    
    agent.close()
    environment.close()
    
    return state_action_list

def PlotPolicy(state_action_list, out):
    # initialize empty lists
    x, y, u, v = ([] for i in range(4))
    
    for item in state_action_list:
        x.append(item.xPos)
        y.append(item.yPos)
        if item.action == 0:
            u.append(0)
            v.append(1)
        elif item.action == 1:
            u.append(0)
            v.append(-1)
        elif item.action == 2:
            u.append(-1)
            v.append(0)
        elif item.action == 3:
            u.append(1)
            v.append(0)
        else:
            u.append(0)
            v.append(0)
    
    fig, ax = plt.subplots(figsize=(7,7))
    ax.quiver(x,y,u,v)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')
    plt.savefig(out, dpi=150)

if __name__ == '__main__':
    parser = setupArgumentParser()
    args = parser.parse_args()
    config = loadConfigurationFromFile(args.configuration)
    dungeon, environment = CreateEnvironment(config)
    agent = CreateAgent(config, environment)
    evaluation = EvaluateModel(dungeon, environment, agent)
    PlotPolicy(evaluation, args.out)
