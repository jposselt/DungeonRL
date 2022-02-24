import jpype
import jpype.imports
from jpype.types import *
 
# launch the JVM
lib_dir = '../lib/'
jpype.startJVM(
    classpath = [
        lib_dir + 'dungeon/core-1.0.9.jar',
        lib_dir + 'gdx/gdx-1.10.0.jar',
        lib_dir + 'gdx/gdx-ai-1.8.2.jar',
    ]
)

from level.elements import Level
import gym

class Dungeon(gym.Env):
    def __init__(self, dungeon: Level, n_actions: int):
        self.dungeon           = dungeon
        self.action_space      = n_actions
        self.observation_space = None # TODO
        self.state             = None # TODO
 
    def reset(self):
        pass
 
    def step(self, action):
        pass
