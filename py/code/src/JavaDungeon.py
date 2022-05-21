import jpype
import jpype.imports
from jpype.types import *
 
# launch the JVM
lib_dir = '../lib/'
jpype.startJVM(
    classpath = [
        lib_dir + 'dungeon/code-2.0.0.jar',
        lib_dir + 'gdx/gdx-1.10.0.jar',
        lib_dir + 'gdx/gdx-ai-1.8.2.jar',
        lib_dir + 'google/gson-2.9.0.jar',
    ]
)

from tools import Point
from level.elements import Level
from level.elements.room import Room
from level.elements.room import Tile
from level.generator.dummy import DummyGenerator
from level.generator.LevelLoader import LevelLoader