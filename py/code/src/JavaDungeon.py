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

def is_position_accessible(dungeon: Level, point: Point):
    """Checks if a position is accessible in a dungeon

    Args:
        dungeon (Level): a dungeon
        point (Point): a position

    Returns:
        bool: True if point is accessible in the dungeon else False
    """
    return dungeon.getTileAt(point.toCoordinate()).isAccessible()

def get_accessible_coordinates(dungeon: Level):
    """Lists coordinates of all accessible tiles in a dungeon

    Args:
        dungeon (Level): a dungeon

    Returns:
        [Coordinate]: Array of accessible coordinates
    """
    return [
        tile.getGlobalPosition()
        for room in dungeon.getRooms()
        for sub_list in room.getLayout()
        for tile in sub_list
        if tile.isAccessible()
    ]

def get_dungeon_bounds(dungeon: Level):
    """Returns the minimum and maximum coordinate values of accessible tiles in a dungeon.

    Args:
        dungeon (Level): a dungeon

    Returns:
        int, int, int, int: Minimum x, minimum y, maximum x, maximum y
    """
    accessible_coordinates = get_accessible_coordinates(dungeon)
    x = [coordinate.x for coordinate in accessible_coordinates]
    y = [coordinate.y for coordinate in accessible_coordinates]
    return min(x), min(y), max(x), max(y)

def point_as_array(point: Point):
    """Convert a point into an array

    Args:
        point (Point): a point

    Returns:
        array: x,y coordinates as array
    """
    return [point.x, point.y]