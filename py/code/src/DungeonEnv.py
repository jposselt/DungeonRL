import gym
import random
import numpy as np
from JavaDungeon import Point, Level

class DungeonEnv(gym.Env):
    """An OpenAI Gym compatible environment for navigating in a dungeon with RF learning
    """
    def __init__(self, dungeon: Level, n_actions: int):
        """Initialize the environment

        Args:
            dungeon (Level): the dungeon
            n_actions (int): number of possible actions

        Raises:
            ValueError: raised if given number of actions is invalid (<1)
        """
        if n_actions <= 0:
            raise ValueError('Number of action cannot be less than 1. Given value %d' % (n_actions))

        super(DungeonEnv, self).__init__()
        self.dungeon           = dungeon
        self.action_space      = gym.spaces.Discrete(n_actions)
        self.observation_space = self._getObservationSpace(dungeon)
        self.goal              = dungeon.getEndTile().getGlobalPosition().toPoint()
        self.start_tiles       = [
            tile
            for room in dungeon.getRooms()
            for sub_list in room.getLayout()
            for tile in sub_list
            if tile.isAccessible() and not tile.getGlobalPosition().equals(self.goal.toCoordinate())
        ]
        self.position          = random.choice(self.start_tiles).getGlobalPosition().toPoint()
        self.step_size         = 1
 
    def reset(self):
        """Reset the environment
        Important: the observation must be a numpy array
        
        Returns:
            np.array: initial state after reset
        """
        # Choose random tile (must be accessible and not be the goal tile)
        self.position = random.choice(self.start_tiles).getGlobalPosition().toPoint()

        return np.array([self.position.x, self.position.y]).astype(np.float32)
 
    def step(self, action):
        """Perform an action in the environment

        Args:
            action (int): code for the action to perform

        Raises:
            ValueError: raised if given action code is unknown

        Returns:
            np.array, int, bool, dict: Resulting state, reward, termination indicator and additional information
        """
        if not self.action_space.contains(action):
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Set movement direction
        dx, dy = 0, 0
        if action == 0:
            dy = self.step_size
        elif action == 1:
            dy = -self.step_size
        elif action == 2:
            dx = -self.step_size
        elif action == 3:
            dx = self.step_size

        # Calculate new position
        new_position = Point(self.position.x + dx, self.position.y + dy)

        # Perform movement if new position is accessible
        if (self.dungeon.getTileAt(new_position.toCoordinate()).isAccessible()):
            self.position = new_position

        # Check if goal position has been reached
        done = self.position.toCoordinate().equals(self.goal.toCoordinate())

        # Null reward everywhere except when reaching the goal
        reward = 1 if done else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        # returned observation must be a numpy array
        observation = np.array([self.position.x, self.position.y]).astype(np.float32)

        return observation, reward, done, info

    def _getObservationSpace(self, dungeon: Level):
        """Calculate the observation space for a given dungeon object

        Args:
            dungeon (Level): the dungeon

        Returns:
            gym.space: bounded state space
        """
        tiles = [tile for room in dungeon.getRooms() for sub_list in room.getLayout() for tile in sub_list]
        positions = [(p.x, p.y) for p in [tile.getGlobalPosition().toPoint() for tile in tiles]]
        limits =  np.array(list(map(min, zip(*positions)))), np.array(list(map(max, zip(*positions))))
        return gym.spaces.Box(limits[0], limits[1])
