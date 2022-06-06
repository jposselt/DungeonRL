import random
import numpy as np

from tensorforce import Environment
from JavaDungeon import Point, Level
from collections import namedtuple

class MultiActorDungeon(Environment):
    """An RF learning environment with multiple actors.
    Based on https://github.com/tensorforce/tensorforce/blob/master/examples/multiactor_environment.py
    """

    def __init__(self, dungeon: Level):
        super().__init__()

        # Dungeon level (java class)
        self.dungeon = dungeon

        # State space
        state_indices = [0, 1, 2, 3] # 2D coordinates for two actors
        self._state_indices = np.array(state_indices, np.int32)
        self._state_bounds = self.getStateBounds(dungeon)

        # Coordinates of all accessible tiles
        self.accessible_coords = [
            tile.getGlobalPosition()
            for room in dungeon.getRooms()
            for sub_list in room.getLayout()
            for tile in sub_list
            if tile.isAccessible()
        ]

        # Step size
        self.step_size = 1

        # Named tuple for 2d positions
        self.Point2D = namedtuple('Point2D',['x','y'])

    def states(self):
        return dict(
            type=float,
            shape=tuple(self._state_indices.shape),
            min_value=self._state_bounds[0, self._state_indices],
            max_value=self._state_bounds[1, self._state_indices]
        )

    def actions(self):
        return dict(type=int, num_values=5)

    def num_actors(self):
        return 2  # Indicates that environment has multiple actors

    def reset(self):
        # Always for multi-actor environments: initialize parallel indices
        self._parallel_indices = np.arange(self.num_actors())

        # Single shared environment logic, plus per-actor perspective
        start_actor_1 = random.choice(self.start_coords)
        start_actor_2 = random.choice(self.start_coords)
        while (start_actor_1.equals(start_actor_2)):
            start_actor_2 = random.choice(self.start_coords)
        actor_1_perspective = np.array([start_actor_1.x, start_actor_1.y, start_actor_2.x, start_actor_2.y]).astype(np.float32)
        actor_2_perspective = np.array([start_actor_2.x, start_actor_2.y, start_actor_1.x, start_actor_1.y]).astype(np.float32)
        states = np.stack([actor_1_perspective, actor_2_perspective], axis=0)
        self._states = actor_1_perspective

        # Always for multi-actor environments: return per-actor values
        return self._parallel_indices.copy(), states

    def execute(self, actions):
        # Current and next postion for each actor
        actor_1_action = actions[0]
        actor_2_action = actions[1]
        actor_1_position = self.Point2D(self._states[0], self._states[1])
        actor_2_position = self.Point2D(self._states[2], self._states[3])
        next_actor_1_position = self.nextPosition(actor_1_position, actor_1_action)
        next_actor_2_position = self.nextPosition(actor_2_position, actor_2_action)

        # Single shared environment logic, plus per-actor perspective
        actor_1_perspective = np.array([next_actor_1_position.x, next_actor_1_position.y, next_actor_2_position.x, next_actor_2_position.y]).astype(np.float32)
        actor_2_perspective = np.array([next_actor_2_position.x, next_actor_2_position.y, next_actor_1_position.x, next_actor_1_position.y]).astype(np.float32)
        terminal = np.stack([False, False], axis=0)
        states = np.stack([actor_1_perspective, actor_2_perspective], axis=0)

        # TODO: Reward calculation
        reward = 0

        # Update internal state
        self._states = actor_1_perspective

        # Always for multi-actor environments: update parallel indices, and return per-actor values
        self._parallel_indices = self._parallel_indices[~terminal]
        return self._parallel_indices.copy(), states, terminal, reward

    def getStateBounds(self, dungeon: Level):
        # Get all accessible tiles
        accessible_tiles = [tile for room in dungeon.getRooms() for sub_list in room.getLayout() for tile in sub_list if tile.isAccessible()]

        # Get list of x/y coordinates
        x_coords = [coord.x for coord in [tile.getGlobalPosition() for tile in accessible_tiles]]
        y_coords = [coord.y for coord in [tile.getGlobalPosition() for tile in accessible_tiles]]

        # Get min/max values
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        return np.array([[x_min, y_min, x_min, y_min],[x_max, y_max, x_max, y_max]])

    def nextPosition(self, position, action: int):
        # Actions
        # 0: go north
        # 1: go south
        # 2: go west
        # 3: go east
        # 4: do nothing
        new_x = position.x, new_y = position.y
        if action == 0:
            new_y + self.step_size
        elif action == 1:
            new_y - self.step_size
        elif action == 2:
            new_x - self.step_size
        elif action == 3:
            new_x + self.step_size

        if (self.dungeon.getTileAt(Point(new_x, new_y).toCoordinate()).isAccessible()):
            return self.Point2D(new_x, new_y)
        return position