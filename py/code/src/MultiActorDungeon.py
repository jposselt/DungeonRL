import numpy as np

from tensorforce import Environment
from JavaDungeon import Point, Level

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

        # Step size
        self.step_size = 1

    def states(self):
        return dict(
            type=float,
            shape=tuple(self._state_indices.shape),
            min_value=self._state_bounds[0, self._state_indices],
            max_value=self._state_bounds[1, self._state_indices]
        )

    def actions(self):
        # Actions
        # 0: go north
        # 1: go south
        # 2: go west
        # 3: go east
        # 4: do nothing
        return dict(type=int, num_values=5)

    def num_actors(self):
        return 2  # Indicates that environment has multiple actors

    def reset(self):
        # Always for multi-actor environments: initialize parallel indices
        self._parallel_indices = np.arange(self.num_actors())

        # Single shared environment logic, plus per-actor perspective
        states = np.stack([, ], axis=0)

        # Always for multi-actor environments: return per-actor values
        return self._parallel_indices.copy(), states

    def execute(self, actions):
        # Single shared environment logic, plus per-actor perspective
        terminal = np.stack([False, False], axis=0)
        states = np.stack([, ], axis=0)
        reward = 0

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