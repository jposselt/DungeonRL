import random
import numpy as np
from tensorforce import Environment
from JavaDungeon import Point, Level

class DungeonTFEnvironment(Environment):
    """An RF learning environment based on the Tensorforce environment interface. Similar to DungeonGymEnvironment
    but extended with action masking.
    """
    def __init__(self, dungeon: Level):
        """Initialize the environment

        Args:
            dungeon (Level): the dungeon
        """
        super().__init__()
        
        # Dungeon level (java class)
        self.dungeon = dungeon

        # State space
        state_indices = [0, 1]
        self._state_indices = np.array(state_indices, np.int32)
        self._state_bounds = self.getStateBounds(dungeon)
        
        # Start/Goal
        self.goal_coord = dungeon.getEndTile().getGlobalPosition()
        self.start_coords = [
            tile.getGlobalPosition()
            for room in dungeon.getRooms()
            for sub_list in room.getLayout()
            for tile in sub_list
            if tile.isAccessible() and not tile.getGlobalPosition().equals(self.goal_coord)
        ]
        
        # Step size
        self.step_size = 1


    def states(self):
        """Returns the state space specification.

        Returns:
            specification: Arbitrarily nested dictionary of state descriptions. See base class for more information.
        """
        return dict(
            type=float,
            shape=tuple(self._state_indices.shape),
            min_value=self._state_bounds[0, self._state_indices],
            max_value=self._state_bounds[1, self._state_indices]
        )


    def actions(self):
        """Returns the action space specification.
        
        Returns:
            specification: Arbitrarily nested dictionary of action descriptions. See base class for more information.
        """
        return dict(type=int, num_values=4)


    def reset(self):
        """Resets the environment to start a new episode.

        Returns:
            dict[state, action_mask]: Dictionary containing initial state(s) and action mask. See "getActionMask" method
            for meaning of the mask.
        """
        # Initial state and associated action mask
        start = random.choice(self.start_coords)
        self.state = np.array([start.x, start.y]).astype(np.float32)
        action_mask = self.getActionMask(Point(start.x, start.y))
        
        # Add action mask to states dictionary (mask item is "[NAME]_mask", here "action_mask")
        states = dict(state=self.state, action_mask=action_mask)

        return states


    def execute(self, actions):
        """Executes the given action(s) and advances the environment by one step.

        Args:
            actions (dict[action]): Dictionary containing action(s) to be executed

        Returns:
            dict[state, action_mask], bool | 0 | 1 | 2, float: Dictionary containing next state(s)
            and action mask, whether a terminal state is reached or 2 if the episode was
            aborted and observed reward.
        """
        # Compute next state and associated action mask
        if actions == 0:
            self.state[1] += self.step_size
        elif actions == 1:
            self.state[1] -= self.step_size
        elif actions == 2:
            self.state[0] -= self.step_size
        elif actions == 3:
            self.state[0] += self.step_size
        
        action_mask = self.getActionMask(Point(self.state[0], self.state[1]))
        states = dict(state=self.state, action_mask=action_mask)
        
        # Compute terminal
        terminal = Point(self.state[0], self.state[1]).toCoordinate().equals(self.goal_coord)
        
        # Compute reward
        reward = 1 if terminal else 0

        return states, terminal, reward


    def getStateBounds(self, dungeon: Level):
        """Returns the boundaries of the state space

        Args:
            dungeon (Level): the dungeon

        Returns:
            array[[x_min, y_min],[x_max, y_max]]: Array containing minimum and maximum values of the 2D state space.
        """
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
        
        return np.array([[x_min, y_min],[x_max, y_max]])


    def getActionMask(self, p: Point):
        """Return array possible actions

        Args:
            p (Point): Current state from which an action is to be taken

        Returns:
            array: Array of booleans indicating which action are possible (true=action is possible, false=action is not possible).
        """
        return np.asarray([
            self.dungeon.getTileAt(Point(p.x, p.y + self.step_size).toCoordinate()).isAccessible(),
            self.dungeon.getTileAt(Point(p.x, p.y - self.step_size).toCoordinate()).isAccessible(),
            self.dungeon.getTileAt(Point(p.x - self.step_size, p.y).toCoordinate()).isAccessible(),
            self.dungeon.getTileAt(Point(p.x + self.step_size, p.y).toCoordinate()).isAccessible()
        ])
