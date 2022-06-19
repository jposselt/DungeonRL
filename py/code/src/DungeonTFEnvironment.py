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
        self._state_bounds = self.get_state_bounds(dungeon)
        
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

        # On/Off switch for action masking
        self.action_masking = True


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
        start_position = random.choice(self.start_coords)
        self.state = np.array([start_position.x, start_position.y]).astype(np.float32)
        action_mask = self.get_action_mask(start_position.x, start_position.y)
        
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
        self.state = self.get_next_state(actions)
        action_mask = self.get_action_mask(self.state[0], self.state[1])
        states = dict(state=self.state, action_mask=action_mask)
        
        # Compute terminal
        terminal = Point(self.state[0], self.state[1]).toCoordinate().equals(self.goal_coord)
        
        # Compute reward
        reward = 1 if terminal else 0

        return states, terminal, reward


    def get_state_bounds(self, dungeon: Level):
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


    def get_action_mask(self, x_position, y_position):
        """Return array possible actions

        Args:
            p (Point): Current state from which an action is to be taken

        Returns:
            array: Array of booleans indicating which action are possible (true=action is possible, false=action is not possible).
            If masking is disabled all values are true.
        """
        if not self.action_masking:
            return np.full(self.actions['num_values'], True)

        return np.asarray([
            self.dungeon.getTileAt(Point(x_position, y_position + self.step_size).toCoordinate()).isAccessible(),
            self.dungeon.getTileAt(Point(x_position, y_position - self.step_size).toCoordinate()).isAccessible(),
            self.dungeon.getTileAt(Point(x_position - self.step_size, y_position).toCoordinate()).isAccessible(),
            self.dungeon.getTileAt(Point(x_position + self.step_size, y_position).toCoordinate()).isAccessible()
        ])

    def set_state(self, state: Point):
        """Sets the current state of the environment if given a valid state parameter (i.e. a reachable state).

        Args:
            state (Point): The desired new state of the environment.

        Returns:
            dict[state, action_mask], bool | 0 | 1 | 2, float: Dictionary containing next state(s)
            and action mask, whether a terminal state is reached or 2 if the episode was
            aborted and observed reward.
        """
        if self.dungeon.getTileAt(state.toCoordinate()).isAccessible():
            self.state = np.array([state.x, state.y]).astype(np.float32)

        action_mask = self.get_action_mask(self.state[0], self.state[1])
        states = dict(state=self.state, action_mask=action_mask)

        return states

    def disable_action_masking(self):
        """Disables action masking"""
        self.action_masking = False

    def get_next_state(self, action):
        """Returns the next state of the environment following a given action.

        Args:
            action (int): Action taken by the agent

        Returns:
            np.array: next state
        """
        nextState = self.state.copy()
        if action == 0:
            nextState[1] += self.step_size
        elif action == 1:
            nextState[1] -= self.step_size
        elif action == 2:
            nextState[0] -= self.step_size
        elif action == 3:
            nextState[0] += self.step_size

        if self.action_masking:
            return nextState

        if self.dungeon.getTileAt(Point(nextState[0], nextState[1]).toCoordinate()).isAccessible():
            return nextState
        return self.state
