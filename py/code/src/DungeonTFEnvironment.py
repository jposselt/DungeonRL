import random
import numpy as np
from tensorforce import Environment
from JavaDungeon import Point, Level
from JavaDungeon import is_position_accessible, get_accessible_coordinates, get_dungeon_bounds

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
        self.goal_coordinate = dungeon.getEndTile().getGlobalPosition()
        self.start_positions = [
            coordinate.toPoint()
            for coordinate in get_accessible_coordinates(self.dungeon)
            if not coordinate.equals(self.goal_coordinate)
        ]

        # Step size
        self.step_size = 1

        # On/Off switch for action masking
        self.action_masking = True


    def states(self):
        """Returns the specification for external states.

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
            dict[state, action_mask]: Dictionary containing initial state(s) and action mask. See "get_action_mask" method
            for meaning of the mask.
        """
        self._internal_state = random.choice(self.start_positions)
        return dict(state=self.get_external_state(), action_mask=self.get_action_mask())


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
        self._internal_state = self.get_next_state(actions)
        states = dict(state=self.get_external_state(), action_mask=self.get_action_mask())
        
        # Compute terminal
        terminal = Point(self._internal_state.x, self._internal_state.y).toCoordinate().equals(self.goal_coordinate)
        
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
        x_min, y_min, x_max, y_max = get_dungeon_bounds(dungeon)
        return np.array([[x_min, y_min],[x_max, y_max]])


    def get_action_mask(self):
        """Returns array of possible actions

        Returns:
            array[bool]: Array of booleans indicating which action are possible for the current internal state
            (true=action is possible, false=action is not possible). If action masking is disabled all values are true.
        """
        if not self.action_masking:
            return np.full(self.actions['num_values'], True)

        step_north = Point(self._internal_state.x, self._internal_state.y + self.step_size)
        step_south = Point(self._internal_state.x, self._internal_state.y - self.step_size)
        step_east  = Point(self._internal_state.x + self.step_size, self._internal_state.y)
        step_west  = Point(self._internal_state.x - self.step_size, self._internal_state.y)

        return np.asarray([
            is_position_accessible(self.dungeon, step_north), is_position_accessible(self.dungeon, step_south),
            is_position_accessible(self.dungeon, step_west), is_position_accessible(self.dungeon, step_east),
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
        if is_position_accessible(self.dungeon, state):
            self._internal_state = state

        return dict(state=self.get_external_state(), action_mask=self.get_action_mask())

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
        next_state = Point(self._internal_state)
        if action == 0:
            next_state.y += self.step_size
        elif action == 1:
            next_state.y -= self.step_size
        elif action == 2:
            next_state.x -= self.step_size
        elif action == 3:
            next_state.x += self.step_size

        if self.action_masking or is_position_accessible(self.dungeon, next_state):
            return next_state

        return self._internal_state

    def get_external_state(self):
        """Returns the external representation of the internal state

        Returns:
            array: Array of state variables
        """
        return np.array([self._internal_state.x, self._internal_state.y]).astype(np.float32)
