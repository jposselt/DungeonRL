import random
import numpy as np
from math import sqrt, pow
from itertools import cycle, islice

from tensorforce import Environment
from JavaDungeon import Point, Level
from JavaDungeon import is_position_accessible, get_accessible_coordinates, get_dungeon_bounds, point_as_array
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
        self._state_bounds = self.get_state_bounds(dungeon)

        # Coordinates of all accessible tiles
        self.accessible_positions = [coordinate.toPoint() for coordinate in get_accessible_coordinates(dungeon)]

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
        return dict(type=int, num_values=5)

    def num_actors(self):
        return 2  # Indicates that environment has multiple actors

    def reset(self):
        # Always for multi-actor environments: initialize parallel indices
        self._parallel_indices = np.arange(self.num_actors())

        # get random (but different) initial positions for all actors
        self._internal_state = random.sample(self.accessible_positions, self.num_actors())

        # Always for multi-actor environments: return per-actor values
        return self._parallel_indices.copy(), self.external_state()

    def execute(self, actions):
        current_state = self._internal_state
        next_state = self.next_state(actions)
        terminal = self.is_terminal(current_state, actions, next_state)
        reward = self.reward(current_state, actions, next_state)

        # update internal state
        self._internal_state = next_state

        # always for multi-actor environments: update parallel indices, and return per-actor values
        self._parallel_indices = self._parallel_indices[~terminal]

        return self._parallel_indices.copy(), self.external_state(), terminal, reward

    def is_terminal(self, current_state, actions, next_state):
        # return False for all active actors
        return np.full_like(self._parallel_indices, False)

    def reward(self, current_state, actions, next_state):
        # actor indices
        actor_1 = 0
        actor_2 = 1

        # distance between actor positions in the current state
        current_distance = self.straight_line_distance(current_state[actor_1], current_state[actor_2])

        # after action change of distance with respect to the current state
        new_distances = np.array([
            self.straight_line_distance(next_state[actor_1], current_state[actor_2]),
            self.straight_line_distance(next_state[actor_2], current_state[actor_1])
        ])
        delta = new_distances - current_distance

        # set reward to -1, 0 or 1 depending on distance change
        reward_value = lambda x : 1 if x > 0 else (-1 if x < 0 else 0)
        return np.array([
             reward_value(delta[actor_1]),
            -reward_value(delta[actor_2])
        ])

    def get_state_bounds(self, dungeon: Level):
        x_min, y_min, x_max, y_max = get_dungeon_bounds(dungeon)
        return np.array([[x_min, y_min, x_min, y_min],[x_max, y_max, x_max, y_max]])

    def next_position(self, current_position: Point, action: int):
        # Actions {0: go north, 1: go south, 2: go west, 3: go east, 4: do nothing}
        next_position = Point(current_position)
        if action == 0:
            next_position.y += self.step_size
        elif action == 1:
            next_position.y -= self.step_size
        elif action == 2:
            next_position.x -= self.step_size
        elif action == 3:
            next_position.x += self.step_size

        if is_position_accessible(self.dungeon, next_position):
            return next_position
        return current_position

    def next_state(self, actions: list[int]):
        return [
            self.next_position(self._internal_state[i], actions[i])
            for i in self._parallel_indices
        ]

    def straight_line_distance(self, source: Point, destination: Point):
        return sqrt(pow(destination.x - source.x, 2) + pow(destination.y - source.y, 2))

    def actor_perspectives(self):
        """Returns the external state from each actors perspective.

        Each actors perspective is a representation of the evironments state consisting of a list of
        x and y coordinates. The perspective of one actor is such that its own x and y coordinates are
        first in the list followed be the other actors coordinates. In general coordinate values are
        ordered cyclical, meaning for three actor the perspective of the third would be
        [x_3, y_3, x_1, y_1, x_2, y_2].

        Returns:
            list: List of external states
        """
        cyclic_perspective = lambda x: list(islice(cycle(self._internal_state), x, x + len(self._internal_state)))
        perspectives = [
            cyclic_perspective(start)
            for start in range(len(self._internal_state))
        ]

        return [
            np.array(list(map(point_as_array, perspective)), dtype=np.float32).flatten()
            for perspective in perspectives
        ]

    def external_state(self):
        # TODO: Check if stacking is even necessary here. If not actor_perspectives method can be renamed and this part removed.
        return np.stack(self.actor_perspectives(), axis=0)

    def disable_action_masking(self):
        pass
