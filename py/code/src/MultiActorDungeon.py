import random
import numpy as np
from math import sqrt, pow

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

        # Actor indices
        self._actor_1_index = 0
        self._actor_2_index = 1
        self._actor_indices = [self._actor_1_index, self._actor_2_index]

        # State space
        state_indices = [0, 1, 2, 3] # 2D coordinates for two actors
        self._state_indices = np.array(state_indices, np.int32)
        self._state_bounds = self.get_state_bounds(dungeon)

        # Coordinates of all accessible tiles
        self.accessible_coordinates = [
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
        start_actor_1 = random.choice(self.accessible_coordinates)
        start_actor_2 = random.choice(self.accessible_coordinates)
        while (start_actor_1.equals(start_actor_2)):
            start_actor_2 = random.choice(self.accessible_coordinates)
        actor_1_perspective = np.array([start_actor_1.x, start_actor_1.y, start_actor_2.x, start_actor_2.y]).astype(np.float32)
        actor_2_perspective = np.array([start_actor_2.x, start_actor_2.y, start_actor_1.x, start_actor_1.y]).astype(np.float32)
        states = np.stack([actor_1_perspective, actor_2_perspective], axis=0)
        self._states = actor_1_perspective

        # Always for multi-actor environments: return per-actor values
        return self._parallel_indices.copy(), states

    def execute(self, actions):
        # Current and next postion for each actor
        actor_1_action = actions[self._actor_1_index]
        actor_2_action = actions[self._actor_2_index]
        actor_1_position = self.get_current_position(self._actor_1_index)
        actor_2_position = self.get_current_position(self._actor_2_index)
        next_actor_1_position = self.get_next_position(actor_1_position, actor_1_action)
        next_actor_2_position = self.get_next_position(actor_2_position, actor_2_action)

        # Single shared environment logic, plus per-actor perspective
        actor_1_perspective = np.array([next_actor_1_position.x, next_actor_1_position.y, next_actor_2_position.x, next_actor_2_position.y]).astype(np.float32)
        actor_2_perspective = np.array([next_actor_2_position.x, next_actor_2_position.y, next_actor_1_position.x, next_actor_1_position.y]).astype(np.float32)
        terminal = np.stack([False, False], axis=0)
        states = np.stack([actor_1_perspective, actor_2_perspective], axis=0)

        # Reward calculation
        current_distance = self.straight_line_distance(actor_1_position, actor_2_position)
        delta_actor_1 = self.straight_line_distance(next_actor_1_position, actor_2_position) - current_distance
        delta_actor_2 = self.straight_line_distance(next_actor_2_position, actor_1_position) - current_distance
        r = lambda x : 1 if x > 0 else (-1 if x < 0 else 0)
        reward = np.stack([r(delta_actor_1), -r(delta_actor_2)], axis=0)

        # Update internal state
        self._states = actor_1_perspective

        # Always for multi-actor environments: update parallel indices, and return per-actor values
        self._parallel_indices = self._parallel_indices[~terminal]
        return self._parallel_indices.copy(), states, terminal, reward

    def get_state_bounds(self, dungeon: Level):
        # Get all accessible tiles
        accessible_tiles = [tile for room in dungeon.getRooms() for sub_list in room.getLayout() for tile in sub_list if tile.isAccessible()]

        # Get list of x/y coordinates
        x_coordinates = [coordinate.x for coordinate in [tile.getGlobalPosition() for tile in accessible_tiles]]
        y_coordinates = [coordinate.y for coordinate in [tile.getGlobalPosition() for tile in accessible_tiles]]

        # Get min/max values
        x_min = min(x_coordinates)
        x_max = max(x_coordinates)
        y_min = min(y_coordinates)
        y_max = max(y_coordinates)

        return np.array([[x_min, y_min, x_min, y_min],[x_max, y_max, x_max, y_max]])

    def get_next_position(self, position, action: int):
        # Actions
        # 0: go north
        # 1: go south
        # 2: go west
        # 3: go east
        # 4: do nothing
        new_x = position.x
        new_y = position.y
        if action == 0:
            new_y += self.step_size
        elif action == 1:
            new_y -= self.step_size
        elif action == 2:
            new_x -= self.step_size
        elif action == 3:
            new_x += self.step_size

        if (self.dungeon.getTileAt(Point(new_x, new_y).toCoordinate()).isAccessible()):
            return self.Point2D(new_x, new_y)
        return position

    def get_current_position(self, actor_index: int):
        if actor_index not in self._actor_indices:
            raise ValueError(f"{actor_index} is not an allowed actor index. Allowed indices are {self._actor_indices}")

        x_index = 2 * actor_index + 0
        y_index = 2 * actor_index + 1
        return self.Point2D(self._states[x_index], self._states[y_index])

    def straight_line_distance(self, point_1, point_2):
        return sqrt(pow(point_1.x - point_2.x, 2) + pow(point_1.y - point_2.y, 2))

    def disable_action_masking(self):
        pass


if __name__ == '__main__':
    from JavaDungeon import LevelLoader
    from tensorforce.agents import Agent
    from tensorforce.execution import Runner

    dungeon_environment = MultiActorDungeon(
        dungeon=LevelLoader().loadLevel("../../level/level0.json")
    )

    environment = Environment.create(
        environment=dungeon_environment,
        max_episode_timesteps=100
    )

    agent = Agent.create(
        agent="src/Configuration/Agent/multiactor_ppo.json",
        environment=environment
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        max_episode_timesteps=100
    )

    runner.run(num_episodes=10)
    runner.close()
    agent.close()
    environment.close()