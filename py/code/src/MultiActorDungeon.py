import numpy as np

from tensorforce import Environment

class MultiActorDungeon(Environment):
    """An RF learning environment with multiple actors.
    Based on https://github.com/tensorforce/tensorforce/blob/master/examples/multiactor_environment.py
    """

    def __init__(self):
        super().__init__()
        pass

    def states(self):
        pass

    def actions(self):
        pass

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