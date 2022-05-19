import random
import numpy as np
from tensorforce import Environment
from JavaDungeon import Point, Level

class ActionMaskEnv(Environment):
    def __init__(self, dungeon: Level):
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
        return dict(
            type=float,
            shape=tuple(self._state_indices.shape),
            min_value=self._state_bounds[0, self._state_indices],
            max_value=self._state_bounds[1, self._state_indices]
        )


    def actions(self):
        return dict(type=int, num_values=4)


    def reset(self):
        # Initial state and associated action mask
        start = random.choice(self.start_coords)
        self.state = np.array([start.x, start.y]).astype(np.float32)
        action_mask = self.getActionMask(Point(start.x, start.y))
        
        # Add action mask to states dictionary (mask item is "[NAME]_mask", here "action_mask")
        states = dict(state=self.state, action_mask=action_mask)

        return states


    def execute(self, actions):
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
        return np.asarray([
            self.dungeon.getTileAt(Point(p.x, p.y + self.step_size).toCoordinate()).isAccessible(),
            self.dungeon.getTileAt(Point(p.x, p.y - self.step_size).toCoordinate()).isAccessible(),
            self.dungeon.getTileAt(Point(p.x - self.step_size, p.y).toCoordinate()).isAccessible(),
            self.dungeon.getTileAt(Point(p.x + self.step_size, p.y).toCoordinate()).isAccessible()
        ])


if __name__ == '__main__':
    from tensorforce.environments import Environment
    from tensorforce.agents import Agent
    from tensorforce.execution import Runner
    from JavaDungeon import LevelLoader
    
    agentType     = 'ppo'
    topDir        = '../../'
    levelDir      = topDir + 'level/'
    summaryDir    = topDir + 'summaries/tf/base'
    summaryName   = agentType
    modelsDir     = topDir + 'models/tf/base'
    modelName     = agentType
    episodes      = 10000
    max_timesteps = 100

    env = ActionMaskEnv(LevelLoader().loadLevel(levelDir + 'level0.json'))
    
    environment = Environment.create(
        environment=env,
        max_episode_timesteps=max_timesteps
    )

    agent = Agent.create(
        agent=agentType,
        environment=environment,
        batch_size=10,
        learning_rate=1e-3,
        exploration=0.1,
        summarizer=dict(directory=summaryDir, filename=summaryName, summaries='all'),
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        max_episode_timesteps=max_timesteps
    )

    runner.run(num_episodes=episodes)
    
    agent.save(directory=modelsDir,          filename=modelName, format='saved-model', append='episodes')
    agent.save(directory=modelsDir+'/numpy', filename=modelName, format='numpy',       append='episodes')
    
    runner.close()
    agent.close()
    environment.close()