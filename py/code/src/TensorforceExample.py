from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from DungeonEnv import DungeonEnv
from level.generator.LevelLoader import LevelLoader

def main():
    agentType     = 'ppo'
    topDir        = '../../'
    levelDir      = topDir + 'level/'
    summaryDir    = topDir + 'summaries/tf/base'
    summaryName   = agentType
    modelsDir     = topDir + 'models/tf/base'
    modelName     = agentType
    episodes      = 10000
    max_timesteps = 100

    env = DungeonEnv(
        LevelLoader().loadLevel(levelDir + 'level0.json'),
        4
    )
    
    environment = Environment.create(
        environment='gym',
        level=env,
        max_episode_timesteps=max_timesteps
    )

    agent = Agent.create(
        agent=agentType,
        environment=environment,
        batch_size=10,
        learning_rate=1e-3,
        exploration=0.1,
        summarizer=dict(directory=summaryDir, filename=summaryName, summaries='all')
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        max_episode_timesteps=max_timesteps
    )

    runner.run(num_episodes=episodes)
    
    agent.save(directory=modelsDir, filename=modelName, format='numpy',       append='episodes')
    agent.save(directory=modelsDir, filename=modelName, format='saved-model', append='episodes')
    
    runner.close()
    agent.close()
    environment.close()

if __name__ == '__main__':
    main()