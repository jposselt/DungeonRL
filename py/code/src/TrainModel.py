from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from JavaDungeon import LevelLoader
from DungeonTFEnvironment import DungeonTFEnvironment

def train():
    agentType     = 'ppo'
    topDir        = '../../'
    levelDir      = topDir + 'level/'
    summaryDir    = topDir + 'summaries/tf/test'
    summaryName   = agentType
    modelsDir     = topDir + 'models/tf/test'
    modelName     = agentType
    episodes      = 2
    max_timesteps = 100

    env = DungeonTFEnvironment(LevelLoader().loadLevel(levelDir + 'level0.json'))
    
    environment = Environment.create(
        environment=env,
        max_episode_timesteps=max_timesteps
    )

    agent = Agent.create(
       agent=agentType,
       environment=environment,
       batch_size=10,
       learning_rate=1e-3,
       exploration=0.3,
       #summarizer=dict(directory=summaryDir, filename=summaryName, summaries='all'),
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


if __name__ == '__main__':
    train()