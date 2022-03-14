from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from DungeonEnv import DungeonEnv
from level.generator.LevelLoader import LevelLoader

def main():
    topDir = '../../'
    levelDir = topDir + 'level/'
    #summariesDir = topDir + 'summaries/tf'
    #modelsDir = topDir + 'models/tf'
    #agentType = 'ppo'
    episodes = 2

    env = DungeonEnv(
        LevelLoader().loadLevel(levelDir + 'level0.json'),
        4
    )
    
    environment = Environment.create(
        environment='gym',
        level=env,
        max_episode_timesteps=500
    )

    agent = Agent.create(
        agent='ppo',
        environment=environment,
        batch_size=10,
        learning_rate=1e-3,
        #summarizer=dict(directory=summariesDir, filename=agentType, summaries='all')
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        max_episode_timesteps=500
    )

    runner.run(num_episodes=episodes)
    
    #agent.save(directory='model-numpy', filename=agentType, format='numpy', append='episodes')
    #agent.save(directory='model-tf', filename=agentType, format='saved-model', append='episodes')
    
    runner.close()
    agent.close()
    environment.close()

if __name__ == '__main__':
    main()