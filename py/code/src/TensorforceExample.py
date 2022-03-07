from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from DungeonEnv import DungeonEnv
from level.generator.dummy import DummyGenerator

def main():
    env = DungeonEnv(DummyGenerator().getLevel(), 4)
    
    environment = Environment.create(
        environment='gym',
        level=env,
        max_episode_timesteps=500
    )

    agent_type = "dqn"
    
    agent = Agent.create(
        agent=agent_type,
        environment=environment,
        batch_size=10,
        memory=10000,
        learning_rate=1e-3,
        #summarizer=dict(directory='../../summaries/tf', filename=agent_type, summaries='all')
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        max_episode_timesteps=500
    )

    runner.run(num_episodes=200)
    
    agent.save(directory='../../models/tf', filename=agent_type, format='saved-model', append='episodes')
    
    runner.close()
    agent.close()
    environment.close()

if __name__ == '__main__':
    main()