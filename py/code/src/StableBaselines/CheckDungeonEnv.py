from stable_baselines3.common.env_checker import check_env
from DungeonGymEnvironment import DungeonGymEnvironment
from JavaDungeon import DummyGenerator

def main():
    env = DungeonGymEnvironment(DummyGenerator().getLevel(), 4)
    check_env(env, warn=True)
    
    print(env.reset())
    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

if __name__ == '__main__':
    main()