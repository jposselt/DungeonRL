from stable_baselines3.common.env_checker import check_env
from DungeonEnv import DungeonEnv
from level.generator.dummy import DummyGenerator

def main():
    env = DungeonEnv(DummyGenerator().getLevel(), 4)
    check_env(env, warn=True)
    
    print(env.reset())
    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

if __name__ == '__main__':
    main()