from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from DungeonEnv import DungeonEnv
from level.generator.dummy import DummyGenerator

def main():
    env = DungeonEnv(DummyGenerator().getLevel(), 4)
    
    print(env.reset())
    print(env.goal.x, env.goal.y)
    
    env = make_vec_env(lambda: env, n_envs=1)
    
    model = A2C(
        'MlpPolicy',
        env,
        verbose=1,
        #tensorboard_log="./summaries/baseline/"
    )
    
    model.learn(total_timesteps=10_000)
    model.save("model-bl")

if __name__ == '__main__':
    main()