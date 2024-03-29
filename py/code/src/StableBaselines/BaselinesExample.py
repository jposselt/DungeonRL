from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from DungeonGymEnvironment import DungeonGymEnvironment
from JavaDungeon import LevelLoader

def main():
    topDir       = '../../'
    levelDir     = topDir + 'level/'
    summariesDir = topDir + 'summaries/baseline/'
    modelsDir    = topDir + 'models/baseline/'
    agentType    = "ppo"
    timesteps    = 100
    
    environment = DungeonGymEnvironment(LevelLoader().loadLevel(levelDir + 'level0.json'), 4)
    environment = make_vec_env(lambda: environment, n_envs=1)
    
    model = PPO(
        'MlpPolicy',
        environment,
        verbose=1,
        #tensorboard_log=summariesDir + agentType
    )
    
    model.learn(total_timesteps=timesteps)
    model.save(modelsDir + agentType)

if __name__ == '__main__':
    main()