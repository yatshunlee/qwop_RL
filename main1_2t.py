# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 03:24:50 2022
rOPrrWO
@author: damieQWQ
"""

import os
from time import sleep, time
from game_env_2t import qwopEnv
from stable_baselines3 import DQN
import stable_baselines3.common.vec_env

def get_new_model():
    env= qwopEnv()
    model = DQN("MlpPolicy", env,policy_kwargs = dict(net_arch=[256, 128]),
                exploration_final_eps = 0.06,learning_rate = 0.00008,verbose=1)
    
    return model

def run_train():
    model = get_new_model()
    #model.train_freq=5
    model.learning_starts = 1800
    model.exploration_fraction=0.2
    model.learn(total_timesteps=12000, log_interval=5)
    model.save("YDv2_1")

def run_test():
    env= qwopEnv()
    model = DQN.load("YDv2_1")
    
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

def run_train_old():
    model_name = "YDv2_1"

    model = DQN.load(model_name)
    env =qwopEnv()  # SubprocVecEnv([lambda: QWOPEnv()])
    model.set_env(env)
    sleep(1)
    model.learning_rate = 0.00004
    #model.train_freq=5
    model.learning_starts = 200
    model.learn(total_timesteps=500, log_interval=5)
    model.save(model_name)



run_test()


'''
W
env= qwopEnv() 
sleep (5)

for i in range(20):
    env.step(0)
    env.step(1)P
    sleep(0.5)
    
env.resetr() '''