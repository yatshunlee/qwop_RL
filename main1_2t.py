# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 03:24:50 2022
rOPrrWO
@author: damieQWQ
"""
import os

import os
from time import sleep, time
from game_env_2t import qwopEnv
from stable_baselines3 import DQN
import stable_baselines3.common.vec_env

def get_new_model():
    env= qwopEnv()
    model = DQN("MlpPolicy", env,policy_kwargs = dict(net_arch=[256, 128]),
                exploration_final_eps = 0.075,learning_rate = 0.00008,verbose=1,device="cpu")
    
    return model

def run_train():
    model_name = "YDv3_3"
    model = get_new_model()
    #model.train_freq=5
    model.learning_starts = 1000
    model.exploration_fraction=0.2
    model.learn(total_timesteps=10000, log_interval=5)
    model.save(model_name)
    
    for i in range(50):
        model.learn(total_timesteps=1000, log_interval=5)
        model.save(model_name+str("_f")+str(i))
        print("round {}".format(i))
        
def run_test():
    env= qwopEnv()
    model = DQN.load("YDv3_3_f78")
    
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

def run_train_old():
    model_name = "YDv3_3_f78"
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model_name2 = "YDv3_3"
    model = DQN.load(model_name,learning_rate = 0.00005,device="cpu")
    env =qwopEnv()  # SubprocVecEnv([lambda: QWOPEnv()])
    model.set_env(env)
    sleep(1)
    model.learning_rate = 0.00005
    #model.train_freq=5
    model.learning_starts = 100
    model.exploration_final_eps = 0.05
    for i in range(79,100):
        model.learn(total_timesteps=1000, log_interval=5)
        model.save(model_name2+str("_f")+str(i))
        print("rpound {}".format(i))


run_test()


#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

'''
W
env= qwopEnv() r
sleep (5)

for i in range(20):
    env.step(0)
    env.step(1)P
    sleep(0.5)
    
env.resetr() '''