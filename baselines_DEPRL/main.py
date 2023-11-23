""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Modifications made by Adan Dominguez
Based on the work done by:
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import myosuite
import gym
import deprl
import numpy as np

import cv2
import os




# env = gym.make('myoLegWalk-v0', reset_type='random')
#foldername = "baselines_DEPRL\deprl_baseline\myoLeg_Test_0210"
foldername = "baselines_DEPRL\myoLegWalk_20230514\myoLeg"
#foldername = "baselines_DEPRL\deprl_baseline\myo_amputation_1"
env_walk = "myoLegWalk-v0"
env_rough = "myoLegRoughTerrainWalk-v0"
env_chase = "myoChallengeChaseTagP1-v1"
env_amp = "myoAmpWalk-v0"
env = gym.make(env_amp, reset_type='init')


##policy = deprl.load(path=foldername,environment=env)
policy = deprl.load(foldername, env)
tot_episodes = 5
visual = True
randAction = True




# TODO
# Guardar cada uno de los pasos en frames para armar un video.
# Guardar las diferencias entre chaseTag_v0 y ChaseTag_V1, las diferentes variables que se estan leyendo y la ecuacion de rewards.
# Describir lo que se castiga y las opciones de ganar.
# Describir como se entrenó al sistema de red neuronal, compuesto por 2 capas para cada uno de los músculos del cuerpo, con un sistema de entrenamiento MPO
# Describir los métodos fallidos, por medio de los objetivos: izquierda, arriba, abajo y el que ha mejorado.



for ep in range(tot_episodes):
    print(f'Episode: {ep + 1} of {tot_episodes}')
    obs = env.reset()
    done = False
    while not done:
        if randAction:
            action = env.action_space.sample()
        else:
            action = policy(obs)
        if visual:
            env.mj_render()

        # Render env output to video
        next_state, reward, done, info = env.step(action)
        obs = next_state
    print("Reward: ", reward)
    env.close()
print("Process Finished")


"""


for i in range(1000):
    action = policy(obs)
    env.mj_render()
    # action = env.action_space.sample() ### Random Action
    # action = policy(obs)
    next_state, reward, done, info = env.step(action)
    obs = next_state
    ##if done:
    ##    break
env.close()

"""
exit()
