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
import matplotlib.pyplot as plt

import cv2
import os


# env = gym.make('myoLegWalk-v0', reset_type='random')
# foldername = "baselines_DEPRL\deprl_baseline\myoLeg_Test_0210"
foldername = "baselines_DEPRL\myoLegWalk_20230514\myoLeg"
amp_foldername = "baselines_DEPRL\deprl_baseline\myo_amputation_1"
# foldername = "baselines_DEPRL\deprl_baseline\myo_amputation_1"
env_walk = "myoLegWalk-v0"
 # 80 actions and muscles, 34 DoF
env_rough = "myoLegRoughTerrainWalk-v0"
env_chase = "myoChallengeChaseTagP1-v1"
env_amp = "myoAmpWalk-v0"
# 22 DoF 78 actions and muscles
env_w = gym.make(env_walk, reset_type="random", max_episode_steps=1000)
env = gym.make(env_amp, reset_type="random")

tot_episodes = 5
visual = True
randAction = False


# TODO
# Guardar cada uno de los pasos en frames para armar un video.
# Guardar las diferencias entre chaseTag_v0 y ChaseTag_V1, las diferentes variables que se estan leyendo y la ecuacion de rewards.
# Describir lo que se castiga y las opciones de ganar.
# Describir como se entrenó al sistema de red neuronal, compuesto por 2 capas para cada uno de los músculos del cuerpo, con un sistema de entrenamiento MPO


# Initialization to test action space
policy = deprl.load(amp_foldername, env)
obs = env.reset()
action = env.action_space.sample()

policy_w = deprl.load(foldername, env_w)
obs_w = env_w.reset()
action_w = policy_w(obs_w)

#Testing of muscles and actuators

"""
position = env.sim.data.qpos.tolist()
velocity = env.sim.data.qvel.tolist()
muscles = env.sim.data.actuator_force.tolist()
muscle_length = env.sim.data.ten_length.tolist()

position_w = env_w.sim.data.qpos.tolist()
velocity_w = env_w.sim.data.qvel.tolist()
muscles_w = env_w.sim.data.actuator_force.tolist()
muscle_length_w = env_w.sim.data.ten_length.tolist()

print("total number of DoF in the model: ", env.sim.model.nv)
print("total number of DoF in the regular model: ", env_w.sim.model.nv)

print("Number of actions in the model: ", len(action))
print("Number of actions in the regular model: ", len(action_w))

print("Muscles: ", muscles)
print("total muscles: ", len(muscles))
print("total regular muscles: ", len(muscles_w))
print("muscle ", 0, ": ",  muscles[0])
print("action ", 0, ": ",  action[0])
"""
#get the names of every group
#for i in range(env.sim.model.ntendon):
#    print('name of geom ', i, ' : ', env.sim.model.tendon(i).name)

tb_ant_l = []
psoas_l = []
iliac_l = []
muscle_length = []

for ep in range(tot_episodes):
    print(f"Episode: {ep + 1} of {tot_episodes}")
    obs = env.reset()
    done = False
    while not done:
        if randAction:
            action = env.action_space.sample()
            action = action*0
        else:
            action = policy(obs)
        if visual:
            env.mj_render()
        # Render env output to video
        # tb_ant_l.append(action[75])
        # psoas_l.append(action[68])
        # iliac_l.append(action[64])
        # muscle_length.append(env.sim.data.ten_length.tolist()[75])
        next_state, reward, done, info = env.step(action)
        obs = next_state
    print("Reward: ", reward)
    env.close()
"""

plt.plot(tb_ant_l, label="tb_ant_l", color="blue")
plt.plot(psoas_l, label="illipsoas_l", color="red")
plt.plot(iliac_l, label="illiacus_l", color="green")
plt.xlabel("Timesteps")
plt.ylabel("Muscle")
plt.title("Muscle force and muscle")
plt.legend()
plt.show()

"""
print("Process Finished")


exit()
