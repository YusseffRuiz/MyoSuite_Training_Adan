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
env_walk_sar = "myoSarcLegWalk-v0"
# 80 actions and muscles, 34 DoF
env_rough = "myoLegRoughTerrainWalk-v0"
env_chase = "myoChallengeChaseTagP1-v1"
env_amp = "myoAmpWalk-v0"
env_amp_sar = "myoSarcAmpWalk-v0"
# 22 DoF 78 actions and muscles


tot_episodes = 5
visual = True
randAction = False
plotFlag = False
ampFlag = False


# TODO
# Guardar cada uno de los pasos en frames para armar un video.
# Describir como se entrenó al sistema de red neuronal, compuesto por 2 capas para cada uno de los músculos del cuerpo, con un sistema de entrenamiento MPO



# Initialization to test action space

if ampFlag:
    env = gym.make(env_amp, reset_type="random")
    policy = deprl.load(amp_foldername, env)
    obs = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
else:
    env_w = gym.make(env_walk, reset_type="random")
    policy_w = deprl.load(foldername, env_w)
    obs_w = env_w.reset()
    action_w = policy_w(obs_w)

# Testing of muscles and actuators


# position = env.sim.data.qpos.tolist()
# velocity = env.sim.data.qvel.tolist()
# muscles = env.sim.data.actuator_force.tolist()
# muscle_length = env.sim.data.ten_length.tolist()

# position_w = env_w.sim.data.qpos.tolist()
# velocity_w = env_w.sim.data.qvel.tolist()
# muscles_w = env_w.sim.data.actuator_force.tolist()
# muscle_length_w = env_w.sim.data.ten_length.tolist()

# print("total number of DoF in the model: ", env.sim.model.nv)
# print("total number of DoF in the regular model: ", env_w.sim.model.nv)

# print("Number of actions in the model: ", len(action))
# print("Number of actions in the regular model: ", len(action_w))

# print("Muscles: ", muscles)
# print("Action: ", action)
# print("total muscles: ", len(muscles))
# print("total regular muscles: ", len(muscles_w))


# get the names of every group
# for i in range(env.sim.model.ntendon):
#    print('name of geom ', i, ' : ', env.sim.model.geom(i).name)

"""
Muscles are the one we need to keep tracking, action space only indicates the torque required to perform movement on the muscles.

"""

tb_ant_l = []
psoas_l = []
iliac_l = []
muscle_length = []
motor_action = []


for ep in range(tot_episodes):
    print(f"Episode: {ep + 1} of {tot_episodes}")
    if ampFlag:
        obs = env.reset()
    else:
        obs_w = env_w.reset()
    done = False
    while not done:
        if randAction:
            if ampFlag:
                action = env.action_space.sample()
                action = action * 0
            else:
                action_w = env_w.action_space.sample()
                action_w = action_w * 0
        else:
            if ampFlag:
                action = policy(obs)
            else:
                action_w = policy_w(obs_w)
        if visual:
            if ampFlag:
                env.mj_render()
            else:
                env_w.mj_render()
        # Render env output to video
        if plotFlag:
            muscles = env_w.sim.data.actuator_force.tolist()
            tb_ant_l.append(muscles[75])
            # psoas_l.append(muscles[68])
            # iliac_l.append(muscles[64])
            # motor_action.append(action[])
        if ampFlag:
            next_state, reward, done, info = env.step(action)
            obs = next_state
        else:
            next_state, reward, done, info = env_w.step(action_w)
            obs_w = next_state
    print("Reward: ", reward)
    if ampFlag:
        env.close()
    else:
        env_w.close()

if plotFlag:
    if len(tb_ant_l) < 300:
        print("Not enough Data, please repeat the experiment")
    else:
        tb_ant_l = tb_ant_l[200:300]
        plt.plot(tb_ant_l, label="tb_ant_l", color="blue")
        # plt.plot(psoas_l, label="illipsoas_l", color="red")
        # plt.plot(iliac_l, label="illiacus_l", color="green")
        plt.xlabel("Timesteps")
        plt.ylabel("Muscle")
        plt.title("Muscle force and muscle")
        plt.legend()
        plt.show()


print("Process Finished")


exit()
