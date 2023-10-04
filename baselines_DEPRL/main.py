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

#import cv2

def grabFrame(env):
    # Get RGB rendering of env
    frame = env.physics.render(480, 600, camera_id=0)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



# env = gym.make('myoLegWalk-v0', reset_type='random')
foldername = "baselines_DEPRL\deprl_baseline\myoLeg_Test_0210"
#foldername = "baselines_DEPRL\myoLegWalk_20230514\myoLeg"
env_name = "myoLegWalk-v0"
env_chase = "myoChallengeChaseTagP1-v1"
env = gym.make(env_chase, reset_type='random')


policy = deprl.load(path=foldername,environment=env)
#policy = deprl.load(foldername, env)
tot_episodes = 5
visual = True
randAction = False

# Setup video writer - mp4 at 30 fps
#video_name = 'walking.mp4'
#frame = grabFrame(env)
#height, width, layers = frame.shape
#video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))



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
        #frame = grabFrame(env)
        # Render env output to video
        #video.write(grabFrame(env))
        next_state, reward, done, info = env.step(action)
        obs = next_state
    print("Reward: ", reward)
    env.close()
print("Process Finished")


# End render to video file
#video.release()

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
