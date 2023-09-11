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



# env = gym.make('myoLegWalk-v0', reset_type='random')
env = gym.make('myoChallengeChaseTagP1-v0')


policy = deprl.load_baseline(env)
obs = env.reset()
tot_episodes = 5

for ep in range(tot_episodes):
    print(f'Episode: {ep + 1} of {tot_episodes}')
    obs = env.reset()
    done = False
    while not done:
        action = policy(obs)
        # action = env.action_space.sample()
        # uncomment if you want to render the task
        env.mj_render()

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

