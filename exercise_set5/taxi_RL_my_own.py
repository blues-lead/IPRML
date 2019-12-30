# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:19:58 2019

@author: Anton
"""

import gym
import random
import numpy
import time
env = gym.make("Taxi-v2")

q_table = np.zeros([env.observation_space.n,env.action_space.n])

N = 20000
for episode in range(0,N):
    state = env.reset()
    done = False
    reward = 0
    while not done:
        action = np.argmax(q_table[state])
        #print("Selected_action = ", action)
        new_state, reward, done, info = env.step(action)
        q_table[state,action] += 0.4*reward + 0.6*(np.max(q_table[new_state])-q_table[state,action])
        state = new_state

#q_table = np.zeros([env.observation_space.n,env.action_space.n])

test_tot_reward = 0
test_tot_actions = 0
past_observation = -1
observation = env.reset();
for t in range(50):
    test_tot_actions = test_tot_actions+1
    action = numpy.argmax(q_table[observation])
    if (observation == past_observation):
        # This is done only if gets stuck
        action = random.sample(range(0,6),1)
        action = action[0]
    past_observation = observation
    observation, reward, done, info = env.step(action)
    test_tot_reward = test_tot_reward+reward
    env.render()
    time.sleep(1)
    if done:
        break
print("Total reward: ")
print(test_tot_reward)
print("Total actions: ")
print(test_tot_actions)
    