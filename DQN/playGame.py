# -------------------------
# Project: DQN Nature implementation
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import os
import cv2
import gym
from DQN import DQN
import numpy as np
import cPickle as cpickle
import random
from config import createstateDict 
#from config import SAMPLE_STATES, START_NEW_GAME, MAX_FRAMES, CURR_REWARD, EVAL, GAME, K
# SAMPLE_STATES = 100
# START_NEW_GAME = True
# MAX_FRAMES = 50000000
# CURR_REWARD = 0
# EVAL = 50000
# # input the game name here
# GAME='Breakout-v0'
# # number of frames to skip
# K = 4
# preprocess raw image to 84*84 Y channel(luminance)
def preprocess(observation):
    # change color space from RGB to YCrCb
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2YCR_CB)
    #extract the Y channel and resize
    observation = cv2.resize(observation[:,:,0],(84,84))
    return observation

def playKFrames(action,env,stateDict):
    # global START_NEW_GAME
    # global CURR_REWARD
    Reward = 0
    K = stateDict['K']
    for _ in xrange(K):
        env.render()
        observation,localreward,terminal,__=env.step(action)
#        cv2.imshow('game',observation)
#        print observation
        Reward+=localreward
        stateDict['CURR_REWARD'] = Reward
        if _ == K-2:
            prevframe=observation
    observation_=prevframe
    indices = np.where(observation>prevframe)
    observation_[indices] = observation[indices]
    phi = preprocess(observation_)
    change_reward = 0
    if Reward > 0:
        change_reward = 1
    elif Reward < 0:
        change_reward = -1


    if terminal:
        stateDict['START_NEW_GAME'] = True

    return (phi, action, change_reward, terminal)

def evaluate(brain, stateDict):
    # global START_NEW_GAME
    # global CURR_REWARD
    # global EVAL
    #create  anew environment just for evaluation
    env = gym.make(stateDict['GAME'])
    env.reset()
    env.render()
    evalStep = 0
    numEpisode = 1.
    totalReward = 0
    while True:
        if evalStep >= stateDict['NUM_EVAL_STEPS']:
            break
        if stateDict['START_NEW_GAME']:
            numEpisode += 1
            stateDict['START_NEW_GAME'] = False
            env.reset()
            init_state = []
            for i in range(4):
                action0 = env.action_space.sample()
                init_state.append(playKFrames(action0, env, stateDict)[0])
            brain.setInitState(init_state)
        action = brain.getAction(True)

        brain.setPerception(playKFrames(action, env, stateDict), True)
        totalReward += stateDict['CURR_REWARD']
        evalStep += 1

    totalReward /= numEpisode
    env.render(close=True)
    return totalReward
    



def playgame(stateDict):
    # global START_NEW_GAME
    # global MAX_FRAMES
    # global EVAL
    # global SAMPLE_STATES
    # Step 1: init Game
    env = gym.make(stateDict['GAME'])
    # Step 2: init DQN
    actions = env.action_space.n
    brain = DQN(actions, stateDict)
    checkStates = None
    while True:
        if stateDict['START_NEW_GAME']:
            stateDict['START_NEW_GAME'] = False
            env.reset()
            init_state = []
            for i in range(4):
                action0 = env.action_space.sample()
                init_state.append(playKFrames(action0, env, stateDict)[0])
            brain.setInitState(init_state)
        action = brain.getAction()

        brain.setPerception(playKFrames(action, env, stateDict))

        if (brain.timeStep % stateDict['EVAL'] == 0) and (brain.timeStep != 0):

            if (brain.timeStep / stateDict['EVAL'] == 1):
                if not ((os.path.exists("checkStates.txt")) and (os.path.getsize("checkStates.txt") > 0)):
                    minibatch = random.sample(brain.replayMemory, stateDict['SAMPLE_STATES'])
                    checkStates = [data[0] for data in minibatch]
                    with open("checkStates.txt", "w") as fp:
                        cpickle.dump(checkStates,fp)
                else:
                    with open("checkStates.txt", 'r') as fp:
                        checkStates = cpickle.load(fp)

            evalQValues = brain.QValue.eval(feed_dict={brain.stateInput:checkStates})
            maxEvalQValues = np.max(evalQValues, axis = 1)
            avgEvalQValues = np.mean(maxEvalQValues)

            with open("evalQValue.txt", "a") as fp:
                print >>fp,avgEvalQValues
            print avgEvalQValues
            reward = evaluate(brain, stateDict)
            with open("reward.txt", "a") as fp:
                print >> fp, reward

        if (brain.timeStep * stateDict['K']) > stateDict['MAX_FRAMES']:
            break
    brain.session.close()
    env.render(close=True)
    

def main():
    stateDict = createstateDict()
    playgame(stateDict)

if __name__ == '__main__':
    main()
