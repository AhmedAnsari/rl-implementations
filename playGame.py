# -------------------------
# Project: DQN Nature implementation 
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import cv2
import gym
from BrainDQN_Nature import BrainDQN
import numpy as np

# input the game name here
GAME='Breakout-v0'
# number of frames to skip
K = 4
# preprocess raw image to 84*84 Y channel(luminance)
def preprocess(observation):
    # change color space from RGB to YCrCb
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2YCR_CB)
    #extract the Y channel and resize
    observation = cv2.resize(observation[:,:,0],(84,84))
    return np.array(observation)

def playKFrames(action,env):
    phi=[]
    Reward = 0
    for _ in xrange(K):
        observation,localreward,terminal,__=env.step(action)
        observation = preprocess(observation)
        phi.append(observation)
        Reward+=localreward    
    return (np.array(phi),action,(Reward>0),terminal)
    
def playgame():
    # Step 1: init Game    
    env = gym.make(GAME)    
    # Step 2: init BrainDQN
    actions = env.action_space.n
    brain = BrainDQN(actions)
    # Step 3.1: play game
    env.reset()
    action0 = env.action_space.sample()        
    #remember to edit setInitState    
    brain.setInitState(playKFrames(action0,env)[0])

    # Step 3.2: run the game
    while 1!= 0:
        action = brain.getAction()
        #remember to edit setPerception
        brain.setPerception(playKFrames(action,env))

def main():
    playgame()

if __name__ == '__main__':
    main()
