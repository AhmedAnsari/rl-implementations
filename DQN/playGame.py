# -------------------------
# Project: DQN Nature implementation 
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import cv2
import gym
from DQN import DQN
import numpy as np

START_NEW_GAME = True
MAX_FRAMES = 50000000


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
    Reward = 0
    for _ in xrange(K):
        observation,localreward,terminal,__=env.step(action)
        Reward+=localreward
    phi = preprocess(observation)
    change_reward = 0
    if Reward > 0:
    	change_reward = 1
    elif Reward < 0:
    	change_reward = -1

    if terminal:
    	START_NEW_GAME = True

    return (np.array(phi, dtype = np.float32), action, change_reward, terminal)
    
def playgame():
    # Step 1: init Game    
    env = gym.make(GAME)    
    # Step 2: init DQN
    actions = env.action_space.n
    brain = DQN(actions)
    while True:
    	if START_NEW_GAME:
    		START_NEW_GAME = False
	    	env.reset()
	    	init_state = []
	    	for i in range(4):
			    action0 = env.action_space.sample()
			    init_state.append(playKFrames(action0,env)[0])
			brain.setInitState(init_state)
        action = brain.getAction()
        
        brain.setPerception(playKFrames(action,env))

        if (brain.timeStep * K) > MAX_FRAMES:
        	break
    brain.session.close()

def main():
    playgame()

if __name__ == '__main__':
    main()
