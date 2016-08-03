#Configurations for Atari Games
def createstateDict():
    stateDict = {}
    #parameters for DQN
    stateDict['BATCH_SIZE'] = 32 # size of minibatch
    stateDict['REPLAY_MEMORY'] = 1000000 # number of previous transitions to remember,
    stateDict['GAMMA'] = 0.99 # decay rate of past observations
    stateDict['UPDATE_FREQUENCY'] = 10000 # Number of parameter updates after which the target parameters are updated
    stateDict['LEARNING_RATE'] = 0.00025
    stateDict['GRADIENT_MOMENTUM'] = 0.95
    stateDict['SQUARED_GRADIENT_MOMENTUM'] = 0.95
    stateDict['MIN_SQUARED_GRADIENT'] = 0.01
    stateDict['INITIAL_EPSILON'] = 1 # starting value of epsilon
    stateDict['FINAL_EPSILON'] = 0.01 # final value of epsilon
    stateDict['EXPLORE'] = 1000000. # frames over which to anneal epsilon
    stateDict['REPLAY_START_SIZE'] = 50 #minimum number of previous transitions to be stored before training starts
    #parameters for the game
    stateDict['SAMPLE_STATES'] = 100
    stateDict['START_NEW_GAME'] = True
    stateDict['MAX_FRAMES'] = 50000000
    stateDict['CURR_REWARD'] = 0
    stateDict['EVAL'] = 50000
    stateDict['GAME'] = 'Breakout-v0'
    stateDict['K'] = 4
    return stateDict