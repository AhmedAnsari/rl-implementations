#Configurations for Atari Games


BATCH_SIZE = 32 # size of minibatch
REPLAY_MEMORY = 1000000 # number of previous transitions to remember,
GAMMA = 0.99 # decay rate of past observations
UPDATE_FREQUENCY = 10000 # Number of parameter updates after which the target parameters are updated
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EXPLORE = 1000000. # frames over which to anneal epsilon
REPLAY_START_SIZE = 50 #minimum number of previous transitions to be stored before training starts

SAMPLE_STATES = 100
START_NEW_GAME = True
MAX_FRAMES = 50000000
CURR_REWARD = 0
EVAL = 50000
GAME='Breakout-v0'
K = 4