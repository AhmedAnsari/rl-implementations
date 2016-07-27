import tensorflow as tf
import numpy as np
import random
from collections import deque
#import os
#os.chdir('/Users/ghulamahmedansari/Documents/Python Scripts/iitm-rl/DQN')
# Hyper Parameters:
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
REPLAY_START_SIZE = 50000 #minimum number of previous transitions to be stored before training starts

class DQN:
    def __init__(self, actions):
        global LEARNING_RATE
        global GRADIENT_MOMENTUM
        global INITIAL_EPSILON
        #init replay memory
        self.replayMemory = deque()

        #init parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions

        #init Q network
        self.stateInput, self.QValue, self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

        #init target Q network
        self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

#               self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

        self.target = tf.placeholder(tf.float32, [None])

        self.target_idx = tf.placeholder(tf.int32, [None])

        action_one_hot = tf.one_hot(self.target_idx, self.actions, 1.0, 0.0, name='action_one_hot')

        q_acted = tf.reduce_sum(self.QValue * action_one_hot, reduction_indices = 1, name = 'q_acted')

        self.delta = self.target - q_acted

        self.global_step = tf.Variable(0, trainable = False)

        self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')

        self.optim = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay = 1, momentum=GRADIENT_MOMENTUM).minimize(self.loss)

        self.session = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        tf.initialize_all_variables().run()        
        # self.session.run(self.copyTargetQNetworkOperation)
        self.copyTargetQNetworkOperation()

    def copyTargetQNetworkOperation(self):
        self.W_conv1T.assign(self.W_conv1)
        self.b_conv1T.assign(self.b_conv1)
        self.W_conv2T.assign(self.W_conv2)
        self.b_conv2T.assign(self.b_conv2)
        self.W_conv3T.assign(self.W_conv3)
        self.b_conv3T.assign(self.b_conv3)
        self.W_fc1T.assign(self.W_fc1)
        self.b_fc1T.assign(self.b_fc1)
        self.W_fc2T.assign(self.W_fc2)
        self.b_fc2T.assign(self.b_fc2)

    def createQNetwork(self):
        #network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([11 * 11 * 64, 512]) # Input size yet to confirm
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # input layer

        stateInput = tf.placeholder("float", [None, 84, 84, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 11 * 11 * 64])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

    def trainQNetwork(self):
        global BATCH_SIZE
        global GAMMA
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
        for i in range(0,BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.optim.run(feed_dict={
                self.target : y_batch,                                                                          #CHECK AGAIN
                self.target_idx : action_batch,                                                         #CHECK AGAIN
                self.stateInput : state_batch
                })

        # save network every 10000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'Savednetworks/'+'network' + '-dqn', global_step = self.timeStep)

        if self.timeStep % UPDATE_FREQUENCY == 0:
            # self.session.run(self.copyTargetQNetworkOperation)
            # self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
            self.copyTargetQNetworkOperation()
    def setPerception(self,observation): #nextObservation,action,reward,terminal):
        global EXPLORE
        global REPLAY_MEMORY
        global REPLAY_START_SIZE
        
        nextState = np.append(self.currentState[:,:,1:], observation[0].reshape((84,84,1)),axis = 2)
        self.replayMemory.append((self.currentState,observation[1],observation[2],nextState,observation[3])) #TUPLE : (state, action, reward, nextState, terminal)
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > REPLAY_START_SIZE and len(self.replayMemory) > REPLAY_START_SIZE:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= REPLAY_START_SIZE:
            state = "observe"
        elif self.timeStep > REPLAY_START_SIZE and self.timeStep <= REPLAY_START_SIZE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print "TIMESTEP", self.timeStep,# "/ STATE", state, \
        #    "/ EPSILON", self.epsilon
        self.currentState = nextState
        self.timeStep += 1


    def getAction(self):
        global EXPLORE
        global REPLAY_START_SIZE
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.actions)
        else:
            QValue = self.QValue.eval(feed_dict={self.stateInput:[self.currentState]})[0]
            action_index = np.argmax(QValue)


        if self.epsilon > FINAL_EPSILON and self.timeStep > REPLAY_START_SIZE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action_index

    def setInitState(self,observation):
        self.currentState = np.array(observation).reshape([84,84,4])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        '''
                Normal convolution proposed initially
        '''
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'SAME')
