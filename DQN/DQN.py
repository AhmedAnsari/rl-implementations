import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters:
BATCH_SIZE = 32 # size of minibatch
MAX_REPLAY_SIZE = 1000000 # Maximum replay memory size
FRAME_PER_ACTION = 4
GAMMA = 0.99 # decay rate of past observations
ACTION_REPEAT = 4 
UPDATE_FREQUENCY = 10000 # Number of parameter updates after which the target parameters are updated
LEARNING_RATE = 0.00025 
GRADIENT_MOMENTUM = 0.95 
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.01
EXPLORE = 1000000. # frames over which to anneal epsilon
REPLAY_MEMORY = 1000000 # number of previous transitions to remember, REPLAY_START_SIZE
NOOP_MAX = 30

OBSERVE = 50000. # timesteps to observe before training

# FINAL_EPSILON = 0#0.001 # final value of epsilon
# INITIAL_EPSILON = 0#0.01 # starting value of epsilon


# UPDATE_TIME = 100

class DQN:
	def __init__(self, actions):
		#init replay memory
		self.replayMemory = deque()

		#init parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EXPLORATION
		self.actions = actions

		#init Q network
		self.stateInput, self.QValue, self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

		#init target Q network
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

		self.target = tf.placeholder(tf.float32, [None])

		self.target_idx = tf.placeholder(tf.int32, [None])

		action_one_hot = tf.one_hot(self.target_idx, self.actions, 1.0, 0.0, name='action_one_hot')

		q_acted = tf.reduce_sum(self.QValue * action_one_hot, reduction_indices = 1, name = 'q_acted')
		
		self.delta = self.target - q_acted
		
		self.global_step = tf.Variable(0, trainable = False)
		
		self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')
		
		self.optim = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay = 1, momentum=GRADIENT_MOMENTUM).minimize(self.loss)
		
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

		self.trainStep.run(feed_dict={
			self.target : y_batch,										#CHECK AGAIN
			self.target_idx : action_batch,								#CHECK AGAIN
			self.stateInput : state_batch
			})

		# save network every 100000 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)

		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

	def setPerception(self,observation): #nextObservation,action,reward,terminal):
		self.replayMemory.append((self.currentState,observation[1],observation[2],observation[0],observation[3])) #TUPLE : (state, action, reward, nextState, terminal)
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
			# Train the network
			self.trainQNetwork()
		#ACTION needs to be returned in this function
		#PUT in random policy

		# print info
		state = ""
		if self.timeStep <= OBSERVE:
			state = "observe"
		elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		print "TIMESTEP", self.timeStep,# "/ STATE", state, \
        #    "/ EPSILON", self.epsilon
        self.currentState = observation[0]
		self.timeStep += 1

	def setInitState(self,observation):
		self.currentState = observation[0]


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

	# def max_pool_2x2(self, x):
	# 	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1])
