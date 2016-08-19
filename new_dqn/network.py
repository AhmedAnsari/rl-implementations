import tensorflow as tf
from .replay_memory import ReplayMemory
from .history import History 

class Network:
    def __init__(self, config):
        #init replay memory
        self.replayMemory = ReplayMemory()
        self.frameQueue = History()
#####################
        #Start Index to keep track of start of Replay Memory
        self.startCounter= 0
        self.currentState = -(config.K) # In Init state dunction we are incrementing so we are starting from negative value here
######################
        self.evalCurrentState = []
        self.evalNextState = []
        #init parameters
        self.timeStep = 0
        self.epsilon = config.INITIAL_EPSILON
        self.actions =  config.NUM_ACTIONS

        #init Q network
        self.stateInput, self.QValue, self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()
        #init target Q network
        self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

        #target_q_value
        self.target = tf.placeholder(tf.float32, [None])
        #id of the target action
        self.target_idx = tf.placeholder(tf.int32, [None])
        #convert the id of the target action to onehot encoding
        action_one_hot = tf.one_hot(self.target_idx, self.actions, 1.0, 0.0, name='action_one_hot')
        #q_value output of the q network
        q_acted = tf.reduce_sum(self.QValue * action_one_hot, reduction_indices = 1, name = 'q_acted')
        #error
        self.delta = self.target - q_acted
        #clipping the error to avoid blowing values out of limits
        if config.clipDelta:
            self.delta = tf.clip_by_value(self.delta, config.minDelta, config.maxDelta, name='clipped_delta')
        #counting the number of steps
        self.global_step = tf.Variable(0, trainable = False)
        self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')
        self.optim = tf.train.RMSPropOptimizer(learning_rate=config.LEARNING_RATE, decay = 1, momentum=config.GRADIENT_MOMENTUM).minimize(self.loss)
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

        W_fc1 = self.weight_variable([7 * 7 * 64, 512]) # Input size yet to confirm
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # input layer
        stateInput = tf.placeholder(tf.float32, [None, 84, 84, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 11 * 11 * 64])        
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2
        
        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'SAME')
