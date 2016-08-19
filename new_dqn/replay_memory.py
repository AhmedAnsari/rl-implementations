# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:06:37 2016

@author: ghulamahmedansari
"""
from collections import deque

class ReplayMemory():
    def __init__(self,config,model_dir):
        self.model_dir = model_dir
        self.memory_size = config.REPLAY_MEMORY
        self.actions = deque()
        self.rewards = deque()
        self.screens = deque()
        self.terminals = deque()
        self.history_length = config.history_length
        self.dims = (config.screen_height, config.screen_width)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0