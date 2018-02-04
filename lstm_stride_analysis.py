from __future__ import division, print_function
from collections import Counter, defaultdict

import random
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


class Trace:
    def __init__(self, fileAddr):
        self.fileAddr = fileAddr

    def readFile(self):
        instructions = []
        dataAddr = []
        with open(self.fileAddr) as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(' ')
                trace_type = int(data[0], 10)
                addr = int(data[1], 16)
                if trace_type in [0,1]:
                    dataAddr.append(addr)
                elif trace_type == 2:
                    instructions.append(addr)
                else:
                    print(data[0])
        self.dataAddr = np.array(dataAddr)
        self.instructions = np.array(instructions)

    def getTags(self, num_bits):
    	tmp = [bin(addr) for addr in self.dataAddr]
    	self.tags = [int(item[:-num_bits],2) for item in tmp]
    	return self.tags

    def getTagStrides(self, tags, normalize):
    	tag_strides = [tags[k]-tags[k-1] for k in range(1, len(tags))]
    	if not normalize:
    		return tag_strides
    	m = np.mean(tag_strides)
    	s = np.std(tag_strides)
    	return [(item-m)/s for item in tag_strides]


def split_train_test(data, seq_len, target_len, normalise_window):
    sequence_length = seq_len + target_len
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-target_len]
    y_train = train[:, -target_len:]
    x_test = result[int(row):, :-target_len]
    y_test = result[int(row):, -target_len:]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test] 

def main():
	trace = Trace('trace.din')
	trace.readFile()
	dataAddr = trace.dataAddr

	num_bits = 6
	tags = trace.getTags(num_bits)
	tag_strides = trace.getTagStrides(tags, True)

	sd = 300
	np.random.seed(sd)
	seq_len = 500
	target_len = 64
	[x_train, y_train, x_test, y_test] = split_train_test(tag_strides, 
														  seq_len,
														  target_len,
														  False)

	model = Sequential()
	
	model.add(LSTM(
	    input_dim=1,
	    output_dim=50,
	    return_sequences=True))
	
	model.add(Dropout(0.2))
	
	model.add(LSTM(
    100,
    return_sequences=False))
	
	model.add(Dropout(0.2))

	model.add(Dense(
    output_dim=target_len))

	model.add(Activation('linear'))
	start = time.time()
	model.compile(loss='mse', optimizer='rmsprop')
	print('compilation time : ', time.time() - start)

	pdb.set_trace()
	model.fit(
    x_train,
    y_train,
    batch_size=512,
    nb_epoch=30,
    validation_split=0.05)

if __name__ == "__main__":
	main()
