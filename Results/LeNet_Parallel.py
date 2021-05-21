import random
from mpi4py import MPI
from collections import deque
import time
import math

from __future__ import print_function
import keras
import tensorflow as tf 
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

import keras
import tensorflow as tf 
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from tensorflow import keras
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn import datasets
import scipy.stats as stats

import optunity
import optunity.metrics

%%time
import random
from mpi4py import MPI
from collections import deque
import time
import math

# ARTIFICIALLY EXPENSIVE ALPINE OBJECTIVE FUNCTION
def model(parameter,x_train,y_train,x_test,y_test):
  lr = parameter[0]
  epochs = math.floor(parameter[2])
  batch_size= math.floor(parameter[1])
  val_x = x_train[:5000]
  val_y = y_train[:5000]
  lenet_5_model = keras.models.Sequential([
    keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=x_train[0].shape, padding='same'), #C1
    keras.layers.AveragePooling2D(), #S2
    keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
    keras.layers.AveragePooling2D(), #S4
    keras.layers.Flatten(), #Flatten
    keras.layers.Dense(120, activation='tanh'), #C5
    keras.layers.Dense(84, activation='tanh'), #F6
    keras.layers.Dense(10, activation='softmax') #Output layer
  ])
  opt = keras.optimizers.Adam(learning_rate=lr)
  lenet_5_model.compile(optimizer=opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
  lenet_5_model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))
  scores = lenet_5_model.evaluate(x_test, y_test,verbose = 1)
  print(scores[0])
  return scores[0]

class Particle:
	def __init__(self, x0):
		self.position=[]
		self.velocity=[]
		self.best_pos_in=[]
		self.best_cost_in=float('inf')
		self.cost=float('inf')

		for i in range(0, num_dimensions):
			self.velocity.append(random.uniform(-1, 1))
			self.position.append(random.uniform(bounds[0], bounds[1]))



	def update_velocity(self, best_pos_g):
		w=0.85
		c1=1
		c2=2

		for i in range(0, num_dimensions):
			r1=random.random()
			r2=random.random()

			vel_cognitive=c1*r1*(self.best_pos_in[i]-self.position[i])
			vel_social= c2*r2*(best_pos_g[i]-self.position[i])
			self.velocity[i]= w*self.velocity[i]+vel_social+vel_cognitive


	def update_position(self, bounds):
		for i in range(0, num_dimensions):
			self.position[i]+=self.velocity[i]

			if self.position[i]<bounds[0]:
				self.position[i]=bounds[0]

			if self.position[i]>bounds[1]:
				self.position[i]=bounds[1]


class PSO():
	def __init__(self, num_d, bounds, num_particles, num_iter):
		global num_dimensions
		num_dimensions=num_d

		best_cost_g=float('inf')
		best_pos_g=[]

		swarm=[]
		for i in range(0, num_particles):
			swarm.append(Particle(bounds))

		for i in range(num_iter):

			evalQueue = deque(range(num_particles))


			# POP AND SEND PARTICLES TO EACH SLAVE NODE
			for i in range(1, size):
				p = evalQueue.popleft()
				obj_comm = (p, swarm[p].position)
				comm.send(obj_comm, dest=i)

			idle=0
			# FURTHER LOOPING
			while(1):
				obj_recv = comm.recv(source = MPI.ANY_SOURCE, status=status)
				id_recv = obj_recv[0]
				f_recv = obj_recv[1]
				src_rank = status.Get_source()

				swarm[id_recv].cost = f_recv
				if f_recv < swarm[id_recv].best_cost_in:
					swarm[id_recv].best_pos_in = list(swarm[id_recv].position)
					swarm[id_recv].best_cost_in = float(f_recv)

				if f_recv < best_cost_g :
					best_cost_g = float(f_recv)
					best_pos_g = list(swarm[id_recv].position)

				if len(evalQueue)!=0:
					j= evalQueue.popleft()
					obj_comm = (j, swarm[j].position)
					comm.send(obj_comm, dest = src_rank)
				else:
					idle+=1

				if idle==size-1:
					break


			for j in range(0, num_particles):
				swarm[j].update_velocity(best_pos_g)
				swarm[j].update_position(bounds)

		
		for k in range(1,size):
			comm.send(0, dest=k, tag=200)   
		# print ('Best position : ')
		# print (best_pos_g)
		# print ('Best cost : ')
		# print (best_cost_g)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()

if rank==0:
	start_time = time.time()
	num_d=3
	bounds=[(0.0001,0.1),(16,256),(1,100)]    # upper and lower bounds of variables #learning rate #batch size #epochs 
	PSO(num_d, bounds, num_particles=4, num_iter=20)
	print("time taken:")
	print(time.time()-start_time)
else:
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
	while(1):
		obj_recv = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
		tag = status.Get_tag()
		if (tag == 200):
			break
		f = model(obj_recv[1],x_train,y_train,x_test,y_test)
		obj_sent = (obj_recv[0], f)
		comm.send(obj_sent, dest=0)
