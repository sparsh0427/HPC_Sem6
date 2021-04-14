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

# Training parameters

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

# Training parameters
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





import random
import math
import matplotlib.pyplot as plt

bounds=[(0.0001,0.1),(16,256),(1,100)]   # upper and lower bounds of variables #learning rate #batch size #epochs #growth rate

nv = 3                   # number of variables
mm = -1                   # if minimization problem, mm = -1; if maximization problem, mm = 1
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
# THE FOLLOWING PARAMETERS ARE OPTINAL.
particle_size=4         # number of particles
iterations=20         # max number of iterations
w=0.85                    # inertia constant
c1=1                    # cognative constant
c2=2                     # social constant
# END OF THE CUSTOMIZATION SECTION
#------------------------------------------------------------------------------

class Particle:
    def __init__(self,bounds):
        self.particle_position=[]                     # particle position
        self.particle_velocity=[]                     # particle velocity
        self.local_best_particle_position=[]          # best position of the particle
        self.fitness_local_best_particle_position= initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position=initial_fitness             # objective function value of the particle position
 
        for i in range(nv):
            self.particle_position.append(random.uniform(bounds[i][0],bounds[i][1])) # generate random initial position
            self.particle_velocity.append(random.uniform(-1,1)) # generate random initial velocity
 
    def evaluate(self,objective_function):
        self.fitness_particle_position=model(self.particle_position,x_train,y_train,x_test,y_test)
        if mm == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position=self.particle_position                  # update the local best
                self.fitness_local_best_particle_position=self.fitness_particle_position  # update the fitness of the local best
        if mm == 1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position=self.particle_position                  # update the local best
                self.fitness_local_best_particle_position=self.fitness_particle_position  # update the fitness of the local best
 
    def update_velocity(self,global_best_particle_position):
        for i in range(nv):
            r1=random.random()
            r2=random.random()
 
            cognitive_velocity = c1*r1*(self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2*r2*(global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w*self.particle_velocity[i]+ cognitive_velocity + social_velocity
 
    def update_position(self,bounds):
        for i in range(nv):
            self.particle_position[i]=self.particle_position[i]+self.particle_velocity[i]
 
            # check and repair to satisfy the upper bounds
            if self.particle_position[i]>bounds[i][1]:
                self.particle_position[i]=bounds[i][1]
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i]=bounds[i][0]

class PSO():
    def __init__(self,objective_function,bounds,particle_size,iterations):
 
        fitness_global_best_particle_position=initial_fitness
        global_best_particle_position=[]
 
        swarm_particle=[]
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds))
        A=[]
         
        for i in range(iterations):
            print("iterations = ",i)
            for j in range(particle_size):
                print("particle = ",j)
                print("iterations = ",i)
                swarm_particle[j].evaluate(objective_function)
 
                if mm ==-1:
                    if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
                if mm ==1:
                    if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position(bounds)
                 
            A.append(fitness_global_best_particle_position) # record the best fitness
             
             
        print('Optimal solution:', global_best_particle_position)
        print('Objective function value:', fitness_global_best_particle_position)
        print('Evolutionary process of the objective function value:')
        plt.plot(A)
        plt.savefig('foo.png')
#------------------------------------------------------------------------------
if mm == -1:
    initial_fitness = float("inf") # for minimization problem
if mm == 1:
    initial_fitness = -float("inf") # for maximization problem
#------------------------------------------------------------------------------   
# Main PSO         
# PSO(objective_function,bounds,particle_size,iterations)

PSO(model,bounds,particle_size,iterations)