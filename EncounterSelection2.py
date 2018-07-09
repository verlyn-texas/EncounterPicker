#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:11:31 2018

@author: verlynfischer
"""

from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, InputLayer
from keras.models import Sequential, load_model
import pandas as pd

####################################
#
#  Training Set
#
####################################
   
def loadData2():
    examples = np.genfromtxt('examples2.csv',delimiter=",")
    np.random.shuffle(examples)
    x_train = examples[:300,0:5]
    y_train = examples[:300,5]
    x_test = examples[301:401,0:5]
    y_test = examples[301:401,5]
    return x_train, y_train, x_test, y_test


####################################
#  
#  Model Generation 
#
####################################

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def configModel():    
    
    state_parameters = 5
    tags = 16
    batch_size = 5
    epochs = 25
    
    # load the training and test data
    #x_train, y_train, x_test, y_test = loadData(state_parameters,tags)
    x_train, y_train, x_test, y_test = loadData2()
    
    #print(y_train)
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, tags)
    y_test = keras.utils.to_categorical(y_test, tags)
    
    #print(y_train)
    
    # define dense model
    model = Sequential() #take as input
    model.add(InputLayer(batch_input_shape=(batch_size,state_parameters)))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(tags, activation='softmax'))
            
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    print('Model compiled.')

    history = AccuracyHistory()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])
    print('Fit complete.')
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(range(1, epochs + 1), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    model.save_weights('WIFE_Weights')
    model.save('EntireWIFEModel')
    print('Model saved.')


####################################
#  
#  Model Loading
#
####################################

def loadModel():
    m = load_model('EntireWIFEModel')
    return m



####################################
#  
#  Helper functions
#
####################################

def getIndex(prediction):
    r = random.uniform(0,1)
    cumArray = np.cumsum(prediction)
    location = 0
    for index in range(15,-1,-1):
        if r < cumArray[index]:
            location = index
    return location
    
def loadEncounterDecoder():
    encounterList = pd.read_csv('encounterList.csv',delimiter=',',header=0)
    #print(encounterList)
    return encounterList

def getDescription(df,classCode):
        
    df = df.loc[df['Class']==classCode]
    df_elements = df.sample(n=1)
    description = str(df_elements.iloc[0]['Description'])
    return description

####################################
#  
#  Main
#
####################################

## Train Model
#configModel()
    
def provideEncounters():
    doMore = True
    while doMore == True:
        print('')
        print('----------')
        print('Answer the following questions with 1 for true and 0 for false.')
        pcLowHealth = input('Is PCs health low? ')
        ironMan = input('Is player and ironman? ')
        firstYear = input('Is PC in first year? ')
        playerFrustration = input('Is player frustrated? ')
        prevEncAction = input('Is the previous encounter action based? ')
        
        stateArray = np.array([pcLowHealth,ironMan,firstYear,playerFrustration,prevEncAction])
        stateArray = stateArray.astype(int)
        s = np.array([stateArray,stateArray,stateArray,stateArray,stateArray])
        #print(s)
        p = model.predict(s)
        #print(p[0,:])
        newA = getIndex(p[0,:])
        #print(newA)
        description = getDescription(encounterList,newA)
        print('')
        print(' - - - ')
        print('Encounter Description: ' + description)
        print('- - -')
        getContinue = input('Press Y to continue. ')
        if (getContinue != 'Y') and (getContinue != 'y'):
            doMore = False
            print('')
    
## User Interactions
model = loadModel()
encounterList = loadEncounterDecoder()

provideEncounters()

print('Program stopped.')