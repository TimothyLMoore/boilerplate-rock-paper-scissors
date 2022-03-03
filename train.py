from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player

import itertools
import numpy as np
import pandas as pd

import pickle

from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
#from keras.optimizers import Adam



def build_model():

    model = Sequential()
    model.add(Dense(60, input_dim = 30, activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dense(40, activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

if __name__ == '__main__':
    filepath = "./model"
    model = load_model(filepath, compile = True)
    played = []
    for i in range(1,15):
        #_, j = play(player, quincy, 100)
        #played.append(j)
        _, j = play(player, abbey, 100)
        played.append(j)
        _, j = play(player, kris, 100)
        played.append(j)
        #_, j = play(player, mrugesh, 100)
        #played.append(j)
        print(i)
    played = list(itertools.chain(*played))
    winVs = {"S":"R", "R":"P", "P":"S"}
    y = [winVs[i] for i in played]
    df = pd.DataFrame(played, columns = ['throws'])
    df = pd.get_dummies(df['throws'])
    df_head = pd.DataFrame(np.zeros((10, 3)), columns = ["P", "R", "S"])
    df = pd.concat([df_head,df])
    y = pd.DataFrame(y)
    X = np.empty(0)
    np_df = df.to_numpy()
    for i in range(0,len(df)-10):
        X = np.concatenate([X,np_df[i:i+10].flatten()])
        if i % 1000 == 0:
            print(i)
    X = X.reshape(len(X)//30,30)


    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    y = tf.keras.utils.to_categorical(y, num_classes=3, dtype="int")


    with open("./model/le.obj", 'wb') as f:
        pickle.dump(le, f)

    print(np.shape(X),np.shape(y))

    model.fit(X,y, epochs = 100, batch_size=100)

    filepath = "./model"
    save_model(model, filepath)







