from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player

import itertools
import numpy as np
import pandas as pd

import pickle

from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
#from keras.optimizers import Adam



def build_model():

    model = Sequential()
    model.add(Dense(120, input_dim = 69, activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dropout(0.5))
    model.add(Dense(80, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40, activation = 'sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

if __name__ == '__main__':
    filepath = "./model"
    model = load_model(filepath, compile = True)
    #model = build_model()
    opp_played = []
    my_played = ["R"]
    play_order=[{
              "RR": 1,
              "RP": 1,
              "RS": 1,
              "PR": 1,
              "PP": 1,
              "PS": 1,
              "SR": 1,
              "SP": 1,
              "SS": 1,
          }]
    for i in range(1,2):
        _, j, k = play(player, abbey, 5000)
        opp_played.append(j)
        my_played.append(k)
        _, j, k = play(player, quincy, 5000)
        opp_played.append(j)
        my_played.append(k)
        _, j, k = play(player, kris, 5000)
        opp_played.append(j)
        my_played.append(k)
        _, j, k = play(player, mrugesh, 5000)
        opp_played.append(j)
        my_played.append(k)
        print(i)
    opp_played = list(itertools.chain(*opp_played))
    my_played = list(itertools.chain(*my_played))
    winVs = {"S":"R", "R":"P", "P":"S"}
    y = [winVs[i] for i in opp_played]
    opp_df = pd.DataFrame(opp_played, columns = ['throws'])
    opp_df = pd.get_dummies(opp_df['throws'])
    my_df = pd.DataFrame(my_played, columns = ['throws'])
    my_df = pd.get_dummies(my_df['throws'])
    df_head = pd.DataFrame(np.zeros((10, 3)), columns = ["P", "R", "S"])
    opp_df = pd.concat([df_head,opp_df])
    my_df = pd.concat([df_head,my_df])
    y = pd.DataFrame(y)
    X = np.empty(0)
    opp_np_df = opp_df.to_numpy()
    my_np_df = my_df.to_numpy()
    for i in range(0,len(opp_df)-10):
        play_order_values = list(play_order[0].values())
        po_val = np.array(play_order_values)
        po_val = po_val/po_val.sum()
        X = np.concatenate([X,my_np_df[i:i+10].flatten(),opp_np_df[i:i+10].flatten()])
        X = np.concatenate((X,po_val))
        play_order[0][("".join([my_played[i-1],my_played[i]]))] += 1
        #print(play_order[0],my_played[i])
        if i % 200 == 0:
            print(i)

    X = X.reshape(len(X)//69,69)

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    y = tf.keras.utils.to_categorical(y, num_classes=3, dtype="float")
    y2 = le.transform(pd.DataFrame(opp_played))
    y2 = tf.keras.utils.to_categorical(y2, num_classes=3, dtype="float")
    y = y * 0.6666
    y2 = y2 * 0.3334
    y = np.add(y,y2)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)


    with open("./model/le.obj", 'wb') as f:
        pickle.dump(le, f)

    print(np.shape(X),np.shape(y))

    model.fit(X_train,y_train,validation_data = [X_test, y_test], epochs = 256, batch_size=30)

    filepath = "./model"
    save_model(model, filepath)







