# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random
from keras.models import Sequential, save_model, load_model
import pandas as pd
import numpy as np
import pickle
from time import time

with open("./model/le.obj", 'rb') as f:
    le = pickle.load(f)

filepath = "./model"
model = load_model(filepath, compile = True)


def player(prev_play, opponent_history=["R","P","S"], my_history=["R","P","S"],play_order=[{
              "RR": 1,
              "RP": 1,
              "RS": 1,
              "PR": 1,
              "PP": 1,
              "PS": 1,
              "SR": 1,
              "SP": 1,
              "SS": 1,}]):

    play_order_values = list(play_order[0].values())
    if not(prev_play == ""):
        opponent_history.append(prev_play)
    opp_df = pd.DataFrame(opponent_history, columns = ['throws'])
    my_df = pd.DataFrame(my_history, columns = ['throws'])
    opp_df = pd.get_dummies(opp_df['throws'])
    my_df = pd.get_dummies(my_df['throws'])



    if len(opp_df) < 10:
        df_head = pd.DataFrame(np.zeros((10 - len(opp_df), 3)), columns = ["P", "R", "S"])
        opp_df = pd.concat([df_head,opp_df])
    if len(my_df) < 10:
        df_head = pd.DataFrame(np.zeros((10 - len(my_df), 3)), columns = ["P", "R", "S"])
        my_df = pd.concat([df_head,my_df])


    df =  pd.concat([my_df[-10:],opp_df[-10:]])

    np_df = df.to_numpy()



    X = np_df.flatten()
    po_val = np.array(play_order_values)
    po_val = po_val/po_val.sum()
    #print(X)
    #print(po_val)
    X = np.concatenate((X,po_val))
    #print(np.shape(X))
    X = np.reshape(X,(1,69,1))
    #print(X)
    #print(np.shape(X))

    #print(X)
    pred = model.predict(X)
    max_index = [np.argmax(pred)]
    pred = str(le.inverse_transform(max_index))[2]
    my_history.append(pred)
    play_order[0][("".join(my_history[-2:]))] += 1
    #print(play_order)
    return pred #random.choice(['R',"P","S"])
