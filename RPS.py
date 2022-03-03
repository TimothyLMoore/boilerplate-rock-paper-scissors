# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random
from keras.models import Sequential, save_model, load_model
import pandas as pd
import numpy as np
import pickle

def player(prev_play, opponent_history=["R","P","S"]):
    with open("./model/le.obj", 'rb') as f:
        le = pickle.load(f)

    filepath = "./model"
    model = load_model(filepath, compile = True)
    if not(prev_play == ""):
        opponent_history.append(prev_play)
    df = pd.DataFrame(opponent_history, columns = ['throws'])
    df = pd.get_dummies(df['throws'])
    if len(df) < 10:
        df_head = pd.DataFrame(np.zeros((10 - len(df), 3)), columns = ["P", "R", "S"])
        df = pd.concat([df_head,df])
    df = df[-10:]
    np_df = df.to_numpy()

    X = np_df.flatten()
    #print(np.shape(X))
    X = np.reshape(X,(1,30,1))
    #print(X)
    #print(np.shape(X))

    pred = model.predict(X)


    max_index = [np.argmax(pred)]

    pred = str(le.inverse_transform(max_index))[2]

    return pred #random.choice(['R',"P","S"])
