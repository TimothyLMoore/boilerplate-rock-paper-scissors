# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random
from keras.models import Sequential, save_model, load_model
import pandas as pd
import numpy as np
import pickle
from time import time

def player(prev_opponent_play, opponent_history=[], play_order=[{}]):

    chain_length = 4

    if not prev_opponent_play:
        prev_opponent_play = 'R'
    opponent_history.append(prev_opponent_play)

    last_three = "".join(opponent_history[-(chain_length):])
    if len(last_three) == chain_length:
        if last_three in play_order[0]:
            play_order[0][last_three] += 1
        else:
            play_order[0][last_three] = 1

    potential_plays = [
        "".join(opponent_history[-(chain_length-1):]) + "R",
        "".join(opponent_history[-(chain_length-1):]) + "P",
        "".join(opponent_history[-(chain_length-1):]) + "S",
    ]

    sub_order = {
        k: play_order[0][k]
        for k in potential_plays if k in play_order[0]
    }

    if len(last_three) == chain_length and sub_order != {}:
        prediction = max(sub_order, key=sub_order.get)[-1:]
    else:
        prediction = random.choice(['R', 'P', 'S'])

    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    return ideal_response[prediction]
