from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
#from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(10, activation='relu')(x)
    x = Dense(5, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model

memory = SequentialMemory(limit=50000, window_length=1)

if __name__ == '__main__':
    model = build_model(10, 3)
    print(model.summary())
