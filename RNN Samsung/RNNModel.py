from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, GRU

def simpleRNN():
    model = Sequential()
    model.add(SimpleRNN(units=4, input_length=5, input_dim=1, activation="tanh"))
    model.add(Dense(1))
    model.summary()

    return model

def deepRNN():
    model = Sequential()
    model.add(SimpleRNN(units=4, input_length=5, input_dim=1, activation="tanh", return_sequences=True))
    model.add(SimpleRNN(units=4, activation="tanh"))
    model.add(Dense(1))

    return model
