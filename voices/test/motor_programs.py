import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras import backend as K

def oscillator_control_activation(x):
    u = K.relu(x[:, 0])
    k = x[:, 1]
    return K.concatenate([K.expand_dims(u), K.expand_dims(k)])

def oscillator_control_loss(y_true, y_pred):
    y_sqdiff = K.square(y_pred[:, 1:, :] - y_pred[:, :-1, :])
    return K.mean(y_sqdiff)

n_batch = 2
n_in = 4
n_unit = 5
n_out = 2
t_len = 10

# set up sample data
X_train = np.random.randn(n_batch, t_len, n_in)
Y_train = np.random.randn(n_batch, t_len, n_out)

X_test = np.random.randn(n_batch, t_len, n_in)
Y_test = np.random.randn(n_batch, t_len, n_out)

# construct the network
model = Sequential()
model.add(LSTM(n_unit, return_sequences=True, input_shape=(t_len, n_in)))
model.add(TimeDistributed(Dense(n_out, activation=oscillator_control_activation)))

# compile the network and specify loss, optimizer
model.compile(loss=oscillator_control_loss, optimizer='rmsprop')

# fit the data
model.fit(X_train, Y_train,
          batch_size=n_batch, nb_epoch=30,
          validation_data=(X_test, Y_test))

Y_pred = model.predict(X_test, batch_size=n_batch)
print 'Y_pred.shape=',Y_pred.shape
print Y_pred