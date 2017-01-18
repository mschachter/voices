import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras import backend as K


class MotorProgramRNN(object):

    def __init__(self):
        pass

    def activation_op(self):

        def activation_func(x):
            u = K.relu(x[:, 0])
            k = x[:, 1]
            return K.concatenate([K.expand_dims(u), K.expand_dims(k)])

        return activation_func

    def loss_op(self, alpha=1.0):

        def loss_func(y_true, y_pred):
            y_sqdiff = K.square(y_pred[:, 1:, :] - y_pred[:, :-1, :])
            return alpha * K.mean(y_sqdiff)

        return loss_func

    def construct(self, num_units, num_time_points, num_inputs, num_outputs, alpha=1.0, optimizer='rmsprop'):

        # construct the network
        model = Sequential()
        model.add(LSTM(num_units, return_sequences=True, input_shape=(num_time_points, num_inputs)))
        model.add(TimeDistributed(Dense(num_outputs, activation=self.activation_op())))

        # compile the network and specify loss, optimizer
        model.compile(loss=self.loss_op(alpha), optimizer=optimizer)

        self.model = model

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, batch_size=10, num_epochs=30):
        # fit the data
        self.model.fit(Xtrain, Ytrain, batch_size=batch_size, nb_epoch=num_epochs, validation_data=(Xtest, Ytest))

    def predict(self, X):

        assert len(X.shape) == 3, "X must be shape (batch_size, num_time_steps, num_input_dim)"
        batch_size, num_time_steps, num_input_dim = X.shape

        Ypred = self.model.predict(X, batch_size=batch_size)

        return Ypred


n_batch = 25
n_in = 4
n_unit = 5
n_out = 2
t_len = 50

# set up sample data
X_train = np.random.randn(n_batch, t_len, n_in)
Y_train = np.random.randn(n_batch, t_len, n_out)

X_test = np.random.randn(n_batch, t_len, n_in)
Y_test = np.random.randn(n_batch, t_len, n_out)

net = MotorProgramRNN()
net.construct(n_unit, t_len, n_in, n_out, alpha=1.0)
net.fit(X_train, Y_train, X_test, Y_test, batch_size=n_batch, num_epochs=20)

Y_pred = net.predict(X_test)
print 'Y_pred.shape=',Y_pred.shape

# plt.figure()
# ax = plt.subplot(2, 1, 1)
# plt.plot(Y_pred)

