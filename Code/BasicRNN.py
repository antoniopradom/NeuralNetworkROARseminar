import numpy as np
import tensorflow as tf
import pandas
import matplotlib.pyplot as plt

# here we will create a fake dataset to use for testing
freq = 50
t = np.linspace(0, 100, 100 * freq) # 100s of recording at 10 hz
A = 1  # this is an amplitude

yFn = lambda xx: A * (np.array([np.sin(r * 2 * np.pi * t) for r in [0.16, 0.21, 0.24, 0.49]]).sum(axis=0))
# the output is a single number
f = yFn(t)

windowSize = 10

predFuture = 5
x = np.zeros((t.size-predFuture - windowSize, windowSize))
y = np.zeros(x.shape[0])
for k in range(t.size - predFuture - windowSize):
    x[k] = f[k:k + windowSize]
    y[k] = f[k+predFuture]

x = np.expand_dims(x, 2)

x_test = x[np.round(x.shape[0] * 0.8).astype(np.int):]
y_test = y[np.round(x.shape[0] * 0.8).astype(np.int):]

y = y[:np.round(x.shape[0] * 0.8).astype(np.int)]
x = x[:np.round(x.shape[0] * 0.8).astype(np.int)]
t_test = np.arange(x_test.shape[0]) / freq + t[-1] * 0.8 - (windowSize-predFuture) / freq


# make the network
num_gru_units = [5]
rnn_model = tf.keras.Sequential()
# RNN
rnn_model.add(tf.keras.layers.RNN([tf.keras.layers.GRUCell(n) for n in num_gru_units],
                              return_sequences=True, input_shape=x.shape[1:]))
# Flatten
rnn_model.add(tf.keras.layers.Flatten())
# dense layers
fullyCon = [16]
for k in fullyCon:
    rnn_model.add(tf.keras.layers.Dense(k, activation=tf.nn.relu))

# the last layer reshapes the tensor to the required output size
rnn_model.add(tf.keras.layers.Dense(1))
# we use the built-in optimizer RMSprop. Using the tf.keras.optimizers class, we can easily change optimizers
optimizer = tf.keras.optimizers.RMSprop(0.001)
# in here we declare the training parameters of the network
rnn_model.compile(loss='mse',  # mean square error
              optimizer=optimizer,
              metrics=['mae', 'mse'])
# we fit the rnn_model, we use a batch of 10 and 50 epochs. validation split means that from the training dataset 80% is
# used for training and 20% for validation
history = rnn_model.fit(x, y, batch_size=200, epochs=50, validation_split=0.2)
# in here we can see the training history and plot it
hist = pandas.DataFrame(history.history)
# we predict the y using keras rnn_model
y_pred = np.squeeze(rnn_model.predict(x_test))

plt.plot(t, f, label='Truth')
plt.plot(t_test, y_pred, label='RNN')
plt.legend()
