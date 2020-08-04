import numpy as np
import tensorflow as tf
import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# here we will create a fake dataset to use for testing
t = 1000  # we have 1000 examples
dim = 2
x = np.random.rand(t, dim)  # we have 4 features per sample
x_test = x[800:]
# yFn = lambda xx: xx.dot(np.arange(dim)+1)  # y = sum(n*x)) --> this is a linear function
yFn = lambda xx: np.sin(np.pi * xx).dot(np.arange(dim)+1)  # y = sum(n*sin(pi*x)) --> non-linear function
# the output is a single number
y = yFn(x)  # get the training ground truth
y_test = yFn(x_test)  # get the testing ground truth

# we build the model as a Sequential model. For the first layer we need to define the input size
model = tf.keras.Sequential([tf.keras.layers.Dense(8, activation='relu', input_shape=(dim,))])
# each element of this list is a layer, the value is the number units
dense_layers_units = [16]
# if we put the constructor in a for loop, we can easily increase the number of layers
for n in dense_layers_units:
    model.add(tf.keras.layers.Dense(n, activation='relu'))
# the last layer reshapes the tensor to the required output size
model.add(tf.keras.layers.Dense(1))
# we use the built-in optimizer RMSprop. Using the tf.keras.optimizers class, we can easily change optimizers
optimizer = tf.keras.optimizers.RMSprop(0.01)
# in here we declare the training parameters of the network
model.compile(loss='mse',  # mean square error
              optimizer=optimizer,
              metrics=['mae', 'mse'])
# we print the layer size and summary
model.summary()
# we fit the model, we use a batch of 10 and 50 epochs. validation split means that from the training dataset 80% is
# used for training and 20% for validation
history = model.fit(x, y, batch_size=10, epochs=50, validation_split=0.2)
# in here we can see the training history and plot it
hist = pandas.DataFrame(history.history)
# we predict the y using keras model
y_pred = np.squeeze(model.predict(x_test))
# we do the same using scikit-learn linear regression
model_sci = linear_model.LinearRegression()
model_sci.fit(x, y)
y_pred_sci = model_sci.predict(x_test)

print('Mean squared error tf: %.3f \nMean squared error sk: %.3f' % (mean_squared_error(y_test, y_pred),
                                                                      mean_squared_error(y_test, y_pred_sci)))

plt.plot(y_test, y_test, color='r', label='45 line')
plt.scatter(y_test, y_pred, label='Tensorflow')
plt.scatter(y_test, y_pred_sci, label='Linear Regressor')
plt.legend()

errorPd = pandas.DataFrame({'tf': y_pred - y_test, 'sklearn': y_pred_sci - y_test})
errorPd.describe()
