# this code is being addapter from downloaded from https://github.com/legolas123/cv-tricks.com, if you have questions
# check out https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/

# we are only using a subset for to do an example, download the full dataset at
# https://www.kaggle.com/c/dogs-vs-cats/

import tensorflow as tf
import Code.CNNdataset as cDB
import numpy as np
import matplotlib.pyplot as plt


classes = ['dogs', 'cats']
num_classes = len(classes)

train_path = '../Data/training_data'

# validation split
validation_size = 0.2

# batch size
batch_size = 16
img_size = 128

data = cDB.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

model = tf.keras.Sequential()
filterSize = [64, 128]
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=data.train.images.shape[1:]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
for k in filterSize:
    model.add(tf.keras.layers.Conv2D(k, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

# flatten the image
model.add(tf.keras.layers.Flatten())

# do a fully connected to mix everything up
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
# get a probability vector of the classes
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-4)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


max_epochs = 100

model.fit(x=data.train.images, y=data.train.labels[:, 0], batch_size=batch_size, epochs=max_epochs,
          validation_data=(data.valid.images, data.valid.labels[:, 0]))

# lets get 4 random images
m = np.random.randint(data.valid.images.shape[0], size=4)
x_test = data.valid.images[m]
y = [classes[xx] for xx in np.argmax(model.predict(x_test), 1)]
y_tr = [classes[xx] for xx in np.argmax(data.valid.labels[m], 1)]
f, ax = plt.subplots(2, 2)
for j in range(4):
    ax[j//2][j % 2].imshow(x_test[j])
    ax[j // 2][j % 2].set_title(y[j])
