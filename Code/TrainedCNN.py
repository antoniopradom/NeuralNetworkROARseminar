import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('../Data/modelGood.h5')
image_size = 128
classes = ['cat', 'dog']

def NewImage():
    root = tk.Tk()
    #
    fileN = filedialog.askopenfilename(parent=root, title='Select Shoe binary file',
                                          filetypes=(("Image files", "*.jpg"), ("all files", "*.*")))
    root.withdraw()
    if fileN is None:
        return 0
    image = cv2.imread(fileN)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    image = np.expand_dims(image, 0)
    pred = model.predict(image)
    lab = classes[np.argmax(pred)]
    plt.imshow(np.squeeze(image))
    plt.title('%s (%.2f)' % (lab, pred[0, np.argmax(pred)]))


