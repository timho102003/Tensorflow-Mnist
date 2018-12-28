import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# load MNIST data from tensorflow
# if using MAC and python3.6, copy this command to terminal-->/Applications/Python\ 3.6/Install\ Certificates.command
# to prevent the urlerror while using either keras.datasets or tensorflow.examples.tutorials.mnist
mnist=tf.keras.datasets.mnist
(Train_Image, Train_Label), (Test_Image, Test_Label) = mnist.load_data()
#show image
plt.imshow(Train_Image[0], cmap='binary', interpolation=None)
plt.show()

