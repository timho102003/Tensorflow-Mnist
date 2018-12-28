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

# Parameters
lr = 0.1 #learning rate
steps = 500
batch_size = 36
display_step = 100

# Network Parameters Settings
n_hidden_1 = 256 # 1st fully-connected layer
n_hidden_2 = 256 # 2st fully-connected layer
num_imput = 784 # 28 x 28 (image pixels)
num_classes = 10 # classes to be classify

# tensorflow graph
X = tf.placeholder('float', [None, num_imput])
Y = tf.placeholder('float', [None, num_classes])
# weight and bias
Weights = {
    'w1': tf.Variable(tf.random_normal([num_imput, n_hidden_1]), name='HIDDEN_W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='HIDDEN_W2'),
    'out': tf.Variable(tf.random_normal([n_hidden_2,num_classes]), name='Output_W')
}
Bias = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='HIDDEN_B1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_1]), name='HIDDEN_B2'),
    'out': tf.Variable(tf.random_normal([n_hidden_1]), name='Output_B')
}
