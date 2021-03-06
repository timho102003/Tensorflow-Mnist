import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# load MNIST data from tensorflow
# if using MAC and python3.6, copy this command to terminal-->/Applications/Python\ 3.6/Install\ Certificates.command
# to prevent the urlerror while using either keras.datasets or tensorflow.examples.tutorials.mnist
# mnist=tf.keras.datasets.mnist
# (Train_Image, Train_Label), (Test_Image, Test_Label) = mnist.load_data()
#show image
# plt.imshow(Train_Image[0], cmap='binary', interpolation=None)
# plt.show()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Parameters
import math
import os
model_path = './Model/'
if not os.path.isdir(model_path):
    os.makedirs(model_path)
    print('Create Model Folder: {}'.format(model_path))
lr = 0.1 #learning rate
batch_size = 36
steps = int(math.ceil(mnist.train.images.shape[0] / batch_size))
epochs = 10
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
Biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='HIDDEN_B1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='HIDDEN_B2'),
    'out': tf.Variable(tf.random_normal([num_classes]), name='Output_B')
}

def network(x):
    '''
    :param x: input images, shape --> batch_size x num_input
    :return: prediction, shape --> batch_size x num_classes
    '''
    layer_1 = tf.add(tf.matmul(x, Weights['w1']), Biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, Weights['w2']), Biases['b2'])
    out_layer = tf.add(tf.matmul(layer_2, Weights['out']), Biases['out'])

    return out_layer

# Model
logits = network(X)
prediction = tf.nn.softmax(logits) # Normalization

# Define loss function and optimization
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

# Calculate Performance
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# save model
saver = tf.train.Saver()

init = tf.global_variables_initializer()
# run training
with tf.Session() as sess:
    # Run Initializer to initialize all variables
    sess.run(init)
    # step training
    for epoch in range(1, epochs+1):
        for step in range(1, steps+1):
            batch_images, batch_labels = mnist.train.next_batch(batch_size=batch_size, shuffle=True)
            sess.run(train_op, feed_dict={X: batch_images, Y: batch_labels})
            # Display
            if step % display_step == 0 or step ==1:
                L, ACC = sess.run([loss, accuracy], feed_dict={X: batch_images, Y: batch_labels})
                print('Epoch: {},\tStep: {},\tLOSS: {:.3f},\tTRAIN ACCURACY: {:.3f}'.format(epoch, step, L, ACC))

        save_path = saver.save(sess, model_path + '/model_{}.ckpt'.format(epoch))
        print("Model saved in file: %s" % save_path)

        # Calculate accuracy for MNIST validation images
        print("Validation Accuracy: {:.3f}".format(sess.run(accuracy, feed_dict={X: mnist.validation.images,
                                                                              Y: mnist.validation.labels})))
    print('Finish Optimization...')

    # Calculate accuracy for MNIST test images
    print('Start Testing...')
    print("Test Accuracy: {:.3f}".format(sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                                              Y: mnist.test.labels})))


