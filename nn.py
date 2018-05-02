import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# number of neurons in each layer
input_num_units = 12
hidden_num_units = 3
output_num_units = 12

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 30
learning_rate = 0.01
seed = 10

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.tanh(hidden_layer)
output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
cost = tf.reduce_mean(tf.square(output_layer-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()
#batch_x = np.matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
batch_x = np.matrix([[0,1,2,3,4,5,6,7,8,9,10,11]])
#batch_y = np.matrix([[1 ,1.5 ,4 ,3.3] ,[6 ,5.9 ,7.5 ,6] ,[7 ,6.8 ,8.3 ,9 ]])
batch_y = np.matrix([[1,1.5,4,3.3,6,5.9,7.5,6,7,6.8,8.3,9.9]])
with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    for n in range(epochs):
        avg_cost = 0
        for a in range(200):
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            avg_cost+= c
        print ("Epoch:", (n+1), "cost =", (avg_cost)/200)
        if(n == epochs-1):
            print ("batch_y",batch_y.flatten())
            an = np.transpose(sess.run(output_layer,feed_dict={x:batch_x}))
            an = an.flatten()
            plt.plot([0,1,2,3,4,5,6,7,8,9,10,11],an,'ro')
            plt.plot([1,1.5,4,3.3,6,5.9,7.5,6,7,6.8,8.3,9.9,9.5])
            plt.show()
            print("Answer",an)
            print("Expected",batch_y.flatten())
            n = tf.reduce_sum(tf.square(batch_y.flatten()-an))
            print ("cost1",sess.run(n))
            print ("cost2",c)
    print ("\nTraining complete!")


    #plt.plot([0,1,2,3,4,5,6,7,8,9,10,11],an,'ro')
    #plt.plot([1,1.5,4,3.3,6,5.9,7.5,6,7,6.8,8.3,9.9,9.5])
    #plt.axis([0,10,0,10])
    #plt.show()

    # find predictions on val set
    #pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    #print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)})

    #predict = tf.argmax(output_layer, 1)
    #pred = predict.eval({x: test_x.reshape(-1, input_num_units)})
