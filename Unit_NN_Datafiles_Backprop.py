"""
Neural Network for Soil data with Backpropagation

Cite: http://blog.aloni.org/posts/backprop-with-tensorflow/

Learning Algorithm: Neural Network
Supervised Learning: Regression

Algorithm content:
activation function: sigmoid function
hypothesis function: polynomial function
Cost function: square error
Solving partial derivatives of Cost function: Backpropagation
Minimise cost function: Gradient descent

Data set divided into training, cross-validation, test set
"""

""" First test: only using all 28 Karlsruhe Data 
training data: 18
cross-validation data: 5
test data: 5
"""

import tensorflow as tf
import Data_Set_Generator as soil
import Error_Plots as Plots
#import numpy as np


""" Number of training data 
Splitting:      training data     - 60%
                cross-validation  - 20%
                test data         - 20%
                """
m_training = int(soil.training)
m_cv = int(soil.cv)
m_test = int(soil.test)
total_examples = soil.total

" range for running "
range_training = range(0,m_training)
range_cv = range(m_training,(m_training+m_cv))
range_test = range((m_training+m_cv),total_examples)

""" Using interactive Sessions"""
sess = tf.InteractiveSession()

""" creating input and output vectors """
x = tf.placeholder(tf.float32, shape=[None, soil.input_number])
y_true = tf.placeholder(tf.float32, shape=[None, soil.output_number])

""" Weights and Biases """

def weights(shape):
    """ shape = [row,column]"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    """ shape = [row,column]"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

""" Creating weights and biases for all layers """
theta1 = weights([soil.weight_1_1,soil.weight_1_2])
bias1 = bias([1,soil.bias_1])

theta2 = weights([soil.weight_2_1,soil.weight_2_2])
bias2 = bias([1,soil.bias_2])

"Last layer"
theta3 = weights([soil.weight_3_1,soil.weight_3_2])
bias3 = bias([1,soil.bias_3])


""" Hidden layer input (Sum of weights, activation functions and bias)
z = theta^T * activation + bias
 """
def Z_Layer(activation,theta,bias):
    return tf.add(tf.matmul(activation,theta),bias)

""" Creating the sigmoid function 
sigmoid = 1 / (1 + exp(-z))
"""
def Sigmoid(z):
    return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0), tf.exp(tf.neg(z))))

def Diff_Sigmoid(z):
    return tf.truediv(tf.exp(z), tf.pow( tf.add(tf.exp(z),1) , 2) )

""" hypothesis functions - predicted output """    
' layer 1 - input layer '
hyp1 = x
' layer 2 '
z2 = Z_Layer(x, theta1, bias1)
hyp2 = Sigmoid(z2)
' layer 3 '
z3 = Z_Layer(hyp2, theta2, bias2)
hyp3 = Sigmoid(z3)
' layer 4 - output layer '
zL = Z_Layer(hyp3, theta3, bias3)
#hypL = zL
hypL = tf.add( tf.add(tf.pow(zL,4) , tf.add(tf.pow(zL,3), tf.pow(zL,2) ) ), zL)
#hypL = tf.add(tf.add(tf.pow(zL,3), tf.pow(zL,2) ) , zL) + 1.0 / ( 1.0 + tf.exp(-zL))


""" Cost function """
" Cost function can't be the logarithmic version because there are negative stress increments "
cost_function1 = tf.mul( 0.5, tf.pow( tf.sub(hypL[:,0], y_true[:,0]), 2))
cost_function2 = tf.mul( 0.5, tf.pow( tf.sub(hypL[:,1], y_true[:,1]), 2))
cost_function = tf.add(cost_function1, cost_function2)


'-----------------------------------------------------------'
""" Backpropagation """

' error terms delta '
""" name = delta'layer_number'_'feature_number' """
deltaL = tf.mul(tf.sub(hypL,y_true) , tf.add(3*tf.pow(zL, 2) + 2*zL , 1.0))
delta3 = tf.mul(tf.matmul(deltaL, tf.transpose(theta3)) , Diff_Sigmoid(z3) )
delta2 = tf.mul(tf.matmul(delta3, tf.transpose(theta2)) , Diff_Sigmoid(z2) )

' partial derivatives '
part_theta3 = tf.matmul(tf.transpose(hyp3) , deltaL)
part_theta2 = tf.matmul(tf.transpose(hyp2) , delta3)
part_theta1 = tf.matmul(tf.transpose(hyp1) , delta2)

part_bias3 = deltaL
part_bias2 = delta3
part_bias1 = delta2

' Update weights and biases '
alpha = tf.constant(0.003)
step = [
        tf.assign(theta3, tf.sub(theta3, tf.mul(alpha, part_theta3))),
        tf.assign(theta2, tf.sub(theta2, tf.mul(alpha, part_theta2))),
        tf.assign(theta1, tf.sub(theta1, tf.mul(alpha, part_theta1))),
        tf.assign(bias3, tf.sub(bias3, tf.mul(alpha, tf.reduce_mean(part_bias3, reduction_indices=[0])))),
        tf.assign(bias2, tf.sub(bias2, tf.mul(alpha, tf.reduce_mean(part_bias2, reduction_indices=[0])))),
        tf.assign(bias1, tf.sub(bias1, tf.mul(alpha, tf.reduce_mean(part_bias1, reduction_indices=[0]))))        
        ]


'-----------------------------------------------------------'


""" Gradient Descent """
#train_step = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost_function)       
        

"""    Training and Evaluation     """

correct_prediction = tf.equal(tf.arg_max(hypL, 1), tf.arg_max(y_true, 1))

#correct_prediction = 1.0 - cost_function

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

keep_prob = tf.placeholder(tf.float32)


training_error=[]
cost_function_training = []
for j in range_training:
    feedj = {x: soil.input_matrixs[j], y_true: soil.output_matrixs[j] , keep_prob: 1.0}
    sess.run(step, feed_dict=feedj)
    train_accuracy = accuracy.eval(feed_dict=feedj)
    cost_function_train = sess.run(tf.reduce_mean(cost_function), feed_dict=feedj)
    print("step %d, training accuracy %g" % (j, train_accuracy))
    print("Cost function: %f" % cost_function_train)
    #train_step.run(feed_dict=feedj)
    training_error.append(1 - train_accuracy)
    cost_function_training.append(cost_function_train)

cv_error=[]  
#cost_function_validation =[]  
for k in range_cv:
    feedk = {x: soil.input_matrixs[k], y_true: soil.output_matrixs[k] , keep_prob: 1.0}
    cv_accuracy = accuracy.eval(feed_dict=feedk)
    #cost_function_cv = sess.run(tf.reduce_mean(cost_function), feed_dict=feedk)
    print("cross-validation accuracy %g" % cv_accuracy)
    #print("Cost function: %f" % cost_function_cv)
    cv_error.append(1-cv_accuracy) 
    #cost_function_validation.append(cost_function_cv)

for l in range_test:
    print("test accuracy %g" % accuracy.eval(feed_dict={x: soil.input_matrixs[l], y_true: soil.output_matrixs[l], keep_prob: 1.0}))


""" Error Analysis """

" Learning Curves "

figure1 = Plots.LearningCurvesT(cv_error, training_error,'error plot')
#figure2 = Plots.LearningCurvesT(cost_function_validation, cost_function_training, 'cost functions')


