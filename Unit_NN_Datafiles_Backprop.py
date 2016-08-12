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
x = tf.placeholder(tf.float32, shape=[None, 1])
y_true = tf.placeholder(tf.float32, shape=[None, 2])

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
theta1 = weights([1,4])
bias1 = bias([1,4])

theta2 = weights([4,4])
bias2 = bias([1,4])

"Last layer"
theta3 = weights([4,2])
bias3 = bias([1,2])


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
hypL = tf.add( tf.add(tf.pow(zL,3), tf.pow(zL,2) ), zL)


""" Cost function """
" Cost function can't be the logarithmic version because there are negative stress increments "
cost_function = tf.mul( tf.div(0.5, m_training), tf.pow( tf.sub(hypL, y_true), 2)) 


'-----------------------------------------------------------'
""" Backpropagation """
error1 = cost_function[:,1]
error2 = cost_function[:,2]
error3 = cost_function[:,3]

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

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

keep_prob = tf.placeholder(tf.float32)


training_error=[]
for j in range_training:
    train_accuracy = accuracy.eval(feed_dict={x: soil.input_matrixs[j], y_true: soil.output_matrixs[j]})
    print("step %d, training accuracy %g" % (j, train_accuracy))
    sess.run(step, feed_dict={x: soil.input_matrixs[j], y_true: soil.output_matrixs[j]})
    training_error.append(1 - train_accuracy)

cv_error=[]    
for k in range_cv:
    cv_accuracy = accuracy.eval(feed_dict={x: soil.input_matrixs[k], y_true: soil.output_matrixs[k], keep_prob: 1.0})
    print("cross-validation accuracy %g" % cv_accuracy)
    cv_error.append(1-cv_accuracy) 

for l in range_test:
    print("test accuracy %g" % accuracy.eval(feed_dict={x: soil.input_matrixs[k], y_true: soil.output_matrixs[k], keep_prob: 1.0}))


""" Error Analysis """

" Learning Curves "

figure1 = Plots.LearningCurves(cv_error, training_error)
 



