# -*- coding: utf-8 -*-
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

X=np.array([1,2,3])
Y=np.array([1,2,3])

def cost_func(W,X,Y):
    hypothesis=X*W
    return tf.reduce_mean(tf.square(hypothesis-Y))
W_values=np.linspace(-3,5,num=15)
cost_values=[]

for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

'''
-3.000 |   74.66667
-2.429 |   54.85714
-1.857 |   38.09524
-1.286 |   24.38095
-0.714 |   13.71429
-0.143 |    6.09524
 0.429 |    1.52381
 1.000 |    0.00000
 1.571 |    1.52381
 2.143 |    6.09524
 2.714 |   13.71429
 3.286 |   24.38095
 3.857 |   38.09524
 4.429 |   54.85714
 5.000 |   74.66667
 '''