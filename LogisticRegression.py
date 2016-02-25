__author__ = 'Dimitris'

#sudo THEANO_FLAGS='floatX=float32,device=cpu' python2.7

import numpy as np
import theano
import theano.tensor as T
rng = np.random

# training sample size
N = 400
# number of input variables randomly selected
feats = 784
# number of training steps
training_steps = 10

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

# append 1 as last feature of each instance instead of using bias
ones = np.array([[1]*D[0].shape[0]])
temp = np.append(D[0], ones.T, 1)
D = (temp,D[1])


# x is the features' matrix NxF where N is the num of
# instances and F the number of features
x = T.matrix("x")
# y is the labels' N dimentional vector
y = T.vector("y")
# w is the weigths matrix
w = theano.shared(rng.randn(feats+1), name="w")

# logistic func
p_1 = 1 / (1 + T.exp(-T.dot(x, w)))

prediction = p_1 > 0.5

# maximum entropy
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
cost = xent.mean() + 0.01 * (w ** 2).sum()

# gradients of the cost function in response to w
gw = T.grad(cost, w)

train = theano.function( inputs=[x,y], outputs=[prediction, xent], updates=([(w, w - 0.1 * gw)]), allow_input_downcast=True)

predict = theano.function(inputs=[x], outputs=prediction, allow_input_downcast=True)

# run for training steps
for i in range(training_steps):
	print(i)
	pred, err = train(D[0], D[1])



