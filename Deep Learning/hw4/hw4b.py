"""
Source Code for Homework 4.b of ECBM E4040, Fall 2016, Columbia University

"""

import os
import timeit
import inspect
import sys
import numpy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from hw4_utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir
from hw4_nn import myMLP, train_nn

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num, nbit))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)

    # Modify the gen parity pair function, but it seems it would have some problem to calculate the cost
    # if RNN:
    #     Y = numpy.zeros((num, nbit)).astype('int64')
    #     for index in range(X.shape[1]):
    #         Y[:, index] = numpy.mod(numpy.sum(X[:, :index + 1], axis=1), 2).astype('int64')
    # else:
    #     Y = numpy.mod(numpy.sum(X, axis=1), 2)
    
    return X, Y

#TODO: implement RNN class to learn parity function
class RNN(object):

    def __init__(self, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        self.input = input

        W_in_init = numpy.asarray(numpy.random.uniform(size=(n_in, n_hidden),
                                                 low=-.01, high=.01),
                               dtype=theano.config.floatX)

        self.wx = theano.shared(value=W_in_init, name='wx')

        W_init = numpy.asarray(numpy.random.uniform(size=(n_hidden, n_hidden),
                                              low=-.01, high=.01),
                            dtype=theano.config.floatX)

        self.wh = theano.shared(value=W_init, name='wh')

        W_out_init = numpy.asarray(numpy.random.uniform(size=(n_hidden, n_out),
                                                  low=-.01, high=.01),
                                dtype=theano.config.floatX)

        self.w = theano.shared(value=W_out_init, name='w')

        bh_init = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name='bh')

        by_init = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=by_init, name='b')

        h0_init = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
        self.h0 = theano.shared(value=h0_init, name='h0')


        # bundle
        self.params = [self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]

        # as many columns as context window size
        # as many lines as words in the sentence

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) +
                                 T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.dot(h_t, self.w) + self.b
            return [h_t, s_t]

        [self.h, self.s], _ = theano.scan(fn=recurrence,
                                sequences=self.input,
                                outputs_info=[self.h0, None],
                                n_steps=self.input.shape[0])

        self.p_y_given_x = T.nnet.softmax(self.s)
        self.y_out = T.argmax(self.p_y_given_x, axis=1)
        self.loss = lambda y: self.negative_log_likelihood(y)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the sequence
        over the total number of examples in the sequence ; zero one
        loss over the size of the sequence
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                            ('y', y.type, 'y_out', self.y_out.type))

            # check if y is of the correct datatype
        if y.dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
            return T.mean(T.neq(self.y_out, y))
        else:
            raise NotImplementedError()

        
    

#TODO: implement LSTM class to learn parity function
class LSTM(object):
    def __init__(self, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        n_i = n_c = n_o = n_f = n_hidden
        self.input = input

        # TODO: Use a more appropriate initialization method
        def sample_weights(sizeX, sizeY):
            values = numpy.ndarray([sizeX, sizeY], dtype=theano.config.floatX)
            for dx in xrange(sizeX):
                vals = numpy.random.uniform(low=-1., high=1., size=(sizeY,))
                # vals_norm = np.sqrt((vals**2).sum())
                # vals = vals / vals_norm
                values[dx, :] = vals
            _, svs, _ = numpy.linalg.svd(values)
            # svs[0] is the largest singular value
            values = values / svs[0]
            return values

        self.W_xi = theano.shared(sample_weights(n_in, n_i))
        self.W_hi = theano.shared(sample_weights(n_hidden, n_i))
        self.W_ci = theano.shared(sample_weights(n_c, n_i))
        self.b_i = theano.shared(numpy.cast[theano.config.floatX](numpy.random.uniform(-0.5, .5, size=n_i)))
        self.W_xf = theano.shared(sample_weights(n_in, n_f))
        self.W_hf = theano.shared(sample_weights(n_hidden, n_f))
        self.W_cf = theano.shared(sample_weights(n_c, n_f))
        self.b_f = theano.shared(numpy.cast[theano.config.floatX](numpy.random.uniform(0, 1., size=n_f)))
        self.W_xc = theano.shared(sample_weights(n_in, n_c))
        self.W_hc = theano.shared(sample_weights(n_hidden, n_c))
        self.b_c = theano.shared(numpy.zeros(n_c, dtype=theano.config.floatX))
        self.W_xo = theano.shared(sample_weights(n_in, n_o))
        self.W_ho = theano.shared(sample_weights(n_hidden, n_o))
        self.W_co = theano.shared(sample_weights(n_c, n_o))
        self.b_o = theano.shared(numpy.cast[theano.config.floatX](numpy.random.uniform(-0.5, .5, size=n_o)))
        self.W_hy = theano.shared(sample_weights(n_hidden, n_out))
        self.b_y = theano.shared(numpy.zeros(n_out, dtype=theano.config.floatX))

        self.c0 = theano.shared(numpy.zeros(n_hidden, dtype=theano.config.floatX))
        self.h0 = T.tanh(self.c0)

        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, self.W_cf, self.b_f,
                       self.W_xc, self.W_hc, self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o, self.W_hy,
                       self.b_y, self.c0]



        # as many columns as context window size
        # as many lines as words in the sentence


        sigma = lambda x: 1 / (1 + T.exp(-x))

        act = T.tanh


        def recurrence(x_t, h_tm1, c_tm1):
            i_t = sigma(theano.dot(x_t, self.W_xi) + theano.dot(h_tm1, self.W_hi) + theano.dot(c_tm1, self.W_ci) + self.b_i)
            f_t = sigma(theano.dot(x_t, self.W_xf) + theano.dot(h_tm1, self.W_hf) + theano.dot(c_tm1, self.W_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * act(theano.dot(x_t, self.W_xc) + theano.dot(h_tm1, self.W_hc) + self.b_c)
            o_t = sigma(theano.dot(x_t, self.W_xo) + theano.dot(h_tm1, self.W_ho) + theano.dot(c_t, self.W_co) + self.b_o)
            h_t = o_t * act(c_t)
            s_t = sigma(theano.dot(h_t, self.W_hy) + self.b_y)
            return [h_t, c_t, s_t]

        [self.h, _, self.s], _ = theano.scan(fn=recurrence,
                                          sequences=self.input,
                                          outputs_info=[self.h0, self.c0, None],
                                          n_steps=self.input.shape[0])

        self.p_y_given_x = T.nnet.softmax(self.s)
        self.y_out = T.argmax(self.p_y_given_x, axis=1)
        self.loss = lambda y: self.negative_log_likelihood(y)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the sequence
        over the total number of examples in the sequence ; zero one
        loss over the size of the sequence
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                            ('y', y.type, 'y_out', self.y_out.type))

            # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_out, y))
        else:
            raise NotImplementedError()



#TODO: build and train a MLP to learn parity function
def test_mlp_parity(n_bit, learning_rate=0.01, L1_reg=0, L2_reg=0, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=False):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    rng = numpy.random.RandomState(22345)

    # TODO: construct a neural network, either MLP or CNN.
    # classifier = myMLP(...)
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=n_bit,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=2
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
        ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    

    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(n_bit, learning_rate=0.09, n_epochs=100,
             batch_size=8, n_hidden=500, verbose = True):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(22345)

    classifier = RNN(
        input=x,
        n_in=n_bit,
        n_hidden=n_hidden,
        n_out=2)
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.loss(y)


    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
        ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
                        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)




    
    
#TODO: build and train a LSTM to learn parity function
def test_lstm_parity(n_bit, learning_rate=0.09, n_epochs=100,
             batch_size=8, n_hidden=500, verbose = True):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    rng = numpy.random.RandomState(22345)

    classifier = LSTM(
        input=x,
        n_in=n_bit,
        n_hidden=n_hidden,
        n_out=2)
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.loss(y)

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
        ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

    
if __name__ == '__main__':
    test_mlp_parity()
