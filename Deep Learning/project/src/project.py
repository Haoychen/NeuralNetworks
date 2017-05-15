"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
from __future__ import print_function

import timeit
import inspect
import sys
import numpy
from scipy import ndimage

import theano
import theano.tensor as T
from theano.tensor.nnet import relu
from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import bilinear_upsampling


from project_utils import shared_dataset, load_data
# from project_nn import LogisticRegression, HiddenLayer, drop, DropoutHiddenLayer, LeNetConvPoolLayer, \
#     DropoutLeNetConvPoolLayer, train_nn, global_contrast_normalize, RMSprop, ConvLayer, Adam, \
#     batchNormLenetConvPoolLayer, gradient_updates_momentum
from project_nn import ConvLayer, LogisticRegression, train_nn, DropoutConvLayer, DropoutPoolLayer, DropoutHiddenLayer, batchNormLenetConvAvgPoolLayer

def drop(data, p=0.2):
    mask = numpy.random.binomial(n=1, p=1-p, size=(data.shape[0], data.shape[1]))
    return numpy.multiply(data, mask)

def Strided_CNN_C(learning_rate=0.1, n_epochs=500, batch_size=64, verbose=False):
    """
        Wrapper function for testing LeNet on SVHN dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        :type batch_size: int
        :param batch_szie: number of examples in minibatch.

        :type verbose: boolean
        :param verbose: to print out epoch summary or not to.

        """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    # training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction


    # learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
    #                                             dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))


    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # 4D output tensor is thus of shape (batch_size, 96, 15, 15)
    layer0 = ConvLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(96, 3, 3, 3),
        border_mode='half'
    )

    # (32 - 3)/2 + 1 = 15
    layer1 = ConvLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        stride=2
    )

    # 13 - 3 + 1 = 11
    layer2 = ConvLayer(
        rng=rng,
        input=layer1.output,
        image_shape=(batch_size, 96, 15, 15),
        filter_shape=(192, 96, 3, 3),
        border_mode='half'
    )



    # (15 - 3) /2 + 1 = 7
    layer3 = ConvLayer(
        rng=rng,
        input=layer2.output,
        image_shape=(batch_size, 192, 15, 15),
        filter_shape=(192, 192, 3, 3),
        stride=2
    )

    # 3 - 3 + 1 = 1
    layer4 = ConvLayer(
        rng=rng,
        input=layer3.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(192, 192, 3, 3),
        border_mode='half'
    )

    layer5 = ConvLayer(
        rng=rng,
        input=layer4.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(192, 192, 1, 1)
    )

    layer6 = ConvLayer(
        rng=rng,
        input=layer5.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(10, 192, 1, 1)
    )


    layer7 = pool.pool_2d(
        input=layer6.output,
        ds=(6, 6),
        ignore_border=True,
        mode='average_exc_pad'

    )


    layer8_input = layer7.flatten(2)

    # construct a fully-connected sigmoidal layer

    # classify the values of the fully-connected sigmoidal layer
    layer8 = LogisticRegression(input=layer8_input, n_in=10, n_out=10)


    # the cost we minimize during training is the NLL of the model
    # L2_norm = layer0.L2_sqr + layer1.L2_sqr + layer3.L2_sqr + layer4.L2_sqr + layer5.L2_sqr + layer6.L2_sqr + layer7.L2_sqr

    cost = layer8.negative_log_likelihood(y)

    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params\
              + layer8.params

    # grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * numpy.cast[theano.config.floatX](0.))
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update,
                        momentum * param_update + (numpy.cast[theano.config.floatX](1.) - momentum) * T.grad(cost,
                                                                                                             param)))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    # # Theano function to decay the learning rate
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #                                       updates={learning_rate: learning_rate * learning_rate_decay})
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

def Strided_CNN_C_Dropout(init_learning_rate=0.25, n_epochs=500, batch_size=64, verbose=False):
    """
        Wrapper function for testing LeNet on SVHN dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        :type batch_size: int
        :param batch_szie: number of examples in minibatch.

        :type verbose: boolean
        :param verbose: to print out epoch summary or not to.

        """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(theano_shared=False)

    train_set_x, train_set_y = datasets[0]
    train_set_x = drop(train_set_x, p=0.2)
    train_set = shared_dataset((train_set_x, train_set_y))
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])




    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction

    learning_rate = theano.shared(numpy.asarray(init_learning_rate,
                                             dtype=theano.config.floatX))
    # learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
    #                                             dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # 4D output tensor is thus of shape (batch_size, 96, 15, 15)
    layer0 = ConvLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(96, 3, 3, 3),
        border_mode=1
    )

    # (32 - 3 + 2)/2 + 1 = 16
    layer1 = DropoutConvLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        border_mode=1,
        stride=2,
        is_train=training_enabled
    )

    # 13 - 3 + 1 = 11
    layer2 = ConvLayer(
        rng=rng,
        input=layer1.output,
        image_shape=(batch_size, 96, 16, 16),
        filter_shape=(192, 96, 3, 3),
        border_mode=1
    )


    # (16 - 3 + 2) /2 + 1 = 8
    layer3 = DropoutConvLayer(
        rng=rng,
        input=layer2.output,
        image_shape=(batch_size, 192, 16, 16),
        filter_shape=(192, 192, 3, 3),
        stride=2,
        border_mode=1,
        is_train=training_enabled
    )

    # 3 - 3 + 1 = 1
    layer4 = ConvLayer(
        rng=rng,
        input=layer3.output,
        image_shape=(batch_size, 192, 8, 8),
        filter_shape=(192, 192, 3, 3),
        border_mode=1
    )

    layer5 = ConvLayer(
        rng=rng,
        input=layer4.output,
        image_shape=(batch_size, 192, 8, 8),
        filter_shape=(192, 192, 1, 1)
    )

    layer6 = ConvLayer(
        rng=rng,
        input=layer5.output,
        image_shape=(batch_size, 192, 8, 8),
        filter_shape=(10, 192, 1, 1)
    )

    layer7 = pool.pool_2d(
        input=layer6.output,
        ds=(8, 8),
        ignore_border=True,
        mode='average_exc_pad'

    )

    layer8_input = layer7.flatten(2)

    # construct a fully-connected sigmoidal layer

    # classify the values of the fully-connected sigmoidal layer
    layer8 = LogisticRegression(input=layer8_input, n_in=10, n_out=10)

    # the cost we minimize during training is the NLL of the model
    L2_norm = layer0.L2_sqr + layer1.L2_sqr + layer3.L2_sqr + layer4.L2_sqr + layer5.L2_sqr + layer6.L2_sqr \
               + layer8.L2_sqr

    cost = layer8.negative_log_likelihood(y) + 0.001 * L2_norm

    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params \
              + layer8.params


    # grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * numpy.cast[theano.config.floatX](0.))
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update,
                        momentum * param_update + (numpy.cast[theano.config.floatX](1.) - momentum) * T.grad(cost,
                                                                                                             param)))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)
        }
    )

    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                                          updates={learning_rate: learning_rate * 0.1})

    # # Theano function to decay the learning rate
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #                                       updates={learning_rate: learning_rate * learning_rate_decay})
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        if epoch in [200, 250, 300]:
            new_learning_rate = decay_learning_rate()
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                        ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)



def ConvPool_CNN_C(learning_rate=0.1, n_epochs=500, batch_size=64, verbose=False):
    """
        Wrapper function for testing LeNet on SVHN dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        :type batch_size: int
        :param batch_szie: number of examples in minibatch.

        :type verbose: boolean
        :param verbose: to print out epoch summary or not to.

        """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    # training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction


    # learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
    #                                             dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))


    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # 4D output tensor is thus of shape (batch_size, 96, 15, 15)
    layer0 = ConvLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(96, 3, 3, 3),
        border_mode='half'
    )
    # (30 - 3 + 1) = 28
    layer1 = ConvLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        border_mode='half'
    )
    # (32 - 3)/2 + 1 = 15
    layer2 = ConvLayer(
        rng=rng,
        input=layer1.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        border_mode='half'
    )

    layer3 = pool.pool_2d(
        input=layer2.output,
        ds=(3, 3),
        ignore_border=True,
        st=(2, 2)
    )

    # 13 - 3 + 1 = 11
    layer4 = ConvLayer(
        rng=rng,
        input=layer3,
        image_shape=(batch_size, 96, 15, 15),
        filter_shape=(192, 96, 3, 3),
        border_mode='half'
    )

    # 11 - 3 + 1 = 9
    layer5 = ConvLayer(
        rng=rng,
        input=layer4.output,
        image_shape=(batch_size, 192, 15, 15),
        filter_shape=(192, 192, 3, 3),
        border_mode='half'
    )

    # (15 - 3) /2 + 1 = 7
    layer6 = ConvLayer(
        rng=rng,
        input=layer5.output,
        image_shape=(batch_size, 192, 15, 15),
        filter_shape=(192, 192, 3, 3),
        border_mode='half'
    )

    # 3 - 3 + 1 = 1
    layer7 = pool.pool_2d(
        input=layer6.output,
        ds=(3, 3),
        ignore_border=True,
        st=(2, 2)
    )

    layer8 = ConvLayer(
        rng=rng,
        input=layer7,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(192, 192, 3, 3),
        border_mode='half'
    )

    layer9 = ConvLayer(
        rng=rng,
        input=layer8.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(192, 192, 1, 1)
    )

    layer10 = ConvLayer(
        rng=rng,
        input=layer9.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(10, 192, 1, 1)
    )


    layer11 = pool.pool_2d(
        input=layer10.output,
        ds=(6, 6),
        ignore_border=True,
        mode='average_exc_pad'

    )


    layer12_input = layer11.flatten(2)

    # construct a fully-connected sigmoidal layer

    # classify the values of the fully-connected sigmoidal layer
    layer12 = LogisticRegression(input=layer12_input, n_in=10, n_out=10)


    # the cost we minimize during training is the NLL of the model
    # L2_norm = layer0.L2_sqr + layer1.L2_sqr + layer3.L2_sqr + layer4.L2_sqr + layer5.L2_sqr + layer6.L2_sqr + layer7.L2_sqr

    cost = layer12.negative_log_likelihood(y)

    params = layer0.params + layer1.params + layer2.params + layer4.params + layer5.params + layer6.params\
              + layer8.params + layer9.params + layer10.params + layer12.params

    # grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * numpy.cast[theano.config.floatX](0.))
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update,
                        momentum * param_update + (numpy.cast[theano.config.floatX](1.) - momentum) * T.grad(cost,
                                                                                                             param)))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer12.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer12.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    # # Theano function to decay the learning rate
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #                                       updates={learning_rate: learning_rate * learning_rate_decay})
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

def ConvPool_CNN_C_Dropout(init_learning_rate=0.25, n_epochs=500, batch_size=64, verbose=False):
    """
        Wrapper function for testing LeNet on SVHN dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        :type batch_size: int
        :param batch_szie: number of examples in minibatch.

        :type verbose: boolean
        :param verbose: to print out epoch summary or not to.

        """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(theano_shared=False)

    train_set_x, train_set_y = datasets[0]
    train_set_x = drop(train_set_x, p=0.2)
    train_set = shared_dataset((train_set_x, train_set_y))
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction
    learning_rate = theano.shared(numpy.asarray(init_learning_rate,
                                                dtype=theano.config.floatX))

    # learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
    #                                             dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))


    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # 4D output tensor is thus of shape (batch_size, 96, 15, 15)
    layer0 = ConvLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(96, 3, 3, 3),
        border_mode='half'
    )
    # (30 - 3 + 1) = 28
    layer1 = ConvLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        border_mode='half'
    )
    # (32 - 3)/2 + 1 = 15
    layer2 = ConvLayer(
        rng=rng,
        input=layer1.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        border_mode='half'
    )

    layer3 = DropoutPoolLayer(
        input=layer2.output,
        ds=3,
        st=2,
        is_train=training_enabled
    )

    # 13 - 3 + 1 = 11
    layer4 = ConvLayer(
        rng=rng,
        input=layer3.output,
        image_shape=(batch_size, 96, 15, 15),
        filter_shape=(192, 96, 3, 3),
        border_mode='half'
    )

    # 11 - 3 + 1 = 9
    layer5 = ConvLayer(
        rng=rng,
        input=layer4.output,
        image_shape=(batch_size, 192, 15, 15),
        filter_shape=(192, 192, 3, 3),
        border_mode='half'
    )

    # (15 - 3) /2 + 1 = 7
    layer6 = ConvLayer(
        rng=rng,
        input=layer5.output,
        image_shape=(batch_size, 192, 15, 15),
        filter_shape=(192, 192, 3, 3),
        border_mode='half'
    )

    layer7 = DropoutPoolLayer(
        input=layer6.output,
        ds=3,
        st=2,
        is_train=training_enabled
    )

    layer8 = ConvLayer(
        rng=rng,
        input=layer7.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(192, 192, 3, 3),
        border_mode='half'
    )

    layer9 = ConvLayer(
        rng=rng,
        input=layer8.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(192, 192, 1, 1)
    )

    layer10 = ConvLayer(
        rng=rng,
        input=layer9.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(10, 192, 1, 1)
    )


    layer11 = pool.pool_2d(
        input=layer10.output,
        ds=(6, 6),
        ignore_border=True,
        mode='average_exc_pad'

    )


    layer12_input = layer11.flatten(2)

    # construct a fully-connected sigmoidal layer

    # classify the values of the fully-connected sigmoidal layer
    layer12 = LogisticRegression(input=layer12_input, n_in=10, n_out=10)


    # the cost we minimize during training is the NLL of the model
    L2_norm = layer0.L2_sqr + layer1.L2_sqr + layer2.L2_sqr + layer4.L2_sqr + layer5.L2_sqr + layer6.L2_sqr \
              + layer8.L2_sqr + layer9.L2_sqr + layer10.L2_sqr + layer12.L2_sqr

    cost = layer12.negative_log_likelihood(y) + 0.001 * L2_norm

    params = layer0.params + layer1.params + layer2.params + layer4.params + layer5.params + layer6.params\
              + layer8.params + layer9.params + layer10.params + layer12.params

    # grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * numpy.cast[theano.config.floatX](0.))
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update,
                        momentum * param_update + (numpy.cast[theano.config.floatX](1.) - momentum) * T.grad(cost,
                                                                                                             param)))

    # create a function to compute the mistakes that are made by the model
        # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer12.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer12.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)
        }
    )

    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                                          updates={learning_rate: learning_rate * 0.1})


    # # Theano function to decay the learning rate
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #                                       updates={learning_rate: learning_rate * learning_rate_decay})
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        if epoch in [200, 250, 300]:
            new_learning_rate = decay_learning_rate()
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                        ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)




#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset
def ALL_CNN_C(learning_rate=0.1, n_epochs=500, batch_size=64, verbose=False):
    """
        Wrapper function for testing LeNet on SVHN dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        :type batch_size: int
        :param batch_szie: number of examples in minibatch.

        :type verbose: boolean
        :param verbose: to print out epoch summary or not to.

        """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    # training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction


    # learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
    #                                             dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))



    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # 4D output tensor is thus of shape (batch_size, 96, 15, 15)
    layer0 = ConvLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(96, 3, 3, 3),
        border_mode='half'
    )
    # (30 - 3 + 1) = 28
    layer1 = ConvLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        border_mode='half'
    )
    # (32 - 3)/2 + 1 = 15
    layer2 = ConvLayer(
        rng=rng,
        input=layer1.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        stride=2
    )

    # 13 - 3 + 1 = 11
    layer3 = ConvLayer(
        rng=rng,
        input=layer2.output,
        image_shape=(batch_size, 96, 15, 15),
        filter_shape=(192, 96, 3, 3),
        border_mode='half'
    )

    # 11 - 3 + 1 = 9
    layer4 = ConvLayer(
        rng=rng,
        input=layer3.output,
        image_shape=(batch_size, 192, 15, 15),
        filter_shape=(192, 192, 3, 3),
        border_mode='half'
    )

    # (15 - 3) /2 + 1 = 7
    layer5 = ConvLayer(
        rng=rng,
        input=layer4.output,
        image_shape=(batch_size, 192, 15, 15),
        filter_shape=(192, 192, 3, 3),
        stride=2
    )

    # 3 - 3 + 1 = 1
    layer6 = ConvLayer(
        rng=rng,
        input=layer5.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(192, 192, 3, 3),
        border_mode='half'
    )

    layer7 = ConvLayer(
        rng=rng,
        input=layer6.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(192, 192, 1, 1)
    )

    layer8 = ConvLayer(
        rng=rng,
        input=layer7.output,
        image_shape=(batch_size, 192, 7, 7),
        filter_shape=(10, 192, 1, 1)
    )


    layer9 = pool.pool_2d(
        input=layer8.output,
        ds=(6, 6),
        ignore_border=True,
        mode='average_exc_pad'

    )


    layer10_input = layer9.flatten(2)

    # construct a fully-connected sigmoidal layer

    # classify the values of the fully-connected sigmoidal layer
    layer10 = LogisticRegression(input=layer10_input, n_in=10, n_out=10)


    # the cost we minimize during training is the NLL of the model
    # L2_norm = layer0.L2_sqr + layer1.L2_sqr + layer3.L2_sqr + layer4.L2_sqr + layer5.L2_sqr + layer6.L2_sqr + layer7.L2_sqr

    cost = layer10.negative_log_likelihood(y)

    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params\
             + layer7.params + layer8.params + layer10.params

    # grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * numpy.cast[theano.config.floatX](0.))
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update,
                        momentum * param_update + (numpy.cast[theano.config.floatX](1.) - momentum) * T.grad(cost,
                                                                                                             param)))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer10.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer10.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    # # Theano function to decay the learning rate
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #                                       updates={learning_rate: learning_rate * learning_rate_decay})
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)


def ALL_CNN_C_Dropout(init_learning_rate=0.25, n_epochs=500, batch_size=64, verbose=False):
    """
        Wrapper function for testing LeNet on SVHN dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        :type batch_size: int
        :param batch_szie: number of examples in minibatch.

        :type verbose: boolean
        :param verbose: to print out epoch summary or not to.

        """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    # datasets = load_data(theano_shared=False)
    #
    # train_set_x, train_set_y = datasets[0]
    # train_set_x = drop(train_set_x, p=0.2)
    # train_set = shared_dataset((train_set_x, train_set_y))
    # train_set_x, train_set_y = train_set
    # valid_set_x, valid_set_y = shared_dataset(datasets[1])
    # test_set_x, test_set_y = shared_dataset(datasets[2])




    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction

    learning_rate = theano.shared(numpy.asarray(init_learning_rate,
                                             dtype=theano.config.floatX))
    # learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
    #                                             dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+2+1 , 32-3+1) = (31, 30)
    # 4D output tensor is thus of shape (batch_size, 96, 15, 15)
    layer0 = ConvLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(96, 3, 3, 3),
        border_mode=1
    )
    # (31 - 3 + 1 + 1) = 30
    layer1 = ConvLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        border_mode=1
    )
    # (32 - 3 + 2)/2 + 1 = 16
    layer2 = DropoutConvLayer(
        rng=rng,
        input=layer1.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        stride=2,
        border_mode=1,
        is_train=training_enabled
    )

    # 15 - 3 + 2 + 1 = 15
    layer3 = ConvLayer(
        rng=rng,
        input=layer2.output,
        image_shape=(batch_size, 96, 16, 16),
        filter_shape=(192, 96, 3, 3),
        border_mode=1
    )

    # 15 - 3 + 1 + 1 = 14
    layer4 = ConvLayer(
        rng=rng,
        input=layer3.output,
        image_shape=(batch_size, 192, 16, 16),
        filter_shape=(192, 192, 3, 3),
        border_mode=1
    )

    # (16 - 3 + 2) /2 + 1 = 8
    layer5 = DropoutConvLayer(
        rng=rng,
        input=layer4.output,
        image_shape=(batch_size, 192, 16, 16),
        filter_shape=(192, 192, 3, 3),
        stride=2,
        border_mode=1,
        is_train=training_enabled
    )

    # 7 - 3 + 1 + 1  = 6
    layer6 = ConvLayer(
        rng=rng,
        input=layer5.output,
        image_shape=(batch_size, 192, 8, 8),
        filter_shape=(192, 192, 3, 3),
        border_mode=1
    )

    layer7 = ConvLayer(
        rng=rng,
        input=layer6.output,
        image_shape=(batch_size, 192, 8, 8),
        filter_shape=(192, 192, 1, 1)
    )

    layer8 = ConvLayer(
        rng=rng,
        input=layer7.output,
        image_shape=(batch_size, 192, 8, 8),
        filter_shape=(16, 192, 1, 1)
    )

    layer9 = pool.pool_2d(
        input=layer8.output,
        ds=(8, 8),
        ignore_border=True,
        mode='average_inc_pad'

    )

    layer10_input = layer9.flatten(2)

    # construct a fully-connected sigmoidal layer

    # classify the values of the fully-connected sigmoidal layer
    layer10 = LogisticRegression(input=layer10_input, n_in=16, n_out=10)

    # the cost we minimize during training is the NLL of the model
    L2_norm = layer0.L2_sqr + layer1.L2_sqr + layer3.L2_sqr + layer4.L2_sqr + layer5.L2_sqr + layer6.L2_sqr \
              + layer7.L2_sqr + layer8.L2_sqr + layer10.L2_sqr

    cost = layer10.negative_log_likelihood(y) + 0.001 * L2_norm

    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params \
             + layer7.params + layer8.params + layer10.params


    # grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * numpy.cast[theano.config.floatX](0.))
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update,
                        momentum * param_update + (numpy.cast[theano.config.floatX](1.) - momentum) * T.grad(cost,
                                                                                                             param)))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer10.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer10.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)
        }
    )

    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                                          updates={learning_rate: learning_rate * 0.1})

    # # Theano function to decay the learning rate
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #                                       updates={learning_rate: learning_rate * learning_rate_decay})
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        if epoch in [200, 250, 300]:
            new_learning_rate = decay_learning_rate()
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                        ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)



def rotate_image(train_set):
    angle = numpy.random.randint(-4, 5)
    train_set = numpy.reshape(train_set, (3, 32, 32)).transpose(1, 2, 0)
    train_set = ndimage.interpolation.rotate(train_set, angle=angle, reshape=False)
    train_set = train_set.transpose(2, 0, 1).reshape(3072)
    return train_set

def flip_image(training_set, switch=1):
    if switch == 1:
        training_set = training_set.reshape(3, 32, 32)
        for i in range(3):
            training_set[i] = numpy.flipud(training_set[i])
        return training_set.reshape(3072)
    else: return training_set



def CNN_PLUS_FC(init_learning_rate=0.01, n_epochs=350, batch_size=64, verbose=False):
    """
        Wrapper function for testing LeNet on SVHN dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        :type batch_size: int
        :param batch_szie: number of examples in minibatch.

        :type verbose: boolean
        :param verbose: to print out epoch summary or not to.

        """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(theano_shared=False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    new_train_set_x_1 = numpy.apply_along_axis(rotate_image, 1, train_set_x)
    new_train_set_y_1 = numpy.copy(train_set_y)
    train_set_x = numpy.vstack([train_set_x, new_train_set_x_1])
    train_set_y = numpy.append(train_set_y, new_train_set_y_1)
    new_train_set_x_2 = numpy.apply_along_axis(rotate_image, 1, train_set_x)
    new_train_set_y_2 = numpy.copy(train_set_y)
    train_set_x = numpy.vstack([train_set_x, new_train_set_x_2])
    train_set_y = numpy.append(train_set_y, new_train_set_y_2)

    def flip_image_random(img):
        switch = int(numpy.random.randint(0, 5))
        img = flip_image(img, switch)
        return img

    train_set_x = numpy.apply_along_axis(flip_image_random, 1, train_set_x)
    train_set_x = drop(train_set_x, p=0.2)


    # shared dataset
    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))
    


    # datasets = load_data(theano_shared=False)
    # 
    # train_set_x, train_set_y = datasets[0]
    # train_set_x = drop(train_set_x, p=0.2)
    # train_set = shared_dataset((train_set_x, train_set_y))
    # train_set_x, train_set_y = train_set
    # valid_set_x, valid_set_y = shared_dataset(datasets[1])
    # test_set_x, test_set_y = shared_dataset(datasets[2])




    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction

    learning_rate = theano.shared(numpy.asarray(init_learning_rate,
                                             dtype=theano.config.floatX))
    # learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
    #                                             dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+2+1 , 32-3+1) = (31, 30)
    # 4D output tensor is thus of shape (batch_size, 96, 15, 15)
    layer0 = ConvLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(96, 3, 3, 3),
        border_mode=1
    )
    # (31 - 3 + 1 + 1) = 30
    layer1 = ConvLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        border_mode=1
    )
    # (32 - 3 + 2)/2 + 1 = 16
    layer2 = DropoutConvLayer(
        rng=rng,
        input=layer1.output,
        image_shape=(batch_size, 96, 32, 32),
        filter_shape=(96, 96, 3, 3),
        stride=2,
        border_mode=1,
        is_train=training_enabled
    )

    # 15 - 3 + 2 + 1 = 15
    layer3 = ConvLayer(
        rng=rng,
        input=layer2.output,
        image_shape=(batch_size, 96, 16, 16),
        filter_shape=(192, 96, 3, 3),
        border_mode=1
    )

    # 15 - 3 + 1 + 1 = 14
    layer4 = ConvLayer(
        rng=rng,
        input=layer3.output,
        image_shape=(batch_size, 192, 16, 16),
        filter_shape=(192, 192, 3, 3),
        border_mode=1
    )

    # (16 - 3 + 2) /2 + 1 = 8
    layer5 = DropoutConvLayer(
        rng=rng,
        input=layer4.output,
        image_shape=(batch_size, 192, 16, 16),
        filter_shape=(192, 192, 3, 3),
        stride=2,
        border_mode=1,
        is_train=training_enabled
    )

    # 7 - 3 + 1 + 1  = 6
    layer6 = ConvLayer(
        rng=rng,
        input=layer5.output,
        image_shape=(batch_size, 192, 8, 8),
        filter_shape=(192, 192, 3, 3),
        border_mode=1
    )

    layer7 = ConvLayer(
        rng=rng,
        input=layer6.output,
        image_shape=(batch_size, 192, 8, 8),
        filter_shape=(192, 192, 1, 1)
    )

    # layer8 = ConvLayer(
    #     rng=rng,
    #     input=layer7.output,
    #     image_shape=(batch_size, 192, 8, 8),
    #     filter_shape=(16, 192, 1, 1)
    # )
    #
    # layer9 = pool.pool_2d(
    #     input=layer8.output,
    #     ds=(8, 8),
    #     ignore_border=True,
    #     mode='average_inc_pad'
    # )

    layer8 = batchNormLenetConvAvgPoolLayer(
        rng=rng,
        input=layer7.output,
        image_shape=(batch_size, 192, 8, 8),
        filter_shape=(16, 192, 1, 1),
        activation=relu,
        border_mode="half",
        poolsize=(8, 8),
        batch_norm=True
    )

    layer9_input = layer8.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer9 = DropoutHiddenLayer(
        rng,
        input=layer9_input,
        n_in=16,
        n_out=10,
        activation=relu,
        is_train=training_enabled,
        p=0.5
    )

    # classify the values of the fully-connected sigmoidal layer
    layer10 = LogisticRegression(input=layer9.output, n_in=10, n_out=10)

    # the cost we minimize during training is the NLL of the model
    L2_norm = layer0.L2_sqr + layer1.L2_sqr + layer3.L2_sqr + layer4.L2_sqr + layer5.L2_sqr + layer6.L2_sqr \
              + layer7.L2_sqr + layer8.L2_sqr + layer9.L2_sqr + layer10.L2_sqr

    cost = layer10.negative_log_likelihood(y) + 0.001 * L2_norm

    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params \
             + layer7.params + layer8.params + layer9.params + layer10.params


    # grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * numpy.cast[theano.config.floatX](0.))
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update,
                        momentum * param_update + (numpy.cast[theano.config.floatX](1.) - momentum) * T.grad(cost,
                                                                                                             param)))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer10.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer10.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)
        }
    )

    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                                          updates={learning_rate: learning_rate * 0.1})

    # # Theano function to decay the learning rate
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #                                       updates={learning_rate: learning_rate * learning_rate_decay})
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        if epoch in [200, 250, 300]:
            new_learning_rate = decay_learning_rate()
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                        ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
