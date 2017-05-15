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


from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, drop, DropoutHiddenLayer, LeNetConvPoolLayer, \
    DropoutLeNetConvPoolLayer, train_nn, global_contrast_normalize, RMSprop, ConvLayer, Adam, \
    batchNormLenetConvPoolLayer, gradient_updates_momentum

#Problem 1
#Implement the convolutional neural network architecture depicted in HW3 problem 1
#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py
def test_lenet(learning_rate=0.1, n_epochs=500,
               nkerns=[16, 64], batch_size=200,
               filtershape=2, pool_size=2,
               imageshape=32,verbose=False):
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
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, imageshape, imageshape),
        filter_shape=(nkerns[0], 3, filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], imageshape, imageshape),
        filter_shape=(nkerns[1], nkerns[0], filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * imageshape * imageshape,
        n_out=512,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
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





#Problem 2.1
#Write a function to add translations
def translate_image(train_set, shiftX, shiftY):
    for i in range(3):
        temp = numpy.copy(train_set)[i]
        xsize, ysize = temp.shape
        neg = lambda s: max(0, s)
        posX = lambda s: min(xsize, s)
        posY = lambda s: min(ysize, s)
        shift_img = numpy.zeros_like(temp)
        shift_img[neg(shiftX):posX(xsize+shiftX), neg(shiftY):posY(ysize+shiftY)] = temp[neg(-shiftX):posX(xsize-shiftX), neg(-shiftY):posY(ysize-shiftY)]
        train_set[i] = shift_img
    return train_set




# Implement a convolutional neural network with the translation method for augmentation
def test_lenet_translation(learning_rate=0.1, n_epochs=500,
               nkerns=[16, 64], batch_size=200,
               filtershape=2, pool_size=2,
               imageshape=32,verbose=False):
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

    # image shifting for train set
    def shift(img):
        shiftX = int(numpy.random.randint(-2, 3, size=1))
        shiftY = int(numpy.random.randint(-2, 3, size=1))
        img = numpy.reshape(img, (3, 32, 32))
        img = translate_image(img, shiftX, shiftY)
        img = img.reshape(3072)
        return img

    new_train_set_x = numpy.apply_along_axis(shift, 1, train_set_x)
    new_train_set_y = numpy.copy(train_set_y)
    train_set_x = numpy.vstack([train_set_x, new_train_set_x])
    train_set_y = numpy.append(train_set_y, new_train_set_y)


    # shared dataset
    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))

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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, imageshape, imageshape),
        filter_shape=(nkerns[0], 3, filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], imageshape, imageshape),
        filter_shape=(nkerns[1], nkerns[0], filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * imageshape * imageshape,
        n_out=512,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
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




#Problem 2.2
#Write a function to add roatations
def rotate_image(train_set):
    angle = numpy.random.randint(-4, 5)
    train_set = numpy.reshape(train_set, (3, 32, 32)).transpose(1, 2, 0)
    train_set = ndimage.interpolation.rotate(train_set, angle=angle, reshape=False)
    train_set = train_set.transpose(2, 0, 1).reshape(3072)
    return train_set


#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation(learning_rate=0.1, n_epochs=500,
               nkerns=[16, 64], batch_size=200,
               filtershape=2, pool_size=2,
               imageshape=32,verbose=False):
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

    new_train_set_x = numpy.apply_along_axis(rotate_image, 1, train_set_x)
    new_train_set_y = numpy.copy(train_set_y)
    train_set_x = numpy.vstack([train_set_x, new_train_set_x])
    train_set_y = numpy.append(train_set_y, new_train_set_y)

    # shared dataset
    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))

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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, imageshape, imageshape),
        filter_shape=(nkerns[0], 3, filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], imageshape, imageshape),
        filter_shape=(nkerns[1], nkerns[0], filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * imageshape * imageshape,
        n_out=512,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
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


#Problem 2.3
#Write a function to flip images
def flip_image(training_set, switch=1):
    if switch == 1:
        training_set = training_set.reshape(3, 32, 32)
        for i in range(3):
            training_set[i] = numpy.flipud(training_set[i])
        return training_set.reshape(3072)
    else: return training_set

#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip(learning_rate=0.1, n_epochs=500,
               nkerns=[16, 64], batch_size=200,
               filtershape=2, pool_size=2,
               imageshape=32,verbose=False):
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

    # image shifting for train set
    def flip_image_random(img):
        switch = int(numpy.random.randint(0, 2))
        img = flip_image(img, switch)
        return img


    train_set_x = numpy.apply_along_axis(flip_image_random, 1, train_set_x)

    # shared dataset
    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))

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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, imageshape, imageshape),
        filter_shape=(nkerns[0], 3, filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], imageshape, imageshape),
        filter_shape=(nkerns[1], nkerns[0], filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * imageshape * imageshape,
        n_out=512,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
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
    
    
#Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean
def noise_injection(training_set, mean=0, var=10**-5, switch=True):
    if switch == 1:
        training_set = training_set + numpy.random.normal(mean, var, training_set.shape[0])
    else:
        training_set = training_set + numpy.random.uniform(-var, var, training_set.shape[0])
    return training_set

#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input(learning_rate=0.1, n_epochs=500,
               nkerns=[16, 64], batch_size=200,
               filtershape=2, pool_size=2,
               imageshape=32,verbose=False):
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

    # image shifting for train set
    def noise_injection_on_data(img):
        switch = int(numpy.random.randint(0, 2))
        mean = 0
        var = 0.05
        img = noise_injection(img, mean, var, switch)
        return img

    train_set_x = numpy.apply_along_axis(noise_injection_on_data, 1, train_set_x)

    # shared dataset
    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))

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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, imageshape, imageshape),
        filter_shape=(nkerns[0], 3, filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], imageshape, imageshape),
        filter_shape=(nkerns[1], nkerns[0], filtershape, filtershape),
        poolsize=(pool_size, pool_size)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)
    imageshape = ((imageshape - filtershape + 1) / (pool_size))

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * imageshape * imageshape,
        n_out=512,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
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





#Problem 3 
#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset
def MY_lenet(learning_rate=0.1, n_epochs=500, nkerns=[32, 64, 128], batch_size=200, verbose=False):
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




    # train_set_x = global_contrast_normalize(train_set_x)
    # valid_set_x = global_contrast_normalize(valid_set_x)
    # test_set_x = global_contrast_normalize(test_set_x)
    #
    #
    # # shared dataset
    # test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    # valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    # train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))

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


    # learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
    #                                             dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))


    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = batchNormLenetConvPoolLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        activation=relu,
        border_mode="half",
        batch_norm=True
    )

    layer1 = batchNormLenetConvPoolLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 16, 16),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        activation=relu,
        border_mode="half",
        batch_norm=True
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer2 = ConvLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 8, 8),
        filter_shape=(nkerns[2], nkerns[1], 3, 3)
    )

    layer3 = ConvLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 8, 8),
        filter_shape=(nkerns[2], nkerns[2], 3, 3)
    )

    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[2], 8, 8),
        filter_shape=(nkerns[1], nkerns[2], 3, 3),
        activation=relu,
        border_mode="half"
    )


    layer5_input = layer4.output.flatten(2)

    # construct a fully-connected sigmoidal layer

    layer5 = DropoutHiddenLayer(
        rng,
        input=layer5_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=4096,
        activation=relu,
        is_train=training_enabled,
        p=0.5
    )

    layer6 = DropoutHiddenLayer(
        rng,
        input=layer5.output,
        n_in=4096,
        n_out=512,
        activation=relu,
        is_train=training_enabled,
        p=0.5
    )

    # classify the values of the fully-connected sigmoidal layer
    layer7 = LogisticRegression(input=layer6.output, n_in=512, n_out=10)


    # the cost we minimize during training is the NLL of the model
    # L2_norm = layer0.L2_sqr + layer1.L2_sqr + layer3.L2_sqr + layer4.L2_sqr + layer5.L2_sqr + layer6.L2_sqr + layer7.L2_sqr

    cost = layer7.negative_log_likelihood(y)

    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params + layer7.params

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
        layer7.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer7.errors(y),
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


    # # Theano function to decay the learning rate
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #                                       updates={learning_rate: learning_rate * learning_rate_decay})
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)









#Problem4
#Implement the convolutional neural network depicted in problem4 
def MY_CNN(corrupted_imgs, original_imgs, learning_rate=0.1, l2_reg=0.00001, n_epochs=128, batch_size=50, verbose=False):
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
    train_set = shared_dataset(corrupted_imgs[0])
    valid_set = shared_dataset(corrupted_imgs[1])
    test_set = shared_dataset(corrupted_imgs[2])

    train_set_orignal = shared_dataset(original_imgs[0])
    valid_set_orignal = shared_dataset(original_imgs[1])
    test_set_orignal = shared_dataset(original_imgs[2])

    train_set_x, train_set_y = train_set
    train_set_x_orignal, train_set_orignal_y = train_set_orignal

    valid_set_x, valid_set_y = valid_set
    valid_set_x_orignal, valid_set_y_orignal = valid_set_orignal

    test_set_x, test_set_y = test_set
    test_set_x_orignal, test_set_y_orignal = test_set_orignal

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    x_bar = T.matrix('x_bar')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    layer0_input = x.reshape((batch_size, 3, 32, 32))
    right_answer = x_bar.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

    layer0 = ConvLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(64, 3, 3, 3)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = ConvLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3)
    )

    layer2 = pool.pool_2d(
        input=layer1.output,
        ds=(2, 2),
        ignore_border=True
    )

    layer3 = ConvLayer(
        rng,
        input=layer2,
        image_shape=(batch_size, 64, 16, 16),
        filter_shape=(128, 64, 3, 3)
    )

    layer4 = ConvLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3)
    )

    layer5 = pool.pool_2d(
        input=layer4.output,
        ds=(2, 2),
        ignore_border=True
    )

    layer6 = ConvLayer(
        rng,
        input=layer5,
        image_shape=(batch_size, 128, 8, 8),
        filter_shape=(256, 128, 3, 3)
    )

    layer7 = bilinear_upsampling(layer6.output, 2, batch_size=batch_size, num_input_channels=256)

    layer8 = ConvLayer(
        rng,
        input=layer7,
        image_shape=(batch_size, 256, 16, 16),
        filter_shape=(128, 256, 3, 3)
    )

    layer9 = ConvLayer(
        rng,
        input=layer8.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3)
    )

    output0 = layer4.output + layer9.output

    layer10 = bilinear_upsampling(output0, 2, batch_size=batch_size, num_input_channels=128)

    layer11 = ConvLayer(
        rng,
        input=layer10,
        image_shape=(batch_size, 128, 32, 32),  # sohuld be 2
        filter_shape=(64, 128, 3, 3)
    )

    layer12 = ConvLayer(
        rng,
        input=layer11.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3)
    )

    output1 = layer12.output + layer1.output

    output2 = ConvLayer(
        rng,
        input=output1,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(3, 64, 3, 3)
    )

    # the cost we minimize during training is the NLL of the model

    cost = ((output2.output - right_answer) ** 2).sum() / (batch_size * 1.0 * 3072)
    # cost_train = cost
    cost_train = ((output2.output - right_answer) ** 2).sum() / (batch_size * 1.0 * 3072) + l2_reg * (
    layer0.L2_sqr + layer1.L2_sqr + layer3.L2_sqr + layer4.L2_sqr + layer6.L2_sqr + layer8.L2_sqr + layer9.L2_sqr + layer11.L2_sqr + layer12.L2_sqr + output2.L2_sqr)

    # create a list of all model parameters to be fit by gradient descent



    params = layer0.params + layer1.params + layer3.params + layer4.params + layer6.params + layer8.params + layer9.params + layer11.params + layer12.params + output2.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost_train, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = Adam(params, grads, lr=learning_rate, b1=0.1, b2=0.001, e=1e-8)
    # updates = Adam(params,grads)
    # updates = gradient_updates_momentum(params, cost)
    test_model = theano.function(
        [index],
        cost,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            x_bar: test_set_x_orignal[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            x_bar: valid_set_x_orignal[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_model = theano.function(
        [index],
        cost_train,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            x_bar: train_set_x_orignal[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_result = theano.function(
        [index],
        output2.output,
        givens={
            x: test_set_x[:index]
        }
    )

    print('... training')
    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 1.5  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant

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
    validation_frequency = min(n_train_batches, patience // 2)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
            # print (cost_ij)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation mse %f ' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           this_validation_loss))

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
                        print(('     epoch %i, minibatch %i/%i, test mse of '
                               'best model %f ') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    test_image_result = test_result(batch_size)
    return test_image_result