__author__ = 'tangyh'

"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from theanoTutorial.logistic_sgd import LogisticRegression, load_data
from theanoTutorial.mlp import HiddenLayer

from dataProcess import PreProcess

from parsing.evalb import evalb

import pickle

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=False
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def shared_dataset(data_x, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return T.cast(shared_x, 'int32')


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=1):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = np.random.RandomState(23455)

    try:
        pkl_file = open(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/train.pkl'), 'rb')
        prodata, train_set, train_candcnt, valid_set, validtreestrs, valid_candcnt, test_set, testtreestrs, test_candcnt = pickle.load(pkl_file)
        pkl_file.close()
    except IOError:
        prodata= PreProcess(np.random.RandomState(123),
                 os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/embeddings.txt')
        )

        train_set,_, train_candcnt = prodata.finaldata(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/train1.txt'))
        valid_set, validtreestrs, valid_candcnt = prodata.finaldata(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/valid1.txt'))
        test_set, testtreestrs, test_candcnt  = prodata.finaldata(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/test1.txt'))

        output = open(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/train.pkl'), 'wb')
        pickle.dump((prodata,train_set, train_candcnt, valid_set, validtreestrs, valid_candcnt, test_set, testtreestrs, test_candcnt), output)
        output.close()

    print prodata.terminalset

    # compute number of minibatches for training, validation and testing
    # n_train_batches = train_set.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set.get_value(borrow=True).shape[0]
    # n_test_batches = test_set.get_value(borrow=True).shape[0]
    # n_train_batches /= batch_size
    # n_valid_batches /= batch_size
    # n_test_batches /= batch_size
    n_train_batches = len(train_set)
    n_valid_batches = len(valid_set)
    n_test_batches = len(test_set)

    train_set=shared_dataset(train_set)
    valid_set=shared_dataset(valid_set)
    test_set = shared_dataset(test_set)

    train_candcnt=shared_dataset(train_candcnt)
    valid_candcnt=shared_dataset(valid_candcnt)
    test_candcnt = shared_dataset(test_candcnt)

    terminalsize = len(prodata.terminalset)
    terminalEmbeding = theano.shared(
        np.asarray(
            rng.uniform(low=prodata.minv, high=prodata.maxv, size=[terminalsize,prodata.embedingsize]),
            dtype=theano.config.floatX),
        borrow=True
    )

    embeddings= T.concatenate([theano.shared(np.asarray(prodata.vocabulary)), terminalEmbeding], axis=0)


    # datasets = load_data(dataset)

    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]



    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.itensor3('x')   # the data is presented as rasterized images
    xlen = T.iscalar('xlen')
    # y = T.ivector('y')  # the labels are presented as 1D vector of
    #                     # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    # 201*25*41*40
    layer0_input = embeddings[x].dimshuffle(0,3,1,2)
    # terminalEmbeding[x].reshape(201,25,41,40)  wrong!
    # layer0_input = terminalEmbeding[x].reshape((prodata.kbest+1, prodata.embedingsize, prodata.maxsenlen+1, prodata.maxsenlen))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (41-5+1 , 40-5+1) = (37, 36)
    # maxpooling reduces this further to (37/2, 36/2) = (19, 18)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 19, 18)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(prodata.kbest+1, prodata.embedingsize, prodata.maxsenlen+1, prodata.maxsenlen),
        filter_shape=(nkerns[0], prodata.embedingsize, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (19-5+1, 18-5+1) = (15, 14)
    # maxpooling reduces this further to (15/2, 14/2) = (8, 7)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 8, 7)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(prodata.kbest+1, nkerns[0], 19, 18),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 8 * 7),
    # or (201, 50 * 8 * 7) = (201, 2800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 8 * 7,
        n_out=prodata.kbest+1,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    # layer3 = LogisticRegression(input=layer2.output, n_in=prodata.kbest+1, n_out=10)

    # the cost we minimize during training is the NLL of the model
    Ws = theano.shared(
        value=np.asarray(
            rng.uniform(low=prodata.minv, high=prodata.maxv, size=[prodata.kbest+1, 1]),
            dtype=theano.config.floatX
        ),
        name='Ws',
        borrow=True
    )

    score=T.dot(layer2.output, Ws)[0:xlen]
    best=T.argmax(score[1:],axis=0)
    cost=1+T.max(score[1:])-score[0]
    cost=T.mean(cost*(cost>0))
    # cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        best,
        givens={
            x: test_set[index],
            xlen: test_candcnt[index]
        }
    )

    validate_model = theano.function(
        [index],
        best,
        givens={
            x: valid_set[index],
            xlen: valid_candcnt[index]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer2.params + layer1.params + layer0.params
    params.append(Ws)
    params.append(terminalEmbeding)

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
            x: train_set[index],
            xlen: train_candcnt[index]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = -np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    myevalb=evalb()

    f=open('goldvalid.txt', 'wb')
    g=open('goldtest.txt','wb')
    for i in range(len(validtreestrs)):
        f.write(validtreestrs[i][0]+'\n')

    for i in range(len(testtreestrs)):
        g.write(testtreestrs[i][0]+'\n')

    g.close()
    f.close()

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_best = [validate_model(i)[0] for i in xrange(yu)]
                assert len(validation_best) == n_valid_batches
                this_validation_loss = getF1(validation_best, validtreestrs, myevalb, False)
                # this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss > best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss > best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_best = [test_model(i)[0] for i in xrange(n_test_batches)]
                    assert len(test_best) == n_test_batches
                    test_score= getF1(test_best, testtreestrs, myevalb, True)
                    # test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

def getF1(bestids, alltreestrs, myevalb, istest):
    tempfilename='temptest%d.txt'%np.random.uniform(1,10000,size=(1))
    f=open(tempfilename, 'wb')
    for i in range(len(bestids)):
        try:
            tree=alltreestrs[i][bestids[i]+1]
        except IndexError:
            # print bestids
            print i
            print bestids[i]
            print len(alltreestrs[i])
            raise Exception(len(alltreestrs))

        f.write(tree+'\n')
    f.close()
    f1=0.0
    if istest:
        f1=myevalb.fscore_extractor(myevalb.evaluate(tempfilename, 'goldtest.txt'))
    else:
        f1=myevalb.fscore_extractor(myevalb.evaluate(tempfilename, 'goldvalid.txt'))
    os.remove(tempfilename)
    return f1



if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
