__author__ = 'tangyh'

import numpy as np
import re
import os
from parsing.evalb import evalb

import numpy as np

import theano
import theano.tensor as T
from theanoTutorial.mlp import HiddenLayer
from CNN import LeNetConvPoolLayer, shared_dataset
from dataProcess import PreProcess

class Mytest():

    def __init__(self, rng, tfilename, wefilename):
        self.rng=rng
        self.textfilename=tfilename
        self.wordembedfilename=wefilename
        self.vocabulary=[]
        self.vocabularyset=[]
        self.terminalset=[]
        self.maxv=-np.inf
        self.minv=np.inf
        self.embedingsize=None
        self.kbest=200
        self.maxsenlen=40
        self.terminalEmbeding=None
        self.paras=None


    def readWordEmbeding(self):
        try:
            with open(self.wordembedfilename) as inf:
                for line in inf:
                    if line:
                        embeds=line.split()
                        if embeds[0] not in self.vocabularyset:
                        # if not self.vocabulary.has_key(embeds[0]):
                            embedsv=np.array(embeds[1:],dtype=float)
                            if self.embedingsize==None:
                                self.embedingsize=len(embedsv)
                            else:
                                assert self.embedingsize==len(embedsv)
                            self.vocabulary.append(embedsv)
                            self.vocabularyset.append(embeds[0])
                            if self.minv>np.min(embedsv):
                                self.minv=np.min(embedsv)
                            if self.maxv<np.max(embedsv):
                                self.maxv=np.max(embedsv)

        except IOError:
            raise Exception("the word embedding file cannot be read")


def testfunction(prodata, valid_set, valid_candcnt):
    rng = np.random.RandomState(23455)
    nkerns=[20, 50]
    n_valid_batches = 1
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
    # y = T.ivector('y')  # the labels are presented as 1D vector of
    #                     # [int] labels
    xlen = T.iscalar('xlen')
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


    validate_model = theano.function(
        [index],
        score,
        givens={
            x: valid_set[index],
            xlen: valid_candcnt[index]
        }
    )

    validate_model2 = theano.function(
        [index],
        best,
        givens={
            x: valid_set[index],
            xlen: valid_candcnt[index]
        }
    )

    for i in xrange(n_valid_batches):
        print validate_model(i)
        print validate_model2(i)
    # validation_best = []

    # print validation_best
    # print n_valid_batches




# f=open(os.path.join(os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0], 'data/sample.txt'))
# a=f.readline()
#
# p1=re.compile(r'\t')
# p2=re.compile(r'(([TW]{1}) (\S+) (\d{1,2}) (\d{1,2}))')
#
# b=p1.split(a)
#
# c=p2.findall(b[0])
# print c
#
# imginverse=[[[] for i in range(26)]for j in range(27)]
# for x,y,z,u,v in c:
#     if y=='T':
#         imginverse[26-int(u)][int(v)]=z
#     if y=='W':
#         imginverse[0][int(v)]=z
#
# d=np.asarray(imginverse)
# np.savetxt('c6.txt',d,fmt="%s")

if __name__ == '__main__':
    # prodata= Mytest(np.random.RandomState(123),
    #          os.path.join(os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0], 'data/sample.txt'),
    #          os.path.join(os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0], 'data/embeddings1.txt')
    # )
    # print 'startingttttttttttttttttt'
    # prodata.readWordEmbeding()
    # print len(prodata.vocabulary)
    # print prodata.vocabulary[0].shape
    # print prodata.vocabularyset
    # print prodata.vocabularyset.index('to')
    # print prodata.embedingsize

    # myevalb=evalb()
    #
    # print myevalb.evaluate('/home/tangyh/Dropbox/PycharmProjects/CNNParsing/EVALB/gold.mrg',
    #                  '/home/tangyh/Dropbox/PycharmProjects/CNNParsing/EVALB/my.mrg')
    #
    # print myevalb.fscore_extractor(myevalb.evaluate('/home/tangyh/Dropbox/PycharmProjects/CNNParsing/EVALB/gold.mrg',
    #                  '/home/tangyh/Dropbox/PycharmProjects/CNNParsing/EVALB/my.mrg'))
    #
    # f=open('a.txt','wb')
    # f.write('i love u')
    # f.write('i love u again')
    # f.write('i love u again')
    # f.close()
    # os.remove('a.txt')


    # index = T.lscalar()  # index to a [mini]batch
    #
    # # start-snippet-1
    # x = T.itensor3('x')
    # img = T.matrix(dtype='int32')
    #
    # img=np.asarray(np.random.uniform(1,100,size=(201,41,40)))
    # alldata=[]
    # alldata.append(img)
    #
    # shared_x = theano.shared(np.asarray(alldata, dtype=theano.config.floatX), borrow=True)
    # sharedalldata = T.cast(shared_x, 'int32')
    #
    # # cost = x
    #
    #
    # rng = np.random.RandomState(23455)
    #
    # terminalEmbeding = theano.shared(
    #     np.asarray(
    #         rng.uniform(low=-1, high=1, size=[100,25]),
    #         dtype=theano.config.floatX),
    #     borrow=True
    # )
    #
    # nkerns=[20, 50]
    # layer0_input = terminalEmbeding[x].dimshuffle(0,3,1,2)
    #
    # layer0 = LeNetConvPoolLayer(
    #     rng,
    #     input=layer0_input,
    #     image_shape=(201, 25, 41, 40),
    #     filter_shape=(nkerns[0], 25, 5, 5),
    #     poolsize=(2, 2)
    # )
    #
    # layer1 = LeNetConvPoolLayer(
    #     rng,
    #     input=layer0.output,
    #     image_shape=(201, nkerns[0], 19, 18),
    #     filter_shape=(nkerns[1], nkerns[0], 5, 5),
    #     poolsize=(2, 2)
    # )
    #
    # layer2_input = layer1.output.flatten(2)
    #
    # layer2 = HiddenLayer(
    #     rng,
    #     input=layer2_input,
    #     n_in=nkerns[1] * 8 * 7,
    #     n_out=201,
    #     activation=T.tanh
    # )
    #
    # Ws = theano.shared(
    #     value=np.asarray(
    #         rng.uniform(low=-1, high=1, size=[201, 1]),
    #         dtype=theano.config.floatX
    #     ),
    #     name='Ws',
    #     borrow=True
    # )
    # score=T.dot(layer2.output, Ws)
    # best=T.argmax(score[1:],axis=0)
    # cost=1+T.max(score[1:])-score[0]
    # cost=cost*(cost>0)
    #
    #
    #
    # # cost = terminalEmbeding[x].reshape(201,25,41,40)
    # # cost = terminalEmbeding[x].dimshuffle(0,3,1,2)
    # # cost = terminalEmbeding[x]
    # train_model = theano.function([index], best,
    #     givens={
    #         x: sharedalldata[index]
    #     }
    # )
    #
    # params = layer2.params + layer1.params + layer0.params
    # params.append(Ws)
    # params.append(terminalEmbeding)
    #
    # # create a list of gradients for all model parameters
    # grads = T.grad(cost[0], params)
    # results1=train_model(0)
    # print results1
    # print results1.shape
    # print results1[1:]
    # print train_model(0)
    # print train_model(0).shape



    # prodata= PreProcess(np.random.RandomState(123),
    #          os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/embeddings.txt')
    # )
    #
    # valid_set, validtreestrs, valid_candcnt = prodata.finaldata(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/sample2.txt'))
    #
    #
    # valid_set=shared_dataset(valid_set)
    #
    # valid_candcnt=shared_dataset(valid_candcnt)
    #
    #
    # testfunction(prodata, valid_set, valid_candcnt)
    # myevalb=evalb()
    # f1=myevalb.fscore_extractor(myevalb.evaluate('temptest4846.txt', 'goldvalid.txt'))

    a=[1,2,3,4,5]
    # b=shared_dataset(a)
    # print b







