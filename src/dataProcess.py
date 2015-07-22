__author__ = 'tangyh'

import numpy as np
import os
import re
import theano
import theano.tensor as T

class PreProcess(object):

    def __init__(self, rng, wefilename):
        self.rng=rng
        # self.textfilename=tfilename
        self.wordembedfilename=wefilename
        print wefilename
        self.vocabulary=[]
        self.vocabularyset=[]
        self.nonofwords=0
        self.terminalset=[]
        self.maxv=-np.inf
        self.minv=np.inf
        self.embedingsize=None
        self.kbest=200
        self.maxsenlen=40

        self.readWordEmbeding()

    def readTreeToArray(self, textfilename):
        ### we assume the data structure is like this:
        ### dates       SFG   GTS   EIG
        ### 31-Jan-2000 20.10 21.00 56.23
        ### 29-Feb-2000 20.50 27.00 57.23
        ### 31-Mar-2000 20.60 20.00 59.23
        # p1=re.compile(r'(\{([^\{\}]*)\})')
        # p2=re.compile(r'([^\[\]\s]+)\[(\d+) (\d+)\];')
        p0=re.compile(r'\t\t')
        p1=re.compile(r'\t')
        p2=re.compile(r'(([TW]{1}) (\S+) (\d{1,2}) (\d{1,2}))')


        # a=p1.split(sample)
        # assert len(a)<202
        # b=p2.findall(a[0])
        # assert len(b)==len(a[0].split())/4
        # assert(b[0])==5
        allTrees=[]
        allTreeStrs=[]
        try:
            with open(textfilename) as inf:
                for line in inf:
                    if line:
                        trees=[]
                        treeStrs=[]
                        length=0
                        candidates=p0.split(line)[:-1]
                        assert len(candidates)<self.kbest+2
                        candcnt = len(candidates)
                        for candidate in candidates:
                            treeString,nodescandidate=p1.split(candidate)
                            nodes=p2.findall(nodescandidate)
                            assert len(nodes)==len(nodescandidate.split())/4
                            for x, y, z, u, v in nodes:
                                if y=='T':
                                    if z not in self.terminalset:
                                        # print z
                                        self.terminalset.append(z)
                                elif y=='W':
                                    pass
                                    # if z not in self.vocabularyset:
                                    #     print 'word is not in vocabularyset', z
                                        # z= self.change_word(z)
                                else:
                                    # print self.vocabulary.has_key(z)
                                    print x,y
                                    raise Exception('y is neither W nor T ')
                                # tree[z]=(int(y), int(z))
                                if length<int(v): length=int(v)

                            trees.append(nodes)
                            treeStrs.append(treeString)
                        allTrees.append((candcnt,length+1,trees))
                        allTreeStrs.append(treeStrs)
            # print self.terminalset
            # self.terminalset=list(set(self.terminalset))
            assert len(set(self.terminalset)) == len(self.terminalset)
            # print self.terminalset
        except IOError:
            raise Exception("the treebank file cannot be read")

        return allTrees, allTreeStrs

    def readTextToArray(self, textfilename):
        ### we assume the data structure is like this:
        ### dates       SFG   GTS   EIG
        ### 31-Jan-2000 20.10 21.00 56.23
        ### 29-Feb-2000 20.50 27.00 57.23
        ### 31-Mar-2000 20.60 20.00 59.23
        # p1=re.compile(r'(\{([^\{\}]*)\})')
        # p2=re.compile(r'([^\[\]\s]+)\[(\d+) (\d+)\];')
        p0=re.compile(r'\t\t')
        p1=re.compile(r'\t')
        p2=re.compile(r'(([TW]{1}) (\S+) (\d{1,2}) (\d{1,2}))')


        # a=p1.split(sample)
        # assert len(a)<202
        # b=p2.findall(a[0])
        # assert len(b)==len(a[0].split())/4
        # assert(b[0])==5
        allTrees=[]
        try:
            with open(textfilename) as inf:
                for line in inf:
                    if line:
                        trees=[]
                        length=0
                        candidates=p1.split(line)[:-1]
                        # print len(candidates)
                        assert len(candidates)<self.kbest+2
                        for candidate in candidates:
                            # tree={}
                            nodes=p2.findall(candidate)
                            assert len(nodes)==len(candidate.split())/4
                            for x, y, z, u, v in nodes:
                                if y=='T':
                                    if z not in self.terminalset:
                                        # print z
                                        self.terminalset.append(z)
                                elif y=='W':
                                    pass
                                    # if z not in self.vocabularyset:
                                    #     print 'word is not in vocabularyset', z
                                        # z= self.change_word(z)
                                else:
                                    # print self.vocabulary.has_key(z)
                                    print x,y
                                    raise Exception('y is neither W nor T ')
                                # tree[z]=(int(y), int(z))
                                if length<int(v): length=int(v)

                            trees.append(nodes)
                        allTrees.append((length+1,trees))
            # print self.terminalset
            # self.terminalset=list(set(self.terminalset))
            assert len(set(self.terminalset)) == len(self.terminalset)
            # print self.terminalset
        except IOError:
            raise Exception("the treebank file cannot be read")

        return allTrees


    def change_word(self, word):
        if word =="-lsb-" or word=="-lcb-" or word=="-lrb-":
            v= "("
        elif word=="-rsb-" or word=="-rcb-" or word=="-rrb-":
            v= ")"
        elif word.lower() in self.vocabularyset:
            v = word.lower()
        elif word.capitalize() in self.vocabularyset:
            v = word.capitalize()
        elif word.isdigit():
            v = '1995'
        else:
            #print('Warning: the word %s is not in the look up table' % word)
            v = '*UNKNOWN*'

        return v

    def readWordEmbeding(self):
        try:
            with open(self.wordembedfilename) as inf:
                for line in inf:
                    if line:
                        embeds=line.split()
                        if embeds[0].strip() not in self.vocabularyset:
                        # if not self.vocabulary.has_key(embeds[0]):
                            embedsv=np.array(embeds[1:], dtype=theano.config.floatX)
                            if self.embedingsize==None:
                                print 'embedingsssssssssss'
                                self.embedingsize=len(embedsv)
                                # insert all zeros token embedding for the blank(0) cells
                                self.vocabularyset.append('000')
                                self.vocabulary.append(np.zeros((self.embedingsize),dtype=theano.config.floatX))
                                self.nonofwords+=1
                            else:
                                assert self.embedingsize==len(embedsv)
                            self.vocabulary.append(embedsv)
                            self.nonofwords+=1
                            self.vocabularyset.append(embeds[0].strip())
                            if self.minv>np.min(embedsv):
                                self.minv=np.min(embedsv)
                            if self.maxv<np.max(embedsv):
                                self.maxv=np.max(embedsv)

        except IOError:
            raise Exception("the word embedding file cannot be read")

    def finaldata(self, textfilename):
        alltrees, alltreestrs=self.readTreeToArray(textfilename)
        alldata=[]
        candcnts=[]
        for candcnt,length,trees in alltrees:
            assert length<=40
            img=np.asarray(np.zeros((self.kbest+1,self.maxsenlen+1,self.maxsenlen),dtype=int))
            # print img.shape
            # img=[[[[] for k in range(40)] for i in range(41)]for j in range(201)]
            # self.kbest* self.maxsenlen* self.maxsenlen*self.embedingsize: 200*40*40*25
            for treei in range(len(trees)):
                for x,y,z,u,v in trees[treei]:
                    if y=='T':
                        try:
                            img[treei, length-int(u), int(v)]=self.nonofwords+self.terminalset.index(z)
                        except ValueError:
                            raise Exception('T',treei, length-int(u), int(v), z)
                        # img[treei][int(v)][int(u)+1]=self.terminalEmbeding[self.terminalset.index(z),]
                    elif y=='W':
                        try:
                            img[treei, 0, int(v)]=self.vocabularyset.index(z)
                        except ValueError:
                            img[treei, 0, int(v)]=self.vocabularyset.index(self.change_word(z))
                            # raise Exception('W', treei, int(v), z)
                    else:
                        raise Exception('y is neither W nor T in function finaldata')
                # if treei==0:
                #     print img[0][0][25]
                #     print img[0][0][2]
                #     print img[0][0][0]
                #     print img[0][0][1]
            alldata.append(img)
            candcnts.append(candcnt)
        assert len(alldata) == len(alltreestrs)
        return alldata, alltreestrs, candcnts


if __name__ == '__main__':
    # print os.path.abspath(__file__)
    # print os.path.dirname(os.path.abspath(__file__))
    # print os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
    # print os.path.dirname(os.path.abspath(__file__))
    # print os.path.join(os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0], 'data/sample.txt')
    prodata= PreProcess(np.random.RandomState(123),
             os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/embeddings.txt')
    )

    print prodata.vocabularyset[0]
    print prodata.vocabulary[0]
    print prodata.vocabulary[1]
    # train_set = prodata.finaldata(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'data/train.txt'))

    # prodata.readWordEmbeding()
    # prodata.initialTerminalEmbeding()
    # print prodata.maxv,prodata.minv, prodata.embedingsize, len(prodata.vocabulary)
    # print prodata.vocabulary['Corera']
    # prodata.readTextToArray()












