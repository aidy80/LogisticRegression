import autograd.numpy as np
#import numpy as np
from autograd import grad
from autograd import elementwise_grad
from random import random, seed
import math
import scipy.special

def Lexp(wtVec, w0, trainInstMat, trainLabelsArr, lamb):
        normalizer = (lamb / 2.0) * (np.dot(wtVec, wtVec))
        
        etas = np.dot(trainInstMat, wtVec)
        #print wtVec
        #print trainInstMat
        #print etas
        etas = etas + w0

        likli = etas * trainLabelsArr
        
        likli -= np.where(etas < 30, np.log(1 + np.exp(etas)), etas)

        liksum = np.sum(likli)

        return normalizer - liksum

class MyLogisticReg:
    def __init__(self):
        self.allInstances = []
        self.allLabels = []
        self.trainInst = []
        self.trainLabels = []
        self.testInst = []
        self.testLabels = []
        self.wt = []
        self.w0 = 0

        self.numFolds = 5
        self.epsilon = .00001
        self.maxStep = 2000000

        self.wtOld = []
        self.w0Old = 0

        self.lamb = 1.0

        self.read_in_titanic_data()
        #self.read_in_mnist_data()
        self.initRand()

    def fit(self, X, y):
        step = 1
        highError = True

        #d_LsimpWt = grad(Lsimp, 0)
        #d_LsimpW0 = grad(Lsimp, 1)

        d_LexpWt = grad(Lexp, 0)
        d_LexpW0 = grad(Lexp, 1)
        while (step < self.maxStep and highError):
            #stepSize = max([1.0 / (1.0 + step), 0.000005])
            stepSize = 0.00001
            #stepSize = 1.0 / (1.0 + float(step))

            #below30 = createBinaryMask(self.wt, self.w0, self.trainInst)
            
            #wtSimpGrad = d_LsimpWt(self.wt, self.w0, self.trainInst, self.trainLabels, self.lamb)
            #w0SimpGrad = d_LsimpW0(self.wt, self.w0, self.trainInst, self.trainLabels, self.lamb)

            wtExpGrad = d_LexpWt(self.wt, self.w0, self.trainInst, \
                                        self.trainLabels, self.lamb)
            w0ExpGrad = d_LexpW0(self.wt, self.w0, self.trainInst, \
                                        self.trainLabels, self.lamb)
            #wtGrad = LgradWt(self.wt, self.w0, self.trainInst, self.trainLabels, self.lamb)
            #w0Grad = LgradW0(self.wt, self.w0, self.trainInst, self.trainLabels, self.lamb)

            #self.wt = self.wt - stepSize * wtGrad - self.lamb * self.wt
            #self.wt = self.wt - stepSize * wtSimpGrad
            #self.w0 = self.w0 - stepSize * w0SimpGrad 
            self.wt = self.wt - stepSize * wtExpGrad
            self.w0 = self.w0 - stepSize * w0ExpGrad 

            if step % 100 == 0:
                #print "step: ", step
                totFeatErr = 0
                for index, currFeat in enumerate(self.wt):
                    totFeatErr += math.fabs(currFeat - self.wtOld[index])
                totFeatErr += math.fabs(self.w0 - self.w0Old)
                diff = (1.0 / float(len(self.allInstances) + 1)) * totFeatErr
                #print "Curr Vec", self.wt[0], self.wt[len(self.wt) - 1], self.w0
                #print "diff:", diff
                #print "loss: ", Lexp(self.wt, self.w0, self.trainInst,\
                #                               self.trainLabels, self.lamb)

                if (diff < self.epsilon):
                    highError = False
                    print "highError = False!"

                self.w0Old = self.w0
                self.wtOld = self.wt

            step += 1
    
    def predict(self, X):
        etas = np.dot(X, self.wt)
        etas = etas + self.w0

        y_pred = []

        for eta in etas:
            if eta < 0:
                y_pred.append(1)
            else:
                y_pred.append(0)

        y_pred = np.array(y_pred)
        return y_pred

    def evaluate(self, y_test, y_pred):
        error_rate = np.sum(np.equal(y_test, y_pred).astype(np.float)) / \
                                                                y_test.size
        return 1 - error_rate

    def fold(self, foldNum):
        numTest = len(self.allInstances) / self.numFolds
        firstFoldInst = foldNum * numTest
        lastFoldInst = (foldNum + 1) * numTest

        #print "Len instances: ", len(self.allInstances)
        #print "Label instances: ", len(self.allLabels)

        for index, instance in enumerate(self.allInstances):
            if (index >= firstFoldInst and index < lastFoldInst):
                self.testInst.append(instance)
                self.testLabels.append(self.allLabels[index])
            else:
                self.trainInst.append(instance)
                self.trainLabels.append(self.allLabels[index])

        self.trainInst = np.array(self.trainInst)
        self.testInst = np.array(self.testInst)
        self.trainLabels = np.array(self.trainLabels)
        self.testLabels = np.array(self.testLabels)

    def read_in_mnist_data(self):
        with open('mnist-train.txt', 'r') as file:
            for line in file:
                pixels = line.split()
                if (pixels[0] != 'label'):
                    newInstance = []
                    for index, data in enumerate(pixels):
                        if (index == 0 and data == '8'):
                            self.allLabels.append(0)  
                        elif (index == 1 and data == '9'):
                            self.allLabels.append(1)  
                        else:
                            newInstance.append(float(data))

                    self.allInstances.append(newInstance)

    def read_in_titanic_data(self):
        with open('titanic_train.txt', 'r') as file:
            for line in file:
                features = line.split()
                if (features[0] == '0' or features[0] == '1'):
                    newInstance = []
                    for index, data in enumerate(features):
                        if (index == 0):
                            self.allLabels.append(int(data))
                        else:
                            newInstance.append(float(data))
                                                    #/maxFeatures[index - 1])
                    self.allInstances.append(newInstance)

    def initRand(self):
        seed()
        for i in range(len(self.allInstances[0])):
            self.wt.append(2 * random() - 1)
            self.w0 = 2 * random() - 1

        self.wt = np.array(self.wt)
        self.wtOld = self.wt
        self.w0Old = self.w0

    def main(self):
        for i in range(5):
            self.fold(i)
            self.fit(self.trainInst, self.trainLabels)
            y_pred = self.predict(self.testInst)
            #print y_pred
            #print self.testLabels
            acc = self.evaluate(self.testLabels, y_pred)
            print "Accuracy: ", acc
            print "Final weights: ", self.wt, self.w0
            self.trainInst = []
            self.trainLabels = []
            self.testInst = []
            self.testLabels = []

test = MyLogisticReg()
test.main()

def LgradWt(wtVec, w0, trainInstMat, trainLabelsArr, lamb):
    etas = np.matmul(trainInstMat, wtVec)
    etas = etas + w0

    gradWeights = trainLabelsArr - scipy.special.expit(etas)
    grad = np.transpose(np.matmul(np.transpose(trainInstMat), gradWeights))

    return grad * -1

def LgradW0(wtVec, w0, trainInstMat, trainLabelsArr, lamb):
    etas = np.matmul(trainInstMat, wtVec)
    etas = etas + w0

    gradWeights = trainLabelsArr - scipy.special.expit(etas)
    grad = np.sum(gradWeights)

    return grad * -1

def splitTrainTest(self, numTrain, numTest):
        totNumSplits = numTrain + numTest
        numInstPerSplit = len(allInstances) / totNumSplits  
        for index, instance in enumerate(allInstances):
            if (index < numInstPerSplit * numTrain):
                self.trainInst.append(instance)
                self.trainLabels.append(allLabels[index])
            else:
                self.testInst.append(instance)
                self.testLabels.append(allLabels[index])

        self.trainInst = np.array(self.trainInst)
        self.testInst = np.array(self.testInst)
        self.trainLabels = np.array(self.trainLabels)
        self.testLabels = np.array(self.testLabels)

"""
        maxFeatures = np.zeros(7)
        with open('titanic_train.txt', 'r') as file:
            for line in file:
                features = line.split()
                if (features[0] == '0' or features[0] == '1'):
                    for index, feature in enumerate(features):
                        if (index != 0):
                            if (float(feature) > maxFeatures[index - 1]):
                                maxFeatures[index - 1] = float(feature)
                            """

def createBinaryMask(wtVec, w0, trainInstMat):
    etas = np.matmul(trainInstMat, np.transpose(wtVec))
    below30 = []
    for eta in etas:
        if eta < 5:
            below30.append(True)
        else:
            below30.append(False)

    return below30

def Lsimp(wtVec, w0, trainInstMat, trainLabelsArr, lamb):
        normalizer = (lamb / 2.0) * (np.dot(wtVec, wtVec))
        
        etas = np.matmul(trainInstMat, np.transpose(wtVec))
        etas = etas + w0

        likli = etas * trainLabelsArr
        likli = likli - etas
        
        liksum = np.sum(likli)

        return normalizer - liksum
        #return -1 * liksum
