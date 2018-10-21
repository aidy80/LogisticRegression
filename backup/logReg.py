import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad
#import numpy as np
from random import random, seed
import math

def L(wtVec, wo, trainInstMat, trainLabelsArr, lamb):
        normalizer = (lamb / 2.0) * (np.dot(wtVec, wtVec))
        
        etas = np.matmul(trainInstMat, np.transpose(wtVec))
        etas = etas + wo

        likli = etas * trainLabelsArr
        likli = likli - etas
        
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
        self.threshold = 0.3

        self.numFolds = 10
        self.epsilon = 10**(-3)
        self.maxStep = 400000

        self.wtOld = []
        self.w0Old = 0
        self.lamb = 1.0

        self.read_in_titanic_data()
        self.initRand()

    def fit(self, X, y):
        step = 1
        highError = True

        while (step < self.maxStep and highError):
            #stepSize = max([1.0 / (1.0 + step), 0.0001])
            stepSize = 1.0 / (1.0 + float(step))
            d_Lwt = grad(L, 0)
            d_Lw0 = grad(L, 1)

            wtGrad = d_Lwt(self.wt, self.w0, self.trainInst, self.trainLabels, self.lamb)
            w0Grad = d_Lw0(self.wt, self.w0, self.trainInst, self.trainLabels, self.lamb)

            self.wt -= stepSize * wtGrad
            self.w0 -= stepSize * w0Grad

            if step % 100 == 0:
                print "step: ", step
                totFeatErr = 0
                for index, currFeat in enumerate(self.wt):
                    totFeatErr += math.fabs(currFeat - self.wtOld[index])
                totFeatErr += math.fabs(self.w0 - self.w0Old)
                diff = (1.0 / float(len(self.allInstances) + 1)) * totFeatErr
                print "Curr Vec", self.wt[0], self.w0
                print "diff:", diff

                if (diff < self.epsilon):
                    highError = False
                    print "highError = False!"

                self.w0Old = self.w0
                self.wtOld = self.wt

            step += 1
    
    def predict(self, X):
        etas = etas + self.w0

        y_pred = []

        for eta in etas:
            if eta < 0:
                y_pred.append(0)
            else:
                y_pred.append(1)

        y_pred = np.array(y_pred)

    def evaluate(y_test, y_pred):
        error_rate = np.sum(np.equal(y_test, p_pred).astype(np.float)) / \
                                                                y_test.size
        return 1 - error_rate

    def fold(self, foldNum):
        numTest = len(self.allInstances) / self.numFolds
        firstFoldInst = foldNum * numTest
        lastFoldInst = (foldNum + 1) * numTest
        for index, instance in enumerate(self.allInstances):
            if (index >= firstFoldInst and index <= lastFoldInst):
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
                if (pixels[0] != 'labels'):
                    newInstance = []
                    for index, data in enumerate(features):
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
                    self.allInstances.append(newInstance)

    def initRand(self):
        seed()
        for i in range(len(self.allInstances[0])):
            self.wt.append(random())
            self.w0 = random()

        self.wt = np.array(self.wt)
        self.wtOld = self.wt
        self.w0Old = self.w0

    def main(self):
        self.fold(0)
        self.fit(self.trainInst, self.trainLabels)
        y_pred = self.predict(self.testInst)
        acc = self.evaluate(self.testLabels, y_pred)
        print "Accuracy: ", acc

test = MyLogisticReg()
test.main()
