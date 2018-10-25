#Aidan Fike
#October 23, 2018
#Comp 135
#
#Logistic regression class. This can be used to fit and predict data 
#from the mnist or titanic dataset.

import autograd.numpy as np
#import numpy as np
from autograd import grad
from autograd import elementwise_grad
from random import random, seed
from numpy.random import randint
import math
import time
import scipy.special

#The logistic loss function whose gradient is used to update the feature
#weights
#
#Params: wtVec - np.array of our feature weights
#        w0 - Additional constant used to calculate eta
#        trainInstMat - The (np.array) matrix filled with our training instances
#        trainLabelsArr - The corresponding true labels of our training instances
#                         in an np.array
#        lamb - lambda used in our regularizer
def Loss(wtVec, w0, trainInstMat, trainLabelsArr, lamb):
        regularizer = (lamb / 2.0) * (np.dot(wtVec, wtVec))
        
        etas = np.dot(trainInstMat, wtVec)
        etas = etas + w0

        likli = etas * trainLabelsArr
        
        likli -= np.where(etas < 30, np.log(1 + np.exp(etas)), etas)

        liksum = np.sum(likli)

        return regularizer - liksum

#Able to fit the titanic and mnist data sets using 
#gradient descent or stochastic gradient descent relying on a logistic loss 
#function. After fitting, the class also able to predict the labels of new data. 
#There are also methods used to find the optimal hyperparameters, and predict
#the accuracy of the classifier on new data
class MyLogisticReg:


    #Constructor for logistic regression class
    #
    #Params: dataSet - the dataSet the user wants tested. 'm' for mnist data,
    #                   't' for titanic data
    #        lamb - the hyperparameter lambda used in the loss function
    def __init__(self, dataSet, lamb):
        self.dataSet = dataSet
        self.lamb = lamb

        self.wt = []
        self.w0 = 0

        #Stochastic gradient descent methods
        self.numRandBatches = 10000
        self.batchSize = 10

        #Parameters dependent on the dataSet being fit. Includes stepsize and
        #epsilon for gradient descent and functions to read in the datasets
        #themselves
        if (dataSet == 'm'):
            self.stepSize = 0.00000001 
            self.epsilon = 0.000017
            self.decreaseInit = 10000
        elif (dataSet == 't'):
            self.stepSize = 0.00001
            self.epsilon = 0.002
            self.decreaseInit = 1
        else:
            print "Invalid Dataset!!!!"

        #Variables used to save the data from 100 steps previous 
        self.wtOld = []
        self.w0Old = 0


    #Fit the desired training data using logistic gradient descent
    #
    #Params: X - the np.array of training instances
    #        y - the np.array of training labels
    def fit(self, X, y):
        #Initialize the weights to small random numbers
        self.initRand(X.shape[1])

        step = 1
        highError = True

        startTime = time.time()

        stepOutput = open('normStep.txt', 'w')
        timeOutput = open('normTime.txt', 'w')

        #Gradient functions using the autograd package.
        d_LossWt = grad(Loss, 0)
        d_LossW0 = grad(Loss, 1)

        #Run gradient descent while the desired weights are still changing
        #significantly
        while (highError):
            #The gradients for the weights of eta at the current steps
            wtGrad = d_LossWt(self.wt, self.w0, X, \
                                        y, self.lamb)
            w0Grad = d_LossW0(self.wt, self.w0, X, \
                                        y, self.lamb)

            #Update the weights using the current gradient
            self.wt = self.wt - self.stepSize * wtGrad
            self.w0 = self.w0 - self.stepSize * w0Grad 

            #Every 100 steps determine if the gradient descent should
            #conclude becaue the weights are no longer significantly changing
            if step % 100 == 0:
                totFeatErr = 0
                for index, currFeat in enumerate(self.wt):
                    totFeatErr += math.fabs(currFeat - self.wtOld[index])
                totFeatErr += math.fabs(self.w0 - self.w0Old)
                diff = (1.0 / float(X.shape[1] + 1)) * totFeatErr

                stepOutput.write(str(step) + "      " + str(Loss(self.wt, self.w0,
                    X, y, self.lamb)) + "\n")
                timeOutput.write(str(time.time() - startTime) + "      " + \
                    str(Loss(self.wt, self.w0, X, y, self.lamb)) + "\n")

                if (diff < self.epsilon):
                    highError = False

                self.w0Old = self.w0
                self.wtOld = self.wt

            if step == 1:
                stepOutput.write(str(step) + "      " + str(Loss(self.wt, self.w0,
                    X, y, self.lamb)) + "\n")
                timeOutput.write(str(time.time() - startTime) + "      " + \
                    str(Loss(self.wt, self.w0, X, y, self.lamb)) + "\n")
            step += 1


        stepOutput.close()
        timeOutput.close()

    #Fit the current dataset using stochastic gradient descent
    #Params: X - The np.array of training instances
    #        y - The np.array of binary training labels
    def SGDfit(self, X, y):
        #Initialize the weights to small fractional numbers
        self.initRand(X.shape[1])

        step = 1
        highError = True
        randBatches = self.chooseRandomInsts(X)

        startTime = time.time()

        stepOutput = open('sgdStep.txt', 'w')
        timeOutput = open('sgdTime.txt', 'w')

        #Gradient functions using the autograd package.
        d_LossWt = grad(Loss, 0)
        d_LossW0 = grad(Loss, 1)

        #Run gradient descent while the desired weights are still changing
        #significantly
        while (highError):
            currInst = []
            currLabels = []

            #Choose the current training instances based on the indexes of the 
            #current batch of random numbers
            for index in randBatches[step % self.numRandBatches]:
                currInst.append(X[index])
                currLabels.append(y[index])

            currInst = np.array(currInst)
            currLabels = np.array(currLabels)

            #The gradients for the weights of eta at the current steps given
            #the current training data set
            wtGrad = d_LossWt(self.wt, self.w0, currInst, \
                                        currLabels, self.lamb)
            w0Grad = d_LossW0(self.wt, self.w0, currInst, \
                                        currLabels, self.lamb)

            #Update the weights using the current gradient
            self.wt = self.wt - self.stepSize * wtGrad
            self.w0 = self.w0 - self.stepSize * w0Grad 

            #Every 100 steps determine if the gradient descent should
            #conclude becaue the weights are no longer significantly changing
            if step % 100 == 0:
                totFeatErr = 0
                for index, currFeat in enumerate(self.wt):
                    totFeatErr += math.fabs(currFeat - self.wtOld[index])
                totFeatErr += math.fabs(self.w0 - self.w0Old)
                diff = (1.0 / float(X.shape[1] + 1)) * totFeatErr

                if step % self.numRandBatches == 0:
                    randBatches = chooseRandomInsts(X)
                currLoss = Loss(self.wt, self.w0, currInst, currLabels, self.lamb) *\
                                    float(X.shape[0])/10.0
                
                stepOutput.write(str(step) + "         " + str(currLoss) + "\n")
                timeOutput.write(str(time.time() - startTime) + "          " + \
                                                        str(currLoss) + "\n")

                if (diff < self.epsilon):
                    highError = False

                self.w0Old = self.w0
                self.wtOld = self.wt

            if step == 1:
                currLoss = Loss(self.wt, self.w0, currInst, currLabels, self.lamb) *\
                                    float(X.shape[0])/10.0
                stepOutput.write(str(step) + "         " + str(currLoss) + "\n")
                timeOutput.write(str(time.time() - startTime) + "          " + \
                                                        str(currLoss) + "\n")
                 
            step += 1

        stepOutput.close()
        timeOutput.close()
    

    #Predict the labels of given testing instances with the current weights
    #
    #Params: X - the np.array of testing instances
    #
    #Return: An np.array of the predicting training labels (made up of 8s and
    #9s for mnist dataset and 0 and 1s for titanic dataset)
    def predict(self, X):
        etas = np.dot(X, self.wt)
        etas = etas + self.w0

        y_pred = []

        if (self.dataSet == 'm'):
            for eta in etas:
                if eta < 0:
                    y_pred.append(8)
                else:
                    y_pred.append(9)
        elif (self.dataSet == 't'):
            for eta in etas:
                if eta < 0:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
        else:
            print "Invalid Dataset!!!!"

        y_pred = np.array(y_pred)
        return y_pred


    #Calculate the accuracy of a given set of predicted labels
    #
    #Params: y_test - the true labels of the testing data
    #        y_pred - the labels of the testing data
    #
    #Return: The accuracy of the predicted labels in a decimal for (i.e. 0.96)
    def evaluate(self, y_test, y_pred):
        accuracy = np.sum(np.equal(y_test, y_pred).astype(np.float)) / \
                                                                y_test.size
        return accuracy

    #Initialize the feature weights with small weights from -1:1
    #These weights are then scaled by a factor of 1 / self.decreaseInit to
    #prevent overflow issues in the loss function
    def initRand(self, numFeats):
        seed()
        self.wt = []
        self.w0 = 0
        for i in range(numFeats):
            self.wt.append((2 * random() - 1) / float(self.decreaseInit))
            self.w0 = (2 * random() - 1) / float(self.decreaseInit)

        self.wt = np.array(self.wt)
        self.wtOld = self.wt
        self.w0Old = self.w0

    #Create a 2D list with self.numRandBatches number of batches of batchSize.
    #These batches contain indexes meant to represent indexes of the
    #allInstances list
    #
    #Params: allInstances - A list of instances the batches will be made from
    #Return: The 2D list containing batches of indexes
    def chooseRandomInsts(self, allInstances):
        randBatches = []
        rand = []
        for i in range(self.numRandBatches):
            rand = randint(0, allInstances.shape[0], self.batchSize) 
            while len(rand) != len(set(rand)):
                rand = randint(0, allInstances.shape[0], self.batchSize) 
            randBatches.append(rand)

        return randBatches

    #Change the value of lambda, the coefficient for the regulizer
    #
    #Params: newLamb - the new desired value of lambda
    def setLamb(self, newLamb):
        self.lamb = newLamb

    #Give user the weight for the random feature (the last weight)
    def getRandFeatWeight(self):
        return self.wt[len(self.wt) - 1]
