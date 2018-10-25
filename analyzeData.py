#Aidan Fike
#October 24, 2018
#Comp135 Machine Learning
#
# File to read through data, split it into testing and training and find
# qualities about Logistic classifier such as accuracy and best hyperparameters

import logReg

import autograd.numpy as np
#import numpy as np
from autograd import grad
from autograd import elementwise_grad
from random import random, seed
from numpy.random import randint
import math
import time
import scipy.special

#Convert all of the 0s in Mnist to 8s and all of the 1s in Mnist to 9s.
#Apply this to both testLabels and trainLabels
#
#Params: changeLabels - an np.array filled with the labels the user wants
#                       changed
def convertToBinary(changeLabels):
    newLabels = [] 
    labels = changeLabels.tolist()

    for label in labels:
        if (label == 8):
            newLabels.append(0)
        elif (label == 9):
            newLabels.append(1)
        else:
            newLabels.append(label)

    return np.array(newLabels)

#Class to parse through data and separate it into training and testing.
#Additionally, it has the ability to analyze the timing of the
#MyLogisticRegression class as well as discovering its accuracies and optimal
#hyperparameters
class AnalyzeData:
    def __init__(self, dataSet, initLamb):
        self.dataSet = dataSet
        self.lamb = initLamb

        #Training and testing data as well as weights
        self.allInstances = []
        self.allLabels = []
        self.trainInst = []
        self.trainLabels = []
        self.testInst = []
        self.testLabels = []

        if dataSet == 'm':
            self.read_in_mnist_data()
        elif dataSet == 't':
            self.read_in_titanic_data()
        else:
            print "Invalid Dataset!!!!"

        self.classifier = logReg.MyLogisticReg(dataSet, self.lamb)

    #Read in instances and labels from the file mnist-train.txt.
    def read_in_mnist_data(self):
        with open('mnist-train.txt', 'r') as file:
            for line in file:
                pixels = line.split()
                if (pixels[0] != 'label'):
                    newInstance = []
                    for index, data in enumerate(pixels):
                        if (index == 0 and data == '8'):
                            self.allLabels.append(8)  
                        elif (index == 0 and data == '9'):
                            self.allLabels.append(9)  
                        else:
                            newInstance.append(float(data))

                    self.allInstances.append(newInstance)

    #Read in instances and labels from the file titanic_train.txt
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


    #Make the training data be equivalent to all of the instances read in from 
    #the datasets
    def trainAllInst(self):
        self.trainInst = self.allInstances
        self.trainLabels = self.allLabels

        self.trainLabels = np.array(self.trainLabels)
        self.trainInst = np.array(self.trainInst)
        self.testInst = np.zeros(1)
        self.testLabels = np.zeros(1)

    #Separate the data into a given fold 
    #
    #Params: foldNum - The section of the data used for testing. There are 
    #                  numFolds number of possible folds so this integer
    #                  can be anything between 0 and numFolds - 1. 
    #                  0 will be the first fraction of data, 1 will be the 2nd
    #                  fraction, etc
    #        numFolds - The number of folds that the data set will eventually
    #                   be split into
    def fold(self, foldNum, numFolds):
        self.trainInst = []
        self.testInst = []
        self.trainLabels = []
        self.testLabels = []

        #Find the indexes of the instnaces where the test data should begin
        #and end
        numTest = len(self.allInstances) / numFolds
        firstFoldInst = foldNum * numTest
        lastFoldInst = (foldNum + 1) * numTest

        #Create the training and testing data
        for index, instance in enumerate(self.allInstances):
            if (index >= firstFoldInst and index < lastFoldInst):
                self.testInst.append(instance)
                self.testLabels.append(self.allLabels[index])
            else:
                self.trainInst.append(instance)
                self.trainLabels.append(self.allLabels[index])

        #Convert the training and testing instances into np arrays
        self.trainInst = np.array(self.trainInst)
        self.testInst = np.array(self.testInst)
        self.trainLabels = np.array(self.trainLabels)
        self.testLabels = np.array(self.testLabels)

    #Deperate instances into training and testing data with a ratio of
    #numTrain:numTest
    #
    #Params: numTrain - training side of the ratio
    #        numTest - testing side of the ratio
    def splitTrainTest(self, numTrain, numTest):
        totNumSplits = numTrain + numTest
        numInstPerSplit = len(self.allInstances) / totNumSplits  
        for index, instance in enumerate(self.allInstances):
            if (index < numInstPerSplit * numTrain):
                self.trainInst.append(instance)
                self.trainLabels.append(self.allLabels[index])
            else:
                self.testInst.append(instance)
                self.testLabels.append(self.allLabels[index])

        self.trainInst = np.array(self.trainInst)
        self.testInst = np.array(self.testInst)
        self.trainLabels = np.array(self.trainLabels)
        self.testLabels = np.array(self.testLabels)

    #Time the process of fitting all of the training data using normal gradient
    #descent
    def timeNorm(self):
        self.trainAllInst()
        if (self.dataSet == 'm'):
            self.classifier.fit(self.trainInst, convertToBinary(self.trainLabels))
        elif (self.dataSet == 't'):
            self.classifier.fit(self.trainInst, convertToBinary(self.trainLabels))


    #Time the process of fitting training data using stochastic 
    #gradient descent with a batchsize of 10
    def testSGD(self):
        self.classifier.SGDfit(np.array(self.allInstances),\
                    np.array(convertToBinary(np.array(self.allLabels))))
 
    #Find the best lambda for the given dataSet by splitting the data into a
    #7:3 training:testing ratio and finding the resulting accuracy.
    #
    #Params: possibleLambs - A list containing the lambdas the user would like
    #                        to test
    #        trainAcc - A list that will be filled with tuples containing
    #                   training accuracies and the random Feature weight 
    #                   corresponding to the possibleLambs
    #        testAcc - A list that will be filled with tuples containing
    #                   testing accuracies and the random Feature weight 
    #                   corresponding to the possibleLambs
    def findLamb(self, possibleLambs, trainAcc, testAcc):
        self.splitTrainTest(7,3)
        for lamb in possibleLambs:
            self.classifier.initRand(len(self.allInstances[0]))
            self.classifier.setLamb(lamb)
            self.classifier.fit(self.trainInst, convertToBinary(self.trainLabels))

            #Predict testing accuracies 
            y_pred = self.classifier.predict(self.testInst)
            acc = self.classifier.evaluate(self.testLabels, y_pred)
            testAcc.append((acc, self.classifier.getRandFeatWeight()))

            #Predict training accuracies 
            y_pred = self.classifier.predict(self.trainInst)
            acc = self.classifier.evaluate(self.trainLabels, y_pred)
            trainAcc.append((acc, self.classifier.getRandFeatWeight()))
            
    #Use numFolds cross validation to determine the mean and standard 
    #deviation accuracy of classifying a given dataset using logistic
    #regression
    #
    #Return: An array with the found accuracies of each fold
    def crossValid(self, numFolds):
        accuracies = []
        for i in range(numFolds):
            self.classifier.initRand(len(self.allInstances[0]))
            self.fold(i, numFolds)
            self.classifier.fit(self.trainInst, convertToBinary(self.trainLabels))
            y_pred = self.classifier.predict(self.testInst)
            acc = self.classifier.evaluate(self.testLabels, y_pred)
            accuracies.append(acc)

        return accuracies

    #Return the training labels
    def getTrainLabels(self):
        return self.trainLabels

    #Return the training instances
    def getTrainInsts(self):
        return self.trainInst
