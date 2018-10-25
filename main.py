#written by: Aidan Fike
#Oct 23, 2018
#
#Main function to test qualities of MyLogisticReg such as optimal lambda,
#training time, and predicted accuracy. Also gives ability to pickalize the
#classifier

from logReg import MyLogisticReg
from analyzeData import AnalyzeData
from analyzeData import convertToBinary
import pickle
import numpy as np
import math

#Create an mnist classifier and a titanic classifier and picklize them both
#into mnist_classifier.pkl and titanic_classifier.pkl. For section 4.3
def pickleIt():
    analyze = AnalyzeData('m', 100.0)
    analyze.trainAllInst()
    classifier = MyLogisticReg('m', 100.0)
    classifier.fit(analyze.getTrainInsts(), \
                            convertToBinary(analyze.getTrainLabels()))
    pickle_out = open("mnist_classifier.pkl", "w")        
    pickle.dump(classifier, pickle_out)
    pickle_out.close()

    print "Finished picklizing mnist"

    analyze = AnalyzeData('t', 0.01)
    analyze.trainAllInst()
    classifier = MyLogisticReg('t', 0.01)
    classifier.fit(analyze.getTrainInsts(), analyze.getTrainLabels())
    pickle_out = open("titanic_classifier.pkl", "w")        
    pickle.dump(classifier, pickle_out)
    pickle_out.close()

#Test many lambdas values on the mnist and titanic datasets and find the one
#which yields the best accuracies in each. Test both train and test accuracies
#for each
def testFindLamb():
    print "BEST LAMBDA TEST\n"
    lambs = [0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] #Lambdas to test

    #Output training and testing accuracies for mnist dataset
    testAcc = []
    trainAcc = []
    analyze = AnalyzeData('m', -1.0)
    analyze.findLamb(lambs, trainAcc, testAcc)
    with open("mnistTrainAcc.dat", "w") as mnist:
        for index, accWeight in enumerate(trainAcc):
            mnist.write(str(lambs[index]) + " " + str(accWeight) + "\n")
    with open("mnistTestAcc.dat", "w") as mnist:
        for index, accWeight in enumerate(testAcc):
            mnist.write(str(lambs[index]) + " " + str(accWeight) + "\n")

    print "Finished Mnist lambdas\n"

    #Output training and testing accuracies for titanic dataset
    testAcc = []
    trainAcc = []
    analyze = AnalyzeData('t', -1.0)
    analyze.findLamb(lambs, trainAcc, testAcc)
    with open("titTrainAcc.dat", "w") as titanic:
        for index, accWeight in enumerate(trainAcc):
            titanic.write(str(lambs[index]) + " accuracy, random feature weight" + str(accWeight) + "\n")
    with open("titTestAcc.dat", "w") as titanic:
        for index, accWeight in enumerate(testAcc):
            titanic.write(str(lambs[index]) + " accuracy, random feature weight" + str(accWeight) + "\n")

    print "Outputted Data now in titTrain/TestAcc.dat and \
                                                mnistTrain/TestAcc.dat"

#Find the amount of time it takes for stochastic gradient descent to fit
#training data and how long it takes for the normal gradient descent to fit all
#of the training instances.
def timingData():
    print "SGD/NORM TIMING TEST"

    analyze = AnalyzeData('m', 100.0)
    analyze.testSGD()
    analyze.timeNorm()
    print "Outputted data now in normStep/Time.dat and sgdStep/Time.dat"

#Run 5-fold cross validation on both the mnist dataset with a lambda of 100
#and on the titanic dataset with a lambda of 0.01. Output mean and standard 
#deviation of found accuracies
def findTrueAcc():
    analyze = AnalyzeData('m', 100.0)
    accuracies = analyze.crossValid(5)
    calcMeanStdev(accuracies, 'mnistAcc.dat')

    "Finished cross validation on mnist"

    analyze = AnalyzeData('t', 0.01)
    accuracies = analyze.crossValid(5)
    calcMeanStdev(accuracies, 'titAcc.dat')

    print "Outputted data now in mnistAcc.dat and titAcc.dat"

#Calculate the mean and stdev of a list of accuracies
def calcMeanStdev(accuracies, filename):
    meanAcc = np.sum(np.array(accuracies)) / float(len(accuracies))
    stdev = 0
    for acc in accuracies:
        stdev += (acc - meanAcc)**2
    stdev = math.sqrt(stdev / len(accuracies))
    with open(filename, "w") as titanic:
        titanic.write("Accuracy: " +  str(meanAcc) + ", stdev: " + str(stdev) + "\n")

#Test that the pickling process was successful
def testPickle():
    testData = AnalyzeData('t', 0.01)
    testData.splitTrainTest(5,5)
    pickleIn = open('titanic_classifier.pkl', 'rb')
    classifier = pickle.load(pickleIn)
    y_pred = classifier.predict(testData.getTrainInsts()) 
    accuracy = classifier.evaluate(testData.getTrainLabels(), y_pred)

    print "Titanic accuracy: ", accuracy
    
    testData = AnalyzeData('m', 0.01)
    testData.splitTrainTest(5,5)
    pickleIn = open('mnist_classifier.pkl', 'rb')
    classifier = pickle.load(pickleIn)
    y_pred = classifier.predict(testData.getTrainInsts()) 
    accuracy = classifier.evaluate(testData.getTrainLabels(), y_pred)

    print "Mnist accuracy: ", accuracy
    


#Main functions! Sections 4.1, 4.2, 4.3, 4.4 in order

timingData()
#testFindLamb()
pickleIt()
#findTrueAcc()

testPickle()
