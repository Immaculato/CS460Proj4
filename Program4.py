#Tristan Basil
#Assignment: Project 4 - cS460G Machine Learning, Dr. Harrison

#https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.matrix.html
#https://stackoverflow.com/questions/4455076/how-to-access-the-ith-column-of-a-numpy-multidimensional-array
#https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range


import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import copy
import random

class Node:
    weights = list()
    forwardNodes = list()

    def __init__(self):
        self.weights = list()
        self.forwardNodes = list()


#this class is only designed to work for the data in this project.
class MultilayerPerceptron:
    hiddenLayers = -1
    debug = False
    inputNodes = list()
    inputs = list()
    classLabelList = list()
    distinctClassLabels = set()
    weightsMatrices = list()
    nodesLayersList = list()
    inputVectorsAtLayerList = list()
    activationVectorsAtLayerList = list()
    inputMatrix = None
    numEpochs = 10
    alpha = 0.1
    numHiddenNodes = 50
    numInputs = 784

    '''
    def __debugPrint(self):
        if self.debug:
            print " --- DEBUG INFO --- "
            print "class labels: " + str(self.distinctClassLabels)
            for i in range(len(self.featureLists)):
                print "max feature " + str(i) + ": " + str(max(self.featureLists[i]))
                print "min feature " + str(i) + ": " + str(min(self.featureLists[i]))
                print "discretized bins feature 1:",
                for j in range(len(self.binLists[i])):
                    print(self.binLists[i][j]),
                    print "-",
                print ""
            print " --- END DEBUG --- "
    '''

    #initialization takes a filename.
    def __init__(self, filename, hiddenLayers, debug):
        self.debug = debug
        self.hiddenLayers = hiddenLayers
        file = None
        try:
            file = open(filename, "r")
        except:
            invalidNumFields = True
            print(filename, 'not found')
            exit(-1)

        #for each line in the file, parse the inputs and class labels into parallel lists.
        lineIndex = 0
        for line in file:
            parsedLine = line.rstrip().split(',')
            self.classLabelList.append(int(parsedLine[0]))
            self.inputs.append(list())
            for i in range(1, len(parsedLine)):
                self.inputs[lineIndex].append(int(parsedLine[i]))
            #self.distinctClassLabels.add(classLabel)
            lineIndex+=1

        self.inputMatrix = np.matrix(self.inputs)

        #build the tree from the output up.
        outputNode = Node()
        #create a temporary list to hold weights to build matrices later.
        weightsLists = list()
        for i in range(hiddenLayers+1):
            weightsLists.append(list())
            self.nodesLayersList.append(list())
            #just give these vectors dummy objects so we don't throw errors later, when storing inputs between each layer.
            self.inputVectorsAtLayerList.append(None)
            self.activationVectorsAtLayerList.append(None)

        
        #create nodes for the last layer and all internal layers.
        for i in range(hiddenLayers, 0, -1):
            #create 10 nodes
            for j in range(self.numHiddenNodes):
                #for the last layer, tie each node to the output node.
                if i == hiddenLayers:
                    self.nodesLayersList[i].append(Node())
                    self.nodesLayersList[i][j].forwardNodes.append(outputNode)
                    self.nodesLayersList[i][j].weights.append(random.uniform(0.00001, 0.0001))
                    weightsLists[i].append(self.nodesLayersList[i][j].weights)
                #for internal layers, tie each node to the next set of internal nodes, except the bias node.
                else:
                    self.nodesLayersList[i].append(Node())
                    self.nodesLayersList[i][j].forwardNodes = self.nodesLayersList[i+1]
                    #we're gonna need 10 weights per node here.
                    for k in range(self.numHiddenNodes):
                        self.nodesLayersList[i][j].weights.append(random.uniform(0.00001, 0.0001))
                    weightsLists[i].append(self.nodesLayersList[i][j].weights)

        '''
        #make a fixed amount of hidden nodes that point to this output
        weightsList2 = list()
        hiddenNodes = list()
        for i in range(10):
            hiddenNodes.append(Node())
            hiddenNodes[i].forwardNodes.append(outputNode)
            hiddenNodes[i].weights.append(0.0)
            weightsList2.append(hiddenNodes[i].weights)
        '''
        #make the input nodes, then connect each input node to the earliest nodeLayer
        for i in range(self.numInputs):
            self.nodesLayersList[0].append(Node())
            for j in range(self.numHiddenNodes):
                self.nodesLayersList[0][i].forwardNodes.append(self.nodesLayersList[1][j])
                self.nodesLayersList[0][i].weights.append(random.uniform(0.00001, 0.0001))
            weightsLists[0].append(self.nodesLayersList[0][i].weights)

        '''
        #make the input nodes, then connect each input node to each hidden node
        for i in range(self.numInputs):
            self.inputNodes.append(Node())
            for j in range(10):
                self.inputNodes[i].forwardNodes.append(hiddenNodes[j])
                self.inputNodes[i].weights.append(0.0)
            weightsList1.append(self.inputNodes[i].weights)
        '''

        #build the weight matrix for each pair of layers.
        for i in range(hiddenLayers+1):
            self.weightsMatrices.append(np.matrix(weightsLists[i]).T)
            print self.weightsMatrices[i]
        '''
        self.weightsMatrix2 = np.matrix(weightsList2).T
        self.weightsMatrix1 = np.matrix(weightsList1).T
        '''

        #print '0', self.weightsMatrices[0]
        #print '0', self.weightsMatrices[0].shape
        #print '1', self.weightsMatrices[1]
        #print '1', self.weightsMatrices[1].shape

    def train(self, alpha, numEpochs):
        self.alpha = alpha
        self.numEpochs = numEpochs
        for epoch in range(self.numEpochs):
            #for each input
            print 'On epoch', epoch+1, 'out of', self.numEpochs
            for i in range(len(self.inputs)):
                print 'before'
                #print 'on input', i
                #get the prediction and error for the example
                prediction = self.__prediction(self.inputs[i])
                error = self.classLabelList[i]-prediction
                print 'on input', i, 'error', error
                #calculate DeltaJ for the output layer.
                inputVector = self.inputVectorsAtLayerList[self.hiddenLayers]
                #print 'before', inputVector
                #inputVector = self.weightsMatrices[self.hiddenLayers+1]*np.matrix(self.inputVectorsAtLayerList[self.hiddenLayers+1]).T
                vectorSize = len(inputVector)
                #print 'before delta change', inputVector 
                for j in range(vectorSize):
                    inputVector[j, 0] = self.__deltaOutputLayer(inputVector[j, 0], error)
                #inputVector now contains the deltaJ values in a vector. store it to another one with a better name so my brain doesn't explode.
                upperInputVector = inputVector
                #print 'after', inputVector

                #for each weight in the layer before the output, do the updates.
                weightsCount = self.weightsMatrices[self.hiddenLayers].shape[1]
                #print weightsCount
                upperWeightsCount = self.weightsMatrices[self.hiddenLayers].shape[0]
                #print 'before updates', self.weightsMatrices[self.hiddenLayers]
                
                for k in range(weightsCount):
                    for l in range(upperWeightsCount):
                        #this needs to use the activation value, NOT the actual value.
                        #print self.weightsMatrices[j][k][l]
                        #print self.alpha
                        #print 'hello world', (self.activationVectorsAtLayerList[self.hiddenLayers][k,0]*upperInputVector[l])
                        #print 'here'
                        #print 'activation', self.activationVectorsAtLayerList[self.hiddenLayers-1][k,0]
                        #print 'deltaj', upperInputVector[l]
                        #print 'weight to update', self.weightsMatrices[self.hiddenLayers][l,k]
                        self.weightsMatrices[self.hiddenLayers][l,k] = self.weightsMatrices[self.hiddenLayers][l,k] + (self.alpha*self.activationVectorsAtLayerList[self.hiddenLayers-1][k,0]*upperInputVector[l])
                #print 'after updates', self.weightsMatrices[self.hiddenLayers]

                #print 'final weights matrix:\n', self.weightsMatrices[self.hiddenLayers]
                
                #for each layer before the output layer going backwards,
                for j in range(self.hiddenLayers-1, -1, -1):
                    #for each node in that layer,
                    internalInputVector = self.inputVectorsAtLayerList[j]
                    #print 'internalInputVector', internalInputVector
                    #print 'upper deltaJvector', upperInputVector
                    vectorSize = len(internalInputVector)
                    print 'vectorSize', vectorSize
                    for l in range(vectorSize):
                        internalInputVector[l, 0] = self.__deltaInternalLayer(internalInputVector[l, 0], l, j, upperInputVector)
                    print 'after'
                    #now, the upper input vector is the vector we just calculated.
                    upperInputVector = internalInputVector
                    #for each weight in this layer, do the updates.
                    weightsCount = self.weightsMatrices[j].shape[1]
                    upperWeightsCount = self.weightsMatrices[j].shape[0]
                    if j > 0:
                        for k in range(weightsCount):
                            for l in range(upperWeightsCount):
                                #this needs to use the activation value, NOT the actual value.
                                #print self.weightsMatrices[j][k][l]
                                #print upperInputVector[l]
                                #print self.activationVectorsAtLayerList[j][k,0]
                                #print k, l
                                #print self.weightsMatrices[j].shape
                                #print self.weightsMatrices[j]
                                #print self.weightsMatrices[j][l,k]
                                #print self.activationVectorsAtLayerList[j].shape
                                self.weightsMatrices[j][l,k] = self.weightsMatrices[j][l,k] + (self.alpha*self.activationVectorsAtLayerList[j-1][k,0]*upperInputVector[l])
                    elif j==0:
                        #print 'before\n', self.weightsMatrices[j]
                        example = self.inputs[i]
                        #print 'start'
                        for k in range(weightsCount):
                            for l in range(upperWeightsCount):
                                #this needs to use the activation value, NOT the actual value.
                                #print self.weightsMatrices[j][k][l]
                                #print upperInputVector[l]
                                #print self.activationVectorsAtLayerList[j][k,0]
                                #print k, l
                                #print self.weightsMatrices[j].shape
                                #print self.weightsMatrices[j]
                                #rint self.weightsMatrices[j][l,k]
                                #print self.activationVectorsAtLayerList[j].shape
                                #print 'activation', example[k]
                                #print 'deltaj', upperInputVector[l]
                                #print 'weight to update', self.weightsMatrices[j][l,k]
                                self.weightsMatrices[j][l,k] = self.weightsMatrices[j][l,k] + (self.alpha*(example[k]*upperInputVector[l]))
                        #print 'end'
                        #print 'after', self.weightsMatrices[j]

                    

                


    #get the prediction for a given input index.
    def __prediction(self, inputList):
        #print 'matrix 1 shape', self.weightsMatrix1.shape
        #print 'matrix 2 shape', self.weightsMatrix2.shape
        inputVector = np.matrix(inputList).T
        for j in range(self.hiddenLayers+1):
            #print 'between layers', j, j+1, self.weightsMatrices[j].shape
            #print 'between layers', j, j+1, inputVector.shape
            inputVector = self.weightsMatrices[j]*inputVector
            self.inputVectorsAtLayerList[j] = copy.deepcopy(inputVector)
            vectorSize = len(inputVector)
            #run sigmoid on the vector.
            for i in range(vectorSize):
                inputVector[i, 0] = self.__sigmoid(inputVector[i, 0])
            self.activationVectorsAtLayerList[j] = copy.deepcopy(inputVector)


        #print 'input at 0', self.inputVectorsAtLayerList[0]
        #print 'input at 1', self.inputVectorsAtLayerList[1]
        #print 'activation at 0', self.activationVectorsAtLayerList[0]
        #print 'activation at 1', self.activationVectorsAtLayerList[1]
        return inputVector[0, 0]


        '''
        inputVector = self.weightsMatrix1*np.matrix(self.inputs[inputIndex]).T
        vectorSize = len(inputVector)

        #run sigmoid on the input vector.
        for i in range(vectorSize):
            inputVector[i, 0] = self.__sigmoid(inputVector[i, 0])

        #now, run these inputs through the next layer to get the final answer.
        inputVector = self.weightsMatrix2*inputVector
        vectorSize = len(inputVector)
        for i in range(vectorSize):
            inputVector[i, 0] = self.__sigmoid(inputVector[i, 0])

        return inputVector[0, 0]
        '''

    def __sigmoid(self, inputVal):
        try:
            return 1.0/(1+np.e**-inputVal)
        except:
            #print inputVal
            return 0.000001


    def __deltaOutputLayer(self, inputVal, error):
        sigmoid = self.__sigmoid(inputVal)
        return (sigmoid*(1-sigmoid)) * (error)

    def __deltaInternalLayer(self, inputVal, toNode, layer, deltaTo):
        #print 'inputVal', inputVal
        sigmoid = self.__sigmoid(inputVal)
        #print len(self.weightsLists[layer+1])
        #for the number of nodes in the above layer,
        #test = self.weightsMatrices[layer+1][:,toNode].T*deltaTo
        #print 'delta to first element', deltaTo[0, 0]
        #print 'inputVal', inputVal
        #print 'sigmoid', sigmoid
        jSum = 0.0
        itojWeightsVector = self.weightsMatrices[layer][toNode,:].T
        weightsVectorSize = len(itojWeightsVector)
        for i in range(weightsVectorSize):
            jSum+=itojWeightsVector[i]*deltaTo
        #print 'weightDeltaJSUm', jSum
        return (sigmoid*(1-sigmoid)) * jSum


        '''
        #make a list of the example indexes to start ID3 (all of them)
        examplesIndexList = list()
        for i in range(len(self.classLabelList)):
            examplesIndexList.append(i)
        #set a root node
        self.rootTreeNode = Node()
        #kick off ID3!
        self.__ID3(examplesIndexList, self.unbranchedFeatures, self.rootTreeNode, 0)
        '''

    def runTestExamples(self, filename):
        try:
            file = open(filename, "r")
        except:
            invalidNumFields = True
            print(filename, 'not found')
            exit(-1)

        #for each line in the file, parse the inputs and class labels into parallel lists.
        testClassLabelList = list()
        testInputs = list()
        lineIndex = 0
        for line in file:
            parsedLine = line.rstrip().split(',')
            testClassLabelList.append(int(parsedLine[0]))
            testInputs.append(list())
            for i in range(1, len(parsedLine)):
                testInputs[lineIndex].append(int(parsedLine[i]))
            #self.distinctClassLabels.add(classLabel)
            lineIndex+=1

        #get the cumulative accuracy of the network on the test data.
        numInputs = len(testInputs)
        successes = 0
        total=0
        for i in range(numInputs):
            prediction = int(round(self.__prediction(testInputs[i])))
            if prediction == testClassLabelList[i]:
                successes+=1
            total+=1

        print 'Final accuracy:', successes, 'out of', total, '- Percentage:', float(successes)/total, '%'


    

def main():
    random.seed(1)
    if (len(sys.argv) != 3):
        print "Takes 2 command line argument: the name of the csv training file, and the name of the csv test file."
        exit(-1)
    trainingFilename = sys.argv[1]
    testFilename = sys.argv[2]
    isDebugMode = False
    hiddenLayers = 1
    alpha =0.5
    numEpochs = 1
    #initialize the network
    neuralNet = MultilayerPerceptron(trainingFilename, hiddenLayers, isDebugMode)
    neuralNet.runTestExamples(testFilename)
    neuralNet.train(alpha, numEpochs)
    #neuralNet.runTestExamples(testFilename)




main()