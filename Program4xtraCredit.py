#Tristan Basil
#Assignment: Project 4 - cS460G Machine Learning, Dr. Harrison

#https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.matrix.html -> for using a matrix in general
#https://stackoverflow.com/questions/4455076/how-to-access-the-ith-column-of-a-numpy-multidimensional-array -> for getting a vector from a matrix
#https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range -> for generating random weights


import numpy as np
import sys
import copy
import random

#this class is only designed to work for the data in this project.
class MultilayerPerceptron:
    hiddenLayers = -1
    debug = False
    inputNodes = list()
    inputs = list()
    classLabelList = list()
    weightsMatrices = list()
    nodesLayersList = list()
    inputVectorsAtLayerList = list()
    activationVectorsAtLayerList = list()
    biasNodesList = list()
    numEpochs = 10
    alpha = 0.1
    numHiddenNodes = 10
    numInputs = 784

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
                self.inputs[lineIndex].append(float(parsedLine[i])/255.0)
            lineIndex+=1

        #build the tree from the output up.
        #create a temporary list to hold weights to build matrices later.
        weightsLists = list()
        for i in range(hiddenLayers+1):
            weightsLists.append(list())
            self.nodesLayersList.append(list())
            #just give these vectors dummy objects so we don't throw errors later, when storing inputs between each layer.
            self.inputVectorsAtLayerList.append(None)
            self.activationVectorsAtLayerList.append(None)
            self.biasNodesList.append(list())
        #to make things simpler, just add another bias layer element so that the layers line up later on.
        self.biasNodesList.append(list())

        #add bias nodes for the output layer.
        for i in range(5):
            self.biasNodesList[2].append(random.uniform(-0.1, 0.1))

        #create nodes for the last layer and all internal layers.
        for i in range(hiddenLayers, 0, -1):
            #create 10 nodes
            for j in range(self.numHiddenNodes):
                #for the last layer, tie each node to each output node.
                if i == hiddenLayers:
                    weightsList = list()
                    for k in range(5):
                        weightsList.append(random.uniform(-0.1, 0.1))
                    self.biasNodesList[i].append(random.uniform(-0.1, 0.1))

                #for internal layers, we'll need bias nodes.
                else:
                    #we're gonna need more weights per node here.
                    weightsList = list()
                    for k in range(self.numHiddenNodes):
                        #weightsList = list()
                        weightsList.append(random.uniform(-0.1, 0.1))
                        self.biasNodesList[i].append(random.uniform(-0.1, 0.1))
                #add an extra weight for the bias node at the each internal and the output layer
               
                weightsLists[i].append(weightsList)

        if self.debug:
            print 'biasnodelist1', self.biasNodesList[1]
            print 'biasnodelist2', self.biasNodesList[2]
                
        #make the input nodes, then connect each input node to the earliest nodeLayer
        for i in range(self.numInputs):
            weightsList = list()
            for j in range(self.numHiddenNodes):
                weightsList.append(random.uniform(-0.1, 0.1))
            weightsLists[0].append(weightsList)

        #build the weight matrix for each pair of layers.
        for i in range(hiddenLayers+1):
            self.weightsMatrices.append(np.matrix(weightsLists[i]).T)
            if self.debug:
                print self.weightsMatrices[i]

    def train(self, alpha, numEpochs):
        self.alpha = alpha
        self.numEpochs = numEpochs
        for epoch in range(self.numEpochs):
            #for each input
            print 'On epoch', epoch+1, 'out of', self.numEpochs
            for i in range(len(self.inputs)):
                #get the prediction and error for the example
                predictionVector = self.__prediction(self.inputs[i])
                errorVector = predictionVector
                for j in range(len(predictionVector)):
                    if j == self.classLabelList[i]:
                        errorVector[j] = 1.0 - errorVector[j]
                    else:
                        errorVector[j] = 0 - errorVector[j]
                print 'on input', i, 'error', errorVector
                #calculate DeltaJ for the output layer.
                inputVector = self.inputVectorsAtLayerList[self.hiddenLayers]
                vectorSize = len(inputVector)
                for j in range(vectorSize):
                    inputVector[j, 0] = self.__deltaOutputLayer(inputVector[j, 0], errorVector[j])
                #inputVector now contains the deltaJ values in a vector. store it to another one with a better name so my brain doesn't explode.
                upperInputVector = inputVector

                #for each weight in the layer before the output, do the updates.
                weightsCount = self.weightsMatrices[self.hiddenLayers].shape[1]
                upperWeightsCount = self.weightsMatrices[self.hiddenLayers].shape[0]
                #extra weight for bias
                for k in range(weightsCount+1):
                    for l in range(upperWeightsCount):
                        #this needs to use the activation value, NOT the actual value.
                        if k < weightsCount:
                            self.weightsMatrices[self.hiddenLayers][l,k] = self.weightsMatrices[self.hiddenLayers][l,k] + (self.alpha*self.activationVectorsAtLayerList[self.hiddenLayers-1][k,0]*upperInputVector[l])
                        #weight update for bias node
                        elif k == weightsCount:
                            self.biasNodesList[self.hiddenLayers+1][l] = self.biasNodesList[self.hiddenLayers+1][l] + (self.alpha*1.0*upperInputVector[l,0])
                
                #for each layer before the output layer going backwards,
                for j in range(self.hiddenLayers-1, -1, -1):
                    currentWeightsMatrix = self.weightsMatrices[j]
                    internalInputVector = self.inputVectorsAtLayerList[j]
                    #get the deltas for each input value
                    vectorSize = len(internalInputVector)
                    for l in range(vectorSize):
                        internalInputVector[l, 0] = self.__deltaInternalLayer(internalInputVector[l, 0], l, j, upperInputVector[j,0])
                    #now, the upper input vector is the vector we just calculated.
                    upperInputVector = internalInputVector
                    #for each weight in this layer, do the updates.
                    weightsCount = self.weightsMatrices[j].shape[1]
                    upperWeightsCount = self.weightsMatrices[j].shape[0]
                    if j > 0:
                        #extra weight for bias
                        for k in range(weightsCount+1):
                            for l in range(upperWeightsCount):
                                if k < weightsCount:
                                    currentWeightsMatrix[l,k] = currentWeightsMatrix[l,k] + (self.alpha*self.activationVectorsAtLayerList[j-1][k,0]*upperInputVector[l])
                                #weight update for bias node
                                elif k == weightsCount:
                                    self.biasNodesList[j+1][l] = self.biasNodesList[j+1][l] + (self.alpha*1.0*upperInputVector[l,0])
                    elif j==0:
                        example = self.inputs[i]
                        count = 0
                        for l in range(upperWeightsCount):
                            fasterDeltaJ = upperInputVector[l]
                            fasterBiasNode = self.biasNodesList[j+1][l]
                            for k in range(weightsCount):
                                #this needs to use the input value
                                if k < weightsCount:
                                    currentWeightsMatrix[l,k] = currentWeightsMatrix[l,k] + (self.alpha*(example[k]*fasterDeltaJ))
                                elif k==weightsCount:
                                    fasterBiasNode = fasterBiasNode + (self.alpha*1.0*fasterDeltaJ)

    #get the prediction for a given input index.
    def __prediction(self, inputList):
        inputVector = np.matrix(inputList).T
        for j in range(self.hiddenLayers+1):
            #neato matrix operations make this fast
            inputVector = self.weightsMatrices[j]*inputVector
            self.inputVectorsAtLayerList[j] = copy.deepcopy(inputVector)
            vectorSize = len(inputVector)
            #run sigmoid on the vector.
            #print 'vectorSize', vectorSize
            for i in range(vectorSize):
                #don't forget the bias node in this part.
                #print 'a', inputVector[i,0]
                #print j, i
                #print 'b', self.biasNodesList[j+1][i]
                inputVector[i, 0] = self.__sigmoid(inputVector[i, 0] + self.biasNodesList[j+1][i])
            self.activationVectorsAtLayerList[j] = copy.deepcopy(inputVector)

        return inputVector

    def __sigmoid(self, inputVal):
        try:
            return 1.0/(1+np.e**-inputVal)
        except:
            return 0.000001


    def __deltaOutputLayer(self, inputVal, error):
        sigmoid = self.__sigmoid(inputVal)
        return (sigmoid*(1-sigmoid)) * (error)

    def __deltaInternalLayer(self, inputVal, toNode, layer, deltaTo):
        sigmoid = self.__sigmoid(inputVal)
        #for the number of nodes in the above layer,
        jSum = 0.0
        itojWeightsVector = self.weightsMatrices[layer][toNode,:].T
        weightsVectorSize = len(itojWeightsVector)
        for i in range(weightsVectorSize):
            jSum+=itojWeightsVector[i]*deltaTo

        return (sigmoid*(1-sigmoid)) * jSum

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
            lineIndex+=1

        #get the cumulative accuracy of the network on the test data.
        numInputs = len(testInputs)
        successes = 0
        total=0
        for i in range(numInputs):
            predictionVector = self.__prediction(testInputs[i])
            partialSuccesses = 0
            #print 'predictionVector', predictionVector, 'prediction', testClassLabelList[i]
            for j in range(len(predictionVector)):
                if j == testClassLabelList[i] and int(round(predictionVector[j])) == 1:
                    partialSuccesses+=1
                elif j != testClassLabelList[i] and int(round(predictionVector[j])) == 0:
                    partialSuccesses+=1
            #print 'partialSuccesses', partialSuccesses
            if partialSuccesses == 5:
                successes+=1
            total+=1

        print 'Final accuracy:', successes, 'out of', total, '- Percentage:', (float(successes)/total)*100, '%'


    

def main():
    random.seed(1)
    if (len(sys.argv) != 3):
        print "Takes 2 command line argument: the name of the csv training file, and the name of the csv test file."
        exit(-1)
    trainingFilename = sys.argv[1]
    testFilename = sys.argv[2]
    isDebugMode = True
    hiddenLayers = 1
    alpha = 0.1
    numEpochs = 1
    #initialize the network
    neuralNet = MultilayerPerceptron(trainingFilename, hiddenLayers, isDebugMode)
    neuralNet.runTestExamples(testFilename)
    neuralNet.train(alpha, numEpochs)
    neuralNet.runTestExamples(testFilename)




main()