#Tristan Basil
#Assignment: Project 4 - cS460G Machine Learning, Dr. Harrison

import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import copy

class Node:
    weights = list()
    forwardNodes = list()

    def printNode(self):
        print "featureIndex:", self.featureIndex, "featureMin/Max:", self.featureMin, self.featureMax, "label:", self.label,


#this class is only designed to work for the data in this project.
class MultilayerPerceptron:
    hiddenLayers = -1
    debug = False
    inputNodes = list()
    inputs = list()
    classLabelList = list()
    distinctClassLabels = set()
    unbranchedFeatures = set()
    rootTreeNode = Node()

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
    def __init__(self, filename, numLayers, debug):
        self.debug = debug
        self.numLayers = numLayers
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
            self.classLabelList.append(parsedLine[0])
            self.inputs.append(list())
            for i in range(1, len(parsedLine)):
                self.inputs[lineIndex].append(int(parsedLine[i]))
            #self.distinctClassLabels.add(classLabel)
            lineIndex+=1

        #build the tree from backwards up.
        outputNode = Node()

        #make a fixed amount of hidden nodes that point to this output
        weightsList2 = list()
        hiddenNodes = list()
        for i in range(50):
            hiddenNodes.append(Node())
            hiddenNodes[i].forwardNodes.append(outputNode)
            hiddenNodes[i].weights.append(0.5)
            weightsList2.append(hiddenNodes[i].weights)

        #make the input nodes, then connect each input node to each hidden node
        weightsList1 = list()
        for i in range(784):
            self.inputNodes.append(Node())
            for j in range(50):
                self.inputNodes[i].forwardNodes.append(hiddenNodes[j])
                self.inputNodes[i].weights.append(0.5)
            weightsList1.append(self.inputNodes[i].weights)

        #build the weight matrix for each pair of layers.
        print(np.matrix(weightsList2))
        print(np.matrix(weightsList1))
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

    

def main():
    if (len(sys.argv) != 2):
        print "Takes 1 command line argument: the name of the csv file."
        exit(-1)
    filename = sys.argv[1]
    isDebugMode = False
    numLayers = 3
    #initialize the network
    neuralNet = MultilayerPerceptron(filename, numLayers, isDebugMode)



main()