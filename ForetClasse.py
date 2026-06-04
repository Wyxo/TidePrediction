#!/home/araulin/venv_python/bin/python

from math import log2
import numpy as np
from random import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import csv

## Criterion

class Criterion:
    
    def initialize(self, ySorted):
        pass
    
    def moveSample(self, y):
        pass
    def score(self):
        pass
    def output(self, ySorted):
        pass
    
class VarianceCriterion(Criterion):
    def initialize(self, ySorted):
        self.N = len(ySorted)
        self.leftN = 0 
        self.leftSum = 0 
        self.leftSum2 = 0

        self.rightN = self.N
        self.rightSum = np.sum(np.array(ySorted))
        self.rightSum2 = np.sum(np.array(ySorted)**2)
    
    def moveSample(self, y):
        self.leftN += 1
        self.rightN -= 1

        self.leftSum += y
        self.rightSum -= y

        self.leftSum2 += y**2
        self.rightSum2 -= y**2
        
    def score(self):
        if self.N == 0:
            return 0
        varianceLeft = self.leftSum2/self.leftN  - (self.leftSum/self.leftN)**2 
        varianceRight = self.rightSum2/self.rightN  - (self.rightSum/self.rightN)**2 
        return -(varianceLeft*self.leftN + varianceRight*self.rightN)/self.N
    def output(self, ySorted):
        return np.mean(ySorted)
    
class EntropyCriterion(Criterion):
    def initialize(self, ySorted):
        self.N = len(ySorted)

        classes = set(ySorted)
        self.mapping = {val: i for i, val in enumerate(classes)}

        self.leftCount = 0
        self.leftNumbers = np.zeros(len(classes))

        self.rightCount = self.N
        self.rightNumbers = np.array([np.sum(ySorted == c) for c in classes])
    
    def moveSample(self, y):
        self.leftCount += 1
        self.rightCount -= 1

        self.leftNumbers[self.mapping[y]] += 1
        self.rightNumbers[self.mapping[y]] -= 1
        
    def score(self):
        entropyLeft = entropy(self.leftNumbers/self.leftCount)
        entropyRight = entropy(self.rightNumbers/self.rightCount)
        return (entropyLeft*self.leftCount + entropyRight*self.rightCount)/self.N
    
    def output(self, ySorted):
        counts = np.bincount(ySorted.astype(int))
        return np.argmax(counts)

def entropy(probabilities):
    return np.sum(probabilities * np.log2(probabilities + 1e-10))

class Node :
    def __init__(self):
        self.score = None

        self.nSamples = None
        self.depth = None

        self.featureIndex = None
        self.featureThreshold = None
        
        self.left = None
        self.right = None

        self.output = None

class DecisionTree :
    def __init__(self, criterion, maxDepth=10, minSamplesSplit=2, minSamplesLeaf=1):
        self.max_depth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf

        self.root = None

        self.criterion = criterion
        
        self.tree = None
    def best_split(self, X, y, numberOfArgs):
        N = len(y)

        bestIndex , bestThreshold, bestScore = None, None, -1000
        attributs = sample([k for k in range(len(X[0]))], numberOfArgs)
        for index in attributs :
            #Sort by the current attribute
            seuils, classes = zip(*sorted(zip(X[:, index], y)))
            self.criterion.initialize(classes)
            
            for i in range(self.minSamplesLeaf, N-self.minSamplesLeaf):
                self.criterion.moveSample(classes[i-1])
                if seuils[i] == seuils[i-1]:
                    continue
                
                score = self.criterion.score()
                if score > bestScore :
                    bestScore = score
                    bestIndex = index
                    bestThreshold = (seuils[i] + seuils[i-1]) / 2 
        return bestIndex, bestThreshold, bestScore

    def planting(self, X, y, numberOfArgs,  depth=0):
        m = y.size
        node = Node()
        node.depth = depth
        node.nSamples = m
        if m <= self.minSamplesSplit :
            node.output = self.criterion.output(y)
            return node
        
        if depth <= self.max_depth :
            index, seuil, score = self.best_split(X, y, numberOfArgs)
            if index!= None :
                indexLeft = X[:, index] < seuil
                XLeft, yLeft = X[indexLeft] , y[indexLeft]
                XRight, yRight = X[np.logical_not(indexLeft)], y[np.logical_not(indexLeft)]
                
                if len(yLeft) == 0 or len(yRight) == 0:
                  node.output = self.criterion.output(y)
                  return node
                
                node.featureIndex = index
                node.featureThreshold = seuil

                node.left = self.planting(XLeft, yLeft, numberOfArgs,  depth + 1)
                node.right = self.planting(XRight, yRight, numberOfArgs,  depth + 1)
                node.score = score
            else :
              node.output = self.criterion.output(y)
        else:
            node.output = self.criterion.output(y)      
        return node

    def fit(self, X, y, numberOfArgs):
        self.tree = self.planting(X, y, numberOfArgs)
        
    def prediction(self, input):
        
        node = self.tree
        while node.left!= None :
            if input[node.featureIndex] < node.featureThreshold  :
                node = node.left
            else :
                node = node.right
        return node.output

    def predictions(self, X):
        return [self.prediction(entrees) for entrees in X]
    
def bagging(data, numerOfBags):
    
    samples = []
    n = len(data)
    for _ in range(numerOfBags): 
        tempBag = []
        for _ in range(n):
            tempBag += [data[randrange(0, n)]]
        samples += [tempBag]
    return samples
    
    
def makeForest(data, numerOfTrees, numberOfArgs, criterion = VarianceCriterion(),  maxDepth = 10, minSampleSplit = 2, minSamplesLeaf = 1):
    
    forest = []
    samples = bagging(data, numerOfTrees)
    for sample in samples :
        a = DecisionTree(criterion, maxDepth, minSampleSplit,minSamplesLeaf)
        a.fit(np.array(sample)[:, :-1], np.array(sample)[:, -1], numberOfArgs)
        forest += [a]
    return forest

def predictionForest(forest, inputs) :
    predit = []
    m = len(forest)
    for input in inputs :
        predit += [1/m * np.sum([tree.prediction(input) for tree in forest])]
    return predit
    
def predictor(data, numerOfTrees, numberOfArgs, criterion = VarianceCriterion(), maxDepth = 8, minSampleSplit = 2, minSamplesLeaf = 1):
    '''Crée une forêt sur 3/4 des données, puis test la précision du modèle avec le 1/4
    restant. La fonction d'erreur employée est l'erreur moyenne'''
    train, test = train_test_split(data, test_size = 0.25)
    forest = makeForest(train, numerOfTrees, numberOfArgs, criterion, maxDepth, minSampleSplit, minSamplesLeaf)
    erreurTrain = erreurTest = 0
    L1 = predictionForest(forest, np.array(test)[:, :-1])
    L2 = predictionForest(forest, np.array(train)[:, :-1])
    for k in range(len(L1)) :
        erreurTest += (L1[k] - test[k][-1])**2
    for k in range(len(L2)) :
        erreurTrain += (L2[k] - train[k][-1])**2
    erreurTest = np.sqrt(erreurTest/len(L1))
    erreurTrain = np.sqrt(erreurTrain/len(L2))
    return forest, erreurTest, erreurTrain, L1, test
        
def printTree(tree, espace = 0):
    '''Implémentation sommaire permettant la visualisation de petits arbres'''
    if espace ==0 :
        node = tree.tree
    else : 
        node = tree
    if node.left == node.right == None :
        print(" " * espace + "[{:.1f}]".format(node.output))
    else :
        text = " " * espace + "[X{:.0f} {:.1f}]".format(node.featureIndex, node.featureThreshold)
        print(text)
    if node.left !=None :
        printTree(node.left, espace + 2)
    if node.right !=None :
        printTree(node.right, espace + 2)

def readData(file):
    donnees = csv.reader(file, delimiter=',')
    k, X, Y=0, [], []
    for k, ligne in enumerate(donnees):
        if k == 0:
            continue  # skip header
        def f(i):
            return float(ligne[i].replace(",", "."))
        y = f(1)
        #x = [1/f(4)**2, 1/f(7)**2,f(8),f(9), f(10), f(11), abs(f(12)), abs(f(15))]
        x = [f(4), f(7), abs(f(12)), abs(f(15))]
        x = [1/f(4)**2, 1/f(7)**2, f(8), f(9), f(10), f(11), abs(f(12))]
        if k == 1:
            print(y, x)
        X.append(x)
        Y.append(y)

    X, Y = X[:2497], Y[2:]
    for k in range(len(X)):
        X[k] += [Y[k] - 20]
    return np.array(X)

if __name__ == "__main__":
    file = open("data.csv","r")
    data = readData(file)
    numerOfTrees = 1
    numberOfArgs = 6
    maxDepth = 4
    minSamplesSplit = 2
    minSamplesLeaf = 1
    criterion = VarianceCriterion()
    forest, erreur_test, erreur_train, prediction, test = predictor(data, numerOfTrees, numberOfArgs, criterion, maxDepth, minSamplesSplit, minSamplesLeaf )
    print("erreur test : ", erreur_test)
    print("erreur train : ", erreur_train)
    printTree(forest[0])