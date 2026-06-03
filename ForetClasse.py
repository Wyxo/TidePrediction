from math import log2
import numpy as np
from random import *
from sklearn.model_selection import train_test_split
import time

## Criterion

class Criterion:
    
    def initialize(self, ySorted):
        pass
    
    def move_sample(self, y):
        pass
    def score(self):
        pass
    

class VarianceCriterion(Criterion):
    def initialize(self, ySorted):
        self.N = len(ySorted)

        self.leftN = 0 
        self.leftSum = 0 
        self.leftSum2 = 0

        self.rightN = self.N
        self.rightSum = np.sum(ySorted)
        self.rightSum2 = np.sum(ySorted**2)
    
    def move_sample(self, y):
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
    
class EntropyCriterion(Criterion):
    def initialize(self, ySorted):
        self.N = len(ySorted)

        classes = set(ySorted)
        self.mapping = {val: i for i, val in enumerate(classes)}

        self.leftCount = 0
        self.leftNumbers = np.zeros(len(classes))

        self.rightCount = self.N
        self.rightNumbers = np.array([np.sum(ySorted == c) for c in classes])
    
    def move_sample(self, y):
        self.leftCount += 1
        self.rightCount -= 1

        self.leftNumbers[self.mapping[y]] += 1
        self.rightNumbers[self.mapping[y]] -= 1
        
    def score(self):
        entropyLeft = entropy(self.leftNumbers/self.leftCount)
        entropyRight = entropy(self.rightNumbers/self.rightCount)
        return (entropyLeft*self.leftCount + entropyRight*self.rightCount)/self.N
    

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

#Dans toute la suite les classes sont des entiers de 0 à n_classes
#Quitte à translater les coefficients de marée de 20

class Arbre_de_decision :
    def __init__(self, criterion, leafValue, maxDepth=10, minSamplesSplit=2, minSamplesLeaf=1):
        self.max_depth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf

        self.root = None

        self.criterion = criterion
        self.criterion = leafValue

    def best_split(self, X, y, arg):
        '''Renvoie l'index de l'argument, et le seuil correspondant, qui maximisent le gain d'information
        parmi les exemples X. En cherchant parmi arg arguments choisis au hasard.'''
        #Séparation si il y a assez d'élements

        N = len(y)
        if N < self.minSamplesLeaf :
            return None, None

        #Initialisation
        bestScore = -1000
        bestIndex , bestThreshold = None, None
        #Parcours des différents arguments avec un facteur aléatoire
        attributs = sample([k for k in range(len(X[0]))], arg)
        for index in attributs :
            #Tri des données selon l'argument que l'on observe
            seuils, classes = zip(*sorted(zip(X[:, index], y)))
            self.criterion.initialize(classes)
            #Test des potentielles séparations
            for i in range(1, N):
                self.criterion.move_sample(classes[i-1])
                score = self.criterion.score()
                if score > bestScore :
                    bestScore = score
                    bestIndex = index
                    bestThreshold = (seuils[i] + seuils[i-1]) / 2 
        return bestIndex, bestThreshold, bestScore

    def planting(self, X, y,arg,  depth=0):
        '''Fonction qui va récursivement fabriquer l'abre décisionnel en se réappelant sur
        les deux sous-ensembles de l'ensemble X, séparées suivant le meilleur seuil'''
        m = y.size
        node = Node()

        node.depth = depth
        node.nSamples = m

        if depth <= self.max_depth :
            index, seuil, score = self.best_split(X, y, arg)
            if index!= None :
                #Tableau des indices des exemples dont le index-e argument est inférieur au seuil
                indexLeft = X[:, index] < seuil
                XLeft, yLeft = X[indexLeft] , y[indexLeft]
                XRight, yRight = X[np.logical_not(indexLeft)], y[np.logical_not(indexLeft)]
                node.featureIndex = index
                node.featureThreshold = seuil
                #Rappel de le fonction sur les deux sous-ensembles en incrémentant la profondeur
                node.left = self.planting(XLeft, yLeft, arg,  depth + 1)
                node.right = self.planting(XRight, yRight, arg,  depth + 1)
                node.score = score
        return node

    def fit(self, X, y, arg):
        self.tree = self.planting(X, y, arg)
        
    def predict(self, input):
        '''Parcourt l'arbre jusqu'à arriver à la feuille correspondante à l'entrée puis renvoie la valeur prédite'''
        node = self.tree
        while node.left!= None :
            if input[node.featureIndex] < node.threshold  :
                node = node.left
            else :
                node = node.right
        return node.classe_predite

    def prediction(self, X):
        return [self.predict(entrees) for entrees in X]
    
def bagging(data, nombre):
    """Crée de nouveaux échantillons par tirage au hasard dans 
    data, avec remise"""
    échantillons = []
    n = int(len(data)/3)
    for k in range(nombre): 
        D = []
        for k in range(n):
            i = randrange(0, n)
            D += [data[i]]
        échantillons += [D]
    return échantillons
    
    
def foret(data, population, profondeur, taillemin, arg):
    """Crée une forêt d'arbres, chacun développé sur un
    sous échantillon du set de données initial"""
    start = time.time()
    foret = []
    echantillons = bagging(data, population)
    k = 0
    for ech in echantillons :
        a = Arbre_de_decision(profondeur, taillemin)
        a.fit(np.array(ech)[:, :-1], np.array(ech)[:, -1], arg)
        foret += [a]
        k+=1
    return time.time() - start

def prediction_foret(foret, entrees) :
    predit = []
    m = len(foret)
    for entre in entrees :
        predit += [1/m * np.sum([arbre.predict(entre) for arbre in foret])]
    return predit
    
def modele(data, population, profondeur, taillemin, arg):
    '''Crée une forêt sur 3/4 des données, puis test la précision du modèle avec le 1/4
    restant. La fonction d'erreur employée est l'erreur moyenne'''
    train, test = train_test_split(data, test_size = 0.25)
    foret = foret(train, population, profondeur, taillemin, arg)
    erreur_train = erreur_test = 0
    L1 = prediction_foret(foret, np.array(test)[:, :-1])
    L2 = prediction_foret(foret, np.array(train)[:, :-1])
    for k in range(len(L1)) :
        erreur_test += abs(L1[k] - test[k][-1])
    for k in range(len(L2)) :
        erreur_train += abs(L2[k] - train[k][-1])
    erreur_test = erreur_test/len(L1)
    erreur_train = erreur_train/len(L2)
    return foret, erreur_test, erreur_train
        
def affichageArbre(Arbre, espace = 0):
    '''Implémentation sommaire permettant la visualisation de petits arbres'''
    if espace ==0 :
        noeud = Arbre.arbre
    else : 
        noeud = Arbre
    if noeud.gauche == noeud.droite == None :
        print(" " * espace + "[" + str(noeud.classe_predite) + "]")
    else :
        text = " " * espace + "[" + "X" + str(noeud.index) + " < " + str(noeud.seuil) + "]"
        print(text)
    if noeud.gauche !=None :
        affichageArbre(noeud.gauche, espace + 2)
    if noeud.droite !=None :
        affichageArbre(noeud.droite, espace + 2)
                  
def jardinage(queue, echantillons ,profondeur, taillemin, arg):
    for ech in echantillons :
        a = Arbre_de_decision(profondeur,taillemin)
        a.fit(np.array(ech)[:, :-1], np.array(ech)[:, -1], arg)
        queue.put([a])
    