from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

class KNN:
    def __init__(self, X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.numClasses = len(set(list(y_train)+list(y_test))) #Infers number of labels.

    def predict(self, index:int, k:int):
        """Computes a prediction for a single sample, given index of that sample in test set."""
        distances = self.euclideanDist(self.X_train,self.X_test[index])
        neighbours = self.getNeighbours(k, distances)
        prediction = self.getResponse(neighbours,self.y_train)
        print("We predict that entry",index,"in the model is a number:",prediction)

    def predictAll(self, k:int) -> [float, list]:
        """Obtains predictions of test set returning the overall accruacy and list of predictions."""
        correctGuesses = 0
        euclideanArray = [] #Array to hold euclidean distances for each sample.
        neighbours = []#Array to hold k nearest neighbours of each test sample.
        guesses = []#Array to hold predictions of each test sample.

        #For each sample in the test set, compute distance to each sample in the train set.
        for i in range(len(self.X_test)):
            euclideanArray.append(self.euclideanDist(self.X_train,self.X_test[i]))

        #Produce a list of the k nearest neighbours.
        for i in range(len(euclideanArray)):
            neighbours.append(self.getNeighbours(k, euclideanArray[i]))

        #Produce a list of labels corresponding to k nearest neighbours.
        for i in range(len(neighbours)):
            guesses.append(self.getResponse(neighbours[i],self.y_train))

        #Compare with actual labels.
        for i in range(len(guesses)):
            if guesses[i] == self.y_test[i]:
                correctGuesses = correctGuesses + 1

        return self.accuracy(correctGuesses,len(guesses)), guesses


    def euclideanDist(self, X_train:np.array, X_test:np.array) -> list:
        """Obtains list of euclidean distances between each point in x_train with each point in y_train."""
        distances = []
        sumOfDistances = 0
        for x in range(len(X_train)):
            for i in range(len(X_train[x])):
                sumOfDistances = sumOfDistances + (((X_train[x][i]-X_test[i])**2)**0.5)

            distances.append(sumOfDistances)
            sumOfDistances = 0
        return distances

    def getNeighbours(self, k:int, distances:list) -> list:
        """Returns a list of the k nearest neighbours for sample in x_train, grouped in an overall list."""
        neighbours = []
        for _ in range(k):#Make K passes to find k nearest neighbours
            smallest = distances[0]
            smallestIndex = 0
            for j in range(len(distances)):#Loop through distances array and find smallest element
                if smallest>distances[j]:
                    smallestIndex = j
                    smallest = distances[j]

            neighbours.append(smallestIndex)#Save index for predictions
            distances[smallestIndex] = 10000 #Make entry arbitrarily large so that next loop doesn"t pick it up again.

        return neighbours

    def getResponse(self, neighbours:list, y_train:list) -> int:
        """Generates prediction for a single response given a list of its k-nearest neighbours."""
        predictions = []
        predictionCount = [0]*self.numClasses #Tracks number of votes for each class. (K votes altogether.)

        #Get labels for each of the k nearest neighbors.
        for i in range(len(neighbours)):
            predictions.append(self.y_train[neighbours[i]])

        #Update votes tracker.
        for i in range(len(predictions)):
            predictionCount[(predictions[i])] = predictionCount[(predictions[i])]+1

        #Find the index with the most votes and return.
        projected = predictionCount[0]
        prediction = 0;
        for i in range(len(predictionCount)):
            if projected<predictionCount[i]:
                projected = predictionCount[i]
                prediction = i

        return prediction

    def accuracy(self, correctGuesses:int, numOfGuesses:int):#Takes the array containing all correct guesses so that we can find difference between that number and num of entries to calc accuracy.
        accuracy = (correctGuesses/numOfGuesses)*100
        return accuracy


def confusionMatrix(predictions:list, evidence:list, numClasses:int) -> np.array:
    '''Computes confusion matrix given a set of predictions, true labels, and number of classes.'''
    confusionMatrix = np.zeros((numClasses, numClasses)) #Intialise confusion matrix.

    #Populate confusion matrix where rows represent predictions and columns represent true values/
    for i in range(len(predictions)):
        confusionMatrix[predictions[i]][evidence[i]] = confusionMatrix[predictions[i]][evidence[i]] + 1

    #Display confusion matrix.
    for i in range(numClasses):
        for j in range(numClasses):
            print(str(int(confusionMatrix[i][j])) + " ", end = "")
        print()

    return confusionMatrix



data = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20, random_state=0, stratify=data.target)

myKNN = KNN(X_train, y_train, X_test, y_test)
myAccuracy, myPredictions = myKNN.predictAll(1)
print("Accuracy for My KNN is: ", end="")
print(str(round(myAccuracy,2))+ "%")
print("Confusion Matrix for my KNN implementation is:")
confusionMatrix(myPredictions, y_test, 10)
print()

libraryKnn = KNeighborsClassifier(n_neighbors = 1)
libraryKnn.fit(X_train, y_train)
predictions = libraryKnn.predict(X_test)
print("Accuracy for the libary KNN is: ", end ="")
print(str(round(accuracy_score(y_test, predictions)*100,2))+"%")
print("Confusion Matrix for the library KNN implementation is:")
confusionMatrix(predictions, y_test, 10)
