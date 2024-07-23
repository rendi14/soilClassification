import time
import numpy as np
from scipy.linalg import pinv

class elm():
    def __init__(self, hidden_units, activation_function, data, label):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.data = data
        self.label = label
        self.class_num = np.unique(self.label).shape[0]
        self.beta = np.zeros((self.hidden_units, self.class_num))

        self.setOneHotLabel()
        self.setWeight()
        self.setBias()

    def setOneHotLabel(self):
        self.one_hot_label = np.zeros((self.label.shape[0], self.class_num))
        for i in range(self.label.shape[0]):
            self.one_hot_label[i, int(self.label[i])] = 1

    def setWeight(self):
        self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.data.shape[1]))

    def setBias(self):
        self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __inputOfHiddenLayer(self, data):
        self.temH = np.dot(self.W, data.T) + self.b

        if self.activation_function == 'sigmoid':
            self.H = 1/(1 + np.exp(-self.temH))

        if self.activation_function == 'relu':
            self.H = self.temH * (self.temH > 0)

        return self.H

    def __outputOfHiddenLayer(self, H):
        self.output = np.dot(H.T, self.beta)
        return self.output

    def elm_fit(self):
        self.time1 = time.time()
        self.H = self.__inputOfHiddenLayer(self.data)
        self.beta = np.dot(pinv(self.H.T), self.one_hot_label)

        self.time2 = time.time()
        self.result = np.exp(self.__outputOfHiddenLayer(self.H)) / np.sum(np.exp(self.__outputOfHiddenLayer(self.H)), axis=1).reshape(-1, 1)
        self.label_ = np.where(self.result == np.max(self.result, axis=1).reshape(-1, 1))[1]

        self.correct = 0
        for i in range(self.label.shape[0]):
            if self.label_[i] == self.label[i]:
                self.correct += 1
        self.train_score = self.correct/self.label.shape[0]

        return self.beta, self.train_score, str(self.time2 - self.time1)

    # def elm_predict(self, data):
    #     self.H = self.__inputOfHiddenLayer(data)
    #     output = self.__outputOfHiddenLayer(self.H)
    #     self.label_ = np.where(output == np.max(output, axis=1).reshape(-1, 1))[1]
    #     # probabilities = np.exp(output) / np.sum(np.exp(output), axis=1).reshape(-1, 1)

        # return self.label_

    def elm_predict(self, data):
        self.H = self.__inputOfHiddenLayer(data)
        self.label_ = np.where(self.__outputOfHiddenLayer(self.H) == np.max(self.__outputOfHiddenLayer(self.H), axis=1).reshape(-1, 1))[1]
        return self.label_

    def score(self, data, label):
        self.prediction = self.elm_predict(data)

        self.correct = 0
        for i in range(label.shape[0]):
            if self.prediction[i] == label[i]:
                self.correct += 1
        self.test_score = self.correct / label.shape[0]

        return self.test_score