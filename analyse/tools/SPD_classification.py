from abc import abstractmethod
import numpy as np
import pandas as pd
import queue

class singlePointDataModel(object):


    @abstractmethod
    def __init__(self):
        pass

    # calculate the right rate and error after classfication
    @abstractmethod
    def evaluate(self,preY,reaY):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass



class SPD_classfication(singlePointDataModel):

    def __init__(self):
        self.__isProgress = None
        self.__totalProgress = None
        self.__currentProgress = None
        pass

    def choeseFunction(self):
        pass

    def progressBar(self):

        if self.__isProgress == None:
            raise ("there are not initial ProcessBar")
        if self.__totalProgress == None:
            raise ("task do not start")
        if self.__isProgress:
            return int(100*float(self.__currentProgress)/self.__totalProgress)
        else:
            return -1

    def evaluate(self,preY,reaY):

        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
        accuracy = accuracy_score(preY,reaY)
        precision = precision_score(preY,reaY,average='weighted')
        recall = recall_score(preY,reaY,average='weighted')
        F = f1_score(preY,reaY,average='weighted')

        return accuracy,precision,recall,F