from analyse.tools.SPD_classification import SPD_classfication
from keras.models import Model
from keras.layers import Input,PReLU,Dense,LSTM,multiply,concatenate,Activation
from keras.layers import Conv1D,BatchNormalization,GlobalAveragePooling1D,Permute,Dropout
from analyse.tools import operateDB
from analyse.models import Spd_data
import numpy as np
import pandas as pd
import queue


class DTW_kNN(object):
    def __init__(self,k = 1,method='default'):
        self.__k = k
        self.__method = method

    def dtw(self,sqe1, sqe2):

        n = len(sqe1)
        m = len(sqe2)


        d = np.zeros([n, m])
        for i in range(n):
            for j in range(m):
                d[i][j] = (sqe1[i] - sqe2[j]) * (sqe1[i] - sqe2[j])
        realmax = np.inf
        D = np.ones([n, m]) * realmax

        D[0][0] = d[0][0]
        for i in range(1, n):
            for j in range(m):
                D1 = D[i - 1][j]
                if j > 0:
                    D2 = D[i - 1][j - 1]
                else:
                    D2 = realmax

                if j > 1:
                    D3 = D[i - 1][j - 2]
                else:
                    D3 = realmax
                D[i][j] = d[i][j] + np.min(np.array([D1, D2, D3]))

        return D[n - 1][m - 1]

    def predict(self,trainData,testData):
        if type(trainData)!=type(np.zeros(0)) and type(testData)!=type(np.zeros(0)):
            raise ("train or test data need to be numpy")
        else:
            pass
        trainX = trainData[:,1:-1]
        trainY = trainData[:,0]
        testX = testData[:,1:-1]
        preY = []
        pp = 0
        for currentTestSample in testX:
            index = 0

            print (pp)
            pp = pp + 1
            circleDistance = []
            circleClass = []
            for campareSample in trainX:
                distance = self.dtw(currentTestSample,campareSample)
                circleDistance.append(distance)
                index = index + 1

            circleDistance = pd.Series(circleDistance).sort_values()[0:self.__k]

            for jj in circleDistance.index:
                circleClass.append(trainY[jj])

            preY.append(self.circleJudge(np.array(circleClass), circleDistance.values))

        return np.array(preY)

    def circleJudge(self,sampleClass,distance):
        if len(sampleClass)==1:
            return sampleClass[0]

        elif self.__method == 'default':
            sampleClass = sampleClass.astype(int)
            return np.argmax(np.bincount(sampleClass))

        else:

            dmin,dmax = distance.min(),distance.max()
            distance = (distance - dmin)/(dmax - dmin)

    def evaluate(self, preY, reaY):
        rightNumber = 0
        errorNumber = 0

        for i in range(len(preY)):
            if reaY[i] == preY[i]:
                rightNumber = rightNumber + 1
            else:
                errorNumber = errorNumber + 1

        if rightNumber + errorNumber != len(preY):
            raise ("there are errors in count right and wrong ")


        rightRatio = float(rightNumber) / len(preY)
        errorRatio = float(errorNumber) / len(preY)

        return rightRatio, errorRatio

class DTW_kNN(object):
    def __init__(self,k = 1,method='default'):
        self.__k = k
        self.__method = method

    def dtw(self,sqe1, sqe2):

        n = len(sqe1)
        m = len(sqe2)


        d = np.zeros([n, m])
        for i in range(n):
            for j in range(m):
                d[i][j] = (sqe1[i] - sqe2[j]) * (sqe1[i] - sqe2[j])
        realmax = np.inf
        D = np.ones([n, m]) * realmax

        D[0][0] = d[0][0]
        for i in range(1, n):
            for j in range(m):
                D1 = D[i - 1][j]
                if j > 0:
                    D2 = D[i - 1][j - 1]
                else:
                    D2 = realmax

                if j > 1:
                    D3 = D[i - 1][j - 2]
                else:
                    D3 = realmax
                D[i][j] = d[i][j] + np.min(np.array([D1, D2, D3]))

        return D[n - 1][m - 1]

    def predict(self,trainData,testData):
        if type(trainData)!=type(np.zeros(0)) and type(testData)!=type(np.zeros(0)):
            raise ("train or test data need to be numpy")
        else:
            pass
        trainX = trainData[:,1:-1]
        trainY = trainData[:,0]
        testX = testData[:,1:-1]
        preY = []
        pp = 0
        for currentTestSample in testX:
            index = 0

            print (pp)
            pp = pp + 1
            circleDistance = []
            circleClass = []
            for campareSample in trainX:
                distance = self.dtw(currentTestSample,campareSample)
                circleDistance.append(distance)
                index = index + 1

            circleDistance = pd.Series(circleDistance).sort_values()[0:self.__k]

            for jj in circleDistance.index:
                circleClass.append(trainY[jj])

            preY.append(self.circleJudge(np.array(circleClass), circleDistance.values))

        return np.array(preY)

    def circleJudge(self,sampleClass,distance):
        if len(sampleClass)==1:
            return sampleClass[0]

        elif self.__method == 'default':
            sampleClass = sampleClass.astype(int)
            return np.argmax(np.bincount(sampleClass))

        else:

            dmin,dmax = distance.min(),distance.max()
            distance = (distance - dmin)/(dmax - dmin)

    def evaluate(self, preY, reaY):
        rightNumber = 0
        errorNumber = 0

        for i in range(len(preY)):
            if reaY[i] == preY[i]:
                rightNumber = rightNumber + 1
            else:
                errorNumber = errorNumber + 1

        if rightNumber + errorNumber != len(preY):
            raise ("there are errors in count right and wrong ")

        rightRatio = float(rightNumber) / len(preY)
        errorRatio = float(errorNumber) / len(preY)

        return rightRatio, errorRatio












class Euclidean_kNN(SPD_classfication):

    def __init__(self,k=1 , method = 'default'):
        self.__k = k
        self.__method = method
        self.__isProgress = True


    def predict(self,trainData,testData):

        if type(trainData)!= type(np.zeros(0)) and type(testData)!=type(np.zeros(0)):
            raise ("train or test data need to be numpy")
        else:
            pass

        trainX = trainData[:,1:-1]
        trainY = trainData[:,0]
        testX = testData[:,1:-1]


        preY = []
        self.__currentProgress = 0
        self.__totalProgress = len(testX)

        for currentTestSample in testX:

            index = 0
            circleClass = []
            circleDistance=[]


            for campareSampel in trainX:
                distance = (currentTestSample-campareSampel)**2
                distance = np.sum(distance)
                circleDistance.append(distance)
                index = index + 1

            circleDistance = pd.Series(circleDistance).sort_values()[0:self.__k]


            for jj in circleDistance.index:
                circleClass.append(trainY[jj])


            preY.append(self.circleJudge(np.array(circleClass),circleDistance.values))
            self.__currentProgress = self.__currentProgress+1

        return np.array(preY)


    def circleJudge(self,sampleClass,distance):
        if len(sampleClass)==1:
            return sampleClass[0]

        elif self.__method == 'default':
            sampleClass = sampleClass.astype(int)
            return np.argmax(np.bincount(sampleClass))

        else:

            dmin,dmax = distance.min(),distance.max()
            distance = (distance - dmin)/(dmax - dmin)



class LSTM_FCN(SPD_classfication):

    def __init__(self):
        self.__model = None
        pass

    def generate_model1(self,input_length,output_length):

        ip = Input(shape=(1,input_length))

        x = LSTM(8)(ip)
        x = Dropout(0.8)(x)

        y = Permute((2,1))(ip)
        y = Conv1D(128,8,padding='same',kernel_initializer = 'he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(256,5,padding = 'same',kernel_initializer = 'he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(128,3,padding='same',kernel_initializer = 'he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D(y)

        x = concatenate([x,y])

        out = Dense(output_length,activation = 'softmax')(x)

        model = Model(ip,out)

        model.summary()

        return model

    def train_model(self,epoch = 50,batch_size = 128,val_subset = None,learning_rate = 1e-3):
        if self.__model==None:
            raise ("you need to generate model first ")
        else:
            pass
        pass
    def fit(self,trainData,testData):

        if type(trainData)!= type(np.zeros(0)) and type(testData)!=type(np.zeros(0)):
            raise ("train or test data need to be numpy")
        else:
            pass

        trainX = trainData[:,1:-1]
        trainY = trainData[:,0]
        testX = testData[:,1:-1]
        input_length = len(trainX[0])
        output_length = len(list(set(trainY)))

        self.__model = self.generate_model1(input_length,output_length)



# if __name__ == '__main__':


def selectModel(method_name = 'Euclidean_KNN'):
    if method_name == 'Euclidean_KNN':
        model = Euclidean_kNN(k=1,method='default')
    elif method_name == 'DTW_KNN':
        model = DTW_kNN(k=1, method='default')

    return model


# import os
def execute_data(filename = 'synthetic_control',method = 'ed_knn'):
    results = []

    trainData,testData = Spd_data().getDataAccordingName(filename)

    results.append(filename)

    model = selectModel(method)
    # model2 = DTW_kNN(k=1, method='default')

    preY1 = model.predict(trainData, testData)
    # preY2 = model2.predict(trainData, testData)

    accuracy, precision, recall, F = model.evaluate(testData[:, 0], preY1)
    results.extend([accuracy, precision, recall, F])

    # print("ek right ratio is:",rightRatio,"\n","error ratio is:",errorRatio)
    # rightRatio2, errorRatio2 = model2.evaluate(testData[:, 0], preY2)
    # results.append(rightRatio2)
    # results.append(errorRatio2)
    # # print("dtwk right ratio is:",rightRatio,"\n","error ratio is:",errorRatio)
    return results