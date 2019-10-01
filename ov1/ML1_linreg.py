import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

train1data = genfromtxt("C:\\Users\\T. Laurvik\\Desktop\\dataset\\dataset\\regression\\train_1d_reg_data.csv", delimiter=',')[1:]
test1data = genfromtxt("C:\\Users\\T. Laurvik\\Desktop\\dataset\\dataset\\regression\\test_1d_reg_data.csv", delimiter=',')[1:]
train2data = genfromtxt("C:\\Users\\T. Laurvik\\Desktop\\dataset\\dataset\\regression\\train_2d_reg_data.csv", delimiter=',')[1:]
test2data = genfromtxt("C:\\Users\\T. Laurvik\\Desktop\\dataset\\dataset\\regression\\test_2d_reg_data.csv", delimiter=',')[1:]

############Task 2.1.2##########

def splitdata(data):
    return data[:,:-1],data[:,-1] ##[start:stop,start:stop]

def linregOLS(X,y): ##Task 2.1.1##
    return np.linalg.pinv(X.T@X)@X.T@y

def Emse(w,X,y):
    return (w.T@X.T@X@w - 2*(X@w).T@y + y.T@y)/len(X)

X_train2,y_train2 = splitdata(train2data) ##splitter
X_test2,y_test2 = splitdata(test2data) ##splitter

train2_weights = linregOLS(X_train2,y_train2)
train2_error = Emse(train2_weights,X_train2,y_train2)
test2_error = Emse(train2_weights,X_test2,y_test2)

print('Weights:[w0,w1,w2] ',train2_weights)
print(f'Train2 error: {train2_error}\nTest2 error: {test2_error}\n\n')
    
############Task 2.1.3###############
X_train1,y_train1 = splitdata(train1data)
X_test1,y_test1 = splitdata(test1data)
train1_weights = linregOLS(X_train1,y_train1)

train1_error = Emse(train1_weights,X_train1,y_train1)
test1_error = Emse(train1_weights,X_test1,y_test1)

print(f'Weights[w0,w1]: {train1_weights}')
print(f'Train1 error: {train1_error}\nTest1 error: {test1_error}')
x1=np.linspace(0,max(y_test1),num=175)
##print(np.shape(X_test1[:,1]))
##print(np.shape(y_test1))
plt.plot(X_test1[:,1],y_test1,"o")
plt.plot(x1,train1_weights[0]+train1_weights[1]*x1,"-")
plt.show()
