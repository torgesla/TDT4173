import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

cltrain1 = genfromtxt("C:\\Users\\T. Laurvik\\Desktop\\dataset\\dataset\\classification\\cl_train_1.csv", delimiter=',')
cltest1 = genfromtxt("C:\\Users\\T. Laurvik\\Desktop\\dataset\\dataset\\classification\\cl_test_1.csv", delimiter=',')

lr = 0.03 ##Learning rate
k = 1000 ##Number of iterations

def prepare_dataset(dataset):
    return dataset[:,:-1],dataset[:,-1] ##returns X,y
def sigma_func(z): ##Correct
    return 1/(1+np.exp(-z))
def cross_ent_error(y,z): #Correct
    N = len(y)
    summ = y.dot(np.log(sigma_func(z)))+(1-y).dot(np.log(1-sigma_func(z))) 
    return -1/N*summ
def gd(lr,w,X,y): ##Correct
    w -= lr*X.T.dot(sigma_func(X.dot(w))-y) ##Written on vector form
    return w    
def log_reg(k,lr,X,y): ##Correct?
    cross_ents = np.zeros(k)
    w_list = np.ones((k,len(X[0]))) ##(rader,kolonner)
    w = np.ones(len(X[0]))
    for i in range(k): ##Each iteration of w
        w = gd(lr,w,X,y)
        w_list[i] = w
        cross_ents[i] = cross_ent_error(y,X.dot(w))
    return w,w_list,cross_ents
def tester(w_list,X,y,k):
    cross_ents = np.zeros(k)
    for i in range(k):
        cross_ents[i] = cross_ent_error(y,X.dot(w_list[i].T))
    return cross_ents
    
  
##########Task 2.1#######
X,y = prepare_dataset(cltrain1) ##Training set
w,w_list,cross_ents = log_reg(k,lr,X,y)
plt.scatter(X[:,1],X[:,2],c = y)

line_x = np.linspace(0,1,2)
line_y = (-w[0]-w[1]*line_x)/w[2]
plt.plot(line_x,line_y)

plt.show()

a = np.linspace(1,k,k)
plt.plot(a,cross_ents)
plt.show()
############     #############

X_test,y_test = prepare_dataset(cltest1)##Test set
cross_ents_test = tester(w_list,X_test,y_test,k)
plt.scatter(X_test[:,1],X_test[:,2],c = y_test)

line_x = np.linspace(0,1,2)
line_y = (-w[0]-w[1]*line_x)/w[2]
plt.plot(line_x,line_y)

plt.show()

a = np.linspace(1,k,k) ##Visualize cross error for test set
plt.plot(a,cross_ents_test)

plt.show()

        
    
