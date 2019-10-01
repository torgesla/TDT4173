import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

cltrain2 = genfromtxt("C:\\Users\\T. Laurvik\\Desktop\\dataset\\dataset\\classification\\cl_train_2.csv", delimiter=',')
cltest2 = genfromtxt("C:\\Users\\T. Laurvik\\Desktop\\dataset\\dataset\\classification\\cl_test_2.csv", delimiter=',')

lr = 0.01 ##Learning rate
k = 1000 ##Number of iterations

def prepare_dataset(dataset):
    return dataset[:,:-1],dataset[:,-1] ##returns X,y
def linear_X(X):
    return X
def quadratic_X(X):
    X_second_degree = np.zeros((len(X),6))
    X_second_degree[:,0] = X[:,1]**2 ##X_1^2
    X_second_degree[:,1] = X[:,2]**2 ##X_2^2
    X_second_degree[:,2] = X[:,1]*X[:,2] ##X_1*X_2
    X_second_degree[:,3] = X[:,1] ##X_1
    X_second_degree[:,4] = X[:,2] ##X_2
    X_second_degree[:,5] = np.ones(len(X)) ## vector of ones
    return X_second_degree
def sigma_func(z): ##Correct
    return 1/(1+np.exp(-z))
def cross_ent_error(y,z): #Correct
    N = len(y)
    summ = y.dot(np.log(sigma_func(z)))+(1-y).dot(np.log(1-sigma_func(z))) 
    return -1/N*summ
def gd(lr,w,X,y): ##Correct
    w -= lr*X.T.dot(sigma_func(X.dot(w))-y) ##Written on vector form
    return w    
def log_reg(k,lr,X,y,X_func): ##Correct?
    cross_ents = np.zeros(k)
    w_list = np.ones((k,len(X[0])))
    X = X_func(X)
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
 
##########Task 2.2#######



##Plot linear
X,y = prepare_dataset(cltrain2)
w,w_list,cross_ents = log_reg(k,lr,X,y,linear_X)
plt.scatter(X[:,1],X[:,2],c = y)
line_x = np.linspace(0,1,2)
line_y = (-w[0]-w[1]*line_x)/w[2]
plt.plot(line_x,line_y)
plt.show()
a = np.linspace(1,k,k)
plt.plot(a,cross_ents)
plt.show()
##########
X_test,y_test = prepare_dataset(cltest2)##Test set
cross_ents_test = tester(w_list,X_test,y_test,k)
plt.scatter(X_test[:,1],X_test[:,2],c = y_test)

line_x = np.linspace(0,1,2)
line_y = (-w[0]-w[1]*line_x)/w[2]
plt.plot(line_x,line_y)

plt.show()

a = np.linspace(1,k,k) ##Visualize cross error for test set
plt.plot(a,cross_ents_test)

plt.show()

#Plot circle
w,w_list,cross_ents = log_reg(k,lr,X,y,quadratic_X)
plt.scatter(X[:,1],X[:,2],c = y)
x_circle=np.linspace(0,1,101)
y_circle=np.linspace(0,1,101)
X_Circle, Y_Circle = np.meshgrid(x_circle, y_circle)
F = w[0] * X_Circle**2 + w[1] * Y_Circle**2 + w[2] * X_Circle * Y_Circle + w[3] * X_Circle +  w[4] * Y_Circle + w[5]
cp = plt.contour(X_Circle, Y_Circle, F, [0])
plt.show()
a = np.linspace(1,k,k)
plt.plot(a,cross_ents)
plt.show()
##




    
