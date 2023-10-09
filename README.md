# Implementation-of-Logistic-Regression-Using-Gradient-Descent
# AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent. Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Use the standard libraries in python for finding linear regression.
Set variables for assigning dataset values.
Import linear regression from sklearn.
Predict the values of array.
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
Obtain the graph.
# Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Nivesha P
RegisterNumber:  212222040108
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt", delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:,0],X[y == 1][:,1], label="Admitted")
plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label ="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costfunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return j,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta =np.array([0,0,0])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min()-1, X[:,0].max() +1
  y_min, y_max = X[:,1].min()-1, X[:,1].max() +1
  xx,yy =np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min,y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:,0],X[y== 1][:,1],label="Admitted")
  plt.scatter(X[y== 0][:,0],X[y ==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels =[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
# Output:
# Array Value of x
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/7c2282a6-5abf-4744-b01d-f8d5382c58cd)


# Array Value of y
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/6893a6d9-5e96-449f-85b9-120ab28d9381)


# Exam 1 - score graph
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/c571733f-64f2-41b9-a49d-1414d92d8eb2)


# Sigmoid function graph
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/ac2f7ee2-f950-46ab-ae83-08163d2e6630)


# X_train_grad value
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/c8bc90a5-a736-46e4-aaca-619c32847d81)


# Y_train_grad value
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/1ad4b199-76b1-428d-896f-94d45497d3a6)


# Print res.x
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/c185abb8-cb9e-4ac6-a5d5-2227de6add72)


# Decision boundary - graph for exam score
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/46d42063-dab7-478a-b509-a38f926787b2)


# Probablity value
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/87401384-d153-4f41-8d7c-64a9d3d362ec)


# Prediction value of mean
![image](https://github.com/niveshaprabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122986499/5a4728fa-146a-42d4-af3b-9d74f2c0daa4)


# Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
