# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
 
2. Set variables for assigning dataset values.
 
3. Import linear regression from sklearn.
 
4. Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: T.LISIANA
RegisterNumber: 212222240053 
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
```

## Output:
## Array value of x:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/d0f0c258-ed8b-4f6e-9689-e4273a45d0e7)

## Array value of y:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/18435309-f1b3-49c7-8b77-0173ce224ffa)

## Score graph:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/ff35e82e-8ba1-47e9-9903-b1f018b18c6b)

## Sigmoid function graph:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/d77a11c9-89ed-489c-9ee8-a5c580f43680)

## X train grad value:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/deb49f4e-18d5-476a-9c9f-0c9aee356e96)

## Y train grad value:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/b5a9ece6-5875-4e22-afbe-24eb2ca4b367)

## Regression value:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/f7cfbeb1-49d0-4715-80d4-93c2cb400565)

## decision boundary graph:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/820c3e10-dc77-407e-82b0-e49c770f1326)

## Probability value:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/2f40a1c1-61ec-4b65-a790-6a946882be80)

## Prediction value of mean:

![image](https://github.com/lisianathiruselvan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119389971/2d5e180b-0378-4123-b79b-7bfe44b3fa76)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

